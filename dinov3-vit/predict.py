"""Run a trained probe on new imagery: image in, chinee apple mask out.

This is inference. Training lives in train_probe.py, metrics in probe_eval.py.

    image -> frozen DINO ViT -> patch tokens -> probe -> p(weed) per patch
          -> threshold + close -> mask -> connected components -> plants

The probe file records the backbone family, image size and patch size it was
trained with, and they are reused here. Scoring tokens produced at a different
resolution to training is the easiest way to get quiet nonsense, so a mismatch
is refused rather than silently allowed.

METRICS NEED GROUND TRUTH. With --annotations pointing at CVAT polygons for
these same images, the run reports AP, IoU and instance mAP. Without it you get
predictions and overlays only, because there is nothing to score against. An
unannotated image can never produce a metric.

Outputs per image, into --out:
    <stem>_prob.png     probability heatmap, upsampled to the input size
    <stem>_mask.png     binary mask at the same size
    <stem>_overlay.png  the input with the mask edge drawn on
and one instances.csv for every plant found, with area and mean score.

Usage:
    python predict.py --probe probe_out/probe.joblib --image tile.jpg --out preds
    python predict.py --probe probe_out/probe.joblib --image 'tiles/*.jpg' --out preds
    python predict.py --probe probe_out/probe.joblib --image 'tiles/*.jpg' \
        --annotations 'exports/batch_*.zip' --out preds        # also score it
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None


def resolve_images(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.is_dir():
            out.extend(sorted(p.rglob("*.jpg")))
        else:
            out.extend(Path(x) for x in sorted(glob.glob(pat)))
    seen, uniq = set(), []
    for p in out:
        if p.is_file() and p not in seen:
            seen.add(p); uniq.append(p)
    return uniq


def colourise(prob: np.ndarray) -> Image.Image:
    """Probability grid -> magma-ish RGB without pulling in matplotlib."""
    p = np.clip(prob, 0, 1)
    r = np.clip(2.2 * p - 0.3, 0, 1)
    g = np.clip(1.6 * p - 0.55, 0, 1)
    b = np.clip(1.1 * p ** 2 + 0.25 * (1 - p), 0, 1) * (0.4 + 0.6 * p)
    rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def overlay(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Draw the mask boundary in green over the original."""
    edge = ndi.binary_dilation(mask, np.ones((3, 3), bool)) & ~mask
    base = np.asarray(img.convert("RGB")).copy()
    inner = mask & ~edge
    base[inner] = (0.65 * base[inner] + 0.35 * np.array([0, 230, 90])).astype(np.uint8)
    base[edge] = np.array([0, 255, 100], dtype=np.uint8)
    return Image.fromarray(base)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", type=Path, required=True, help="probe.joblib from train_probe.py")
    ap.add_argument("--image", nargs="+", required=True, help="image files, globs or directories")
    ap.add_argument("--out", type=Path, default=Path("predictions"))
    ap.add_argument("--repo", default=None, help="local dinov3 repo, if the probe used dinov3")
    ap.add_argument("--weights", default=None, help="dinov3 .pth")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--smooth", type=int, default=1, help="closing radius on the patch grid")
    ap.add_argument("--min-size", type=int, default=8, help="drop components smaller than this many patches")
    ap.add_argument("--annotations", nargs="+", default=None,
                    help="CVAT zips/xml for these images, to also report metrics")
    ap.add_argument("--no-images", action="store_true", help="skip writing pngs, just the csv")
    args = ap.parse_args()

    import joblib
    bundle = joblib.load(args.probe)
    clf = bundle["clf"]
    family, img_size, patch = bundle["family"], bundle["img_size"], bundle["patch"]
    print(f"probe trained with {family}, img_size {img_size}, patch {patch}")

    images = resolve_images(args.image)
    if not images:
        raise SystemExit(f"no images matched {args.image}")
    print(f"images to score   {len(images)}")

    import vit_features as vf
    model, meta = vf.load_backbone(family=family, repo=args.repo, weights=args.weights)
    if meta["patch"] != patch:
        raise SystemExit(f"patch size mismatch, probe {patch} vs backbone {meta['patch']}. "
                         "The probe must run on the features it was trained on.")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    scored = []

    for i, path in enumerate(images, 1):
        feats, (gh, gw) = vf.patch_features(model, str(path), meta, img_size=img_size)
        feats = np.asarray(feats, dtype=np.float32)
        feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        prob = clf.predict_proba(feats)[:, 1].reshape(gh, gw)

        mask = prob >= args.thresh
        if args.smooth > 0:
            st = np.ones((2 * args.smooth + 1, 2 * args.smooth + 1), bool)
            mask = ndi.binary_closing(mask, st, border_value=0)
            mask = ndi.binary_opening(mask, np.ones((3, 3), bool), border_value=0)

        lab, n = ndi.label(mask)
        kept = 0
        with Image.open(path) as im:
            size = im.size
            src = im.convert("RGB")
        px_per_patch = (size[0] / gw) * (size[1] / gh)
        for k in range(1, n + 1):
            m = lab == k
            npatch = int(m.sum())
            if npatch < args.min_size:
                mask[m] = False
                continue
            kept += 1
            rows.append({
                "image": path.name, "instance": kept,
                "patches": npatch,
                "approx_px": int(npatch * px_per_patch),
                "mean_score": round(float(prob[m].mean()), 4),
                "max_score": round(float(prob[m].max()), 4),
            })

        if not args.no_images:
            stem = path.stem
            colourise(prob).resize(size, Image.BILINEAR).save(args.out / f"{stem}_prob.png")
            big = np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                             .resize(size, Image.NEAREST)) > 127
            Image.fromarray((big * 255).astype(np.uint8)).save(args.out / f"{stem}_mask.png")
            overlay(src, big).save(args.out / f"{stem}_overlay.png")

        scored.append({"name": path.name, "prob": prob, "mask": mask, "grid": (gh, gw), "size": size})
        if i % 10 == 0 or i == len(images):
            print(f"  {i}/{len(images)}")

    with (args.out / "instances.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "instance", "patches", "approx_px",
                                           "mean_score", "max_score"])
        w.writeheader(); w.writerows(rows)

    n_with = len({r["image"] for r in rows})
    print(f"\nplants found      {len(rows)} across {n_with} of {len(images)} images")
    print(f"wrote             {args.out}")

    # ---- optional scoring, only possible where ground truth exists ---------- #
    if args.annotations:
        import train_probe as tp
        import probe_eval as pe
        polys = tp.read_cvat_zips(args.annotations)
        have = [s for s in scored if s["name"] in polys]
        if not have:
            print("\nno ground truth matched these images, metrics skipped")
            return 0
        print(f"\nscoring against ground truth for {len(have)} of {len(images)} images")
        ys, ps, tiles = [], [], []
        for s in have:
            gh, gw = s["grid"]
            side = gh * patch
            cov = tp.patch_coverage(tp.tile_mask(s["size"], polys[s["name"]]), side, patch)
            y = tp.label_patches(cov, 0.5, 0.1)
            k = y >= 0
            ys.append(y[k]); ps.append(s["prob"].ravel()[k])
            inst = []
            for pts in polys[s["name"]]:
                g = tp.patch_coverage(tp.tile_mask(s["size"], [pts]), side, patch) >= 0.5
                if g.any():
                    inst.append(g)
            tiles.append({"prob": s["prob"], "gt": cov >= 0.5, "gt_instances": inst})
        pe.report(np.concatenate(ys), np.concatenate(ps), tiles,
                  thresh=args.thresh, min_size=args.min_size, smooth=args.smooth)
    else:
        print("\nno --annotations given, so no metrics. Metrics need ground truth "
              "for these same images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
