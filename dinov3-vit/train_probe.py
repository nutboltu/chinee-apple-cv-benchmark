"""Stage 2: supervised linear probe on frozen DINOv3 patch tokens.

Takes the CVAT "CVAT for images 1.1" polygon annotations produced in CVAT with
SAM3, turns them into per-patch labels over the frozen backbone's token grid,
and fits a logistic-regression head. This is the supervised counterpart to the
KMeans pseudo-labels in vit_features.py, and reuses its backbone loading and
patch_features() unchanged, so the probe sits on exactly the same features.

    tile + polygons -> mask -> patch-grid coverage -> {pos, neg, ignore}
    frozen ViT patch tokens (N, D) -> L2 norm -> logistic regression -> p(weed)

WHY THE SPLIT IS BY SOURCE IMAGE
Tiles are cut 8x6 from each 8192x5460 frame, and the bottom tile row is shifted
flush to the edge, so it overlaps the row above by 684px. Neighbouring tiles
from one frame therefore share both content and, in the overlap band, literal
pixels. A random split over tiles or patches leaks train into test and inflates
every metric. Grouping by source frame is the only honest option, so --folds
runs GroupKFold over source frames.

BOUNDARY PATCHES ARE DISCARDED
A patch straddling the polygon edge is neither weed nor background. Patches
between --neg-thresh and --pos-thresh coverage are dropped rather than forced
into a class, which keeps the decision boundary off the annotation's own
uncertainty.

CARDBOARD IS MASKED OUT
The field markers are white cardboard that exists only because this is a
ground-truthing flight. Measured over this set they are 0.12% of mask area, so
excluding them changes little, but a probe that learned "cardboard" would score
well here and fail on any unmarked imagery. Pass --keep-markers to disable.

Usage:
    python train_probe.py --zips '../chinee-apple-dataset/.../processed/batch_*.zip' \
        --tiles '../chinee-apple-dataset/.../processed' --family dinov2
    python train_probe.py ... --family dinov3 --repo /path/dinov3 --weights /path/w.pth
    python train_probe.py ... --limit 20 --dry-run     # label geometry only, no backbone
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None


# --------------------------------------------------------------------------- #
# CVAT annotations -> per-tile polygon lists
# --------------------------------------------------------------------------- #
def read_cvat_zips(patterns: list[str]) -> dict[str, list[list[tuple[float, float]]]]:
    """Map tile filename -> list of polygons, each a list of (x, y)."""
    out: dict[str, list] = collections.defaultdict(list)
    files: list[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise SystemExit(f"no annotation zips matched: {patterns}")
    for fp in files:
        p = Path(fp)
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                names = [n for n in z.namelist() if n.endswith(".xml")]
                if not names:
                    continue
                root = ET.fromstring(z.read(names[0]))
        else:
            root = ET.parse(p).getroot()
        for im in root.findall("image"):
            name = Path(im.get("name", "")).name
            for poly in im.findall("polygon"):
                pts = [tuple(map(float, xy.split(","))) for xy in poly.get("points").split(";")]
                out[name].append(pts)
    return dict(out)


def read_marker_boxes(crib: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    boxes: dict[str, list] = collections.defaultdict(list)
    if not crib.exists():
        return {}
    for r in csv.DictReader(crib.open()):
        boxes[r["tile"]].append((int(r["marker_x0"]), int(r["marker_y0"]),
                                 int(r["marker_x1"]), int(r["marker_y1"])))
    return dict(boxes)


def index_tiles(root: Path) -> dict[str, Path]:
    return {p.name: p for p in root.rglob("*.jpg")}


def source_of(tile_name: str) -> str:
    """DJI_20251118124919_0185_r0_c5.jpg -> DJI_20251118124919_0185"""
    return tile_name.rsplit("_r", 1)[0]


# --------------------------------------------------------------------------- #
# Masks -> patch-grid labels
# --------------------------------------------------------------------------- #
def tile_mask(size: tuple[int, int], polys, markers=None) -> np.ndarray:
    """Binary mask at tile resolution, with marker boxes punched back out."""
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for pts in polys:
        d.polygon(pts, fill=1)
    if markers:
        for x0, y0, x1, y1 in markers:
            d.rectangle([x0, y0, x1, y1], fill=0)
    return np.asarray(m, dtype=np.uint8)


def patch_coverage(mask: np.ndarray, side: int, patch: int) -> np.ndarray:
    """Fraction of each patch covered by the mask, matching preprocess()'s resize."""
    m = Image.fromarray(mask * 255).resize((side, side), Image.BILINEAR)
    a = np.asarray(m, dtype=np.float32) / 255.0
    g = side // patch
    return a.reshape(g, patch, g, patch).mean(axis=(1, 3))


def label_patches(cov: np.ndarray, pos_t: float, neg_t: float) -> np.ndarray:
    """1 positive, 0 negative, -1 ignore."""
    y = np.full(cov.size, -1, dtype=np.int8)
    f = cov.ravel()
    y[f >= pos_t] = 1
    y[f <= neg_t] = 0
    return y


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zips", nargs="+", required=True, help="CVAT export zips or xml files (globs ok)")
    ap.add_argument("--tiles", type=Path, required=True, help="directory holding the tile jpgs")
    ap.add_argument("--crib", type=Path, default=None, help="tile_markers.csv, for masking cardboard out")
    ap.add_argument("--out", type=Path, default=Path("probe_out"))
    ap.add_argument("--family", choices=("dinov3", "dinov2"), default="dinov2")
    ap.add_argument("--repo", default=None, help="local dinov3 repo (gated weights)")
    ap.add_argument("--weights", default=None, help="dinov3 .pth")
    ap.add_argument("--img-size", type=int, default=768)
    ap.add_argument("--pos-thresh", type=float, default=0.50)
    ap.add_argument("--neg-thresh", type=float, default=0.10)
    ap.add_argument("--folds", type=int, default=5, help="GroupKFold folds over source frames")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N annotated tiles")
    ap.add_argument("--keep-markers", action="store_true", help="do NOT mask cardboard out of the masks")
    ap.add_argument("--dry-run", action="store_true", help="label geometry only, never loads a backbone")
    args = ap.parse_args()

    polys_by_tile = read_cvat_zips(args.zips)
    tiles = index_tiles(args.tiles)
    crib = read_marker_boxes(args.crib) if (args.crib and not args.keep_markers) else {}

    usable = [n for n in sorted(polys_by_tile) if n in tiles]
    missing = [n for n in sorted(polys_by_tile) if n not in tiles]
    if args.limit:
        usable = usable[: args.limit]
    print(f"annotated tiles in export : {len(polys_by_tile)}")
    print(f"tile images found         : {len(usable)}")
    if missing:
        print(f"WARNING missing images    : {len(missing)} (e.g. {missing[:3]})")
    if not usable:
        raise SystemExit("no annotated tile images found, check --tiles")

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- label geometry, independent of any backbone ----------------------- #
    if args.dry_run:
        patch = 16 if args.family == "dinov3" else 14
        side = max(patch, (args.img_size // patch) * patch)
        pos = neg = ign = 0
        for n in usable:
            with Image.open(tiles[n]) as im:
                size = im.size
            mask = tile_mask(size, polys_by_tile[n], crib.get(n))
            y = label_patches(patch_coverage(mask, side, patch), args.pos_thresh, args.neg_thresh)
            pos += int((y == 1).sum()); neg += int((y == 0).sum()); ign += int((y == -1).sum())
        tot = pos + neg + ign
        print(f"\ngrid {side//patch}x{side//patch} per tile, patch {patch}px")
        print(f"positive patches : {pos:>8}  ({100*pos/tot:.1f}%)")
        print(f"negative patches : {neg:>8}  ({100*neg/tot:.1f}%)")
        print(f"ignored boundary : {ign:>8}  ({100*ign/tot:.1f}%)")
        print(f"source frames    : {len({source_of(n) for n in usable})}")
        print("\nDRY RUN, no backbone loaded and no model trained")
        return 0

    # ---- features ---------------------------------------------------------- #
    import vit_features as vf  # noqa: E402  (deferred, needs torch)

    model, meta = vf.load_backbone(family=args.family, repo=args.repo, weights=args.weights)
    X, Y, G = [], [], []
    for i, n in enumerate(usable, 1):
        feats, (gh, gw) = vf.patch_features(model, str(tiles[n]), meta, img_size=args.img_size)
        feats = np.asarray(feats, dtype=np.float32)
        feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        with Image.open(tiles[n]) as im:
            size = im.size
        mask = tile_mask(size, polys_by_tile[n], crib.get(n))
        cov = patch_coverage(mask, gh * meta["patch"], meta["patch"])
        if cov.shape != (gh, gw):                      # non-square grid, recompute
            cov = np.asarray(Image.fromarray(mask * 255).resize((gw, gh), Image.BILINEAR),
                             dtype=np.float32) / 255.0
        y = label_patches(cov, args.pos_thresh, args.neg_thresh)
        keep = y >= 0
        X.append(feats[keep]); Y.append(y[keep]); G.extend([source_of(n)] * int(keep.sum()))
        if i % 20 == 0 or i == len(usable):
            print(f"  features {i}/{len(usable)}")

    X = np.concatenate(X); Y = np.concatenate(Y); G = np.asarray(G)
    print(f"\npatches kept     : {len(Y)}  positive {int(Y.sum())} ({100*Y.mean():.1f}%)")
    print(f"feature dim      : {X.shape[1]}   source frames: {len(set(G))}")

    # ---- cross-validated probe, grouped by source frame -------------------- #
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

    folds = min(args.folds, len(set(G)))
    gkf = GroupKFold(n_splits=folds)
    rows = []
    for k, (tr, te) in enumerate(gkf.split(X, Y, groups=G), 1):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(X[tr], Y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        rows.append({
            "fold": k,
            "test_frames": len(set(G[te])),
            "ap": float(average_precision_score(Y[te], p)),
            "auc": float(roc_auc_score(Y[te], p)),
            "f1": float(f1_score(Y[te], (p >= 0.5).astype(int))),
        })
        print(f"  fold {k}: AP={rows[-1]['ap']:.3f}  AUC={rows[-1]['auc']:.3f}  F1={rows[-1]['f1']:.3f}")

    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("ap", "auc", "f1")}
    std = {k: float(np.std([r[k] for r in rows])) for k in ("ap", "auc", "f1")}
    print(f"\ngrouped {folds}-fold: AP={mean['ap']:.3f}+-{std['ap']:.3f}  "
          f"AUC={mean['auc']:.3f}+-{std['auc']:.3f}  F1={mean['f1']:.3f}+-{std['f1']:.3f}")

    final = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(X, Y)
    import joblib
    joblib.dump({"clf": final, "family": args.family, "img_size": args.img_size,
                 "patch": meta["patch"], "l2_normalized": True}, args.out / "probe.joblib")
    (args.out / "metrics.json").write_text(json.dumps(
        {"folds": rows, "mean": mean, "std": std, "n_patches": int(len(Y)),
         "positive_rate": float(Y.mean()), "n_frames": len(set(G)),
         "tiles": len(usable)}, indent=2))
    print(f"\nwrote {args.out/'probe.joblib'} and {args.out/'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
