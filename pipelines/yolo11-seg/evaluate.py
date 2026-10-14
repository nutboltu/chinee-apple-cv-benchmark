#!/usr/bin/env python3
"""Score a trained YOLO11-seg model with the shared metrics, on the shared grid.

Ultralytics already prints a mask mAP, but it is not comparable with anything
else in this repo: it runs its own NMS and its own matching at full mask
resolution, while dinov3-linear-probe and dinov3-kmeans-cluster are scored on a
32x32 patch grid by `metrics/`. A table mixing the two would read as a
comparison and would not be one. So this re-scores YOLO the repo's way.

    predicted instance masks -> per-pixel max confidence -> 32x32 grid
    CVAT polygons            -> 1024px mask             -> 32x32 grid

32x32 is where the probe can be read: ViT-B/14 at 448px gives a 32x32 token
grid, one token per ~32x32 tile pixels. Downsampling YOLO to meet it throws
away real boundary precision that YOLO has and the probe never had, so the mask
IoU reported here is a floor on what YOLO can do, not its best. That is the
price of a comparable number, and `--grid 0` scores at full 1024px resolution
instead for the standalone figure.

TWO INSTANCE NUMBERS, DELIBERATELY
`metrics.instance_map` derives instances from connected components, because a
per-patch probe has no notion of an object. Scoring YOLO that way merges two
adjacent canopies into one component and discards the instance separation that
is the entire reason to train a detector. So both are reported:

  instance.mAP_50_95         connected components, apples to apples with the probe
  instance.native_mAP_50_95  YOLO's own instances, same 101-point AP convention

Expect native to be the higher of the two. If it is not, the model is producing
one blob per tile and the detector is not earning its keep.

CONFIDENCE MUST BE LOW
AP integrates over the precision-recall curve, so it needs the low-confidence
tail. Predicting at the Ultralytics default conf=0.25 truncates that curve and
understates AP. The default here is 0.001.

Usage:
    python evaluate.py --weights runs/yolo11s-seg_1024px/weights/best.pt --data yolo_dataset
    python evaluate.py --self-test        # metric plumbing only, loads no torch
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

Image.MAX_IMAGE_PIXELS = None


# --------------------------------------------------------------------------- #
# YOLO labels -> masks on the scoring grid
# --------------------------------------------------------------------------- #
def read_label(path: Path, w: int, h: int) -> list[list[tuple[float, float]]]:
    """A YOLO-seg label file back to pixel-space rings."""
    polys = []
    if not path.exists():
        return polys
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) < 7:
            continue
        v = np.asarray(f[1:], dtype=np.float64).reshape(-1, 2)
        polys.append([(x * w, y * h) for x, y in v])
    return polys


def rasterize(polys, size: tuple[int, int]) -> np.ndarray:
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for pts in polys:
        if len(pts) >= 3:
            d.polygon([(float(x), float(y)) for x, y in pts], fill=255)
    return np.asarray(m, dtype=np.float32) / 255.0


def to_grid(a: np.ndarray, grid: int) -> np.ndarray:
    """Area-average a float map down to grid x grid.

    Applied identically to predictions and to ground truth. `grid <= 0` is the
    identity, which is how --grid 0 scores at full resolution.
    """
    if grid <= 0 or a.shape == (grid, grid):
        return a
    im = Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8))
    return np.asarray(im.resize((grid, grid), Image.BILINEAR), dtype=np.float32) / 255.0


def gt_for_tile(label: Path, size: tuple[int, int], grid: int, cover: float = 0.5):
    """(binary gt on the grid, [binary gt per instance]) matching the probe's rule."""
    polys = read_label(label, *size)
    gt = to_grid(rasterize(polys, size), grid) >= cover
    inst = [to_grid(rasterize([p], size), grid) >= cover for p in polys]
    inst = [m for m in inst if m.any()]      # an instance that vanishes at this grid
    return gt, inst, len(polys)


# --------------------------------------------------------------------------- #
# instance AP over the model's own instances
# --------------------------------------------------------------------------- #
def native_instance_ap(records: list[dict], iou_thresholds=None) -> dict:
    """COCO-style AP using predicted instances as given, not components.

    Greedy highest-confidence-first, each ground truth claimed once, then the
    same 101-point interpolated AP the shared metrics use, so only the source
    of the instances differs.
    """
    from metrics.segmentation import _ap_from_matches

    if iou_thresholds is None:
        iou_thresholds = np.round(np.arange(0.50, 0.96, 0.05), 2)

    n_gt = sum(len(r["gt_instances"]) for r in records)
    n_pred = sum(len(r["pred_instances"]) for r in records)
    per_t = {}
    for iou_t in iou_thresholds:
        scores, hits = [], []
        for r in records:
            gts = [g.astype(bool) for g in r["gt_instances"]]
            claimed = [False] * len(gts)
            for m, s in sorted(r["pred_instances"], key=lambda x: -x[1]):
                m = m.astype(bool)
                best_i, best_iou = -1, 0.0
                for i, g in enumerate(gts):
                    if claimed[i]:
                        continue
                    union = float((m | g).sum())
                    if union == 0:
                        continue
                    iou = float((m & g).sum()) / union
                    if iou > best_iou:
                        best_i, best_iou = i, iou
                if best_i >= 0 and best_iou >= iou_t:
                    claimed[best_i] = True
                    scores.append(s); hits.append(1)
                else:
                    scores.append(s); hits.append(0)
        per_t[float(iou_t)] = _ap_from_matches(scores, hits, n_gt)

    vals = [v for v in per_t.values() if not np.isnan(v)]
    return {
        "native_mAP_50_95": float(np.mean(vals)) if vals else float("nan"),
        "native_AP_50": per_t.get(0.5, float("nan")),
        "native_AP_75": per_t.get(0.75, float("nan")),
        "native_n_pred_instances": n_pred,
        "native_n_gt_instances": n_gt,
    }


# --------------------------------------------------------------------------- #
def component_ceiling(records: list[dict], min_size: int, smooth: int) -> dict:
    """What `metrics.instance_map` scores when handed a PERFECT prediction.

    Connected components cannot separate two canopies that touch, so feeding
    the ground truth back in as the prediction does not score 1.0. On this
    val split at grid 32 with the probe's tuned smooth=1 and min_size=8 it
    scores 0.605, because the closing merges adjacent annotated plants: 57
    ground truth instances come back as 49 components.

    That is the real ceiling for any pipeline scored this way, so it is
    reported next to the figure it bounds. Read dinov3-linear-probe's 0.147
    against 0.605 rather than against 1.0.
    """
    import metrics

    oracle = [{"prob": r["gt"].astype(np.float32), "gt": r["gt"],
               "gt_instances": r["gt_instances"]} for r in records]
    im = metrics.instance_map(oracle, 0.5, min_size=min_size, smooth=smooth)
    return {
        "component_ceiling_mAP_50_95": im["mAP_50_95"],
        "component_ceiling_AP_50": im["AP_50"],
        "component_ceiling_n_pred": im["n_pred_instances"],
    }


def score(records: list[dict], min_size: int, smooth: int, thresh: float | None) -> dict:
    """Shared three-level report, plus native instances and the metric's ceiling."""
    import metrics

    y = np.concatenate([r["gt"].ravel().astype(np.int8) for r in records])
    p = np.concatenate([r["prob"].ravel().astype(np.float32) for r in records])
    tiles = [{"prob": r["prob"], "gt": r["gt"], "gt_instances": r["gt_instances"]}
             for r in records]
    out = metrics.report(y, p, tiles, thresh=thresh, min_size=min_size, smooth=smooth)
    nat = native_instance_ap(records)
    ceil = component_ceiling(records, min_size, smooth)
    out["instance"].update(nat)
    out["instance"].update(ceil)
    print("\nINSTANCE LEVEL, model's own instances")
    print(f"  mAP@[.50:.95]       {nat['native_mAP_50_95']:.3f}")
    print(f"  AP@0.50             {nat['native_AP_50']:.3f}")
    print(f"  AP@0.75             {nat['native_AP_75']:.3f}")
    print(f"  gt / pred instances {nat['native_n_gt_instances']} / {nat['native_n_pred_instances']}")
    print(f"\nceiling of the component metric at these settings")
    print(f"  perfect prediction  {ceil['component_ceiling_mAP_50_95']:.3f} mAP, "
          f"{ceil['component_ceiling_AP_50']:.3f} AP@0.50")
    print(f"  it resolves {ceil['component_ceiling_n_pred']} of "
          f"{nat['native_n_gt_instances']} annotated plants as separate components")
    return out


def self_test(grid: int = 32) -> int:
    """Run the scoring path on synthetic predictions, with no model and no torch.

    Checks the plumbing rather than the model: a perfect predictor must score
    near 1 at every level, and a constant predictor must sit at the positive
    rate. Both have caught sign and shape mistakes here that a real run would
    have buried under a plausible-looking number.
    """
    rng = np.random.default_rng(0)
    records = []
    for _ in range(12):
        size = (1024, 1024)
        polys = []
        for _ in range(int(rng.integers(1, 4))):
            cx, cy = rng.integers(200, 824, 2)
            r = rng.integers(80, 260)
            t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
            polys.append([(cx + r * np.cos(a), cy + r * np.sin(a)) for a in t])
        gt = to_grid(rasterize(polys, size), grid) >= 0.5
        inst = [m for m in (to_grid(rasterize([p], size), grid) >= 0.5 for p in polys) if m.any()]
        records.append({"gt": gt, "gt_instances": inst, "polys": polys, "size": size,
                        "prob": gt.astype(np.float32),
                        "pred_instances": [(m, 0.9) for m in inst]})

    print("=== perfect predictor, expect ~1.0 everywhere ===")
    r1 = score(records, min_size=4, smooth=0, thresh=0.5)
    print("\n=== constant 0.5 predictor, expect patch AP ~ positive rate ===")
    flat = [dict(r, prob=np.full_like(r["prob"], 0.5), pred_instances=[]) for r in records]
    r2 = score(flat, min_size=4, smooth=0, thresh=0.5)

    ok = True
    checks = [
        ("perfect patch AP >= 0.99", r1["patch"]["ap"] >= 0.99),
        ("perfect mask IoU >= 0.99", r1["mask"]["mean_iou"] >= 0.99),
        ("perfect native mAP >= 0.95", r1["instance"]["native_mAP_50_95"] >= 0.95),
        ("constant patch AP ~ prevalence",
         abs(r2["patch"]["ap"] - r2["patch"]["positive_rate"]) < 0.05),
        ("constant native mAP == 0", r2["instance"]["native_mAP_50_95"] == 0.0),
    ]
    print()
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok &= passed
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=None, help="best.pt from train_yolo.py")
    ap.add_argument("--data", type=Path, default=Path("yolo_dataset"))
    ap.add_argument("--split", default="val", choices=("val", "train"))
    ap.add_argument("--grid", type=int, default=32,
                    help="scoring grid. 32 matches the probe, 0 scores at full resolution")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="keep it low, AP needs the low-confidence tail of the PR curve")
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--min-size", type=int, default=8, help="matches the probe's tuned default")
    ap.add_argument("--smooth", type=int, default=1, help="matches the probe's tuned default")
    ap.add_argument("--thresh", type=float, default=None, help="default picks the best-F1 threshold")
    ap.add_argument("--out", type=Path, default=Path("eval_out"))
    ap.add_argument("--self-test", action="store_true", help="check the metrics plumbing, no model")
    args = ap.parse_args()

    if args.self_test:
        return self_test(args.grid)
    if not args.weights:
        raise SystemExit("--weights is required (or use --self-test)")

    img_dir = args.data / "images" / args.split
    lab_dir = args.data / "labels" / args.split
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images in {img_dir}, run cvat_to_yolo.py first")
    print(f"scoring {len(images)} {args.split} tiles at grid "
          f"{args.grid or args.imgsz}x{args.grid or args.imgsz}")

    from ultralytics import YOLO  # deferred, needs torch

    model = YOLO(str(args.weights))
    records = []
    for i, path in enumerate(images, 1):
        with Image.open(path) as im:
            size = im.size
        gt, gt_inst, n_poly = gt_for_tile(lab_dir / f"{path.stem}.txt", size, args.grid)

        res = model.predict(str(path), imgsz=args.imgsz, conf=args.conf,
                            max_det=args.max_det, retina_masks=True, verbose=False)[0]

        gh, gw = gt.shape
        prob = np.zeros((gh, gw), dtype=np.float32)
        pred_inst = []
        if res.masks is not None and len(res.masks) > 0:
            confs = res.boxes.conf.cpu().numpy().astype(np.float32)
            md = res.masks.data.cpu().numpy().astype(np.float32)   # (N, h, w)
            for m, c in zip(md, confs):
                if m.shape != (size[1], size[0]):
                    m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(
                        size, Image.BILINEAR), dtype=np.float32) / 255.0
                g = to_grid(m, args.grid)
                # Per pixel the score is the most confident instance covering
                # it, which is the dense map a probe would have produced.
                prob = np.maximum(prob, g * c)
                pred_inst.append((g >= 0.5, float(c)))
            pred_inst = [(m, c) for m, c in pred_inst if m.any()]

        records.append({"name": path.name, "gt": gt, "gt_instances": gt_inst,
                        "prob": prob, "pred_instances": pred_inst, "polygons": n_poly})
        if i % 10 == 0 or i == len(images):
            print(f"  {i}/{len(images)}")

    print()
    out = score(records, args.min_size, args.smooth, args.thresh)

    args.out.mkdir(parents=True, exist_ok=True)
    split_meta = {}
    sp = args.data / "split.json"
    if sp.exists():
        split_meta = json.loads(sp.read_text())
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=Path(__file__).parent).stdout.strip() or None
    except Exception:
        commit = None

    (args.out / "metrics.json").write_text(json.dumps({
        "pipeline": "yolo11-seg",
        "config": {
            "backbone": Path(args.weights).parent.parent.name or "yolo11-seg",
            "weights": str(args.weights),
            "img_size": args.imgsz,
            "scoring_grid": args.grid or args.imgsz,
            "conf": args.conf,
            "min_size": args.min_size,
            "smooth": args.smooth,
            "markers_masked": split_meta.get("config", {}).get("markers_painted_out"),
            "split": "held-out source frames, grouped",
        },
        "data": {
            "tiles": len(records),
            "frames": len(split_meta.get("frames", {}).get(args.split, [])) or None,
            "polygons": sum(r["polygons"] for r in records),
        },
        "patch": out["patch"], "mask": out["mask"], "instance": out["instance"],
        "run": {"date": datetime.date.today().isoformat(), "commit": commit},
    }, indent=2))
    print(f"\nwrote {args.out/'metrics.json'}")
    print(f"render it with: python -m metrics.render {args.out/'metrics.json'} --format md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
