#!/usr/bin/env python3
"""Stage 3: fine-tune YOLO11-seg on the CVAT canopy polygons.

The counterpart to dinov3-linear-probe. That pipeline freezes a self-supervised
backbone and fits 769 numbers on top, which is the right move on 180 annotated
tiles but gives the head no spatial prior, so canopies fragment and instance
mAP sits at 0.147 against a patch AP of 0.769. This trains a real instance
segmenter end to end instead, which is the principled fix for that gap, at the
cost of updating every weight on a small dataset.

WHY -seg AND NOT PLAIN DETECTION
Measured over the 202 annotated polygons, the median instance covers 20% of its
1024px tile, the median bounding box covers 27%, and 64% of instances are
clipped by a tile edge. A box around a sprawling, non-convex, edge-truncated
canopy says almost nothing about where the plant is, and box IoU between two
such boxes is high whether or not the masks agree. The polygons are already
drawn, the repo already reports mask IoU and Dice, so the segmentation head is
both the cheaper and the more informative target.

WHY THE SMALL MODEL
145 training instances over 39 source frames. Capacity is not the binding
constraint, data is. yolo11s-seg is the default here and n/m are a sweep, not
an upgrade path, which `--model` exists to test rather than assume.

NADIR IMAGERY HAS NO CANONICAL UP
These are straight-down UAV frames, so a rotated or vertically flipped tile is
a perfectly plausible one. Ultralytics leaves `flipud` and `degrees` at zero
because COCO photographs have a ground plane. Here they are free augmentation
and both are turned on, which matters more than usual at this dataset size.

Usage:
    python train_yolo.py --data yolo_dataset
    python train_yolo.py --data yolo_dataset --model yolo11n-seg.pt --imgsz 640
    python train_yolo.py --data yolo_dataset --show-config     # no torch needed
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path


def build_config(args) -> dict:
    """Every knob that differs from an Ultralytics default, in one place."""
    cfg = {
        "data": str(Path(args.data).resolve() / "data.yaml"),
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "seed": args.seed,
        "deterministic": True,
        "project": str(Path(args.out).resolve()),
        "name": args.name,
        "exist_ok": True,
        "val": True,
        "plots": True,
        # Nadir UAV frames have no canonical orientation, so give the model the
        # full dihedral group rather than left-right only.
        "fliplr": 0.5,
        "flipud": 0.5,
        "degrees": 180.0,
        # Mosaic is the main regulariser at 130 training tiles, but it
        # downscales objects that already run off the tile edge, so it is
        # switched off for the last stretch and the model finishes on real
        # tiles at real scale.
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        # Instances are few, so pasting extra canopies in is worth more here
        # than it would be on a crowded dataset.
        "copy_paste": args.copy_paste,
        "copy_paste_mode": "flip",
        # Dry-season chinee apple is distinguished partly by being brown while
        # much of the scrub is not, so hue is a real cue and is left alone at
        # the Ultralytics default rather than widened.
        "hsv_h": 0.015,
        "scale": args.scale,
        "cos_lr": True,
    }
    if args.freeze:
        cfg["freeze"] = args.freeze
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("yolo_dataset"),
                    help="dataset root written by cvat_to_yolo.py")
    ap.add_argument("--model", default="yolo11s-seg.pt",
                    help="yolo11{n,s,m,l,x}-seg.pt, or a .yaml to train from scratch")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="tiles are 1024px native. 640 is ~2.5x faster and loses canopy texture")
    ap.add_argument("--batch", type=int, default=4,
                    help="4 fits yolo11s-seg at 1024px on a 16GB T4. Raise it at --imgsz 640")
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--close-mosaic", type=int, default=30,
                    help="final epochs trained without mosaic, at real scale")
    ap.add_argument("--copy-paste", type=float, default=0.3)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--freeze", type=int, default=None,
                    help="freeze the first N layers, e.g. 10 for backbone-only transfer")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--name", default=None, help="run name, defaults to model and imgsz")
    ap.add_argument("--show-config", action="store_true",
                    help="print the training config and exit, loads no torch")
    args = ap.parse_args()

    stem = Path(args.model).stem
    args.name = args.name or f"{stem}_{args.imgsz}px"

    data_yaml = Path(args.data).resolve() / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"no data.yaml at {data_yaml}, run cvat_to_yolo.py first")

    split = Path(args.data) / "split.json"
    meta = json.loads(split.read_text()) if split.exists() else {}
    if meta:
        d = meta.get("data", {})
        print(f"dataset   {d.get('tiles')} tiles, {d.get('instances')} instances")
        print(f"split     {d.get('frames')}")
        if not meta.get("config", {}).get("markers_painted_out", False):
            print("WARNING   cardboard markers are still in the imagery. A detector can "
                  "learn 'board, therefore canopy' and will not transfer to unmarked flights.")
        if meta.get("config", {}).get("background_frac", 0):
            print("WARNING   unannotated tiles were added as background. On this dataset they "
                  "are unlabelled, not empty, so this trains the model to suppress true plants.")

    cfg = build_config(args)
    print("\ntraining config, deviations from Ultralytics defaults included")
    for k, v in cfg.items():
        print(f"  {k:<16} {v}")

    if args.show_config:
        print("\n--show-config, nothing trained")
        return 0

    from ultralytics import YOLO  # deferred, needs torch

    model = YOLO(args.model)
    model.train(**cfg)
    results = model.val(data=cfg["data"], imgsz=args.imgsz, split="val")

    run_dir = Path(cfg["project"]) / cfg["name"]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=Path(__file__).parent).stdout.strip() or None
    except Exception:
        commit = None

    # Ultralytics' own numbers, kept separate from the shared-metrics numbers
    # that evaluate.py writes. The two use different matching conventions, so
    # mixing them in one table would invite a false comparison.
    seg = getattr(results, "seg", None)
    box = getattr(results, "box", None)
    native = {
        "ultralytics_mask_mAP50_95": float(seg.map) if seg else None,
        "ultralytics_mask_mAP50": float(seg.map50) if seg else None,
        "ultralytics_box_mAP50_95": float(box.map) if box else None,
        "ultralytics_box_mAP50": float(box.map50) if box else None,
    }
    (run_dir / "train_summary.json").write_text(json.dumps({
        "pipeline": "yolo11-seg/train",
        "config": cfg,
        "data": meta.get("data", {}),
        "ultralytics_val": native,
        "weights": str(run_dir / "weights" / "best.pt"),
        "run": {"date": datetime.date.today().isoformat(), "commit": commit},
    }, indent=2))

    print(f"\nUltralytics val, its own matching convention:")
    for k, v in native.items():
        print(f"  {k:<28} {v if v is None else f'{v:.3f}'}")
    print(f"\nweights   {run_dir/'weights'/'best.pt'}")
    print(f"summary   {run_dir/'train_summary.json'}")
    print("\nNow score it on the shared grid so it is comparable with the other pipelines:")
    print(f"  python evaluate.py --weights {run_dir/'weights'/'best.pt'} --data {args.data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
