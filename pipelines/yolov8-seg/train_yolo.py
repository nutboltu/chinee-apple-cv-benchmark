#!/usr/bin/env python3
"""YOLOv8 pipeline entry point for training.

Defaults to `yolov8s-seg.pt`. The implementation is shared with the other YOLO
segmentation pipeline and lives in the repo-root `yoloseg` package, so the two
families differ by this one string and nothing else.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from yoloseg.train_yolo import *   # noqa: F401,F403
from yoloseg.train_yolo import main

DEFAULT_MODEL = "yolov8s-seg.pt"

if __name__ == "__main__":
    raise SystemExit(main(DEFAULT_MODEL))
