#!/usr/bin/env python3
"""YOLO11 pipeline entry point for `cvat_to_yolo`.

The implementation is shared with the other YOLO segmentation pipeline and
lives in the repo-root `yoloseg` package, so a fix reaches both families at
once. Run this file, or `python -m yoloseg.cvat_to_yolo`, they are the same code.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from yoloseg.cvat_to_yolo import *          # noqa: F401,F403  (re-export for importers)
from yoloseg.cvat_to_yolo import main

if __name__ == "__main__":
    raise SystemExit(main())
