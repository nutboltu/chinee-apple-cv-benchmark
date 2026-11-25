#!/usr/bin/env python3
"""Group source frames into spatial blocks, so a split can avoid shared ground.

WHY THIS EXISTS
Grouping a train/test split by source frame stops tiles from one frame landing
on both sides. It does not stop the frames themselves overlapping, and on this
survey they overlap heavily. A Zenmuse P1 at 30m AGL with the 35mm lens covers
about 30.8 x 20.5 m of ground, and consecutive frames are a median 4.1 m apart,
so neighbouring frames share 80 percent or more of what they see.

Measured on the wet probe split, all 13 held-out frames sat within 4.2 m of a
training frame. The same individual plants were on both sides. Patch AP of
0.982 is partly a memory test.

This reads the GPS in each frame's EXIF, joins frames that sit within --buffer
metres of each other into a block, and writes the frame to block mapping. Pass
it to train_probe.py --blocks and the split groups by block, so held-out ground
is ground the model has never seen.

    python spatial_blocks.py --dataset matured-wet-30m
    python train_probe.py ... --blocks <processed>/frame_blocks.csv

The default buffer is one frame footprint. Raise it to be stricter.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ExifTags
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tile_images import DEFAULT_DATASET, dataset_root

Image.MAX_IMAGE_PIXELS = None
_GPS = {v: k for k, v in ExifTags.TAGS.items()}["GPSInfo"]


def frame_gps(path: Path) -> tuple[float, float] | None:
    """(lat, lon) in signed degrees, or None when the frame carries no GPS."""
    with Image.open(path) as im:
        info = im.getexif().get_ifd(_GPS)
    if not info:
        return None

    def dec(v, ref):
        d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        return -d if ref in ("S", "W") else d

    try:
        return dec(info[2], info[1]), dec(info[4], info[3])
    except Exception:
        return None


def local_xy(pts: np.ndarray) -> np.ndarray:
    """Degrees to metres on a local tangent plane. Fine over a few tens of km."""
    lat0 = float(pts[:, 0].mean())
    return np.stack([(pts[:, 1] - pts[:, 1].mean()) * 111320 * math.cos(math.radians(lat0)),
                     (pts[:, 0] - lat0) * 110540], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help=f"dataset name under datasets/, or a path (default: {DEFAULT_DATASET})")
    ap.add_argument("--raw", type=Path, default=None, help="source images (default: <dataset>/raw)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output csv (default: <dataset>/processed/frame_blocks.csv)")
    ap.add_argument("--buffer", type=float, default=31.0,
                    help="metres of empty ground that must separate two blocks. The default is "
                         "one frame footprint at 30m with the 35mm lens (default: 31)")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    root = dataset_root(args.dataset)
    raw = (args.raw or root / "raw").resolve()
    out = (args.out or root / "processed" / "frame_blocks.csv").resolve()
    if not raw.is_dir():
        print(f"error: no raw directory at {raw}", file=sys.stderr)
        return 1

    frames = sorted(raw.glob("*.JPG")) + sorted(raw.glob("*.jpg"))
    pos = {p.stem: frame_gps(p) for p in frames}
    missing = sorted(k for k, v in pos.items() if v is None)
    pos = {k: v for k, v in pos.items() if v}
    if not pos:
        print("error: no frame carried GPS, cannot build spatial blocks", file=sys.stderr)
        return 1

    names = sorted(pos)
    xy = local_xy(np.array([pos[n] for n in names]))
    labels = (fcluster(linkage(pdist(xy), "single"), t=args.buffer, criterion="distance")
              if len(names) > 1 else np.array([1]))

    # Number blocks west to east so the naming is stable across reruns.
    order = {old: i + 1 for i, old in enumerate(
        sorted(set(labels), key=lambda c: xy[labels == c, 0].mean()))}
    block = {n: f"block_{order[c]:02d}" for n, c in zip(names, labels)}

    print(f"dataset : {args.dataset}")
    print(f"frames  : {len(names)} with GPS" + (f", {len(missing)} without" if missing else ""))
    print(f"buffer  : {args.buffer:g} m")
    print(f"extent  : {np.ptp(xy[:, 0]):.0f} m by {np.ptp(xy[:, 1]):.0f} m")
    print()
    counts = {b: sum(1 for v in block.values() if v == b) for b in sorted(set(block.values()))}
    print(f"{'block':<10}{'frames':>8}   extent")
    for b, c in counts.items():
        m = np.array([block[n] == b for n in names])
        print(f"{b:<10}{c:>8}   {np.ptp(xy[m, 0]):.0f} m by {np.ptp(xy[m, 1]):.0f} m")

    if len(counts) > 1:
        D = squareform(pdist(xy))
        gaps = []
        for b in counts:
            m = np.array([block[n] == b for n in names])
            gaps.append(D[np.ix_(m, ~m)].min())
        print(f"\nsmallest gap between blocks: {min(gaps):.0f} m")
    else:
        print("\nWARNING only one block at this buffer, so no spatially clean holdout exists.")
        print("Every frame is within the buffer of another. Lower --buffer to split anyway,")
        print("knowing the halves will share ground, or fly a separate site.")

    if args.dry_run:
        print("\nDRY RUN, nothing written")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "lat", "lon", "block"])
        w.writeheader()
        for n in names:
            w.writerow({"source": n, "lat": f"{pos[n][0]:.8f}",
                        "lon": f"{pos[n][1]:.8f}", "block": block[n]})
    print(f"\nwrote {out}")
    print("pass it to train_probe.py --blocks, and upload it to Drive beside the tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
