#!/usr/bin/env python3
"""CVAT "CVAT for images 1.1" polygons -> an Ultralytics YOLO segmentation dataset.

    <polygon points="x1,y1;x2,y2;..."/>   ->   0 x1/W y1/H x2/W y2/H ...

One text file per tile, one line per instance, class id then the ring as
normalised xy pairs. YOLO-seg carries a single ring per instance, so there is no
place to put a hole or an ignore region, which drives two of the decisions below.

Depends on pillow and numpy only. No torch, no ultralytics, so the conversion
can be checked on a laptop before any GPU is booked.

ONLY ANNOTATED TILES ARE WRITTEN
440 of the 620 tiles came back from CVAT empty, and that records that nobody
annotated them, not that they hold no chinee apple. YOLO treats every pixel
outside a polygon as background, so feeding those tiles in would train the head
to suppress the plants it is supposed to find. `--background-frac` can add some
anyway, for the ablation that demonstrates this, and defaults to 0.

THE SPLIT IS GROUPED BY SOURCE FRAME
Tiles are cut 8x6 from one 8192x5460 frame and the bottom row is shifted flush
to the edge, so it overlaps the row above by 684px. Neighbouring tiles share
content and, in the overlap band, literal pixels. Splitting over tiles leaks
train into val. Frames are shuffled under `--seed` and split whole, and the
resulting frame lists are written to split.json so a run can be repeated.

CARDBOARD IS PAINTED OUT OF THE IMAGERY
The field markers are white cardboard that exists only because this was a
ground-truthing flight, and each board was placed beside its target plant. A
detector with whole-tile context can learn "cardboard, therefore canopy next to
it" and then score well here and find nothing on unmarked imagery. The probe
pipeline drops those pixels in label space, which a single-ring polygon cannot
express, so here each board is overwritten with texture copied from elsewhere in
the same tile. `--keep-markers` disables it.

Usage:
    P=../../datasets/matured-dry-30m/processed
    python cvat_to_yolo.py --zips "$P/batch_*.zip" --tiles $P --crib $P/tile_markers.csv \
        --out yolo_dataset
    python cvat_to_yolo.py --zips "$P/batch_*.zip" --tiles $P --dry-run
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import glob
import json
import random
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

CLASS_NAME = "chinee_apple"


# --------------------------------------------------------------------------- #
# CVAT exports -> per-tile polygon lists
# --------------------------------------------------------------------------- #
def read_cvat_zips(patterns: list[str]) -> tuple[dict[str, list], dict[str, tuple[int, int]], set[str]]:
    """Return (tile -> polygons, tile -> (W, H), every tile named in the exports).

    The third value is what separates "annotated and empty" from "not in the
    task at all", which is the distinction `--background-frac` draws on.
    """
    polys: dict[str, list] = collections.defaultdict(list)
    sizes: dict[str, tuple[int, int]] = {}
    seen: set[str] = set()
    files: list[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise SystemExit(f"no annotation exports matched: {patterns}")
    for fp in files:
        p = Path(fp)
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                names = [n for n in z.namelist() if n.endswith(".xml")]
                if not names:
                    print(f"  warning: no xml inside {p.name}, skipped")
                    continue
                root = ET.fromstring(z.read(names[0]))
        else:
            root = ET.parse(p).getroot()
        for im in root.findall("image"):
            name = Path(im.get("name", "")).name
            if not name:
                continue
            seen.add(name)
            sizes[name] = (int(im.get("width")), int(im.get("height")))
            for poly in im.findall("polygon"):
                pts = [tuple(map(float, xy.split(",")))
                       for xy in poly.get("points", "").split(";") if xy]
                if len(pts) >= 3:
                    polys[name].append(pts)
    return dict(polys), sizes, seen


def read_marker_boxes(crib: Path | None) -> dict[str, list[tuple[int, int, int, int]]]:
    boxes: dict[str, list] = collections.defaultdict(list)
    if not crib or not crib.exists():
        return {}
    for r in csv.DictReader(crib.open()):
        boxes[r["tile"]].append((int(r["marker_x0"]), int(r["marker_y0"]),
                                 int(r["marker_x1"]), int(r["marker_y1"])))
    return dict(boxes)


def index_tiles(root: Path, skip: Path | None = None) -> dict[str, Path]:
    """tile filename -> path on disk, ignoring anything under `skip`."""
    out = {}
    for p in root.rglob("*.jpg"):
        if skip and skip in p.parents:
            continue
        out[p.name] = p
    return out


def source_of(tile_name: str) -> str:
    """DJI_20251118124919_0185_r0_c5.jpg -> DJI_20251118124919_0185"""
    return tile_name.rsplit("_r", 1)[0]


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def poly_area(pts) -> float:
    a = np.asarray(pts, dtype=np.float64)
    x, y = a[:, 0], a[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def simplify(pts: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """Douglas-Peucker on a closed ring. tol <= 0 returns the ring untouched.

    Ultralytics rasterises polygons for the mask loss, so point count mostly
    costs file size and dataloader time, not accuracy. It stays off by default
    all the same, because thinning is an edit to ground truth.
    """
    if tol <= 0 or len(pts) <= 4:
        return pts

    def dp(seq):
        if len(seq) < 3:
            return seq
        a, b = np.asarray(seq[0]), np.asarray(seq[-1])
        ab = b - a
        n = np.linalg.norm(ab)
        p = np.asarray(seq[1:-1])
        if n == 0:
            d = np.linalg.norm(p - a, axis=1)
        else:
            d = np.abs(np.cross(np.broadcast_to(ab, p.shape), p - a)) / n
        i = int(np.argmax(d))
        if d[i] <= tol:
            return [seq[0], seq[-1]]
        k = i + 1
        return dp(seq[: k + 1])[:-1] + dp(seq[k:])

    # Split the ring at its two extreme points so the result does not depend on
    # where CVAT happened to start the ring.
    a = np.asarray(pts)
    i0 = int(np.argmin(a[:, 0] + a[:, 1]))
    rot = pts[i0:] + pts[:i0]
    i1 = int(np.argmax(np.asarray(rot)[:, 0] + np.asarray(rot)[:, 1]))
    out = dp(rot[: i1 + 1])[:-1] + dp(rot[i1:] + [rot[0]])[:-1]
    return out if len(out) >= 3 else pts


def to_yolo_line(pts, w: int, h: int, cls: int = 0) -> str | None:
    """One YOLO-seg line, or None if clipping collapsed the ring below a triangle.

    Coordinates are written with repr, the shortest form that reads back as the
    same float, rather than a fixed %.6f. CVAT emits whole-pixel coordinates,
    which land exactly on pixel edges, so rounding them even by 5e-7px flips
    entire scanline spans when the ring is rasterised: on the worst tile here
    that was 236 pixels of a 7727 pixel canopy, because these shapes are
    filamentary and carry a long perimeter for their area. Tiles are 1024px, so
    x/1024 is exact in binary and repr makes the conversion lossless rather
    than merely close. It changes nothing about training and removes a
    discrepancy that would otherwise need explaining every time the labels are
    checked against CVAT.
    """
    a = np.asarray(pts, dtype=np.float64)
    a[:, 0] = np.clip(a[:, 0] / w, 0.0, 1.0)
    a[:, 1] = np.clip(a[:, 1] / h, 0.0, 1.0)
    # Drop points that clipping collapsed onto their neighbour.
    keep = [0] + [i for i in range(1, len(a)) if not np.allclose(a[i], a[i - 1], atol=1e-12)]
    a = a[keep]
    if len(a) < 3:
        return None
    body = " ".join(repr(float(v)) for v in a.reshape(-1))
    return f"{cls} {body}"


# --------------------------------------------------------------------------- #
# cardboard removal
# --------------------------------------------------------------------------- #
def paint_out_markers(img: Image.Image, boxes, pad: int = 6) -> Image.Image:
    """Overwrite each marker box with same-tile texture of the same shape.

    A flat grey fill would simply swap one constant cue for another, so the
    replacement is copied from a donor region one box-height away, which keeps
    the local texture statistics roughly intact.
    """
    if not boxes:
        return img
    W, H = img.size
    out = img.copy()
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            continue
        # Prefer a donor above the board, fall back to below, then sideways.
        for dx, dy in ((0, -bh), (0, bh), (-bw, 0), (bw, 0)):
            sx, sy = x0 + dx, y0 + dy
            if 0 <= sx and sx + bw <= W and 0 <= sy and sy + bh <= H:
                out.paste(out.crop((sx, sy, sx + bw, sy + bh)), (x0, y0))
                break
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zips", nargs="+", required=True, help="CVAT export zips or xml files (globs ok)")
    ap.add_argument("--tiles", type=Path, required=True, help="directory holding the tile jpgs")
    ap.add_argument("--crib", type=Path, default=None, help="tile_markers.csv, for painting cardboard out")
    ap.add_argument("--out", type=Path, default=Path("yolo_dataset"))
    ap.add_argument("--val-frac", type=float, default=0.30,
                    help="fraction of SOURCE FRAMES held out (0.30 of 56 frames ~ the probe's 17)")
    ap.add_argument("--seed", type=int, default=0, help="frame shuffle seed, recorded in split.json")
    ap.add_argument("--simplify", type=float, default=0.0,
                    help="Douglas-Peucker tolerance in px, 0 keeps every annotated point")
    ap.add_argument("--background-frac", type=float, default=0.0,
                    help="fraction of UNANNOTATED tiles to add as empty background. "
                         "Wrong on this dataset, see the module docstring")
    ap.add_argument("--keep-markers", action="store_true",
                    help="do NOT paint the cardboard out of the copied tiles")
    ap.add_argument("--symlink", action="store_true",
                    help="symlink tiles instead of copying. Ignored unless --keep-markers")
    ap.add_argument("--dry-run", action="store_true", help="report what would be written, write nothing")
    args = ap.parse_args()

    polys_by_tile, sizes, in_task = read_cvat_zips(args.zips)
    tiles = index_tiles(args.tiles, skip=args.out.resolve())
    markers = {} if args.keep_markers else read_marker_boxes(args.crib)

    annotated = sorted(n for n in polys_by_tile if n in tiles)
    missing = sorted(n for n in polys_by_tile if n not in tiles)
    empty = sorted(n for n in in_task if n not in polys_by_tile and n in tiles)

    print(f"tiles in CVAT tasks      : {len(in_task)}")
    print(f"annotated, image found   : {len(annotated)}")
    print(f"annotated, image MISSING : {len(missing)}" + (f"  e.g. {missing[:2]}" if missing else ""))
    print(f"annotated but empty      : {len(empty)}  (not negatives, see docstring)")
    if not annotated:
        raise SystemExit("no annotated tile images found, check --tiles")
    if not markers and not args.keep_markers:
        print("warning: no --crib given, cardboard stays in the imagery")

    # ---- background tiles, opt in only ------------------------------------- #
    rng = random.Random(args.seed)
    chosen_bg: list[str] = []
    if args.background_frac > 0:
        k = int(round(len(empty) * min(args.background_frac, 1.0)))
        chosen_bg = sorted(rng.sample(empty, k))
        print(f"background tiles added   : {len(chosen_bg)}")

    # ---- split whole source frames ---------------------------------------- #
    members: dict[str, list[str]] = collections.defaultdict(list)
    for n in annotated + chosen_bg:
        members[source_of(n)].append(n)
    frames = sorted(members)
    shuffled = frames[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * args.val_frac))) if args.val_frac > 0 else 0
    val_frames = sorted(shuffled[:n_val])
    train_frames = sorted(shuffled[n_val:])
    if not train_frames:
        raise SystemExit(f"--val-frac {args.val_frac} left no training frames")

    split = {"train": train_frames, "val": val_frames}
    counts = {s: {"frames": len(f), "tiles": sum(len(members[k]) for k in f)} for s, f in split.items()}
    n_poly = {s: sum(len(polys_by_tile.get(t, ())) for k in f for t in members[k]) for s, f in split.items()}
    print(f"\nsplit seed {args.seed}, grouped by source frame")
    for s in ("train", "val"):
        print(f"  {s:<5} {counts[s]['frames']:>3} frames  {counts[s]['tiles']:>4} tiles  "
              f"{n_poly[s]:>4} polygons")

    # ---- geometry summary, useful before committing a GPU ------------------ #
    areas, bfrac, border = [], [], 0
    for n in annotated:
        W, H = sizes.get(n, (1024, 1024))
        for pts in polys_by_tile[n]:
            areas.append(poly_area(pts) / (W * H))
            x = [p[0] for p in pts]; y = [p[1] for p in pts]
            bfrac.append((max(x) - min(x)) * (max(y) - min(y)) / (W * H))
            if min(x) <= 1 or min(y) <= 1 or max(x) >= W - 1 or max(y) >= H - 1:
                border += 1
    q = lambda v, p: float(np.quantile(v, p))  # noqa: E731
    print(f"\ninstances                : {len(areas)}")
    print(f"polygon area / tile      : p05 {q(areas,.05)*100:.1f}%  median {q(areas,.5)*100:.1f}%"
          f"  p95 {q(areas,.95)*100:.1f}%")
    print(f"bbox area / tile         : median {q(bfrac,.5)*100:.1f}%  p95 {q(bfrac,.95)*100:.1f}%")
    print(f"clipped by a tile edge   : {border}/{len(areas)} ({border/len(areas)*100:.0f}%)")
    print("these are large, mostly edge-clipped regions, so prefer -seg over box detection")

    if args.dry_run:
        print("\nDRY RUN, nothing written")
        return 0

    # ---- write ------------------------------------------------------------- #
    for s in ("train", "val"):
        (args.out / "images" / s).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / s).mkdir(parents=True, exist_ok=True)

    written = {"train": 0, "val": 0}
    lines_out = {"train": 0, "val": 0}
    dropped = 0
    for s, fl in split.items():
        for frame in fl:
            for name in sorted(members[frame]):
                src = tiles[name]
                W, H = sizes.get(name, Image.open(src).size)
                lines = []
                for pts in polys_by_tile.get(name, ()):
                    line = to_yolo_line(simplify(pts, args.simplify), W, H)
                    if line is None:
                        dropped += 1
                        continue
                    lines.append(line)
                dst_img = args.out / "images" / s / name
                boxes = markers.get(name)
                if boxes:
                    with Image.open(src) as im:
                        paint_out_markers(im.convert("RGB"), boxes).save(dst_img, quality=95)
                elif args.symlink and args.keep_markers:
                    if dst_img.is_symlink() or dst_img.exists():
                        dst_img.unlink()
                    dst_img.symlink_to(src.resolve())
                else:
                    with Image.open(src) as im:
                        im.convert("RGB").save(dst_img, quality=95)
                (args.out / "labels" / s / f"{Path(name).stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""))
                written[s] += 1
                lines_out[s] += len(lines)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=Path(__file__).parent).stdout.strip() or None
    except Exception:
        commit = None

    (args.out / "data.yaml").write_text(
        f"# generated by cvat_to_yolo.py, do not hand edit\n"
        f"path: {args.out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names:\n  0: {CLASS_NAME}\n")

    (args.out / "split.json").write_text(json.dumps({
        "pipeline": "yolo11-seg/cvat_to_yolo",
        "config": {
            "zips": args.zips, "seed": args.seed, "val_frac": args.val_frac,
            "simplify_px": args.simplify, "background_frac": args.background_frac,
            "markers_painted_out": bool(markers),
            "split": "whole source frames, grouped",
        },
        "data": {
            "class": CLASS_NAME,
            "tiles": written, "instances": lines_out, "frames": counts,
            "polygons_dropped": dropped,
            "annotated_but_empty_tiles_excluded": len(empty) - len(chosen_bg),
        },
        "frames": split,
        "run": {"date": datetime.date.today().isoformat(), "commit": commit},
    }, indent=2))

    print(f"\nwrote {written['train']} train and {written['val']} val tiles, "
          f"{lines_out['train']}/{lines_out['val']} instances")
    if dropped:
        print(f"dropped {dropped} degenerate polygons")
    print(f"dataset root {args.out.resolve()}")
    print(f"  data.yaml   label schema and split paths")
    print(f"  split.json  which frames landed where, and the seed that put them there")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
