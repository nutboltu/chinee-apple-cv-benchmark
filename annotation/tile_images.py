#!/usr/bin/env python3
"""Tile full-resolution drone stills, keeping only tiles that contain a cardboard marker.

Each source image (8192 x 5460 for this set) is split into a grid of 1024 x 1024
tiles: 8 columns x 6 rows = 48 tiles. 8192 divides evenly into 8 columns; 5460
does not divide evenly into rows (5 x 1024 = 5120, leaving a 340 px strip), so
the bottom band is handled by --edge:

  shift (default) : the last row is pulled up to sit flush with the bottom edge,
                    so every tile is a full 1024 x 1024. Costs a 684 px vertical
                    overlap with the row above (no padding, no lost pixels).
  pad             : the last row starts at y=5120 and the 340 px strip is padded
                    out to 1024 px with a solid fill. No overlap, but adds
                    synthetic pixels.
  crop            : the last row starts at y=5120 and is saved as 1024 x 340.
                    No overlap and no synthetic pixels, but tiles vary in size.

MARKER DETECTION
The field markers are white cardboard squares with a sprayed arrow. Plain
brightness thresholding does not find them: the dry grass is bright and
desaturated too. What separates cardboard is that it is a *solid, smooth* blob,
so detection is: threshold on bright + desaturated, apply a morphological
opening to erase thin grass stems, label connected components, and discard
anything below --min-blob-area.

Detection runs ONCE on the full image, not per tile. A marker straddling a tile
boundary is therefore found as a single blob and every tile it overlaps is kept,
rather than each half risking falling under the size floor independently.

With --keep markers (the default) only tiles containing a marker are written.
Use --keep all to write the full 48-tile grid.

Outputs in the output directory:
  <stem>_r<row>_c<col>.jpg   the kept tiles
  manifest.csv               every tile considered, its pixel window, whether it
                             was kept, and how much marker area it contains
  markers.csv                one row per detected marker blob, in full-image
                             coordinates, with the tiles it lands in

Examples:
    python3 tile_images.py --limit 1              # smoke test on one image
    python3 tile_images.py                        # whole directory, markers only
    python3 tile_images.py --keep all             # every tile
    python3 tile_images.py --dry-run              # detect and report, write nothing
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# Dataset lives outside the repo tree in git terms (see .gitignore), one level up
# from this scripts directory.
DATASET = Path(__file__).resolve().parent.parent / "chinee-apple-dataset" / "matured-dry-30m"

# Drone stills are far above Pillow's decompression-bomb threshold.
Image.MAX_IMAGE_PIXELS = None

MANIFEST_FIELDS = [
    "tile", "source", "row", "col", "x0", "y0", "x1", "y1",
    "width", "height", "padded", "kept", "marker_px", "marker_blobs",
]
MARKER_FIELDS = [
    "source", "blob", "area_px", "x0", "y0", "x1", "y1",
    "bbox_w", "bbox_h", "fill", "tiles",
]


def tile_origins(extent: int, tile: int, edge: str) -> list[tuple[int, int]]:
    """Return (origin, length) pairs covering `extent` along one axis."""
    origins: list[tuple[int, int]] = []
    pos = 0
    while pos < extent:
        remaining = extent - pos
        if remaining >= tile:
            origins.append((pos, tile))
        elif edge == "shift":
            origins.append((max(0, extent - tile), min(tile, extent)))
        else:
            origins.append((pos, remaining))
        pos += tile
    return origins


def detect_markers(
    im: Image.Image, v_min: int, s_max: int, open_size: int, close_iter: int, min_blob_area: int
) -> tuple[np.ndarray, list[dict]]:
    """Find solid bright/desaturated blobs. Returns (clean bool mask, blob records)."""
    hsv = im.convert("HSV")
    s = np.asarray(hsv.getchannel("S"))
    v = np.asarray(hsv.getchannel("V"))
    del hsv

    mask = (v > v_min) & (s < s_max)
    del s, v

    cross = np.ones((3, 3), bool)
    if open_size > 1:
        # Erase thin bright grass stems while preserving solid cardboard.
        mask = ndi.binary_opening(mask, np.ones((open_size, open_size), bool), border_value=0)

    if close_iter > 0:
        # The sprayed arrow and shadowed creases cut each board into fragments.
        # Close across those gaps so one board labels as one blob. Iterated 3x3
        # is equivalent to a (2*close_iter+1) kernel but much cheaper at 45MP.
        mask = ndi.binary_dilation(mask, cross, iterations=close_iter)
        mask = ndi.binary_erosion(mask, cross, iterations=close_iter, border_value=1)

    lab, n = ndi.label(mask)
    if n == 0:
        return np.zeros_like(mask), []

    areas = ndi.sum_labels(mask, lab, index=np.arange(1, n + 1))
    keep_ids = np.nonzero(areas >= min_blob_area)[0] + 1
    if keep_ids.size == 0:
        return np.zeros_like(mask), []

    blobs: list[dict] = []
    for bid, sl in zip(keep_ids, [ndi.find_objects(lab)[i - 1] for i in keep_ids]):
        ys, xs = sl
        comp_area = int(areas[bid - 1])
        bw, bh = xs.stop - xs.start, ys.stop - ys.start
        blobs.append({
            "blob": int(bid),
            "area_px": comp_area,
            "x0": int(xs.start), "y0": int(ys.start),
            "x1": int(xs.stop), "y1": int(ys.stop),
            "bbox_w": int(bw), "bbox_h": int(bh),
            "fill": round(comp_area / (bw * bh), 3),
        })

    clean = np.isin(lab, keep_ids)
    del lab
    return clean, blobs


def process_one(
    src: Path, out_dir: Path, tile: int, edge: str, quality: int,
    fill: tuple[int, int, int], keep_exif: bool, overwrite: bool, keep_mode: str,
    v_min: int, s_max: int, open_size: int, close_iter: int, min_blob_area: int, min_tile_px: int,
    dry_run: bool,
) -> tuple[str, list[dict], list[dict], str | None]:
    """Tile one image, writing only the tiles selected by `keep_mode`."""
    tile_rows: list[dict] = []
    marker_rows: list[dict] = []
    try:
        with Image.open(src) as raw:
            im = raw.convert("RGB")
            width, height = im.size
            exif = raw.info.get("exif", b"") if keep_exif else b""

        mask, blobs = detect_markers(im, v_min, s_max, open_size, close_iter, min_blob_area)

        xs = tile_origins(width, tile, edge)
        ys = tile_origins(height, tile, edge)
        blob_tiles: dict[int, list[str]] = {b["blob"]: [] for b in blobs}

        for r, (y0, th) in enumerate(ys):
            for c, (x0, tw) in enumerate(xs):
                name = f"{src.stem}_r{r}_c{c}.jpg"
                window = mask[y0:y0 + th, x0:x0 + tw]
                marker_px = int(window.sum())
                n_blobs = 0
                if marker_px:
                    for b in blobs:
                        # bbox overlap test first, then exact pixel check
                        if b["x0"] < x0 + tw and b["x1"] > x0 and b["y0"] < y0 + th and b["y1"] > y0:
                            n_blobs += 1
                            blob_tiles[b["blob"]].append(f"r{r}_c{c}")

                kept = keep_mode == "all" or marker_px >= min_tile_px
                padded = False
                out_w, out_h = tw, th

                if kept and not dry_run:
                    crop = im.crop((x0, y0, x0 + tw, y0 + th))
                    if edge == "pad" and (tw < tile or th < tile):
                        canvas = Image.new("RGB", (tile, tile), fill)
                        canvas.paste(crop, (0, 0))
                        crop = canvas
                        padded = True
                    out_w, out_h = crop.size
                    dest = out_dir / name
                    if overwrite or not dest.exists():
                        kwargs = {
                            "quality": quality,
                            # 4:4:4 - no chroma subsampling, keeps small-object detail
                            "subsampling": 0,
                            "optimize": True,
                        }
                        if exif:
                            kwargs["exif"] = exif
                        crop.save(dest, "JPEG", **kwargs)

                tile_rows.append({
                    "tile": name, "source": src.name, "row": r, "col": c,
                    "x0": x0, "y0": y0, "x1": x0 + tw, "y1": y0 + th,
                    "width": out_w, "height": out_h, "padded": int(padded),
                    "kept": int(bool(kept)), "marker_px": marker_px,
                    "marker_blobs": n_blobs,
                })

        for b in blobs:
            marker_rows.append({
                "source": src.name, **{k: b[k] for k in
                    ("blob", "area_px", "x0", "y0", "x1", "y1", "bbox_w", "bbox_h", "fill")},
                "tiles": " ".join(blob_tiles[b["blob"]]),
            })

    except Exception as exc:  # noqa: BLE001 - report and keep going
        return src.name, tile_rows, marker_rows, f"{type(exc).__name__}: {exc}"
    return src.name, tile_rows, marker_rows, None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, default=DATASET / "raw", help="directory of source images")
    ap.add_argument("--output", type=Path, default=None, help="output directory (default: the dataset processed/ dir)")
    ap.add_argument("--pattern", default="*.JPG", help="glob for source images (default: *.JPG)")
    ap.add_argument("--tile", type=int, default=1024, help="tile size in pixels (default: 1024)")
    ap.add_argument("--edge", choices=("shift", "pad", "crop"), default="shift", help="how to handle partial edge tiles")
    ap.add_argument("--keep", choices=("markers", "all"), default="markers", help="write only marker tiles (default) or every tile")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (default: 95)")
    ap.add_argument("--fill", default="0,0,0", help="pad colour as R,G,B (default: 0,0,0)")
    ap.add_argument("--keep-exif", action="store_true", help="copy source EXIF into every tile")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N images")
    ap.add_argument("--workers", type=int, default=4, help="parallel processes (default: 4; ~400MB peak each)")
    ap.add_argument("--overwrite", action="store_true", help="rewrite tiles that already exist")
    ap.add_argument("--dry-run", action="store_true", help="detect and report, write no tiles")
    ap.add_argument("--clean", action="store_true", help="delete tiles in the output dir that are not kept this run")
    ap.add_argument("--fresh", action="store_true", help="delete the entire output directory before running")
    # detection tuning
    ap.add_argument("--v-min", type=int, default=215, help="min HSV value for marker pixels (default: 215)")
    ap.add_argument("--s-max", type=int, default=45, help="max HSV saturation for marker pixels (default: 45)")
    ap.add_argument("--open-size", type=int, default=3, help="morphological opening kernel (default: 3)")
    ap.add_argument("--close-iter", type=int, default=8, help="3x3 closing iterations to merge board fragments (default: 8)")
    ap.add_argument("--min-blob-area", type=int, default=800, help="min solid blob area in px (default: 800)")
    ap.add_argument("--min-tile-px", type=int, default=200, help="min marker px for a tile to be kept (default: 200)")
    args = ap.parse_args()

    in_dir: Path = args.input.resolve()
    out_dir: Path = (args.output or DATASET / "processed").resolve()

    if not in_dir.is_dir():
        print(f"error: input directory not found: {in_dir}", file=sys.stderr)
        return 1

    sources = sorted(p for p in in_dir.glob(args.pattern) if p.is_file())
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        print(f"error: no images matching {args.pattern!r} in {in_dir}", file=sys.stderr)
        return 1

    try:
        fill = tuple(int(v) for v in args.fill.split(","))
        if len(fill) != 3:
            raise ValueError
    except ValueError:
        print(f"error: --fill must be three comma-separated ints, got {args.fill!r}", file=sys.stderr)
        return 1

    if args.fresh:
        if out_dir == in_dir or in_dir == out_dir.parent and out_dir.name == in_dir.name:
            print(f"error: refusing --fresh on the source directory: {out_dir}", file=sys.stderr)
            return 1
        if out_dir.exists():
            if args.dry_run:
                print(f"fresh    : would delete {out_dir}")
            else:
                shutil.rmtree(out_dir)
                print(f"fresh    : deleted {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)

    print(f"input    : {in_dir}")
    print(f"output   : {out_dir}")
    print(f"images   : {len(sources)}")
    print(f"tile     : {args.tile}x{args.tile}  edge={args.edge}  keep={args.keep}  quality={args.quality}  workers={workers}")
    print(f"detect   : v>{args.v_min} s<{args.s_max} open={args.open_size} close={args.close_iter} min_blob={args.min_blob_area}px min_tile={args.min_tile_px}px")
    if args.dry_run:
        print("mode     : DRY RUN (no tiles written)")
    print()

    tile_rows: list[dict] = []
    marker_rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    job = dict(
        out_dir=out_dir, tile=args.tile, edge=args.edge, quality=args.quality,
        fill=fill, keep_exif=args.keep_exif, overwrite=args.overwrite,
        keep_mode=args.keep, v_min=args.v_min, s_max=args.s_max,
        open_size=args.open_size, close_iter=args.close_iter, min_blob_area=args.min_blob_area,
        min_tile_px=args.min_tile_px, dry_run=args.dry_run,
    )

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, src, **job): src for src in sources}
        for i, fut in enumerate(as_completed(futures), start=1):
            name, rows, markers, err = fut.result()
            tile_rows.extend(rows)
            marker_rows.extend(markers)
            if err:
                failures.append((name, err))
                print(f"[{i}/{len(sources)}] {name}: FAILED {err}")
            else:
                kept = sum(r["kept"] for r in rows)
                tiles = sorted({f"r{r['row']}_c{r['col']}" for r in rows if r["kept"]})
                shown = " ".join(tiles) if len(tiles) <= 10 else " ".join(tiles[:10]) + " ..."
                print(f"[{i}/{len(sources)}] {name}: {len(markers)} markers -> {kept}/{len(rows)} tiles  {shown}")

    tile_rows.sort(key=lambda r: (r["source"], r["row"], r["col"]))
    marker_rows.sort(key=lambda r: (r["source"], -r["area_px"]))

    if args.dry_run:
        # A dry run must not touch the CSVs; they describe the last real run.
        print("\nDRY RUN, no tiles written and manifest/markers left untouched")
        print(f"would keep : {sum(r['kept'] for r in tile_rows)} of {len(tile_rows)} tiles")
        return 1 if failures else 0

    with (out_dir / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(tile_rows)
    with (out_dir / "markers.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MARKER_FIELDS)
        w.writeheader()
        w.writerows(marker_rows)

    kept_names = {r["tile"] for r in tile_rows if r["kept"]}
    removed = 0
    if args.clean and not args.dry_run:
        for p in out_dir.glob("*.jpg"):
            if p.name not in kept_names:
                p.unlink()
                removed += 1

    imgs_with = len({r["source"] for r in marker_rows})
    print()
    print(f"images with markers : {imgs_with}/{len(sources)}")
    print(f"markers detected    : {len(marker_rows)}")
    print(f"tiles kept          : {len(kept_names)} of {len(tile_rows)} considered")
    if args.clean and not args.dry_run:
        print(f"stale tiles removed : {removed}")
    print(f"manifest            : {out_dir / 'manifest.csv'}")
    print(f"markers             : {out_dir / 'markers.csv'}")
    if failures:
        print(f"failures            : {len(failures)}")
        for name, err in failures:
            print(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
