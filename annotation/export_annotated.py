#!/usr/bin/env python3
"""Collect only the annotated images out of one or more CVAT exports.

CVAT always exports every frame in a task, including the ones nobody annotated.
This walks one or more exports, keeps only images carrying at least one
annotation, merges them into a single dataset directory, and writes a filtered
annotation file alongside.

Handles the two export formats that matter for polygon work:

  COCO 1.0              annotations/instances_*.json  (segmentation polygons)
  CVAT for images 1.1   annotations.xml               (polygon / box / mask elements)

Inputs may be .zip files straight from CVAT, or already-extracted directories,
and may be mixed. Several batch exports can be merged in one go.

Tile filenames already encode source image plus row and column, so they are
unique across batches. A genuine name collision between two different files is
reported and skipped rather than silently overwriting.

Usage:
    python3 export_annotated.py --input exports/*.zip --output dataset_v1
    python3 export_annotated.py --input exports/ --dry-run
    python3 export_annotated.py --input a.zip b.zip --output out --format coco
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

SHAPE_TAGS = ("polygon", "box", "mask", "polyline", "points", "ellipse", "cuboid")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack(src: Path, workdir: Path) -> Path:
    """Return a directory for `src`, extracting it first if it is a zip."""
    if src.is_dir():
        return src
    if src.suffix.lower() != ".zip":
        raise ValueError(f"not a directory or .zip: {src}")
    dest = workdir / src.stem
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as z:
        z.extractall(dest)
    return dest


def index_images(root: Path) -> dict[str, Path]:
    """Map basename -> path for every image file under root."""
    out: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.setdefault(p.name, p)
    return out


def read_cvat_xml(path: Path) -> tuple[dict[str, int], ET.ElementTree]:
    """Return {image name: shape count} and the parsed tree."""
    tree = ET.parse(path)
    counts: dict[str, int] = {}
    for img in tree.getroot().findall("image"):
        name = Path(img.get("name", "")).name
        n = sum(len(img.findall(tag)) for tag in SHAPE_TAGS)
        counts[name] = counts.get(name, 0) + n
    return counts, tree


def read_coco(path: Path) -> tuple[dict[str, int], dict]:
    data = json.loads(path.read_text())
    per_id: dict[int, int] = defaultdict(int)
    for a in data.get("annotations", []):
        per_id[a["image_id"]] += 1
    counts: dict[str, int] = {}
    for im in data.get("images", []):
        counts[Path(im["file_name"]).name] = per_id.get(im["id"], 0)
    return counts, data


def find_annotation_files(root: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for p in sorted(root.rglob("*.xml")):
        try:
            if ET.parse(p).getroot().tag == "annotations":
                found.append(("cvat", p))
        except ET.ParseError:
            continue
    for p in sorted(root.rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(d, dict) and "images" in d and "annotations" in d:
            found.append(("coco", p))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, nargs="+", required=True, help="CVAT export zips and/or directories")
    ap.add_argument("--output", type=Path, default=Path("annotated_dataset"), help="output dataset directory")
    ap.add_argument("--format", choices=("auto", "coco", "cvat"), default="auto",
                    help="which annotation file to trust if an export contains both")
    ap.add_argument("--dry-run", action="store_true", help="report only, copy nothing")
    ap.add_argument("--move", action="store_true", help="move images instead of copying")
    args = ap.parse_args()

    keep: dict[str, Path] = {}          # image name -> source path
    shapes: dict[str, int] = {}         # image name -> shape count
    cvat_imgs: list[ET.Element] = []
    coco_images: list[dict] = []
    coco_anns: list[dict] = []
    coco_cats: list[dict] = []
    seen_total = 0
    collisions: list[str] = []
    per_export: list[tuple[str, int, int]] = []

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for src in args.input:
            if not src.exists():
                print(f"error: no such input: {src}", file=sys.stderr)
                return 1
            root = unpack(src, work)
            ann_files = find_annotation_files(root)
            if args.format != "auto":
                ann_files = [a for a in ann_files if a[0] == args.format]
            if not ann_files:
                print(f"warning: no annotation file found in {src.name}, skipped", file=sys.stderr)
                continue
            kind, ann_path = ann_files[0]
            images = index_images(root)

            if kind == "cvat":
                counts, tree = read_cvat_xml(ann_path)
                for img in tree.getroot().findall("image"):
                    name = Path(img.get("name", "")).name
                    if counts.get(name, 0) > 0:
                        img.set("name", name)
                        cvat_imgs.append(img)
            else:
                counts, data = read_coco(ann_path)
                if not coco_cats:
                    coco_cats = data.get("categories", [])
                wanted = {im["id"]: im for im in data.get("images", [])
                          if counts.get(Path(im["file_name"]).name, 0) > 0}
                for im in wanted.values():
                    im = dict(im)
                    im["file_name"] = Path(im["file_name"]).name
                    coco_images.append(im)
                for a in data.get("annotations", []):
                    if a["image_id"] in wanted:
                        coco_anns.append(a)

            seen_total += len(counts)
            kept_here = 0
            for name, n in counts.items():
                if n <= 0:
                    continue
                path = images.get(name)
                if path is None:
                    print(f"warning: {name} is annotated but its image is missing from {src.name}", file=sys.stderr)
                    continue
                if name in keep:
                    if digest(keep[name]) != digest(path):
                        collisions.append(name)
                        continue
                    shapes[name] += n          # same file in two exports, sum the shapes
                    continue
                keep[name] = path
                shapes[name] = n
                kept_here += 1
            per_export.append((src.name, len(counts), kept_here))

        print(f"{'export':<40}{'frames':>9}{'annotated':>11}")
        for name, total, kept in per_export:
            print(f"{name:<40}{total:>9}{kept:>11}")
        print(f"{'TOTAL':<40}{seen_total:>9}{len(keep):>11}")
        empty = seen_total - len(keep)
        if seen_total:
            print(f"\nempty frames dropped: {empty} ({100*empty/seen_total:.1f}%)")
        print(f"shapes kept         : {sum(shapes.values())}")
        if collisions:
            print(f"\nNAME COLLISIONS, skipped: {len(collisions)}", file=sys.stderr)
            for c in collisions[:10]:
                print(f"  {c}", file=sys.stderr)

        if args.dry_run:
            print("\nDRY RUN, nothing written")
            return 0
        if not keep:
            print("\nnothing to export", file=sys.stderr)
            return 1

        out = args.output.resolve()
        img_dir = out / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for name, path in sorted(keep.items()):
            dest = img_dir / name
            if args.move:
                shutil.move(str(path), dest)
            else:
                shutil.copy2(path, dest)

        if cvat_imgs:
            root_el = ET.Element("annotations")
            ET.SubElement(root_el, "version").text = "1.1"
            for i, img in enumerate(sorted(cvat_imgs, key=lambda e: e.get("name"))):
                img.set("id", str(i))
                root_el.append(img)
            ET.ElementTree(root_el).write(out / "annotations.xml", encoding="utf-8", xml_declaration=True)
            print(f"\nwrote {out / 'annotations.xml'}")
        if coco_images:
            (out / "instances.json").write_text(json.dumps({
                "images": sorted(coco_images, key=lambda i: i["file_name"]),
                "annotations": coco_anns,
                "categories": coco_cats,
            }, indent=1))
            print(f"wrote {out / 'instances.json'}")

        with (out / "manifest.csv").open("w") as fh:
            fh.write("image,shapes\n")
            for name in sorted(keep):
                fh.write(f"{name},{shapes[name]}\n")
        print(f"wrote {out / 'manifest.csv'}")
        print(f"images              : {img_dir}  ({len(keep)} files)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
