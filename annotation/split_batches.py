#!/usr/bin/env python3
"""Split the marker tiles in processed/ into fixed-size batch subdirectories.

Tiles are moved, in sorted order, into batch_01, batch_02, ... of --size each
(the last batch holds the remainder). Sorted order keeps tiles from the same
source image adjacent, so a source is split across a batch boundary at most once.

Because the tiles move, anything referring to them is rewritten too:

  <batch>/cvat_preannot.xml   per-batch CVAT import, bare filenames, so each
                              batch folder can become its own CVAT task
  processed/cvat_preannot.xml combined import, names prefixed with the batch dir,
                              for one task over everything
  processed/manifest.csv      gains a `batch` column for kept tiles

Idempotent: re-running collects the tiles back out of existing batch dirs and
re-splits them, so changing --size is safe.

Usage:
    python3 split_batches.py                 # 100 per batch
    python3 split_batches.py --size 50
    python3 split_batches.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom

# Dataset lives outside the repo tree in git terms (see .gitignore), one level up
# from this scripts directory.
DATASET = Path(__file__).resolve().parent.parent / "chinee-apple-dataset" / "matured-dry-30m"

BATCH_RE = re.compile(r"^batch_\d+$")


def collect_tiles(proc: Path) -> list[Path]:
    """All tile jpgs, whether loose in processed/ or already in batch dirs."""
    tiles = list(proc.glob("*.jpg"))
    for d in proc.iterdir():
        if d.is_dir() and BATCH_RE.match(d.name):
            tiles.extend(d.glob("*.jpg"))
    return sorted(tiles, key=lambda p: p.name)


def write_xml(root: ET.Element, path: Path) -> None:
    xml = minidom.parseString(ET.tostring(root, "utf-8")).toprettyxml(indent="  ")
    path.write_text("\n".join(l for l in xml.splitlines() if l.strip()) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", type=Path, default=DATASET / "processed")
    ap.add_argument("--size", type=int, default=100, help="tiles per batch (default: 100)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    proc: Path = args.processed.resolve()
    tiles = collect_tiles(proc)
    if not tiles:
        print(f"error: no tiles found in {proc}")
        return 1

    n_batches = (len(tiles) + args.size - 1) // args.size
    width = max(2, len(str(n_batches)))
    assign: dict[str, str] = {}
    batches: dict[str, list[Path]] = {}
    for i, t in enumerate(tiles):
        name = f"batch_{i // args.size + 1:0{width}d}"
        assign[t.name] = name
        batches.setdefault(name, []).append(t)

    print(f"tiles   : {len(tiles)}")
    print(f"batches : {n_batches} x {args.size} -> " +
          ", ".join(f"{b}={len(v)}" for b, v in sorted(batches.items())))
    if args.dry_run:
        print("\nDRY RUN - nothing moved")
        return 0

    for bname, members in sorted(batches.items()):
        bdir = proc / bname
        bdir.mkdir(exist_ok=True)
        for src in members:
            dest = bdir / src.name
            if src.resolve() != dest.resolve():
                shutil.move(str(src), str(dest))

    # drop any now-empty batch dirs left over from a previous, larger split
    for d in sorted(proc.iterdir()):
        if d.is_dir() and BATCH_RE.match(d.name) and not any(d.glob("*.jpg")):
            shutil.rmtree(d)

    # ---- manifest gains a batch column -------------------------------------
    man = proc / "manifest.csv"
    if man.exists():
        rows = list(csv.DictReader(man.open()))
        fields = [f for f in rows[0].keys() if f != "batch"] + ["batch"]
        for r in rows:
            r["batch"] = assign.get(r["tile"], "") if r["kept"] == "1" else ""
        with man.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"manifest: +batch column ({man})")

    # ---- rewrite the CVAT XMLs ---------------------------------------------
    combined = proc / "cvat_preannot.xml"
    if not combined.exists():
        print("note: no cvat_preannot.xml found, skipping XML rewrite")
        return 0

    tree = ET.parse(combined)
    root = tree.getroot()
    meta = root.find("meta")

    per_batch: dict[str, list[ET.Element]] = {}
    for img in root.findall("image"):
        base = Path(img.get("name")).name          # tolerate an earlier prefix
        b = assign.get(base)
        if b is None:
            continue
        img.set("name", f"{b}/{base}")
        per_batch.setdefault(b, []).append((base, img))

    write_xml(root, combined)
    print(f"combined: {combined} ({sum(len(v) for v in per_batch.values())} images, batch-prefixed)")

    for bname, items in sorted(per_batch.items()):
        broot = ET.Element("annotations")
        ET.SubElement(broot, "version").text = "1.1"
        if meta is not None:
            import copy
            broot.append(copy.deepcopy(meta))
        for i, (base, img) in enumerate(sorted(items)):
            import copy
            c = copy.deepcopy(img)
            c.set("name", base)                     # bare name within the batch task
            c.set("id", str(i))
            broot.append(c)
        out = proc / bname / "cvat_preannot.xml"
        write_xml(broot, out)
        print(f"  {bname}: {len(items)} images -> {out.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
