#!/usr/bin/env python3
"""Build CVAT pre-annotations for the marker tiles.

Reads the tiling outputs (processed/manifest.csv + processed/markers.csv), and for
every detected cardboard marker works out:

  arrow_colour   blue | pink | blank   - the sprayed arrow, which encodes the
                                         plant class/condition. "blank" means a
                                         board with no arrow on it at all.
  arrow_deg      empty by default      - automatic arrow-bearing estimation was
                                         tested and is only ~50% correct at this
                                         GSD (strokes are ~15px wide), so it is
                                         OFF. --bearing turns it on, but do not
                                         trust it without checking. The annotator
                                         reads the arrow direction by eye.

Only the plant is annotated, under the label chinee_apple_matured_30m, which is
also used verbatim as the SAM3 text prompt. Markers are NOT annotation objects,
they stay visible
in the imagery as a reference the annotator reads by eye. The script therefore
writes a single-label schema (cvat_labels.json) for task creation, plus a
crib sheet (tile_markers.csv) listing which board sits in which tile, with its uid
and arrow colour, so the annotator can fill in marker_uid without drawing a box.
Pass --include-markers to also export marker rectangles as a pre-annotation XML.

ARROW DIRECTION (disabled)
Two heuristics were tried for recovering which way the arrow points: mean
perpendicular spread per half (read backwards, ~0% correct) and perpendicular
extent at the outer ends (~50% correct). Neither is usable - the strokes are only
~15px wide at 30m, so head and tail are not reliably distinguishable. The code is
kept behind --bearing for future work at lower altitude, but is off by default.

Usage:
    python3 make_cvat_task.py                      # plants only (default)
    python3 make_cvat_task.py --include-markers    # also export marker rectangles
    python3 make_cvat_task.py --preview 12         # also render a validation sheet
"""

from __future__ import annotations

import argparse
import csv
import collections
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

# Dataset lives outside the repo tree in git terms (see .gitignore), one level up
# from this scripts directory.
DATASET = Path(__file__).resolve().parent.parent / "chinee-apple-dataset" / "matured-dry-30m"


# Hue windows in PIL's 0-255 space. Pink is kept away from the 0-20 red-brown
# band because sunlit soil at the board edge otherwise reads as pink.
BLUE = (105, 175)
PINK = (210, 252)
# The single annotation label. It doubles as the text prompt handed to SAM3,
# so it is also what the model is asked to segment.
PLANT_LABEL = "chinee_apple_matured_30m"

MIN_PAINT_PX = 60          # below this a board counts as blank
MIN_BLANK_AREA = 6000      # only call a board "blank" if it is big enough to be sure


def arrow_mask(hsv: np.ndarray, lo: int, hi: int, s_min: int, v_min: int) -> np.ndarray:
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    return (h >= lo) & (h <= hi) & (s > s_min) & (v > v_min)


def arrow_bearing(mask: np.ndarray) -> float | None:
    """Bearing in degrees the arrow points: 0=right, 90=down. None if indeterminate."""
    ys, xs = np.nonzero(mask)
    if ys.size < 40:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(axis=0)
    # principal axis of the stroke
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    axis = vt[0]
    perp = np.array([-axis[1], axis[0]])
    t = pts @ axis          # position along the stroke
    d = np.abs(pts @ perp)  # perpendicular offset from the stroke

    # Compare the two ENDS, not the two halves: averaging over a half is diluted
    # by the long shaft, which made this read backwards. The barbed head flares
    # perpendicular to the axis, so measure perpendicular extent in the outer
    # 25% of each end.
    lo, hi = np.quantile(t, 0.25), np.quantile(t, 0.75)
    tail_end, head_end = d[t <= lo], d[t >= hi]
    if tail_end.size < 10 or head_end.size < 10:
        return None
    pos_width = float(np.ptp(head_end))
    neg_width = float(np.ptp(tail_end))
    if max(pos_width, neg_width) == 0:
        return None
    # require a clear winner; an ambiguous arrow is better left unlabelled
    if abs(pos_width - neg_width) / max(pos_width, neg_width) < 0.15:
        return None
    vec = axis if pos_width > neg_width else -axis
    return math.degrees(math.atan2(vec[1], vec[0])) % 360


def classify(src_path: Path, rows: list[dict], bearing_on: bool = False) -> list[dict]:
    hsv_full = np.asarray(Image.open(src_path).convert("RGB").convert("HSV"))
    out = []
    for r in rows:
        x0, y0, x1, y1 = (int(r[k]) for k in ("x0", "y0", "x1", "y1"))
        reg = hsv_full[y0:y1, x0:x1]
        bm = arrow_mask(reg, *BLUE, 60, 60)
        pm = arrow_mask(reg, *PINK, 80, 80)
        nb, np_ = int(bm.sum()), int(pm.sum())

        if max(nb, np_) < MIN_PAINT_PX:
            colour = "blank" if int(r["area_px"]) >= MIN_BLANK_AREA else "unknown"
            bearing = None
        elif nb >= np_:
            colour, bearing = "blue", (arrow_bearing(bm) if bearing_on else None)
        else:
            colour, bearing = "pink", (arrow_bearing(pm) if bearing_on else None)

        out.append({
            **r,
            "arrow_colour": colour,
            "blue_px": nb,
            "pink_px": np_,
            "arrow_deg": "" if bearing is None else f"{bearing:.1f}",
        })
    return out


def build_xml(tiles: list[dict], markers: list[dict], out_path: Path,
              include_markers: bool = False) -> int:
    by_tile: dict[str, list[dict]] = collections.defaultdict(list)
    tile_index = {(t["source"], f"r{t['row']}_c{t['col']}"): t for t in tiles}
    for m in markers:
        for tname in m["tiles"].split():
            key = (m["source"], tname)
            t = tile_index.get(key)
            if t is None or t["kept"] != "1":
                continue
            by_tile[t["tile"]].append((m, t))

    ann = ET.Element("annotations")
    ET.SubElement(ann, "version").text = "1.1"
    meta = ET.SubElement(ann, "meta")
    task = ET.SubElement(meta, "task")
    labels = ET.SubElement(task, "labels")

    def add_label(name, ltype, colour, attrs=()):
        lab = ET.SubElement(labels, "label")
        ET.SubElement(lab, "name").text = name
        ET.SubElement(lab, "type").text = ltype
        ET.SubElement(lab, "color").text = colour
        a = ET.SubElement(lab, "attributes")
        for an, itype, default, values, mutable in attrs:
            at = ET.SubElement(a, "attribute")
            ET.SubElement(at, "name").text = an
            ET.SubElement(at, "mutable").text = "true" if mutable else "false"
            ET.SubElement(at, "input_type").text = itype
            ET.SubElement(at, "default_value").text = default
            ET.SubElement(at, "values").text = values

    if include_markers:
        add_label("marker", "rectangle", "#00e5ff", [
            ("arrow_colour", "select", "blank", "blue\npink\nblank\nunknown", True),
            ("arrow_deg", "text", "", "", True),
            ("marker_uid", "text", "", "", False),
        ])
    add_label(PLANT_LABEL, "polygon", "#32cd32", [
        ("marker_uid", "text", "", "", True),
        ("class", "select", "unsure", "blue\npink\nblank\nunsure", True),
        ("partial", "checkbox", "false", "", True),
    ])

    n_boxes = 0
    for i, tname in enumerate(sorted(by_tile)):
        t = by_tile[tname][0][1]
        img = ET.SubElement(ann, "image", id=str(i), name=tname,
                            width=str(t["width"]), height=str(t["height"]))
        if not include_markers:
            continue
        tx, ty = int(t["x0"]), int(t["y0"])
        for m, _ in by_tile[tname]:
            # full-image coords -> tile-local, clipped to the tile
            xtl = max(0, int(m["x0"]) - tx); ytl = max(0, int(m["y0"]) - ty)
            xbr = min(int(t["width"]), int(m["x1"]) - tx)
            ybr = min(int(t["height"]), int(m["y1"]) - ty)
            if xbr <= xtl or ybr <= ytl:
                continue
            box = ET.SubElement(img, "box", label="marker", occluded="0", source="auto",
                                xtl=f"{xtl}", ytl=f"{ytl}", xbr=f"{xbr}", ybr=f"{ybr}", z_order="0")
            uid = f"{Path(m['source']).stem}#{m['blob']}"
            for an, av in (("arrow_colour", m["arrow_colour"]),
                           ("arrow_deg", m["arrow_deg"]),
                           ("marker_uid", uid)):
                ET.SubElement(box, "attribute", name=an).text = str(av)
            n_boxes += 1

    xml = minidom.parseString(ET.tostring(ann, "utf-8")).toprettyxml(indent="  ")
    out_path.write_text(xml)
    return n_boxes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=DATASET / "raw", help="directory of source images")
    ap.add_argument("--processed", type=Path, default=DATASET / "processed", help="tiling output directory")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--include-markers", action="store_true", help="also export marker rectangles (default: plants only)")
    ap.add_argument("--bearing", action="store_true", help="estimate arrow bearing (UNRELIABLE at this GSD; ~50%% correct)")
    ap.add_argument("--preview", type=int, default=0, help="render N validation crops with the arrow overlay")
    args = ap.parse_args()

    tiles = list(csv.DictReader(open(args.processed / "manifest.csv")))
    markers = list(csv.DictReader(open(args.processed / "markers.csv")))
    by_src: dict[str, list[dict]] = collections.defaultdict(list)
    for m in markers:
        by_src[m["source"]].append(m)

    classified: list[dict] = []
    for src in sorted(by_src):
        classified.extend(classify(args.raw / src, by_src[src], args.bearing))

    fields = list(markers[0].keys()) + ["arrow_colour", "blue_px", "pink_px", "arrow_deg"]
    with (args.processed / "markers_classified.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(classified)

    counts = collections.Counter(m["arrow_colour"] for m in classified)
    with_dir = sum(1 for m in classified if m["arrow_deg"])
    print("arrow colour:")
    for k, v in counts.most_common():
        print(f"  {k:<8} {v:>4}  ({100*v/len(classified):.1f}%)")
    print(f"bearing resolved: {with_dir}/{len(classified)}")

    # Label schema for task creation. Plants only, unless markers are asked for.
    labels = []
    if args.include_markers:
        labels.append({"name": "marker", "type": "rectangle", "color": "#00e5ff", "attributes": [
            {"name": "arrow_colour", "mutable": True, "input_type": "select",
             "default_value": "blank", "values": ["blue", "pink", "blank", "unknown"]},
            {"name": "marker_uid", "mutable": False, "input_type": "text",
             "default_value": "", "values": []},
        ]})
    labels.append({"name": PLANT_LABEL, "type": "polygon", "color": "#32cd32", "attributes": [
        {"name": "marker_uid", "mutable": True, "input_type": "text",
         "default_value": "", "values": []},
        {"name": "class", "mutable": True, "input_type": "select",
         "default_value": "unsure", "values": ["blue", "pink", "blank", "unsure"]},
        {"name": "partial", "mutable": True, "input_type": "checkbox",
         "default_value": "false", "values": []},
    ]})
    lab_path = args.processed / "cvat_labels.json"
    lab_path.write_text(json.dumps(labels, indent=2) + "\n")
    print(f"\nlabel schema  {lab_path}  ({', '.join(l['name'] for l in labels)})")

    # Crib sheet. Which boards sit in each tile, so the annotator can copy the uid
    # and read the expected class without the markers being annotation objects.
    tile_index = {(t["source"], f"r{t['row']}_c{t['col']}"): t for t in tiles}
    crib = []
    for m in classified:
        for tn in m["tiles"].split():
            t = tile_index.get((m["source"], tn))
            if t is None or t["kept"] != "1":
                continue
            crib.append({
                "tile": t["tile"],
                "marker_uid": f"{Path(m['source']).stem}#{m['blob']}",
                "arrow_colour": m["arrow_colour"],
                "marker_x0": int(m["x0"]) - int(t["x0"]),
                "marker_y0": int(m["y0"]) - int(t["y0"]),
                "marker_x1": int(m["x1"]) - int(t["x0"]),
                "marker_y1": int(m["y1"]) - int(t["y0"]),
            })
    crib.sort(key=lambda r: (r["tile"], r["marker_uid"]))
    crib_path = args.processed / "tile_markers.csv"
    with crib_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(crib[0].keys()))
        w.writeheader(); w.writerows(crib)
    print(f"crib sheet    {crib_path}  ({len(crib)} board sightings)")

    out = args.processed / "cvat_preannot.xml"
    if args.include_markers:
        n = build_xml(tiles, classified, out, include_markers=True)
        n_imgs = len({t["tile"] for t in tiles if t["kept"] == "1"})
        print(f"CVAT XML      {out}  ({n} marker boxes across {n_imgs} tiles)")
    else:
        for stale in [out, *args.processed.glob("batch_*/cvat_preannot.xml")]:
            if stale.exists():
                stale.unlink()
        print("CVAT XML      not written, plants only. Define labels from cvat_labels.json")

    if args.preview:
        render_preview(args.raw, classified, args.preview, args.processed / "arrow_preview.png")
        print(f"preview : {args.processed / 'arrow_preview.png'}")
    return 0


def render_preview(raw: Path, classified: list[dict], n: int, out: Path) -> None:
    sel = [m for m in classified if m["arrow_deg"]][:n]
    S, cols = 200, 6
    rows = (len(sel) + cols - 1) // cols
    sheet = Image.new("RGB", (S * cols, S * rows), (20, 20, 20))
    cache: dict[str, Image.Image] = {}
    for i, m in enumerate(sel):
        if m["source"] not in cache:
            cache[m["source"]] = Image.open(raw / m["source"]).convert("RGB")
        im = cache[m["source"]]
        cx = (int(m["x0"]) + int(m["x1"])) // 2
        cy = (int(m["y0"]) + int(m["y1"])) // 2
        h = 150
        c = im.crop((cx - h, cy - h, cx + h, cy + h)).resize((S, S), Image.LANCZOS)
        d = ImageDraw.Draw(c)
        ang = math.radians(float(m["arrow_deg"]))
        x0, y0 = S / 2, S / 2
        x1, y1 = x0 + 70 * math.cos(ang), y0 + 70 * math.sin(ang)
        d.line([x0, y0, x1, y1], fill=(255, 255, 0), width=3)
        d.ellipse([x1 - 5, y1 - 5, x1 + 5, y1 + 5], fill=(255, 0, 0))
        d.text((4, 4), f"{m['arrow_colour']} {float(m['arrow_deg']):.0f}deg", fill=(255, 255, 0))
        sheet.paste(c, ((i % cols) * S, (i // cols) * S))
    sheet.save(out)


if __name__ == "__main__":
    raise SystemExit(main())
