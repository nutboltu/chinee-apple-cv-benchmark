"""Fast visual triage for UAV frames: find the images worth annotating.

The problem
-----------
Thousands of full-resolution DJI frames sit in Google Drive. Only some of them
actually contain chinee apple, and the only way to know is to look at them.

Measured on real frames from this project (8192x5460, ~26 MB each):

    local disk read        9 ms/frame
    JPEG decode          110-170 ms/frame
    SAME read over Drive  ~2.8 s/frame at 10 MB/s

So decoding is not the problem -- moving 26 MB per frame across the network is.
Triaging 3000 frames that way is ~2.5 hours of pure transfer, and it is paid
again every time you scroll back to re-check something.

The fix is to pay that transfer ONCE and never re-read an original during
triage:

  1. `thumbs` makes one threaded pass over the originals and writes a ~100 KB
     proxy per frame. Run it WHERE THE DATA IS -- on a Colab VM next to the
     Drive mount, not on a laptop pulling through a Drive client. That one
     pass is the only time full-resolution bytes move.
  2. Everything after that reads proxies: 3000 proxies are ~300 MB total
     instead of ~78 GB, so the whole set fits on a laptop and opens instantly.
  3. You triage in bulk, not one frame at a time: `sheets` writes contact
     sheets with dozens of frames per page, `review` writes a self-contained
     HTML page with keyboard keep/skip that exports a CSV.
  4. `plan` turns the kept list back into real originals for CVAT. You only
     ever upload and annotate frames that have the weed in them.

Why `draft()` is used in the proxy pass: it decodes in the JPEG DCT domain at
1/2, 1/4 or 1/8 scale. That is only ~1.2x faster than a full decode (Huffman
decoding dominates, and draft only shrinks the IDCT), but it cuts peak memory
per frame from ~163 MB to under 10 MB. That is what lets --workers run at 16-32
without exhausting RAM, and high concurrency is exactly what hides Drive's
per-file latency.

Typical run
-----------
    # once, wherever the originals are readable (Colab next to the Drive mount
    # is far faster than a local Drive client)
    python fast_triage.py thumbs --src /content/drive/MyDrive/uav --out proxies

    # then, on your laptop, against the (small) proxy folder
    python fast_triage.py review --proxies proxies --out triage.html
    python fast_triage.py sheets --proxies proxies --out sheets
    python fast_triage.py plan   --decisions decisions.csv --src /path/uav --out keep

    # optional, any time: EXIF/GPS manifest without decoding pixels
    python fast_triage.py index --src /content/drive/MyDrive/uav --out manifest.csv
"""

import argparse
import concurrent.futures as cf
import csv
import html
import json
import math
import os
import shutil
import sys
import time

from PIL import Image, ImageDraw, ImageOps

# Pillow refuses very large files by default; UAV frames are legitimately big.
Image.MAX_IMAGE_PIXELS = None

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
GPS_IFD = 0x8825
EXIF_IFD = 0x8769
DATETIME_ORIGINAL = 0x9003
DATETIME = 0x0132


# --------------------------------------------------------------------------- #
# Input discovery
# --------------------------------------------------------------------------- #
def list_images(src: str) -> list[str]:
    """Accept a directory (walked recursively) or a text file of paths."""
    if os.path.isfile(src) and os.path.splitext(src)[1].lower() not in EXTS:
        with open(src) as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    out = []
    for root, _, files in os.walk(src):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS and not f.startswith("."):
                out.append(os.path.join(root, f))
    return sorted(out)


def proxy_name(src_path: str, src_root: str) -> str:
    """Flatten a nested source path into one collision-free proxy filename."""
    rel = os.path.relpath(src_path, src_root) if os.path.isdir(src_root) else os.path.basename(src_path)
    stem = os.path.splitext(rel)[0].replace(os.sep, "__")
    return stem + ".jpg"


# --------------------------------------------------------------------------- #
# thumbs -- the one pass that touches full-resolution frames
# --------------------------------------------------------------------------- #
def make_proxy(src: str, dst: str, size: int, quality: int) -> None:
    with Image.open(src) as im:
        # The whole point: draft() picks a 1/2, 1/4 or 1/8 DCT scale and decodes
        # only that, so we never materialise the full-resolution bitmap.
        im.draft("RGB", (size, size))
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.BILINEAR)
        tmp = dst + ".part"
        im.save(tmp, "JPEG", quality=quality)
    os.replace(tmp, dst)


def cmd_thumbs(args) -> None:
    srcs = list_images(args.src)
    if not srcs:
        sys.exit(f"no images found under {args.src}")
    os.makedirs(args.out, exist_ok=True)

    jobs = []
    for s in srcs:
        dst = os.path.join(args.out, proxy_name(s, args.src))
        if args.force or not os.path.exists(dst):
            jobs.append((s, dst))

    skipped = len(srcs) - len(jobs)
    print(f"[thumbs] {len(srcs)} frames, {len(jobs)} to build, {skipped} already done")
    if not jobs:
        return

    t0, done, failed = time.time(), 0, []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(make_proxy, s, d, args.size, args.quality): s for s, d in jobs}
        for fut in cf.as_completed(futs):
            src = futs[fut]
            try:
                fut.result()
            except Exception as exc:  # a corrupt frame must not kill the pass
                failed.append((src, repr(exc)))
            done += 1
            if done % 50 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(jobs) - done) / max(rate, 1e-6)
                print(f"  {done}/{len(jobs)}  {rate:.1f} img/s  eta {eta/60:.1f} min",
                      flush=True)

    print(f"[thumbs] wrote {done - len(failed)} proxies to {args.out} "
          f"in {(time.time()-t0)/60:.1f} min")
    if failed:
        log = os.path.join(args.out, "_failed.txt")
        with open(log, "w") as fh:
            for s, e in failed:
                fh.write(f"{s}\t{e}\n")
        print(f"[thumbs] {len(failed)} failed, listed in {log}")


# --------------------------------------------------------------------------- #
# index -- EXIF only, no pixel decode at all
# --------------------------------------------------------------------------- #
def _ratio(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _dms(vals, ref) -> float:
    d, m, s = (_ratio(x) for x in vals)
    deg = d + m / 60.0 + s / 3600.0
    return -deg if str(ref).upper() in ("S", "W") else deg


def exif_row(path: str) -> dict:
    row = {"path": path, "name": os.path.basename(path),
           "bytes": os.path.getsize(path), "width": "", "height": "",
           "datetime": "", "lat": "", "lon": "", "alt": ""}
    try:
        with Image.open(path) as im:          # header only, no decode
            row["width"], row["height"] = im.size
            exif = im.getexif()
        if exif:
            # DateTimeOriginal lives in the Exif sub-IFD, not the root IFD;
            # fall back to the root DateTime tag if the sub-IFD is absent.
            sub = exif.get_ifd(EXIF_IFD) or {}
            row["datetime"] = sub.get(DATETIME_ORIGINAL) or exif.get(DATETIME, "") or ""
            gps = exif.get_ifd(GPS_IFD) or {}
            if 2 in gps and 4 in gps:
                row["lat"] = round(_dms(gps[2], gps.get(1, "N")), 7)
                row["lon"] = round(_dms(gps[4], gps.get(3, "E")), 7)
            if 6 in gps:
                row["alt"] = round(_ratio(gps[6]), 2)
    except Exception as exc:
        row["datetime"] = f"ERROR {exc!r}"
    return row


def cmd_index(args) -> None:
    srcs = list_images(args.src)
    print(f"[index] reading EXIF from {len(srcs)} frames ({args.workers} threads)")
    with cf.ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(exif_row, srcs))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    located = sum(1 for r in rows if r["lat"] != "")
    print(f"[index] wrote {args.out} -- {len(rows)} frames, {located} with GPS")


# --------------------------------------------------------------------------- #
# sheets -- dozens of frames per page, for offline / printed scanning
# --------------------------------------------------------------------------- #
def cmd_sheets(args) -> None:
    proxies = list_images(args.proxies)
    if not proxies:
        sys.exit(f"no proxies found under {args.proxies} -- run `thumbs` first")
    os.makedirs(args.out, exist_ok=True)

    cell, cols, rows_n, cap = args.cell, args.cols, args.rows, 20
    per = cols * rows_n
    pages = 0
    for start in range(0, len(proxies), per):
        chunk = proxies[start:start + per]
        sheet = Image.new("RGB", (cols * cell, rows_n * (cell + cap)), (20, 20, 20))
        draw = ImageDraw.Draw(sheet)
        for i, p in enumerate(chunk):
            r, c = divmod(i, cols)
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    im.thumbnail((cell, cell), Image.BILINEAR)
                    sheet.paste(im, (c * cell + (cell - im.width) // 2,
                                     r * (cell + cap) + (cell - im.height) // 2))
            except Exception:
                continue
            draw.text((c * cell + 4, r * (cell + cap) + cell + 4),
                      f"{start + i:>5}  {os.path.basename(p)[:30]}", fill=(205, 205, 205))
        out = os.path.join(args.out, f"sheet_{pages:04d}.jpg")
        sheet.save(out, "JPEG", quality=88)
        pages += 1
    print(f"[sheets] wrote {pages} sheets ({per} frames each) to {args.out}")


# --------------------------------------------------------------------------- #
# review -- self-contained HTML, keyboard triage, CSV out
# --------------------------------------------------------------------------- #
REVIEW_CSS = """
:root{color-scheme:dark;--bg:#151515;--fg:#eee;--keep:#37b24d;--skip:#e03131}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:#101010;border-bottom:1px solid #2a2a2a;
  padding:10px 14px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
button{background:#262626;color:var(--fg);border:1px solid #3a3a3a;border-radius:6px;
  padding:6px 11px;cursor:pointer;font:inherit}
button:hover{background:#333}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--w,190px),1fr));
  gap:8px;padding:12px}
figure{margin:0;position:relative;border:3px solid transparent;border-radius:6px;
  overflow:hidden;cursor:pointer;background:#000}
figure img{display:block;width:100%;aspect-ratio:4/3;object-fit:cover}
figure figcaption{font-size:10px;padding:3px 5px;color:#aaa;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
figure.keep{border-color:var(--keep)}
figure.skip{border-color:var(--skip);opacity:.4}
figure.cursor{outline:3px solid #4dabf7;outline-offset:2px}
figure.keep::after{content:"KEEP";position:absolute;top:5px;left:5px;background:var(--keep);
  color:#fff;font-size:10px;font-weight:700;padding:1px 5px;border-radius:3px}
.hide{display:none}
#out{width:100%;height:150px;background:#0c0c0c;color:#ccc;border-top:1px solid #2a2a2a;
  font:11px ui-monospace,monospace;padding:8px;display:none}
kbd{background:#2a2a2a;border:1px solid #444;border-radius:3px;padding:1px 5px;font-size:11px}
"""

REVIEW_JS = """
const files = __FILES__;
const KEY = 'triage:' + location.pathname;
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
let cur = 0, onlyUndecided = false;

const grid = document.getElementById('grid');
grid.innerHTML = files.map((f, i) =>
  `<figure data-i="${i}"><img loading="lazy" src="${f.src}" alt=""><figcaption>${i} &middot; ${f.name}</figcaption></figure>`
).join('');
const cells = [...grid.children];

function save(){ try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
function paint(){
  cells.forEach((el, i) => {
    const s = state[files[i].name];
    el.classList.toggle('keep', s === 'keep');
    el.classList.toggle('skip', s === 'skip');
    el.classList.toggle('cursor', i === cur);
    el.classList.toggle('hide', onlyUndecided && !!s);
  });
  const k = Object.values(state).filter(v => v === 'keep').length;
  const s = Object.values(state).filter(v => v === 'skip').length;
  document.getElementById('stats').textContent =
    `${files.length} frames \\u2014 ${k} keep, ${s} skip, ${files.length - k - s} undecided`;
}
function set(i, v){
  const n = files[i].name;
  state[n] === v ? delete state[n] : state[n] = v;
  save(); paint();
}
function move(d){
  let i = cur;
  for (let n = 0; n < files.length; n++){
    i = (i + d + files.length) % files.length;
    if (!onlyUndecided || !state[files[i].name]) break;
  }
  cur = i; paint();
  cells[cur].scrollIntoView({block:'nearest', behavior:'smooth'});
}
grid.addEventListener('click', e => {
  const fig = e.target.closest('figure');
  if (!fig) return;
  cur = +fig.dataset.i; set(cur, 'keep');
});
addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA') return;
  const k = e.key.toLowerCase();
  if (k === 'arrowright' || k === 'd') { move(1); e.preventDefault(); }
  else if (k === 'arrowleft' || k === 'a') { move(-1); e.preventDefault(); }
  else if (k === 'k' || k === ' ') { set(cur, 'keep'); move(1); e.preventDefault(); }
  else if (k === 'x' || k === 's') { set(cur, 'skip'); move(1); e.preventDefault(); }
  else if (k === 'u') { delete state[files[cur].name]; save(); paint(); }
});
document.getElementById('undec').onclick = () => { onlyUndecided = !onlyUndecided; paint(); };
document.getElementById('bigger').onclick = () => bump(60);
document.getElementById('smaller').onclick = () => bump(-60);
function bump(d){
  const w = parseInt(getComputedStyle(grid).getPropertyValue('--w')) || 190;
  grid.style.setProperty('--w', Math.max(90, w + d) + 'px');
}
function csv(){
  const rows = [['name','decision','source']];
  files.forEach(f => rows.push([f.name, state[f.name] || 'undecided', f.source]));
  return rows.map(r => r.map(c => `"${String(c).replace(/"/g,'""')}"`).join(',')).join('\\n');
}
document.getElementById('export').onclick = () => {
  const text = csv();
  const ta = document.getElementById('out');
  ta.style.display = 'block'; ta.value = text; ta.select();
  try {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], {type:'text/csv'}));
    a.download = 'decisions.csv'; a.click();
  } catch (e) {
    alert('Download blocked - the CSV is in the box below, copy it out.');
  }
};
paint();
"""


def cmd_review(args) -> None:
    proxies = list_images(args.proxies)
    if not proxies:
        sys.exit(f"no proxies found under {args.proxies} -- run `thumbs` first")

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    files = [{"name": os.path.basename(p),
              "src": os.path.relpath(p, out_dir).replace(os.sep, "/"),
              "source": p} for p in proxies]

    page = f"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UAV triage &mdash; {html.escape(os.path.basename(args.proxies))}</title>
<style>{REVIEW_CSS}</style>
<header>
  <strong>UAV triage</strong>
  <span id="stats"></span>
  <span><kbd>K</kbd>/<kbd>space</kbd> keep &middot; <kbd>X</kbd> skip &middot;
        <kbd>U</kbd> undo &middot; <kbd>&larr;</kbd><kbd>&rarr;</kbd> move &middot; click = keep</span>
  <button id="undec">Only undecided</button>
  <button id="bigger">Bigger</button>
  <button id="smaller">Smaller</button>
  <button id="export">Export CSV</button>
</header>
<div id="grid"></div>
<textarea id="out" readonly></textarea>
<script>{REVIEW_JS.replace("__FILES__", json.dumps(files))}</script>
"""
    with open(args.out, "w") as fh:
        fh.write(page)
    print(f"[review] wrote {args.out} with {len(files)} frames")
    print(f"[review] open it:  open {args.out}")
    print("[review] decisions autosave to localStorage; Export CSV when done")


# --------------------------------------------------------------------------- #
# dedup -- drop the redundant frames before paying to transfer them
# --------------------------------------------------------------------------- #
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Local flat-earth approximation -- exact enough over a single flight."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    x = math.radians(lon2 - lon1) * math.cos((p1 + p2) / 2)
    y = p2 - p1
    return math.hypot(x, y) * 6371000.0


def cmd_dedup(args) -> None:
    with open(args.manifest) as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: (r.get("datetime") or "", r["name"]))

    if args.every > 1:
        kept = rows[::args.every]
        why = f"every {args.every}th frame"
    else:
        located = [r for r in rows if r.get("lat")]
        if not located:
            sys.exit("no GPS in manifest -- use --every N instead of --spacing")
        kept, last = [], None
        for r in located:
            lat, lon = float(r["lat"]), float(r["lon"])
            if last is None or haversine_m(last[0], last[1], lat, lon) >= args.spacing:
                kept.append(r)
                last = (lat, lon)
        # frames without GPS can't be placed, so never silently drop them
        kept += [r for r in rows if not r.get("lat")]
        why = f"min {args.spacing} m spacing"

    with open(args.out, "w") as fh:
        fh.write("\n".join(r["path"] for r in kept) + "\n")
    cut = 100 * (1 - len(kept) / max(len(rows), 1))
    print(f"[dedup] {len(rows)} -> {len(kept)} frames ({why}), {cut:.0f}% dropped")
    print(f"[dedup] wrote {args.out} -- feed it straight to: thumbs --src {args.out}")


# --------------------------------------------------------------------------- #
# plan -- turn the decisions CSV back into a real set of originals
# --------------------------------------------------------------------------- #
def cmd_plan(args) -> None:
    with open(args.decisions) as fh:
        keep = {r["name"] for r in csv.DictReader(fh) if r.get("decision") == "keep"}
    if not keep:
        sys.exit(f"no rows marked keep in {args.decisions}")

    # Proxy names are the flattened source paths, so map back by that stem.
    originals = list_images(args.src)
    by_stem = {proxy_name(p, args.src): p for p in originals}
    hits = [by_stem[k] for k in sorted(keep) if k in by_stem]
    missing = len(keep) - len(hits)

    os.makedirs(args.out, exist_ok=True)
    listing = os.path.join(args.out, "keep_list.txt")
    with open(listing, "w") as fh:
        fh.write("\n".join(hits) + "\n")

    if args.copy:
        for i, src in enumerate(hits, 1):
            shutil.copy2(src, os.path.join(args.out, os.path.basename(src)))
            if i % 25 == 0:
                print(f"  copied {i}/{len(hits)}", flush=True)
        print(f"[plan] copied {len(hits)} originals into {args.out}")

    print(f"[plan] {len(hits)} frames to annotate, listed in {listing}"
          + (f" ({missing} names had no matching original)" if missing else ""))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("thumbs", help="build small proxies (the one full-res pass)")
    t.add_argument("--src", required=True, help="image dir, or a text file of paths")
    t.add_argument("--out", default="proxies", help="proxy output dir")
    t.add_argument("--size", type=int, default=1024, help="longest proxy side in px")
    t.add_argument("--quality", type=int, default=82, help="proxy JPEG quality")
    t.add_argument("--workers", type=int, default=16, help="threads (raise for Drive/network)")
    t.add_argument("--force", action="store_true", help="rebuild proxies that already exist")
    t.set_defaults(func=cmd_thumbs)

    i = sub.add_parser("index", help="EXIF/GPS manifest without decoding pixels")
    i.add_argument("--src", required=True)
    i.add_argument("--out", default="manifest.csv")
    i.add_argument("--workers", type=int, default=16)
    i.set_defaults(func=cmd_index)

    d = sub.add_parser("dedup", help="drop overlapping frames using the manifest GPS")
    d.add_argument("--manifest", required=True, help="CSV from `index`")
    d.add_argument("--out", default="keep_frames.txt", help="path list, usable as thumbs --src")
    d.add_argument("--spacing", type=float, default=40.0,
                   help="minimum metres between kept frames")
    d.add_argument("--every", type=int, default=1,
                   help="simple every-Nth fallback when there is no GPS")
    d.set_defaults(func=cmd_dedup)

    s = sub.add_parser("sheets", help="contact sheets from proxies")
    s.add_argument("--proxies", required=True)
    s.add_argument("--out", default="sheets")
    s.add_argument("--cols", type=int, default=6)
    s.add_argument("--rows", type=int, default=5)
    s.add_argument("--cell", type=int, default=320, help="cell side in px")
    s.set_defaults(func=cmd_sheets)

    r = sub.add_parser("review", help="self-contained HTML triage page")
    r.add_argument("--proxies", required=True)
    r.add_argument("--out", default="triage.html")
    r.set_defaults(func=cmd_review)

    p = sub.add_parser("plan", help="decisions CSV -> list (or copy) of originals")
    p.add_argument("--decisions", required=True, help="CSV exported from the review page")
    p.add_argument("--src", required=True, help="where the originals live")
    p.add_argument("--out", default="keep")
    p.add_argument("--copy", action="store_true", help="copy originals, not just list them")
    p.set_defaults(func=cmd_plan)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
