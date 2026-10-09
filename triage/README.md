# triage — find the UAV frames worth annotating

Thousands of DJI frames, only some containing chinee apple, and no way to know
which without looking. This is the pre-annotation step: it narrows the set
*before* anything reaches CVAT, so annotation time goes into frames that have
the weed in them.

## Why opening each frame is slow

Measured on real frames from this project (8192×5460, ~26 MB each):

| step | cost per frame |
| --- | --- |
| local disk read | 9 ms |
| JPEG decode | 110–170 ms |
| **same read over Google Drive** | **~2.8 s** at 10 MB/s |

Decoding is not the bottleneck — moving 26 MB per frame across the network is.
Browsing 3000 frames that way is ~2.5 hours of pure transfer, paid again every
time you scroll back to re-check one.

Two things follow, and they are the whole design:

1. **Pay the transfer once.** One pass builds a ~200 KB proxy per frame. Every
   later step reads proxies, so the working set drops from ~78 GB to ~600 MB
   and opens instantly.
2. **Run that pass where the data is.** On a Colab VM next to the Drive mount,
   not on a laptop pulling through a Drive client.

`draft()` is used during the pass not for speed (it is only ~1.2× faster —
Huffman decoding dominates, and draft only shrinks the IDCT) but for **memory**:
it cuts peak RSS per frame from ~163 MB to under 10 MB, which is what lets
`--workers 16–32` run without exhausting RAM. High concurrency is what actually
hides Drive's per-file latency.

## The redundancy nobody exploits

UAV survey frames overlap heavily. From the EXIF GPS of this project's own
frames, at 97 m AGL:

```
mean spacing between consecutive frames   22 m
footprint width (4/3 sensor, 24mm-equiv)  137 m
forward overlap                           ~84%
```

At 84% overlap you can keep **every 3rd frame** and still have 50% overlap —
full ground coverage, a third of the frames. `dedup` does this from the EXIF
GPS alone, and because `index` reads only file headers (no pixel transfer), you
can cut the set *before* paying to move a single full-resolution frame.

## Workflow

```bash
pip install -r requirements.txt
```

**On Colab, next to the Drive mount** — the only steps that touch originals:

```bash
# 1. EXIF/GPS manifest. Reads headers only, so it is cheap even over Drive.
python fast_triage.py index --src /content/drive/MyDrive/uav --out manifest.csv

# 2. Drop the ~2/3 of frames that are redundant overlap.
python fast_triage.py dedup --manifest manifest.csv --spacing 40 --out keep_frames.txt

# 3. The one full-resolution pass. Resumable — rerun after a disconnect and it
#    skips what it already built.
python fast_triage.py thumbs --src keep_frames.txt --out proxies --workers 24
```

Then download `proxies/` (small) and work locally:

```bash
# 4a. Keyboard triage in the browser: K keep, X skip, ←/→ move, Export CSV.
#     Decisions autosave to localStorage, so closing the tab loses nothing.
python fast_triage.py review --proxies proxies --out triage.html

# 4b. Or scan contact sheets, 30 frames per page.
python fast_triage.py sheets --proxies proxies --out sheets

# 5. Turn the kept list back into real originals for CVAT.
python fast_triage.py plan --decisions decisions.csv --src /path/to/uav --out keep --copy
```

`keep/keep_list.txt` is the set to upload. Everything else never gets opened
again.

## Commands

| command | reads | writes | notes |
| --- | --- | --- | --- |
| `index` | file headers only | `manifest.csv` | path, size, dimensions, EXIF datetime, GPS lat/lon/alt |
| `dedup` | `manifest.csv` | path list | `--spacing` metres, or `--every N` when there is no GPS; frames without GPS are always kept |
| `thumbs` | originals (once) | proxy JPEGs | threaded, resumable, `--size` default 1024; failures are logged, not fatal |
| `sheets` | proxies | contact sheet JPEGs | `--cols`/`--rows`/`--cell` |
| `review` | proxies | one HTML file | self-contained, works over `file://` |
| `plan` | decisions CSV | list (or copy) of originals | `--copy` to stage the files themselves |

`--src` accepts a directory (walked recursively) or a text file of paths, so
the output of `dedup` feeds straight into `thumbs`.

## Scaling further

Once a few hundred frames are triaged by hand, that keep/skip split is a
labelled set — train the [`dinov3-vit`](../dinov3-vit) linear probe on the
proxies and let it rank the remaining frames by weed likelihood, so you review
in descending order and stop when the hits dry up. The proxies are already the
right input size for a ViT, so no extra preprocessing is needed.
