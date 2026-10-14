# yolo11-seg - fine-tuned instance segmentation on the CVAT canopy polygons

Supervised instance segmentation of chinee apple, trained end to end on the
same CVAT and SAM3 polygons as [`../dinov3-linear-probe`](../dinov3-linear-probe).
Where the probe freezes a self-supervised backbone and fits a 769 number head,
this updates every weight and predicts separated objects rather than a dense
score map.

```
CVAT polygon (x1,y1,x2,y2,...)  ->  normalised ring  ->  YOLO-seg label
frozen nothing. backbone + neck + seg head, fine-tuned from COCO weights
```

## How the annotations map onto YOLO

The CVAT export is `CVAT for images 1.1`, one `<polygon>` per plant:

```xml
<image name="DJI_20251118124919_0185_r0_c5.jpg" width="1024" height="1024">
  <polygon label="chinee_apple_matured_dry_30m" source="semi-auto"
           points="752.00,770.00,749.00,773.00,747.00,782.00,..."/>
</image>
```

YOLO-seg wants one text file per image, one line per instance, class id then
the ring as xy pairs normalised by image width and height:

```
0 0.734375 0.751953125 0.7314453125 0.7548828125 0.7294921875 0.763671875 ...
```

So the conversion is a divide by 1024 and a reorder. `cvat_to_yolo.py` does it
and the round trip is lossless: rasterising the written labels back and
comparing against the CVAT rings gives IoU 1.000000 on all 180 tiles. Four
things about this dataset are not that mechanical, and they are where a
careless conversion quietly costs you the experiment.

**One ring per instance, no holes.** YOLO-seg carries a single closed ring, so
there is nowhere to put a hole or an ignore region. The probe exploits both: it
punches the cardboard markers out of the mask and drops boundary patches
between 10% and 50% coverage. Neither is expressible here, which forces the
marker decision into image space (below) and means boundary pixels are simply
trained as whatever side of the ring they fall on.

**Attributes are lost, and were never recorded.** `cvat_labels.json` defines
`marker_uid`, `class` (blue/pink/blank/unsure) and `partial`. The exports carry
none of them, the label element's `<attributes>` block is empty, so the task was
run without them. Nothing is dropped in conversion because nothing is there. If
the arrow colour is ever meant to become a second class, it has to be
re-annotated or joined back through `tile_markers.csv` on `marker_uid`.

**The label name differs from the docs.** The exports say
`chinee_apple_matured_dry_30m`, `annotations/ANNOTATION.md` and
`cvat_labels.json` say `chinee_apple_matured_30m`. Harmless with one class, and
the converter reads whatever is in the file, but it is worth fixing before a
second class makes the mismatch ambiguous.

**Empty tiles are not background.** 440 of the 620 tiles came back with no
polygon. That records that nobody annotated them, not that they hold no chinee
apple. YOLO trains every pixel outside a ring as background, so including those
tiles teaches the model to suppress the plants it is meant to find. Only the 180
annotated tiles are written. `--background-frac` adds them anyway, for the
ablation that demonstrates the point.

## Which YOLO, and why

**Segmentation, not detection.** Measured over the 202 annotated polygons:

| | value |
| --- | --- |
| instances | 202 over 180 tiles, 1.12 per tile |
| polygon area, median | 20% of a 1024x1024 tile |
| bounding box area, median | 27% of the tile, p95 73% |
| clipped by a tile edge | 129 / 202, 64% |
| instances under 32x32 px | 0 |

These are large, sprawling, non-convex, mostly edge-truncated canopies. A box
around one covers a quarter of the tile and says almost nothing about where the
plant is, and two such boxes score high IoU whether or not the masks agree. The
polygons are already drawn and the repo already reports mask IoU and Dice, so
`-seg` is both the cheaper and the more informative target. This is also the
opposite of the usual aerial-imagery problem: there are no small objects here at
all, so none of the small-object tricks apply.

**yolo11s-seg.** YOLO11 is the current Ultralytics segmentation line and beats
v8-seg at equal parameter count. On size, the binding constraint is 145 training
instances over 39 source frames, not capacity, so `s` (about 10M parameters) is
the default and `n`/`m` are a sweep rather than an upgrade path. Run
`--model yolo11n-seg.pt` and `--model yolo11m-seg.pt` and report all three
before believing any one of them.

**1024px.** Tiles are 1024 native. The objects are large enough that 640 keeps
their geometry, but the class cue here is canopy texture and leaf detail on a
brown dry-season shrub, and that is what downsampling destroys. 1024 is the
default, `--imgsz 640` is about 2.5x faster and is the honest speed comparison.

**Vertical flips and full rotation.** Ultralytics ships `flipud=0.0` and
`degrees=0.0` because COCO photographs have a ground plane. These are nadir UAV
frames with no canonical up, so both are turned on. At 130 training tiles that
matters more than the model size does.

A note for publication: Ultralytics is AGPL-3.0, which reaches any code you
distribute that links it. The other pipelines in this repo are Apache or
CC-BY-NC backbones, so this one is the only directory with that constraint.

## Files

| file | purpose |
| --- | --- |
| `cvat_to_yolo.py` | CVAT exports to a YOLO-seg dataset, split by source frame. `--dry-run` reports the geometry with no writes. Needs only pillow and numpy. |
| `train_yolo.py` | Fine-tune and validate. `--show-config` prints every setting and loads no torch. |
| `evaluate.py` | Re-score with the shared `metrics` package on the shared grid. `--self-test` checks the metric plumbing with no model. |
| `yolo_colab.ipynb` | The whole loop end to end on a free T4. The notebook calls the scripts above rather than reimplementing them, so a run there and a run on the command line train the same model. |

## Run

```bash
P=../../datasets/matured-dry-30m/processed

python cvat_to_yolo.py --zips "$P/batch_*.zip" --tiles $P --crib $P/tile_markers.csv \
    --out yolo_dataset
python train_yolo.py --data yolo_dataset
python evaluate.py --weights runs/yolo11s-seg_1024px/weights/best.pt --data yolo_dataset
python -m metrics.render eval_out/metrics.json --format md
```

The conversion is worth checking before booking a GPU:

```bash
python cvat_to_yolo.py --zips "$P/batch_*.zip" --tiles $P --dry-run
python evaluate.py --self-test
```

## The cardboard has to come out of the imagery

The field markers are white cardboard that exists only because this was a
ground-truthing flight, and each board was placed beside its target plant. They
are 0.12% of annotated mask area, which is why the probe can afford to drop
those pixels in label space, but each board is about 166x171px, roughly 2.7% of
a tile, and usually sits just outside the canopy it marks.

A detector sees the whole tile at once, so it can learn "cardboard, therefore
canopy next to it" and then score well on this set and find nothing on an
unmarked flight. A single-ring polygon cannot express an ignore region, so
`cvat_to_yolo.py` overwrites each board with texture copied from one box-height
away in the same tile, which keeps local texture statistics roughly intact
rather than swapping a white constant for a grey one. `--keep-markers` disables
it and is the ablation that measures how much the boards were worth.

## Scoring, and why there are two instance numbers

Ultralytics prints its own mask mAP, but it runs its own NMS and matches at full
resolution, while every other pipeline here is scored by `metrics/` on a 32x32
patch grid. Putting both in one table would read as a comparison and would not
be one, so `train_yolo.py` records the Ultralytics figures in
`train_summary.json` and `evaluate.py` writes the comparable ones to
`metrics.json`.

`metrics.instance_map` builds instances from connected components, because a
per-patch probe has no notion of an object. That is the right call for the probe
and the wrong one for a detector, since it merges two adjacent canopies and
discards exactly the instance separation that justifies training one. So
`evaluate.py` reports both, under `instance`:

- `mAP_50_95`, connected components, comparable with the probe
- `native_mAP_50_95`, YOLO's own instances, same 101-point AP convention

### The component metric cannot reach 1.0

Feed the ground truth back in as a perfect prediction and score it the
component way, and it does not score 1.0, because components cannot separate
canopies that touch. On the 17 held-out frames:

| scoring grid | smooth, min_size | perfect prediction scores | instances resolved |
| --- | --- | --- | --- |
| 32x32 | 1, 8 (the probe's tuned default) | **0.605** mAP, 0.842 AP@0.50 | 49 of 57 |
| 32x32 | 0, 4 | 0.927 mAP, 0.960 AP@0.50 | 55 of 57 |
| 64x64 | 1, 8 | 0.755 mAP, 0.911 AP@0.50 | 53 of 57 |
| 1024px | 0, 4 | 0.852 mAP, 0.867 AP@0.50 | 67 of 57 |

So dinov3-linear-probe's instance mAP of 0.147 should be read against a ceiling
of 0.605, not against 1.0. It is about 24% of what that metric can award, not
15%. The `smooth=1` closing that the probe needed in order to stop fragmenting
its own canopies is itself costing a third of the ceiling by merging real
neighbours, and at 1024px the metric starts inventing instances, 67 components
for 57 plants, by splitting single canopies at narrow waists.

`evaluate.py` recomputes this ceiling on whatever split and settings it is given
and writes it into `metrics.json` as `component_ceiling_mAP_50_95`, so the
figure never travels without its bound.

## What to expect

No trained numbers yet, so this directory has no measured baseline row. Two
predictions worth recording in advance, because they are what the experiment is
for:

**Instance mAP should beat the probe's 0.147 clearly.** A detector has the
spatial prior the probe lacks, which the probe's own README names as the
principled fix. If it does not, the 180 tile dataset is the binding constraint
and the next move is more annotation, not more model.

**Patch AP may well lose to the probe's 0.769.** A frozen DINOv2 backbone saw
142M images, this fine-tunes from COCO on 130 tiles. Being beaten per-patch
while winning per-instance is the expected and informative outcome, not a
contradiction, and it is the argument for a light convolutional decoder on
frozen DINO features as a third design.

## Honest limits

**Sparse annotation is the floor on all of this.** Only the plant each cardboard
pointed at was annotated. Where a tile holds unmarked chinee apple, those pixels
are trained as background, and the model is penalised at evaluation for finding
them. This depresses precision and the true precision is unknown, not merely
unmeasured. Nothing in this pipeline fixes it, it needs exhaustive annotation of
a subset of tiles to even quantify.

**57 held-out instances.** Every figure from this split carries wide error bars.
Re-run with `--seed 1`, `--seed 2` and so on and report the spread, which is
what `cvat_to_yolo.py` records the seed in `split.json` for.

**The split is grouped by source frame.** Tiles are cut 8x6 from one 8192x5460
frame and the bottom row overlaps the row above by 684px, so neighbouring tiles
share literal pixels. Splitting over tiles leaks train into test and inflates
everything.
