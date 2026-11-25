# dinov3-linear-probe — logistic regression head on frozen DINO tokens

Supervised localization of chinee apple, trained on the CVAT and SAM3 polygon
annotations. The backbone is frozen and never fine-tuned. The only thing
trained is a 769 number head, so this runs on a free Colab T4 in minutes.

```
tile + polygons -> mask -> patch-grid coverage -> {positive, negative, ignore}
frozen ViT patch tokens -> L2 norm -> logistic regression -> p(weed)
```

Contrast with [`../dinov3-kmeans-cluster`](../dinov3-kmeans-cluster), which needs no labels
but requires a human to pick the right cluster every time. This pipeline needs
labels once and then runs unattended.

## Why logistic regression

A strong self-supervised backbone is trained so that a straight line separates
classes in feature space, which is what "linear probe" measures. Over 20
configurations, five heads by two resolutions by two smoothing settings, patch
AP spans only 0.758 to 0.771. A random forest and a small MLP land within noise
of logistic regression, so the simplest of equals wins. Run `compare.py` to
reproduce that table on your own split.

## Files

| file | purpose |
| --- | --- |
| `train_probe.py` | CVAT polygons to a trained head. `--dry-run` reports label geometry with no backbone. |
| `predict.py` | Run a trained head on any imagery. Heatmap, mask, overlay and a plant list. |
| `compare.py` | Score several heads and resolutions on one split so rows are comparable. |
| `probe_colab.ipynb` | The whole loop end to end on a free GPU. |

Scoring comes from the shared [`metrics`](../../metrics) package, and features
from the shared [`backbones`](../../backbones) package.

## Two seasons, two shape kinds

The pipeline runs on either dataset. What differs is how CVAT exported the
annotations, and it is not cosmetic.

| | `matured-dry-30m` | `matured-wet-30m` |
| --- | --- | --- |
| CVAT exports | `batch_*.zip` | `wet_batch_*.zip` |
| Shapes | 202 polygons | 400 masks, 9 polygons |
| Annotated tiles | 180 of 620 | 260 of 300 |
| Source frames | 56 | 41 |

SAM3 produces a raster mask. CVAT writes it out as a `<mask>` element holding a
run-length encoding unless the annotator converts it to a polygon first, which
happened on the dry tasks and not on the wet ones. `read_cvat_zips` reads both
kinds and `tile_mask` rasterises either, so a tile may freely hold a mix.

This mattered more than it sounds. A reader that takes only `<polygon>` finds 9
shapes in the wet export instead of 409, and nothing raises, because an export
with no polygons is indistinguishable from a task nobody annotated. The
`yolo11-seg` converter cannot consume masks at all, so it now refuses the wet
export rather than quietly building a dataset from 2 percent of the labels.

## Run

```bash
# wet season
P=../../datasets/matured-wet-30m/processed
python train_probe.py --zips "$P/wet_batch_*.zip" --tiles $P --crib $P/tile_markers.csv

# dry season
P=../../datasets/matured-dry-30m/processed
python train_probe.py --zips "$P/batch_*.zip" --tiles $P --crib $P/v1/tile_markers.csv

python predict.py --probe probe_out/probe.joblib --image my_tile.jpg --out preds
python compare.py --zips "$P/batch_*.zip" --tiles $P --img-sizes 448 768
```

Add `--dry-run` to either training command to report the label geometry without
loading a backbone. It is the fastest way to confirm a new export was read
properly, because it prints the shape counts.

```
annotated tiles in export : 260
shapes                    : 409  (9 polygons, 400 masks)
```

The scripts default to the ungated DINOv2 fallback. For DINOv3, clone Meta's
repo and point at your weights, since loading needs both the code and the
checkpoint.

```bash
pip install torchmetrics termcolor omegaconf ftfy
git clone https://github.com/facebookresearch/dinov3.git /content/dinov3
... --family dinov3 --repo /content/dinov3 --weights /path/dinov3_vitb16_pretrain_lvd1689m.pth
```

Those four packages are not optional. Meta's `hubconf.py` imports their
segmentation and dinotxt heads, so loading even a plain ViT-B/16 backbone pulls
in `torchmetrics` and `termcolor`, and fails with `ModuleNotFoundError` without
them.

The notebook defaults to DINOv3 and clones the repo for you.

## Where results go

Locally, wherever `--out` points. In the notebook, predictions are written to
`/content` first and then copied to Drive in one step, under
`MyDrive/chinee-apple/predictions/<date>_<backbone>_<resolution>/`, with a zip
of the same run beside it. Writing hundreds of small files straight to a
mounted Drive is slow, and a timestamped folder means runs never overwrite each
other and stay identifiable months later.

## Measured baseline

Dry season only. The wet season has not been run yet, and the dry numbers
should not be assumed to carry over, because the wet annotation is denser and
the target is green against brown grass rather than brown against brown.

This is the **DINOv2 fallback**, so it is the number DINOv3 should beat.
DINOv2 ViT-B/14 at 448px, 180 annotated tiles over 56 frames, 17 frames held
out, split grouped by source frame.

| level | metric | value |
| --- | --- | --- |
| patch | AP | 0.769 |
| patch | precision / recall | 0.651 / 0.807 |
| mask | IoU / Dice | 0.502 / 0.631 |
| instance | mAP@[.50:.95] | 0.147 |
| instance | AP@0.50 | 0.382 |

## Three things that decide whether those numbers mean anything

**Grouping the split by source frame is necessary and not sufficient.** Tiles
are cut 8 by 6 from one frame and the bottom row overlaps the row above by 684
px, so a random split over tiles leaks outright. Grouping by frame stops that.

It does not stop the frames overlapping each other, which is the larger leak. A
Zenmuse P1 at 30 m with the 35 mm lens covers about 30.8 by 20.5 m, and
consecutive frames sit a median 4.1 m apart, so neighbours share 80 percent or
more of their ground. On the wet split every one of the 13 held-out frames sat
within 4.2 m of a training frame, so the same plants were on both sides.

Run `annotations/spatial_blocks.py` and pass `--blocks`. It reads each frame's
EXIF GPS, joins frames closer than one footprint into a block, and the split
then groups by patch of ground. Both datasets have two sites about 23 km apart,
so a genuinely clean holdout exists for each. Any number produced without
`--blocks` is an upper bound.

```bash
python ../../annotations/spatial_blocks.py --dataset matured-wet-30m --buffer 100
python train_probe.py ... --blocks "$P/frame_blocks.csv"
```

**Empty tiles are not negatives.** 440 of 620 tiles came back unannotated, but
that means nobody annotated them, not that they hold no chinee apple. All
negatives are drawn from inside annotated tiles. Using the empty ones as
background would train the head to suppress the plants it should find.

**Instance mAP is far below patch AP.** This is a good per-patch classifier and
a weak instance detector, because the head has no spatial prior and canopies
fragment. A light convolutional decoder over the token grid is the principled
fix, not threshold tuning.
