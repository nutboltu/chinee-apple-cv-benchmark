# supervised-probe — logistic regression head on frozen DINO tokens

Supervised localization of chinee apple, trained on the CVAT and SAM3 polygon
annotations. The backbone is frozen and never fine-tuned. The only thing
trained is a 769 number head, so this runs on a free Colab T4 in minutes.

```
tile + polygons -> mask -> patch-grid coverage -> {positive, negative, ignore}
frozen ViT patch tokens -> L2 norm -> logistic regression -> p(weed)
```

Contrast with [`../kmeans-clusters`](../kmeans-clusters), which needs no labels
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

## Run

```bash
P=../../datasets/matured-dry-30m/processed

python train_probe.py --zips "$P/batch_*.zip" --tiles $P --crib $P/tile_markers.csv
python predict.py --probe probe_out/probe.joblib --image my_tile.jpg --out preds
python compare.py --zips "$P/batch_*.zip" --tiles $P --img-sizes 448 768
```

Add `--family dinov3 --repo /path/dinov3 --weights /path/w.pth` for the gated
backbone. The default is the ungated DINOv2 fallback.

## Measured baseline

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

**The split is grouped by source frame.** Tiles are cut 8 by 6 from one frame
and the bottom row overlaps the row above by 684 px, so neighbouring tiles
share literal pixels. A random split leaks train into test.

**Empty tiles are not negatives.** 440 of 620 tiles came back unannotated, but
that means nobody annotated them, not that they hold no chinee apple. All
negatives are drawn from inside annotated tiles. Using the empty ones as
background would train the head to suppress the plants it should find.

**Instance mAP is far below patch AP.** This is a good per-patch classifier and
a weak instance detector, because the head has no spatial prior and canopies
fragment. A light convolutional decoder over the token grid is the principled
fix, not threshold tuning.
