# yolov8-seg - fine-tuned instance segmentation on the YOLOv8 line

The same pipeline as [`../yolo11-seg`](../yolo11-seg), trained on the previous
Ultralytics segmentation family. It exists to check one claim on this data
rather than to take it on trust.

## Why this directory exists

Ultralytics reports that YOLO11 beats v8-seg at equal parameter count. That
number is from COCO, which is 80 classes of everyday objects photographed from
the ground. This dataset is one class of sprawling, mostly edge-clipped canopy
seen from 30 m straight down, and `yolov8s-seg` actually carries more
parameters than `yolo11s-seg`, about 11.8M against 10M. Whether the ranking
survives the change of domain has not been measured here, and measuring it
costs one training run.

## How it relates to yolo11-seg

The two families differ by a checkpoint string and nothing else, so the
conversion, training and scoring code is shared rather than copied. It lives in
the repo-root [`yoloseg`](../../yoloseg) package, next to `backbones` and
`metrics`, and the scripts in this directory are entry points that set the
default model.

```
yoloseg/                      one implementation
  cvat_to_yolo.py             CVAT exports -> YOLO-seg dataset
  train_yolo.py               main(default_model) 
  evaluate.py                 scoring on the shared grid

pipelines/yolov8-seg/         defaults to yolov8s-seg.pt   <- you are here
pipelines/yolo11-seg/         defaults to yolo11s-seg.pt
```

This is not tidiness for its own sake. The CVAT mask tracing the wet season
needed went into `cvat_to_yolo` after the YOLO11 pipeline was already written.
A copied directory would have silently kept reading polygons only, found 9
shapes in a 409-shape export, and raised nothing. Sharing the implementation is
what stops that happening twice.

Everything in the [`yolo11-seg` README](../yolo11-seg/README.md) about how the
annotations map onto YOLO, why segmentation rather than boxes, what the mask
tracing costs, and why cardboard is painted out of the imagery applies here
unchanged. It is not repeated.

## Files

| file | purpose |
| --- | --- |
| `cvat_to_yolo.py` | entry point, identical behaviour to the yolo11-seg one |
| `train_yolo.py` | entry point, `--model` defaults to `yolov8s-seg.pt` |
| `evaluate.py` | entry point, identical behaviour |
| `yolov8_colab.ipynb` | the whole loop end to end on a free T4, `MODEL` defaults to v8 |

## Run

```bash
P=../../datasets/matured-wet-30m/processed

python cvat_to_yolo.py --zips "$P/wet_batch_*.zip" --tiles $P --crib $P/tile_markers.csv \
    --out yolo_dataset
python train_yolo.py --data yolo_dataset                 # yolov8s-seg.pt
python evaluate.py --weights runs/yolov8s-seg_1024px/weights/best.pt --data yolo_dataset
python -m metrics.render eval_out/metrics.json --format md
```

Check the conversion and the metric plumbing before booking a GPU.

```bash
python cvat_to_yolo.py --zips "$P/wet_batch_*.zip" --tiles $P --dry-run
python evaluate.py --self-test
python train_yolo.py --data yolo_dataset --show-config
```

Run names derive from the model stem, so `yolov8s-seg_1024px` and
`yolo11s-seg_1024px` never collide and both families can share one
`yolo_dataset` and one `runs/` directory.

## The comparison is genuinely paired

Both families read the same dataset built once by `cvat_to_yolo.py` at seed 0,
so they train on the same 190 tiles and are scored on the same 70 val tiles and
the same 113 plants. Most cross-model tables in this project carry a caveat
that the held-out halves differ. This one does not, which is the whole reason
it is worth running.

## What to expect

**A small v11 win, or nothing.** The COCO margin at the `s` size is a point or
two of mask mAP, v8s carries more parameters, and data has been the binding
constraint at every step of this project. The honest prior is that the two land
within noise.

**One seed cannot settle it.** 113 held-out instances is a small sample and the
YOLO11 wet run is a single draw. Treat any gap under about 0.05 as no gap until
both are repeated at `--seed 1` and `--seed 2`.

## Measured baseline

No v8 run yet. The YOLO11 wet run it is measured against reached patch AP
0.947, mask IoU 0.800, and mAP@[.50:.95] of 0.326 by components or 0.580 on its
own objects, on 12 held-out photographs and 113 plants.

Ultralytics is AGPL-3.0, which reaches any code you distribute that links it.
That applies to both families equally.
