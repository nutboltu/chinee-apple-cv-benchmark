# chinee-apple-cv-benchmark

[![kmeans-clusters](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nutboltu/chinee-apple-cv-benchmark/blob/main/pipelines/kmeans-clusters/kmeans_colab.ipynb)
[![supervised-probe](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nutboltu/chinee-apple-cv-benchmark/blob/main/pipelines/supervised-probe/probe_colab.ipynb)


PhD research repository for evaluating computer-vision models on detecting
**Chinee apple** (*Ziziphus mauritiana*) — a Weed of National Significance in
Australia that invades rangelands and pastures and is difficult to distinguish
from surrounding vegetation in aerial and ground imagery.

Each subdirectory is a self-contained experiment: a single model family, its
runtime, and the scripts used to evaluate it on Chinee apple imagery.

## Experiments

| Directory | Model | Task | Status |
| --- | --- | --- | --- |
| [`pipelines/kmeans-clusters/`](./pipelines/kmeans-clusters) | DINOv3 / DINOv2 ViT | **Annotation-free.** KMeans over frozen patch tokens, then a human picks which cluster is the weed, once per run. Needs no labels, but needs a person every time. | Stage 1 |
| [`pipelines/supervised-probe/`](./pipelines/supervised-probe) | DINOv3 / DINOv2 ViT | **Supervised.** Logistic-regression head on frozen tokens, trained on the CVAT and SAM3 polygons. Trains once, then runs unattended. Patch AP 0.77, instance mAP 0.15 on held-out frames. | Trained |
| [`pipelines/dinov3-colab/`](./pipelines/dinov3-colab) | DINOv3 ViT-L/16, CAFe-DINO | **Mixed exploration.** CLS-token saliency with a Gradio UI, an early crop-based probe, and CAFe-DINO open-vocabulary detection prompted with text. Superseded in part by supervised-probe. | Exploratory |
| [`triage/`](./triage) | — (Pillow only) | **Pre-annotation triage.** Narrows thousands of UAV frames to the ones actually containing chinee apple, before anything reaches CVAT. EXIF/GPS manifest → GPS dedup (~84% frame overlap, so ~2/3 are redundant) → one threaded proxy pass → bulk keep/skip in an HTML page or contact sheets → list of originals to annotate. | Tool |
| [`annotations/`](./annotations) | — | **Dataset preparation.** Tiles 8192x5460 stills into 1024x1024, detects the cardboard ground-truth markers, keeps only tiles containing one, batches them for CVAT, and filters the finished exports down to annotated frames. | Tool |
| [`metrics/`](./metrics) | — | **Shared metrics.** Backbone-agnostic patch, mask and instance scoring (AP, IoU, Dice, COCO mAP) so every pipeline reports comparable numbers. | Shared |

More experiments (e.g. supervised detectors, segmentation backbones,
fine-tuned classifiers) will be added as sibling directories.

## Layout

```
chinee-apple-cv-benchmark/
├── README.md            ← you are here
├── annotations/         ← tiling, marker detection, CVAT task prep and export
├── datasets/            ← imagery (gitignored, not in the clone)
│   └── matured-dry-30m/
│       ├── raw/         ← source stills
│       └── processed/   ← tiles, batches, CVAT exports
├── backbones/           ← shared DINO loading + patch tokens
├── metrics/             ← shared scoring, used by every pipeline
├── pipelines/           ← one folder per model/approach
│   ├── kmeans-clusters/   annotation-free, pick a cluster
│   ├── supervised-probe/  trained on CVAT + SAM3 polygons
│   └── dinov3-colab/      exploratory, incl. CAFe-DINO
└── triage/              ← pre-annotation frame triage
```

The imagery is deliberately outside git. Clone the repo, then point the
scripts at wherever you keep `datasets/`.

## Adding a new pipeline

1. Create a directory under `pipelines/` named after the model or approach.
2. Include a short README covering the model, the dataset split, how to run
   it, and what it produces.
3. Pin dependencies in a local `requirements.txt` so runs stay reproducible.
4. Take features from `backbones/` and score with `metrics/` rather than
   rolling your own, so results stay comparable across pipelines.
5. Add a row to the table above.

## Target weed

Chinee apple (*Ziziphus mauritiana*) — thorny shrub / small tree, dense
canopy, small ovate leaves. Visual cues used by these experiments include
leaf shape, canopy texture, and (where in season) fruit colour.
