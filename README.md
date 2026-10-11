# chinee-apple-cv-benchmark

[![Open dinov3-colab in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nutboltu/chinee-apple-cv-benchmark/blob/main/pipelines/dinov3-colab/dinov3_colab.ipynb)
[![Open dinov3-vit in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nutboltu/chinee-apple-cv-benchmark/blob/main/pipelines/dinov3-vit/dinov3_vit_colab.ipynb)

PhD research repository for evaluating computer-vision models on detecting
**Chinee apple** (*Ziziphus mauritiana*) — a Weed of National Significance in
Australia that invades rangelands and pastures and is difficult to distinguish
from surrounding vegetation in aerial and ground imagery.

Each subdirectory is a self-contained experiment: a single model family, its
runtime, and the scripts used to evaluate it on Chinee apple imagery.

## Experiments

| Directory | Model | Task | Status |
| --- | --- | --- | --- |
| [`pipelines/dinov3-colab/`](./pipelines/dinov3-colab) | DINOv3 (ViT-B/16) | Two paths in `dinov3_colab.ipynb`: (1) unsupervised foreground localization via [CLS]-token similarity over patch tokens, and (2) a **supervised linear probe** on the frozen backbone — train a logistic-regression head on labeled crops to classify Chinee apple vs other trees, then apply it per-patch for class-aware detection heatmaps. Also runs by SSHing into a Colab VM (`colab_ssh_bootstrap.ipynb` + `dinov3_detect.py`). | Probe |
| [`pipelines/dinov3-vit/`](./pipelines/dinov3-vit) | DINOv3 (ViT-B/16), DINOv2 fallback | **Frozen-backbone probe.** Stage 1 is annotation-free KMeans over patch tokens (`vit_features.py`). Stage 2 is a supervised logistic-regression head trained on the CVAT + SAM3 polygons (`train_probe.py`), run on new imagery with `predict.py`, and compared across heads and resolutions with `compare.py`. End to end in `dinov3_probe_colab.ipynb`. | Probe trained |
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
├── metrics/             ← shared scoring, used by every pipeline
├── pipelines/           ← one folder per model/approach
│   ├── dinov3-colab/
│   └── dinov3-vit/
└── triage/              ← pre-annotation frame triage
```

The imagery is deliberately outside git. Clone the repo, then point the
scripts at wherever you keep `datasets/`.

## Adding a new pipeline

1. Create a directory under `pipelines/` named after the model or approach.
2. Include a short README covering the model, the dataset split, how to run
   it, and what it produces.
3. Pin dependencies in a local `requirements.txt` so runs stay reproducible.
4. Score with `metrics/` rather than rolling your own, so results stay
   comparable across pipelines.
5. Add a row to the table above.

## Target weed

Chinee apple (*Ziziphus mauritiana*) — thorny shrub / small tree, dense
canopy, small ovate leaves. Visual cues used by these experiments include
leaf shape, canopy texture, and (where in season) fruit colour.
