# weed-detection-experiments

[![Open dinov3-colab in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nutboltu/weed-detection-experiments/blob/main/dinov3-colab/dinov3_colab.ipynb)
[![Open rf-detr in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nutboltu/weed-detection-experiments/blob/main/rf-detr/rf_detr_colab.ipynb)
[![Open dinov3-vit in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nutboltu/weed-detection-experiments/blob/main/dinov3-vit/dinov3_vit_colab.ipynb)

PhD research repository for evaluating computer-vision models on detecting
**Chinee apple** (*Ziziphus mauritiana*) — a Weed of National Significance in
Australia that invades rangelands and pastures and is difficult to distinguish
from surrounding vegetation in aerial and ground imagery.

Each subdirectory is a self-contained experiment: a single model family, its
runtime, and the scripts used to evaluate it on Chinee apple imagery.

## Experiments

| Directory | Model | Task | Status |
| --- | --- | --- | --- |
| [`dinov3-colab/`](./dinov3-colab) | DINOv3 (ViT-B/16) | Two paths in `dinov3_colab.ipynb`: (1) unsupervised foreground localization via [CLS]-token similarity over patch tokens, and (2) a **supervised linear probe** on the frozen backbone — train a logistic-regression head on labeled crops to classify Chinee apple vs other trees, then apply it per-patch for class-aware detection heatmaps. Also runs by SSHing into a Colab VM (`colab_ssh_bootstrap.ipynb` + `dinov3_detect.py`). | Probe |
| [`dinov3-vit/`](./dinov3-vit) | DINOv3 (ViT-B/16) | **Annotation-free localization** in `dinov3_vit_colab.ipynb`: frozen ViT patch tokens → KMeans over patches → pick which cluster is chinee apple *once* → mask/heatmap/boxes. Cluster ids double as pseudo-labels for a stage-2 linear probe. DINOv2 fallback for the gated DINOv3 weights. | Stage 1 |
| [`rf-detr/`](./rf-detr) | RF-DETR (Base) | COCO-pretrained DETR-style detector run on the same imagery. Self-contained Colab notebook (`rf_detr_colab.ipynb`) or local script (`rf_detr_detect.py`). Default checkpoint detects COCO classes only — fine-tune for actual Chinee apple detection. | Baseline |

More experiments (e.g. supervised detectors, segmentation backbones,
fine-tuned classifiers) will be added as sibling directories.

## Layout

```
weed-detection-experiments/
├── README.md            ← you are here
└── <experiment-name>/   ← one folder per model/approach
    ├── README*.md       ← setup + how to run this experiment
    ├── requirements.txt
    └── <scripts / notebooks>
```

## Adding a new experiment

1. Create a new sibling directory named after the model or approach.
2. Include a short README covering: model, dataset split, how to run, and
   what metric/output the experiment produces.
3. Pin dependencies in a `requirements.txt` (or equivalent) local to that
   experiment so runs stay reproducible.
4. Add a row to the table above.

## Target weed

Chinee apple (*Ziziphus mauritiana*) — thorny shrub / small tree, dense
canopy, small ovate leaves. Visual cues used by these experiments include
leaf shape, canopy texture, and (where in season) fruit colour.
