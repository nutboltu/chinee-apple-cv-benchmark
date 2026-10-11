# dinov3-cafe — open-vocabulary detection, prompted with text

Finds chinee apple by *naming* it. No training, no labels, no probe. Change the
prompt list and you change the classes.

```
UAV tile -> DINOv3.txt patch tokens (x) class text embeddings -> cost volume
         -> cost aggregation (Swin + channel attention) -> AnyUp upsample
         -> per-pixel per-class logits -> "chinee apple" channel
         -> threshold -> connected components -> mask + boxes
```

Uses [CAFe-DINO](https://github.com/rfaulk/DINO_Soars) over the text-aligned
**DINOv3.txt ViT-L/16** backbone, which is a different and larger checkpoint
than the ViT-B/16 the other pipelines use. Both DINOv3 weight files are gated
and must be downloaded from Meta after accepting the licence.

## How it compares

| | [dinov3-kmeans-cluster](../dinov3-kmeans-cluster) | [dinov3-linear-probe](../dinov3-linear-probe) | dinov3-cafe |
| --- | --- | --- | --- |
| Labels needed | none | CVAT + SAM3 polygons | none |
| Human per run | picks a cluster | none | writes a prompt |
| Output | patch mask | patch probability | full-resolution mask |
| New class | recluster | retrain the head | edit the prompt |
| Scored | no | yes, AP / IoU / mAP | not yet |

## Contents

| path | what it is |
| --- | --- |
| `cafedino/` | CAFe-DINO. Standalone CLI mirror of section 8 of the notebook. |
| `dinov3_colab.ipynb` | Sections 0 to 5 are a CLS-saliency demo with a Gradio UI, 6 to 7 an early crop-based probe, 8 is CAFe-DINO. |
| `dinov3_detect.py` | The CLS-saliency localizer as a script. |

**Be aware the notebook is mixed.** Sections 6 and 7 train a probe on labelled
*crops*, which [`dinov3-linear-probe`](../dinov3-linear-probe) supersedes by
training on real polygons and reporting honest held-out metrics. Section 8 is
the part that nothing else in this repo does.

## Not yet scored

This pipeline produces masks but has never been run against the annotations, so
there are no AP or IoU numbers for it. Scoring it with the shared
[`metrics`](../../metrics) package would make it directly comparable with the
probe, and is the obvious next step. The prompt wording matters a lot, since
"chinee apple" is rare in DINOv3's text pretraining.
