# CAFe-DINO for chinee apple detection

Open-vocabulary weed detection on UAV imagery using **CAFe-DINO**
([DINO Soars](https://github.com/rfaulk/DINO_Soars), Faulkenberry & Prasad,
CVPRW 2026). No training required — we run the authors' pretrained model and
prompt it with `"chinee apple"` plus distractor land-cover classes.

**Simplest path:** run **Section 8** of `../dinov3_colab.ipynb` in Colab. It is
self-contained (clone → patch → install → detect), same style as the rest of the
notebook. `weed_detect.py` here is just the standalone CLI mirror of that flow.

## Why this over the linear probe (notebook §6–7)

| | Linear probe (existing) | CAFe-DINO (this) |
|---|---|---|
| Supervision | needs labeled chinee-apple crops | none (open-vocabulary) |
| Granularity | mean-pooled patch grid | full-resolution mask (AnyUp) |
| Similarity quality | raw DINOv3.txt (noisy on aerial) | cost-aggregated (denoised) |
| New classes | retrain probe | just change the prompt list |

The paper's core result: DINOv3 (natural-image-trained) beats remote-sensing
foundation models on aerial segmentation, and cost aggregation "unlocks" its
noisy text–image similarity maps without any RS fine-tuning.

## How it works

```
UAV tile ─▶ DINOv3.txt patch tokens ⊗ class text embeddings ─▶ cost volume
        ─▶ cost aggregation (Swin + channel attn) ─▶ AnyUp upsample
        ─▶ per-pixel per-class logits ─▶ softmax ─▶ "chinee apple" channel
        ─▶ threshold ─▶ connected components ─▶ mask + boxes
```

## Standalone CLI

```bash
git clone https://github.com/rfaulk/DINO_Soars.git /content/DINO_Soars

# gated ViT-L/16 DINOv3 weights (accept Meta's license & download both files):
#   https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
export DINOV3_VITL16_WEIGHTS=/path/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
export DINOV3_DINOTXT_WEIGHTS=/path/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth

pip install einops timm 'albumentations>=2.0' omegaconf ftfy regex \
            opencv-python-headless tifffile huggingface_hub scipy matplotlib

python cafedino/weed_detect.py \
  --dinosoars /content/DINO_Soars \
  --cafedino-weights <cafedino_ckpt.pth from huggingface.co/rfaulken/cafedino> \
  --image field_tile.jpg --out out/
```

`weed_detect.py` auto-patches the two hardcoded developer paths
(`torch.load('/home/rfaulken/...')`) the upstream repo ships with, auto-detects
`aggregator_dim` from the checkpoint, and writes `*_heatmap.png`, `*_boxes.png`,
`*_mask.png`.

> CAFe-DINO uses the **ViT-L/16** backbone (`dino_dim=1024`), not the ViT-B/16
> used elsewhere in this repo. You need the L/16 weights.

### Useful flags
- `--classes "chinee apple" grass tree "bare soil"` — target class must be **first**.
- `--resize 0` — run at native resolution (large orthomosaics; downscaling erases small shrubs).
- `--threshold 0.4` — lower to increase recall on faint detections.

## Tuning for chinee apple

`chinee apple` is a rare term in DINOv3's text pretraining, so prompt wording
matters. If detections are weak, try `"thorny shrub"`, `"green bush"`,
`"chinee apple bush"`. The paper reports the rural-texture case (grass vs.
crop/shrub) as the model's hardest — expect to tune the threshold and distractor
set on a few tiles.
