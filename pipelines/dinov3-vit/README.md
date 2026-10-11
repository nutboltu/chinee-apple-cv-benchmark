# DINOv3 ViT — unsupervised chinee apple localization

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nutboltu/chinee-apple-cv-benchmark/blob/main/dinov3-vit/dinov3_vit_colab.ipynb)

Stage 1 of a **"model first, labels later"** pipeline for detecting
**Chinee apple** (*Ziziphus mauritiana*) in UAV imagery — with **no manual
annotation**. We run a frozen DINOv3 ViT-B/16 backbone, cluster its per-patch
features, and let you assign *one integer* ("cluster 3 is the weed") after
looking at the result. Those cluster ids become the pseudo-labels for a linear
probe in stage 2.

This is a fresh, minimal, self-contained experiment — distinct from
[`../dinov3-colab/`](../dinov3-colab), which mixes `[CLS]`-saliency, a supervised
probe, and CAFe-DINO in one large notebook.

## Why this answers "how do we label without annotating?"

DINOv3 is self-supervised, so the **features are free**. The only thing missing
is the *name* "chinee apple". Instead of hand-drawing boxes, we recover it from
structure in the features:

```
UAV tile ─▶ DINOv3 ViT-B/16 patch tokens ─▶ L2-normalize
        ─▶ KMeans (k≈12) over patches ─▶ cluster map overlay
        ─▶ you glance once: "cluster 3 = chinee apple"      ← the only human input
        ─▶ hard mask ∩ soft centroid-similarity heatmap ─▶ boxes
```

Total manual effort: **one number per site**, not thousands of boxes. And
because `localize()` returns the per-patch features *and* their cluster ids,
promoting these to pseudo-labels for a trained head (stage 2) is a direct
hand-off — no re-extraction.

## Run it

**Colab (recommended):** click the badge, run top to bottom. It clones this
repo, installs deps, loads the backbone, and walks the cluster → pick → mask
flow on a tile you upload.

**Local / server:**

```bash
pip install -r requirements.txt

# DINOv2 fallback — ungated, works today, same code path:
python vit_features.py --image tile.jpg --out out/ --family dinov2 -k 12
#   → writes tile_clusters.png; open it, find the weed's cluster id, then:
python vit_features.py --image tile.jpg --out out/ --family dinov2 -k 12 \
       --target-cluster 3

# DINOv3 ViT-B/16 — gated: accept Meta's license, clone the repo, download weights
#   https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
git clone https://github.com/facebookresearch/dinov3.git
python vit_features.py --image tile.jpg --out out/ \
       --repo ./dinov3 --weights ./dinov3_vitb16_pretrain_lvd1689m.pth \
       -k 12 --target-cluster 3
```

Outputs: `*_clusters.png` (pick from this), `*_heatmap.png`, `*_mask.png`,
`*_boxes.png`.

### DINOv3 weights are gated

Official DINOv3 checkpoints sit behind Meta's license. Until yours clears, run
`--family dinov2` (ViT-B/14, loads straight from `torch.hub`) — it exercises the
entire pipeline unchanged, then swap `--family dinov3 --repo … --weights …` when
the weights land. `forward_features` returns the same dict for both.

## Tuning

- **`-k` (clusters):** more clusters = finer land-cover separation. Start at 12.
  If chinee apple shares a cluster with other trees, raise `k`; if it fragments
  across several, lower it or merge ids.
- **`--img-size`:** larger keeps small shrubs resolvable (snapped to the patch
  size). Very large orthomosaics should be tiled first (sliding window is a
  planned addition).
- **`--threshold`:** relative cutoff on the min-max-normalized centroid
  similarity; lower for recall on faint canopy.

## Stage 2 — linear probe (notebook §7–8)

Once the cluster picks look right, the notebook promotes them to labels and
trains a probe on the frozen features — no hand-drawn crops:

1. Save the validated cluster's centroid as the chinee-apple **reference**.
2. Re-cluster each training tile and assign the positive cluster by nearest
   centroid to that reference — this handles per-tile KMeans ids being
   arbitrary and not comparable across tiles.
3. Pool `(feats, binary label)` and fit a logistic-regression head on the
   frozen tokens (same head as `dinov3-colab`).
4. Apply the probe per patch on new tiles for class-aware heatmaps + boxes — no
   re-clustering, and it generalizes across tiles where raw KMeans ids would not.

## Files

| File | Purpose |
|---|---|
| `dinov3_vit_colab.ipynb` | Self-contained Colab walkthrough (cluster → pick → mask → probe). |
| `vit_features.py` | Importable API + CLI mirror of the notebook. |
| `requirements.txt` | Pinned-light deps. |
