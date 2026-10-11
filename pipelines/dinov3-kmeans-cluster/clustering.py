"""Annotation-free localization by clustering DINO patch tokens.

Stage 1 of the "model first, labels later" plan. No boxes are ever drawn by
hand. The only human input is a single integer, "cluster 3 is the weed",
chosen after looking at the cluster overlay.

    UAV tile -> DINO patch tokens -> L2-normalize
             -> KMeans over patches -> cluster map
             -> you pick which cluster is chinee apple ONCE, by eye
             -> binary mask + soft heatmap + boxes

Features come from the shared backbones package, so this pipeline and the
supervised probe sit on identical tokens.

CLI:
    python clustering.py --image tile.jpg --out out/ --family dinov2
    python clustering.py --image tile.jpg --out out/ \
        --repo /path/to/dinov3 --weights /path/dinov3_vitb16_*.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# backbones/ is a shared package at the repo root, two levels up.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from backbones import load_backbone, patch_features, pick_device  # noqa: E402



# --------------------------------------------------------------------------- #
# Unsupervised clustering (the annotation-free "labelling")
# --------------------------------------------------------------------------- #
def cluster_patches(feats: np.ndarray, k: int = 12, seed: int = 0):
    """KMeans over L2-normalized patch features. Returns (labels [N], centroids [k, D]).

    On normalized vectors, Euclidean KMeans approximates spherical/cosine
    clustering, which is what we want for DINO features.
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(feats)
    return labels, km.cluster_centers_


def cluster_grid(labels: np.ndarray, grid) -> np.ndarray:
    """Reshape flat patch labels back to the (gh, gw) spatial grid."""
    gh, gw = grid
    return labels.reshape(gh, gw)


def centroid_heatmap(feats: np.ndarray, centroid: np.ndarray, grid) -> np.ndarray:
    """Soft score per patch = cosine similarity to the chosen cluster centroid.

    Gives a smooth (gh, gw) confidence map instead of a hard cluster membership,
    which upsamples into a nicer heatmap and mirrors the few-shot / probe score.
    """
    c = centroid / (np.linalg.norm(centroid) + 1e-8)
    sim = feats @ c                                   # feats already L2-normalized
    gh, gw = grid
    return sim.reshape(gh, gw)


# --------------------------------------------------------------------------- #
# Grid -> image-space mask / boxes
# --------------------------------------------------------------------------- #
def upsample_grid(grid_arr: np.ndarray, size, nearest: bool = True) -> np.ndarray:
    """Upsample a (gh, gw) grid to image (W, H). nearest for labels, bilinear for heat."""
    W, H = size
    mode = Image.NEAREST if nearest else Image.BILINEAR
    im = Image.fromarray(grid_arr.astype(np.float32))
    return np.asarray(im.resize((W, H), mode))


def boxes_from_mask(mask: np.ndarray, min_area: int = 64):
    """Connected-component boxes over a binary mask -> [(x0,y0,x1,y1,area), ...]."""
    from scipy.ndimage import find_objects, label

    lbl, n = label(mask)
    boxes = []
    # find_objects wants the integer label array, not a boolean mask. Passing
    # `lbl == i` raised TypeError on scipy >= 1.15, and rescanned the whole
    # array once per component. One call returns every bounding slice.
    for i, sl in enumerate(find_objects(lbl), start=1):
        if sl is None:
            continue
        area = int((lbl[sl] == i).sum())
        if area < min_area:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        boxes.append((x0, y0, x1, y1, area))
    boxes.sort(key=lambda b: -b[4])
    return boxes


def localize(feats, labels, centroids, target_cluster, grid, base_image,
             heat_threshold: float = 0.5, min_area: int = 64):
    """Turn a chosen cluster into an image-space mask, soft heatmap, and boxes.

    base_image: PIL image (used only for output size). heat_threshold is applied
    to the min-max-normalized centroid similarity, so it is a relative cutoff.
    """
    if isinstance(base_image, (str, os.PathLike)):
        base_image = Image.open(base_image)
    base_image = base_image.convert("RGB")
    W, H = base_image.size

    lbl_grid = cluster_grid(labels, grid)
    hard = (lbl_grid == target_cluster).astype(np.uint8)

    heat = centroid_heatmap(feats, centroids[target_cluster], grid)
    heat_n = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    mask_full = upsample_grid(hard, (W, H), nearest=True).astype(np.uint8)
    heat_full = upsample_grid(heat_n, (W, H), nearest=False)
    # Intersect the hard cluster mask with the soft-confidence cutoff.
    mask = ((mask_full > 0) & (heat_full >= heat_threshold)).astype(np.uint8)

    return {
        "target_cluster": int(target_cluster),
        "size": (W, H),
        "cluster_grid": lbl_grid,          # (gh, gw) int
        "heat": heat_full,                 # (H, W) float in [0,1]
        "mask": mask,                      # (H, W) uint8
        "boxes": boxes_from_mask(mask, min_area),
        "base_image": base_image,
    }


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def _tab_colors(k: int) -> np.ndarray:
    import matplotlib.cm as cm

    cmap = cm.get_cmap("tab10" if k <= 10 else "tab20", k)
    return (np.array([cmap(i)[:3] for i in range(k)]) * 255).astype(np.uint8)


def cluster_overlay(base_image, labels, grid, k: int, alpha: float = 0.55):
    """Colour every patch by its cluster id, overlaid on the tile (for the 'pick' step)."""
    if isinstance(base_image, (str, os.PathLike)):
        base_image = Image.open(base_image)
    base = np.asarray(base_image.convert("RGB"))
    W, H = base_image.size
    lbl_full = upsample_grid(cluster_grid(labels, grid), (W, H), nearest=True).astype(int)
    colors = _tab_colors(k)
    colored = colors[lbl_full]
    return (alpha * colored + (1 - alpha) * base).astype(np.uint8)


def save_outputs(result: dict, out_dir: str, stem: str = "weed") -> None:
    import matplotlib.cm as cm

    os.makedirs(out_dir, exist_ok=True)
    base = np.asarray(result["base_image"])

    hcol = (cm.inferno(np.clip(result["heat"], 0, 1))[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.55 * base + 0.45 * hcol).astype(np.uint8)
    Image.fromarray(overlay).save(os.path.join(out_dir, f"{stem}_heatmap.png"))

    boxed = result["base_image"].copy()
    draw = ImageDraw.Draw(boxed)
    for (x0, y0, x1, y1, _area) in result["boxes"]:
        draw.rectangle([x0, y0, x1, y1], outline="lime", width=3)
    boxed.save(os.path.join(out_dir, f"{stem}_boxes.png"))

    Image.fromarray(result["mask"] * 255).save(os.path.join(out_dir, f"{stem}_mask.png"))

    cov = 100.0 * result["mask"].mean()
    print(f"[out] cluster {result['target_cluster']}: {len(result['boxes'])} region(s), "
          f"{cov:.1f}% of tile flagged -> {out_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="DINOv3 ViT unsupervised weed localization")
    ap.add_argument("--image", required=True, help="UAV image / tile")
    ap.add_argument("--out", default="vit_out", help="Output directory")
    ap.add_argument("--family", default="dinov3", choices=list(FAMILY_DEFAULTS),
                    help="dinov3 (gated, needs --repo/--weights) or dinov2 (ungated fallback)")
    ap.add_argument("--repo", default=None, help="Local clone of facebookresearch/dinov3")
    ap.add_argument("--weights", default=None, help="DINOv3 .pth checkpoint")
    ap.add_argument("--arch", default=None, help="Hub entrypoint override")
    ap.add_argument("--img-size", type=int, default=768, help="Square input side (snapped to patch)")
    ap.add_argument("-k", "--clusters", type=int, default=12, help="Number of KMeans clusters")
    ap.add_argument("--target-cluster", type=int, default=None,
                    help="Cluster id = chinee apple. If omitted, only writes the cluster overlay "
                         "so you can pick, then re-run with this set.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Relative heatmap cutoff [0,1]")
    ap.add_argument("--min-area", type=int, default=64, help="Min box area in px")
    args = ap.parse_args()

    model, meta = load_backbone(args.family, args.repo, args.weights, args.arch)
    feats, grid = patch_features(model, args.image, meta, args.img_size)
    labels, centroids = cluster_patches(feats, k=args.clusters)

    base = Image.open(args.image).convert("RGB")
    stem = os.path.splitext(os.path.basename(args.image))[0]
    os.makedirs(args.out, exist_ok=True)

    overlay = cluster_overlay(base, labels, grid, args.clusters)
    Image.fromarray(overlay).save(os.path.join(args.out, f"{stem}_clusters.png"))
    print(f"[clusters] wrote {stem}_clusters.png (k={args.clusters}). "
          "Open it, note which cluster covers chinee apple, then re-run with "
          "--target-cluster <id>.")

    if args.target_cluster is None:
        return

    result = localize(feats, labels, centroids, args.target_cluster, grid, base,
                      heat_threshold=args.threshold, min_area=args.min_area)
    save_outputs(result, args.out, stem=stem)


if __name__ == "__main__":
    main()
