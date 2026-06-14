"""
DINOv3 single-object localization demo.

Loads the official DINOv3 backbone (https://github.com/facebookresearch/dinov3),
extracts per-patch features, finds the foreground via cosine similarity between
the [CLS] token and each patch token, and draws a bounding box around the
dominant object. A Gradio UI lets you drag in an image.

Setup
-----
1. Clone the official repo:
       git clone https://github.com/facebookresearch/dinov3.git
2. Accept the DINOv3 license and download weights from
       https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
   ViT-B/16 (~330 MB) is a good default.
3. pip install -r requirements.txt
4. Run:
       python dinov3_detect.py \
           --repo    /path/to/dinov3 \
           --weights /path/to/dinov3_vitb16_pretrain_lvd1689m.pth

Gradio opens a browser tab where you can upload an image and see the box.
"""

import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

PATCH = 16
IMG_SIZE = 768

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(repo: str, weights: str, arch: str, device: str):
    model = torch.hub.load(repo, arch, source="local", weights=weights)
    return model.to(device).eval()


def preprocess(image: Image.Image) -> torch.Tensor:
    tx = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tx(image.convert("RGB")).unsqueeze(0)


@torch.inference_mode()
def cls_saliency(model, image: Image.Image, device: str) -> np.ndarray:
    x = preprocess(image).to(device)
    out = model.forward_features(x)
    cls = out["x_norm_clstoken"][0]
    patches = out["x_norm_patchtokens"][0]
    sim = torch.nn.functional.cosine_similarity(patches, cls.unsqueeze(0), dim=-1)
    grid = IMG_SIZE // PATCH
    return sim.float().cpu().numpy().reshape(grid, grid)


def foreground_mask(sim: np.ndarray) -> np.ndarray:
    s = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    return (s > s.mean()).astype(np.uint8)


def largest_component_bbox(mask: np.ndarray):
    from scipy.ndimage import find_objects, label

    lbl, n = label(mask)
    if n == 0:
        return None
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    idx = int(sizes.argmax())
    sl = find_objects(lbl == idx)[0]
    return sl[1].start, sl[0].start, sl[1].stop, sl[0].stop


def annotate(image: Image.Image, bbox_grid, grid_size: int) -> Image.Image:
    if bbox_grid is None:
        return image
    W, H = image.size
    sx, sy = W / grid_size, H / grid_size
    x0, y0, x1, y1 = bbox_grid
    out = image.copy()
    ImageDraw.Draw(out).rectangle(
        [x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline="red", width=4
    )
    return out


def build_inferer(repo: str, weights: str, arch: str):
    device = pick_device()
    model = load_model(repo, weights, arch, device)

    def infer(image: Image.Image) -> Image.Image:
        sim = cls_saliency(model, image, device)
        bbox = largest_component_bbox(foreground_mask(sim))
        return annotate(image, bbox, sim.shape[0])

    return infer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="Path to a local clone of facebookresearch/dinov3")
    p.add_argument("--weights", required=True, help="Path to a DINOv3 .pth checkpoint")
    p.add_argument("--arch", default="dinov3_vitb16", help="Hub entrypoint name (default: dinov3_vitb16)")
    p.add_argument("--share", action="store_true", help="Expose a public Gradio URL")
    args = p.parse_args()

    import gradio as gr

    infer = build_inferer(args.repo, args.weights, args.arch)
    gr.Interface(
        fn=infer,
        inputs=gr.Image(type="pil", label="Upload"),
        outputs=gr.Image(type="pil", label="Detected object"),
        title="DINOv3 object localization",
        description=(
            "Foreground is the set of patches most similar to the [CLS] token; "
            "the largest connected blob is boxed. Backbone-only — no class label."
        ),
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
