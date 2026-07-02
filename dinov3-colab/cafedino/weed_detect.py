"""CAFe-DINO open-vocabulary weed detection for UAV imagery.

Runs the pretrained CAFe-DINO model (DINO Soars, Faulkenberry & Prasad, CVPRW
2026) as an *open-vocabulary* segmenter on drone imagery, with no training of
our own. We simply prompt it with "chinee apple" plus a handful of distractor
classes (grass, tree, bare soil, ...), take the per-class similarity maps that
the cost-aggregation network cleans up, and turn the chinee-apple channel into a
weed probability heatmap, binary mask, and bounding boxes.

Pipeline (identical core to the repo's analysis.py, repackaged):
    image -> DINOv3.txt patch tokens x class text embeddings -> cost volume
          -> cost aggregation -> AnyUp upsample -> per-pixel per-class logits
          -> softmax -> chinee-apple channel -> threshold -> connected components

Prerequisites:
    * git clone https://github.com/rfaulk/DINO_Soars.git  (this script auto-patches
      the two hardcoded developer weight paths it ships with)
    * env DINOV3_VITL16_WEIGHTS   -> ViT-L/16 LVD-1689M backbone .pth
    * env DINOV3_DINOTXT_WEIGHTS  -> dinotxt vision-head + text-encoder .pth
    * trained CAFe-DINO checkpoint (huggingface.co/rfaulken/cafedino)

The Colab notebook (dinov3_colab.ipynb §8) does the same thing inline; this
module is the standalone CLI mirror.

Example:
    python cafedino/weed_detect.py \
        --dinosoars /content/DINO_Soars \
        --cafedino-weights /content/DINO_Soars/checkpoints/best.pth \
        --image field_tile.jpg --out out/

Importable API:
    model, tok = load_cafedino(dinosoars, cafedino_weights)
    result = detect(model, tok, "tile.jpg")          # dict of arrays
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# ImageNet stats — DINOv3 and AnyUp both expect ImageNet-normalized RGB input.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Prompt-ensemble templates (same set the repo uses for evaluation).
PROMPT_TEMPLATES = (
    "a photo of {}", "an image of {}", "a photograph of {}", "a picture of {}",
    "a photo of a {}", "an image of a {}", "a photo of the {}", "an image of the {}",
    "a close-up photo of {}", "an aerial image of {}", "a drone image of {}",
    "a satellite image of {}",
)

# Chinee apple (Ziziphus mauritiana) is index 0. The other classes are
# distractors so the argmax/softmax has something to compete against — good
# open-vocabulary segmentation needs the scene's other land cover named too.
DEFAULT_CLASSES = [
    "chinee apple",   # target weed (index 0)
    "grass",
    "tree",
    "bare soil",
    "shrub",
    "dry grass",
    "road",
]


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _patch_hardcoded_paths(dinosoars_root: str) -> None:
    """Rewrite the two `torch.load('/home/rfaulken/...')` lines DINO_Soars ships
    with so they read the DINOV3_* env vars instead. Idempotent."""
    import re
    targets = [
        ("dinov3/hub/backbones.py", "DINOV3_VITL16_WEIGHTS", "lvd1689m"),
        ("dinov3/hub/dinotxt.py", "DINOV3_DINOTXT_WEIGHTS", "dinotxt"),
    ]
    for rel, var, tag in targets:
        path = os.path.join(dinosoars_root, rel)
        if not os.path.isfile(path):
            continue
        text = open(path, encoding="utf-8").read()
        new_text, n = re.subn(
            r"torch\.load\(\s*['\"]/home/rfaulken/[^'\"]*" + tag + r"[^'\"]*['\"]\s*\)",
            f"torch.load(os.environ['{var}'], map_location='cpu')",
            text,
        )
        if n == 0:
            continue  # already patched or upstream changed
        if not re.search(r"^import os$", new_text, re.MULTILINE):
            new_text = "import os\n" + new_text
        open(path, "w", encoding="utf-8").write(new_text)
        print(f"[patch] {rel} -> {var}")


def _add_paths(dinosoars_root: str) -> None:
    for p in (dinosoars_root,
              os.path.join(dinosoars_root, "CAFe_DINO"),
              os.path.join(dinosoars_root, "anyup")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_upsampler(dinosoars_root: str, device: str):
    """Prefer the vendored AnyUp (offline); fall back to torch.hub."""
    local_anyup = os.path.join(dinosoars_root, "anyup")
    try:
        return torch.hub.load(local_anyup, "anyup", source="local",
                              verbose=False).to(device).eval()
    except Exception as exc:  # noqa: BLE001 - want any failure to trigger fallback
        print(f"[anyup] local load failed ({exc}); trying torch.hub remote ...")
        return torch.hub.load("wimmerth/anyup", "anyup",
                              verbose=False).to(device).eval()


def _infer_aggregator_dim(state_dict: dict) -> int:
    """corr_embed is Conv2d(1, aggregator_dim, 7) -> out_channels == agg dim."""
    w = state_dict.get("corr_embed.weight")
    if w is None:
        raise KeyError("checkpoint has no 'corr_embed.weight'; is this a CAFe-DINO ckpt?")
    return int(w.shape[0])


def load_cafedino(dinosoars_root: str, cafedino_weights: str, device: str | None = None,
                  input_size: int = 224):
    """Load DINOv3.txt backbone + AnyUp + the trained CAFe-DINO aggregator."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    for var in ("DINOV3_VITL16_WEIGHTS", "DINOV3_DINOTXT_WEIGHTS"):
        if not os.environ.get(var):
            raise EnvironmentError(
                f"Environment variable {var} is not set. Point it at the gated "
                "DINOv3 ViT-L/16 checkpoint (see cafedino/setup.sh)."
            )

    _patch_hardcoded_paths(dinosoars_root)
    _add_paths(dinosoars_root)
    from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
    from CAFe_DINO.modeling.cafedino import CAFe_DINO

    torch.set_float32_matmul_precision("high")

    backbone, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
    backbone.to(device).eval()
    upsampler = _load_upsampler(dinosoars_root, device)

    ckpt = torch.load(cafedino_weights, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    # Strip torch.compile's "_orig_mod." prefix if present.
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    agg_dim = _infer_aggregator_dim(state_dict)
    print(f"[cafedino] detected aggregator_dim={agg_dim}")

    model = CAFe_DINO(
        backbone, tokenizer, upsampler,
        input_resolution=(input_size // 16, input_size // 16),
        device=device, aggregator_dim=agg_dim,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # The frozen backbone/upsampler weights come from their own checkpoints, so
    # they legitimately appear in `missing` here — only flag aggregator gaps.
    agg_missing = [k for k in missing if k.startswith(("corr_embed", "aggregator",
                                                       "reduce_d", "text_guidance",
                                                       "vis_guidance"))]
    if agg_missing:
        print(f"[cafedino] WARNING missing aggregator keys: {agg_missing[:6]} ...")
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Text + inference primitives (ported verbatim from DINO_Soars/CAFe_DINO/utils.py)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _text_embed(backbone, tokenizer, class_names, device):
    prompts = [t.format(n) for n in class_names for t in PROMPT_TEMPLATES]
    toks = tokenizer.tokenize(prompts).to(device)
    embs = backbone.encode_text(toks)[:, 1024:]          # keep the patch-aligned half
    C, K, D = len(class_names), len(PROMPT_TEMPLATES), embs.size(1)
    embs = embs.view(C, K, D).mean(dim=1)                # average over templates
    return F.normalize(embs, p=2, dim=1)                 # [C, D]


@torch.no_grad()
def build_text_embeddings(backbone, tokenizer, class_names, device):
    return _text_embed(backbone, tokenizer, class_names, device)


@torch.no_grad()
def strided_inference(model, img, text_emb, side, stride, num_classes, device):
    """Overlapping sliding-window inference, averaged where windows overlap."""
    _, _, H, W = img.shape
    probs = torch.zeros(num_classes, H, W, device=device)
    counts = torch.zeros(H, W, device=device)
    h_grids = max(H - side + stride - 1, 0) // stride + 1
    w_grids = max(W - side + stride - 1, 0) // stride + 1
    for i in range(h_grids):
        for j in range(w_grids):
            y1, x1 = i * stride, j * stride
            y2, x2 = min(y1 + side, H), min(x1 + side, W)
            y1, x1 = max(y2 - side, 0), max(x2 - side, 0)
            window = img[:, :, y1:y2, x1:x2]
            out = model(window, text_emb, pre_text_emb=True).squeeze(0)  # [C, h, w]
            probs[:, y1:y2, x1:x2] += out
            counts[y1:y2, x1:x2] += 1
    probs /= counts.clamp(min=1)
    return probs  # [C, H, W] (aggregated logits)


# --------------------------------------------------------------------------- #
# Preprocessing + postprocessing
# --------------------------------------------------------------------------- #
def _preprocess(pil_img: Image.Image, size: int | None, device: str) -> torch.Tensor:
    img = pil_img.convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return t.to(device)


def _bboxes_from_mask(mask: np.ndarray, min_area: int):
    """Connected-component boxes over a binary mask -> [(x0,y0,x1,y1,area), ...]."""
    from scipy.ndimage import find_objects, label

    lbl, n = label(mask)
    boxes = []
    for i in range(1, n + 1):
        sl = find_objects(lbl == i)[0]
        area = int((lbl[sl] == i).sum())
        if area < min_area:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        boxes.append((x0, y0, x1, y1, area))
    boxes.sort(key=lambda b: -b[4])
    return boxes


@torch.no_grad()
def detect(model, tokenizer, image_path, class_names=None, target_index=0,
           resize=512, side=224, stride=112, threshold=0.5, min_area=64,
           device=None):
    """Run open-vocabulary detection; returns a dict of numpy arrays + boxes.

    resize=None runs strided inference at the image's native resolution (use for
    large UAV tiles where downscaling would erase small weeds).
    """
    if device is None:
        device = next(model.parameters()).device.type
    class_names = class_names or DEFAULT_CLASSES

    pil = Image.open(image_path)
    x = _preprocess(pil, resize, device)
    text_emb = build_text_embeddings(model.backbone, tokenizer, class_names, device)

    with torch.amp.autocast(device) if device == "cuda" else _nullctx():
        logits = strided_inference(model, x, text_emb, side, stride,
                                   len(class_names), device)  # [C, H, W]

    prob = torch.softmax(logits.float(), dim=0)          # per-pixel class probs
    seg = prob.argmax(0).cpu().numpy().astype(np.int32)  # class index map
    heat = prob[target_index].cpu().numpy()              # target weed probability
    mask = (heat >= threshold).astype(np.uint8)
    boxes = _bboxes_from_mask(mask, min_area)

    _, _, H, W = x.shape
    return {
        "class_names": class_names,
        "target_index": target_index,
        "size": (H, W),
        "seg": seg,           # (H, W) int class map
        "heat": heat,         # (H, W) float P(target)
        "mask": mask,         # (H, W) uint8 binary
        "boxes": boxes,       # list of (x0, y0, x1, y1, area)
        "base_image": pil.convert("RGB").resize((W, H), Image.BILINEAR),
    }


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# --------------------------------------------------------------------------- #
# Visualization / saving
# --------------------------------------------------------------------------- #
def save_outputs(result: dict, out_dir: str, stem: str = "weed") -> None:
    import matplotlib.cm as cm

    os.makedirs(out_dir, exist_ok=True)
    base = result["base_image"]
    heat = result["heat"]

    # 1. Heatmap overlay (inferno).
    hcol = (cm.inferno(np.clip(heat, 0, 1))[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.55 * np.asarray(base) + 0.45 * hcol).astype(np.uint8)
    Image.fromarray(overlay).save(os.path.join(out_dir, f"{stem}_heatmap.png"))

    # 2. Boxed detections on the RGB tile.
    boxed = base.copy()
    draw = ImageDraw.Draw(boxed)
    for (x0, y0, x1, y1, area) in result["boxes"]:
        draw.rectangle([x0, y0, x1, y1], outline="lime", width=3)
    boxed.save(os.path.join(out_dir, f"{stem}_boxes.png"))

    # 3. Raw binary mask.
    Image.fromarray(result["mask"] * 255).save(os.path.join(out_dir, f"{stem}_mask.png"))

    n = len(result["boxes"])
    cov = 100.0 * result["mask"].mean()
    print(f"[out] {n} chinee-apple region(s), {cov:.1f}% of tile flagged -> {out_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="CAFe-DINO chinee apple weed detection")
    ap.add_argument("--dinosoars", required=True, help="Path to cloned+patched DINO_Soars")
    ap.add_argument("--cafedino-weights", required=True, help="Trained CAFe-DINO .pth")
    ap.add_argument("--image", required=True, help="UAV image / tile")
    ap.add_argument("--out", default="cafedino_out", help="Output directory")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="Class prompts; target must be first (default: chinee apple + distractors)")
    ap.add_argument("--resize", type=int, default=512,
                    help="Square resize before inference; 0 = native resolution")
    ap.add_argument("--side", type=int, default=224, help="Sliding window size")
    ap.add_argument("--stride", type=int, default=112, help="Sliding window stride")
    ap.add_argument("--threshold", type=float, default=0.5, help="P(weed) mask threshold")
    ap.add_argument("--min-area", type=int, default=64, help="Min box area in px")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = ap.parse_args()

    model, tok = load_cafedino(args.dinosoars, args.cafedino_weights, args.device)
    result = detect(
        model, tok, args.image,
        class_names=args.classes,
        resize=(None if args.resize == 0 else args.resize),
        side=args.side, stride=args.stride,
        threshold=args.threshold, min_area=args.min_area,
    )
    save_outputs(result, args.out, stem=os.path.splitext(os.path.basename(args.image))[0])


if __name__ == "__main__":
    main()
