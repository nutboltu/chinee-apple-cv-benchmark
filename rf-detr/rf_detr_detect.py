"""
RF-DETR object detection demo.

Loads RF-DETR (https://github.com/roboflow/rf-detr) with its default
COCO-pretrained weights, runs detection on an uploaded image, and draws
boxes + class labels via supervision. A Gradio UI lets you drag in an image
and tune the confidence threshold.

Notes
-----
- The default checkpoint detects COCO classes (80). On weed imagery it
  will usually fire on "potted plant" or nothing — fine-tune on a labeled
  Chinee apple dataset for real detection.
- To use a fine-tuned checkpoint, pass --weights /path/to/checkpoint.pth
  (RF-DETR's standard fine-tuning export format).

Setup
-----
    pip install -r requirements.txt
    python rf_detr_detect.py
"""

import argparse

import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES


def build_inferer(weights: str | None):
    model = RFDETRBase(pretrain_weights=weights) if weights else RFDETRBase()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    def infer(image: Image.Image, threshold: float) -> Image.Image:
        detections = model.predict(image, threshold=threshold)
        labels = [
            f"{COCO_CLASSES[cid]} {conf:.2f}"
            for cid, conf in zip(detections.class_id, detections.confidence)
        ]
        annotated = image.copy()
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels)
        return annotated

    return infer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None, help="Optional path to fine-tuned RF-DETR weights")
    p.add_argument("--threshold", type=float, default=0.3, help="Default confidence threshold")
    p.add_argument("--share", action="store_true", help="Expose a public Gradio URL")
    args = p.parse_args()

    import gradio as gr

    infer = build_inferer(args.weights)
    gr.Interface(
        fn=infer,
        inputs=[
            gr.Image(type="pil", label="Upload"),
            gr.Slider(0.05, 0.95, value=args.threshold, step=0.05, label="Confidence threshold"),
        ],
        outputs=gr.Image(type="pil", label="Detections"),
        title="RF-DETR object detection",
        description=(
            "COCO-pretrained baseline. Boxes + class labels via supervision. "
            "Swap in a fine-tuned checkpoint via --weights for Chinee apple detection."
        ),
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
