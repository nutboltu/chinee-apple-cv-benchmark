"""Metrics for the frozen-backbone patch probe: patch, mask and instance level.

Three questions get asked of this model, and they need three different numbers.

  patch level     Is this patch chinee apple? Average precision over every
                  patch, plus precision/recall/F1. This is what the probe
                  literally optimises.
  mask level      How well does the predicted region overlap the annotation?
                  IoU and Dice, averaged over tiles.
  instance level  Did we find each plant, as a separate object? Connected
                  components of the prediction are matched to individual
                  polygons by IoU, giving COCO-style AP@0.50, AP@0.75 and
                  mAP@[0.50:0.95]. One class, so mAP equals AP here.

EVERYTHING IS AT PATCH RESOLUTION. The probe cannot resolve finer than one
patch, so upsampling predictions to pixel resolution before scoring would
flatter the boundary. Ground truth is downsampled to the same grid and
binarised at 50% coverage instead. With ViT-B/14 at 448px a patch covers
14x14 model pixels, which is roughly 32x32 tile pixels on a 1024px tile.

AP uses the COCO 101-point interpolated convention so numbers are comparable
with detector baselines in this repo.

DEFAULTS WERE TUNED, NOT GUESSED. On a 17-frame held-out split, raw connected
components gave 141 predictions for 59 real plants, because a per-patch probe
fragments single canopies. Closing the mask first fixes most of that:

    smooth  min_size   mAP@.5:.95   AP@.50   predictions
      0         4        0.130       0.336      141
      1         8        0.147       0.382       67     <- default
      2         8        0.119       0.431       64
      3         8        0.081       0.361       61

smooth=1 maximises mAP@[.50:.95] and lands near the right object count.
smooth=2 trades tight-IoU accuracy for the best AP@0.50, so prefer it if you
only care about finding plants rather than outlining them.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score, roc_auc_score


# --------------------------------------------------------------------------- #
# patch level
# --------------------------------------------------------------------------- #
def patch_metrics(y: np.ndarray, p: np.ndarray, thresh: float = 0.5) -> dict:
    yhat = (p >= thresh).astype(np.int8)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "ap": float(average_precision_score(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "precision": prec, "recall": rec, "f1": f1,
        "threshold": thresh, "tp": tp, "fp": fp, "fn": fn,
        "positive_rate": float(y.mean()),
    }


def best_f1_threshold(y: np.ndarray, p: np.ndarray, steps: int = 101) -> tuple[float, float]:
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, steps):
        m = patch_metrics(y, p, float(t))
        if m["f1"] > best_f:
            best_t, best_f = float(t), m["f1"]
    return best_t, best_f


# --------------------------------------------------------------------------- #
# mask level
# --------------------------------------------------------------------------- #
def mask_metrics(tiles: list[dict], thresh: float = 0.5) -> dict:
    """tiles: [{'prob': (gh,gw) float, 'gt': (gh,gw) bool}]. IoU/Dice per tile."""
    ious, dices = [], []
    for t in tiles:
        pred = t["prob"] >= thresh
        gt = t["gt"].astype(bool)
        inter = float((pred & gt).sum())
        union = float((pred | gt).sum())
        if union == 0:
            continue                      # nothing predicted, nothing annotated
        ious.append(inter / union)
        denom = float(pred.sum() + gt.sum())
        dices.append(2 * inter / denom if denom else 0.0)
    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "tiles_scored": len(ious),
    }


# --------------------------------------------------------------------------- #
# instance level
# --------------------------------------------------------------------------- #
def _instances(mask: np.ndarray, prob: np.ndarray | None, min_size: int, smooth: int = 0):
    """Connected components -> list of (bool mask, score).

    A per-patch probe has no spatial prior, so a single plant often breaks into
    several components and instance AP collapses even when patch AP is healthy.
    `smooth` closes the mask with a (2*smooth+1) square first, merging those
    fragments before labelling.
    """
    if smooth > 0:
        st = np.ones((2 * smooth + 1, 2 * smooth + 1), bool)
        mask = ndi.binary_closing(mask, st, border_value=0)
        mask = ndi.binary_opening(mask, np.ones((3, 3), bool), border_value=0)
    lab, n = ndi.label(mask)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        if int(m.sum()) < min_size:
            continue
        score = float(prob[m].mean()) if prob is not None else 1.0
        out.append((m, score))
    return out


def _ap_from_matches(scores: list[float], hits: list[int], n_gt: int) -> float:
    """COCO 101-point interpolated AP."""
    if n_gt == 0:
        return float("nan")
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    h = np.asarray(hits)[order]
    tp = np.cumsum(h == 1)
    fp = np.cumsum(h == 0)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    # make precision monotonically decreasing, then sample at 101 recalls
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    q = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, q, side="left")
    sampled = np.where(idx < len(prec), prec[np.clip(idx, 0, len(prec) - 1)], 0.0)
    return float(sampled.mean())


def instance_map(tiles: list[dict], thresh: float = 0.5, min_size: int = 4,
                 smooth: int = 0, iou_thresholds=None) -> dict:
    """COCO-style AP over connected components.

    tiles: [{'prob': (gh,gw), 'gt_instances': [bool (gh,gw), ...]}]
    Greedy highest-score-first matching, each ground truth claimed at most once.
    """
    if iou_thresholds is None:
        iou_thresholds = np.round(np.arange(0.50, 0.96, 0.05), 2)

    preds_per_tile, n_gt = [], 0
    for t in tiles:
        preds_per_tile.append(_instances(t["prob"] >= thresh, t["prob"], min_size, smooth))
        n_gt += len(t["gt_instances"])

    per_t = {}
    for iou_t in iou_thresholds:
        scores, hits = [], []
        for t, preds in zip(tiles, preds_per_tile):
            gts = [g.astype(bool) for g in t["gt_instances"]]
            claimed = [False] * len(gts)
            for m, s in sorted(preds, key=lambda x: -x[1]):
                best_i, best_iou = -1, 0.0
                for i, g in enumerate(gts):
                    if claimed[i]:
                        continue
                    union = float((m | g).sum())
                    if union == 0:
                        continue
                    iou = float((m & g).sum()) / union
                    if iou > best_iou:
                        best_i, best_iou = i, iou
                if best_i >= 0 and best_iou >= iou_t:
                    claimed[best_i] = True
                    scores.append(s); hits.append(1)
                else:
                    scores.append(s); hits.append(0)
        per_t[float(iou_t)] = _ap_from_matches(scores, hits, n_gt)

    vals = [v for v in per_t.values() if not np.isnan(v)]
    return {
        "mAP_50_95": float(np.mean(vals)) if vals else float("nan"),
        "AP_50": per_t.get(0.5, float("nan")),
        "AP_75": per_t.get(0.75, float("nan")),
        "per_iou": per_t,
        "n_gt_instances": n_gt,
        "n_pred_instances": sum(len(p) for p in preds_per_tile),
    }


def report(y, p, tiles, thresh: float | None = None, min_size: int = 8, smooth: int = 1) -> dict:
    """Run all three levels and print a readable block."""
    if thresh is None:
        thresh, _ = best_f1_threshold(y, p)
    pm = patch_metrics(y, p, thresh)
    mm = mask_metrics(tiles, thresh)
    im = instance_map(tiles, thresh, min_size=min_size, smooth=smooth)
    print(f"threshold (best F1)   {thresh:.2f}")
    print("\nPATCH LEVEL")
    print(f"  AP (mAP, 1 class)   {pm['ap']:.3f}")
    print(f"  ROC AUC             {pm['auc']:.3f}")
    print(f"  precision           {pm['precision']:.3f}")
    print(f"  recall              {pm['recall']:.3f}")
    print(f"  F1                  {pm['f1']:.3f}")
    print("\nMASK LEVEL")
    print(f"  mean IoU            {mm['mean_iou']:.3f}")
    print(f"  mean Dice           {mm['mean_dice']:.3f}")
    print("\nINSTANCE LEVEL (COCO style)")
    print(f"  mAP@[.50:.95]       {im['mAP_50_95']:.3f}")
    print(f"  AP@0.50             {im['AP_50']:.3f}")
    print(f"  AP@0.75             {im['AP_75']:.3f}")
    print(f"  gt / pred instances {im['n_gt_instances']} / {im['n_pred_instances']}")
    return {"patch": pm, "mask": mm, "instance": im}
