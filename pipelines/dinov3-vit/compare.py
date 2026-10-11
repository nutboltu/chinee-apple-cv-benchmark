"""Compare probe options side by side on one honest split.

Answers "would a different head, resolution or post-processing be better?" with
numbers instead of opinion. Features are the expensive part, so they are
extracted once per resolution and reused across every head.

Every configuration is scored on the same frame-grouped split, so the rows are
comparable. The split is by SOURCE FRAME because tiles from one frame overlap,
and a random split would make everything look better than it is.

    python compare.py --zips '.../batch_*.zip' --tiles '.../processed' \
        --crib '.../tile_markers.csv' --img-sizes 448 768

Reported per configuration:

  patch     AP        average precision, the headline number for the classifier
            AUC       ranking quality, insensitive to class balance
            P / R     precision and recall at the best-F1 threshold
            F1        their harmonic mean
  mask      IoU       overlap of predicted region with the annotation
            Dice      same idea, more forgiving of small regions
  instance  mAP       COCO mAP@[.50:.95] over connected components
            AP50      the looser "did you find the plant at all" number
            n         predicted instance count, against the true count
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# metrics/ is a shared package at the repo root, two levels up from this file.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from PIL import Image

from metrics import segmentation as pe
import train_probe as tp


def build_heads(seed: int = 0) -> dict:
    """Candidate heads. All take L2-normalised patch tokens."""
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        "logreg_C0.1": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1),
        "linear_svm": SGDClassifier(loss="modified_huber", class_weight="balanced",
                                    max_iter=2000, tol=1e-4, random_state=seed),
        "mlp_64": MLPClassifier(hidden_layer_sizes=(64,), max_iter=400,
                                early_stopping=True, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=120, min_samples_leaf=4,
                                                class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=seed),
    }


def extract(names, tiles, polys, crib, model, meta, img_size):
    """One pass of the frozen backbone. Returns per-tile records."""
    recs = []
    for i, n in enumerate(names, 1):
        import vit_features as vf
        f, (gh, gw) = vf.patch_features(model, str(tiles[n]), meta, img_size=img_size)
        f = np.asarray(f, dtype=np.float32)
        f /= (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        with Image.open(tiles[n]) as im:
            size = im.size
        side = gh * meta["patch"]
        cov = tp.patch_coverage(tp.tile_mask(size, polys[n], crib.get(n)), side, meta["patch"])
        inst = []
        for pts in polys[n]:
            g = tp.patch_coverage(tp.tile_mask(size, [pts], crib.get(n)), side, meta["patch"]) >= 0.5
            if g.any():
                inst.append(g)
        recs.append(dict(name=n, feat=f, cov=cov, grid=(gh, gw), inst=inst, src=tp.source_of(n)))
        if i % 40 == 0:
            print(f"    {i}/{len(names)}")
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zips", nargs="+", required=True)
    ap.add_argument("--tiles", type=Path, required=True)
    ap.add_argument("--crib", type=Path, default=None)
    ap.add_argument("--family", choices=("dinov3", "dinov2"), default="dinov2")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--img-sizes", type=int, nargs="+", default=[448])
    ap.add_argument("--heads", nargs="+", default=None, help="subset of head names")
    ap.add_argument("--smooth", type=int, nargs="+", default=[1])
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("compare_out"))
    args = ap.parse_args()

    from sklearn.model_selection import GroupShuffleSplit
    import vit_features as vf

    polys = tp.read_cvat_zips(args.zips)
    tiles = tp.index_tiles(args.tiles)
    crib = tp.read_marker_boxes(args.crib) if args.crib and args.crib.exists() else {}
    names = [n for n in sorted(polys) if n in tiles]
    if args.limit:
        names = names[: args.limit]
    print(f"annotated tiles {len(names)}, frames {len({tp.source_of(n) for n in names})}")

    model, meta = vf.load_backbone(family=args.family, repo=args.repo, weights=args.weights)
    heads = build_heads(args.seed)
    if args.heads:
        heads = {k: v for k, v in heads.items() if k in args.heads}

    rows = []
    for img_size in args.img_sizes:
        print(f"\nextracting features at {img_size}px")
        recs = extract(names, tiles, polys, crib, model, meta, img_size)
        G = np.array([r["src"] for r in recs])
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                        random_state=args.seed).split(np.zeros(len(recs)), groups=G))
        print(f"  grid {recs[0]['grid']}, train {len(set(G[tr]))} frames, test {len(set(G[te]))} frames")

        def pack(idx):
            X, Y = [], []
            for j in idx:
                r = recs[j]
                y = tp.label_patches(r["cov"], 0.5, 0.1)
                k = y >= 0
                X.append(r["feat"][k]); Y.append(y[k])
            return np.concatenate(X), np.concatenate(Y)

        Xtr, Ytr = pack(tr)
        n_gt = sum(len(recs[j]["inst"]) for j in te)

        for hname, head in heads.items():
            import copy
            clf = copy.deepcopy(head)
            clf.fit(Xtr, Ytr)
            ys, ps, tl = [], [], []
            for j in te:
                r = recs[j]; gh, gw = r["grid"]
                prob = clf.predict_proba(r["feat"])[:, 1]
                y = tp.label_patches(r["cov"], 0.5, 0.1); k = y >= 0
                ys.append(y[k]); ps.append(prob[k])
                tl.append({"prob": prob.reshape(gh, gw), "gt": r["cov"] >= 0.5,
                           "gt_instances": r["inst"]})
            Y = np.concatenate(ys); P = np.concatenate(ps)
            thr, _ = pe.best_f1_threshold(Y, P)
            pm = pe.patch_metrics(Y, P, thr)
            mm = pe.mask_metrics(tl, thr)
            for sm in args.smooth:
                im = pe.instance_map(tl, thr, min_size=8, smooth=sm)
                rows.append({
                    "img_size": img_size, "head": hname, "smooth": sm, "thresh": round(thr, 2),
                    "ap": pm["ap"], "auc": pm["auc"], "precision": pm["precision"],
                    "recall": pm["recall"], "f1": pm["f1"],
                    "iou": mm["mean_iou"], "dice": mm["mean_dice"],
                    "mAP": im["mAP_50_95"], "ap50": im["AP_50"],
                    "n_pred": im["n_pred_instances"], "n_gt": n_gt,
                })
                print(f"  {hname:<14} sm={sm}  AP={pm['ap']:.3f} F1={pm['f1']:.3f} "
                      f"IoU={mm['mean_iou']:.3f} mAP={im['mAP_50_95']:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(rows, indent=2))

    hdr = (f"{'size':>5}{'head':>15}{'sm':>4}{'thr':>6}"
           f"{'AP':>7}{'AUC':>7}{'P':>7}{'R':>7}{'F1':>7}"
           f"{'IoU':>7}{'Dice':>7}{'mAP':>7}{'AP50':>7}{'pred/gt':>10}")
    print("\n" + "=" * len(hdr)); print(hdr); print("=" * len(hdr))
    for r in sorted(rows, key=lambda r: -r["ap"]):
        print(f"{r['img_size']:>5}{r['head']:>15}{r['smooth']:>4}{r['thresh']:>6.2f}"
              f"{r['ap']:>7.3f}{r['auc']:>7.3f}{r['precision']:>7.3f}{r['recall']:>7.3f}"
              f"{r['f1']:>7.3f}{r['iou']:>7.3f}{r['dice']:>7.3f}{r['mAP']:>7.3f}"
              f"{r['ap50']:>7.3f}{str(r['n_pred'])+'/'+str(r['n_gt']):>10}")
    print(f"\nwrote {args.out/'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
