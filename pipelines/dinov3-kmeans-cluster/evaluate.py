"""Score the clustering pipeline against the CVAT polygons.

The difficulty is that clustering has no inherent notion of "weed". A human
picks the cluster. So scoring it means deciding how that pick happens, and the
pick IS the method. Two modes, both reported, because the gap between them is
the interesting number.

  oracle    For every tile, choose the cluster that best overlaps the
            annotation. This is the ceiling: what the pipeline would score if a
            person never once chose wrongly. It uses the ground truth to make
            the choice, so it is not an achievable result, it is an upper bound.

  centroid  Derive one reference centroid from the training frames, then on
            each test frame pick the cluster whose centroid is closest to it.
            No ground truth is consulted at test time, so this is a real,
            automatable result. It is what the notebook's own stage 2 does.

The distance between oracle and centroid is the cost of the cluster id not
being stable across tiles.

Scores for AP are the cosine similarity between each patch and the chosen
cluster centroid, which gives the ranking a hard assignment cannot.

The split matches the probe exactly, GroupShuffleSplit by source frame with
test_size 0.3 and seed 0, so the two pipelines are directly comparable.

Usage:
    python evaluate.py --zips '.../batch_*.zip' --tiles '.../processed' \
        --crib '.../tile_markers.csv' --family dinov2 -k 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "pipelines" / "dinov3-linear-probe") not in sys.path:
    sys.path.insert(0, str(_ROOT / "pipelines" / "dinov3-linear-probe"))

import clustering as cl                      # noqa: E402
import train_probe as tp                     # noqa: E402  shared CVAT + label helpers
from metrics import segmentation as pe       # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = float((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zips", nargs="+", required=True)
    ap.add_argument("--tiles", type=Path, required=True)
    ap.add_argument("--crib", type=Path, default=None)
    ap.add_argument("--family", choices=("dinov3", "dinov2"), default="dinov2")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("-k", "--clusters", type=int, default=12)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--min-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    from sklearn.model_selection import GroupShuffleSplit
    from backbones import dino as vf

    polys = tp.read_cvat_zips(args.zips)
    tiles = tp.index_tiles(args.tiles)
    crib = tp.read_marker_boxes(args.crib) if args.crib and args.crib.exists() else {}
    names = [n for n in sorted(polys) if n in tiles]
    if args.limit:
        names = names[: args.limit]
    print(f"annotated tiles {len(names)}, frames {len({tp.source_of(n) for n in names})}")

    model, meta = vf.load_backbone(family=args.family, repo=args.repo, weights=args.weights)

    recs = []
    for i, n in enumerate(names, 1):
        feats, (gh, gw) = vf.patch_features(model, str(tiles[n]), meta, img_size=args.img_size)
        feats = np.asarray(feats, dtype=np.float32)
        feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        labels, cents = cl.cluster_patches(feats, k=args.clusters, seed=args.seed)
        cents = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-8)

        with Image.open(tiles[n]) as im:
            size = im.size
        side = gh * meta["patch"]
        cov = tp.patch_coverage(tp.tile_mask(size, polys[n], crib.get(n)), side, meta["patch"])
        gt = cov >= 0.5
        inst = []
        for pts in polys[n]:
            g = tp.patch_coverage(tp.tile_mask(size, [pts], crib.get(n)), side, meta["patch"]) >= 0.5
            if g.any():
                inst.append(g)

        # the cluster a perfect human would pick on this tile
        best = max(range(args.clusters),
                   key=lambda c: iou((labels == c).reshape(gh, gw), gt))
        recs.append(dict(name=n, feats=feats, labels=labels, cents=cents, grid=(gh, gw),
                         cov=cov, gt=gt, inst=inst, oracle=best, src=tp.source_of(n)))
        if i % 25 == 0 or i == len(names):
            print(f"  {i}/{len(names)}")

    G = np.array([r["src"] for r in recs])
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                    random_state=args.seed).split(np.zeros(len(recs)), groups=G))
    print(f"train {len(set(G[tr]))} frames, test {len(set(G[te]))} frames / {len(te)} tiles")

    # one reference centroid, averaged over the oracle picks on the TRAIN frames only
    ref = np.mean([recs[j]["cents"][recs[j]["oracle"]] for j in tr], axis=0)
    ref /= (np.linalg.norm(ref) + 1e-8)

    results = {}
    for mode in ("oracle", "centroid"):
        ys, ps, tl, picks = [], [], [], []
        for j in te:
            r = recs[j]; gh, gw = r["grid"]
            chosen = r["oracle"] if mode == "oracle" else int(np.argmax(r["cents"] @ ref))
            picks.append(chosen)
            score = (r["feats"] @ r["cents"][chosen]).reshape(gh, gw)   # cosine to centroid
            score = (score - score.min()) / (np.ptp(score) + 1e-8)        # to 0..1 for AP
            y = tp.label_patches(r["cov"], 0.5, 0.1); k = y >= 0
            ys.append(y[k]); ps.append(score.ravel()[k])
            tl.append({"prob": score, "gt": r["gt"], "gt_instances": r["inst"]})
        Y = np.concatenate(ys); P = np.concatenate(ps)
        thr, _ = pe.best_f1_threshold(Y, P)
        results[mode] = {
            "patch": pe.patch_metrics(Y, P, thr),
            "mask": pe.mask_metrics(tl, thr),
            "instance": pe.instance_map(tl, thr, min_size=args.min_size, smooth=args.smooth),
            "cluster_ids_chosen": sorted(set(picks)),
        }
        print(f"\n=== {mode} ===")
        pe.report(Y, P, tl, thresh=thr, min_size=args.min_size, smooth=args.smooth)

    agree = sum(1 for j, c in zip(te, [int(np.argmax(recs[j]['cents'] @ ref)) for j in te])
                if c == recs[j]["oracle"])
    print(f"\ncentroid matching picked the oracle cluster on {agree} of {len(te)} test tiles")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": "dinov3-kmeans-cluster",
        "config": {
            "backbone": args.family, "arch": meta.get("arch"),
            "img_size": args.img_size, "patch": meta["patch"],
            "grid": f"{args.img_size // meta['patch']}x{args.img_size // meta['patch']}",
            "head": f"KMeans k={args.clusters}, no trained parameters",
            "split": f"GroupShuffleSplit by source frame, {int(args.test_size*100)} percent held out",
        },
        "data": {
            "annotated_tiles": len(names), "frames_total": len(set(G)),
            "frames_train": len(set(G[tr])), "frames_test": len(set(G[te])),
            "tiles_test": len(te),
            "gt_instances_test": sum(len(recs[j]["inst"]) for j in te),
            "oracle_agreement": f"{agree}/{len(te)}",
        },
        "run": {"date": time.strftime("%Y-%m-%d")},
        **{m: {k: v for k, v in d.items()} for m, d in results.items()},
    }
    (args.out / "metrics.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {args.out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
