"""Turn a run's metrics.json into a Confluence-ready page body.

Retyping numbers from a notebook into a wiki page is where results quietly go
wrong. A digit slips, a run is described with the config of a different run, or
a page says 0.77 long after the code that produced it changed. This reads the
JSON a pipeline already wrote and emits the page, so the figures on Confluence
and the figures on disk cannot drift apart.

Every pipeline emits the same top-level shape, which is what makes their pages
comparable:

    {"pipeline": str,
     "config":   {...},            what was run
     "data":     {...},            what it was run on
     "mean"/"std"/"folds"          cross-validated results, or
     "patch"/"mask"/"instance"     single held-out split results
     "run":      {"date", "commit"}}

Usage:
    python -m metrics.render run/metrics.json                 # HTML for Confluence
    python -m metrics.render run/metrics.json --format md     # Markdown
    python -m metrics.render a/metrics.json b/metrics.json    # comparison table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PCT = {"ap", "auc", "f1", "precision", "recall", "iou", "dice",
       "mean_iou", "mean_dice", "mAP_50_95", "AP_50", "AP_75", "positive_rate"}

LABEL = {
    "ap": "AP (mAP, one class)", "auc": "ROC AUC", "f1": "F1",
    "precision": "Precision", "recall": "Recall",
    "mean_iou": "Mean IoU", "mean_dice": "Mean Dice",
    "mAP_50_95": "mAP@[.50:.95]", "AP_50": "AP@0.50", "AP_75": "AP@0.75",
}


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def _rows(d: dict) -> list[tuple[str, str]]:
    return [(k.replace("_", " "), fmt(v)) for k, v in d.items()]


def _table(headers: list[str], rows: list[list[str]], html: bool) -> str:
    if html:
        h = "".join(f"<th>{c}</th>" for c in headers)
        b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render_one(d: dict, html: bool = True) -> str:
    p, nl = ("<p>{}</p>", "") if html else ("{}", "\n")
    out: list[str] = []
    name = d.get("pipeline", "pipeline")
    run = d.get("run", {})
    out.append((f"<h2>{name}</h2>" if html else f"## {name}"))
    stamp = f"Run {run.get('date', 'unknown date')} at commit <code>{run.get('commit', 'unknown')}</code>."
    if not html:
        stamp = stamp.replace("<code>", "`").replace("</code>", "`")
    out.append(p.format(stamp))

    if "config" in d:
        out.append("<h3>Configuration</h3>" if html else "### Configuration")
        out.append(_table(["Setting", "Value"], [list(r) for r in _rows(d["config"])], html))
    if "data" in d:
        out.append("<h3>Data</h3>" if html else "### Data")
        out.append(_table(["Measure", "Value"], [list(r) for r in _rows(d["data"])], html))

    # cross-validated shape
    if "mean" in d and "std" in d:
        out.append("<h3>Results, cross validated</h3>" if html else "### Results, cross validated")
        rows = [[LABEL.get(k, k), f"{d['mean'][k]:.3f}", f"{d['std'][k]:.3f}"] for k in d["mean"]]
        out.append(_table(["Metric", "Mean", "Std"], rows, html))
        if d.get("folds"):
            out.append(p.format(f"Across {len(d['folds'])} folds, grouped by source frame."))

    # single-split shape
    for level in ("patch", "mask", "instance"):
        if level in d and isinstance(d[level], dict):
            out.append((f"<h3>{level.title()} level</h3>" if html else f"### {level.title()} level"))
            rows = [[LABEL.get(k, k.replace("_", " ")), fmt(v)]
                    for k, v in d[level].items() if not isinstance(v, (dict, list))]
            out.append(_table(["Metric", "Value"], rows, html))
    return (nl + nl).join(out) if not html else "".join(out)


def render_compare(ds: list[dict], html: bool = True) -> str:
    """One row per run, so pipelines can be read side by side."""
    def pick(d):
        m = d.get("mean", {})
        inst = d.get("instance", {})
        mask = d.get("mask", {})
        patch = d.get("patch", {})
        return {
            "pipeline": d.get("pipeline", "?"),
            "backbone": d.get("config", {}).get("backbone", "?"),
            "img_size": d.get("config", {}).get("img_size", "?"),
            "AP": m.get("ap", patch.get("ap")),
            "F1": m.get("f1", patch.get("f1")),
            "IoU": mask.get("mean_iou"),
            "mAP": inst.get("mAP_50_95"),
            "date": d.get("run", {}).get("date", "?"),
        }
    cols = ["pipeline", "backbone", "img_size", "AP", "F1", "IoU", "mAP", "date"]
    rows = [[fmt(pick(d).get(c)) if pick(d).get(c) is not None else "n/a" for c in cols] for d in ds]
    head = ("<h2>Pipeline comparison</h2>" if html else "## Pipeline comparison")
    note = ("Blank cells mean the pipeline does not report that level yet, not that it scored zero.")
    body = _table(cols, rows, html)
    return head + (f"<p>{note}</p>" if html else f"\n\n{note}\n\n") + body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics", nargs="+", type=Path, help="one or more metrics.json")
    ap.add_argument("--format", choices=("html", "md"), default="html")
    ap.add_argument("--out", type=Path, default=None, help="write to a file instead of stdout")
    args = ap.parse_args()

    ds = [json.loads(p.read_text()) for p in args.metrics]
    html = args.format == "html"
    body = render_compare(ds, html) if len(ds) > 1 else render_one(ds[0], html)
    if args.out:
        args.out.write_text(body + "\n"); print(f"wrote {args.out}")
    else:
        print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
