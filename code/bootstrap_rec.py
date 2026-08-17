"""Paired image-clustered bootstrap intervals for REC/GIoU/precise-IoU gains."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from recovery_utils import precise_iou_score


def load_jsonl(path):
    rows = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            rows[row["uid"]] = row
    return rows


def per_example(row, metric):
    if metric == "rec":
        return float(row["iou"] >= 0.5)
    if metric == "giou":
        return float(row["giou"])
    if metric == "precise_iou":
        return precise_iou_score(float(row["iou"]))
    raise ValueError(metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--subset", required=True, help="subset JSON used to recover image IDs for historical logs")
    ap.add_argument("--metric", choices=("rec", "giou", "precise_iou"), required=True)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    baseline = load_jsonl(args.baseline)
    method = load_jsonl(args.method)
    if set(baseline) != set(method):
        raise SystemExit("paired logs do not contain identical UIDs")
    with open(args.subset) as f:
        image_by_uid = {row["uid"]: row["image_id"] for row in json.load(f)}
    if set(baseline) - set(image_by_uid):
        raise SystemExit("subset is missing UIDs from result logs")

    clusters = defaultdict(list)
    for uid in sorted(baseline):
        delta = per_example(method[uid], args.metric) - per_example(baseline[uid], args.metric)
        clusters[image_by_uid[uid]].append(delta)
    cluster_values = list(clusters.values())
    rng = np.random.default_rng(args.seed)
    boot = np.empty(args.resamples, dtype=np.float64)
    for sample in range(args.resamples):
        chosen = rng.integers(0, len(cluster_values), size=len(cluster_values))
        values = [value for index in chosen for value in cluster_values[index]]
        boot[sample] = np.mean(values)
    observed = np.mean([value for values in cluster_values for value in values])
    result = {
        "metric": args.metric,
        "method_minus_baseline": float(observed),
        "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "n_examples": len(baseline),
        "n_images": len(cluster_values),
        "resamples": args.resamples,
        "seed": args.seed,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
