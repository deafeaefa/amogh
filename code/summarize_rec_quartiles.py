"""Summarize BF16/W4/GCQ by relative target-area quartile with clustered CIs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from eval_rec import area_quartiles
from recovery_utils import precise_iou_score


def load_jsonl(path):
    with open(path) as f:
        return {row["uid"]: row for row in map(json.loads, f)}


def cluster_ci(deltas, image_ids, resamples, rng):
    clusters = defaultdict(list)
    for value, image_id in zip(deltas, image_ids):
        clusters[image_id].append(value)
    values = list(clusters.values())
    boot = np.empty(resamples)
    for sample in range(resamples):
        chosen = rng.integers(0, len(values), len(values))
        boot[sample] = np.mean([x for i in chosen for x in values[i]])
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def score(row, metric):
    if metric == "rec":
        return float(row["iou"] >= 0.5)
    if metric == "giou":
        return float(row["giou"])
    if metric == "precise_iou":
        return precise_iou_score(float(row["iou"]))
    raise ValueError(metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--w4", required=True)
    ap.add_argument("--gcq", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(args.subset) as f:
        records = json.load(f)
    by_uid = {row["uid"]: row for row in records}
    quartile_by_index = area_quartiles(records)
    quartile_by_uid = {records[index]["uid"]: quartile for index, quartile in quartile_by_index.items()}
    runs = {"bf16": load_jsonl(args.bf16), "w4": load_jsonl(args.w4), "gcq": load_jsonl(args.gcq)}
    expected = set(by_uid)
    if any(set(run) != expected for run in runs.values()):
        raise SystemExit("result logs and subset do not have identical UIDs")

    rng = np.random.default_rng(args.seed)
    output = {"n": len(records), "resamples": args.resamples, "quartiles": {}}
    for quartile in range(1, 5):
        uids = [uid for uid in expected if quartile_by_uid[uid] == quartile]
        images = [by_uid[uid]["image_id"] for uid in uids]
        result = {"n_expressions": len(uids), "n_images": len(set(images)), "metrics": {}}
        for metric in ("rec", "giou", "precise_iou"):
            values = {name: np.asarray([score(run[uid], metric) for uid in uids]) for name, run in runs.items()}
            w4_drop = values["w4"] - values["bf16"]
            gcq_gain = values["gcq"] - values["w4"]
            result["metrics"][metric] = {
                **{name: float(value.mean()) for name, value in values.items()},
                "w4_minus_bf16": float(w4_drop.mean()),
                "w4_minus_bf16_ci95": cluster_ci(w4_drop, images, args.resamples, rng),
                "gcq_minus_w4": float(gcq_gain.mean()),
                "gcq_minus_w4_ci95": cluster_ci(gcq_gain, images, args.resamples, rng),
            }
        output["quartiles"][f"q{quartile}"] = result
    print(json.dumps(output, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
