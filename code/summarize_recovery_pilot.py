"""Aggregate the corrected recovery pilot and apply its screening gates."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from recovery_utils import BASE_REVISION, precise_iou_score


ARMS = (
    "w4rtn_lora_ce_s0",
    "w4rtn_lora_cwce_g5_s0",
    "gcq425_lora_ce_s0",
    "gcq425_lora_cwce_g5_s0",
)
REC_TAGS = {
    "bf16": "bf16_recoverydev",
    "w4_untrained": "w4rtn_untrained_recoverydev",
    "gcq_untrained": "gcq425_untrained_recoverydev",
    "w4_plain": "w4rtn_lora_ce_s0_recoverydev",
    "w4_weighted": "w4rtn_lora_cwce_g5_s0_recoverydev",
    "gcq_plain": "gcq425_lora_ce_s0_recoverydev",
    "gcq_weighted": "gcq425_lora_cwce_g5_s0_recoverydev",
}
TAG_DIR = {
    "bf16": "w4rtn_lora_ce_s0",
    "w4_untrained": "w4rtn_lora_ce_s0",
    "gcq_untrained": "gcq425_lora_ce_s0",
    "w4_plain": "w4rtn_lora_ce_s0",
    "w4_weighted": "w4rtn_lora_cwce_g5_s0",
    "gcq_plain": "gcq425_lora_ce_s0",
    "gcq_weighted": "gcq425_lora_cwce_g5_s0",
}
METRICS = ("rec", "giou", "precise_iou")


def load_csv_rows(path: Path) -> dict[str, dict]:
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tag = row["tag"]
            if tag in rows:
                raise ValueError(f"duplicate tag {tag!r} in {path}")
            rows[tag] = row
    return rows


def metric(row: dict, name: str) -> float:
    value = row.get(name, "")
    if value == "":
        raise ValueError(f"missing {name} in {row.get('tag')}")
    return float(value)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_jsonl_unique(path: Path) -> dict[str, dict]:
    rows = {}
    with open(path) as f:
        for line_number, line in enumerate(f, 1):
            row = json.loads(line)
            uid = row["uid"]
            if uid in rows:
                raise ValueError(f"duplicate UID {uid!r} in {path}:{line_number}")
            rows[uid] = row
    return rows


def example_metric(row: dict, name: str) -> float:
    if name == "rec":
        return float(float(row["iou"]) >= 0.5)
    if name == "giou":
        return float(row["giou"])
    if name == "precise_iou":
        return precise_iou_score(float(row["iou"]))
    raise ValueError(name)


def paired_cluster_contrast(
    runs: dict[str, dict[str, dict]],
    coefficients: dict[str, float],
    metric_name: str,
    task: str = "rec",
    resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Image-clustered paired bootstrap for an arbitrary linear contrast."""
    names = tuple(coefficients)
    expected_uids = set(runs[names[0]])
    for name in names[1:]:
        if set(runs[name]) != expected_uids:
            raise ValueError(f"paired result logs differ in UIDs: {names[0]} vs {name}")

    clusters = defaultdict(list)
    n_examples = 0
    for uid in sorted(expected_uids):
        reference = runs[names[0]][uid]
        metadata = (reference.get("image_id"), reference.get("task"), reference.get("source"))
        for name in names[1:]:
            row = runs[name][uid]
            if (row.get("image_id"), row.get("task"), row.get("source")) != metadata:
                raise ValueError(f"paired metadata mismatch for UID {uid!r}")
        if reference.get("task") != task:
            continue
        image_id = reference.get("image_id")
        if image_id is None:
            raise ValueError(f"missing image_id for UID {uid!r}")
        value = sum(
            coefficients[name] * example_metric(runs[name][uid], metric_name)
            for name in names
        )
        clusters[image_id].append(value)
        n_examples += 1
    if not clusters:
        raise ValueError(f"no examples for task {task!r}")

    cluster_sums = np.asarray([sum(values) for values in clusters.values()], dtype=np.float64)
    cluster_counts = np.asarray([len(values) for values in clusters.values()], dtype=np.int64)
    observed = float(cluster_sums.sum() / cluster_counts.sum())
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples, dtype=np.float64)
    cluster_count = len(cluster_sums)
    chunk_size = 512
    for start in range(0, resamples, chunk_size):
        end = min(start + chunk_size, resamples)
        chosen = rng.integers(0, cluster_count, size=(end - start, cluster_count))
        bootstrap[start:end] = (
            cluster_sums[chosen].sum(axis=1) / cluster_counts[chosen].sum(axis=1)
        )
    return {
        "observed": observed,
        "ci95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "n_examples": n_examples,
        "n_images": len(clusters),
        "resamples": resamples,
        "seed": seed,
    }


def group_scores(metrics: dict, group: str) -> dict:
    source = metrics if group == "overall" else metrics["by_task"][group]
    return {
        "n": int(source["n"]),
        "rec": float(source["acc_iou_0.5"]),
        "giou": float(source["mean_giou"]),
        "precise_iou": float(source["mean_acc_iou_0.50_0.95"]),
        "parse_fail": float(source["parse_fail"]),
    }


def recovery_fraction(bf16: float, quantized: float, recovered: float) -> float | None:
    gap = bf16 - quantized
    return None if gap <= 0 else (recovered - quantized) / gap


def main() -> None:
    root_runs = Path(os.environ["GCQ_RUNS"])
    pilot = root_runs / "recovery_pilot"
    eval_root = pilot / "eval"

    csv_rows = {}
    for arm in ARMS:
        for tag, row in load_csv_rows(eval_root / arm / "results.csv").items():
            if tag in csv_rows:
                raise ValueError(f"duplicate tag {tag!r} across evaluation directories")
            csv_rows[tag] = row

    rec_metrics = {}
    rec_logs = {}
    for name, tag in REC_TAGS.items():
        directory = eval_root / TAG_DIR[name]
        metrics_path = directory / f"{tag}.rec.metrics.json"
        rec_metrics[name] = load_json(metrics_path)
        if rec_metrics[name].get("base_revision") != BASE_REVISION:
            raise ValueError(f"unmatched base revision in {metrics_path}")
        rec_logs[name] = load_jsonl_unique(directory / f"{tag}.rec.jsonl")

    scores = {
        name: {
            "primary_rec": group_scores(values, "rec"),
            "auxiliary_coco_grounding": group_scores(values, "coco_grounding"),
            "overall": group_scores(values, "overall"),
            "by_source": values["by_source"],
        }
        for name, values in rec_metrics.items()
    }

    comparisons = {}
    comparison_specs = {
        "gcq_weighted_minus_gcq_untrained": {"gcq_weighted": 1.0, "gcq_untrained": -1.0},
        "gcq_weighted_minus_gcq_plain": {"gcq_weighted": 1.0, "gcq_plain": -1.0},
        "gcq_weighted_minus_w4_weighted": {"gcq_weighted": 1.0, "w4_weighted": -1.0},
        "factorial_interaction_D_minus_C_minus_B_plus_A": {
            "gcq_weighted": 1.0, "gcq_plain": -1.0,
            "w4_weighted": -1.0, "w4_plain": 1.0,
        },
    }
    seed = 20260814
    for comparison_index, (comparison, coefficients) in enumerate(comparison_specs.items()):
        comparisons[comparison] = {
            metric_name: paired_cluster_contrast(
                rec_logs, coefficients, metric_name, task="rec",
                resamples=10_000, seed=seed + comparison_index * len(METRICS) + metric_index,
            )
            for metric_index, metric_name in enumerate(METRICS)
        }

    primary = {name: values["primary_rec"] for name, values in scores.items()}
    gap_recovery = {}
    for metric_name in METRICS:
        bf16 = primary["bf16"][metric_name]
        gcq = primary["gcq_untrained"][metric_name]
        recovered = primary["gcq_weighted"][metric_name]
        gap_recovery[metric_name] = {
            "bf16": bf16,
            "untrained_gcq": gcq,
            "trained_gcq_weighted": recovered,
            "bf16_minus_untrained_gcq": bf16 - gcq,
            "trained_minus_untrained_gcq": recovered - gcq,
            "fraction_of_bf16_gap_recovered": recovery_fraction(bf16, gcq, recovered),
        }

    d_vqa = csv_rows["gcq425_lora_cwce_g5_s0_vqa5k"]
    d_pope = csv_rows["gcq425_lora_cwce_g5_s0_pope"]
    base_vqa = csv_rows["gcq425_untrained_recoverypilot_vqa5k"]
    base_pope = csv_rows["gcq425_untrained_recoverypilot_pope"]
    general_capability = {
        "untrained_gcq": {
            "vqa": metric(base_vqa, "acc"), "pope": metric(base_pope, "acc"),
            "pope_parse_fail": metric(base_pope, "parse_fail"),
        },
        "gcq_weighted": {
            "vqa": metric(d_vqa, "acc"), "pope": metric(d_pope, "acc"),
            "pope_parse_fail": metric(d_pope, "parse_fail"),
        },
    }

    d_minus_gcq = comparisons["gcq_weighted_minus_gcq_untrained"]
    gates = {
        "D_primary_REC_at_least_1pt_over_untrained_GCQ": (
            primary["gcq_weighted"]["rec"] - primary["gcq_untrained"]["rec"] >= 0.010
        ),
        "D_primary_GIoU_above_untrained_GCQ": (
            primary["gcq_weighted"]["giou"] > primary["gcq_untrained"]["giou"]
        ),
        "D_primary_precise_IoU_above_untrained_GCQ": (
            primary["gcq_weighted"]["precise_iou"] > primary["gcq_untrained"]["precise_iou"]
        ),
        "D_primary_REC_above_GCQ_plain_LoRA": (
            primary["gcq_weighted"]["rec"] > primary["gcq_plain"]["rec"]
        ),
        "D_primary_REC_above_W4_weighted_LoRA": (
            primary["gcq_weighted"]["rec"] > primary["w4_weighted"]["rec"]
        ),
        "D_primary_parse_fail_within_0.5pt": (
            primary["gcq_weighted"]["parse_fail"]
            <= primary["gcq_untrained"]["parse_fail"] + 0.005
        ),
        "D_primary_REC_gain_CI_excludes_zero": d_minus_gcq["rec"]["ci95"][0] > 0,
        "D_primary_GIoU_gain_CI_excludes_zero": d_minus_gcq["giou"]["ci95"][0] > 0,
        "D_primary_precise_IoU_gain_CI_excludes_zero": (
            d_minus_gcq["precise_iou"]["ci95"][0] > 0
        ),
        "D_VQA_within_1.5pt_of_fresh_GCQ": (
            general_capability["gcq_weighted"]["vqa"]
            >= general_capability["untrained_gcq"]["vqa"] - 0.015
        ),
        "D_POPE_within_1.5pt_of_fresh_GCQ": (
            general_capability["gcq_weighted"]["pope"]
            >= general_capability["untrained_gcq"]["pope"] - 0.015
        ),
        "D_POPE_parse_fail_within_0.5pt": (
            general_capability["gcq_weighted"]["pope_parse_fail"]
            <= general_capability["untrained_gcq"]["pope_parse_fail"] + 0.005
        ),
    }
    summary = {
        "schema_version": 2,
        "evaluation_role": "development screening; not final paper evidence",
        "base_revision": BASE_REVISION,
        "primary_task": "rec (750 referring-expression examples)",
        "auxiliary_task": "coco_grounding (250 category-grounding examples)",
        "scores": scores,
        "gap_recovery": gap_recovery,
        "paired_image_clustered_comparisons": comparisons,
        "general_capability": general_capability,
        "gates": gates,
        "pilot_pass": all(gates.values()),
        "paper_claim_ready": False,
        "next_if_pass": "repeat winning method/strongest control at seeds 1-2 and confirm once on untouched RefCOCO+",
    }
    output = pilot / "pilot_summary.json"
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
