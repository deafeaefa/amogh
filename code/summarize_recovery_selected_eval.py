"""Audit the one-time selected-checkpoint VQA holdout and full POPE evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from recovery_utils import BASE_REVISION
from summarize_recovery_checkpoint_sweep import load_vqa_rows, mean_vqa


VQA_HOLDOUT_START = 1000
VQA_HOLDOUT_COUNT = 4000
POPE_COUNT = 9000
PRESERVATION_MARGIN = 0.015
PARSE_MARGIN = 0.005


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pope_rows(path: Path, expected_count: int) -> list[dict]:
    rows = []
    seen = set()
    with open(path) as f:
        for line_number, line in enumerate(f, 1):
            row = json.loads(line)
            uid = row.get("uid")
            if not isinstance(uid, str) or not uid:
                raise ValueError(f"missing POPE uid in {path}:{line_number}")
            if uid in seen:
                raise ValueError(f"duplicate POPE uid {uid!r} in {path}")
            if not isinstance(row.get("correct"), bool):
                raise ValueError(f"invalid POPE correctness in {path}:{line_number}")
            if not isinstance(row.get("parse_fail"), bool):
                raise ValueError(f"invalid POPE parse flag in {path}:{line_number}")
            seen.add(uid)
            rows.append({
                "uid": uid,
                "score": float(row["correct"]),
                "parse_fail": float(row["parse_fail"]),
            })
    if len(rows) != expected_count:
        raise ValueError(f"expected exactly {expected_count} POPE rows in {path}, found {len(rows)}")
    return rows


def require_same_uids(reference: list[dict], candidate: list[dict], label: str) -> None:
    reference_uids = [row["uid"] for row in reference]
    candidate_uids = [row["uid"] for row in candidate]
    if candidate_uids != reference_uids:
        raise ValueError(f"{label} candidate UIDs/order do not match the frozen baseline slice")


def mean_field(rows: list[dict], field: str) -> float:
    if not rows:
        raise ValueError(f"cannot average empty field {field!r}")
    return sum(float(row[field]) for row in rows) / len(rows)


def paired_image_delta(
    reference: list[dict],
    candidate: list[dict],
    image_by_uid: dict[str, str],
    field: str = "score",
    resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Paired candidate-minus-reference delta with image-clustered bootstrap."""
    require_same_uids(reference, candidate, field)
    clusters = defaultdict(list)
    for base_row, candidate_row in zip(reference, candidate):
        uid = base_row["uid"]
        if uid not in image_by_uid:
            raise ValueError(f"missing image mapping for UID {uid!r}")
        clusters[image_by_uid[uid]].append(
            float(candidate_row[field]) - float(base_row[field])
        )
    cluster_sums = np.asarray([sum(values) for values in clusters.values()], dtype=np.float64)
    cluster_counts = np.asarray([len(values) for values in clusters.values()], dtype=np.int64)
    observed = float(cluster_sums.sum() / cluster_counts.sum())
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples, dtype=np.float64)
    cluster_count = len(cluster_sums)
    for start in range(0, resamples, 512):
        end = min(start + 512, resamples)
        chosen = rng.integers(0, cluster_count, size=(end - start, cluster_count))
        bootstrap[start:end] = (
            cluster_sums[chosen].sum(axis=1) / cluster_counts[chosen].sum(axis=1)
        )
    return {
        "observed": observed,
        "ci95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "n_examples": len(reference),
        "n_images": len(clusters),
        "resamples": resamples,
        "seed": seed,
    }


def vqa_image_map(data_dir: Path) -> dict[str, str]:
    rows = load_json(data_dir / "subsets" / "vqa_val_5k.json")
    mapping = {f"vqa:{row['question_id']}": str(row["image_id"]) for row in rows}
    if len(rows) != 5000 or len(mapping) != 5000:
        raise ValueError("frozen VQA manifest must contain 5000 unique question IDs")
    return mapping


def pope_image_map(data_dir: Path) -> dict[str, str]:
    mapping = {}
    for variant in ("random", "popular", "adversarial"):
        path = data_dir / "pope" / f"coco_pope_{variant}.json"
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                uid = f"pope_{variant}:{row['question_id']}"
                if uid in mapping:
                    raise ValueError(f"duplicate frozen POPE UID {uid!r}")
                mapping[uid] = row["image"]
    if len(mapping) != POPE_COUNT:
        raise ValueError(f"frozen POPE inputs must contain {POPE_COUNT} unique rows")
    return mapping


def validate_candidate_metrics(path: Path, task: str, count: int, start: int, limit: int) -> None:
    metrics = load_json(path)
    expected = {
        "base_revision": BASE_REVISION,
        "task": task,
        "n": count,
        "start": start,
        "requested_limit": limit,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(f"unexpected {key} in {path}: {metrics.get(key)!r} != {value!r}")


def main() -> None:
    root_runs = Path(os.environ["GCQ_RUNS"])
    data_dir = Path(os.environ["GCQ_DATA"])
    pilot = root_runs / "recovery_pilot"
    checkpoint_summary_path = pilot / "checkpoint_sweep_summary.json"
    checkpoint_summary = load_json(checkpoint_summary_path)
    if checkpoint_summary.get("selection_succeeded") is not True:
        raise ValueError("checkpoint selection did not succeed")
    selected = checkpoint_summary["selected"]
    step = int(selected["step"])
    if step != 300 or selected["scores"].get("eligible") is not True:
        raise ValueError("holdout is hard-locked to eligible checkpoint step 300")
    adapter = Path(selected["adapter_dir"])
    expected_adapter = (
        pilot / "adapters" / "gcq425_lora_cwce_g5_s0" / "checkpoint-000300"
    )
    if adapter != expected_adapter:
        raise ValueError(f"selected adapter is not the canonical step-300 path: {adapter}")
    tag = f"gcq425_cwce_step{step}_selected"

    output_dir = pilot / "selected_eval"
    launch = load_json(output_dir / "selection_launch_manifest.json")
    if launch.get("selected_step") != step or launch.get("adapter_dir") != str(adapter):
        raise ValueError("holdout launch does not match the frozen checkpoint selection")
    base_dir = pilot / "eval" / "gcq425_lora_ce_s0"
    promote = root_runs / "promote_gcq_b4.25.json"
    hash_paths = {
        "checkpoint_summary": checkpoint_summary_path,
        "adapter": adapter / "adapter_model.safetensors",
        "adapter_manifest": adapter / "gcq_recovery_manifest.json",
        "promotion": promote,
        "baseline_vqa": base_dir / "gcq425_untrained_recoverypilot_vqa5k.vqa.jsonl",
        "baseline_pope": base_dir / "gcq425_untrained_recoverypilot_pope.pope.jsonl",
        "frozen_vqa": data_dir / "subsets" / "vqa_val_5k.json",
        "pope_random": data_dir / "pope" / "coco_pope_random.json",
        "pope_popular": data_dir / "pope" / "coco_pope_popular.json",
        "pope_adversarial": data_dir / "pope" / "coco_pope_adversarial.json",
    }
    if launch.get("base_revision") != BASE_REVISION:
        raise ValueError("holdout launch base revision changed")
    if launch.get("vqa_slice") != {
        "start": VQA_HOLDOUT_START, "count": VQA_HOLDOUT_COUNT
    } or launch.get("pope_count") != POPE_COUNT:
        raise ValueError("holdout launch slice/count protocol changed")
    for key, path in hash_paths.items():
        expected_hash = launch.get("hashes", {}).get(key)
        if not expected_hash or expected_hash != sha256(path):
            raise ValueError(f"{key} changed after the holdout launch: {path}")

    base_vqa = load_vqa_rows(
        base_dir / "gcq425_untrained_recoverypilot_vqa5k.vqa.jsonl", expected_count=5000
    )
    selected_vqa = load_vqa_rows(
        pilot / "checkpoint_sweep" / f"step{step}"
        / f"gcq425_cwce_step{step}_vqa_select1k.vqa.jsonl",
        expected_count=VQA_HOLDOUT_START,
    )
    require_same_uids(base_vqa[:VQA_HOLDOUT_START], selected_vqa, "VQA selection")
    candidate_vqa = load_vqa_rows(
        output_dir / f"{tag}_vqa_holdout4k.vqa.jsonl",
        expected_count=VQA_HOLDOUT_COUNT,
    )
    base_vqa_holdout = base_vqa[VQA_HOLDOUT_START:]
    require_same_uids(base_vqa_holdout, candidate_vqa, "VQA holdout")
    validate_candidate_metrics(
        output_dir / f"{tag}_vqa_holdout4k.vqa.metrics.json",
        task="vqa", count=VQA_HOLDOUT_COUNT,
        start=VQA_HOLDOUT_START, limit=VQA_HOLDOUT_COUNT,
    )

    base_pope = load_pope_rows(
        base_dir / "gcq425_untrained_recoverypilot_pope.pope.jsonl", expected_count=POPE_COUNT
    )
    candidate_pope = load_pope_rows(
        output_dir / f"{tag}_pope_full.pope.jsonl", expected_count=POPE_COUNT
    )
    require_same_uids(base_pope, candidate_pope, "POPE")
    validate_candidate_metrics(
        output_dir / f"{tag}_pope_full.pope.metrics.json",
        task="pope", count=POPE_COUNT, start=0, limit=0,
    )

    vqa_pair = paired_image_delta(
        base_vqa_holdout, candidate_vqa, vqa_image_map(data_dir),
        resamples=10_000, seed=20260840,
    )
    pope_pair = paired_image_delta(
        base_pope, candidate_pope, pope_image_map(data_dir),
        resamples=10_000, seed=20260841,
    )
    pope_parse_pair = paired_image_delta(
        base_pope, candidate_pope, pope_image_map(data_dir), field="parse_fail",
        resamples=10_000, seed=20260842,
    )

    base_vqa_holdout_score = mean_vqa(base_vqa_holdout)
    candidate_vqa_holdout_score = mean_vqa(candidate_vqa)
    base_pope_score = mean_field(base_pope, "score")
    candidate_pope_score = mean_field(candidate_pope, "score")
    base_pope_parse = mean_field(base_pope, "parse_fail")
    candidate_pope_parse = mean_field(candidate_pope, "parse_fail")
    all_candidate_vqa = selected_vqa + candidate_vqa

    gates = {
        "VQA_holdout_within_1.5pt_of_untrained_GCQ": (
            candidate_vqa_holdout_score >= base_vqa_holdout_score - PRESERVATION_MARGIN
        ),
        "POPE_full_within_1.5pt_of_untrained_GCQ": (
            candidate_pope_score >= base_pope_score - PRESERVATION_MARGIN
        ),
        "POPE_parse_fail_within_0.5pt": (
            candidate_pope_parse <= base_pope_parse + PARSE_MARGIN
        ),
    }
    summary = {
        "schema_version": 1,
        "evaluation_role": "one-time general-capability holdout for the development-selected checkpoint",
        "selected_step": step,
        "adapter_dir": str(adapter),
        "base_revision": BASE_REVISION,
        "vqa": {
            "selection_1k": {
                "untrained_gcq": mean_vqa(base_vqa[:VQA_HOLDOUT_START]),
                "selected_checkpoint": mean_vqa(selected_vqa),
            },
            "holdout_4k": {
                "untrained_gcq": base_vqa_holdout_score,
                "selected_checkpoint": candidate_vqa_holdout_score,
                "paired_delta": vqa_pair,
                "ci95_entirely_inside_1.5pt_noninferiority_margin": (
                    vqa_pair["ci95"][0] >= -PRESERVATION_MARGIN
                ),
            },
            "combined_5k_secondary_selection_biased": {
                "untrained_gcq": mean_vqa(base_vqa),
                "selected_checkpoint": mean_vqa(all_candidate_vqa),
            },
        },
        "pope_full_9k": {
            "untrained_gcq": base_pope_score,
            "selected_checkpoint": candidate_pope_score,
            "paired_delta": pope_pair,
            "ci95_entirely_inside_1.5pt_noninferiority_margin": (
                pope_pair["ci95"][0] >= -PRESERVATION_MARGIN
            ),
            "untrained_gcq_parse_fail": base_pope_parse,
            "selected_checkpoint_parse_fail": candidate_pope_parse,
            "paired_parse_fail_delta": pope_parse_pair,
        },
        "gates": gates,
        "holdout_preservation_pass": all(gates.values()),
        "recovery_pipeline_pass": bool(selected["scores"]["eligible"] and all(gates.values())),
        "next_if_pass": "verify a BF16 adapter export, then run seeds 1-2 and untouched grounding confirmation",
        "next_if_fail": "retain the frozen gate and improve the general-replay/training recipe",
    }
    output = pilot / "selected_eval_summary.json"
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
