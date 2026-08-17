"""Select a recovery checkpoint using predeclared grounding/VQA development gates."""
from __future__ import annotations

import json
import os
from pathlib import Path

from summarize_recovery_pilot import (
    group_scores,
    load_json,
    load_jsonl_unique,
    paired_cluster_contrast,
)


STEPS = (100, 200, 300, 400)


def load_vqa_rows(path: Path, expected_count: int) -> list[dict]:
    """Load a complete VQA JSONL and reject missing or duplicate frozen UIDs."""
    rows = []
    seen = set()
    with open(path) as f:
        for line_number, line in enumerate(f, 1):
            row = json.loads(line)
            uid = row.get("uid")
            if not isinstance(uid, str) or not uid:
                raise ValueError(f"missing VQA uid in {path}:{line_number}")
            if uid in seen:
                raise ValueError(f"duplicate VQA uid {uid!r} in {path}")
            try:
                score = float(row["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid VQA score in {path}:{line_number}") from exc
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"VQA score outside [0,1] in {path}:{line_number}")
            seen.add(uid)
            rows.append({"uid": uid, "score": score})
    if len(rows) != expected_count:
        raise ValueError(f"expected exactly {expected_count} VQA rows in {path}, found {len(rows)}")
    return rows


def mean_vqa(rows: list[dict]) -> float:
    if not rows:
        raise ValueError("cannot average an empty VQA result")
    return sum(row["score"] for row in rows) / len(rows)


def main() -> None:
    root_runs = Path(os.environ["GCQ_RUNS"])
    pilot = root_runs / "recovery_pilot"
    eval_root = pilot / "eval"
    sweep_root = pilot / "checkpoint_sweep"

    base_dir = eval_root / "gcq425_lora_ce_s0"
    base_tag = "gcq425_untrained_recoverydev"
    base_metrics = load_json(base_dir / f"{base_tag}.rec.metrics.json")
    base_primary = group_scores(base_metrics, "rec")
    base_log = load_jsonl_unique(base_dir / f"{base_tag}.rec.jsonl")
    base_vqa_path = base_dir / "gcq425_untrained_recoverypilot_vqa5k.vqa.jsonl"
    base_vqa_rows = load_vqa_rows(base_vqa_path, expected_count=5000)
    base_vqa_select_rows = base_vqa_rows[:1000]
    base_vqa_select_uids = [row["uid"] for row in base_vqa_select_rows]
    base_vqa_select = mean_vqa(base_vqa_select_rows)

    w4_dir = eval_root / "w4rtn_lora_cwce_g5_s0"
    w4_tag = "w4rtn_lora_cwce_g5_s0_recoverydev"
    w4_primary = group_scores(load_json(w4_dir / f"{w4_tag}.rec.metrics.json"), "rec")

    candidates = {}
    candidate_logs = {}
    for step in STEPS:
        tag = f"gcq425_cwce_step{step}"
        directory = sweep_root / f"step{step}"
        metrics = load_json(directory / f"{tag}_recoverydev.rec.metrics.json")
        primary = group_scores(metrics, "rec")
        candidate_logs[step] = load_jsonl_unique(directory / f"{tag}_recoverydev.rec.jsonl")
        candidate_vqa_rows = load_vqa_rows(
            directory / f"{tag}_vqa_select1k.vqa.jsonl", expected_count=1000
        )
        candidate_vqa_uids = [row["uid"] for row in candidate_vqa_rows]
        if candidate_vqa_uids != base_vqa_select_uids:
            raise ValueError(
                f"checkpoint {step} VQA UIDs/order do not match the frozen first-1k slice"
            )
        vqa = mean_vqa(candidate_vqa_rows)
        gates = {
            "primary_REC_gain_at_least_1pt": primary["rec"] - base_primary["rec"] >= 0.010,
            "primary_GIoU_above_untrained_GCQ": primary["giou"] > base_primary["giou"],
            "primary_precise_IoU_above_untrained_GCQ": (
                primary["precise_iou"] > base_primary["precise_iou"]
            ),
            "primary_REC_above_W4_weighted_control": primary["rec"] > w4_primary["rec"],
            "primary_parse_fail_within_0.5pt": (
                primary["parse_fail"] <= base_primary["parse_fail"] + 0.005
            ),
            "VQA_select1k_within_1pt": vqa >= base_vqa_select - 0.010,
        }
        candidates[step] = {
            "primary_rec": primary,
            "vqa_select1k": vqa,
            "vqa_select1k_delta": vqa - base_vqa_select,
            "gates": gates,
            "eligible": all(gates.values()),
        }

    eligible = [step for step in STEPS if candidates[step]["eligible"]]
    selected_step = max(
        eligible,
        key=lambda step: (
            candidates[step]["primary_rec"]["rec"],
            candidates[step]["primary_rec"]["precise_iou"],
            -step,
        ),
        default=None,
    )
    selection = None
    if selected_step is not None:
        comparisons = {
            metric_name: paired_cluster_contrast(
                {"base": base_log, "candidate": candidate_logs[selected_step]},
                {"candidate": 1.0, "base": -1.0}, metric_name,
                task="rec", resamples=10_000, seed=20260830 + metric_index,
            )
            for metric_index, metric_name in enumerate(("rec", "giou", "precise_iou"))
        }
        selection = {
            "step": selected_step,
            "adapter_dir": str(
                pilot / "adapters" / "gcq425_lora_cwce_g5_s0"
                / f"checkpoint-{selected_step:06d}"
            ),
            "scores": candidates[selected_step],
            "paired_vs_untrained_gcq": comparisons,
        }

    summary = {
        "schema_version": 1,
        "selection_role": "development-only checkpoint selection",
        "selection_slice": "first 1000 rows of frozen vqa_val_5k plus recovery_dev_1k",
        "holdout_slice": "remaining 4000 VQA rows; evaluate selected checkpoint once",
        "rule": "among fully gated checkpoints, maximize primary REC, then precise-IoU, then prefer earlier step",
        "untrained_gcq": {"primary_rec": base_primary, "vqa_select1k": base_vqa_select},
        "w4_weighted_control": {"primary_rec": w4_primary},
        "candidates": {str(step): candidates[step] for step in STEPS},
        "selected": selection,
        "selection_succeeded": selection is not None,
    }
    output = pilot / "checkpoint_sweep_summary.json"
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))
    if selection is None:
        raise SystemExit("no checkpoint passed the predeclared recovery/preservation gates")


if __name__ == "__main__":
    main()
