"""Audit and select the preregistered development-round-2 checkpoints.

This is development-only and never reads a confirmation manifest.  It reuses
the parent round's audited loaders, metric recomputation, gates, bootstrap,
and tie-break, with the bound primary-REC parse-failure repair.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import summarize_recovery_vqa_development as parent
from recovery_utils import BASE_MODEL, BASE_REVISION, sha256_file
from summarize_recovery_vqa_development_amended import paired_image_delta_primary_rec


ROUND_ID = "balanced-replay-remaining-checkpoints-development-round-2"
RECIPE_ID = "gcq425_lora_cwce_vqa50_g5_lr5e5_s0"
CANDIDATE_STEPS = (50, 100, 150, 250, 350, 450, 550, 650, 700)
PARENT_VALID_STEPS = (200, 300, 400, 500, 600)
REC_EXAMPLES = 1_000
PRIMARY_REC_EXAMPLES = 750
VQA_EXAMPLES = 5_000
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260850
PARENT_SUMMARY_SHA = "4b923471b3af3244b7874995e871ae38619c61298bdf07851ba93fac5cc35698"
AMENDMENT_SHA = "3261c3cfa13401d5577d831f9d52a500b7a74d186c8314289e51395b4e78d3f6"
AMENDED_WRAPPER_SHA = "63ccce3388abd53ea46a793d6b2fbde923639eb5bb9460acf8bb9c30e673206d"


require = parent.require
load_json = parent.load_json
require_close = parent.require_close
require_hash = parent.require_hash
gates_for_candidate = parent.gates_for_candidate


def selection_key(step: int, candidate: dict, recipe_id: str = RECIPE_ID) -> tuple:
    return parent.selection_key(step, candidate, recipe_id)


def expected_paths(runs: Path, data: Path, code_dir: Path) -> dict[str, Path]:
    root = runs / "recovery_vqa_replay"
    base_dir = runs / "recovery_pilot" / "eval" / "gcq425_lora_ce_s0"
    w4_dir = runs / "recovery_pilot" / "eval" / "w4rtn_lora_cwce_g5_s0"
    return {
        "validation": root / "development_round2_artifact_validation.json",
        "environment": code_dir / "env.sh",
        "protocol": code_dir / "recovery_vqa_development_round2_protocol.json",
        "validator": code_dir / "validate_recovery_vqa_development_round2.py",
        "launcher": code_dir / "launch_recovery_vqa_development_round2.sh",
        "batch_script": code_dir / "batch_recovery_vqa_development_round2.sh",
        "development_summarizer": Path(__file__).resolve(),
        "test": code_dir / "test_recovery_vqa_development_round2.py",
        "eval_rec": code_dir / "eval_rec.py",
        "eval_vqa": code_dir / "eval_vqa.py",
        "recovery_utils": code_dir / "recovery_utils.py",
        "quant_utils": code_dir / "quant_utils.py",
        "gcq_patches": code_dir / "gcq_patches.py",
        "summary_pilot_support": code_dir / "summarize_recovery_pilot.py",
        "summary_checkpoint_support": code_dir / "summarize_recovery_checkpoint_sweep.py",
        "summary_selected_support": code_dir / "summarize_recovery_selected_eval.py",
        "parent_summarizer": code_dir / "summarize_recovery_vqa_development.py",
        "parent_amended_wrapper": code_dir / "summarize_recovery_vqa_development_amended.py",
        "parent_amendment": code_dir / "development_summarizer_amendment.json",
        "original_protocol": code_dir / "recovery_vqa_replay_protocol.json",
        "training_launch": root / "training_launch_manifest.json",
        "parent_launch": root / "development_launch_manifest.json",
        "parent_summary": root / "development_summary.json",
        "promotion": runs / "promote_gcq_b4.25.json",
        "recovery_dev": data / "subsets" / "recovery_dev_1k.json",
        "vqa_dev": data / "subsets" / "vqa_val_5k.json",
        "baseline_rec": base_dir / "gcq425_untrained_recoverydev.rec.jsonl",
        "baseline_rec_metrics": base_dir / "gcq425_untrained_recoverydev.rec.metrics.json",
        "baseline_vqa": base_dir / "gcq425_untrained_recoverypilot_vqa5k.vqa.jsonl",
        "baseline_vqa_metrics": base_dir / "gcq425_untrained_recoverypilot_vqa5k.vqa.metrics.json",
        "w4_rec_metrics": w4_dir / "w4rtn_lora_cwce_g5_s0_recoverydev.rec.metrics.json",
    }


def validate_runtime_provenance(path: Path, launch_path: Path, step: int) -> dict:
    provenance = load_json(path)
    for key, expected in {
        "schema_version": 1,
        "round_id": ROUND_ID,
        "recipe_id": RECIPE_ID,
        "checkpoint_step": step,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "development_launch_manifest": str(launch_path),
        "development_launch_manifest_sha256": sha256_file(launch_path),
        "hardware_contract": "exactly one visible NVIDIA L40S",
        "hardware_gate_pass": True,
    }.items():
        require(provenance.get(key) == expected,
                f"runtime provenance {key} mismatch in {path}")
    scheduler = provenance.get("scheduler", {})
    require(isinstance(scheduler.get("job_id"), str) and scheduler["job_id"],
            f"missing scheduler job ID in {path}")
    require(scheduler.get("task_id") == str(CANDIDATE_STEPS.index(step) + 1),
            f"wrong scheduler task ID in {path}")
    require(isinstance(scheduler.get("hostname"), str) and scheduler["hostname"],
            f"missing hostname in {path}")
    python = provenance.get("python", {})
    require(all(isinstance(python.get(key), str) and python[key]
                for key in ("executable", "version")), f"bad Python provenance in {path}")
    packages = provenance.get("packages", {})
    require(all(isinstance(packages.get(key), str) and packages[key]
                for key in ("torch", "transformers", "peft", "safetensors", "numpy")),
            f"bad package provenance in {path}")
    cuda = provenance.get("cuda", {})
    require(cuda.get("available") is True and cuda.get("device_count") == 1,
            f"runtime did not expose exactly one CUDA device in {path}")
    require(isinstance(cuda.get("visible_devices"), str) and cuda["visible_devices"],
            f"missing CUDA_VISIBLE_DEVICES in {path}")
    require(isinstance(cuda.get("driver_versions"), list) and cuda["driver_versions"],
            f"missing driver provenance in {path}")
    devices = cuda.get("devices", [])
    require(len(devices) == 1 and devices[0].get("index") == 0
            and "L40S" in str(devices[0].get("name", "")),
            f"non-L40S runtime in {path}")
    require(tuple(devices[0].get("compute_capability", [])) >= (8, 0)
            and int(devices[0].get("total_memory_bytes", 0)) >= 40 * 2**30,
            f"insufficient GPU contract in {path}")
    return provenance


def verify_launch_and_inputs(
    runs: Path, data: Path, code_dir: Path
) -> tuple[dict, dict, dict[str, Path]]:
    root = runs / "recovery_vqa_replay"
    launch_path = root / "development_round2_launch_manifest.json"
    launch = load_json(launch_path)
    require(launch.get("schema_version") == 1 and launch.get("round_id") == ROUND_ID,
            "unexpected round-2 launch schema or ID")
    require(launch.get("recipe_id") == RECIPE_ID
            and launch.get("base_model") == BASE_MODEL
            and launch.get("base_revision") == BASE_REVISION,
            "round-2 model/recipe changed")
    require(launch.get("candidate_steps") == list(CANDIDATE_STEPS),
            "round-2 candidates changed")
    require(launch.get("preflight_complete") is True,
            "round-2 launch lacks completed preflight")
    require(launch.get("evaluation") == {
        "grounding_subset": "recovery_dev_1k", "grounding_examples": 1000,
        "primary_task": "rec", "primary_examples": 750, "vqa_examples": 5000,
    }, "round-2 evaluation contract changed")

    paths = expected_paths(runs, data, code_dir)
    hashes = launch.get("hashes")
    launched_paths = launch.get("paths")
    require(isinstance(hashes, dict) and set(hashes) == set(paths),
            "round-2 launch hash key set changed")
    require(launched_paths == {key: str(path) for key, path in paths.items()},
            "round-2 launch path map changed")
    for label, path in paths.items():
        require_hash(path, hashes[label], label)
    require(hashes["parent_summary"] == PARENT_SUMMARY_SHA
            and hashes["parent_amendment"] == AMENDMENT_SHA
            and hashes["parent_amended_wrapper"] == AMENDED_WRAPPER_SHA,
            "parent failure/amendment binding changed")

    protocol = load_json(paths["protocol"])
    require(protocol.get("round_id") == ROUND_ID
            and protocol.get("candidate_steps") == list(CANDIDATE_STEPS),
            "round-2 protocol changed")
    require(protocol.get("gates") == load_json(paths["original_protocol"])["development"]["gates"],
            "round-2 gates differ from parent protocol")
    require(protocol.get("selection_order")
            == load_json(paths["original_protocol"])["development"]["selection_order"],
            "round-2 selection rule differs from parent protocol")
    incident = protocol.get("step750_integrity_incident", {})
    require(incident.get("exactly_zero_tensor_count") == 296
            and incident.get("allocated_file_bytes") == 16_777_216
            and incident.get("status", "").startswith("excluded before"),
            "step-750 integrity incident is not bound")

    parent_summary = load_json(paths["parent_summary"])
    require(parent_summary.get("selection_succeeded") is False
            and parent_summary.get("eligible_steps") == [],
            "parent development result is no longer a failed selection")
    validation = load_json(paths["validation"])
    require(validation.get("round_id") == ROUND_ID
            and validation.get("candidate_steps") == list(CANDIDATE_STEPS),
            "round-2 artifact validation changed")
    require(validation.get("all_valid_checkpoint_integrity", {}).get("all_tensors_nonzero") is True
            and validation.get("all_valid_checkpoint_integrity", {}).get("all_files_fully_allocated") is True,
            "valid checkpoint integrity gates did not pass")
    require(validation.get("step750_integrity_incident", {}).get("status")
            == "invalid_excluded_before_round2_launch",
            "step-750 invalid status changed")

    checkpoints = launch.get("checkpoints", {})
    require(set(checkpoints) == set(map(str, CANDIDATE_STEPS)),
            "round-2 launch checkpoint map changed")
    for step in CANDIDATE_STEPS:
        launched = checkpoints[str(step)]
        validated = validation["checkpoints"][str(step)]
        directory = root / "adapters" / RECIPE_ID / f"checkpoint-{step:06d}"
        require(launched.get("adapter_dir") == str(directory)
                and validated.get("directory") == str(directory),
                f"noncanonical adapter path at step {step}")
        for file_key, filename, validation_key in (
            ("adapter_sha256", "adapter_model.safetensors", "artifact"),
            ("adapter_config_sha256", "adapter_config.json", "adapter_config_sha256"),
            ("manifest_sha256", "gcq_recovery_manifest.json", "manifest_sha256"),
        ):
            actual = sha256_file(directory / filename)
            expected = (validated[validation_key]["sha256"]
                        if validation_key == "artifact" else validated[validation_key])
            require(launched.get(file_key) == expected == actual,
                    f"checkpoint {step} {filename} changed")
    return launch, protocol, paths


def validate_completion(path: Path, launch_path: Path, step: int) -> dict:
    completion = load_json(path)
    require(completion.get("schema_version") == 1
            and completion.get("round_id") == ROUND_ID
            and completion.get("checkpoint_step") == step
            and completion.get("launch_sha256") == sha256_file(launch_path),
            f"bad completion record at step {step}")
    files = completion.get("files", {})
    required = {
        "runtime_provenance.json", "results.csv",
        f"vqa50_round2_step{step}_recoverydev.rec.jsonl",
        f"vqa50_round2_step{step}_recoverydev.rec.metrics.json",
        f"vqa50_round2_step{step}_vqa5k_dev.vqa.jsonl",
        f"vqa50_round2_step{step}_vqa5k_dev.vqa.metrics.json",
    }
    require(set(files) == required, f"unexpected completion file set at step {step}")
    for name, expected in files.items():
        require_hash(path.parent / name, expected, f"step {step} output {name}")
    return completion


def main() -> None:
    runs = Path(os.environ["GCQ_RUNS"])
    data = Path(os.environ["GCQ_DATA"])
    code_dir = Path(__file__).resolve().parent
    root = runs / "recovery_vqa_replay"
    launch_path = root / "development_round2_launch_manifest.json"
    launch, protocol, paths = verify_launch_and_inputs(runs, data, code_dir)

    rec_manifest = parent.load_rec_manifest(paths["recovery_dev"])
    vqa_uids, vqa_images = parent.expected_vqa_uids_and_images(paths["vqa_dev"])
    base_rec_ordered, base_rec_unique = parent.load_rec_rows(paths["baseline_rec"], rec_manifest)
    _, base_primary = parent.validate_rec_metrics(
        paths["baseline_rec_metrics"], base_rec_ordered,
        expected_tag="gcq425_untrained_recoverydev",
    )
    base_vqa_rows = parent.load_and_validate_vqa_rows(paths["baseline_vqa"], vqa_uids)
    parent.validate_vqa_metrics(
        paths["baseline_vqa_metrics"], base_vqa_rows,
        expected_tag="gcq425_untrained_recoverypilot_vqa5k",
        vqa_manifest_path=paths["vqa_dev"], require_hardened_schema=False,
    )
    base_vqa_score = parent.mean_vqa(base_vqa_rows)
    w4_metrics = load_json(paths["w4_rec_metrics"])
    w4_primary = parent.group_scores(w4_metrics, "rec")
    require(w4_primary["n"] == PRIMARY_REC_EXAMPLES, "bad W4 primary REC count")

    references = protocol["reference_scores"]
    for metric, key in {
        "rec": "untrained_gcq_primary_rec", "giou": "untrained_gcq_primary_giou",
        "precise_iou": "untrained_gcq_primary_precise_iou",
        "parse_fail": "untrained_gcq_primary_parse_fail",
    }.items():
        require_close(base_primary[metric], references[key], f"reference {metric}")
    require_close(w4_primary["rec"], references["w4_cwce_primary_rec"], "reference W4 REC")
    require_close(base_vqa_score, references["untrained_gcq_vqa5k"], "reference VQA")

    candidates: dict[str, dict] = {}
    runtime_signatures: dict[str, dict] = {}
    for index, step in enumerate(CANDIDATE_STEPS):
        directory = root / "development_round2" / f"step{step}"
        tag = f"vqa50_round2_step{step}"
        provenance_path = directory / "runtime_provenance.json"
        rec_log = directory / f"{tag}_recoverydev.rec.jsonl"
        rec_metrics = directory / f"{tag}_recoverydev.rec.metrics.json"
        vqa_log = directory / f"{tag}_vqa5k_dev.vqa.jsonl"
        vqa_metrics = directory / f"{tag}_vqa5k_dev.vqa.metrics.json"
        completion_path = directory / "completion_manifest.json"
        validate_completion(completion_path, launch_path, step)
        provenance = validate_runtime_provenance(provenance_path, launch_path, step)
        device = provenance["cuda"]["devices"][0]
        runtime_signatures[str(step)] = {
            "python": provenance["python"], "packages": provenance["packages"],
            "cuda_runtime_version": provenance["cuda"]["runtime_version"],
            "cudnn_version": provenance["cuda"]["cudnn_version"],
            "driver_versions": provenance["cuda"]["driver_versions"],
            "device_name": device["name"], "compute_capability": device["compute_capability"],
            "total_memory_bytes": device["total_memory_bytes"],
        }
        rec_ordered, rec_unique = parent.load_rec_rows(rec_log, rec_manifest)
        _, primary = parent.validate_rec_metrics(
            rec_metrics, rec_ordered, expected_tag=f"{tag}_recoverydev"
        )
        vqa_rows = parent.load_and_validate_vqa_rows(vqa_log, vqa_uids)
        parent.validate_vqa_metrics(
            vqa_metrics, vqa_rows, expected_tag=f"{tag}_vqa5k_dev",
            vqa_manifest_path=paths["vqa_dev"], require_hardened_schema=True,
        )
        parent.require_same_uids(base_vqa_rows, vqa_rows, f"checkpoint {step} VQA")

        seed = BOOTSTRAP_SEED + index * 5
        paired_rec = {
            metric: parent.paired_cluster_contrast(
                {"baseline": base_rec_unique, "candidate": rec_unique},
                {"candidate": 1.0, "baseline": -1.0}, metric, task="rec",
                resamples=BOOTSTRAP_RESAMPLES, seed=seed + metric_index,
            )
            for metric_index, metric in enumerate(("rec", "giou", "precise_iou"))
        }
        all_rec_images = {row["uid"]: str(row["image_id"]) for row in rec_manifest}
        paired_rec["parse_fail"] = paired_image_delta_primary_rec(
            base_rec_ordered, rec_ordered, all_rec_images, field="parse_fail",
            resamples=BOOTSTRAP_RESAMPLES, seed=seed + 3,
        )
        vqa_pair = parent.paired_image_delta(
            base_vqa_rows, vqa_rows, vqa_images,
            resamples=BOOTSTRAP_RESAMPLES, seed=seed + 4,
        )
        candidate_vqa = parent.mean_vqa(vqa_rows)
        for metric in ("rec", "giou", "precise_iou", "parse_fail"):
            require_close(paired_rec[metric]["observed"],
                          primary[metric] - base_primary[metric],
                          f"step {step} paired {metric}")
        require_close(vqa_pair["observed"], candidate_vqa - base_vqa_score,
                      f"step {step} paired VQA")
        gates = gates_for_candidate(
            primary, base_primary, w4_primary, candidate_vqa, base_vqa_score,
            vqa_pair, protocol["gates"],
        )
        checkpoint = launch["checkpoints"][str(step)]
        candidates[str(step)] = {
            "step": step, "recipe_id": RECIPE_ID,
            "adapter_dir": checkpoint["adapter_dir"],
            "adapter_sha256": checkpoint["adapter_sha256"],
            "adapter_config_sha256": checkpoint["adapter_config_sha256"],
            "manifest_sha256": checkpoint["manifest_sha256"],
            "runtime_provenance": provenance,
            "primary_rec": primary,
            "paired_primary_vs_untrained_gcq": paired_rec,
            "vqa_dev_5k": {
                "untrained_gcq": base_vqa_score, "candidate": candidate_vqa,
                "point_delta": candidate_vqa - base_vqa_score,
                "paired_delta": vqa_pair,
            },
            "gates": gates, "eligible": all(gates.values()),
            "output_hashes": {
                "completion_manifest": sha256_file(completion_path),
                "runtime_provenance": sha256_file(provenance_path),
                "rec_jsonl": sha256_file(rec_log), "rec_metrics": sha256_file(rec_metrics),
                "vqa_jsonl": sha256_file(vqa_log), "vqa_metrics": sha256_file(vqa_metrics),
            },
        }

    reference_runtime = runtime_signatures[str(CANDIDATE_STEPS[0])]
    for step in CANDIDATE_STEPS[1:]:
        require(runtime_signatures[str(step)] == reference_runtime,
                f"runtime software/hardware differs at step {step}")
    eligible_steps = [step for step in CANDIDATE_STEPS if candidates[str(step)]["eligible"]]
    selected_step = max(
        eligible_steps, key=lambda step: selection_key(step, candidates[str(step)]),
        default=None,
    )
    selected = None
    if selected_step is not None:
        winner = candidates[str(selected_step)]
        selected = {
            "step": selected_step, "recipe_id": RECIPE_ID,
            "adapter_dir": winner["adapter_dir"],
            "adapter_sha256": winner["adapter_sha256"],
            "adapter_config_sha256": winner["adapter_config_sha256"],
            "manifest_sha256": winner["manifest_sha256"],
            "selection_key": list(selection_key(selected_step, winner)),
            "scores": {"primary_rec": winner["primary_rec"],
                       "vqa_dev_5k": winner["vqa_dev_5k"]},
            "gates": winner["gates"], "eligible": True,
        }

    summary = {
        "schema_version": 1,
        "evaluation_role": "post-failure development-only checkpoint gating and selection",
        "round_id": ROUND_ID, "recipe_id": RECIPE_ID,
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "protocol_sha256": launch["hashes"]["protocol"],
        "development_launch_manifest": str(launch_path),
        "development_launch_manifest_sha256": sha256_file(launch_path),
        "parent_development": {
            "summary": str(paths["parent_summary"]),
            "summary_sha256": PARENT_SUMMARY_SHA,
            "selection_succeeded": False,
            "valid_evaluated_steps": list(PARENT_VALID_STEPS),
            "invalid_step750_excluded": True,
            "amendment_sha256": AMENDMENT_SHA,
            "amended_wrapper_sha256": AMENDED_WRAPPER_SHA,
        },
        "development_data": {
            "grounding_manifest": str(paths["recovery_dev"]),
            "grounding_manifest_sha256": launch["hashes"]["recovery_dev"],
            "grounding_examples": REC_EXAMPLES, "primary_rec_examples": PRIMARY_REC_EXAMPLES,
            "vqa_manifest": str(paths["vqa_dev"]),
            "vqa_manifest_sha256": launch["hashes"]["vqa_dev"],
            "vqa_examples": VQA_EXAMPLES,
        },
        "candidate_scope": protocol["candidate_scope"],
        "candidate_steps": list(CANDIDATE_STEPS),
        "bootstrap": {"unit": "image-clustered paired candidate-minus-untrained-GCQ",
                      "resamples": BOOTSTRAP_RESAMPLES, "base_seed": BOOTSTRAP_SEED,
                      "parse_fail_scope": "primary REC rows only"},
        "selection_rule": (
            "among candidates passing every frozen gate, lexicographically maximize "
            "primary REC, primary precise-IoU, primary GIoU, negative optimizer step, "
            "then stable recipe identifier"
        ),
        "references": {"untrained_gcq": {"primary_rec": base_primary,
                                           "vqa_dev_5k": base_vqa_score},
                       "w4_cwce": {"primary_rec": w4_primary}},
        "frozen_gate_thresholds": protocol["gates"],
        "candidates": candidates, "eligible_steps": eligible_steps,
        "selected": selected, "selection_succeeded": selected is not None,
        "confirmation_authorization": (
            "only this single selected checkpoint may advance under the separately frozen confirmation protocol"
            if selected is not None else "none; no round-2 checkpoint passed every frozen gate"
        ),
    }
    output = root / "development_round2_summary.json"
    with open(output, "x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"DEVELOPMENT ROUND-2 SUMMARY: {output}")
    if selected is None:
        raise SystemExit("no remaining valid checkpoint passed every frozen development gate")


if __name__ == "__main__":
    main()
