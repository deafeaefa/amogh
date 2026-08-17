#!/bin/bash -l
#$ -P rise-tower
#$ -N gcq-vqa-confirm
#$ -q l40s
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_m=40G
#$ -l h_rt=04:00:00
#$ -pe omp 4
#$ -j y
#$ -o /projectnb/rise-tower/eric1/GCQ/runs/recovery_vqa_replay/logs

set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh
cd /usr4/spclpgm/eric1/GCQ/code

PROJECT_RUNS="$GCQ_RUNS"
ROOT="$PROJECT_RUNS/recovery_vqa_replay"
CONFIRMATION="$ROOT/confirmation"
LAUNCH="$CONFIRMATION/confirmation_launch_manifest.json"
SUMMARY="$ROOT/vqa_confirmation_summary.json"
RECIPE="gcq425_lora_cwce_vqa50_g5_lr5e5_s0"
MODEL="Qwen/Qwen3-VL-2B-Instruct"
REVISION="89644892e4d85e24eaac8bacfd4f463576704203"
PROMOTION="$PROJECT_RUNS/promote_gcq_b4.25.json"
FRESH_VQA="$GCQ_DATA/subsets/vqa_fresh_confirm_5k.json"
FRESH_META="$GCQ_DATA/subsets/vqa_fresh_confirm_5k.meta.json"
ENV_FILE="/usr4/spclpgm/eric1/GCQ/code/env.sh"
LAUNCHER="/usr4/spclpgm/eric1/GCQ/code/launch_recovery_vqa_confirmation.sh"
BATCH_SCRIPT="/usr4/spclpgm/eric1/GCQ/code/batch_recovery_vqa_confirmation.sh"
SUMMARIZER="/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_vqa_confirmation.py"
EVAL_VQA="/usr4/spclpgm/eric1/GCQ/code/eval_vqa.py"
BASE_TAG="gcq425_untrained_vqa_fresh5k"
MINIMUM_STORAGE_HEADROOM_BYTES=1073741824

die() {
  echo "$*" >&2
  exit 2
}

hash_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_file() {
  [[ -f "$1" && -s "$1" ]] || die "required confirmation input is missing: $1"
}

require_path_and_hash() {
  local key="$1" expected_path="$2" launched_path expected_hash actual_hash
  launched_path=$(jq -er --arg key "$key" '.paths[$key]' "$LAUNCH") || \
    die "confirmation launch is missing path key: $key"
  [[ "$launched_path" == "$expected_path" ]] || \
    die "confirmation launch path mismatch for $key: $launched_path != $expected_path"
  require_file "$expected_path"
  expected_hash=$(jq -er --arg key "$key" '.hashes[$key]' "$LAUNCH") || \
    die "confirmation launch is missing hash key: $key"
  actual_hash=$(hash_file "$expected_path")
  [[ "$actual_hash" == "$expected_hash" ]] || \
    die "confirmation launch hash mismatch for $key: $expected_path"
}

require_file "$LAUNCH"
[[ "$(stat -c '%a' "$LAUNCH")" == "444" ]] || \
  die "confirmation launch manifest must be immutable (mode 0444): $LAUNCH"
if [[ -e "$SUMMARY" ]]; then
  die "refusing to replace an existing VQA confirmation summary: $SUMMARY"
fi

if ! jq -e --arg recipe "$RECIPE" --arg model "$MODEL" --arg revision "$REVISION" '
    .schema_version == 1 and .recipe_id == $recipe and
    .base_model == $model and .base_revision == $revision and
    .quantization == {method: "rtn_quantize_dequantize", bits: 4,
      group_size: 128, average_decoder_bits: 4.25, max_pixels: 1003520} and
    .evaluation.task == "vqa" and .evaluation.questions == 5000 and
    .evaluation.unique_images == 4571 and .evaluation.start == 0 and
    .evaluation.limit == 5000 and .evaluation.batch_size == 24 and
    .evaluation.device == "cuda:0" and
    .evaluation.execution_order ==
      ["untrained_gcq_baseline", "single_selected_candidate"] and
    .evaluation.same_job_and_gpu_required == true and
    (.evaluation.baseline_tag | type) == "string" and
    (.evaluation.candidate_tag | type) == "string" and
    (.trained_candidates | length) == 1 and
    .trained_candidates[0].recipe_id == $recipe and
    (.trained_candidates[0].step | type) == "number" and
    (.trained_candidates[0].adapter_dir | type) == "string" and
    (.trained_candidates[0].adapter_sha256 | type) == "string" and
    (.trained_candidates[0].adapter_config_sha256 | type) == "string" and
    (.trained_candidates[0].manifest_sha256 | type) == "string" and
    .bootstrap == {unit: "image-clustered paired candidate-minus-untrained-GCQ",
      resamples: 10000, seed: 20260850} and
    .only_scientific_gate.minimum == -0.015' "$LAUNCH" >/dev/null; then
  die "frozen confirmation launch manifest is invalid"
fi

STEP=$(jq -er '.trained_candidates[0].step' "$LAUNCH")
case "$STEP" in
  200|300|400|500|600)
    ADAPTER="$ROOT/adapters/$RECIPE/checkpoint-$(printf '%06d' "$STEP")"
    ;;
  750)
    ADAPTER="$ROOT/adapters/$RECIPE"
    ;;
  *)
    die "confirmation selected a step outside the frozen candidate set: $STEP"
    ;;
esac
CANDIDATE_TAG="vqa50_step${STEP}_vqa_fresh5k"
[[ "$(jq -er '.evaluation.baseline_tag' "$LAUNCH")" == "$BASE_TAG" ]] || \
  die "confirmation baseline tag is not the frozen tag"
[[ "$(jq -er '.evaluation.candidate_tag' "$LAUNCH")" == "$CANDIDATE_TAG" ]] || \
  die "confirmation candidate tag is not the step-derived frozen tag"
[[ "$(jq -er '.trained_candidates[0].adapter_dir' "$LAUNCH")" == "$ADAPTER" ]] || \
  die "confirmation adapter path is not canonical"

require_path_and_hash development_summary "$ROOT/development_summary.json"
require_path_and_hash development_launch "$ROOT/development_launch_manifest.json"
require_path_and_hash development_summarizer "/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_vqa_development.py"
require_path_and_hash artifact_validation "$ROOT/artifact_validation.json"
require_path_and_hash artifact_validator "/usr4/spclpgm/eric1/GCQ/code/validate_recovery_vqa_replay.py"
require_path_and_hash training_launch "$ROOT/training_launch_manifest.json"
require_path_and_hash protocol "/usr4/spclpgm/eric1/GCQ/code/recovery_vqa_replay_protocol.json"
require_path_and_hash promotion "$PROMOTION"
require_path_and_hash fresh_manifest "$FRESH_VQA"
require_path_and_hash fresh_metadata "$FRESH_META"
require_path_and_hash selected_adapter "$ADAPTER/adapter_model.safetensors"
require_path_and_hash selected_adapter_config "$ADAPTER/adapter_config.json"
require_path_and_hash selected_adapter_manifest "$ADAPTER/gcq_recovery_manifest.json"
require_path_and_hash eval_vqa "$EVAL_VQA"
require_path_and_hash recovery_utils "/usr4/spclpgm/eric1/GCQ/code/recovery_utils.py"
require_path_and_hash quant_utils "/usr4/spclpgm/eric1/GCQ/code/quant_utils.py"
require_path_and_hash gcq_patches "/usr4/spclpgm/eric1/GCQ/code/gcq_patches.py"
require_path_and_hash environment "$ENV_FILE"
require_path_and_hash launcher_script "$LAUNCHER"
require_path_and_hash batch_script "$BATCH_SCRIPT"
require_path_and_hash confirmation_summarizer "$SUMMARIZER"
require_path_and_hash analysis_vqa_helper "/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_checkpoint_sweep.py"
require_path_and_hash analysis_pairing_helper "/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_selected_eval.py"
require_path_and_hash analysis_recovery_helper "/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_pilot.py"

[[ "$(jq '.paths | length' "$LAUNCH")" -eq 24 ]] || \
  die "confirmation launch path set contains unexpected entries"
[[ "$(jq '.hashes | length' "$LAUNCH")" -eq 24 ]] || \
  die "confirmation launch hash set contains unexpected entries"

if ! jq -e --argjson minimum "$MINIMUM_STORAGE_HEADROOM_BYTES" \
    --arg output "$(realpath "$ROOT")" --arg cache "$(realpath "$HF_HOME")" '
    .untouched_audit.fresh_manifest_sha256 ==
      "416aea5f9dd6f4a4cfa7061c3b5ca0af88647e49d5f17ed00a8d83885ea21038" and
    .untouched_audit.fixed_tag_outputs_found == 0 and
    .untouched_audit.matching_input_metrics_found == 0 and
    .untouched_audit.fresh_uid_prediction_logs_found == 0 and
    .untouched_audit.fixed_tag_results_rows_found == 0 and
    .untouched_audit.checked_before_launch == true and
    .untouched_audit.recheck_at_job_start == true and
    .image_inventory.schema == "sorted-filename-tab-size-tab-sha256-newline-v1" and
    .image_inventory.images == 4571 and
    (.image_inventory.total_bytes | type) == "number" and
    (.image_inventory.aggregate_sha256 | type) == "string" and
    .storage_headroom.minimum_available_bytes == $minimum and
    .storage_headroom.recheck_at_job_start == true and
    (.storage_headroom.launcher_probes | keys | sort) == ["model_cache", "output"] and
    .storage_headroom.launcher_probes.output.path == $output and
    .storage_headroom.launcher_probes.model_cache.path == $cache and
    .storage_headroom.launcher_probes.output.available_bytes >= $minimum and
    .storage_headroom.launcher_probes.model_cache.available_bytes >= $minimum' \
    "$LAUNCH" >/dev/null; then
  die "confirmation no-peeking, image, or storage contract is invalid"
fi

ADAPTER_SHA=$(jq -er '.trained_candidates[0].adapter_sha256' "$LAUNCH")
CONFIG_SHA=$(jq -er '.trained_candidates[0].adapter_config_sha256' "$LAUNCH")
MANIFEST_SHA=$(jq -er '.trained_candidates[0].manifest_sha256' "$LAUNCH")
[[ "$ADAPTER_SHA" == "$(hash_file "$ADAPTER/adapter_model.safetensors")" ]] || \
  die "selected adapter hash differs from trained_candidates"
[[ "$CONFIG_SHA" == "$(hash_file "$ADAPTER/adapter_config.json")" ]] || \
  die "selected adapter config hash differs from trained_candidates"
[[ "$MANIFEST_SHA" == "$(hash_file "$ADAPTER/gcq_recovery_manifest.json")" ]] || \
  die "selected adapter manifest hash differs from trained_candidates"

DEV_SUMMARY="$ROOT/development_summary.json"
[[ "$(stat -c '%a' "$DEV_SUMMARY")" == "444" ]] || \
  die "development summary must be frozen mode 0444 before confirmation"
if ! jq -e --arg recipe "$RECIPE" --argjson step "$STEP" --arg dir "$ADAPTER" \
    --arg adapter "$ADAPTER_SHA" --arg config "$CONFIG_SHA" --arg manifest "$MANIFEST_SHA" '
    .schema_version == 1 and .selection_succeeded == true and
    .selected.recipe_id == $recipe and .selected.step == $step and
    .selected.adapter_dir == $dir and .selected.adapter_sha256 == $adapter and
    .selected.adapter_config_sha256 == $config and
    .selected.manifest_sha256 == $manifest and .selected.eligible == true' \
    "$DEV_SUMMARY" >/dev/null; then
  die "development summary no longer authorizes exactly this candidate"
fi
VALIDATED_SELECTED=$("$PYT" "$SUMMARIZER" --validate-development-selection \
  --runs "$PROJECT_RUNS")
if ! jq -e --argjson step "$STEP" --arg recipe "$RECIPE" --arg dir "$ADAPTER" \
    --arg adapter "$ADAPTER_SHA" --arg config "$CONFIG_SHA" \
    --arg manifest "$MANIFEST_SHA" '
    .step == $step and .recipe_id == $recipe and .adapter_dir == $dir and
    .adapter_sha256 == $adapter and .adapter_config_sha256 == $config and
    .manifest_sha256 == $manifest' <<<"$VALIDATED_SELECTED" >/dev/null; then
  die "recomputed development winner differs from the confirmation candidate"
fi

if [[ "$(hash_file "$FRESH_VQA")" != \
      "416aea5f9dd6f4a4cfa7061c3b5ca0af88647e49d5f17ed00a8d83885ea21038" ]] || \
   [[ "$(hash_file "$FRESH_META")" != \
      "a7777a0199f7fb432deeee94d478ca092474dd8a5e47cffe92a93d62b57601e8" ]]; then
  die "fresh VQA confirmation artifacts do not match the predeclared hashes"
fi
if ! jq -e '
    length == 5000 and
    ([.[].question_id] | unique | length) == 5000 and
    ([.[].image_id] | unique | length) == 4571' "$FRESH_VQA" >/dev/null; then
  die "fresh VQA confirmation manifest has invalid question/image counts"
fi

JOB_NO_PEEK=$("$PYT" "$SUMMARIZER" --audit-pristine --runs "$PROJECT_RUNS" \
  --fresh-manifest "$FRESH_VQA" --baseline-tag "$BASE_TAG" \
  --candidate-tag "$CANDIDATE_TAG")
if ! jq -e '
    .fresh_manifest_sha256 ==
      "416aea5f9dd6f4a4cfa7061c3b5ca0af88647e49d5f17ed00a8d83885ea21038" and
    .fixed_tag_outputs_found == 0 and .matching_input_metrics_found == 0 and
    .fresh_uid_prediction_logs_found == 0 and
    .fixed_tag_results_rows_found == 0' <<<"$JOB_NO_PEEK" >/dev/null; then
  die "fresh-confirmation no-peeking recheck failed"
fi

CURRENT_IMAGE_INVENTORY=$("$PYT" "$SUMMARIZER" --compute-image-inventory \
  --fresh-manifest "$FRESH_VQA" --image-dir "$GCQ_DATA/images/val2014")
if [[ "$(jq -Sc . <<<"$CURRENT_IMAGE_INVENTORY")" != \
      "$(jq -Sc '.image_inventory' "$LAUNCH")" ]]; then
  die "fresh VQA image bytes changed after confirmation launch"
fi

JOB_STORAGE=$("$PYT" - "$ROOT" "$HF_HOME" "$MINIMUM_STORAGE_HEADROOM_BYTES" <<'PY'
import json
import os
import sys
from pathlib import Path

output, cache, minimum = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
probes = {}
for label, path in (("output", output), ("model_cache", cache)):
    if not path.is_dir():
        raise SystemExit(f"storage probe directory is missing: {path}")
    stats = os.statvfs(path)
    available = int(stats.f_bavail * stats.f_frsize)
    if available < minimum:
        raise SystemExit(
            f"insufficient {label} storage headroom at {path}: "
            f"{available} < {minimum} bytes"
        )
    probes[label] = {"path": str(path.resolve()), "available_bytes": available}
print(json.dumps({
    "minimum_available_bytes": minimum,
    "probes": probes,
    "gate_pass": True,
}, sort_keys=True))
PY
)

UNEXPECTED=$(find "$CONFIRMATION" -mindepth 1 -maxdepth 1 \
  ! -name 'confirmation_launch_manifest.json' -print -quit)
[[ -z "$UNEXPECTED" ]] || \
  die "confirmation directory is not pristine at job start: $UNEXPECTED"

RUNTIME="$CONFIRMATION/runtime_provenance.json"
"$PYT" - "$RUNTIME" "$LAUNCH" "$STEP" "$RECIPE" "$MODEL" "$REVISION" \
  "$BASE_TAG" "$CANDIDATE_TAG" "$JOB_STORAGE" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


(output, launch, step, recipe, model, revision,
 baseline_tag, candidate_tag, storage_json) = sys.argv[1:]
storage_headroom = json.loads(storage_json)
devices = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    devices.append({
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "compute_capability": list(torch.cuda.get_device_capability(index)),
        "total_memory_bytes": int(properties.total_memory),
    })
packages = {}
for distribution in ("torch", "transformers", "peft", "safetensors", "numpy"):
    try:
        packages[distribution] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        packages[distribution] = None
try:
    nvidia_smi = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip().splitlines()
except (OSError, subprocess.SubprocessError) as exc:
    nvidia_smi = [f"unavailable: {type(exc).__name__}: {exc}"]
driver_versions = sorted({
    fields[2].strip()
    for line in nvidia_smi
    if len(fields := line.split(",")) == 4
})
hardware_gate = (
    len(devices) == 1
    and "L40S" in devices[0]["name"]
    and tuple(devices[0]["compute_capability"]) >= (8, 0)
    and devices[0]["total_memory_bytes"] >= 40 * 2**30
    and bool(driver_versions)
)
report = {
    "schema_version": 1,
    "recipe_id": recipe,
    "selected_step": int(step),
    "base_model": model,
    "base_revision": revision,
    "confirmation_launch_manifest": launch,
    "confirmation_launch_manifest_sha256": sha256_file(launch),
    "scheduler": {
        "job_id": os.environ.get("JOB_ID"),
        "task_id": os.environ.get("SGE_TASK_ID"),
        "queue": os.environ.get("QUEUE"),
        "hostname": socket.gethostname(),
        "batch_shell_pid": os.getppid(),
    },
    "python": {
        "executable": sys.executable,
        "version": platform.python_version(),
    },
    "packages": packages,
    "cuda": {
        "available": torch.cuda.is_available(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "driver_versions": driver_versions,
        "device_count": len(devices),
        "devices": devices,
        "nvidia_smi": nvidia_smi,
    },
    "execution": {
        "mode": "single non-array scheduler job",
        "order": ["untrained_gcq_baseline", "single_selected_candidate"],
        "baseline_tag": baseline_tag,
        "candidate_tag": candidate_tag,
        "device_argument_for_both": "cuda:0",
        "same_process_environment_for_both": True,
    },
    "hardware_contract": "exactly one visible NVIDIA L40S",
    "hardware_gate_pass": hardware_gate,
    "storage_headroom": storage_headroom,
}
with open(output, "x") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
if not hardware_gate:
    raise SystemExit(
        f"homogeneous L40S hardware gate failed: devices={devices!r}, "
        f"drivers={driver_versions!r}"
    )
print(json.dumps(report, indent=2, sort_keys=True))
PY

# Both evaluations intentionally execute synchronously in this one non-array
# scheduler allocation.  The untrained-GCQ baseline is always first.
export GCQ_RUNS="$CONFIRMATION"
BASELINE_STARTED_NS=$("$PYT" -c 'import time; print(time.time_ns())')
"$PYT" "$EVAL_VQA" --model "$MODEL" --revision "$REVISION" --task vqa \
  --vqa-file "$FRESH_VQA" --tag "$BASE_TAG" --start 0 --limit 5000 \
  --rtn-bits 4 --rtn-group 128 --promote-file "$PROMOTION" \
  --max-pixels 1003520 --batch 24 --device cuda:0 &
BASELINE_PID=$!
wait "$BASELINE_PID"
BASELINE_ENDED_NS=$("$PYT" -c 'import time; print(time.time_ns())')

CANDIDATE_STARTED_NS=$("$PYT" -c 'import time; print(time.time_ns())')
"$PYT" "$EVAL_VQA" --model "$MODEL" --revision "$REVISION" --task vqa \
  --vqa-file "$FRESH_VQA" --tag "$CANDIDATE_TAG" --start 0 --limit 5000 \
  --rtn-bits 4 --rtn-group 128 --promote-file "$PROMOTION" \
  --adapter-dir "$ADAPTER" --max-pixels 1003520 --batch 24 --device cuda:0 &
CANDIDATE_PID=$!
wait "$CANDIDATE_PID"
CANDIDATE_ENDED_NS=$("$PYT" -c 'import time; print(time.time_ns())')

LEDGER="$CONFIRMATION/execution_ledger.json"
"$PYT" - "$LEDGER" "$LAUNCH" "$EVAL_VQA" "$MODEL" "$REVISION" \
  "$FRESH_VQA" "$BASE_TAG" "$CANDIDATE_TAG" "$PROMOTION" "$ADAPTER" \
  "$ADAPTER_SHA" "$BASELINE_STARTED_NS" "$BASELINE_ENDED_NS" "$BASELINE_PID" \
  "$CANDIDATE_STARTED_NS" "$CANDIDATE_ENDED_NS" "$CANDIDATE_PID" <<'PY'
import hashlib
import json
import os
import socket
import sys
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


(output, launch, evaluator, model, revision, fresh_manifest, baseline_tag,
 candidate_tag, promotion, adapter, adapter_sha, baseline_started,
 baseline_ended, baseline_pid, candidate_started, candidate_ended,
 candidate_pid) = sys.argv[1:]
evaluations = []
for role, tag, started, ended, pid, adapter_path in (
    ("untrained_gcq_baseline", baseline_tag, baseline_started, baseline_ended,
     baseline_pid, None),
    ("single_selected_candidate", candidate_tag, candidate_started, candidate_ended,
     candidate_pid, adapter),
):
    jsonl = Path(output).parent / f"{tag}.vqa.jsonl"
    metrics = Path(output).parent / f"{tag}.vqa.metrics.json"
    argv = [
        sys.executable, evaluator, "--model", model, "--revision", revision,
        "--task", "vqa", "--vqa-file", fresh_manifest, "--tag", tag,
        "--start", "0", "--limit", "5000", "--rtn-bits", "4",
        "--rtn-group", "128", "--promote-file", promotion,
    ]
    if adapter_path is not None:
        argv += ["--adapter-dir", adapter_path]
    argv += ["--max-pixels", "1003520", "--batch", "24", "--device", "cuda:0"]
    evaluations.append({
        "role": role,
        "tag": tag,
        "python_pid": int(pid),
        "started_ns": int(started),
        "ended_ns": int(ended),
        "exit_code": 0,
        "argv": argv,
        "adapter": None if adapter_path is None else {
            "path": adapter_path,
            "sha256": adapter_sha,
        },
        "outputs": {
            "jsonl": {"path": str(jsonl), "sha256": sha256_file(jsonl)},
            "metrics": {"path": str(metrics), "sha256": sha256_file(metrics)},
        },
    })
report = {
    "schema_version": 1,
    "confirmation_launch_manifest": launch,
    "confirmation_launch_manifest_sha256": sha256_file(launch),
    "scheduler": {
        "job_id": os.environ.get("JOB_ID"),
        "task_id": os.environ.get("SGE_TASK_ID"),
        "queue": os.environ.get("QUEUE"),
        "hostname": socket.gethostname(),
        "batch_shell_pid": os.getppid(),
    },
    "shared_inference_contract": {
        "model": model,
        "base_revision": revision,
        "task": "vqa",
        "fresh_manifest": fresh_manifest,
        "fresh_manifest_sha256": sha256_file(fresh_manifest),
        "evaluator": evaluator,
        "evaluator_sha256": sha256_file(evaluator),
        "quantization": {
            "method": "rtn_quantize_dequantize",
            "bits": 4,
            "group_size": 128,
            "promotion_manifest": promotion,
            "promotion_manifest_sha256": sha256_file(promotion),
            "average_decoder_bits": 4.25,
        },
        "max_pixels": 1003520,
        "batch_size": 24,
        "device": "cuda:0",
        "blank_image": False,
        "generation": {"max_new_tokens": 16, "do_sample": False},
    },
    "evaluations": evaluations,
}
with open(output, "x", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

GCQ_RUNS="$PROJECT_RUNS" "$PYT" "$SUMMARIZER"
