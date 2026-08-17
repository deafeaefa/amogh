#!/bin/bash -l
#$ -P rise-tower
#$ -N gcq-ground-confirm
#$ -t 1-2
#$ -q l40s
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_m=40G
#$ -l h_rt=08:00:00
#$ -pe omp 4
#$ -j y
#$ -o /projectnb/rise-tower/eric1/GCQ/runs/recovery_vqa_replay/logs

set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh
cd /usr4/spclpgm/eric1/GCQ/code

if [[ ! "${SGE_TASK_ID:-}" =~ ^[12]$ ]]; then
  echo "SGE_TASK_ID must be 1 (testA) or 2 (testB)" >&2
  exit 2
fi
if [[ "$SGE_TASK_ID" == "1" ]]; then
  SPLIT="testA"
  SUBSET="refcocoplus_testA_confirm_full"
else
  SPLIT="testB"
  SUBSET="refcocoplus_testB_confirm_full"
fi

BASE_RUNS="$GCQ_RUNS"
ROOT="$BASE_RUNS/recovery_vqa_replay"
CONFIRMATION="$ROOT/grounding_confirmation"
LAUNCH="$CONFIRMATION/grounding_confirmation_launch_manifest.json"
SUMMARY="$ROOT/grounding_confirmation_summary.json"
OUTPUT="$CONFIRMATION/$SPLIT"
SUMMARIZER="/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_grounding_confirmation.py"
MODEL="Qwen/Qwen3-VL-2B-Instruct"
REVISION="89644892e4d85e24eaac8bacfd4f463576704203"
PROMOTION="$BASE_RUNS/promote_gcq_b4.25.json"
MINIMUM_STORAGE_HEADROOM_BYTES=1073741824

die() {
  echo "$*" >&2
  exit 2
}

hash_file() {
  sha256sum "$1" | awk '{print $1}'
}

[[ -f "$LAUNCH" && -s "$LAUNCH" ]] || die "frozen grounding launch is missing: $LAUNCH"
[[ "$(stat -c '%a' "$LAUNCH")" == "444" ]] || \
  die "frozen grounding launch must be immutable mode 0444: $LAUNCH"
if ! jq -e --arg split "$SPLIT" --arg subset "$SUBSET" \
    --arg model "$MODEL" --arg revision "$REVISION" --argjson task "$SGE_TASK_ID" \
    --argjson minimum "$MINIMUM_STORAGE_HEADROOM_BYTES" '
    .schema_version == 1 and
    .evaluation_role == "one-time frozen RefCOCO+ grounding confirmation" and
    .base_model == $model and .base_revision == $revision and
    .arm_order == ["bf16", "uniform_rtn_w4", "untrained_gcq4.25", "selected_adapter"] and
    .scheduler_array_tasks == {"1": "testA", "2": "testB"} and
    .splits[$split].task_id == $task and .splits[$split].subset == $subset and
    .evaluation.hardware == "one NVIDIA L40S per split" and
    .evaluation.within_task_execution == "all four arms sequentially on the same physical GPU" and
    .evaluation.max_pixels == 1003520 and .evaluation.batch_size == 16 and
    .evaluation.device == "cuda:0" and .evaluation.decoding == "greedy" and
    .no_peeking.expected_output_tags_absent_when_launch_frozen == true and
    .no_peeking.alternate_tag_identity_scan_before_launch == true and
    .no_peeking.recheck_each_split_at_job_start == true and
    .untouched_audit.fixed_tag_outputs_found == 0 and
    .untouched_audit.matching_subset_metrics_found == 0 and
    .untouched_audit.confirmation_uid_prediction_logs_found == 0 and
    .untouched_audit.checked_before_launch == true and
    .untouched_audit.recheck_each_split_at_job_start == true and
    .storage_headroom.minimum_available_bytes == $minimum and
    .storage_headroom.recheck_at_job_start == true and
    (.storage_headroom.launcher_probes | keys | sort) == ["model_cache", "output"]' \
    "$LAUNCH" >/dev/null; then
  die "frozen grounding launch contract is invalid"
fi

# Revalidate every path/hash in the immutable launch before inference.  The
# preflight independently re-walks the VQA authorization chain, adapter hashes,
# manifests, disjointness evidence, source Arrow files, and all 1,500 images.
while IFS=$'\t' read -r KEY FILE_PATH EXPECTED; do
  [[ -f "$FILE_PATH" && -s "$FILE_PATH" ]] || die "missing launch input $KEY: $FILE_PATH"
  [[ "$(hash_file "$FILE_PATH")" == "$EXPECTED" ]] || \
    die "launch input changed for $KEY: $FILE_PATH"
done < <(jq -r '.paths as $paths | .hashes | to_entries[] |
  [.key, $paths[.key], .value] | @tsv' "$LAUNCH")

PREFLIGHT=$(mktemp /tmp/gcq-grounding-task-preflight.XXXXXX.json)
trap 'rm -f "$PREFLIGHT"' EXIT
"$PYT" "$SUMMARIZER" --preflight > "$PREFLIGHT"
if ! jq -e --slurpfile current "$PREFLIGHT" --arg split "$SPLIT" '
    .selected == $current[0].selected and
    .splits[$split].manifest == $current[0].splits[$split].manifest and
    .splits[$split].manifest_sha256 == $current[0].splits[$split].manifest_sha256 and
    .splits[$split].ordered_uid_sha256 == $current[0].splits[$split].ordered_uid_sha256 and
    .splits[$split].expressions == $current[0].splits[$split].expressions and
    .splits[$split].images == $current[0].splits[$split].images and
    .splits[$split].image_inventory == $current[0].splits[$split].image_inventory' \
    "$LAUNCH" >/dev/null; then
  die "current frozen inputs disagree with the grounding launch"
fi
rm -f "$PREFLIGHT"
trap - EXIT

SELECTED=$(jq -c '.selected' "$LAUNCH")
ADAPTER_DIR=$(jq -er '.selected.adapter_dir' "$LAUNCH")
ADAPTER_SHA=$(jq -er '.selected.adapter_sha256' "$LAUNCH")
if [[ "$(hash_file "$ADAPTER_DIR/adapter_model.safetensors")" != "$ADAPTER_SHA" ]]; then
  die "VQA-confirmed adapter changed before grounding inference"
fi
if [[ -e "$OUTPUT" ]]; then
  die "refusing to mix or overwrite grounding outputs: $OUTPUT"
fi

JOB_NO_PEEK=$("$PYT" "$SUMMARIZER" --audit-pristine --audit-split "$SPLIT")
if ! jq -e --arg split "$SPLIT" '
    .splits == [$split] and .fixed_tag_outputs_found == 0 and
    .matching_subset_metrics_found == 0 and
    .confirmation_uid_prediction_logs_found == 0' \
    <<<"$JOB_NO_PEEK" >/dev/null; then
  die "grounding no-peeking recheck failed for $SPLIT"
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

# A plain mkdir is the per-task concurrency lock; a duplicate/requeued task
# cannot enter the same output directory and interleave CSV/JSONL writes.
mkdir "$OUTPUT"

LAUNCH_SHA=$(hash_file "$LAUNCH")
PROVENANCE="$OUTPUT/runtime_provenance.json"
"$PYT" - "$PROVENANCE" "$LAUNCH" "$LAUNCH_SHA" "$SPLIT" "$SGE_TASK_ID" \
  "$SELECTED" "$MODEL" "$REVISION" "$JOB_STORAGE" <<'PY'
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys

import torch


(
    output, launch, launch_sha, split, task_id, selected_json, model, revision,
    storage_json,
) = sys.argv[1:]
selected = json.loads(selected_json)
storage_headroom = json.loads(storage_json)
packages = {}
for key, distribution in (
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("safetensors", "safetensors"),
    ("numpy", "numpy"),
    ("pillow", "Pillow"),
):
    try:
        packages[key] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        packages[key] = None

try:
    smi_lines = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip().splitlines()
except (OSError, subprocess.SubprocessError) as exc:
    smi_lines = [f"unavailable: {type(exc).__name__}: {exc}"]
driver_versions = sorted(
    {
        fields[2].strip()
        for line in smi_lines
        if len(fields := line.split(",")) == 4
    }
)
devices = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    uuid = str(getattr(properties, "uuid", ""))
    if not uuid and len(smi_lines) == 1 and len(smi_lines[0].split(",")) == 4:
        uuid = smi_lines[0].split(",")[1].strip()
    devices.append(
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "uuid": uuid,
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": int(properties.total_memory),
        }
    )
hardware_gate = (
    len(devices) == 1
    and "L40S" in devices[0]["name"]
    and bool(devices[0]["uuid"])
    and bool(driver_versions)
    and devices[0]["total_memory_bytes"] >= 40 * 2**30
)
report = {
    "schema_version": 1,
    "evaluation_role": "one-time frozen RefCOCO+ grounding confirmation",
    "recipe_id": selected["recipe_id"],
    "base_model": model,
    "base_revision": revision,
    "split": split,
    "scheduler_array_task": int(task_id),
    "grounding_launch_manifest": launch,
    "grounding_launch_manifest_sha256": launch_sha,
    "selected": selected,
    "arm_order": ["bf16", "uniform_rtn_w4", "untrained_gcq4.25", "selected_adapter"],
    "scheduler": {
        "job_id": os.environ.get("JOB_ID"),
        "task_id": os.environ.get("SGE_TASK_ID"),
        "queue": os.environ.get("QUEUE"),
        "hostname": socket.gethostname(),
    },
    "python": {"executable": sys.executable, "version": platform.python_version()},
    "packages": packages,
    "cuda": {
        "available": torch.cuda.is_available(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "driver_versions": driver_versions,
        "device_count": len(devices),
        "devices": devices,
        "nvidia_smi": smi_lines,
    },
    "hardware_contract": "exactly one visible NVIDIA L40S",
    "hardware_gate_pass": hardware_gate,
    "storage_headroom": storage_headroom,
}
with open(output, "x", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
if not hardware_gate:
    raise SystemExit(f"L40S hardware gate failed: {devices!r}; drivers={driver_versions!r}")
print(json.dumps(report, indent=2, sort_keys=True))
PY

EXECUTION_LOG="$OUTPUT/arm_execution.jsonl"
record_arm() {
  local sequence="$1" arm="$2" tag="$3" bits="$4" promotion_sha="$5" adapter_sha="$6"
  [[ "$(hash_file "$LAUNCH")" == "$LAUNCH_SHA" ]] || die "grounding launch changed during inference"
  "$PYT" - "$EXECUTION_LOG" "$PROVENANCE" "$LAUNCH_SHA" "$SPLIT" \
    "$sequence" "$arm" "$tag" "$bits" "$promotion_sha" "$adapter_sha" \
    "$OUTPUT/${tag}.rec.jsonl" "$OUTPUT/${tag}.rec.metrics.json" <<'PY'
import hashlib
import json
import os
import subprocess
import sys

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


(
    execution_path,
    provenance_path,
    launch_sha,
    split,
    sequence,
    arm,
    tag,
    bits,
    promotion_sha,
    adapter_sha,
    jsonl_path,
    metrics_path,
) = sys.argv[1:]
with open(provenance_path, encoding="utf-8") as handle:
    provenance = json.load(handle)
if torch.cuda.device_count() != 1:
    raise SystemExit(f"arm completion sees {torch.cuda.device_count()} CUDA devices, expected one")
current_uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
if not current_uuid:
    try:
        uuids = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip().splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"could not re-read physical GPU UUID: {exc}") from exc
    if len(uuids) != 1:
        raise SystemExit(f"expected one visible physical GPU UUID, found {uuids!r}")
    current_uuid = uuids[0].strip()
if current_uuid != provenance["cuda"]["devices"][0]["uuid"]:
    raise SystemExit(
        "physical GPU changed within the sequential arm job: "
        f"{current_uuid!r} != {provenance['cuda']['devices'][0]['uuid']!r}"
    )
record = {
    "sequence_index": int(sequence),
    "arm": arm,
    "tag": tag,
    "split": split,
    "launch_sha256": launch_sha,
    "gpu_uuid": provenance["cuda"]["devices"][0]["uuid"],
    "configuration": {
        "rtn_bits": int(bits),
        "rtn_group": 128,
        "promotion_sha256": promotion_sha or None,
        "adapter_sha256": adapter_sha or None,
    },
    "output_hashes": {
        "rec_jsonl": sha256_file(jsonl_path),
        "rec_metrics": sha256_file(metrics_path),
    },
}
with open(execution_path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(record, sort_keys=True))
PY
}

export GCQ_RUNS="$OUTPUT"

BF16_TAG="bf16_refcocoplus_${SPLIT}_confirm"
"$PYT" eval_rec.py --model "$MODEL" --revision "$REVISION" \
  --subset "$SUBSET" --tag "$BF16_TAG" \
  --max-pixels 1003520 --batch 16 --device cuda:0
record_arm 1 bf16 "$BF16_TAG" 0 "" ""

W4_TAG="w4rtn_refcocoplus_${SPLIT}_confirm"
"$PYT" eval_rec.py --model "$MODEL" --revision "$REVISION" \
  --subset "$SUBSET" --tag "$W4_TAG" \
  --rtn-bits 4 --rtn-group 128 \
  --max-pixels 1003520 --batch 16 --device cuda:0
record_arm 2 uniform_rtn_w4 "$W4_TAG" 4 "" ""

GCQ_TAG="gcq425_untrained_refcocoplus_${SPLIT}_confirm"
"$PYT" eval_rec.py --model "$MODEL" --revision "$REVISION" \
  --subset "$SUBSET" --tag "$GCQ_TAG" \
  --rtn-bits 4 --rtn-group 128 --promote-file "$PROMOTION" \
  --max-pixels 1003520 --batch 16 --device cuda:0
record_arm 3 'untrained_gcq4.25' "$GCQ_TAG" 4 \
  "78b3138da1c70f2f944e3c63d5cfc754f59bf6062c27564d8cc6b2792ba0c1e6" ""

SELECTED_TAG="gcq425_vqa_selected_refcocoplus_${SPLIT}_confirm"
"$PYT" eval_rec.py --model "$MODEL" --revision "$REVISION" \
  --subset "$SUBSET" --tag "$SELECTED_TAG" \
  --rtn-bits 4 --rtn-group 128 --promote-file "$PROMOTION" \
  --adapter-dir "$ADAPTER_DIR" \
  --max-pixels 1003520 --batch 16 --device cuda:0
record_arm 4 selected_adapter "$SELECTED_TAG" 4 \
  "78b3138da1c70f2f944e3c63d5cfc754f59bf6062c27564d8cc6b2792ba0c1e6" \
  "$ADAPTER_SHA"

[[ "$(wc -l < "$EXECUTION_LOG")" -eq 4 ]] || die "arm execution log is incomplete"
[[ "$(hash_file "$LAUNCH")" == "$LAUNCH_SHA" ]] || die "grounding launch changed after inference"

# Publish an integrity-bound completion marker only after all four sequential
# arms finished.  Whichever array task observes both markers atomically claims
# the summary lock and applies the frozen gates.  Thus the two-task array is a
# complete pipeline without a race-prone manual post-processing step.
COMPLETE="$OUTPUT/task_complete.json"
"$PYT" - "$COMPLETE" "$SPLIT" "$LAUNCH_SHA" "$PROVENANCE" \
  "$EXECUTION_LOG" "$OUTPUT/results.csv" <<'PY'
import hashlib
import json
import os
import sys


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


output, split, launch_sha, provenance, execution, results = sys.argv[1:]
record = {
    "schema_version": 1,
    "split": split,
    "grounding_launch_manifest_sha256": launch_sha,
    "runtime_provenance_sha256": sha256_file(provenance),
    "arm_execution_sha256": sha256_file(execution),
    "results_csv_sha256": sha256_file(results),
    "all_four_arms_complete": True,
}
with open(output, "x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

OTHER_SPLIT="testB"
[[ "$SPLIT" == "testA" ]] || OTHER_SPLIT="testA"
if [[ -s "$CONFIRMATION/$OTHER_SPLIT/task_complete.json" ]]; then
  SUMMARY_LOCK="$CONFIRMATION/.summary_lock"
  if mkdir "$SUMMARY_LOCK" 2>/dev/null; then
    cleanup_summary_lock() { rmdir "$SUMMARY_LOCK" 2>/dev/null || true; }
    trap cleanup_summary_lock EXIT
    # A peer may have completed and removed the lock between our first mkdir
    # attempt and this one.  Never rerun the exclusive scientific publisher.
    if [[ ! -e "$SUMMARY" ]]; then
      GCQ_RUNS="$BASE_RUNS" "$PYT" "$SUMMARIZER"
    fi
    cleanup_summary_lock
    trap - EXIT
  fi
fi
