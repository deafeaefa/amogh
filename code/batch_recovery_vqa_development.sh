#!/bin/bash -l
#$ -P rise-tower
#$ -N gcq-vqa-dev
#$ -t 1-6
#$ -q l40s
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_m=40G
#$ -l h_rt=02:00:00
#$ -pe omp 4
#$ -j y
#$ -o /projectnb/rise-tower/eric1/GCQ/runs/recovery_vqa_replay/logs

set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh
cd /usr4/spclpgm/eric1/GCQ/code

STEPS=(200 300 400 500 600 750)
if [[ ! "${SGE_TASK_ID:-}" =~ ^[1-6]$ ]]; then
  echo "SGE_TASK_ID must select exactly one of the six frozen checkpoints" >&2
  exit 2
fi
STEP="${STEPS[$((SGE_TASK_ID - 1))]}"
ROOT="$GCQ_RUNS/recovery_vqa_replay"
RECIPE="gcq425_lora_cwce_vqa50_g5_lr5e5_s0"
ADAPTER_ROOT="$ROOT/adapters/$RECIPE"
if [[ "$STEP" == "750" ]]; then
  ADAPTER="$ADAPTER_ROOT"
else
  ADAPTER="$ADAPTER_ROOT/checkpoint-$(printf '%06d' "$STEP")"
fi
LAUNCH="$ROOT/development_launch_manifest.json"
PROMOTE="$GCQ_RUNS/promote_gcq_b4.25.json"
ENV_FILE="/usr4/spclpgm/eric1/GCQ/code/env.sh"
VALIDATOR="/usr4/spclpgm/eric1/GCQ/code/validate_recovery_vqa_replay.py"
SUMMARIZER="/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_vqa_development.py"
SUMMARY_PILOT_SUPPORT="/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_pilot.py"
SUMMARY_CHECKPOINT_SUPPORT="/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_checkpoint_sweep.py"
SUMMARY_SELECTED_SUPPORT="/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_selected_eval.py"
BASE_DIR="$GCQ_RUNS/recovery_pilot/eval/gcq425_lora_ce_s0"
BASE_REC="$BASE_DIR/gcq425_untrained_recoverydev.rec.jsonl"
BASE_REC_METRICS="$BASE_DIR/gcq425_untrained_recoverydev.rec.metrics.json"
BASE_VQA="$BASE_DIR/gcq425_untrained_recoverypilot_vqa5k.vqa.jsonl"
BASE_VQA_METRICS="$BASE_DIR/gcq425_untrained_recoverypilot_vqa5k.vqa.metrics.json"
W4_REC_METRICS="$GCQ_RUNS/recovery_pilot/eval/w4rtn_lora_cwce_g5_s0/w4rtn_lora_cwce_g5_s0_recoverydev.rec.metrics.json"
REVISION="89644892e4d85e24eaac8bacfd4f463576704203"
MODEL="Qwen/Qwen3-VL-2B-Instruct"
OUTPUT="$ROOT/development/step$STEP"
TAG="vqa50_step${STEP}"

if [[ ! -s "$LAUNCH" ]] || ! jq -e --arg recipe "$RECIPE" --arg model "$MODEL" \
   --arg revision "$REVISION" \
   '.schema_version == 1 and .recipe_id == $recipe and .base_model == $model and
    .base_revision == $revision and .candidate_steps == [200,300,400,500,600,750] and
    .evaluation == {grounding_subset: "recovery_dev_1k", grounding_examples: 1000,
      primary_task: "rec", primary_examples: 750, vqa_examples: 5000}' \
   "$LAUNCH" >/dev/null; then
  echo "frozen development launch manifest is missing or invalid" >&2
  exit 2
fi

check_top_hash() {
  local key="$1" path="$2" expected actual
  expected=$(jq -er --arg key "$key" '.hashes[$key]' "$LAUNCH")
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "development launch hash mismatch for $key: $path" >&2
    exit 2
  fi
}

check_top_path() {
  local key="$1" expected_path="$2" launched_path
  launched_path=$(jq -er --arg key "$key" '.paths[$key]' "$LAUNCH")
  if [[ "$launched_path" != "$expected_path" ]]; then
    echo "development launch path mismatch for $key: $launched_path" >&2
    exit 2
  fi
}

check_top_path validation "$ROOT/artifact_validation.json"
check_top_path environment "$ENV_FILE"
check_top_path protocol "/usr4/spclpgm/eric1/GCQ/code/recovery_vqa_replay_protocol.json"
check_top_path promotion "$PROMOTE"
check_top_path recovery_dev "$GCQ_DATA/subsets/recovery_dev_1k.json"
check_top_path vqa_dev "$GCQ_DATA/subsets/vqa_val_5k.json"
check_top_path baseline_rec "$BASE_REC"
check_top_path baseline_rec_metrics "$BASE_REC_METRICS"
check_top_path baseline_vqa "$BASE_VQA"
check_top_path baseline_vqa_metrics "$BASE_VQA_METRICS"
check_top_path w4_rec_metrics "$W4_REC_METRICS"

check_top_hash validation "$ROOT/artifact_validation.json"
check_top_hash environment "$ENV_FILE"
check_top_hash artifact_validator "$VALIDATOR"
check_top_hash protocol "/usr4/spclpgm/eric1/GCQ/code/recovery_vqa_replay_protocol.json"
check_top_hash eval_rec "/usr4/spclpgm/eric1/GCQ/code/eval_rec.py"
check_top_hash eval_vqa "/usr4/spclpgm/eric1/GCQ/code/eval_vqa.py"
check_top_hash recovery_utils "/usr4/spclpgm/eric1/GCQ/code/recovery_utils.py"
check_top_hash quant_utils "/usr4/spclpgm/eric1/GCQ/code/quant_utils.py"
check_top_hash gcq_patches "/usr4/spclpgm/eric1/GCQ/code/gcq_patches.py"
check_top_hash batch_script "/usr4/spclpgm/eric1/GCQ/code/batch_recovery_vqa_development.sh"
check_top_hash promotion "$PROMOTE"
check_top_hash recovery_dev "$GCQ_DATA/subsets/recovery_dev_1k.json"
check_top_hash vqa_dev "$GCQ_DATA/subsets/vqa_val_5k.json"
check_top_hash baseline_rec "$BASE_REC"
check_top_hash baseline_rec_metrics "$BASE_REC_METRICS"
check_top_hash baseline_vqa "$BASE_VQA"
check_top_hash baseline_vqa_metrics "$BASE_VQA_METRICS"
check_top_hash w4_rec_metrics "$W4_REC_METRICS"
check_top_hash development_summarizer "$SUMMARIZER"
check_top_hash summary_pilot_support "$SUMMARY_PILOT_SUPPORT"
check_top_hash summary_checkpoint_support "$SUMMARY_CHECKPOINT_SUPPORT"
check_top_hash summary_selected_support "$SUMMARY_SELECTED_SUPPORT"

EXPECTED_ADAPTER_DIR=$(jq -er --arg step "$STEP" '.checkpoints[$step].adapter_dir' "$LAUNCH")
EXPECTED_ADAPTER_SHA=$(jq -er --arg step "$STEP" '.checkpoints[$step].adapter_sha256' "$LAUNCH")
EXPECTED_CONFIG_SHA=$(jq -er --arg step "$STEP" '.checkpoints[$step].adapter_config_sha256' "$LAUNCH")
EXPECTED_MANIFEST_SHA=$(jq -er --arg step "$STEP" '.checkpoints[$step].manifest_sha256' "$LAUNCH")
if [[ "$ADAPTER" != "$EXPECTED_ADAPTER_DIR" ]] || \
   [[ "$(sha256sum "$ADAPTER/adapter_model.safetensors" | awk '{print $1}')" != "$EXPECTED_ADAPTER_SHA" ]] || \
   [[ "$(sha256sum "$ADAPTER/adapter_config.json" | awk '{print $1}')" != "$EXPECTED_CONFIG_SHA" ]] || \
   [[ "$(sha256sum "$ADAPTER/gcq_recovery_manifest.json" | awk '{print $1}')" != "$EXPECTED_MANIFEST_SHA" ]]; then
  echo "checkpoint $STEP changed after development launch" >&2
  exit 2
fi
if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to mix or overwrite checkpoint-$STEP development outputs" >&2
  exit 2
fi
mkdir -p "$OUTPUT"
"$PYT" - "$OUTPUT/runtime_provenance.json" "$LAUNCH" "$STEP" "$RECIPE" \
  "$MODEL" "$REVISION" <<'PY'
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


output, launch, step, recipe, model, revision = sys.argv[1:]
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
    len(devices) == 1 and "L40S" in devices[0]["name"] and bool(driver_versions)
)

report = {
    "schema_version": 1,
    "recipe_id": recipe,
    "checkpoint_step": int(step),
    "base_model": model,
    "base_revision": revision,
    "development_launch_manifest": launch,
    "development_launch_manifest_sha256": sha256_file(launch),
    "scheduler": {
        "job_id": os.environ.get("JOB_ID"),
        "task_id": os.environ.get("SGE_TASK_ID"),
        "queue": os.environ.get("QUEUE"),
        "hostname": socket.gethostname(),
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
    "hardware_contract": "exactly one visible NVIDIA L40S",
    "hardware_gate_pass": hardware_gate,
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
export GCQ_RUNS="$OUTPUT"

"$PYT" eval_rec.py --model "$MODEL" --revision "$REVISION" \
  --subset recovery_dev_1k --tag "${TAG}_recoverydev" \
  --rtn-bits 4 --rtn-group 128 --promote-file "$PROMOTE" --adapter-dir "$ADAPTER" \
  --max-pixels 1003520 --batch 16 --device cuda:0

"$PYT" eval_vqa.py --model "$MODEL" --revision "$REVISION" --task vqa \
  --vqa-file "$GCQ_DATA/subsets/vqa_val_5k.json" --tag "${TAG}_vqa5k_dev" \
  --rtn-bits 4 --rtn-group 128 \
  --promote-file "$PROMOTE" --adapter-dir "$ADAPTER" \
  --max-pixels 1003520 --batch 24 --device cuda:0
