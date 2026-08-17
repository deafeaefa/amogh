#!/bin/bash -l
#$ -P rise-tower
#$ -N gcq-vqa-dev-r2
#$ -t 1-9
#$ -q l40s
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_m=40G
#$ -l h_rt=02:00:00
#$ -pe omp 4
#$ -j y
#$ -o /projectnb/rise-tower/eric1/GCQ/runs/recovery_vqa_replay/logs_round2

set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh
cd /usr4/spclpgm/eric1/GCQ/code

STEPS=(50 100 150 250 350 450 550 650 700)
if [[ ! "${SGE_TASK_ID:-}" =~ ^[1-9]$ ]]; then
  echo "SGE_TASK_ID must select exactly one of the nine frozen round-2 checkpoints" >&2
  exit 2
fi
STEP="${STEPS[$((SGE_TASK_ID - 1))]}"
ROOT="$GCQ_RUNS/recovery_vqa_replay"
ROUND_DIR="$ROOT/development_round2"
OUTPUT="$ROUND_DIR/step$STEP"
LAUNCH="$ROOT/development_round2_launch_manifest.json"
RECIPE="gcq425_lora_cwce_vqa50_g5_lr5e5_s0"
ADAPTER="$ROOT/adapters/$RECIPE/checkpoint-$(printf '%06d' "$STEP")"
MODEL="Qwen/Qwen3-VL-2B-Instruct"
REVISION="89644892e4d85e24eaac8bacfd4f463576704203"
PROMOTE="$GCQ_RUNS/promote_gcq_b4.25.json"
TAG="vqa50_round2_step${STEP}"
MIN_RUNTIME_FREE_BYTES=$((512 * 1024 * 1024))

# Recheck every launch-bound code/data/baseline and adapter hash on the worker.
"$PYT" - "$GCQ_RUNS" "$GCQ_DATA" <<'PY'
import sys
from pathlib import Path
from summarize_recovery_vqa_development_round2 import verify_launch_and_inputs
verify_launch_and_inputs(Path(sys.argv[1]), Path(sys.argv[2]), Path.cwd())
PY

AVAILABLE_BYTES=$(df --output=avail -B1 "$ROOT" | awk 'NR == 2 {gsub(/ /, "", $1); print $1}')
if [[ ! "$AVAILABLE_BYTES" =~ ^[0-9]+$ ]] || (( AVAILABLE_BYTES < MIN_RUNTIME_FREE_BYTES )); then
  echo "round-2 storage headroom gate failed: available=$AVAILABLE_BYTES required=$MIN_RUNTIME_FREE_BYTES" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "refusing to mix or overwrite round-2 checkpoint-$STEP output: $OUTPUT" >&2
  exit 2
fi
mkdir -p "$ROUND_DIR"

if [[ -z "${TMPDIR:-}" ]] || [[ ! -d "$TMPDIR" ]]; then
  echo "scheduler TMPDIR is missing; refusing non-atomic direct project writes" >&2
  exit 2
fi
STAGE=$(mktemp -d "$TMPDIR/gcq-vqa-dev-r2-${JOB_ID:?}-${SGE_TASK_ID}.XXXXXX")
PUBLISH="${OUTPUT}.publish-${JOB_ID}-${SGE_TASK_ID}"
cleanup() {
  rm -rf -- "$STAGE"
  if [[ -d "$PUBLISH" ]]; then
    rm -rf -- "$PUBLISH"
  fi
}
trap cleanup EXIT

"$PYT" - "$STAGE/runtime_provenance.json" "$LAUNCH" "$STEP" "$RECIPE" \
  "$MODEL" "$REVISION" "$AVAILABLE_BYTES" <<'PY'
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
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

output, launch, step, recipe, model, revision, available = sys.argv[1:]
devices = []
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    devices.append({
        "index": index, "name": torch.cuda.get_device_name(index),
        "compute_capability": list(torch.cuda.get_device_capability(index)),
        "total_memory_bytes": int(props.total_memory),
    })
packages = {}
for distribution in ("torch", "transformers", "peft", "safetensors", "numpy"):
    try:
        packages[distribution] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        packages[distribution] = None
try:
    smi = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True, stderr=subprocess.STDOUT).strip().splitlines()
except (OSError, subprocess.SubprocessError) as exc:
    smi = [f"unavailable: {type(exc).__name__}: {exc}"]
drivers = sorted({fields[2].strip() for line in smi
                  if len(fields := line.split(",")) == 4})
hardware_pass = len(devices) == 1 and "L40S" in devices[0]["name"] and bool(drivers)
report = {
    "schema_version": 1,
    "round_id": "balanced-replay-remaining-checkpoints-development-round-2",
    "recipe_id": recipe, "checkpoint_step": int(step),
    "base_model": model, "base_revision": revision,
    "development_launch_manifest": launch,
    "development_launch_manifest_sha256": sha256_file(launch),
    "scheduler": {"job_id": os.environ.get("JOB_ID"),
                  "task_id": os.environ.get("SGE_TASK_ID"),
                  "queue": os.environ.get("QUEUE"), "hostname": socket.gethostname()},
    "python": {"executable": sys.executable, "version": platform.python_version()},
    "packages": packages,
    "cuda": {"available": torch.cuda.is_available(),
             "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "runtime_version": torch.version.cuda,
             "cudnn_version": torch.backends.cudnn.version(),
             "driver_versions": drivers, "device_count": len(devices),
             "devices": devices, "nvidia_smi": smi},
    "storage": {"project_available_bytes_at_start": int(available),
                "minimum_required_bytes": 512 * 1024 * 1024,
                "staging_root": os.path.dirname(output)},
    "hardware_contract": "exactly one visible NVIDIA L40S",
    "hardware_gate_pass": hardware_pass,
}
with open(output, "x") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write("\n")
if not hardware_pass:
    raise SystemExit(f"homogeneous L40S hardware gate failed: {devices!r}, {drivers!r}")
print(json.dumps(report, indent=2, sort_keys=True))
PY

export GCQ_RUNS="$STAGE"
"$PYT" eval_rec.py --model "$MODEL" --revision "$REVISION" \
  --subset recovery_dev_1k --tag "${TAG}_recoverydev" \
  --rtn-bits 4 --rtn-group 128 --promote-file "$PROMOTE" --adapter-dir "$ADAPTER" \
  --max-pixels 1003520 --batch 16 --device cuda:0

"$PYT" eval_vqa.py --model "$MODEL" --revision "$REVISION" --task vqa \
  --vqa-file "$GCQ_DATA/subsets/vqa_val_5k.json" --tag "${TAG}_vqa5k_dev" \
  --rtn-bits 4 --rtn-group 128 --promote-file "$PROMOTE" --adapter-dir "$ADAPTER" \
  --max-pixels 1003520 --batch 24 --device cuda:0

# Score-blind structural validation before publication.
"$PYT" - "$STAGE" "$STEP" "$GCQ_DATA" <<'PY'
import sys
from pathlib import Path
import summarize_recovery_vqa_development as audit

stage, step, data = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
tag = f"vqa50_round2_step{step}"
rec_manifest_path = data / "subsets" / "recovery_dev_1k.json"
vqa_manifest_path = data / "subsets" / "vqa_val_5k.json"
rec_manifest = audit.load_rec_manifest(rec_manifest_path)
rec_rows, _ = audit.load_rec_rows(stage / f"{tag}_recoverydev.rec.jsonl", rec_manifest)
audit.validate_rec_metrics(stage / f"{tag}_recoverydev.rec.metrics.json", rec_rows,
                           expected_tag=f"{tag}_recoverydev")
vqa_uids, _ = audit.expected_vqa_uids_and_images(vqa_manifest_path)
vqa_rows = audit.load_and_validate_vqa_rows(stage / f"{tag}_vqa5k_dev.vqa.jsonl", vqa_uids)
audit.validate_vqa_metrics(stage / f"{tag}_vqa5k_dev.vqa.metrics.json", vqa_rows,
                           expected_tag=f"{tag}_vqa5k_dev",
                           vqa_manifest_path=vqa_manifest_path,
                           require_hardened_schema=True)
results = stage / "results.csv"
audit.require(results.is_file() and sum(1 for _ in open(results)) == 3,
              "round-2 results.csv must contain one header and two result rows")
PY

AVAILABLE_BEFORE_PUBLISH=$(df --output=avail -B1 "$ROOT" | awk 'NR == 2 {gsub(/ /, "", $1); print $1}')
if [[ ! "$AVAILABLE_BEFORE_PUBLISH" =~ ^[0-9]+$ ]] || (( AVAILABLE_BEFORE_PUBLISH < MIN_RUNTIME_FREE_BYTES )); then
  echo "round-2 publish headroom gate failed: available=$AVAILABLE_BEFORE_PUBLISH" >&2
  exit 2
fi
if [[ -e "$PUBLISH" ]] || [[ -e "$OUTPUT" ]]; then
  echo "round-2 publish target already exists" >&2
  exit 2
fi
mkdir "$PUBLISH"
for NAME in \
  runtime_provenance.json results.csv \
  "${TAG}_recoverydev.rec.jsonl" "${TAG}_recoverydev.rec.metrics.json" \
  "${TAG}_vqa5k_dev.vqa.jsonl" "${TAG}_vqa5k_dev.vqa.metrics.json"; do
  cp -- "$STAGE/$NAME" "$PUBLISH/$NAME"
done
"$PYT" - "$PUBLISH/completion_manifest.json" "$PUBLISH" "$LAUNCH" "$STEP" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

output, directory, launch, step = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
files = {path.name: sha(path) for path in sorted(directory.iterdir()) if path.is_file()}
report = {
    "schema_version": 1,
    "round_id": "balanced-replay-remaining-checkpoints-development-round-2",
    "checkpoint_step": step,
    "launch_sha256": sha(launch),
    "files": files,
}
with open(output, "x") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
chmod -R a-w "$PUBLISH"
mv -- "$PUBLISH" "$OUTPUT"
trap - EXIT
rm -rf -- "$STAGE"
echo "ROUND2 TASK COMPLETE step=$STEP output=$OUTPUT"
