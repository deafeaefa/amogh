#!/bin/bash
set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh

CODE_DIR="/usr4/spclpgm/eric1/GCQ/code"
ROOT="$GCQ_RUNS/recovery_vqa_replay"
PROTOCOL="$CODE_DIR/recovery_vqa_development_round2_protocol.json"
VALIDATOR="$CODE_DIR/validate_recovery_vqa_development_round2.py"
BATCH="$CODE_DIR/batch_recovery_vqa_development_round2.sh"
SUMMARIZER="$CODE_DIR/summarize_recovery_vqa_development_round2.py"
TEST="$CODE_DIR/test_recovery_vqa_development_round2.py"
LAUNCHER="$CODE_DIR/launch_recovery_vqa_development_round2.sh"
VALIDATION="$ROOT/development_round2_artifact_validation.json"
LAUNCH="$ROOT/development_round2_launch_manifest.json"
SUBMISSION="$ROOT/development_round2_submission.json"
OUTPUT_DIR="$ROOT/development_round2"
SUMMARY="$ROOT/development_round2_summary.json"
LOG_DIR="$ROOT/logs_round2"
MIN_LAUNCH_FREE_BYTES=$((1024 * 1024 * 1024))

for TARGET in "$VALIDATION" "$LAUNCH" "$SUBMISSION" "$OUTPUT_DIR" "$SUMMARY"; do
  if [[ -e "$TARGET" ]]; then
    echo "refusing to replace or mix frozen round-2 state: $TARGET" >&2
    exit 2
  fi
done

require_hash() {
  local path="$1" expected="$2" actual
  if [[ ! -s "$path" ]]; then
    echo "required round-2 input is missing or empty: $path" >&2
    exit 2
  fi
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "round-2 frozen input changed: $path ($actual != $expected)" >&2
    exit 2
  fi
}

# Bind the completed failed parent round and its one preregistered correctness repair.
require_hash "$ROOT/development_summary.json" "4b923471b3af3244b7874995e871ae38619c61298bdf07851ba93fac5cc35698"
require_hash "$CODE_DIR/development_summarizer_amendment.json" "3261c3cfa13401d5577d831f9d52a500b7a74d186c8314289e51395b4e78d3f6"
require_hash "$CODE_DIR/summarize_recovery_vqa_development_amended.py" "63ccce3388abd53ea46a793d6b2fbde923639eb5bb9460acf8bb9c30e673206d"
require_hash "$ROOT/development_launch_manifest.json" "a218efcf7337fc51f54a7c598ad679542bfeef079388cf480eda4265ff62ead7"
require_hash "$ROOT/training_launch_manifest.json" "803abe1f24dc2b20a91c9bcc43ceebf092dd0d8aaaf9d35e216cfece17eff607"
require_hash "$CODE_DIR/recovery_vqa_replay_protocol.json" "fd938d0d39116b989ffdcde4dd5ce64bbb419a1292e4ab9f32864416953e5e6d"

# Verify every parent-launch-bound evaluator, helper, datum, and baseline in place.
"$PYT" - "$ROOT/development_launch_manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

launch = json.load(open(sys.argv[1]))
path_keys = {
    "validation", "environment", "protocol", "promotion", "recovery_dev", "vqa_dev",
    "baseline_rec", "baseline_rec_metrics", "baseline_vqa", "baseline_vqa_metrics",
    "w4_rec_metrics",
}
for key in path_keys:
    path = Path(launch["paths"][key])
    if not path.is_file() or sha(path) != launch["hashes"][key]:
        raise SystemExit(f"parent launch-bound input changed: {key}: {path}")
code = Path("/usr4/spclpgm/eric1/GCQ/code")
for key, path in {
    "artifact_validator": code / "validate_recovery_vqa_replay.py",
    "eval_rec": code / "eval_rec.py", "eval_vqa": code / "eval_vqa.py",
    "recovery_utils": code / "recovery_utils.py", "quant_utils": code / "quant_utils.py",
    "gcq_patches": code / "gcq_patches.py",
    "development_summarizer": code / "summarize_recovery_vqa_development.py",
    "summary_pilot_support": code / "summarize_recovery_pilot.py",
    "summary_checkpoint_support": code / "summarize_recovery_checkpoint_sweep.py",
    "summary_selected_support": code / "summarize_recovery_selected_eval.py",
}.items():
    if not path.is_file() or sha(path) != launch["hashes"][key]:
        raise SystemExit(f"parent launch-bound code changed: {key}: {path}")
summary = json.load(open(Path(launch["paths"]["validation"]).parent / "development_summary.json"))
if summary.get("selection_succeeded") is not False or summary.get("eligible_steps") != []:
    raise SystemExit("parent development summary is not the bound failed selection")
PY

if ! jq -e '
  .schema_version == 1 and
  .round_id == "balanced-replay-remaining-checkpoints-development-round-2" and
  .candidate_steps == [50,100,150,250,350,450,550,650,700] and
  .training_checkpoint_inventory.systematic_saved_steps == [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750] and
  .training_checkpoint_inventory.structurally_valid_steps == [50,100,150,200,250,300,350,400,450,500,550,600,650,700] and
  .training_checkpoint_inventory.parent_valid_evaluated_steps == [200,300,400,500,600] and
  .training_checkpoint_inventory.invalid_steps == [750] and
  .step750_integrity_incident.exactly_zero_tensor_count == 296 and
  .step750_integrity_incident.allocated_file_bytes == 16777216 and
  .confirmation_policy.fresh_vqa_and_refcocoplus_touched_by_this_round == false
' "$PROTOCOL" >/dev/null; then
  echo "round-2 protocol is incomplete or changed" >&2
  exit 2
fi
for FILE in "$VALIDATOR" "$BATCH" "$SUMMARIZER" "$TEST" "$LAUNCHER"; do
  if [[ ! -s "$FILE" ]]; then
    echo "missing round-2 implementation file: $FILE" >&2
    exit 2
  fi
done

AVAILABLE_BYTES=$(df --output=avail -B1 "$ROOT" | awk 'NR == 2 {gsub(/ /, "", $1); print $1}')
if [[ ! "$AVAILABLE_BYTES" =~ ^[0-9]+$ ]] || (( AVAILABLE_BYTES < MIN_LAUNCH_FREE_BYTES )); then
  echo "round-2 launch storage headroom gate failed: available=$AVAILABLE_BYTES required=$MIN_LAUNCH_FREE_BYTES" >&2
  exit 2
fi

# Tests and score-blind tensor validation both complete before preregistration.
cd "$CODE_DIR"
"$PYT" -m pytest -q \
  test_recovery_vqa_development_round2.py \
  test_recovery_vqa_development.py \
  test_recovery_vqa_development_amendment.py

PREFLIGHT_DIR=$(mktemp -d "$ROOT/.development-round2-preflight.XXXXXX")
cleanup() { rm -rf -- "$PREFLIGHT_DIR"; }
trap cleanup EXIT
VALIDATION_TMP="$PREFLIGHT_DIR/artifact_validation.json"
LAUNCH_TMP="$PREFLIGHT_DIR/launch_manifest.json"
"$PYT" "$VALIDATOR" --output "$VALIDATION_TMP" --protocol "$PROTOCOL" \
  --runs "$GCQ_RUNS" --code-dir "$CODE_DIR"

"$PYT" - "$LAUNCH_TMP" "$VALIDATION_TMP" "$VALIDATION" "$LAUNCH" \
  "$AVAILABLE_BYTES" "$GCQ_RUNS" "$GCQ_DATA" "$CODE_DIR" <<'PY'
import datetime, hashlib, json, sys
from pathlib import Path

def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

output, validation_tmp, validation_path, launch_path, available, runs_arg, data_arg, code_arg = sys.argv[1:]
runs, data, code = Path(runs_arg), Path(data_arg), Path(code_arg)
root = runs / "recovery_vqa_replay"
validation = json.load(open(validation_tmp))
paths = {
    "validation": Path(validation_path), "environment": code / "env.sh",
    "protocol": code / "recovery_vqa_development_round2_protocol.json",
    "validator": code / "validate_recovery_vqa_development_round2.py",
    "launcher": code / "launch_recovery_vqa_development_round2.sh",
    "batch_script": code / "batch_recovery_vqa_development_round2.sh",
    "development_summarizer": code / "summarize_recovery_vqa_development_round2.py",
    "test": code / "test_recovery_vqa_development_round2.py",
    "eval_rec": code / "eval_rec.py", "eval_vqa": code / "eval_vqa.py",
    "recovery_utils": code / "recovery_utils.py", "quant_utils": code / "quant_utils.py",
    "gcq_patches": code / "gcq_patches.py",
    "summary_pilot_support": code / "summarize_recovery_pilot.py",
    "summary_checkpoint_support": code / "summarize_recovery_checkpoint_sweep.py",
    "summary_selected_support": code / "summarize_recovery_selected_eval.py",
    "parent_summarizer": code / "summarize_recovery_vqa_development.py",
    "parent_amended_wrapper": code / "summarize_recovery_vqa_development_amended.py",
    "parent_amendment": code / "development_summarizer_amendment.json",
    "original_protocol": code / "recovery_vqa_replay_protocol.json",
    "training_launch": root / "training_launch_manifest.json",
    "parent_launch": root / "development_launch_manifest.json",
    "parent_summary": root / "development_summary.json",
    "promotion": runs / "promote_gcq_b4.25.json",
    "recovery_dev": data / "subsets" / "recovery_dev_1k.json",
    "vqa_dev": data / "subsets" / "vqa_val_5k.json",
    "baseline_rec": runs / "recovery_pilot/eval/gcq425_lora_ce_s0/gcq425_untrained_recoverydev.rec.jsonl",
    "baseline_rec_metrics": runs / "recovery_pilot/eval/gcq425_lora_ce_s0/gcq425_untrained_recoverydev.rec.metrics.json",
    "baseline_vqa": runs / "recovery_pilot/eval/gcq425_lora_ce_s0/gcq425_untrained_recoverypilot_vqa5k.vqa.jsonl",
    "baseline_vqa_metrics": runs / "recovery_pilot/eval/gcq425_lora_ce_s0/gcq425_untrained_recoverypilot_vqa5k.vqa.metrics.json",
    "w4_rec_metrics": runs / "recovery_pilot/eval/w4rtn_lora_cwce_g5_s0/w4rtn_lora_cwce_g5_s0_recoverydev.rec.metrics.json",
}
hashes = {}
for key, path in paths.items():
    source = Path(validation_tmp) if key == "validation" else path
    if not source.is_file():
        raise SystemExit(f"missing launch-bound path {key}: {source}")
    hashes[key] = sha(source)
checkpoints = {}
for step in validation["candidate_steps"]:
    item = validation["checkpoints"][str(step)]
    checkpoints[str(step)] = {
        "adapter_dir": item["directory"],
        "adapter_sha256": item["artifact"]["sha256"],
        "adapter_config_sha256": item["adapter_config_sha256"],
        "manifest_sha256": item["manifest_sha256"],
        "tensor_count": item["artifact"]["tensor_count"],
        "zero_tensor_count": item["artifact"]["zero_tensor_count"],
        "apparent_bytes": item["artifact"]["apparent_bytes"],
        "allocated_bytes": item["artifact"]["allocated_bytes"],
        "structure_sha256": item["artifact"]["structure_sha256"],
    }
manifest = {
    "schema_version": 1,
    "round_id": "balanced-replay-remaining-checkpoints-development-round-2",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "role": "preregistered post-failure development-only complete checkpoint complement",
    "recipe_id": "gcq425_lora_cwce_vqa50_g5_lr5e5_s0",
    "base_model": "Qwen/Qwen3-VL-2B-Instruct",
    "base_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
    "candidate_steps": validation["candidate_steps"],
    "evaluation": {"grounding_subset": "recovery_dev_1k", "grounding_examples": 1000,
                   "primary_task": "rec", "primary_examples": 750, "vqa_examples": 5000},
    "preflight_complete": True,
    "preflight": {"tests": ["round2", "parent_development", "parent_amendment"],
                  "project_available_bytes": int(available),
                  "minimum_launch_free_bytes": 1024 * 1024 * 1024,
                  "all_valid_checkpoint_tensors_nonzero": True,
                  "all_valid_checkpoint_files_fully_allocated": True,
                  "step750_status": "invalid_excluded_before_round2_launch"},
    "checkpoints": checkpoints,
    "paths": {key: str(path) for key, path in paths.items()},
    "hashes": hashes,
}
with open(output, "x") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

chmod 0444 "$VALIDATION_TMP" "$LAUNCH_TMP"
ln "$VALIDATION_TMP" "$VALIDATION"
ln "$LAUNCH_TMP" "$LAUNCH"

# One final independent verification of the now-canonical immutable bindings.
"$PYT" - "$GCQ_RUNS" "$GCQ_DATA" <<'PY'
import sys
from pathlib import Path
from summarize_recovery_vqa_development_round2 import verify_launch_and_inputs
verify_launch_and_inputs(Path(sys.argv[1]), Path(sys.argv[2]), Path.cwd())
PY

mkdir -p "$LOG_DIR"
QSUB_OUTPUT=$(qsub "$BATCH")
printf '%s\n' "$QSUB_OUTPUT"
JOB_ID=$(printf '%s\n' "$QSUB_OUTPUT" | sed -nE 's/.*job-array ([0-9]+).*/\1/p')
if [[ ! "$JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "could not parse submitted round-2 array job ID" >&2
  exit 2
fi
"$PYT" - "$SUBMISSION" "$LAUNCH" "$JOB_ID" "$QSUB_OUTPUT" <<'PY'
import datetime, hashlib, json, sys
from pathlib import Path

output, launch, job_id, qsub_output = sys.argv[1:]
digest = hashlib.sha256(Path(launch).read_bytes()).hexdigest()
record = {
    "schema_version": 1,
    "round_id": "balanced-replay-remaining-checkpoints-development-round-2",
    "submitted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "launch_manifest": launch, "launch_manifest_sha256": digest,
    "job_id": job_id, "array_tasks": 9, "qsub_output": qsub_output,
}
with open(output, "x") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
chmod 0444 "$SUBMISSION"
trap - EXIT
rm -rf -- "$PREFLIGHT_DIR"
echo "ROUND2 SUBMITTED job_id=$JOB_ID tasks=9"
