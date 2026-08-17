#!/bin/bash
set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh

ROOT="$GCQ_RUNS/recovery_vqa_replay"
RECIPE="gcq425_lora_cwce_vqa50_g5_lr5e5_s0"
VALIDATION="$ROOT/artifact_validation.json"
LAUNCH="$ROOT/development_launch_manifest.json"
DEVELOPMENT="$ROOT/development"
ENV_FILE="/usr4/spclpgm/eric1/GCQ/code/env.sh"
VALIDATOR="/usr4/spclpgm/eric1/GCQ/code/validate_recovery_vqa_replay.py"
PROTOCOL="/usr4/spclpgm/eric1/GCQ/code/recovery_vqa_replay_protocol.json"
PROMOTE="$GCQ_RUNS/promote_gcq_b4.25.json"
REC_DEV="$GCQ_DATA/subsets/recovery_dev_1k.json"
VQA_DEV="$GCQ_DATA/subsets/vqa_val_5k.json"
BASE_DIR="$GCQ_RUNS/recovery_pilot/eval/gcq425_lora_ce_s0"
BASE_REC="$BASE_DIR/gcq425_untrained_recoverydev.rec.jsonl"
BASE_REC_METRICS="$BASE_DIR/gcq425_untrained_recoverydev.rec.metrics.json"
BASE_VQA="$BASE_DIR/gcq425_untrained_recoverypilot_vqa5k.vqa.jsonl"
BASE_VQA_METRICS="$BASE_DIR/gcq425_untrained_recoverypilot_vqa5k.vqa.metrics.json"
W4_REC_METRICS="$GCQ_RUNS/recovery_pilot/eval/w4rtn_lora_cwce_g5_s0/w4rtn_lora_cwce_g5_s0_recoverydev.rec.metrics.json"
BATCH_SCRIPT="/usr4/spclpgm/eric1/GCQ/code/batch_recovery_vqa_development.sh"
SUMMARIZER="/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_vqa_development.py"
SUMMARY_PILOT_SUPPORT="/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_pilot.py"
SUMMARY_CHECKPOINT_SUPPORT="/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_checkpoint_sweep.py"
SUMMARY_SELECTED_SUPPORT="/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_selected_eval.py"
MODEL="Qwen/Qwen3-VL-2B-Instruct"
REVISION="89644892e4d85e24eaac8bacfd4f463576704203"

if [[ -e "$LAUNCH" ]]; then
  echo "refusing to replace frozen development launch manifest: $LAUNCH" >&2
  exit 2
fi

require_hash() {
  local path="$1" expected="$2" actual
  if [[ ! -s "$path" ]]; then
    echo "required frozen development input is missing: $path" >&2
    exit 2
  fi
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "frozen development input changed: $path ($actual != $expected)" >&2
    exit 2
  fi
}

require_hash "$REC_DEV" "e40a1374bdc18c6639615e82dab67cbd0ae9c8d63524b34db4170e159d13b23b"
require_hash "$VQA_DEV" "28091f5c1fa94e28d24b948d2546fd1db06eec41556a6c99f19fbfd163ec0c4d"
require_hash "$PROMOTE" "78b3138da1c70f2f944e3c63d5cfc754f59bf6062c27564d8cc6b2792ba0c1e6"
require_hash "$PROTOCOL" "fd938d0d39116b989ffdcde4dd5ce64bbb419a1292e4ab9f32864416953e5e6d"
require_hash "$BASE_REC" "a976d669237dd004e357b766056d3e4ccd4feba391379422b883c1db0aaeb551"
require_hash "$BASE_REC_METRICS" "a3676be7c615fd3cf07609b70acf2a27c91fd1252c8b7797119aada2b83b67e7"
require_hash "$BASE_VQA" "40eddd91a15890e482cbe9e7cccf81767343a5c46fda03cc54c5317c1f3fe77b"
require_hash "$BASE_VQA_METRICS" "7c174ae672c5b6c630a2413bbc32f77d58f0d579cb64118edae248fcd18c4101"
require_hash "$W4_REC_METRICS" "8545e211da8030ee0114e57356db5ae30811a791574e5848601ad6d2d61ba9b7"
for REQUIRED_CODE in "$ENV_FILE" "$VALIDATOR" "$BATCH_SCRIPT" "$SUMMARIZER" \
  "$SUMMARY_PILOT_SUPPORT" "$SUMMARY_CHECKPOINT_SUPPORT" "$SUMMARY_SELECTED_SUPPORT"; do
  if [[ ! -s "$REQUIRED_CODE" ]]; then
    echo "required development code is missing: $REQUIRED_CODE" >&2
    exit 2
  fi
done

"$PYT" "$VALIDATOR" \
  --require-empty-development
if [[ "$(wc -l < "$BASE_REC")" -ne 1000 ]] || [[ "$(wc -l < "$BASE_VQA")" -ne 5000 ]]; then
  echo "fresh untrained-GCQ development baselines are incomplete" >&2
  exit 2
fi
if ! jq -e 'length == 1000 and ([.[] | select((.task // "rec") == "rec")] | length == 750)' \
    "$REC_DEV" >/dev/null || ! jq -e 'length == 5000' "$VQA_DEV" >/dev/null; then
  echo "frozen development manifests have the wrong evaluation counts" >&2
  exit 2
fi
if ! jq -e --arg model "$MODEL" --arg revision "$REVISION" \
    '.model == $model and .base_revision == $revision and .n == 1000 and
     .by_task.rec.n == 750 and
     .by_task.rec["acc_iou_0.5"] == 0.8053333333333333' \
    "$BASE_REC_METRICS" >/dev/null || \
   ! jq -e '.by_task.rec.n == 750 and
     .by_task.rec["acc_iou_0.5"] == 0.8173333333333334' \
    "$W4_REC_METRICS" >/dev/null || \
   ! jq -e --arg model "$MODEL" --arg revision "$REVISION" \
    '.model == $model and .base_revision == $revision and .n == 5000 and
     .task == "vqa" and ((.accuracy - 0.7737) | fabs) < 1e-12' \
    "$BASE_VQA_METRICS" >/dev/null; then
  echo "frozen development baseline metadata does not match the protocol" >&2
  exit 2
fi

hash_file() { sha256sum "$1" | awk '{print $1}'; }
CHECKPOINTS='{}'
for STEP in 200 300 400 500 600 750; do
  ADAPTER_DIR=$(jq -er --arg step "$STEP" '.checkpoints[$step].directory' "$VALIDATION")
  EXPECTED_DIR="$ROOT/adapters/$RECIPE"
  if [[ "$STEP" != "750" ]]; then
    EXPECTED_DIR="$EXPECTED_DIR/checkpoint-$(printf '%06d' "$STEP")"
  fi
  if [[ "$ADAPTER_DIR" != "$EXPECTED_DIR" ]]; then
    echo "checkpoint $STEP has a noncanonical adapter directory: $ADAPTER_DIR" >&2
    exit 2
  fi
  ADAPTER_SHA=$(jq -er --arg step "$STEP" '.checkpoints[$step].artifact.sha256' "$VALIDATION")
  CONFIG_SHA=$(jq -er --arg step "$STEP" '.checkpoints[$step].adapter_config.sha256' "$VALIDATION")
  MANIFEST_SHA=$(jq -er --arg step "$STEP" '.checkpoints[$step].manifest_sha256' "$VALIDATION")
  CHECKPOINTS=$(jq -c \
    --arg step "$STEP" --arg dir "$ADAPTER_DIR" \
    --arg adapter_sha "$ADAPTER_SHA" --arg config_sha "$CONFIG_SHA" \
    --arg manifest_sha "$MANIFEST_SHA" \
    '. + {($step): {adapter_dir: $dir, adapter_sha256: $adapter_sha,
      adapter_config_sha256: $config_sha, manifest_sha256: $manifest_sha}}' <<<"$CHECKPOINTS")
done

mkdir -p "$ROOT/logs" "$DEVELOPMENT"
LAUNCH_TMP=$(mktemp "$ROOT/.development_launch_manifest.XXXXXX")
trap 'rm -f "$LAUNCH_TMP"' EXIT
jq -n \
  --arg recipe_id "$RECIPE" \
  --arg base_model "$MODEL" \
  --arg base_revision "$REVISION" \
  --argjson candidate_steps '[200,300,400,500,600,750]' \
  --argjson checkpoints "$CHECKPOINTS" \
  --arg validation_path "$VALIDATION" \
  --arg environment_path "$ENV_FILE" \
  --arg protocol_path "$PROTOCOL" \
  --arg promotion_path "$PROMOTE" \
  --arg recovery_dev_path "$REC_DEV" \
  --arg vqa_dev_path "$VQA_DEV" \
  --arg baseline_rec_path "$BASE_REC" \
  --arg baseline_rec_metrics_path "$BASE_REC_METRICS" \
  --arg baseline_vqa_path "$BASE_VQA" \
  --arg baseline_vqa_metrics_path "$BASE_VQA_METRICS" \
  --arg w4_rec_metrics_path "$W4_REC_METRICS" \
  --arg validation "$(hash_file "$VALIDATION")" \
  --arg environment "$(hash_file "$ENV_FILE")" \
  --arg artifact_validator "$(hash_file "$VALIDATOR")" \
  --arg protocol "$(hash_file "$PROTOCOL")" \
  --arg eval_rec "$(hash_file /usr4/spclpgm/eric1/GCQ/code/eval_rec.py)" \
  --arg eval_vqa "$(hash_file /usr4/spclpgm/eric1/GCQ/code/eval_vqa.py)" \
  --arg recovery_utils "$(hash_file /usr4/spclpgm/eric1/GCQ/code/recovery_utils.py)" \
  --arg quant_utils "$(hash_file /usr4/spclpgm/eric1/GCQ/code/quant_utils.py)" \
  --arg gcq_patches "$(hash_file /usr4/spclpgm/eric1/GCQ/code/gcq_patches.py)" \
  --arg batch_script "$(hash_file /usr4/spclpgm/eric1/GCQ/code/batch_recovery_vqa_development.sh)" \
  --arg promotion "$(hash_file "$PROMOTE")" \
  --arg recovery_dev "$(hash_file "$REC_DEV")" \
  --arg vqa_dev "$(hash_file "$VQA_DEV")" \
  --arg baseline_rec "$(hash_file "$BASE_REC")" \
  --arg baseline_rec_metrics "$(hash_file "$BASE_REC_METRICS")" \
  --arg baseline_vqa "$(hash_file "$BASE_VQA")" \
  --arg baseline_vqa_metrics "$(hash_file "$BASE_VQA_METRICS")" \
  --arg w4_rec_metrics "$(hash_file "$W4_REC_METRICS")" \
  --arg development_summarizer "$(hash_file "$SUMMARIZER")" \
  --arg summary_pilot_support "$(hash_file "$SUMMARY_PILOT_SUPPORT")" \
  --arg summary_checkpoint_support "$(hash_file "$SUMMARY_CHECKPOINT_SUPPORT")" \
  --arg summary_selected_support "$(hash_file "$SUMMARY_SELECTED_SUPPORT")" \
  '{schema_version: 1, recipe_id: $recipe_id, base_model: $base_model,
    base_revision: $base_revision, candidate_steps: $candidate_steps,
    evaluation: {grounding_subset: "recovery_dev_1k", grounding_examples: 1000,
      primary_task: "rec", primary_examples: 750, vqa_examples: 5000},
    checkpoints: $checkpoints,
    paths: {validation: $validation_path, environment: $environment_path,
      protocol: $protocol_path,
      promotion: $promotion_path, recovery_dev: $recovery_dev_path,
      vqa_dev: $vqa_dev_path, baseline_rec: $baseline_rec_path,
      baseline_rec_metrics: $baseline_rec_metrics_path,
      baseline_vqa: $baseline_vqa_path,
      baseline_vqa_metrics: $baseline_vqa_metrics_path,
      w4_rec_metrics: $w4_rec_metrics_path},
    hashes: {validation: $validation, environment: $environment,
      artifact_validator: $artifact_validator,
      protocol: $protocol,
      eval_rec: $eval_rec, eval_vqa: $eval_vqa, recovery_utils: $recovery_utils,
      quant_utils: $quant_utils, gcq_patches: $gcq_patches,
      batch_script: $batch_script, promotion: $promotion,
      recovery_dev: $recovery_dev, vqa_dev: $vqa_dev,
      baseline_rec: $baseline_rec, baseline_rec_metrics: $baseline_rec_metrics,
      baseline_vqa: $baseline_vqa, baseline_vqa_metrics: $baseline_vqa_metrics,
      w4_rec_metrics: $w4_rec_metrics,
      development_summarizer: $development_summarizer,
      summary_pilot_support: $summary_pilot_support,
      summary_checkpoint_support: $summary_checkpoint_support,
      summary_selected_support: $summary_selected_support}}' \
  > "$LAUNCH_TMP"
chmod 0444 "$LAUNCH_TMP"
if ! ln "$LAUNCH_TMP" "$LAUNCH"; then
  echo "another process froze the development launch manifest first: $LAUNCH" >&2
  exit 2
fi
rm -f "$LAUNCH_TMP"
trap - EXIT

qsub "$BATCH_SCRIPT"
