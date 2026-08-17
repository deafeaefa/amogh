#!/bin/bash
set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh

PILOT_ROOT="$GCQ_RUNS/recovery_pilot"
SUMMARY="$PILOT_ROOT/checkpoint_sweep_summary.json"
OUTPUT="$PILOT_ROOT/selected_eval"
BASE_DIR="$PILOT_ROOT/eval/gcq425_lora_ce_s0"
PROMOTE="$GCQ_RUNS/promote_gcq_b4.25.json"
FROZEN_VQA="$GCQ_DATA/subsets/vqa_val_5k.json"
POPE_RANDOM="$GCQ_DATA/pope/coco_pope_random.json"
POPE_POPULAR="$GCQ_DATA/pope/coco_pope_popular.json"
POPE_ADVERSARIAL="$GCQ_DATA/pope/coco_pope_adversarial.json"
REVISION="89644892e4d85e24eaac8bacfd4f463576704203"

if [[ ! -s "$SUMMARY" ]] || [[ "$(jq -r '.selection_succeeded' "$SUMMARY")" != "true" ]]; then
  echo "a successful frozen checkpoint selection is required: $SUMMARY" >&2
  exit 2
fi
STEP=$(jq -er '.selected.step' "$SUMMARY")
ADAPTER=$(jq -er '.selected.adapter_dir' "$SUMMARY")
if [[ "$STEP" != "300" ]]; then
  echo "refusing to expose holdout to non-frozen checkpoint step $STEP" >&2
  exit 2
fi
if [[ "$(jq -r '.selected.scores.eligible' "$SUMMARY")" != "true" ]]; then
  echo "frozen checkpoint step 300 is not marked eligible" >&2
  exit 2
fi
EXPECTED_ADAPTER="$PILOT_ROOT/adapters/gcq425_lora_cwce_g5_s0/checkpoint-000300"
if [[ "$ADAPTER" != "$EXPECTED_ADAPTER" ]] || [[ ! -s "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "selected adapter is missing or noncanonical: $ADAPTER" >&2
  exit 2
fi
ADAPTER_MANIFEST="$ADAPTER/gcq_recovery_manifest.json"
if [[ "$(jq -r '.checkpoint_step' "$ADAPTER_MANIFEST")" != "300" ]]; then
  echo "selected adapter manifest is not checkpoint 300: $ADAPTER_MANIFEST" >&2
  exit 2
fi
BASE_VQA="$BASE_DIR/gcq425_untrained_recoverypilot_vqa5k.vqa.jsonl"
BASE_POPE="$BASE_DIR/gcq425_untrained_recoverypilot_pope.pope.jsonl"
if [[ "$(wc -l < "$BASE_VQA")" -ne 5000 ]]; then
  echo "fresh untrained-GCQ VQA baseline must contain exactly 5000 rows" >&2
  exit 2
fi
if [[ "$(wc -l < "$BASE_POPE")" -ne 9000 ]]; then
  echo "fresh untrained-GCQ POPE baseline must contain exactly 9000 rows" >&2
  exit 2
fi
for REQUIRED in "$PROMOTE" "$FROZEN_VQA" "$POPE_RANDOM" "$POPE_POPULAR" "$POPE_ADVERSARIAL"; do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "required frozen input is missing: $REQUIRED" >&2
    exit 2
  fi
done
if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to expose a second checkpoint or mix stale holdout outputs: $OUTPUT" >&2
  exit 2
fi
mkdir -p "$OUTPUT" "$PILOT_ROOT/logs"
SUMMARY_SHA=$(sha256sum "$SUMMARY" | awk '{print $1}')
ADAPTER_SHA=$(sha256sum "$ADAPTER/adapter_model.safetensors" | awk '{print $1}')
ADAPTER_MANIFEST_SHA=$(sha256sum "$ADAPTER_MANIFEST" | awk '{print $1}')
PROMOTE_SHA=$(sha256sum "$PROMOTE" | awk '{print $1}')
BASE_VQA_SHA=$(sha256sum "$BASE_VQA" | awk '{print $1}')
BASE_POPE_SHA=$(sha256sum "$BASE_POPE" | awk '{print $1}')
FROZEN_VQA_SHA=$(sha256sum "$FROZEN_VQA" | awk '{print $1}')
POPE_RANDOM_SHA=$(sha256sum "$POPE_RANDOM" | awk '{print $1}')
POPE_POPULAR_SHA=$(sha256sum "$POPE_POPULAR" | awk '{print $1}')
POPE_ADVERSARIAL_SHA=$(sha256sum "$POPE_ADVERSARIAL" | awk '{print $1}')
jq -n \
  --argjson selected_step "$STEP" \
  --arg adapter_dir "$ADAPTER" \
  --arg base_revision "$REVISION" \
  --arg checkpoint_summary "$SUMMARY_SHA" \
  --arg adapter "$ADAPTER_SHA" \
  --arg adapter_manifest "$ADAPTER_MANIFEST_SHA" \
  --arg promotion "$PROMOTE_SHA" \
  --arg baseline_vqa "$BASE_VQA_SHA" \
  --arg baseline_pope "$BASE_POPE_SHA" \
  --arg frozen_vqa "$FROZEN_VQA_SHA" \
  --arg pope_random "$POPE_RANDOM_SHA" \
  --arg pope_popular "$POPE_POPULAR_SHA" \
  --arg pope_adversarial "$POPE_ADVERSARIAL_SHA" \
  '{schema_version: 1, selected_step: $selected_step, adapter_dir: $adapter_dir,
    base_revision: $base_revision,
    hashes: {checkpoint_summary: $checkpoint_summary, adapter: $adapter,
      adapter_manifest: $adapter_manifest, promotion: $promotion,
      baseline_vqa: $baseline_vqa, baseline_pope: $baseline_pope,
      frozen_vqa: $frozen_vqa, pope_random: $pope_random,
      pope_popular: $pope_popular, pope_adversarial: $pope_adversarial},
    vqa_slice: {start: 1000, count: 4000}, pope_count: 9000}' \
  > "$OUTPUT/selection_launch_manifest.json"
chmod 0444 "$OUTPUT/selection_launch_manifest.json"

qsub /usr4/spclpgm/eric1/GCQ/code/batch_recovery_selected_eval.sh
