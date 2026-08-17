#!/bin/bash -l
#$ -P rise-tower
#$ -N gcq-rec-final
#$ -q l40s
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_m=40G
#$ -l h_rt=03:00:00
#$ -pe omp 4
#$ -j y
#$ -o /projectnb/rise-tower/eric1/GCQ/runs/recovery_pilot/logs

set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh
cd /usr4/spclpgm/eric1/GCQ/code

PILOT_ROOT="$GCQ_RUNS/recovery_pilot"
SUMMARY="$PILOT_ROOT/checkpoint_sweep_summary.json"
OUTPUT="$PILOT_ROOT/selected_eval"
LAUNCH="$OUTPUT/selection_launch_manifest.json"
PROMOTE="$GCQ_RUNS/promote_gcq_b4.25.json"
REVISION="89644892e4d85e24eaac8bacfd4f463576704203"
BASE_DIR="$PILOT_ROOT/eval/gcq425_lora_ce_s0"
BASE_VQA="$BASE_DIR/gcq425_untrained_recoverypilot_vqa5k.vqa.jsonl"
BASE_POPE="$BASE_DIR/gcq425_untrained_recoverypilot_pope.pope.jsonl"
FROZEN_VQA="$GCQ_DATA/subsets/vqa_val_5k.json"
POPE_RANDOM="$GCQ_DATA/pope/coco_pope_random.json"
POPE_POPULAR="$GCQ_DATA/pope/coco_pope_popular.json"
POPE_ADVERSARIAL="$GCQ_DATA/pope/coco_pope_adversarial.json"

if [[ ! -s "$LAUNCH" ]]; then
  echo "selection launch manifest is missing: $LAUNCH" >&2
  exit 2
fi
STEP=$(jq -er '.selected_step' "$LAUNCH")
ADAPTER=$(jq -er '.adapter_dir' "$LAUNCH")
if [[ "$STEP" != "300" ]]; then
  echo "refusing to expose holdout to non-frozen checkpoint step $STEP" >&2
  exit 2
fi
EXPECTED_ADAPTER="$PILOT_ROOT/adapters/gcq425_lora_cwce_g5_s0/checkpoint-000300"
TAG="gcq425_cwce_step${STEP}_selected"

if [[ "$(jq -r '.selection_succeeded' "$SUMMARY")" != "true" ]] || \
   [[ "$(jq -r '.selected.step' "$SUMMARY")" != "300" ]] || \
   [[ "$(jq -r '.selected.adapter_dir' "$SUMMARY")" != "$EXPECTED_ADAPTER" ]] || \
   [[ "$(jq -r '.selected.scores.eligible' "$SUMMARY")" != "true" ]]; then
  echo "checkpoint selection did not succeed: $SUMMARY" >&2
  exit 2
fi
if [[ "$ADAPTER" != "$EXPECTED_ADAPTER" ]]; then
  echo "selected adapter path is not canonical: $ADAPTER" >&2
  exit 2
fi
if [[ "$(jq -r '.checkpoint_step' "$ADAPTER/gcq_recovery_manifest.json")" != "$STEP" ]]; then
  echo "selected checkpoint manifest mismatch: $ADAPTER" >&2
  exit 2
fi
if [[ ! -s "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "selected checkpoint tensor is missing: $ADAPTER" >&2
  exit 2
fi
if [[ "$(jq -r '.base_revision' "$LAUNCH")" != "$REVISION" ]] || \
   [[ "$(jq -r '.vqa_slice.start' "$LAUNCH")" != "1000" ]] || \
   [[ "$(jq -r '.vqa_slice.count' "$LAUNCH")" != "4000" ]] || \
   [[ "$(jq -r '.pope_count' "$LAUNCH")" != "9000" ]]; then
  echo "selection launch protocol does not match the frozen holdout" >&2
  exit 2
fi

check_hash() {
  local key="$1"
  local path="$2"
  local expected actual
  expected=$(jq -er --arg key "$key" '.hashes[$key]' "$LAUNCH")
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "hash mismatch for $key: $path" >&2
    exit 2
  fi
}

check_hash checkpoint_summary "$SUMMARY"
check_hash adapter "$ADAPTER/adapter_model.safetensors"
check_hash adapter_manifest "$ADAPTER/gcq_recovery_manifest.json"
check_hash promotion "$PROMOTE"
check_hash baseline_vqa "$BASE_VQA"
check_hash baseline_pope "$BASE_POPE"
check_hash frozen_vqa "$FROZEN_VQA"
check_hash pope_random "$POPE_RANDOM"
check_hash pope_popular "$POPE_POPULAR"
check_hash pope_adversarial "$POPE_ADVERSARIAL"

for TARGET in \
  "$OUTPUT/${TAG}_vqa_holdout4k.vqa.jsonl" \
  "$OUTPUT/${TAG}_vqa_holdout4k.vqa.metrics.json" \
  "$OUTPUT/${TAG}_pope_full.pope.jsonl" \
  "$OUTPUT/${TAG}_pope_full.pope.metrics.json" \
  "$OUTPUT/results.csv"; do
  if [[ -e "$TARGET" ]]; then
    echo "refusing to overwrite selected-evaluation output: $TARGET" >&2
    exit 2
  fi
done

export GCQ_RUNS="$OUTPUT"

"$PYT" eval_vqa.py --model Qwen/Qwen3-VL-2B-Instruct --revision "$REVISION" --task vqa \
  --tag "${TAG}_vqa_holdout4k" --start 1000 --limit 4000 \
  --rtn-bits 4 --promote-file "$PROMOTE" --adapter-dir "$ADAPTER" \
  --batch 24 --device cuda:0

"$PYT" eval_vqa.py --model Qwen/Qwen3-VL-2B-Instruct --revision "$REVISION" --task pope \
  --tag "${TAG}_pope_full" \
  --rtn-bits 4 --promote-file "$PROMOTE" --adapter-dir "$ADAPTER" \
  --batch 24 --device cuda:0
