#!/bin/bash
set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh

PILOT_ROOT="$GCQ_RUNS/recovery_pilot"
BASE_VQA="$PILOT_ROOT/eval/gcq425_lora_ce_s0/gcq425_untrained_recoverypilot_vqa5k.vqa.jsonl"
if [[ ! -s "$BASE_VQA" ]]; then
  echo "fresh untrained-GCQ VQA baseline is incomplete: $BASE_VQA" >&2
  exit 2
fi
for STEP in 100 200 300 400; do
  STEP_PADDED=$(printf '%06d' "$STEP")
  ADAPTER="$PILOT_ROOT/adapters/gcq425_lora_cwce_g5_s0/checkpoint-$STEP_PADDED"
  if [[ ! -s "$ADAPTER/adapter_model.safetensors" ]]; then
    echo "checkpoint tensor is missing: $ADAPTER" >&2
    exit 2
  fi
  if [[ "$(jq -r '.checkpoint_step' "$ADAPTER/gcq_recovery_manifest.json")" != "$STEP" ]]; then
    echo "checkpoint manifest mismatch: $ADAPTER" >&2
    exit 2
  fi
done
if [[ -d "$PILOT_ROOT/checkpoint_sweep" ]] && \
   find "$PILOT_ROOT/checkpoint_sweep" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to mix stale checkpoint-sweep outputs: $PILOT_ROOT/checkpoint_sweep" >&2
  exit 2
fi
mkdir -p "$PILOT_ROOT/checkpoint_sweep" "$PILOT_ROOT/logs"
qsub /usr4/spclpgm/eric1/GCQ/code/batch_recovery_checkpoint_sweep.sh
