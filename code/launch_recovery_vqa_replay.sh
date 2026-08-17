#!/bin/bash
set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh

MODE="${1:-smoke}"
ROOT="$GCQ_RUNS/recovery_vqa_replay"
TAG="gcq425_lora_cwce_vqa50_g5_lr5e5_s0"
DATA="$GCQ_DATA/subsets/recovery_train_vqa_replay_12k.json"
META="$GCQ_DATA/subsets/recovery_train_vqa_replay_12k.meta.json"
PROMOTE="$GCQ_RUNS/promote_gcq_b4.25.json"
PROTOCOL="/usr4/spclpgm/eric1/GCQ/code/recovery_vqa_replay_protocol.json"
EXPECTED_DATA_SHA="8bf3b6a1589527f5847ea28a7c5f0daeb89f6e0d7fa220451db87c52314aec4a"
EXPECTED_META_SHA="5e11f95ae6baebe0d92ea384e80f78bf6e0929003769e8e4d08767d85346b375"
EXPECTED_PROMOTE_SHA="78b3138da1c70f2f944e3c63d5cfc754f59bf6062c27564d8cc6b2792ba0c1e6"
EXPECTED_PROTOCOL_SHA="fd938d0d39116b989ffdcde4dd5ce64bbb419a1292e4ab9f32864416953e5e6d"

require_hash() {
  local path="$1" expected="$2" actual
  if [[ ! -s "$path" ]]; then
    echo "required frozen file is missing: $path" >&2
    exit 2
  fi
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "frozen-file hash mismatch: $path ($actual != $expected)" >&2
    exit 2
  fi
}

require_hash "$DATA" "$EXPECTED_DATA_SHA"
require_hash "$META" "$EXPECTED_META_SHA"
require_hash "$PROMOTE" "$EXPECTED_PROMOTE_SHA"
require_hash "$PROTOCOL" "$EXPECTED_PROTOCOL_SHA"
if [[ "$(jq -r '.output.examples' "$META")" != "12000" ]] || \
   [[ "$(jq -r '.output.grounding_examples' "$META")" != "6000" ]] || \
   [[ "$(jq -r '.output.vqa_examples' "$META")" != "6000" ]] || \
   [[ "$(jq -r '.output.caption_examples' "$META")" != "0" ]]; then
  echo "unexpected balanced-replay manifest composition" >&2
  exit 2
fi

mkdir -p "$ROOT/logs"
case "$MODE" in
  smoke)
    OUTPUT="$ROOT/smoke/$TAG"
    if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -print -quit | grep -q .; then
      echo "refusing to overwrite smoke output: $OUTPUT" >&2
      exit 2
    fi
    qsub /usr4/spclpgm/eric1/GCQ/code/batch_recovery_vqa_replay_smoke.sh
    ;;
  full)
    SMOKE="$ROOT/smoke/$TAG/gcq_recovery_manifest.json"
    if [[ ! -s "$SMOKE" ]] || \
       [[ "$(jq -r '.completed.optimizer_steps' "$SMOKE")" != "1" ]] || \
       [[ "$(jq -r '.verification.zero_init_max_logit_diff' "$SMOKE")" != "0.0" ]] || \
       [[ "$(jq -r '.verification.first_step_nonzero_lora_grad' "$SMOKE")" != "true" ]] || \
       [[ "$(jq -r '.verification.base_parameters_received_grad' "$SMOKE")" != "false" ]] || \
       [[ "$(jq -r '.data.sha256' "$SMOKE")" != "$EXPECTED_DATA_SHA" ]] || \
       [[ "$(jq -c '.lora.runtime_training_dtypes' "$SMOKE")" != '["torch.float32"]' ]]; then
      echo "the balanced-replay smoke run is missing or did not pass" >&2
      exit 2
    fi
    OUTPUT="$ROOT/adapters/$TAG"
    if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -print -quit | grep -q .; then
      echo "refusing to overwrite full training output: $OUTPUT" >&2
      exit 2
    fi
    LAUNCH_MANIFEST="$ROOT/training_launch_manifest.json"
    if [[ -e "$LAUNCH_MANIFEST" ]]; then
      echo "refusing to replace frozen training launch manifest: $LAUNCH_MANIFEST" >&2
      exit 2
    fi
    TRAINER="/usr4/spclpgm/eric1/GCQ/code/train_recovery.py"
    BATCH_SCRIPT="/usr4/spclpgm/eric1/GCQ/code/batch_recovery_vqa_replay.sh"
    TRAINER_SHA=$(sha256sum "$TRAINER" | awk '{print $1}')
    BATCH_SHA=$(sha256sum "$BATCH_SCRIPT" | awk '{print $1}')
    jq -n \
      --arg recipe_id "$TAG" \
      --arg data_sha256 "$EXPECTED_DATA_SHA" \
      --arg metadata_sha256 "$EXPECTED_META_SHA" \
      --arg promotion_sha256 "$EXPECTED_PROMOTE_SHA" \
      --arg protocol_sha256 "$EXPECTED_PROTOCOL_SHA" \
      --arg trainer_sha256 "$TRAINER_SHA" \
      --arg batch_script_sha256 "$BATCH_SHA" \
      '{schema_version: 1, recipe_id: $recipe_id,
        data_sha256: $data_sha256, metadata_sha256: $metadata_sha256,
        promotion_sha256: $promotion_sha256, protocol_sha256: $protocol_sha256,
        trainer_sha256: $trainer_sha256, batch_script_sha256: $batch_script_sha256,
        objective: "cwce", coordinate_weight: 5, learning_rate: 0.00005,
        effective_batch_size: 16, epochs: 1, planned_optimizer_steps: 750,
        checkpoint_steps: [200, 300, 400, 500, 600, 750], seed: 0}' \
      > "$LAUNCH_MANIFEST"
    chmod 0444 "$LAUNCH_MANIFEST"
    qsub /usr4/spclpgm/eric1/GCQ/code/batch_recovery_vqa_replay.sh
    ;;
  *)
    echo "usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac
