#!/bin/bash
set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh

ROOT="$GCQ_RUNS/recovery_vqa_replay"
CONFIRMATION="$ROOT/confirmation"
LAUNCH="$CONFIRMATION/confirmation_launch_manifest.json"
SUMMARY="$ROOT/vqa_confirmation_summary.json"
RECIPE="gcq425_lora_cwce_vqa50_g5_lr5e5_s0"
MODEL="Qwen/Qwen3-VL-2B-Instruct"
REVISION="89644892e4d85e24eaac8bacfd4f463576704203"
BASE_TAG="gcq425_untrained_vqa_fresh5k"
MINIMUM_STORAGE_HEADROOM_BYTES=1073741824

DEV_SUMMARY="$ROOT/development_summary.json"
DEV_LAUNCH="$ROOT/development_launch_manifest.json"
ARTIFACT_VALIDATION="$ROOT/artifact_validation.json"
TRAINING_LAUNCH="$ROOT/training_launch_manifest.json"
PROTOCOL="/usr4/spclpgm/eric1/GCQ/code/recovery_vqa_replay_protocol.json"
PROMOTION="$GCQ_RUNS/promote_gcq_b4.25.json"
FRESH_VQA="$GCQ_DATA/subsets/vqa_fresh_confirm_5k.json"
FRESH_META="$GCQ_DATA/subsets/vqa_fresh_confirm_5k.meta.json"

LAUNCHER="/usr4/spclpgm/eric1/GCQ/code/launch_recovery_vqa_confirmation.sh"
BATCH_SCRIPT="/usr4/spclpgm/eric1/GCQ/code/batch_recovery_vqa_confirmation.sh"
SUMMARIZER="/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_vqa_confirmation.py"
DEV_SUMMARIZER="/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_vqa_development.py"
ARTIFACT_VALIDATOR="/usr4/spclpgm/eric1/GCQ/code/validate_recovery_vqa_replay.py"
EVAL_VQA="/usr4/spclpgm/eric1/GCQ/code/eval_vqa.py"
RECOVERY_UTILS="/usr4/spclpgm/eric1/GCQ/code/recovery_utils.py"
QUANT_UTILS="/usr4/spclpgm/eric1/GCQ/code/quant_utils.py"
GCQ_PATCHES="/usr4/spclpgm/eric1/GCQ/code/gcq_patches.py"
ENV_FILE="/usr4/spclpgm/eric1/GCQ/code/env.sh"
VQA_HELPER="/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_checkpoint_sweep.py"
PAIRING_HELPER="/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_selected_eval.py"
RECOVERY_ANALYSIS_HELPER="/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_pilot.py"

FRESH_VQA_SHA="416aea5f9dd6f4a4cfa7061c3b5ca0af88647e49d5f17ed00a8d83885ea21038"
FRESH_META_SHA="a7777a0199f7fb432deeee94d478ca092474dd8a5e47cffe92a93d62b57601e8"
PROTOCOL_SHA="fd938d0d39116b989ffdcde4dd5ce64bbb419a1292e4ab9f32864416953e5e6d"
PROMOTION_SHA="78b3138da1c70f2f944e3c63d5cfc754f59bf6062c27564d8cc6b2792ba0c1e6"

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

require_hash() {
  local path="$1" expected="$2" actual
  require_file "$path"
  actual=$(hash_file "$path")
  [[ "$actual" == "$expected" ]] || \
    die "frozen confirmation input changed: $path ($actual != $expected)"
}

scan_for_prior_predictions() {
  local candidate_tag="$1"
  "$PYT" "$SUMMARIZER" --audit-pristine --runs "$GCQ_RUNS" \
    --fresh-manifest "$FRESH_VQA" --baseline-tag "$BASE_TAG" \
    --candidate-tag "$candidate_tag"
}

storage_headroom() {
  "$PYT" - "$ROOT" "$HF_HOME" "$MINIMUM_STORAGE_HEADROOM_BYTES" <<'PY'
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
    "launcher_probes": probes,
    "recheck_at_job_start": True,
}, sort_keys=True))
PY
}

for REQUIRED in \
  "$DEV_SUMMARY" "$DEV_LAUNCH" "$ARTIFACT_VALIDATION" "$TRAINING_LAUNCH" \
  "$PROTOCOL" "$PROMOTION" "$FRESH_VQA" "$FRESH_META" \
  "$LAUNCHER" "$BATCH_SCRIPT" "$SUMMARIZER" "$DEV_SUMMARIZER" \
  "$ARTIFACT_VALIDATOR" "$EVAL_VQA" "$RECOVERY_UTILS" "$QUANT_UTILS" \
  "$GCQ_PATCHES" "$ENV_FILE" "$VQA_HELPER" "$PAIRING_HELPER" \
  "$RECOVERY_ANALYSIS_HELPER"; do
  require_file "$REQUIRED"
done
require_hash "$FRESH_VQA" "$FRESH_VQA_SHA"
require_hash "$FRESH_META" "$FRESH_META_SHA"
require_hash "$PROTOCOL" "$PROTOCOL_SHA"
require_hash "$PROMOTION" "$PROMOTION_SHA"

if [[ -e "$SUMMARY" ]]; then
  die "refusing to replace an existing VQA confirmation summary: $SUMMARY"
fi
if [[ -e "$CONFIRMATION" && ! -d "$CONFIRMATION" ]]; then
  die "confirmation path exists but is not a directory: $CONFIRMATION"
fi
if [[ -d "$CONFIRMATION" ]] && find "$CONFIRMATION" -mindepth 1 -print -quit | grep -q .; then
  die "confirmation directory is not pristine; no second launch is allowed: $CONFIRMATION"
fi

if ! jq -e --arg recipe "$RECIPE" --arg model "$MODEL" --arg revision "$REVISION" '
    .schema_version == 1 and .recipe_id == $recipe and
    .base_model == $model and .base_revision == $revision and
    .selection_succeeded == true and .selected != null and
    .selected.recipe_id == $recipe and .selected.eligible == true and
    (.selected.step | type) == "number" and
    (.selected.adapter_dir | type) == "string" and
    (.selected.adapter_sha256 | type) == "string" and
    (.selected.adapter_config_sha256 | type) == "string" and
    (.selected.manifest_sha256 | type) == "string"' "$DEV_SUMMARY" >/dev/null; then
  die "development selection did not authorize a fresh confirmation: $DEV_SUMMARY"
fi

STEP=$(jq -er '.selected.step' "$DEV_SUMMARY")
ADAPTER_DIR=$(jq -er '.selected.adapter_dir' "$DEV_SUMMARY")
ADAPTER_SHA=$(jq -er '.selected.adapter_sha256' "$DEV_SUMMARY")
ADAPTER_CONFIG_SHA=$(jq -er '.selected.adapter_config_sha256' "$DEV_SUMMARY")
ADAPTER_MANIFEST_SHA=$(jq -er '.selected.manifest_sha256' "$DEV_SUMMARY")
CANDIDATE_TAG="vqa50_step${STEP}_vqa_fresh5k"

case "$STEP" in
  200|300|400|500|600)
    EXPECTED_ADAPTER_DIR="$ROOT/adapters/$RECIPE/checkpoint-$(printf '%06d' "$STEP")"
    ;;
  750)
    EXPECTED_ADAPTER_DIR="$ROOT/adapters/$RECIPE"
    ;;
  *)
    die "development selected a step outside the frozen candidate set: $STEP"
    ;;
esac
[[ "$ADAPTER_DIR" == "$EXPECTED_ADAPTER_DIR" ]] || \
  die "development selected a noncanonical adapter path: $ADAPTER_DIR"
ADAPTER_TENSOR="$ADAPTER_DIR/adapter_model.safetensors"
ADAPTER_CONFIG="$ADAPTER_DIR/adapter_config.json"
ADAPTER_MANIFEST="$ADAPTER_DIR/gcq_recovery_manifest.json"
require_hash "$ADAPTER_TENSOR" "$ADAPTER_SHA"
require_hash "$ADAPTER_CONFIG" "$ADAPTER_CONFIG_SHA"
require_hash "$ADAPTER_MANIFEST" "$ADAPTER_MANIFEST_SHA"

if ! jq -e --arg recipe "$RECIPE" --arg model "$MODEL" --arg revision "$REVISION" \
    --arg step "$STEP" --arg dir "$ADAPTER_DIR" --arg adapter "$ADAPTER_SHA" \
    --arg config "$ADAPTER_CONFIG_SHA" --arg manifest "$ADAPTER_MANIFEST_SHA" '
    .schema_version == 1 and .recipe_id == $recipe and
    .base_model == $model and .base_revision == $revision and
    (.candidate_steps | index($step | tonumber)) != null and
    .checkpoints[$step].adapter_dir == $dir and
    .checkpoints[$step].adapter_sha256 == $adapter and
    .checkpoints[$step].adapter_config_sha256 == $config and
    .checkpoints[$step].manifest_sha256 == $manifest' "$DEV_LAUNCH" >/dev/null; then
  die "selected candidate disagrees with the frozen development launch"
fi
if [[ "$(jq -er '.development_launch_manifest_sha256' "$DEV_SUMMARY")" != \
      "$(hash_file "$DEV_LAUNCH")" ]]; then
  die "development summary does not bind the current development launch"
fi
if ! jq -e --arg recipe "$RECIPE" --arg revision "$REVISION" --arg step "$STEP" \
    --arg dir "$ADAPTER_DIR" --arg adapter "$ADAPTER_SHA" \
    --arg config "$ADAPTER_CONFIG_SHA" --arg manifest "$ADAPTER_MANIFEST_SHA" '
    .schema_version == 1 and .recipe_id == $recipe and .base_revision == $revision and
    .checkpoints[$step].directory == $dir and
    .checkpoints[$step].artifact.sha256 == $adapter and
    .checkpoints[$step].adapter_config.sha256 == $config and
    .checkpoints[$step].manifest_sha256 == $manifest' "$ARTIFACT_VALIDATION" >/dev/null; then
  die "selected candidate disagrees with the frozen artifact validation"
fi
if ! jq -e --arg recipe "$RECIPE" --arg protocol "$PROTOCOL_SHA" \
    --arg promotion "$PROMOTION_SHA" '
    .schema_version == 1 and .recipe_id == $recipe and
    .protocol_sha256 == $protocol and .promotion_sha256 == $promotion and
    .checkpoint_steps == [200,300,400,500,600,750]' "$TRAINING_LAUNCH" >/dev/null; then
  die "training launch does not match the frozen confirmation recipe"
fi
if ! jq -e --arg recipe "$RECIPE" --arg model "$MODEL" --arg revision "$REVISION" \
    --arg fresh "$FRESH_VQA_SHA" '
    .schema_version == 1 and .recipe_id == $recipe and
    .base_model == $model and .base_revision == $revision and
    .confirmation.vqa_manifest == "vqa_fresh_confirm_5k.json" and
    .confirmation.vqa_manifest_sha256 == $fresh and
    .confirmation.questions == 5000 and .confirmation.unique_images == 4571 and
    .confirmation.allowed_trained_candidates == 1 and
    .confirmation.paired_image_bootstrap_resamples == 10000 and
    .confirmation.vqa_paired_ci95_lower_bound_min == -0.015' "$PROTOCOL" >/dev/null; then
  die "confirmation contract in the frozen protocol changed"
fi
if ! jq -e --arg manifest "$FRESH_VQA_SHA" '
    .schema_version == 1 and .output.manifest_sha256 == $manifest and
    .output.questions == 5000 and .output.unique_images == 4571 and
    .output.ordered_question_ids_sha256 ==
      "238e349350af36cd22a3c251d7c71ceda0152d20399ebf361c4373823ce2e383" and
    .output.sorted_unique_image_ids_sha256 ==
      "1c2b9c1b25e358d3c3741cac975d512971d9a5dd2d6c449e5cc8d0ef07de7618"' \
    "$FRESH_META" >/dev/null; then
  die "fresh-confirmation metadata changed"
fi

DEV_SUMMARIZER_SHA=$(hash_file "$DEV_SUMMARIZER")
ARTIFACT_VALIDATOR_SHA=$(hash_file "$ARTIFACT_VALIDATOR")
EVAL_VQA_SHA=$(hash_file "$EVAL_VQA")
RECOVERY_UTILS_SHA=$(hash_file "$RECOVERY_UTILS")
QUANT_UTILS_SHA=$(hash_file "$QUANT_UTILS")
GCQ_PATCHES_SHA=$(hash_file "$GCQ_PATCHES")
ENV_SHA=$(hash_file "$ENV_FILE")
VQA_HELPER_SHA=$(hash_file "$VQA_HELPER")
PAIRING_HELPER_SHA=$(hash_file "$PAIRING_HELPER")
RECOVERY_ANALYSIS_HELPER_SHA=$(hash_file "$RECOVERY_ANALYSIS_HELPER")
ARTIFACT_VALIDATION_SHA=$(hash_file "$ARTIFACT_VALIDATION")
if ! jq -e \
    --arg validation "$ARTIFACT_VALIDATION_SHA" \
    --arg artifact_validator "$ARTIFACT_VALIDATOR_SHA" \
    --arg protocol "$PROTOCOL_SHA" --arg promotion "$PROMOTION_SHA" \
    --arg environment "$ENV_SHA" --arg eval_vqa "$EVAL_VQA_SHA" \
    --arg recovery_utils "$RECOVERY_UTILS_SHA" --arg quant_utils "$QUANT_UTILS_SHA" \
    --arg gcq_patches "$GCQ_PATCHES_SHA" \
    --arg development_summarizer "$DEV_SUMMARIZER_SHA" \
    --arg pilot "$RECOVERY_ANALYSIS_HELPER_SHA" --arg checkpoint "$VQA_HELPER_SHA" \
    --arg selected "$PAIRING_HELPER_SHA" '
    .hashes.validation == $validation and
    .hashes.artifact_validator == $artifact_validator and
    .hashes.protocol == $protocol and .hashes.promotion == $promotion and
    .hashes.environment == $environment and .hashes.eval_vqa == $eval_vqa and
    .hashes.recovery_utils == $recovery_utils and .hashes.quant_utils == $quant_utils and
    .hashes.gcq_patches == $gcq_patches and
    .hashes.development_summarizer == $development_summarizer and
    .hashes.summary_pilot_support == $pilot and
    .hashes.summary_checkpoint_support == $checkpoint and
    .hashes.summary_selected_support == $selected' "$DEV_LAUNCH" >/dev/null; then
  die "development evidence differs from the frozen development launch"
fi

VALIDATED_SELECTED=$("$PYT" "$SUMMARIZER" --validate-development-selection \
  --runs "$GCQ_RUNS")
if ! jq -e --argjson step "$STEP" --arg recipe "$RECIPE" \
    --arg dir "$ADAPTER_DIR" --arg adapter "$ADAPTER_SHA" \
    --arg config "$ADAPTER_CONFIG_SHA" --arg manifest "$ADAPTER_MANIFEST_SHA" '
    .step == $step and .recipe_id == $recipe and .adapter_dir == $dir and
    .adapter_sha256 == $adapter and .adapter_config_sha256 == $config and
    .manifest_sha256 == $manifest' <<<"$VALIDATED_SELECTED" >/dev/null; then
  die "recomputed development winner differs from the claimed selected candidate"
fi
chmod 0444 "$DEV_SUMMARY"

NO_PEEK_AUDIT=$(scan_for_prior_predictions "$CANDIDATE_TAG")
NO_PEEK_AUDIT=$(jq -c '. + {checked_before_launch: true, recheck_at_job_start: true}' \
  <<<"$NO_PEEK_AUDIT")
IMAGE_INVENTORY=$("$PYT" "$SUMMARIZER" --compute-image-inventory \
  --fresh-manifest "$FRESH_VQA" --image-dir "$GCQ_DATA/images/val2014")
[[ "$(jq -er '.images' <<<"$IMAGE_INVENTORY")" -eq 4571 ]] || \
  die "fresh VQA image inventory has the wrong image count"
STORAGE_HEADROOM=$(storage_headroom)

PATHS=$(jq -cn \
  --arg development_summary "$DEV_SUMMARY" \
  --arg development_launch "$DEV_LAUNCH" \
  --arg development_summarizer "$DEV_SUMMARIZER" \
  --arg artifact_validation "$ARTIFACT_VALIDATION" \
  --arg artifact_validator "$ARTIFACT_VALIDATOR" \
  --arg training_launch "$TRAINING_LAUNCH" \
  --arg protocol "$PROTOCOL" --arg promotion "$PROMOTION" \
  --arg fresh_manifest "$FRESH_VQA" --arg fresh_metadata "$FRESH_META" \
  --arg selected_adapter "$ADAPTER_TENSOR" \
  --arg selected_adapter_config "$ADAPTER_CONFIG" \
  --arg selected_adapter_manifest "$ADAPTER_MANIFEST" \
  --arg eval_vqa "$EVAL_VQA" --arg recovery_utils "$RECOVERY_UTILS" \
  --arg quant_utils "$QUANT_UTILS" --arg gcq_patches "$GCQ_PATCHES" \
  --arg environment "$ENV_FILE" --arg launcher_script "$LAUNCHER" \
  --arg batch_script "$BATCH_SCRIPT" --arg confirmation_summarizer "$SUMMARIZER" \
  --arg analysis_vqa_helper "$VQA_HELPER" \
  --arg analysis_pairing_helper "$PAIRING_HELPER" \
  --arg analysis_recovery_helper "$RECOVERY_ANALYSIS_HELPER" '
  {$development_summary, $development_launch, $development_summarizer,
   $artifact_validation, $artifact_validator, $training_launch, $protocol,
   $promotion, $fresh_manifest, $fresh_metadata, $selected_adapter,
   $selected_adapter_config, $selected_adapter_manifest, $eval_vqa,
   $recovery_utils, $quant_utils, $gcq_patches, $environment, $launcher_script,
   $batch_script, $confirmation_summarizer, $analysis_vqa_helper,
   $analysis_pairing_helper, $analysis_recovery_helper}')
HASHES='{}'
while IFS=$'\t' read -r KEY FILE_PATH; do
  HASHES=$(jq -c --arg key "$KEY" --arg value "$(hash_file "$FILE_PATH")" \
    '. + {($key): $value}' <<<"$HASHES")
done < <(jq -r 'to_entries[] | [.key, .value] | @tsv' <<<"$PATHS")

CANDIDATE=$(jq -cn \
  --argjson step "$STEP" --arg recipe_id "$RECIPE" --arg adapter_dir "$ADAPTER_DIR" \
  --arg adapter_sha256 "$ADAPTER_SHA" \
  --arg adapter_config_sha256 "$ADAPTER_CONFIG_SHA" \
  --arg manifest_sha256 "$ADAPTER_MANIFEST_SHA" \
  '{$step, $recipe_id, $adapter_dir, $adapter_sha256,
    $adapter_config_sha256, $manifest_sha256}')

mkdir -p "$CONFIRMATION"
umask 077
LAUNCH_TMP=$(mktemp "$CONFIRMATION/.confirmation_launch_manifest.XXXXXX")
trap 'rm -f "$LAUNCH_TMP"' EXIT
jq -n \
  --arg recipe_id "$RECIPE" --arg base_model "$MODEL" --arg base_revision "$REVISION" \
  --arg baseline_tag "$BASE_TAG" --arg candidate_tag "$CANDIDATE_TAG" \
  --argjson candidates "[$CANDIDATE]" --argjson paths "$PATHS" --argjson hashes "$HASHES" \
  --argjson untouched_audit "$NO_PEEK_AUDIT" \
  --argjson image_inventory "$IMAGE_INVENTORY" \
  --argjson storage_headroom "$STORAGE_HEADROOM" '
  {schema_version: 1,
   evaluation_role: "one-time untouched VQAv2 confirmation of the single development-selected candidate",
   recipe_id: $recipe_id, base_model: $base_model, base_revision: $base_revision,
   quantization: {method: "rtn_quantize_dequantize", bits: 4, group_size: 128,
     average_decoder_bits: 4.25, max_pixels: 1003520},
   evaluation: {task: "vqa", questions: 5000, unique_images: 4571,
     start: 0, limit: 5000, batch_size: 24, device: "cuda:0",
     baseline_tag: $baseline_tag, candidate_tag: $candidate_tag,
     execution_order: ["untrained_gcq_baseline", "single_selected_candidate"],
     same_job_and_gpu_required: true},
   trained_candidates: $candidates,
   bootstrap: {unit: "image-clustered paired candidate-minus-untrained-GCQ",
     resamples: 10000, seed: 20260850},
   only_scientific_gate: {name: "VQA_fresh_paired_CI95_lower_at_least_minus_1.5pt",
     statistic: "paired candidate-minus-untrained-GCQ CI95 lower bound",
     minimum: -0.015},
   untouched_audit: $untouched_audit,
   image_inventory: $image_inventory,
   storage_headroom: $storage_headroom,
   paths: $paths, hashes: $hashes}' > "$LAUNCH_TMP"

chmod 0444 "$LAUNCH_TMP"
if ! ln "$LAUNCH_TMP" "$LAUNCH"; then
  die "another process froze the one-time confirmation launch first: $LAUNCH"
fi
rm -f "$LAUNCH_TMP"
trap - EXIT

mkdir -p "$ROOT/logs"
if ! QSUB_OUTPUT=$(qsub "$BATCH_SCRIPT"); then
  # Submission failed before any evaluation could start. Roll back only the
  # just-created launch hardlink so the identical pristine launch can be retried.
  chmod 0600 "$LAUNCH"
  rm -f "$LAUNCH"
  die "confirmation qsub failed before submission; pristine launch was rolled back"
fi
echo "$QSUB_OUTPUT"
