#!/bin/bash
# Freeze and submit the one-time RefCOCO+ testA/testB confirmation.
#
# This launcher is intentionally unusable until fresh-VQA confirmation passed.
# It evaluates exactly that VQA-confirmed adapter and never another candidate.
set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh

ROOT="$GCQ_RUNS/recovery_vqa_replay"
CONFIRMATION="$ROOT/grounding_confirmation"
LAUNCH="$CONFIRMATION/grounding_confirmation_launch_manifest.json"
SUMMARY="$ROOT/grounding_confirmation_summary.json"
RECIPE="gcq425_lora_cwce_vqa50_g5_lr5e5_s0"
MODEL="Qwen/Qwen3-VL-2B-Instruct"
REVISION="89644892e4d85e24eaac8bacfd4f463576704203"
MINIMUM_STORAGE_HEADROOM_BYTES=1073741824

LAUNCHER="/usr4/spclpgm/eric1/GCQ/code/launch_recovery_grounding_confirmation.sh"
BATCH_SCRIPT="/usr4/spclpgm/eric1/GCQ/code/batch_recovery_grounding_confirmation.sh"
SUMMARIZER="/usr4/spclpgm/eric1/GCQ/code/summarize_recovery_grounding_confirmation.py"
ENV_FILE="/usr4/spclpgm/eric1/GCQ/code/env.sh"
EVAL_REC="/usr4/spclpgm/eric1/GCQ/code/eval_rec.py"
RECOVERY_UTILS="/usr4/spclpgm/eric1/GCQ/code/recovery_utils.py"
QUANT_UTILS="/usr4/spclpgm/eric1/GCQ/code/quant_utils.py"
GCQ_PATCHES="/usr4/spclpgm/eric1/GCQ/code/gcq_patches.py"
BUILDER="/usr4/spclpgm/eric1/GCQ/code/build_refcocoplus_confirmation.py"

die() {
  echo "$*" >&2
  exit 2
}

hash_file() {
  sha256sum "$1" | awk '{print $1}'
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

for REQUIRED in "$LAUNCHER" "$BATCH_SCRIPT" "$SUMMARIZER" "$ENV_FILE" \
  "$EVAL_REC" "$RECOVERY_UTILS" "$QUANT_UTILS" "$GCQ_PATCHES" "$BUILDER"; do
  [[ -f "$REQUIRED" && -s "$REQUIRED" ]] || die "required grounding code is missing: $REQUIRED"
done
if [[ -e "$SUMMARY" ]]; then
  die "refusing to replace an existing grounding confirmation summary: $SUMMARY"
fi
if [[ -e "$CONFIRMATION" && ! -d "$CONFIRMATION" ]]; then
  die "grounding confirmation path exists but is not a directory: $CONFIRMATION"
fi
if [[ -d "$CONFIRMATION" ]] && find "$CONFIRMATION" -mindepth 1 -print -quit | grep -q .; then
  die "grounding confirmation directory is not pristine: $CONFIRMATION"
fi

PREFLIGHT=$(mktemp /tmp/gcq-grounding-preflight.XXXXXX.json)
LAUNCH_TMP=""
trap 'rm -f "$PREFLIGHT" "${LAUNCH_TMP:-}"' EXIT
"$PYT" "$SUMMARIZER" --preflight > "$PREFLIGHT"
if ! jq -e --arg recipe "$RECIPE" '
    .schema_version == 1 and .authorization_pass == true and
    .selected.recipe_id == $recipe and
    (.selected.step | type) == "number" and
    (.selected.adapter_dir | type) == "string" and
    (.selected.adapter_sha256 | type) == "string" and
    (.selected.adapter_config_sha256 | type) == "string" and
    (.selected.manifest_sha256 | type) == "string" and
    .splits.testA.expressions == 5726 and .splits.testA.images == 750 and
    .splits.testB.expressions == 4889 and .splits.testB.images == 750' \
    "$PREFLIGHT" >/dev/null; then
  die "grounding preflight did not produce the frozen authorization contract"
fi

# Scan identities, not scores: fixed filenames catch crashed attempts, while
# subset names and RefCOCO+ UIDs catch alternate output tags.
NO_PEEK_AUDIT=$("$PYT" "$SUMMARIZER" --audit-pristine --audit-split all)
if ! jq -e '
    .splits == ["testA", "testB"] and
    .fixed_tag_outputs_found == 0 and
    .matching_subset_metrics_found == 0 and
    .confirmation_uid_prediction_logs_found == 0' \
    <<<"$NO_PEEK_AUDIT" >/dev/null; then
  die "grounding no-peeking identity audit failed"
fi
NO_PEEK_AUDIT=$(jq -c \
  '. + {checked_before_launch: true, recheck_each_split_at_job_start: true}' \
  <<<"$NO_PEEK_AUDIT")
STORAGE_HEADROOM=$(storage_headroom)

SELECTED=$(jq -c '.selected' "$PREFLIGHT")
SPLITS=$(jq -c '
  {testA: (.splits.testA + {
      task_id: 1, subset: "refcocoplus_testA_confirm_full",
      tags: {
        bf16: "bf16_refcocoplus_testA_confirm",
        uniform_rtn_w4: "w4rtn_refcocoplus_testA_confirm",
        "untrained_gcq4.25": "gcq425_untrained_refcocoplus_testA_confirm",
        selected_adapter: "gcq425_vqa_selected_refcocoplus_testA_confirm"
      }}),
   testB: (.splits.testB + {
      task_id: 2, subset: "refcocoplus_testB_confirm_full",
      tags: {
        bf16: "bf16_refcocoplus_testB_confirm",
        uniform_rtn_w4: "w4rtn_refcocoplus_testB_confirm",
        "untrained_gcq4.25": "gcq425_untrained_refcocoplus_testB_confirm",
        selected_adapter: "gcq425_vqa_selected_refcocoplus_testB_confirm"
      }})}' "$PREFLIGHT")
PATHS=$(jq -c \
  --arg environment "$ENV_FILE" \
  --arg eval_rec "$EVAL_REC" \
  --arg recovery_utils "$RECOVERY_UTILS" \
  --arg quant_utils "$QUANT_UTILS" \
  --arg gcq_patches "$GCQ_PATCHES" \
  --arg grounding_builder "$BUILDER" \
  --arg grounding_launcher "$LAUNCHER" \
  --arg grounding_batch "$BATCH_SCRIPT" \
  --arg grounding_summarizer "$SUMMARIZER" '
  .input_paths + {$environment, $eval_rec, $recovery_utils, $quant_utils,
    $gcq_patches, $grounding_builder, $grounding_launcher, $grounding_batch,
    $grounding_summarizer}' "$PREFLIGHT")
HASHES='{}'
while IFS=$'\t' read -r KEY FILE_PATH; do
  [[ -f "$FILE_PATH" && -s "$FILE_PATH" ]] || die "missing launch input $KEY: $FILE_PATH"
  HASHES=$(jq -c --arg key "$KEY" --arg value "$(hash_file "$FILE_PATH")" \
    '. + {($key): $value}' <<<"$HASHES")
done < <(jq -r 'to_entries[] | [.key, .value] | @tsv' <<<"$PATHS")

mkdir -p "$CONFIRMATION" "$ROOT/logs"
umask 077
LAUNCH_TMP=$(mktemp "$CONFIRMATION/.grounding_launch_manifest.XXXXXX")
jq -n \
  --arg recipe_id "$RECIPE" --arg base_model "$MODEL" --arg base_revision "$REVISION" \
  --argjson selected "$SELECTED" --argjson splits "$SPLITS" \
  --argjson paths "$PATHS" --argjson hashes "$HASHES" \
  --argjson untouched_audit "$NO_PEEK_AUDIT" \
  --argjson storage_headroom "$STORAGE_HEADROOM" '
  {schema_version: 1,
   evaluation_role: "one-time frozen RefCOCO+ grounding confirmation",
   recipe_id: $recipe_id, base_model: $base_model, base_revision: $base_revision,
   selected: $selected,
   arm_order: ["bf16", "uniform_rtn_w4", "untrained_gcq4.25", "selected_adapter"],
   scheduler_array_tasks: {"1": "testA", "2": "testB"},
   splits: $splits,
   evaluation: {hardware: "one NVIDIA L40S per split",
     within_task_execution: "all four arms sequentially on the same physical GPU",
     max_pixels: 1003520, batch_size: 16, device: "cuda:0", decoding: "greedy"},
   bootstrap: {unit: "paired referring expressions clustered by COCO image",
     resamples: 10000, base_seed: 20260860,
     seed_mapping: "base_seed + split_index*12 + comparison_index*4 + metric_index; split order testA,testB; comparison/metric order as stored"},
   failure_policy: {scientific_gate_failure: "publish complete FAIL summary and exit zero",
     integrity_failure: "publish no scientific summary and exit nonzero",
     checkpoint_substitution_after_failure: false},
   no_peeking: {manifest_frozen_before_selection: true,
     model_predictions_and_metrics_unseen_before_selection: true,
     expected_output_tags_absent_when_launch_frozen: true,
     alternate_tag_identity_scan_before_launch: true,
     recheck_each_split_at_job_start: true,
     globally_image_unseen_claim: false},
   untouched_audit: $untouched_audit,
   storage_headroom: $storage_headroom,
   paths: $paths, hashes: $hashes}' > "$LAUNCH_TMP"

chmod 0444 "$LAUNCH_TMP"
if ! ln "$LAUNCH_TMP" "$LAUNCH"; then
  die "another process froze the one-time grounding launch first: $LAUNCH"
fi
rm -f "$LAUNCH_TMP"
LAUNCH_TMP=""
trap - EXIT
rm -f "$PREFLIGHT"

if ! QSUB_OUTPUT=$(qsub "$BATCH_SCRIPT"); then
  # Submission failed before either split could be exposed.  Remove only this
  # just-created hardlink so the byte-identical pristine launch can be retried.
  chmod 0600 "$LAUNCH"
  rm -f "$LAUNCH"
  die "grounding qsub failed before submission; pristine launch was rolled back"
fi
echo "$QSUB_OUTPUT"
