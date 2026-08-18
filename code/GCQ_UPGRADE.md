# Frozen GCQ projection-GPTQ experiment

Status: implementation and CPU validation are complete; no upgraded GPU score
has been produced yet. The paper must continue to report the existing 48--64%
recovery until the frozen run below finishes.

The primary target is at least 75% recovery of the uniform-W4 loss. Ninety
percent remains the stretch hypothesis. Confirmation sets may not influence
the allocation, controls, or checkpoint choice.

## Artifact roles

- `gcq_upgrade_protocol.json` is the score-blind design.
- `gcq_upgrade.launch_frozen.json` is created once after data, calibration,
  candidate bank, and implementation hashes are final.
- `gptq_projection_candidates/` contains matched packed W4/W8 arms for all 196
  decoder projections.
- `shortlist24.json` is the frozen top-24 coordinate-KL proxy catalog.
- `beam/` is the authoritative resumable conditional-beam state machine.
- `comparison_plan.json` deduplicates the selected states and all controls.
- A materialized result contains both a dense BF16-QDQ model for the current
  evaluators and the actual packed decoder payload. Only the latter supports
  the byte/bit claim; the dense directory is not a compressed checkpoint.

## 1. Build all score-blind inputs

Run from the repository root after `source code/env.sh`. The commands below pin
the immutable repository heads returned by the official Hugging Face dataset
API during the pre-score audit, plus the official WikiText Parquet-conversion
commit. Output paths are write-once, so a rerun needs a new run directory rather
than deleting or replacing frozen artifacts.

```bash
$PYT code/extract_gcq_profile_candidates.py \
  --coco-instances "$GCQ_DATA/coco_ann/ann2014.zip" \
  --ref-revision refcoco=9fba8200c5326e996f789191f095bd464ef1d09e \
  --ref-revision refcocoplus=12e20b0b6039fbf656e89e2f26597e84c1037847 \
  --ref-revision refcocog=55319436ad54b9480cefda6b9d64397de92456dd \
  --exclude-manifest "$GCQ_DATA/subsets/dprobe_refcoco_train_512.json" \
  --exclude-manifest "$GCQ_DATA/subsets/recovery_train_8k.json" \
  --exclude-manifest "$GCQ_DATA/subsets/recovery_train_vqa_replay_12k.json" \
  --exclude-manifest "$GCQ_DATA/subsets/recovery_dev_1k.json" \
  --exclude-manifest "$GCQ_DATA/subsets/vqa_val_5k.json" \
  --exclude-manifest "$GCQ_DATA/subsets/vqa_fresh_confirm_5k.json" \
  --out-dir "$GCQ_RUNS/gcq_upgrade/profile_candidates"

$PYT code/build_gcq_profile_data.py \
  --rec-candidates "$GCQ_RUNS/gcq_upgrade/profile_candidates/gcq_profile_rec_candidates.json" \
  --category-candidates "$GCQ_RUNS/gcq_upgrade/profile_candidates/gcq_profile_category_candidates.json" \
  --exclude-image-ids "$GCQ_RUNS/gcq_upgrade/profile_candidates/gcq_profile_excluded_train2014_image_ids.json" \
  --out-dir "$GCQ_RUNS/gcq_upgrade/data"

$PYT code/build_gcq_vqa_control_data.py \
  --questions-zip "$GCQ_DATA/vqa/v2_Questions_Train_mscoco.zip" \
  --annotations-zip "$GCQ_DATA/vqa/v2_Annotations_Train_mscoco.zip" \
  --exclude "$GCQ_DATA/subsets/dprobe_refcoco_train_512.json" \
  --exclude "$GCQ_DATA/subsets/recovery_train_8k.json" \
  --exclude "$GCQ_DATA/subsets/recovery_train_vqa_replay_12k.json" \
  --exclude "$GCQ_DATA/subsets/recovery_dev_1k.json" \
  --exclude "$GCQ_DATA/subsets/vqa_val_5k.json" \
  --exclude "$GCQ_DATA/subsets/vqa_fresh_confirm_5k.json" \
  --exclude "$GCQ_RUNS/gcq_upgrade/data/gcq_profile_proxy_train_512.json" \
  --exclude "$GCQ_RUNS/gcq_upgrade/data/gcq_profile_decode_train_512.json" \
  --image-root "$GCQ_DATA/images/train2014" \
  --out "$GCQ_RUNS/gcq_upgrade/data/gcq_vqa_control_train_512.json"

$PYT code/build_gptq_calibration.py \
  --dataset-revision b08601e04326c79dfdd32d625aee71d232d685c3 \
  --out "$GCQ_RUNS/gcq_upgrade/gptq_calibration_128x512.json"
```

Confirm the exact extractor output names before running the profile-data
command; the extractor prints every created path. Never substitute a validation
or confirmation manifest when one of the training-only files is absent.

## 2. Build the matched packed candidate bank

This is the first expensive GPU step. It uses the unbound design and creates
W4/W8 from the same pristine BF16 weight and Hessian.

```bash
$PYT code/build_gptq_projection_bank.py \
  --protocol code/gcq_upgrade_protocol.json \
  --calibration-manifest "$GCQ_RUNS/gcq_upgrade/gptq_calibration_128x512.json" \
  --out "$GCQ_RUNS/gcq_upgrade/gptq_projection_candidates" \
  --device cuda:0
```

After this command, finish all code edits and rerun the tests. Then bind once:

```bash
$PYT code/freeze_gcq_upgrade.py \
  --design code/gcq_upgrade_protocol.json \
  --calibration-manifest "$GCQ_RUNS/gcq_upgrade/gptq_calibration_128x512.json" \
  --proxy-manifest "$GCQ_RUNS/gcq_upgrade/data/gcq_profile_proxy_train_512.json" \
  --decode-manifest "$GCQ_RUNS/gcq_upgrade/data/gcq_profile_decode_train_512.json" \
  --vqa-control-manifest "$GCQ_RUNS/gcq_upgrade/data/gcq_vqa_control_train_512.json" \
  --candidate-bank-manifest "$GCQ_RUNS/gcq_upgrade/gptq_projection_candidates/manifest.json" \
  --packing-spec code/gptq_packing_spec.json \
  --implementation-dir code \
  --out "$GCQ_RUNS/gcq_upgrade/gcq_upgrade.launch_frozen.json"
```

Any subsequent edit to a hash-checked worker requires a new launch protocol
and a clean experiment namespace. Do not silently update the bound file.

## 3. Profile all 196 projections and freeze the top 24

Three GPU workers can use disjoint module slices. They may share the output
directory because every tagged artifact is exclusive.

```bash
$PYT code/profile_gcq_projections.py --modules 0:66 --tag gpu0 \
  --manifest "$GCQ_RUNS/gcq_upgrade/data/gcq_profile_proxy_train_512.json" \
  --candidate-cache "$GCQ_RUNS/gcq_upgrade/gptq_projection_candidates" \
  --protocol-context "$GCQ_RUNS/gcq_upgrade/gcq_upgrade.launch_frozen.json" \
  --data-dir "$GCQ_DATA" --out-dir "$GCQ_RUNS/gcq_upgrade/proxy" --device cuda:0

$PYT code/profile_gcq_projections.py --modules 66:131 --tag gpu1 \
  --manifest "$GCQ_RUNS/gcq_upgrade/data/gcq_profile_proxy_train_512.json" \
  --candidate-cache "$GCQ_RUNS/gcq_upgrade/gptq_projection_candidates" \
  --protocol-context "$GCQ_RUNS/gcq_upgrade/gcq_upgrade.launch_frozen.json" \
  --data-dir "$GCQ_DATA" --out-dir "$GCQ_RUNS/gcq_upgrade/proxy" --device cuda:1

$PYT code/profile_gcq_projections.py --modules 131:196 --tag gpu2 \
  --manifest "$GCQ_RUNS/gcq_upgrade/data/gcq_profile_proxy_train_512.json" \
  --candidate-cache "$GCQ_RUNS/gcq_upgrade/gptq_projection_candidates" \
  --protocol-context "$GCQ_RUNS/gcq_upgrade/gcq_upgrade.launch_frozen.json" \
  --data-dir "$GCQ_DATA" --out-dir "$GCQ_RUNS/gcq_upgrade/proxy" --device cuda:2

$PYT code/gcq_profile_metrics.py shortlist \
  --summaries "$GCQ_RUNS/gcq_upgrade/proxy/gpu0.coordinate_summaries.json" \
  --summaries "$GCQ_RUNS/gcq_upgrade/proxy/gpu1.coordinate_summaries.json" \
  --summaries "$GCQ_RUNS/gcq_upgrade/proxy/gpu2.coordinate_summaries.json" \
  --top-k 24 --expected-candidates 196 \
  --out "$GCQ_RUNS/gcq_upgrade/shortlist24.json"
```

## 4. Run the resumable conditional beam

The context passed to the allocator is the raw launch-protocol file hash, not
the embedded content hash. The frozen primary cap is 88,080,384 added bytes.

```bash
CONTEXT_SHA256=$(sha256sum "$GCQ_RUNS/gcq_upgrade/gcq_upgrade.launch_frozen.json" | cut -d' ' -f1)

$PYT code/allocate_gcq_beam.py init \
  --candidates "$GCQ_RUNS/gcq_upgrade/shortlist24.json" \
  --budget-bytes 88080384 --beam-width 4 \
  --context-sha256 "$CONTEXT_SHA256" \
  --run-dir "$GCQ_RUNS/gcq_upgrade/beam"

$PYT code/allocate_gcq_beam.py plan --run-dir "$GCQ_RUNS/gcq_upgrade/beam"
```

For each emitted `plan_round_*.json`, run three state shards:

```bash
$PYT code/eval_gcq_plan.py --plan PLAN_JSON --num-shards 3 --shard-index 0 \
  --candidate-cache "$GCQ_RUNS/gcq_upgrade/gptq_projection_candidates" \
  --candidate-catalog "$GCQ_RUNS/gcq_upgrade/shortlist24.json" \
  --decode-manifest "$GCQ_RUNS/gcq_upgrade/data/gcq_profile_decode_train_512.json" \
  --protocol-context "$GCQ_RUNS/gcq_upgrade/gcq_upgrade.launch_frozen.json" \
  --image-root "$GCQ_DATA/images/train2014" \
  --out-dir "$GCQ_RUNS/gcq_upgrade/beam_decodes" --device cuda:0
```

Repeat with shard indices 1/2 and devices `cuda:1`/`cuda:2`. Then score and
record the full plan:

```bash
$PYT code/score_gcq_plan.py --plan PLAN_JSON \
  --results-dir "$GCQ_RUNS/gcq_upgrade/beam_decodes" \
  --manifest "$GCQ_RUNS/gcq_upgrade/data/gcq_profile_decode_train_512.json" \
  --context-sha256 "$CONTEXT_SHA256" --out ROUND_SCORES_JSON

$PYT code/allocate_gcq_beam.py record \
  --run-dir "$GCQ_RUNS/gcq_upgrade/beam" --results ROUND_SCORES_JSON
```

Repeat `plan -> three eval shards -> score -> record` until status is
`complete`. Freeze the primary and secondary selections from the same trace:

```bash
$PYT code/allocate_gcq_beam.py select --run-dir "$GCQ_RUNS/gcq_upgrade/beam" \
  --out "$GCQ_RUNS/gcq_upgrade/gcq_primary.json"
$PYT code/allocate_gcq_beam.py select --run-dir "$GCQ_RUNS/gcq_upgrade/beam" \
  --max-bytes 44040192 --out "$GCQ_RUNS/gcq_upgrade/gcq_secondary_b4_25.json"
```

## 5. Build and evaluate the frozen controls

Profile the same 24 projections in three disjoint slices on the VQAv2-train
control data, then merge exact coverage:

```bash
$PYT code/profile_gcq_control_scores.py profile --modules 0:8 --tag gpu0 \
  --manifest "$GCQ_RUNS/gcq_upgrade/data/gcq_vqa_control_train_512.json" \
  --candidate-cache "$GCQ_RUNS/gcq_upgrade/gptq_projection_candidates" \
  --candidate-catalog "$GCQ_RUNS/gcq_upgrade/shortlist24.json" \
  --protocol-context "$GCQ_RUNS/gcq_upgrade/gcq_upgrade.launch_frozen.json" \
  --image-root "$GCQ_DATA/images/train2014" \
  --out-dir "$GCQ_RUNS/gcq_upgrade/control_scores" --device cuda:0
```

Repeat for `8:16`/`gpu1` and `16:24`/`gpu2`, then:

```bash
$PYT code/profile_gcq_control_scores.py merge \
  --candidate-catalog "$GCQ_RUNS/gcq_upgrade/shortlist24.json" \
  --vqa-slice "$GCQ_RUNS/gcq_upgrade/control_scores/vqa_control_scores.gpu0.json" \
  --vqa-slice "$GCQ_RUNS/gcq_upgrade/control_scores/vqa_control_scores.gpu1.json" \
  --vqa-slice "$GCQ_RUNS/gcq_upgrade/control_scores/vqa_control_scores.gpu2.json" \
  --maba-slice "$GCQ_RUNS/gcq_upgrade/control_scores/maba_control_scores.gpu0.json" \
  --maba-slice "$GCQ_RUNS/gcq_upgrade/control_scores/maba_control_scores.gpu1.json" \
  --maba-slice "$GCQ_RUNS/gcq_upgrade/control_scores/maba_control_scores.gpu2.json" \
  --out-dir "$GCQ_RUNS/gcq_upgrade/control_scores"

TARGET_BYTES=$($PYT -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"]["cost_bytes"])' \
  "$GCQ_RUNS/gcq_upgrade/gcq_primary.json")

$PYT code/build_gcq_controls.py \
  --candidates "$GCQ_RUNS/gcq_upgrade/shortlist24.json" \
  --target-bytes "$TARGET_BYTES" \
  --gcq-scores "$GCQ_RUNS/gcq_upgrade/shortlist24.json" \
  --vqa-scores "$GCQ_RUNS/gcq_upgrade/control_scores/vqa_control_scores.json" \
  --maba-scores "$GCQ_RUNS/gcq_upgrade/control_scores/maba_control_scores.json" \
  --random-seeds 2026081701 2026081702 2026081703 2026081704 2026081705 \
                 2026081706 2026081707 2026081708 2026081709 2026081710 \
  --protocol-context "$GCQ_RUNS/gcq_upgrade/gcq_upgrade.launch_frozen.json" \
  --out "$GCQ_RUNS/gcq_upgrade/controls.json"

$PYT code/build_gcq_comparison_plan.py \
  --primary-selection "$GCQ_RUNS/gcq_upgrade/gcq_primary.json" \
  --secondary-selection "$GCQ_RUNS/gcq_upgrade/gcq_secondary_b4_25.json" \
  --controls "$GCQ_RUNS/gcq_upgrade/controls.json" \
  --protocol-context "$GCQ_RUNS/gcq_upgrade/gcq_upgrade.launch_frozen.json" \
  --out "$GCQ_RUNS/gcq_upgrade/comparison_plan.json"
```

Evaluate and score `comparison_plan.json` exactly like a beam plan. This yields
one decode per unique state even when several labels chose the same members.
Report all ten random seeds and their mean; never report only the best seed.

## 6. Freeze the evaluation checkpoint, then open confirmation

```bash
$PYT code/materialize_gcq_checkpoint.py \
  --state-artifact "$GCQ_RUNS/gcq_upgrade/comparison_plan.json" \
  --label gcq_primary \
  --candidate-cache "$GCQ_RUNS/gcq_upgrade/gptq_projection_candidates" \
  --candidate-catalog "$GCQ_RUNS/gcq_upgrade/shortlist24.json" \
  --protocol-context "$GCQ_RUNS/gcq_upgrade/gcq_upgrade.launch_frozen.json" \
  --out "$GCQ_RUNS/gcq_upgrade/materialized_gcq_primary" --device cuda:0
```

Use `materialized_gcq_primary/dense_qdq_model` as `--model` in `eval_rec.py`,
`eval_vqa.py`, and `eval_odinw.py`, with `--rtn-bits 0` and no promotion file.
Only after the allocation, controls, comparison plan, and materialization are
immutable may the RefCOCO+, RefCOCOg, corrected ODinW, fresh VQAv2, and POPE
confirmation runs begin. If confirmation fails, retain and report the frozen
result; do not allocate a second checkpoint from confirmation feedback.

## Validation snapshot

- Upgrade-specific CPU suite: 113 passed.
- Whole repository suite in the non-data shell: 188 passed; five data-backed
  tests could not run because `/projectnb/rise-tower/eric1/GCQ` was not mounted.
- `py_compile`, CLI help smoke tests, and `git diff --check` pass.
