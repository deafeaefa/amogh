# GCQ recovery training

This pipeline trains LoRA adapters on frozen BF16 quantize-dequantize weights.
It reproduces the W4/GCQ numerics used by the current paper, but it is **not** a
packed-int4 QLoRA implementation. Quantization is always applied before the
adapter is attached, and adapters are evaluated without merging.

## Current pilot status (2026-08-14)

Training array `7184200` completed all four arms at 500/500 optimizer steps.
`validate_recovery_pilot.py` verified the frozen data and promotion hashes,
base revision, objectives, seed, quantization layout, 196 LoRA targets,
17,432,576 trainable parameters, zero-init parity, nonzero adapter gradients,
no base gradients, and all four saved tensor files. Corrected single-hardware
evaluation array `7184512` completed on L40S GPUs; partial mixed-A100/L40S array
`7184503` was stopped and archived before any result was used. The step-500
GCQ+CWCE arm passed 11/12 screening checks: primary REC improved 80.53→82.93,
mean GIoU 0.7087→0.7438, and precise-IoU 0.6675→0.7071, but VQA fell
77.37→75.01 and violated the frozen preservation margin. L40S checkpoint array
`7184550` completed with all tasks exiting cleanly and all four checkpoints
passing every frozen selection screen. The predeclared rule selected step 300:
primary REC 80.53→83.73 (+3.20, paired 95% CI [1.47, 5.07]), GIoU
0.7087→0.7495, precise-IoU 0.6675→0.7091, and first-1k VQA 76.25→75.69.
Its one-time held-out evaluation is now complete: POPE improved 88.77→89.52
(+0.76, paired 95% CI [0.33, 1.21]), but VQA fell 77.65→75.20
(−2.45, CI [−3.20, −1.71]). The frozen preservation gate therefore
fails. The existing VQA5k is development-only from this point forward; the
minimal next recipe replaces caption replay with balanced VQAv2-train
short-answer replay and reserves a new image-disjoint VQA5k for confirmation.

## Balanced VQA-replay recovery run

The follow-up protocol is frozen in `recovery_vqa_replay_protocol.json`. Its
single recipe keeps the original 6,000 grounding examples and replaces caption
replay with 6,000 official VQAv2-train short answers on 6,000 unique images.
The 12,000-row manifest has SHA-256
`8bf3b6a1589527f5847ea28a7c5f0daeb89f6e0d7fa220451db87c52314aec4a`;
it contains no caption rows and has no recovery-development overlap. Training
uses GCQ B=4.25, CWCE gamma 5, rank-16 LoRA, learning rate 5e-5, effective batch
16, seed 0, and one 750-step epoch. Job `7184634` is the only training run for
this recipe. It completed all 750 optimizer steps in 2,415 seconds, using a
peak 6.245 GiB of CUDA memory. The base remained frozen, the first update had
nonzero LoRA gradients, and the full artifact validator accepted every
predeclared checkpoint. Homogeneous-L40S development array `7184903` was then
submitted without inspecting any partial checkpoint score.

Checkpoint screening is restricted to the predeclared steps
200/300/400/500/600/750. Each is evaluated on `recovery_dev_1k` and the complete
now-exposed `vqa_val_5k`; all grounding, parseability, point-estimate VQA, and
paired image-bootstrap VQA gates must pass. Selection then maximizes primary
REC, precise-IoU, and GIoU in that order, with an earlier-step tie break.

The confirmation manifest was also frozen before any predictions from the new
recipe: 5,000 VQAv2-val questions on 4,571 images, disjoint by image from both
the exposed VQA5k and POPE, SHA-256
`416aea5f9dd6f4a4cfa7061c3b5ca0af88647e49d5f17ed00a8d83885ea21038`.
Exactly one selected adapter may be evaluated on it, and its paired
image-clustered 95% CI lower endpoint must be at least -1.5 points. A failure
does not authorize evaluating a second checkpoint on this set.

For the independent grounding confirmation, full RefCOCO+ testA and testB were
frozen before checkpoint selection: 5,726 and 4,889 expressions on 750 images
per split. Their manifest hashes are `542fbbf73fe0623ed79ddba19167fce34647f97613d14e48b12ca5d96457ff83`
and `fafda8c81957baef1417c26e84fe225f51100eb2261ae61d123325a2b1c44462`.
The 1,500 images have zero overlap with recovery training, recovery-dev, or the
allocation probe, and all required files were decoded successfully before any
final-model predictions. These are untouched RefCOCO+ expressions and results,
not globally unseen COCO images: the variant uses the same 750-image testA/testB
partitions as the previously evaluated RefCOCO splits. The paper must state that
distinction explicitly.

## Prepare the frozen pilot data

```bash
source /usr4/spclpgm/eric1/GCQ/code/env.sh
cd /usr4/spclpgm/eric1/GCQ/code
$PYT build_recovery_data.py
$PYT fetch_recovery_images.py --workers 24
$PYT -m pytest -q test_recovery.py
```

The builder writes immutable 8k-train and 1k-development manifests under
`$GCQ_DATA/subsets`. It excludes every RefCOCO-family evaluation image and the
allocation-probe images. The train set is 4.5k RefCOCO-family REC, 1.5k
size-balanced COCO grounding, and 2k caption-replay examples.

## Run the gated pilot

```bash
qsub batch_recovery_smoke.sh
# After the smoke log says RECOVERY SMOKE PASS:
./launch_recovery_pilot.sh
# After all four adapters finish:
./launch_recovery_eval.sh
$PYT summarize_recovery_pilot.py
```

The evaluation launcher is fail-closed: every manifest must record 500 completed
steps and all verification flags, and each per-arm evaluation directory must be
empty. Move an old result directory aside explicitly before retrying; results are
never silently appended to stale rows.

The four array tasks are a matched 2×2 design:

| Base | Ordinary CE | Coordinate-weighted CE |
|---|---|---|
| Uniform RTN-W4 | `w4rtn_lora_ce_s0` | `w4rtn_lora_cwce_g5_s0` |
| GCQ B=4.25 | `gcq425_lora_ce_s0` | `gcq425_lora_cwce_g5_s0` |

All arms use rank 16, alpha 32, dropout 0.05, the same seed/order/data/steps,
and all seven decoder projections. Coordinate weighting is gamma 5 on only the
four numeric box spans; the per-example loss is divided by the sum of weights.

All compared models are loaded at immutable base revision
`89644892e4d85e24eaac8bacfd4f463576704203`. Recovery development evaluation
runs fresh BF16, uniform-W4, and GCQ baselines. Its 750 true referring-expression
examples are primary; the 250 category-grounding examples and every source are
reported separately. New logs retain full-precision IoU/GIoU and use the shared
exact thresholds 0.50:0.05:0.95. Paired differences and the factorial interaction
use 10,000 image-clustered bootstrap resamples.

VQAv2 uses official normalization and leave-one-annotator-out soft accuracy.
POPE requires an explicit leading yes/no and counts malformed generations as
incorrect. Both untrained GCQ and GCQ+weighted recovery are rerun with identical
revision, batch, quantization map, and evaluator settings.

The screening gate requires GCQ+weighted recovery to gain at least one primary
REC point over untrained GCQ, improve primary GIoU and precise-IoU, beat both
relevant trained controls, preserve parseability, obtain positive paired 95% CIs,
and keep VQAv2/POPE within 1.5 points. A pass is still development evidence only;
it advances to seeds 1–2 and one untouched RefCOCO+ confirmation.

### Checkpoint preservation gate

The 500-step GCQ+CWCE arm recovered grounding strongly but scored 75.01 on
VQAv2 versus 77.37 for fresh untrained GCQ, outside the 1.5-point preservation
margin. The saved 100/200/300/400-step checkpoints are therefore screened on
`recovery_dev_1k` and only the first 1,000 rows of frozen `vqa_val_5k`.

A checkpoint is eligible only if it gains at least one primary REC point over
untrained GCQ, improves GIoU and precise-IoU, beats W4+CWCE REC, keeps parse
failures within 0.5 point, and stays within a stricter one-point VQA development
margin. Among eligible checkpoints, selection maximizes primary REC, then
precise-IoU, then prefers the earlier step. The selected checkpoint is evaluated
once on the remaining 4,000 VQA rows and full POPE. This rule was frozen before
any intermediate-checkpoint outputs were generated.

## Adapter storage

The current PEFT checkpoints and runtime LoRA tensors are FP32; each file is
69,788,264 bytes, or 9.90% of nominal W4 decoder storage (9.32% of GCQ B=4.25).
The same 17,432,576 parameters would occupy 34,865,152 parameter bytes in BF16
(about 4.96% of W4), but PEFT's default loader silently re-upcasts a BF16 adapter.
The smaller number must not be reported until a non-overwriting BF16 export is
loaded with autocasting disabled, runtime BF16 is asserted, and full frozen-set
output parity plus VQAv2/POPE preservation are checked. Adapters remain separate
from the quantized base for all memory claims.
