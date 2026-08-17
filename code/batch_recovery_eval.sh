#!/bin/bash -l
#$ -P rise-tower
#$ -N gcq-rec-eval
#$ -t 1-4
#$ -q l40s
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_m=40G
#$ -l h_rt=12:00:00
#$ -pe omp 4
#$ -j y
#$ -o /projectnb/rise-tower/eric1/GCQ/runs/recovery_pilot/logs

set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh
cd /usr4/spclpgm/eric1/GCQ/code

ROOT_RUNS="$GCQ_RUNS"
PILOT_ROOT="$ROOT_RUNS/recovery_pilot"
PROMOTE="$ROOT_RUNS/promote_gcq_b4.25.json"
REVISION="89644892e4d85e24eaac8bacfd4f463576704203"
case "${SGE_TASK_ID:?submit as an array job with tasks 1-4}" in
  1) TAG=w4rtn_lora_ce_s0;       EXTRA=() ;;
  2) TAG=w4rtn_lora_cwce_g5_s0;  EXTRA=() ;;
  3) TAG=gcq425_lora_ce_s0;      EXTRA=(--promote-file "$PROMOTE") ;;
  4) TAG=gcq425_lora_cwce_g5_s0; EXTRA=(--promote-file "$PROMOTE") ;;
esac

ADAPTER="$PILOT_ROOT/adapters/$TAG"
if [[ ! -s "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "adapter is incomplete: $ADAPTER" >&2
  exit 2
fi

# Per-arm result directories avoid concurrent appends to the historical CSV.
export GCQ_RUNS="$PILOT_ROOT/eval/$TAG"
mkdir -p "$GCQ_RUNS"

if [[ "$SGE_TASK_ID" == 1 ]]; then
  "$PYT" eval_rec.py --model Qwen/Qwen3-VL-2B-Instruct --revision "$REVISION" \
    --subset recovery_dev_1k --tag bf16_recoverydev --batch 16 --device cuda:0
  "$PYT" eval_rec.py --model Qwen/Qwen3-VL-2B-Instruct --subset recovery_dev_1k \
    --revision "$REVISION" --tag w4rtn_untrained_recoverydev \
    --rtn-bits 4 --batch 16 --device cuda:0
elif [[ "$SGE_TASK_ID" == 3 ]]; then
  "$PYT" eval_rec.py --model Qwen/Qwen3-VL-2B-Instruct --subset recovery_dev_1k \
    --revision "$REVISION" --tag gcq425_untrained_recoverydev \
    --rtn-bits 4 --promote-file "$PROMOTE" --batch 16 --device cuda:0
fi

"$PYT" eval_rec.py --model Qwen/Qwen3-VL-2B-Instruct --revision "$REVISION" \
  --subset recovery_dev_1k \
  --tag "${TAG}_recoverydev" --rtn-bits 4 --adapter-dir "$ADAPTER" \
  --batch 16 --device cuda:0 "${EXTRA[@]}"

if [[ "$SGE_TASK_ID" == 3 ]]; then
  "$PYT" eval_vqa.py --model Qwen/Qwen3-VL-2B-Instruct --revision "$REVISION" --task vqa \
    --tag gcq425_untrained_recoverypilot_vqa5k --rtn-bits 4 --promote-file "$PROMOTE" \
    --batch 24 --device cuda:0
  "$PYT" eval_vqa.py --model Qwen/Qwen3-VL-2B-Instruct --revision "$REVISION" --task pope \
    --tag gcq425_untrained_recoverypilot_pope --rtn-bits 4 --promote-file "$PROMOTE" \
    --batch 24 --device cuda:0
fi

# Only the proposed final method needs the expensive general-capability guard.
# The other three arms are controls for the held-out grounding factorial.
if [[ "$SGE_TASK_ID" == 4 ]]; then
  "$PYT" eval_vqa.py --model Qwen/Qwen3-VL-2B-Instruct --revision "$REVISION" --task vqa \
    --tag "${TAG}_vqa5k" --rtn-bits 4 --adapter-dir "$ADAPTER" \
    --batch 24 --device cuda:0 "${EXTRA[@]}"
  "$PYT" eval_vqa.py --model Qwen/Qwen3-VL-2B-Instruct --revision "$REVISION" --task pope \
    --tag "${TAG}_pope" --rtn-bits 4 --adapter-dir "$ADAPTER" \
    --batch 24 --device cuda:0 "${EXTRA[@]}"
fi
