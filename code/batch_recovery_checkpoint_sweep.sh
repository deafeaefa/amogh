#!/bin/bash -l
#$ -P rise-tower
#$ -N gcq-rec-ckpt
#$ -t 1-4
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

STEP=$((SGE_TASK_ID * 100))
STEP_PADDED=$(printf '%06d' "$STEP")
TAG="gcq425_cwce_step${STEP}"
PILOT_ROOT="$GCQ_RUNS/recovery_pilot"
ADAPTER="$PILOT_ROOT/adapters/gcq425_lora_cwce_g5_s0/checkpoint-$STEP_PADDED"
PROMOTE="$GCQ_RUNS/promote_gcq_b4.25.json"
REVISION="89644892e4d85e24eaac8bacfd4f463576704203"

if [[ "$(jq -r '.checkpoint_step' "$ADAPTER/gcq_recovery_manifest.json")" != "$STEP" ]]; then
  echo "checkpoint manifest mismatch: $ADAPTER" >&2
  exit 2
fi
if [[ ! -s "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "checkpoint tensor is missing: $ADAPTER" >&2
  exit 2
fi

export GCQ_RUNS="$PILOT_ROOT/checkpoint_sweep/step${STEP}"
mkdir -p "$GCQ_RUNS"
"$PYT" eval_rec.py --model Qwen/Qwen3-VL-2B-Instruct --revision "$REVISION" \
  --subset recovery_dev_1k --tag "${TAG}_recoverydev" \
  --rtn-bits 4 --promote-file "$PROMOTE" --adapter-dir "$ADAPTER" \
  --batch 16 --device cuda:0
"$PYT" eval_vqa.py --model Qwen/Qwen3-VL-2B-Instruct --revision "$REVISION" --task vqa \
  --tag "${TAG}_vqa_select1k" --limit 1000 \
  --rtn-bits 4 --promote-file "$PROMOTE" --adapter-dir "$ADAPTER" \
  --batch 24 --device cuda:0
