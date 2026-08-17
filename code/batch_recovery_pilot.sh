#!/bin/bash -l
#$ -P rise-tower
#$ -N gcq-recovery
#$ -t 1-4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_m=40G
#$ -l h_rt=24:00:00
#$ -pe omp 4
#$ -j y
#$ -o /projectnb/rise-tower/eric1/GCQ/runs/recovery_pilot/logs

set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh
cd /usr4/spclpgm/eric1/GCQ/code

PILOT_ROOT="$GCQ_RUNS/recovery_pilot"
PROMOTE="$GCQ_RUNS/promote_gcq_b4.25.json"
case "${SGE_TASK_ID:?submit as an array job with tasks 1-4}" in
  1) TAG=w4rtn_lora_ce_s0;        OBJECTIVE=ce;   EXTRA=() ;;
  2) TAG=w4rtn_lora_cwce_g5_s0;   OBJECTIVE=cwce; EXTRA=(--gamma 5) ;;
  3) TAG=gcq425_lora_ce_s0;       OBJECTIVE=ce;   EXTRA=(--promote-file "$PROMOTE") ;;
  4) TAG=gcq425_lora_cwce_g5_s0;  OBJECTIVE=cwce; EXTRA=(--gamma 5 --promote-file "$PROMOTE") ;;
esac

"$PYT" train_recovery.py \
  --objective "$OBJECTIVE" \
  --rtn-bits 4 --rtn-group 128 \
  --rank 16 --alpha 32 --dropout 0.05 \
  --learning-rate 1e-4 --epochs 1 \
  --batch-size 2 --grad-accum 8 \
  --seed 0 --device cuda:0 \
  --output-dir "$PILOT_ROOT/adapters/$TAG" \
  "${EXTRA[@]}"
