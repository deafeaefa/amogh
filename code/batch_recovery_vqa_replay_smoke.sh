#!/bin/bash -l
#$ -P rise-tower
#$ -N gcq-vqa-smoke
#$ -q l40s
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_m=40G
#$ -l h_rt=00:30:00
#$ -pe omp 4
#$ -j y
#$ -o /projectnb/rise-tower/eric1/GCQ/runs/recovery_vqa_replay/logs

set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh
cd /usr4/spclpgm/eric1/GCQ/code

ROOT="$GCQ_RUNS/recovery_vqa_replay"
TAG="gcq425_lora_cwce_vqa50_g5_lr5e5_s0"

"$PYT" train_recovery.py \
  --train-file "$GCQ_DATA/subsets/recovery_train_vqa_replay_12k.json" \
  --output-dir "$ROOT/smoke/$TAG" \
  --objective cwce --gamma 5 \
  --rtn-bits 4 --rtn-group 128 --promote-file "$GCQ_RUNS/promote_gcq_b4.25.json" \
  --rank 16 --alpha 32 --dropout 0.05 \
  --learning-rate 5e-5 --weight-decay 0 \
  --epochs 1 --batch-size 2 --grad-accum 8 --warmup-ratio 0.03 \
  --max-samples 16 --save-every 0 --seed 0 --device cuda:0

