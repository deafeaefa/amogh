#!/bin/bash -l
#$ -P rise-tower
#$ -N gcq-rec-smoke
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_m=40G
#$ -l h_rt=01:00:00
#$ -pe omp 4
#$ -j y
#$ -o /projectnb/rise-tower/eric1/GCQ/runs/recovery_smoke.log

set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh
cd /usr4/spclpgm/eric1/GCQ/code

ROOT_RUNS="$GCQ_RUNS"
OUT="$ROOT_RUNS/recovery_smoke/adapter_${JOB_ID}"
"$PYT" -m pytest -q test_recovery.py
"$PYT" train_recovery.py \
  --objective cwce --gamma 5 --rtn-bits 4 --rtn-group 128 \
  --rank 16 --alpha 32 --dropout 0.05 \
  --learning-rate 1e-4 --epochs 1 --batch-size 1 --grad-accum 1 \
  --max-samples 4 --save-every 0 --log-every 1 --seed 0 --device cuda:0 \
  --output-dir "$OUT"

export GCQ_RUNS="$ROOT_RUNS/recovery_smoke/eval_${JOB_ID}"
mkdir -p "$GCQ_RUNS"
"$PYT" eval_rec.py --model Qwen/Qwen3-VL-2B-Instruct --subset recovery_dev_1k \
  --tag recovery_smoke --limit 4 --rtn-bits 4 --adapter-dir "$OUT" --batch 1 --device cuda:0
echo "RECOVERY SMOKE PASS: $OUT"
