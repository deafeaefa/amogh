#!/bin/bash -l
#$ -P rise-tower
#$ -N gcq-fills
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=02:00:00
#$ -j y
#$ -o /projectnb/rise-tower/eric1/GCQ/runs/gcq_fills_batch.log
# Three fill-in evals blocked by the K40 node: GCQ POPE (constraint), GCQ-4.5 ODinW
# (OOD dose-response), GCQ-on-GPTQ ODinW (OOD transfer on GPTQ, needs ckpt rebuild).
source /usr4/spclpgm/eric1/GCQ/code/env.sh
cd /usr4/spclpgm/eric1/GCQ/code
CK=${TMPDIR:-/tmp}/ckpts; mkdir -p "$CK"
$PYT eval_vqa.py --model Qwen/Qwen3-VL-2B-Instruct --task pope --tag gcq_b425_pope --rtn-bits 4 --promote-file $GCQ_RUNS/promote_gcq_b4.25.json --batch 32 --device cuda:0
$PYT eval_odinw.py --tag gcq_b45_odinw500s --rtn-bits 4 --promote-file $GCQ_RUNS/promote_gcq_b4.5.json --images 500 --device cuda:0
$PYT gptq_own.py --bits 4 --promote-file $GCQ_RUNS/promote_gcqgptqspec_b4.25.json --out "$CK/gcqgptqspec" --device cuda:0
$PYT eval_odinw.py --tag gcqGPTQspec_odinw500s --model "$CK/gcqgptqspec" --images 500 --device cuda:0
echo "BATCH DONE"
