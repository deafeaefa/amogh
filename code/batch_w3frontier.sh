#!/bin/bash -l
#$ -P rise-tower
#$ -N gcq-w3frontier
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=04:00:00
#$ -j y
#$ -o /projectnb/rise-tower/eric1/GCQ/runs/gcq_w3frontier_batch.log
# W3 frontier: does protecting the grounding band resurrect grounding at 3-bit,
# where it otherwise dies outright (0% parseable boxes)?
#   1. GPTQ base W3 + GPTQ-profiled band at 8-bit (avg ~3.3 bits): REC + VQA
#   2. RTN  base W3 + RTN-profiled band at 8-bit: REC (RTN-W3 alone collapses globally)
source /usr4/spclpgm/eric1/GCQ/code/env.sh
cd /usr4/spclpgm/eric1/GCQ/code
CK=${TMPDIR:-/tmp}/ckpts; mkdir -p "$CK"
$PYT gptq_own.py --bits 3 --promote-file $GCQ_RUNS/promote_gcqgptqspec_b4.25.json --out "$CK/gptqw3band" --device cuda:0
$PYT eval_rec.py --model "$CK/gptqw3band" --subset rec_eval_refcoco_val_1k --tag gcqGPTQ_w3band_rec1k --batch 24 --device cuda:0
$PYT eval_vqa.py --model "$CK/gptqw3band" --task vqa --tag gcqGPTQ_w3band_vqa5k --batch 32 --device cuda:0
$PYT eval_rec.py --model Qwen/Qwen3-VL-2B-Instruct --subset rec_eval_refcoco_val_1k --tag gcqRTN_w3band_rec1k --rtn-bits 3 --promote-file $GCQ_RUNS/promote_gcq_b4.25.json --batch 24 --device cuda:0
echo "W3FRONTIER DONE"
