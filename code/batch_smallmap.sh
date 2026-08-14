#!/bin/bash -l
#$ -P rise-tower
#$ -N gcq-smallmap
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=08:00:00
#$ -j y
#$ -o /projectnb/rise-tower/eric1/GCQ/runs/gcq_smallmap_batch.log
# Small-object-weighted allocation: re-profile sensitivity on the smallest-quartile
# probe boxes, allocate at the same B=4.25 budget, and test whether it beats the
# standard map where the damage concentrates (size-stratified ODinW).
source /usr4/spclpgm/eric1/GCQ/code/env.sh
cd /usr4/spclpgm/eric1/GCQ/code
$PYT profile_sensitivity.py --probe rec --small-frac 0.25 --out sensitivity_small.csv --device cuda:0
$PYT allocate.py --budget 4.25 --sens-file sensitivity_small.csv --prefix gcqsmall --seeds
$PYT eval_rec.py --model Qwen/Qwen3-VL-2B-Instruct --subset rec_eval_refcoco_val_1k --tag gcqsmall_b425_rec1k --rtn-bits 4 --promote-file $GCQ_RUNS/promote_gcqsmall_b4.25.json --batch 24 --device cuda:0
$PYT eval_odinw.py --tag gcqsmall_b425_odinw500s --rtn-bits 4 --promote-file $GCQ_RUNS/promote_gcqsmall_b4.25.json --images 500 --device cuda:0
$PYT eval_vqa.py --model Qwen/Qwen3-VL-2B-Instruct --task vqa --tag gcqsmall_b425_vqa5k --rtn-bits 4 --promote-file $GCQ_RUNS/promote_gcqsmall_b4.25.json --batch 32 --device cuda:0
echo "SMALLMAP DONE"
