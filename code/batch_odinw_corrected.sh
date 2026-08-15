#!/bin/bash -l
#$ -P rise-tower
#$ -N gcq-odinw-fix
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=12:00:00
#$ -j y
#$ -o /projectnb/rise-tower/eric1/GCQ/runs/gcq_odinw_corrected.log
# Corrected ODinW on the FULL test split for the three models that matter.
# Fixes: exact sampling (old run silently used 376 of 500 images), no 20-box cap,
# maximum-cardinality matching, relative-area logging for predeclared quartiles.
source /usr4/spclpgm/eric1/GCQ/code/env.sh
cd /usr4/spclpgm/eric1/GCQ/code
$PYT eval_odinw.py --tag bf16_odinwFULL --images 0 --device cuda:0
$PYT eval_odinw.py --tag w4rtn_odinwFULL --rtn-bits 4 --images 0 --device cuda:0
$PYT eval_odinw.py --tag gcq_b425_odinwFULL --rtn-bits 4 --promote-file $GCQ_RUNS/promote_gcq_b4.25.json --images 0 --device cuda:0
echo "ODINW CORRECTED DONE"
