#!/bin/bash
# GCQ 4-GPU queue launcher. Usage: launch_queues.sh <BATCH>
# Chains (one per GPU, run serially within each chain):
#   cuda:0  REC:   BF16 -> W4-RTN -> W3-RTN -> floor        (grounding gap + aggressive width)
#   cuda:1  VQA:   BF16 -> W4-RTN -> W3-RTN -> floor(2500)
#   cuda:2  POPE:  BF16 -> W4-RTN -> W3-RTN -> floor(3000)
#   cuda:3  REC grid extras: W8-RTN -> D_dev BF16 -> D_dev W4-RTN  (grid start + selection-set baselines)
source /usr4/spclpgm/eric1/GCQ/code/env.sh
B=${1:?batch size required}
cd /usr4/spclpgm/eric1/GCQ/code
M=Qwen/Qwen3-VL-2B-Instruct
case "$2" in
  rec)
    $PYT eval_rec.py --model $M --subset rec_eval_refcoco_val_1k --tag bf16_rec1k   --batch $B --device cuda:0 &&
    $PYT eval_rec.py --model $M --subset rec_eval_refcoco_val_1k --tag w4rtn_rec1k  --rtn-bits 4 --batch $B --device cuda:0 &&
    $PYT eval_rec.py --model $M --subset rec_eval_refcoco_val_1k --tag w3rtn_rec1k  --rtn-bits 3 --batch $B --device cuda:0 &&
    $PYT eval_rec.py --model $M --subset rec_eval_refcoco_val_1k --tag bf16_rec1k_FLOOR --blank-image --batch $B --device cuda:0 ;;
  vqa)
    $PYT eval_vqa.py --model $M --task vqa --tag bf16_vqa5k  --batch $B --device cuda:1 &&
    $PYT eval_vqa.py --model $M --task vqa --tag w4rtn_vqa5k --rtn-bits 4 --batch $B --device cuda:1 &&
    $PYT eval_vqa.py --model $M --task vqa --tag w3rtn_vqa5k --rtn-bits 3 --batch $B --device cuda:1 &&
    $PYT eval_vqa.py --model $M --task vqa --tag bf16_vqa_FLOOR --blank-image --limit 2500 --batch $B --device cuda:1 ;;
  pope)
    $PYT eval_vqa.py --model $M --task pope --tag bf16_pope  --batch $B --device cuda:2 &&
    $PYT eval_vqa.py --model $M --task pope --tag w4rtn_pope --rtn-bits 4 --batch $B --device cuda:2 &&
    $PYT eval_vqa.py --model $M --task pope --tag w3rtn_pope --rtn-bits 3 --batch $B --device cuda:2 &&
    $PYT eval_vqa.py --model $M --task pope --tag bf16_pope_FLOOR --blank-image --limit 3000 --batch $B --device cuda:2 ;;
  grid)
    $PYT eval_rec.py --model $M --subset rec_eval_refcoco_val_1k --tag w8rtn_rec1k --rtn-bits 8 --batch $B --device cuda:3 &&
    $PYT eval_rec.py --model $M --subset ddev_refcoco_val_1k --tag bf16_ddev1k  --batch $B --device cuda:3 &&
    $PYT eval_rec.py --model $M --subset ddev_refcoco_val_1k --tag w4rtn_ddev1k --rtn-bits 4 --batch $B --device cuda:3 ;;
  *) echo "usage: launch_queues.sh <batch> rec|vqa|pope|grid"; exit 2 ;;
esac
