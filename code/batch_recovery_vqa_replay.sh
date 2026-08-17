#!/bin/bash -l
#$ -P rise-tower
#$ -N gcq-vqa-replay
#$ -q l40s
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_m=40G
#$ -l h_rt=02:00:00
#$ -pe omp 4
#$ -j y
#$ -o /projectnb/rise-tower/eric1/GCQ/runs/recovery_vqa_replay/logs

set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh
cd /usr4/spclpgm/eric1/GCQ/code

ROOT="$GCQ_RUNS/recovery_vqa_replay"
TAG="gcq425_lora_cwce_vqa50_g5_lr5e5_s0"
LAUNCH="$ROOT/training_launch_manifest.json"
DATA="$GCQ_DATA/subsets/recovery_train_vqa_replay_12k.json"
META="$GCQ_DATA/subsets/recovery_train_vqa_replay_12k.meta.json"
PROMOTE="$GCQ_RUNS/promote_gcq_b4.25.json"
PROTOCOL="/usr4/spclpgm/eric1/GCQ/code/recovery_vqa_replay_protocol.json"
TRAINER="/usr4/spclpgm/eric1/GCQ/code/train_recovery.py"
BATCH_SCRIPT="/usr4/spclpgm/eric1/GCQ/code/batch_recovery_vqa_replay.sh"

if [[ ! -s "$LAUNCH" ]] || [[ "$(jq -r '.recipe_id' "$LAUNCH")" != "$TAG" ]] || \
   [[ "$(jq -r '.planned_optimizer_steps' "$LAUNCH")" != "750" ]] || \
   [[ "$(jq -r '.checkpoint_steps | @json' "$LAUNCH")" != '[200,300,400,500,600,750]' ]]; then
  echo "frozen balanced-replay training launch manifest is missing or invalid" >&2
  exit 2
fi

check_hash() {
  local key="$1" path="$2" expected actual
  expected=$(jq -er --arg key "$key" '.[$key]' "$LAUNCH")
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "training launch hash mismatch for $key: $path" >&2
    exit 2
  fi
}

check_hash data_sha256 "$DATA"
check_hash metadata_sha256 "$META"
check_hash promotion_sha256 "$PROMOTE"
check_hash protocol_sha256 "$PROTOCOL"
check_hash trainer_sha256 "$TRAINER"
check_hash batch_script_sha256 "$BATCH_SCRIPT"

"$PYT" "$TRAINER" \
  --train-file "$DATA" \
  --output-dir "$ROOT/adapters/$TAG" \
  --objective cwce --gamma 5 \
  --rtn-bits 4 --rtn-group 128 --promote-file "$PROMOTE" \
  --rank 16 --alpha 32 --dropout 0.05 \
  --learning-rate 5e-5 --weight-decay 0 \
  --epochs 1 --batch-size 2 --grad-accum 8 --warmup-ratio 0.03 \
  --save-every 50 --seed 0 --device cuda:0
