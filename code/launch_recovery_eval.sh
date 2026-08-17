#!/bin/bash
set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh
cd /usr4/spclpgm/eric1/GCQ/code
"$PYT" validate_recovery_pilot.py --require-empty-eval
mkdir -p "$GCQ_RUNS/recovery_pilot/logs" "$GCQ_RUNS/recovery_pilot/eval"
qsub /usr4/spclpgm/eric1/GCQ/code/batch_recovery_eval.sh
