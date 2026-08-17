#!/bin/bash
set -euo pipefail
source /usr4/spclpgm/eric1/GCQ/code/env.sh

PILOT_ROOT="$GCQ_RUNS/recovery_pilot"
mkdir -p "$PILOT_ROOT/logs" "$PILOT_ROOT/adapters" "$PILOT_ROOT/eval"

TRAIN_MANIFEST="$GCQ_DATA/subsets/recovery_train_8k.json"
DEV_MANIFEST="$GCQ_DATA/subsets/recovery_dev_1k.json"
if [[ ! -s "$TRAIN_MANIFEST" || ! -s "$DEV_MANIFEST" ]]; then
  echo "recovery manifests are missing; run build_recovery_data.py first" >&2
  exit 2
fi

MISSING=$(
  "$PYT" - "$TRAIN_MANIFEST" "$DEV_MANIFEST" "$GCQ_DATA/images/train2014" <<'PY'
import json, os, sys
paths = set()
for manifest in sys.argv[1:3]:
    with open(manifest) as f:
        paths.update(row["file_name"] for row in json.load(f))
print(sum(not os.path.exists(os.path.join(sys.argv[3], name)) for name in paths))
PY
)
if [[ "$MISSING" != 0 ]]; then
  echo "$MISSING recovery images are missing; run fetch_recovery_images.py first" >&2
  exit 2
fi

qsub /usr4/spclpgm/eric1/GCQ/code/batch_recovery_pilot.sh

