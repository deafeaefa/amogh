#!/bin/bash
# GCQ run watchdog: checks every 5 min that active eval runs are making progress.
# Exits 1 (-> notification) if eval processes are running but no output has grown
# for 2 consecutive checks (10 min). Exits 0 quietly when no eval runs remain.
source /usr4/spclpgm/eric1/GCQ/code/env.sh
LAST=-1; STALLS=0
for i in $(seq 1 96); do   # up to 8 hours
  sleep 300
  RUNNING=$(pgrep -u eric1 -f "eval_(rec|vqa)|profile_sensitivity|allocate" | wc -l)
  if [ "$RUNNING" -eq 0 ]; then
    # no active runs; if queue script also gone, stop quietly
    pgrep -u eric1 -f "run_queue" > /dev/null || { echo "watchdog: no eval runs active, exiting clean"; exit 0; }
  fi
  SIZE=$(du -cb $GCQ_RUNS/*.jsonl $GCQ_RUNS/*.csv 2>/dev/null | tail -1 | cut -f1)
  if [ "$RUNNING" -gt 0 ] && [ "$SIZE" == "$LAST" ]; then
    STALLS=$((STALLS+1))
  else
    STALLS=0
  fi
  if [ "$STALLS" -ge 2 ]; then
    echo "WATCHDOG STALL: $RUNNING eval process(es) running but no output growth for 10+ min"
    echo "--- processes ---"; ps -o pid,pcpu,etime,cmd -u eric1 | grep -E "eval_(rec|vqa)|profile_sensitivity|allocate" | grep -v grep
    echo "--- gpu ---"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    echo "--- sizes ---"; ls -la $GCQ_RUNS/*.jsonl
    exit 1
  fi
  LAST=$SIZE
done
echo "watchdog: 8h window elapsed"; exit 0
