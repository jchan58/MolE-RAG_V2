#!/usr/bin/env bash
# Resume helper: (re)launch detached collection for any dataset whose front half is not
# finished. Idempotent — safe to run after a login-node restart (skips already-running +
# already-done). Run this (or add to a cron/loop) to keep collection advancing.
set -u
CODE="$(dirname "$0")"
for DS in sider lipo tox21 hiv; do
  source "$CODE/env_common.sh" "$DS"
  if [ -f "$EXP70_RUN_DIR/state/COLLECT_DONE" ]; then
    echo "[$DS] collect DONE"; continue
  fi
  if pgrep -u "$USER" -f "run_collect.py" | xargs -r -I{} sh -c "tr '\0' ' ' </proc/{}/environ 2>/dev/null | grep -q 'EXP70_DATASET=$DS' && echo {}" | grep -q .; then
    echo "[$DS] collect already running"; continue
  fi
  TS="$(date +%Y%m%dT%H%M%S)"; LOG="$EXP70_RUN_DIR/logs/collect_${TS}.log"
  setsid nohup "$PY" "$CODE/run_collect.py" >> "$LOG" 2>&1 &
  echo "[$DS] (re)launched collect pid $! -> $LOG"
done
