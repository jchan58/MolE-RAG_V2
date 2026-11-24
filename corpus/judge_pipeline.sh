#!/usr/bin/env bash
# Proven threaded path: alternate a CHUNK pass (incremental) with a JUDGE pass of
# 32 independent shard processes (judge_run.py 32 <s>), while clean_download runs
# continuously in the background. Sequential chunk/judge passes keep Lustre
# contention down; 32 processes x ~160 threads use the 5x OpenAI headroom.
set -u
RUN_DIR="${EXP70_RUN_DIR:-$(cd "$(dirname "$0")/.." && pwd)/corpus_runs/bbbp}"
PY="${PY:-python3}"
export EXP70_RUN_DIR="$RUN_DIR"
cd "$RUN_DIR/code"
NSHARDS=32
prev=-1
for iter in $(seq 1 500); do
  echo "[$(date -u +%H:%M:%S)] iter $iter: CHUNK pass"
  "$PY" chunk_run.py >> "$RUN_DIR/logs/chunk_run.out" 2>&1
  echo "[$(date -u +%H:%M:%S)] iter $iter: JUDGE pass ($NSHARDS shards)"
  pids=""
  for s in $(seq 0 $((NSHARDS-1))); do
    setsid nohup "$PY" judge_run.py $NSHARDS $s >> "$RUN_DIR/logs/judge_$s.out" 2>&1 &
    pids="$pids $!"
    sleep 0.1
  done
  wait $pids 2>/dev/null
  done_now=$(find "$RUN_DIR/judgments/done" -name '*.done' 2>/dev/null | wc -l)
  dl_phase=$("$PY" - <<PYEOF 2>/dev/null
import json
try: print(json.load(open("$RUN_DIR/state/CLEAN_DL_PROGRESS.json")).get("phase"))
except Exception: print("unknown")
PYEOF
)
  echo "[$(date -u +%H:%M:%S)] iter $iter: judged_done=$done_now clean_dl_phase=$dl_phase"
  if [ "$dl_phase" = "done" ] && [ "$done_now" = "$prev" ]; then
    echo "[$(date -u +%H:%M:%S)] ALL CAUGHT UP"; break
  fi
  prev=$done_now
  sleep 5
done
echo "[$(date -u +%H:%M:%S)] judge_pipeline finished"
