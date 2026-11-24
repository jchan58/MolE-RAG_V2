#!/usr/bin/env bash
# Keep chunk -> judge iterating (both incremental/resumable) until the clean set is
# fully chunked + judged. clean_download.py runs separately/continuously, adding
# downloaded articles. async_judge is run to completion each pass (never overlapped,
# so the global 450/s OpenAI budget is never exceeded).
set -u
RUN_DIR="/scratch/bhrm/jchan5/exp70_bbbp_pmc_collection"
PY="/u/jchan5/miniconda3/bin/python3"
export EXP70_RUN_DIR="$RUN_DIR"
cd "$RUN_DIR/code"
prev=-1
for iter in $(seq 1 200); do
  echo "[$(date -u +%H:%M:%S)] iter $iter: chunk pass"
  "$PY" chunk_run.py >> "$RUN_DIR/logs/chunk_run.out" 2>&1
  echo "[$(date -u +%H:%M:%S)] iter $iter: judge pass"
  "$PY" async_judge.py 1 0 >> "$RUN_DIR/logs/async_judge.out" 2>&1
  done_now=$(find "$RUN_DIR/judgments/done" -name '*.done' 2>/dev/null | wc -l)
  dl_phase=$("$PY" - <<PYEOF 2>/dev/null
import json
try: print(json.load(open("$RUN_DIR/state/CLEAN_DL_PROGRESS.json")).get("phase"))
except Exception: print("unknown")
PYEOF
)
  echo "[$(date -u +%H:%M:%S)] iter $iter: judged_done=$done_now clean_dl_phase=$dl_phase"
  if [ "$dl_phase" = "done" ] && [ "$done_now" = "$prev" ]; then
    echo "[$(date -u +%H:%M:%S)] ALL CAUGHT UP (clean download done, no new judgments)"; break
  fi
  prev=$done_now
  sleep 10
done
echo "[$(date -u +%H:%M:%S)] pipeline_loop finished"
