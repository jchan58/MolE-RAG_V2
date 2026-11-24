#!/bin/bash
set -u
RUN_DIR="${EXP70_RUN_DIR:-$(cd "$(dirname "$0")/.." && pwd)/corpus_runs/bbbp}"; PY="${PY:-python3}"
export EXP70_RUN_DIR="$RUN_DIR"; cd "$RUN_DIR/code"
for i in $(seq 1 500); do
  "$PY" chunk_run.py >> "$RUN_DIR/logs/chunk_run.out" 2>&1
  ph=$("$PY" -c "import json;print(json.load(open('$RUN_DIR/state/CLEAN_DL_PROGRESS.json')).get('phase'))" 2>/dev/null)
  [ "$ph" = "done" ] && { "$PY" chunk_run.py >> "$RUN_DIR/logs/chunk_run.out" 2>&1; echo "chunk_loop: clean_download done, final chunk pass complete"; break; }
  sleep 45
done
