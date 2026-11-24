#!/usr/bin/env bash
set -u
RUN_DIR="${EXP70_RUN_DIR:-$(cd "$(dirname "$0")/.." && pwd)/corpus_runs/bbbp}"
PY="${PY:-python3}"
export EXP70_RUN_DIR="$RUN_DIR"
cd "$RUN_DIR/code"
# wait for the clean set (bounded ~15 min)
for i in $(seq 1 300); do [ -f "$RUN_DIR/data/clean_pmcids.txt" ] && break; sleep 3; done
echo "[$(date -u +%H:%M:%S)] clean set present=$([ -f "$RUN_DIR/data/clean_pmcids.txt" ] && echo yes || echo NO); launching phase"
# 1) clean-set S3 download (fills any undownloaded clean articles) + metadata
setsid nohup "$PY" "$RUN_DIR/code/clean_download.py" >> "$RUN_DIR/logs/clean_download.out" 2>&1 &
# 2) resume parallel chunking (clean-restricted)
setsid nohup "$PY" "$RUN_DIR/code/chunk_run.py" >> "$RUN_DIR/logs/chunk_run.out" 2>&1 &
# 3) judging shards (8 processes x 160 threads; openai limiter caps global RPS ~450)
for s in 0 1 2 3 4 5 6 7; do
  setsid nohup "$PY" "$RUN_DIR/code/judge_run.py" 8 $s >> "$RUN_DIR/logs/judge_$s.out" 2>&1 &
  sleep 0.3
done
echo "[$(date -u +%H:%M:%S)] phase launched: clean_download + chunk_run + 8 judge shards"
