#!/usr/bin/env bash
# Launch the exp70-multi FRONT-HALF collection (ingest->identity->Opus querygen->PMC
# search->merge) for a dataset, detached (survives this shell). Resumable + S3-backed.
# Usage: bash launch_collect.sh <lipo|sider|hiv|tox21>
set -u
DS="${1:?usage: launch_collect.sh <dataset>}"
source "$(dirname "$0")/env_common.sh" "$DS"
TS="$(date +%Y%m%dT%H%M%S)"
LOG="$EXP70_RUN_DIR/logs/collect_${TS}.log"
echo "[collect $DS] run_dir=$EXP70_RUN_DIR  rl_dir=$EXP70_RL_DIR  log=$LOG"
setsid nohup "$PY" "$CODE/run_collect.py" >> "$LOG" 2>&1 &
echo "[collect $DS] launched pid $! -> $LOG"
