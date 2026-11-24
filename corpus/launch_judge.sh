#!/usr/bin/env bash
# Launch the exp70-multi relevance JUDGE for a dataset using Bedrock qwen.qwen3-32b-v1:0
# (EXP70_BACKEND=qwen; 3 valid US regions + backoff). Haiku 4.5 fallback via EXP70_CLAUDE_FALLBACK.
# Reads chunked passages from EXP70_FILTER_BASE/chunks/passages; outputs judgments to
# EXP70_FILTER_BASE/judgments_qwen + s3://molerag/<ds>_chunks/qwen_judgments/. Resumable
# (per-article done markers). Shardable across processes.
set -u
DS="${1:?usage: launch_judge.sh <dataset> [nshards]}"
NSHARDS="${2:-8}"
source "$(dirname "$0")/env_common.sh" "$DS"
export EXP70_BACKEND=qwen
export EXP70_STREAM=qwen
export EXP70_CLAUDE_FALLBACK="${EXP70_CLAUDE_FALLBACK:-1}"
export EXP70_CLAUDE_MODELS="us.anthropic.claude-haiku-4-5-20251001-v1:0"   # fallback pool (Haiku 4.5)
export EXP70_JUDGE_THREADS="${EXP70_JUDGE_THREADS:-48}"
export EXP70_GROUP="${EXP70_GROUP:-6}"
export EXP70_BATCH="${EXP70_BATCH:-120}"
export EXP70_CHARBUDGET="${EXP70_CHARBUDGET:-60000}"
export EXP70_MAXTOK="${EXP70_MAXTOK:-5000}"
export EXP70_SAVE_RAW=0
LOG="$EXP70_RUN_DIR/logs"
mkdir -p "$LOG"
echo "[judge $DS] launching $NSHARDS shards x $EXP70_JUDGE_THREADS threads (qwen3-32b + Haiku fallback)"
for s in $(seq 0 $((NSHARDS-1))); do
  setsid nohup "$PY" "$CODE/judge_opus.py" "$NSHARDS" "$s" >> "$LOG/judge_${s}.out" 2>&1 &
  echo "  shard $s pid $!"
  sleep 0.2
done
echo "[judge $DS] launched; progress -> $EXP70_FILTER_BASE/PROGRESS_qwen.json + s3"
