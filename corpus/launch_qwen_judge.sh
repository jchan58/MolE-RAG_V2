#!/bin/bash
# Launch the FULL-222k Qwen3-32B judge run (Bedrock, 3 regions) with a cheap Claude fallback.
# Primary: qwen.qwen3-32b-v1:0 (us-east-1/us-west-2/us-east-2). Fallback: Sonnet 4.6 + Haiku 4.5.
# Resumable off existing per-article done markers. Outputs -> /tmp/exp70_filter + S3.
set -u
CODE=/scratch/bhrm/jchan5/exp70_bbbp_pmc_collection/code
PY=/u/jchan5/miniconda3/bin/python3
LOGDIR=/tmp/exp70_filter/logs
mkdir -p "$LOGDIR"

NSHARDS=${NSHARDS:-12}

export EXP70_BACKEND=qwen
export EXP70_CHARBUDGET=${EXP70_CHARBUDGET:-60000}   # counts target overhead; +5k out < 32k ctx
export EXP70_BATCH=${EXP70_BATCH:-120}
export EXP70_MAXTOK=${EXP70_MAXTOK:-5000}
export EXP70_JUDGE_THREADS=${EXP70_JUDGE_THREADS:-64}
export EXP70_GROUP=${EXP70_GROUP:-6}
export EXP70_CLAUDE_FALLBACK=1
export EXP70_CLAUDE_MODELS="us.anthropic.claude-sonnet-4-6,us.anthropic.claude-haiku-4-5-20251001-v1:0"
export EXP70_SAVE_RAW=0   # bulk run: skip per-call raw files (verdicts+provenance are in the judgment records)

echo "launching $NSHARDS Qwen shards x $EXP70_JUDGE_THREADS threads = $((NSHARDS*EXP70_JUDGE_THREADS)) workers"
for sh in $(seq 0 $((NSHARDS-1))); do
    nohup "$PY" "$CODE/judge_opus.py" "$NSHARDS" "$sh" \
        > "$LOGDIR/qwen_shard_${sh}.log" 2>&1 &
    echo "  shard $sh pid $!"
done
echo "all shards launched; progress -> /tmp/exp70_filter/PROGRESS.json"
