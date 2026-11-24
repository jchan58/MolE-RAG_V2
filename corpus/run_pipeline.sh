#!/usr/bin/env bash
# In-job per-dataset orchestrator (called by pipeline.sbatch on a compute node).
# Resumable end-to-end: collect (ingest->identity->Opus querygen->ESearch->merge) ->
# fulltext+chunk -> BM25 prescreen -> Qwen3-32B judge. Pushes S3 PROGRESS.json every ~90s.
# State on /work survives node death; bulk on node-local /tmp; everything idempotent.
set -u
DS="${1:?usage: run_pipeline.sh <dataset>}"
CODE="/work/hdd/bhrm/jchan5/exp70_multi/code"
source "$CODE/env_common.sh" "$DS"

# --- NCBI rate: each concurrent dataset job takes a FIXED slice of the account-wide
#     10/s cap via a node-local bucket (no fragile cross-node Lustre lock). 3/s x 3 small
#     jobs ~= 9/s. Override per launch with EXP70_NCBI_RATE. Keep RL_DIR node-local. ---
export EXP70_NCBI_RATE="${EXP70_NCBI_RATE:-3}"
# --- query-gen: Opus 4.6 (+4.5), high concurrency (not throttled) ---
export EXP70_CLAUDE_MODELS="${EXP70_CLAUDE_MODELS:-us.anthropic.claude-opus-4-6-v1,us.anthropic.claude-opus-4-5-20251101-v1:0}"
export EXP70_QG_THREADS="${EXP70_QG_THREADS:-48}"
export EXP70_PROP_THREADS="${EXP70_PROP_THREADS:-12}"
# --- deadline levers. SEARCH_CAP caps retrieval DEPTH per query (safe: already-done
#     searches keep their lists). MOL_QCAP (queries/molecule) is UNSET by default (0):
#     capping it after an uncapped collection would drop molecule associations at merge.
#     Pass EXP70_MOL_QCAP=2 ONLY for a FRESH dataset (e.g. HIV) that has not searched yet. ---
export EXP70_MOL_QCAP="${EXP70_MOL_QCAP:-0}"
export EXP70_SEARCH_CAP="${EXP70_SEARCH_CAP:-300}"

LOG="$EXP70_RUN_DIR/logs"; mkdir -p "$LOG" "$EXP70_FILTER_BASE"

echo "[$(date -u +%H:%M:%S)] pipeline $DS on $(hostname) job=${SLURM_JOB_ID:-none} run_dir=$EXP70_RUN_DIR"

# --- restore state from S3 if the /work run dir is empty (fresh node) ---
if [ ! -f "$EXP70_RUN_DIR/data/structures.jsonl" ]; then
  echo "[$DS] no local structures; attempting S3 restore"
  aws s3 sync "s3://molerag/${DS}_chunks/collect_state/data/" "$EXP70_RUN_DIR/data/" --only-show-errors 2>/dev/null || true
  aws s3 sync "s3://molerag/${DS}_chunks/collect_state/state/" "$EXP70_RUN_DIR/state/" --only-show-errors 2>/dev/null || true
fi

# --- background PROGRESS.json pusher (every 90s) ---
( while true; do "$PY" "$CODE/progress.py" >/dev/null 2>&1 || true; sleep 90; done ) &
PROG_PID=$!
trap 'kill $PROG_PID 2>/dev/null || true' EXIT

# --- Stage 1: COLLECT (resumable) ---
if [ ! -f "$EXP70_RUN_DIR/state/COLLECT_DONE" ]; then
  echo "[$(date -u +%H:%M:%S)] [$DS] COLLECT"
  "$PY" "$CODE/run_collect.py" >> "$LOG/collect.out" 2>&1
fi

# --- Stage 2: FULLTEXT + CHUNK (resumable; bulk on /tmp, tar->S3) ---
if [ ! -f "$EXP70_RUN_DIR/state/FULLTEXT_DONE" ]; then
  echo "[$(date -u +%H:%M:%S)] [$DS] FULLTEXT+CHUNK"
  "$PY" "$CODE/run_fulltext.py" >> "$LOG/fulltext.out" 2>&1
fi

# --- Stage 3: BM25 prescreen ---
echo "[$(date -u +%H:%M:%S)] [$DS] BM25 prescreen"
"$PY" "$CODE/bm25_prescreen.py" >> "$LOG/bm25.out" 2>&1 || echo "[$DS] bm25 returned nonzero"

# --- Stage 4: JUDGE (Bedrock qwen3-32b, Haiku fallback) — restricted to BM25 candidates ---
CANDJSONL="$EXP70_FILTER_BASE/bm25/candidates_passages.jsonl"
if [ -f "$CANDJSONL" ]; then
  "$PY" - "$CANDJSONL" "$EXP70_FILTER_BASE/candidate_pmcids.txt" <<'PYEOF'
import sys,json
seen=set()
with open(sys.argv[1]) as f, open(sys.argv[2],'w') as o:
    for l in f:
        if not l.strip(): continue
        p=str(json.loads(l).get('pmcid'))
        if p and p not in seen: seen.add(p); o.write(p+'\n')
PYEOF
  export EXP70_ASSIGN="$EXP70_FILTER_BASE/candidate_pmcids.txt"
fi
echo "[$(date -u +%H:%M:%S)] [$DS] JUDGE (qwen3-32b, candidates only)"
export EXP70_BACKEND=qwen EXP70_STREAM=qwen EXP70_CLAUDE_FALLBACK=1
export EXP70_CLAUDE_MODELS="us.anthropic.claude-haiku-4-5-20251001-v1:0"
export EXP70_JUDGE_THREADS="${EXP70_JUDGE_THREADS:-48}" EXP70_SAVE_RAW=0
NSH="${EXP70_JUDGE_SHARDS:-8}"
jpids=""
for s in $(seq 0 $((NSH-1))); do
  "$PY" "$CODE/judge_opus.py" "$NSH" "$s" >> "$LOG/judge_$s.out" 2>&1 &
  jpids="$jpids $!"; sleep 0.2
done
wait $jpids 2>/dev/null
touch "$EXP70_RUN_DIR/state/JUDGE_DONE"
"$PY" "$CODE/progress.py" >/dev/null 2>&1 || true
echo "[$(date -u +%H:%M:%S)] [$DS] PIPELINE PASS COMPLETE (judge done)"
