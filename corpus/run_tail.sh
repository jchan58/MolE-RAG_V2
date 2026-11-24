#!/usr/bin/env bash
# Downstream TAIL for one dataset, no-GPU path (runs on the Delta login node where the
# Bedrock token is local). Waits for the collection bridge's COLLECT_DONE (does NOT re-run
# collect, so it never conflicts with the running run_collect bridge), then:
#   run_fulltext (download OA + chunk, bulk on /tmp, passages->S3) -> BM25 prescreen ->
#   JUDGE (Bedrock qwen.qwen3-32b-v1:0, Haiku-4.5 fallback) -> push kept chunks to S3.
# Resumable (per-item done markers + S3). Stops at kept chunks; MedCPT/structuring is the
# GPU-gated next stage (EC2 or Delta GPU).
set -u
DS="${1:?usage: run_tail.sh <dataset>}"
CODE="${CODE:-$(cd "$(dirname "$0")" && pwd)}"
source "$CODE/env_common.sh" "$DS"
export EXP70_SEARCH_CAP=300
# polite parallelism: this runs on a SHARED login node, so keep CPU footprint modest
export EXP70_CHUNK_PROCS="${EXP70_CHUNK_PROCS:-4}"
export EXP70_READ_WORKERS="${EXP70_READ_WORKERS:-8}"
export EXP70_RETRIEVE_THREADS="${EXP70_RETRIEVE_THREADS:-4}"
LOG="$EXP70_RUN_DIR/logs"; mkdir -p "$LOG" "$EXP70_FILTER_BASE"

echo "[$(date -u +%H:%M:%S)] [$DS] tail waiting for COLLECT_DONE..."
for i in $(seq 1 720); do
  [ -f "$EXP70_RUN_DIR/state/COLLECT_DONE" ] && break
  sleep 30
done
if [ ! -f "$EXP70_RUN_DIR/state/COLLECT_DONE" ]; then
  echo "[$DS] COLLECT_DONE not reached in time; exiting tail"; exit 1
fi
echo "[$(date -u +%H:%M:%S)] [$DS] COLLECT_DONE -> downstream"

# background PROGRESS pusher
( while true; do "$PY" "$CODE/progress.py" >/dev/null 2>&1 || true; sleep 120; done ) &
PP=$!; trap 'kill $PP 2>/dev/null || true' EXIT

# 1) fulltext + chunk (resumable; bulk on /tmp, passages tar->S3)
if [ ! -f "$EXP70_RUN_DIR/state/FULLTEXT_DONE" ]; then
  echo "[$(date -u +%H:%M:%S)] [$DS] FULLTEXT+CHUNK"
  "$PY" "$CODE/run_fulltext.py" >> "$LOG/fulltext.out" 2>&1 || echo "[$DS] fulltext nonzero"
fi
# 2) BM25 prescreen
echo "[$(date -u +%H:%M:%S)] [$DS] BM25"
"$PY" "$CODE/bm25_prescreen.py" >> "$LOG/bm25.out" 2>&1 || echo "[$DS] bm25 nonzero"
# 3) JUDGE (Bedrock qwen3-32b, Haiku fallback) — restricted to BM25 CANDIDATE articles only
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
print("candidate articles:",len(seen))
PYEOF
  export EXP70_ASSIGN="$EXP70_FILTER_BASE/candidate_pmcids.txt"
fi
echo "[$(date -u +%H:%M:%S)] [$DS] JUDGE qwen3-32b (candidates only)"
export EXP70_BACKEND=qwen EXP70_STREAM=qwen EXP70_CLAUDE_FALLBACK=1
export EXP70_CLAUDE_MODELS="us.anthropic.claude-haiku-4-5-20251001-v1:0"
export EXP70_JUDGE_THREADS="${EXP70_JUDGE_THREADS:-32}" EXP70_SAVE_RAW=0
NSH="${EXP70_JUDGE_SHARDS:-6}"; jp=""
for s in $(seq 0 $((NSH-1))); do
  "$PY" "$CODE/judge_opus.py" "$NSH" "$s" >> "$LOG/judge_$s.out" 2>&1 & jp="$jp $!"; sleep 0.2
done
wait $jp 2>/dev/null
"$PY" "$CODE/progress.py" >/dev/null 2>&1 || true
touch "$EXP70_RUN_DIR/state/JUDGE_DONE"
echo "[$(date -u +%H:%M:%S)] [$DS] TAIL COMPLETE (kept chunks -> S3)"
