#!/usr/bin/env bash
# Hands-off structuring orchestrator — FULLY Bedrock-API + Delta, NO GPU, NO EC2.
# Per dataset, idempotent via /work sentinels: when JUDGE_DONE appears (from run_tail.sh on
# login, or the e70_<ds> SLURM job), run on DELTA:
#   cohere_rerank.py (Bedrock cohere.embed-english-v3 cosine -> K200/400/600/800/1000, ranker=cohere_cosine)
#   -> extract.py (Qwen3-32B Bedrock, K1000 union) -> build_indexes.py (3 evidence indexes + band comparison)
# No EC2 is launched anywhere. Push results to s3://molerag/<ds>_chunks/structuring/.
#
# PARALLEL: datasets are INDEPENDENT and Bedrock-API-bound, so each ready dataset's structuring
# runs in a BACKGROUNDED subshell -> datasets structure CONCURRENTLY (not one-after-another). A
# per-dataset STRUCT_RUNNING marker (auto-removed on subshell exit) prevents a double-launch on
# the next 60s poll; STRUCT_DONE is written ONLY on full success (idempotent), so a failed run is
# retried on the next poll. Assumes a SINGLE watcher instance (as before — no flock): on startup
# any STALE STRUCT_RUNNING markers are cleared so an orphaned marker can't deadlock a dataset.
# Usage: structuring_watcher.sh "sider lipo tox21 hiv"
set -u
CODE="/work/hdd/bhrm/jchan5/exp70_multi/code"; PY=/u/jchan5/miniconda3/bin/python3
DSLIST="${1:-sider lipo tox21 hiv}"
# Bedrock extraction thread-pool size (larger K1000 union -> more concurrent in-flight calls).
export EXP70_EXTRACT_THREADS="${EXP70_EXTRACT_THREADS:-32}"

# SINGLETON (cross-node, shared Lustre lock): only ONE structuring_watcher may run at a time.
# Two instances could both pass a dataset's STRUCT_RUNNING check and launch concurrent extracts,
# whose append-mode evidence_records.jsonl + shared extract_done markers then corrupt each other.
exec 6>"/work/hdd/bhrm/jchan5/exp70_multi/.structuring_watcher.lock"
if ! flock -n 6; then echo "another structuring_watcher holds the lock; exiting"; exit 0; fi

# Startup: clear any orphaned STRUCT_RUNNING markers (this watcher is the only launcher, so on a
# fresh start nothing is actually running). STRUCT_DONE datasets are skipped regardless.
for DS in $DSLIST; do
  rm -f "/work/hdd/bhrm/jchan5/exp70_multi/$DS/state/STRUCT_RUNNING" 2>/dev/null || true
done

for iter in $(seq 1 2880); do        # ~48h at 60s
  for DS in $DSLIST; do
    RUN=/work/hdd/bhrm/jchan5/exp70_multi/$DS
    [ -f "$RUN/state/STRUCT_DONE" ] && continue
    [ -f "$RUN/state/STRUCT_RUNNING" ] && continue    # a backgrounded run is already in progress
    if [ -f "$RUN/state/JUDGE_DONE" ]; then
      mkdir -p "$RUN/state" "$RUN/logs"
      echo "[$(date -u +%H:%M:%S)] [$DS] JUDGE_DONE -> launching BACKGROUNDED structuring (cohere + extract + indexes)"
      (
        touch "$RUN/state/STRUCT_RUNNING"
        trap 'rm -f "$RUN/state/STRUCT_RUNNING"' EXIT        # always release the marker
        source "$CODE/env_common.sh" "$DS"
        export EXP70_BACKEND=qwen EXP70_CLAUDE_FALLBACK=1
        export EXP70_CLAUDE_MODELS="us.anthropic.claude-opus-4-6-v1,us.anthropic.claude-opus-4-5-20251101-v1:0"
        if "$PY" "$CODE/query_registry.py" >> "$RUN/logs/structuring.out" 2>&1 \
           && "$PY" "$CODE/cohere_rerank.py" >> "$RUN/logs/structuring.out" 2>&1 \
           && "$PY" "$CODE/extract.py"       >> "$RUN/logs/structuring.out" 2>&1 \
           && "$PY" "$CODE/build_indexes.py" >> "$RUN/logs/structuring.out" 2>&1 \
           && "$PY" "$CODE/build_cards.py"   >> "$RUN/logs/structuring.out" 2>&1; then
          touch "$RUN/state/STRUCT_DONE"
          echo "[$(date -u +%H:%M:%S)] [$DS] STRUCT_DONE (evidence indexes -> s3://molerag/${DS}_chunks/structuring/, bands 200/400/600/800/1000, ranker=cohere_cosine)"
        else
          echo "[$(date -u +%H:%M:%S)] [$DS] structuring FAILED (see $RUN/logs/structuring.out) — will retry next poll"
        fi
      ) &
    fi
  done
  alldone=1; for DS in $DSLIST; do [ -f "/work/hdd/bhrm/jchan5/exp70_multi/$DS/state/STRUCT_DONE" ] || alldone=0; done
  [ "$alldone" = 1 ] && { echo "all structuring done"; break; }
  sleep 60
done
wait        # let any in-flight backgrounded structuring subshells finish before exiting
