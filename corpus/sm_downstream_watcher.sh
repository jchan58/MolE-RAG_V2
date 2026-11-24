#!/usr/bin/env bash
# Generic SageMaker downstream orchestrator for ONE dataset (parametrized copy of hiv_watcher_sm.sh).
# Routes a dataset's heavy download+chunk+BM25 to a dedicated SageMaker box (parallel, off the shared
# login node), then runs the Bedrock judge on Delta. Idempotent + relaunch-safe (no flock -- last-wins
# on idempotent S3/sentinels; single-launcher discipline). Usage: sm_downstream_watcher.sh <ds> <itype>
#   COLLECT_DONE        -> sync collect_state to S3 -> submit SageMaker chunk/BM25
#   CPU_DOWNSTREAM_DONE -> pull candidates from S3 -> Delta Bedrock-qwen3-32b judge -> JUDGE_DONE
# (structuring_watcher.sh then does Cohere rerank + extract + v2 cards on JUDGE_DONE)
set -u
DS="${1:?usage: sm_downstream_watcher.sh <ds> <itype>}"; ITYPE="${2:-c6i.8xlarge}"
CODE=/work/hdd/bhrm/jchan5/exp70_multi/code; PY=/u/jchan5/miniconda3/bin/python3
export AWS_DEFAULT_REGION=us-east-1
RUN=/work/hdd/bhrm/jchan5/exp70_multi/$DS
mkdir -p "$RUN/logs" "$RUN/state"
sm_status() { timeout 25 aws sagemaker describe-processing-job --region us-east-1 --processing-job-name "$1" --query ProcessingJobStatus --output text 2>/dev/null; }

for iter in $(seq 1 8640); do
  [ -f "$RUN/state/JUDGE_DONE" ] && { echo "[$DS] JUDGE_DONE -> downstream watcher done"; break; }

  if [ -f "$RUN/state/COLLECT_DONE" ]; then
    resubmit=0
    if [ -f "$RUN/state/SM_SUBMITTED" ]; then
      J=$(cat "$RUN/state/SM_SUBMITTED" 2>/dev/null); st=$(sm_status "$J")
      case "$st" in Failed|Stopped) [ ! -f "$RUN/state/.cpu_done_seen" ] && { echo "[$DS] SM $st -> resubmit"; rm -f "$RUN/state/SM_SUBMITTED"; resubmit=1; };; esac
    else resubmit=1; fi
    if [ "$resubmit" = 1 ] && [ ! -f "$RUN/state/SM_SUBMITTED" ]; then
      echo "[$DS] COLLECT_DONE -> sync collect_state + submit SageMaker ($ITYPE)"
      aws s3 sync "$RUN/data/"  "s3://molerag/${DS}_chunks/collect_state/data/"  --only-show-errors --exclude "*.lock" --exclude "locks/*"
      aws s3 sync "$RUN/state/" "s3://molerag/${DS}_chunks/collect_state/state/" --only-show-errors --exclude "*.lock" --exclude "locks/*"
      JOB=$("$PY" "$CODE/sm_launch_chunk_bm25.py" "$DS" "$ITYPE" 500 2>>"$RUN/logs/sm_launch.out" | tail -1)
      [ -n "$JOB" ] && { echo "$JOB" > "$RUN/state/SM_SUBMITTED"; echo "[$DS] submitted SageMaker $JOB"; }
    fi
  fi

  if [ -f "$RUN/state/SM_SUBMITTED" ] && [ ! -f "$RUN/state/JUDGE_DONE" ]; then
    if aws s3 ls "s3://molerag/${DS}_chunks/CPU_DOWNSTREAM_DONE.txt" >/dev/null 2>&1; then
      touch "$RUN/state/.cpu_done_seen"
      echo "[$DS] CPU_DOWNSTREAM_DONE -> Delta judge"
      ( source "$CODE/env_common.sh" "$DS"
        mkdir -p "$EXP70_FILTER_BASE/chunks" "$EXP70_FILTER_BASE/bm25"
        aws s3 cp "s3://molerag/${DS}_chunks/passages/passages_${DS}.tar.gz" /tmp/ph_${DS}.tgz --only-show-errors && tar xzf /tmp/ph_${DS}.tgz -C "$EXP70_FILTER_BASE/chunks/" 2>/dev/null
        aws s3 sync "s3://molerag/${DS}_chunks/candidates/" "$EXP70_FILTER_BASE/bm25/" --only-show-errors
        CJ="$EXP70_FILTER_BASE/bm25/candidates_passages.jsonl"
        [ -f "$CJ" ] && "$PY" - "$CJ" "$EXP70_FILTER_BASE/candidate_pmcids.txt" <<'PYE'
import sys,json
s=set()
for l in open(sys.argv[1]):
    if l.strip(): s.add(str(json.loads(l).get('pmcid')))
open(sys.argv[2],'w').write("\n".join(x for x in s if x)+"\n")
PYE
        export EXP70_ASSIGN="$EXP70_FILTER_BASE/candidate_pmcids.txt"
        export EXP70_BACKEND=qwen EXP70_STREAM=qwen EXP70_CLAUDE_FALLBACK=1 EXP70_SAVE_RAW=0
        export EXP70_CLAUDE_MODELS="us.anthropic.claude-opus-4-6-v1,us.anthropic.claude-opus-4-5-20251101-v1:0" EXP70_JUDGE_THREADS=32
        jp=""; for s in 0 1 2 3 4 5; do "$PY" "$CODE/judge_opus.py" 6 $s >> "$RUN/logs/judge_$s.out" 2>&1 & jp="$jp $!"; sleep 0.2; done
        wait $jp 2>/dev/null
        "$PY" "$CODE/progress.py" >/dev/null 2>&1 || true )
      touch "$RUN/state/JUDGE_DONE"
      echo "done" | aws s3 cp - "s3://molerag/${DS}_chunks/JUDGE_DONE.txt" --only-show-errors 2>/dev/null || true
      echo "[$DS] JUDGE_DONE"
    fi
  fi
  sleep 60
done
