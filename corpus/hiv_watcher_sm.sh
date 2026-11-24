#!/usr/bin/env bash
# HIV downstream orchestrator -- SageMaker path (no GPU/MedCPT; Cohere handles rerank).
# Idempotent + relaunch-safe (flock singleton + /work sentinels + S3 markers). Survives session
# drops: re-run this exact command and it resumes where it left off.
#   COLLECT_DONE         -> sync hiv collect_state to S3 -> submit SageMaker c6i.24xlarge
#                           (download+chunk+BM25 -> passages+candidates+CPU_DOWNSTREAM_DONE to S3)
#   CPU_DOWNSTREAM_DONE  -> pull candidates from S3 -> Delta Bedrock-qwen3-32b judge -> JUDGE_DONE
#   (structuring_watcher.sh then does Cohere rerank + extract + v2 cards on JUDGE_DONE)
set -u
CODE=${CODE:-$(cd "$(dirname "$0")" && pwd)}; PY=${PY:-python3}
export AWS_DEFAULT_REGION=us-east-1
DS=hiv; RUN=${EXP70_BASE:-$(cd "$(dirname "$0")/.." && pwd)/corpus_runs}/$DS
mkdir -p "$RUN/logs" "$RUN/state"
# singleton: only one orchestrator at a time (relaunch-safe)
exec 9>"$RUN/state/.hiv_watcher.lock"
if ! flock -n 9; then echo "[hiv] another orchestrator holds the lock; exiting"; exit 0; fi

sm_status() { timeout 25 aws sagemaker describe-processing-job --region us-east-1 \
  --processing-job-name "$1" --query ProcessingJobStatus --output text 2>/dev/null; }

for iter in $(seq 1 8640); do   # ~6 days @60s
  [ -f "$RUN/state/JUDGE_DONE" ] && { echo "[hiv] JUDGE_DONE -> orchestrator done"; break; }

  # 1) COLLECT_DONE -> submit SageMaker chunk/BM25 (resubmit only if a prior job failed)
  if [ -f "$RUN/state/COLLECT_DONE" ]; then
    resubmit=0
    if [ -f "$RUN/state/HIV_SM_SUBMITTED" ]; then
      J=$(cat "$RUN/state/HIV_SM_SUBMITTED" 2>/dev/null)
      st=$(sm_status "$J")
      case "$st" in
        Failed|Stopped) [ ! -f "$RUN/state/.cpu_done_seen" ] && { echo "[hiv] prior SM job $st -> resubmit"; rm -f "$RUN/state/HIV_SM_SUBMITTED"; resubmit=1; };;
      esac
    else
      resubmit=1
    fi
    if [ "$resubmit" = 1 ] && [ ! -f "$RUN/state/HIV_SM_SUBMITTED" ]; then
      echo "[hiv] COLLECT_DONE -> sync collect_state to S3 + submit SageMaker"
      aws s3 sync "$RUN/data/"  "s3://molerag/${DS}_chunks/collect_state/data/"  --only-show-errors
      aws s3 sync "$RUN/state/" "s3://molerag/${DS}_chunks/collect_state/state/" --only-show-errors --exclude "*.lock" --exclude "locks/*"
      JOB=$("$PY" "$CODE/sm_launch_chunk_bm25.py" "$DS" c6i.24xlarge 2>>"$RUN/logs/sm_launch.out" | tail -1)
      if [ -n "$JOB" ]; then
        echo "$JOB" > "$RUN/state/HIV_SM_SUBMITTED"
        aws s3 cp "$RUN/state/HIV_SM_SUBMITTED" "s3://molerag/${DS}_chunks/HIV_SM_SUBMITTED.txt" --only-show-errors
        echo "[hiv] submitted SageMaker job $JOB"
      fi
    fi
  fi

  # 2) CPU_DOWNSTREAM_DONE in S3 -> Delta judge (Bedrock, candidates only)
  if [ -f "$RUN/state/HIV_SM_SUBMITTED" ] && [ ! -f "$RUN/state/JUDGE_DONE" ]; then
    if aws s3 ls "s3://molerag/${DS}_chunks/CPU_DOWNSTREAM_DONE.txt" >/dev/null 2>&1; then
      touch "$RUN/state/.cpu_done_seen"
      echo "[hiv] CPU_DOWNSTREAM_DONE -> Delta judge"
      ( source "$CODE/env_common.sh" hiv
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
      aws s3 cp "$RUN/state/JUDGE_DONE" "s3://molerag/${DS}_chunks/JUDGE_DONE.txt" --only-show-errors 2>/dev/null || \
        echo "done" | aws s3 cp - "s3://molerag/${DS}_chunks/JUDGE_DONE.txt" --only-show-errors
      echo "[hiv] JUDGE_DONE"
    fi
  fi
  sleep 60
done
