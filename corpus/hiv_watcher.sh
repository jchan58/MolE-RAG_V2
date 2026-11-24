#!/usr/bin/env bash
# HIV end-to-end downstream orchestrator (detached, idempotent via /work sentinels):
#  COLLECT_DONE        -> launch c6i.12xlarge CPU-EC2 (download OA + chunk + BM25, push to S3)
#  CPU_DOWNSTREAM_DONE -> pull candidates+passages from S3, JUDGE on Delta (Bedrock qwen3-32b,
#                         restricted to BM25 candidate articles), write JUDGE_DONE
#  JUDGE_DONE          -> launch g5.12xlarge MedCPT (needs G-quota=96), then extract + indexes on Delta
set -u
CODE="/work/hdd/bhrm/jchan5/exp70_multi/code"; PY=/u/jchan5/miniconda3/bin/python3
export AWS_DEFAULT_REGION=us-east-1
DS=hiv; RUN=/work/hdd/bhrm/jchan5/exp70_multi/$DS
for iter in $(seq 1 2880); do
  [ -f "$RUN/state/STRUCT_DONE" ] && { echo "hiv structuring done"; break; }
  # 1) CPU-EC2 for chunk+BM25
  if [ -f "$RUN/state/COLLECT_DONE" ] && [ ! -f "$RUN/state/HIV_CPU_LAUNCHED" ]; then
    IID=$(bash "$CODE/launch_hiv_cpu_ec2.sh" hiv c6i.12xlarge 2>>"$RUN/logs/cpu_launch.out" | tail -1)
    echo "$IID" > "$RUN/state/HIV_CPU_LAUNCHED"; echo "[hiv] CPU-EC2 $IID"
  fi
  # 2) Delta candidate-judge once candidates are in S3
  if [ -f "$RUN/state/HIV_CPU_LAUNCHED" ] && [ ! -f "$RUN/state/JUDGE_DONE" ]; then
    if aws s3 ls "s3://molerag/${DS}_chunks/CPU_DOWNSTREAM_DONE.txt" >/dev/null 2>&1; then
      ( source "$CODE/env_common.sh" hiv
        mkdir -p "$EXP70_FILTER_BASE/chunks" "$EXP70_FILTER_BASE/bm25"
        aws s3 cp "s3://molerag/${DS}_chunks/passages/passages_${DS}.tar.gz" /tmp/ph.tgz --only-show-errors && tar xzf /tmp/ph.tgz -C "$EXP70_FILTER_BASE/chunks/"
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
        export EXP70_CLAUDE_MODELS="us.anthropic.claude-haiku-4-5-20251001-v1:0" EXP70_JUDGE_THREADS=32
        jp=""; for s in 0 1 2 3 4 5; do "$PY" "$CODE/judge_opus.py" 6 $s >> "$RUN/logs/judge_$s.out" 2>&1 & jp="$jp $!"; sleep 0.2; done
        wait $jp 2>/dev/null )
      touch "$RUN/state/JUDGE_DONE"; echo "[hiv] JUDGE_DONE"
    fi
  fi
  # 3) g5 MedCPT + extract + indexes
  if [ -f "$RUN/state/JUDGE_DONE" ] && [ ! -f "$RUN/state/MEDCPT_LAUNCHED" ]; then
    IID=$(EC2_ITYPE=g5.12xlarge bash "$CODE/launch_medcpt_ec2.sh" hiv 2>>"$RUN/logs/medcpt_launch.out" | tail -1)
    echo "$IID" > "$RUN/state/MEDCPT_LAUNCHED"; echo "[hiv] MedCPT g5 $IID"
  fi
  if [ -f "$RUN/state/MEDCPT_LAUNCHED" ] && [ ! -f "$RUN/state/STRUCT_DONE" ]; then
    if aws s3 ls "s3://molerag/${DS}_chunks/structuring/shortlists.json" >/dev/null 2>&1; then
      ( source "$CODE/env_common.sh" hiv
        export EXP70_BACKEND=qwen EXP70_CLAUDE_MODELS="us.anthropic.claude-haiku-4-5-20251001-v1:0"
        mkdir -p "$RUN/structuring"; aws s3 sync "s3://molerag/${DS}_chunks/structuring/" "$RUN/structuring/" --only-show-errors
        "$PY" "$CODE/extract.py" >> "$RUN/logs/extract.out" 2>&1
        "$PY" "$CODE/build_indexes.py" >> "$RUN/logs/indexes.out" 2>&1 )
      touch "$RUN/state/STRUCT_DONE"; echo "[hiv] STRUCT_DONE"
    fi
  fi
  sleep 60
done
