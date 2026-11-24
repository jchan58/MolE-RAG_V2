#!/bin/bash
#SBATCH --account=bhrm-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=crmn
#SBATCH --output=/workspace/molerag_reasoning/claude/chemrag_moleculenet_9task_qwen_v1/jobs/logs/%x_%j.out
#SBATCH --error=/workspace/molerag_reasoning/claude/chemrag_moleculenet_9task_qwen_v1/jobs/logs/%x_%j.err
set -u
ROOT=/workspace/molerag_reasoning/claude/chemrag_moleculenet_9task_qwen_v1
PY=/home/user/miniconda3/bin/python3
SITE=/workspace/site-packages
JHOME=/workspace/java21/jdk-21.0.11+10
mkdir -p "$ROOT/jobs/logs"; cd "$ROOT"
echo "HOST=$(hostname) START=$(date -Is) JOB=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
NQ=16648
cnt(){ [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
BM=$(cnt retrieval/bm25_hits.jsonl)
DE=$(cnt retrieval/dense_hits.jsonl)
GEN=$(cat outputs/*_preds.jsonl 2>/dev/null | wc -l | tr -d ' ')
echo "STATUS bm25=$BM/$NQ dense=$DE/$NQ gen=$GEN/$NQ fused=$([ -f retrieval/fused_docs.jsonl ] && echo yes || echo no)"

if [ "$BM" -lt "$NQ" ]; then
  echo "=== STAGE BM25 ==="
  export JAVA_HOME=$JHOME PATH=$JHOME/bin:$PATH JVM_PATH=$JHOME/lib/server/libjvm.so PYTHONPATH=$SITE PYTHONUNBUFFERED=1
  $PY retrieval/bm25_run.py --threads 64 --minutes 50
elif [ "$DE" -lt "$NQ" ]; then
  echo "=== STAGE DENSE ==="
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH
  $PY retrieval/dense_run.py
elif [ ! -f retrieval/fused_docs.jsonl ]; then
  echo "=== STAGE FUSE+FETCH ==="
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH
  $PY retrieval/fuse_top5.py && $PY retrieval/content_fetch.py
elif [ "$GEN" -lt "$NQ" ]; then
  echo "=== STAGE GENERATE ==="
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH
  $PY generate_score.py --regress --classify   # single process -> shared 52-min budget before the wall
else
  echo "=== ALL STAGES COMPLETE ==="
  export PYTHONNOUSERSITE=1; unset PYTHONPATH
  $PY generate_score.py --score
fi
echo "rc=$? END=$(date -Is)"
