#!/bin/bash
#SBATCH --account=bhrm-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=cr_gen
#SBATCH --output=/work/hdd/bhrm/jchan5/molerag_reasoning/claude/chemrag_moleculenet_9task_qwen_v1/jobs/logs/%x_%j.out
#SBATCH --error=/work/hdd/bhrm/jchan5/molerag_reasoning/claude/chemrag_moleculenet_9task_qwen_v1/jobs/logs/%x_%j.err
set -u
ROOT=/work/hdd/bhrm/jchan5/molerag_reasoning/claude/chemrag_moleculenet_9task_qwen_v1
PY=/u/jchan5/miniconda3/bin/python3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
mkdir -p "$ROOT/jobs/logs"; cd "$ROOT"
echo "HOST=$(hostname) START=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
# regression first (small, generative), then classification (forced-choice, the bulk). Resumable (skip-done).
$PY generate_score.py --regress
$PY generate_score.py --classify
echo "rc=$? END=$(date -Is)"
