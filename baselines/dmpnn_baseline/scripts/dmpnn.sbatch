#!/bin/bash
#SBATCH --account=bhrm-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=dmpnn
#SBATCH --output=/work/hdd/bhrm/jchan5/mole-rag_naacl/dmpnn_baseline/logs/%x_%j.out
#SBATCH --error=/work/hdd/bhrm/jchan5/mole-rag_naacl/dmpnn_baseline/logs/%x_%j.err
set -u
ROOT=/work/hdd/bhrm/jchan5/mole-rag_naacl/dmpnn_baseline
PY=/u/jchan5/miniconda3/envs/cp_dmpnn/bin/python
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd "$ROOT"
echo "HOST=$(hostname) START=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
$PY scripts/run_dmpnn.py --seeds 0,1,2 --epochs 50 --minutes 52 --accelerator gpu
echo "rc=$? END=$(date -Is)"
