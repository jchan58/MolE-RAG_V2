#!/bin/bash
#SBATCH --account=bhrm-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=dmpnn_smoke
#SBATCH --output=/work/hdd/bhrm/jchan5/mole-rag_naacl/dmpnn_baseline/logs/%x_%j.out
#SBATCH --error=/work/hdd/bhrm/jchan5/mole-rag_naacl/dmpnn_baseline/logs/%x_%j.err
set -u
ROOT=/work/hdd/bhrm/jchan5/mole-rag_naacl/dmpnn_baseline
CP=/u/jchan5/miniconda3/envs/cp_dmpnn/bin/chemprop
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONNOUSERSITE=1
cd "$ROOT"
echo "HOST=$(hostname) START=$(date -Is)"
$CP train --data-path data/freesolv/combined.csv --smiles-columns smiles --splits-column split \
  --task-type regression --metrics rmse mae \
  --output-dir runs/_smoke_freesolv --epochs 5 --pytorch-seed 0 \
  --accelerator gpu --devices 1 --num-workers 0
echo "rc=$? END=$(date -Is)"
echo "=== output tree ==="; find runs/_smoke_freesolv -maxdepth 2 -type f | head -40
