#!/bin/bash
#SBATCH --job-name=gin_naacl
#SBATCH --account=bhrm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/workspace/mole-rag_naacl/gin/jobs/gin_%j.out
#SBATCH --error=/workspace/mole-rag_naacl/gin/jobs/gin_%j.err

# GIN (trained from scratch) supervised baseline over the 9 MoleculeNet tasks, seed-0 split,
# 3 initialization seeds, lr grid {1e-3,1e-4}, best checkpoint by validation. Same test molecules
# as MolE-RAG.

set -euo pipefail
ENV=/workspace/envs/gin
GIN=/workspace/mole-rag_naacl/gin

export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$GIN"

echo "host=$(hostname) date=$(date)"
"$ENV/bin/python" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

# Datasets ordered small -> large so early results land fast; hiv last.
"$ENV/bin/python" train_gin.py \
    --datasets freesolv esol lipo bace bbbp clintox sider tox21 hiv \
    --seeds 0 1 2 \
    --split-seed 0 \
    --lrs 1e-3 1e-4 \
    --dropout 0.5 \
    --out-dir "$GIN/seed_0" \
    --device cuda

echo "DONE $(date)"
