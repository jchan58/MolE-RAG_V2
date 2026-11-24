#!/bin/bash
#SBATCH --job-name=gimlet_zs
#SBATCH --account=bhrm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=03:00:00
#SBATCH --output=/work/hdd/bhrm/jchan5/mole-rag_naacl/gimlet_baseline/logs/slurm_%j.out
#SBATCH --error=/work/hdd/bhrm/jchan5/mole-rag_naacl/gimlet_baseline/logs/slurm_%j.out

set -e
export HF_HOME=/work/hdd/bhrm/jchan5/models
export HF_HUB_CACHE=/work/hdd/bhrm/jchan5/models/hub
export TRANSFORMERS_CACHE=/work/hdd/bhrm/jchan5/models/hub
export HF_DATASETS_CACHE=/work/hdd/bhrm/jchan5/models/datasets
export HUGGINGFACE_HUB_CACHE=/work/hdd/bhrm/jchan5/models/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

PY=/work/hdd/bhrm/jchan5/envs/gimlet/bin/python
CODE=/work/hdd/bhrm/jchan5/mole-rag_naacl/gimlet_baseline/code
OUT=/work/hdd/bhrm/jchan5/mole-rag_naacl/gimlet_baseline/outputs

echo "=== node $(hostname) $(date) ==="
nvidia-smi -L || true

echo "=== SMOKE TEST (env + frozen checkpoint + GPU generate) $(date) ==="
$PY $CODE/smoke_test.py
echo "=== smoke test returned $? $(date) ==="

# One job, all 9 datasets looped internally (small -> large). Resume-safe (skips done rows).
echo "=== PREDICT $(date) ==="
$PY $CODE/gimlet_predict.py --batch_size 40 \
    --tasks bace bbbp freesolv esol clintox lipo hiv sider tox21

echo "=== SCORING $(date) ==="
$PY $CODE/gimlet_score.py
$PY $CODE/make_results_md.py

echo "=== ALL DONE $(date) ==="
