#!/bin/bash
#SBATCH --job-name=e70pipe
#SBATCH --account=bhrm-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=${EXP70_BASE:-$(cd "$(dirname "$0")/.." && pwd)/corpus_runs}/logs/pipe_%x_%j.out
# Usage: sbatch --job-name=e70_<ds> --export=ALL,EXP70_DS=<ds> pipeline.sbatch
set -u
DS="${EXP70_DS:?set EXP70_DS}"
bash ${CODE:-$(cd "$(dirname "$0")" && pwd)}/run_pipeline.sh "$DS"
