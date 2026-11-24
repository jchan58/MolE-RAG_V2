#!/bin/bash
#SBATCH --job-name=e70judge
#SBATCH --account=bhrm-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=${EXP70_RUN_DIR:-$(cd "$(dirname "$0")/.." && pwd)/corpus_runs/bbbp}/logs/slurm_%x_%j.out
RUN_DIR="${EXP70_RUN_DIR:-$(cd "$(dirname "$0")/.." && pwd)/corpus_runs/bbbp}"
PY="${PY:-python3}"
export EXP70_RUN_DIR="$RUN_DIR" EXP70_JUDGE_THREADS=64
cd "$RUN_DIR/code"
echo "[$(date -u +%H:%M:%S)] judge node $(hostname) job $SLURM_JOB_ID starting 16 judge procs (nshards=1, ItemLock-coordinated)"
pids=""
for i in $(seq 0 15); do
  "$PY" judge_run.py 1 0 >> "$RUN_DIR/logs/judge_node_${SLURM_JOB_ID}_$i.out" 2>&1 &
  pids="$pids $!"
  sleep 0.2
done
wait $pids
echo "[$(date -u +%H:%M:%S)] judge node $SLURM_JOB_ID done"
