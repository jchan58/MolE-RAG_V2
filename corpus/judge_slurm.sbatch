#!/bin/bash
#SBATCH --job-name=e70judge
#SBATCH --account=bhrm-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/bhrm/jchan5/exp70_bbbp_pmc_collection/logs/slurm_%x_%j.out
RUN_DIR="/scratch/bhrm/jchan5/exp70_bbbp_pmc_collection"
PY="/u/jchan5/miniconda3/bin/python3"
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
