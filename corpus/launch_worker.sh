#!/usr/bin/env bash
# Launch ONE additional exp70 worker that helps drain whichever parallel stage is
# active (shares the run dir with the driver; central token buckets keep global
# NCBI/PubChem/OpenAI rates within caps regardless of worker count).
#
# Usage:  bash launch_worker.sh [auto|pubchem|querygen_molecule|search|metadata|download]
set -u
RUN_DIR="/scratch/bhrm/jchan5/exp70_bbbp_pmc_collection"
PY="/u/jchan5/miniconda3/bin/python3"
STAGE="${1:-auto}"
TS="$(date +%s)_$$"
mkdir -p "$RUN_DIR/logs"
export EXP70_RUN_DIR="$RUN_DIR"
setsid nohup "$PY" "$RUN_DIR/code/worker.py" "$STAGE" \
    >> "$RUN_DIR/logs/worker_${TS}.log" 2>&1 &
echo "launched worker (stage=$STAGE) pid $! -> $RUN_DIR/logs/worker_${TS}.log"
