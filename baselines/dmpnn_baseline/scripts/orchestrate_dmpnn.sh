#!/bin/bash
# Chains interactive-GPU DMPNN jobs, ONE at a time (polite to the shared per-user QOS cap).
# Retries sbatch when the submit quota is full (QOSMaxSubmitJobPerUserLimit). Stops when all
# 27 result.json (9 tasks x 3 seeds) exist. Logs to logs/orchestrate.log.
set -u
ROOT=/workspace/mole-rag_naacl/dmpnn_baseline
LOG="$ROOT/logs/orchestrate.log"; cd "$ROOT"
ndone(){ ls runs/*/seed*/result.json 2>/dev/null | wc -l | tr -d ' '; }
PREV=-1; STALL=0
for i in $(seq 1 60); do
  # wait while our own dmpnn job is queued/running
  while squeue -u jchan5 -h -n dmpnn 2>/dev/null | grep -q .; do sleep 60; done
  D=$(ndone); echo "$(date -Is) iter=$i done=$D/27" >> "$LOG"
  if [ "$D" -ge 27 ]; then echo "$(date -Is) ALL_DONE" >> "$LOG"; break; fi
  # stall guard: a completed job that produced no new results => likely a bug, stop (don't loop forever)
  if [ "$PREV" -ge 0 ] && [ "$D" -le "$PREV" ]; then STALL=$((STALL+1)); else STALL=0; fi
  if [ "$STALL" -ge 2 ]; then echo "$(date -Is) STALLED (no progress x2); stop for inspection" >> "$LOG"; break; fi
  PREV=$D
  OUT=$(sbatch --parsable "$ROOT/scripts/dmpnn.sbatch" 2>&1)
  if echo "$OUT" | grep -qiE "QOSMaxSubmitJobPerUserLimit|QOSMax|violates"; then
    echo "$(date -Is) submit blocked (quota full); wait 180s" >> "$LOG"; sleep 180; continue
  fi
  echo "$(date -Is) submitted $OUT" >> "$LOG"; sleep 30
done
echo "$(date -Is) orchestrator exit done=$(ndone)/27" >> "$LOG"
