#!/bin/bash
# Chains interactive-GPU pipeline jobs (one at a time, QOS-friendly) until all stages complete.
# Gates on squeue by job NAME (crmn). Resubmits when the previous ends and work remains.
set -u
ROOT=/work/hdd/bhrm/jchan5/molerag_reasoning/claude/chemrag_moleculenet_9task_qwen_v1
LOG="$ROOT/jobs/orchestrate.log"; NQ=16648
cd "$ROOT"
cnt(){ [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
for i in $(seq 1 20); do
  while squeue -u jchan5 -h -n crmn 2>/dev/null | grep -q .; do sleep 60; done
  BM=$(cnt retrieval/bm25_hits.jsonl); DE=$(cnt retrieval/dense_hits.jsonl)
  GEN=$(cat outputs/*_preds.jsonl 2>/dev/null | wc -l | tr -d ' '); FU=$([ -f retrieval/fused_docs.jsonl ] && echo 1 || echo 0)
  echo "$(date -Is) iter=$i bm25=$BM dense=$DE fused=$FU gen=$GEN /$NQ" >> "$LOG"
  if [ "$BM" -ge "$NQ" ] && [ "$DE" -ge "$NQ" ] && [ "$FU" = "1" ] && [ "$GEN" -ge "$NQ" ]; then
    echo "$(date -Is) ALL_DONE" >> "$LOG"; sbatch "$ROOT/jobs/pipeline.sbatch" >/dev/null 2>&1; break
  fi
  JID=$(sbatch --parsable "$ROOT/jobs/pipeline.sbatch" 2>>"$LOG"); echo "$(date -Is) submitted $JID" >> "$LOG"
  sleep 30
done
echo "$(date -Is) orchestrator exit" >> "$LOG"
