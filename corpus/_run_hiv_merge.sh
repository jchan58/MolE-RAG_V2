#!/usr/bin/env bash
cd /work/hdd/bhrm/jchan5/exp70_multi/code
source env_common.sh hiv
export EXP70_PROP_CAP=200 EXP70_MERGE_THREADS=16
echo "[$(date -u +%FT%TZ)] hiv merge START node=$(hostname -s) prop_cap=$EXP70_PROP_CAP threads=$EXP70_MERGE_THREADS" >> ../logs/hiv_merge.out
/u/jchan5/miniconda3/bin/python3 run_merge_once.py >> ../logs/hiv_merge.out 2>&1
echo "[$(date -u +%FT%TZ)] hiv merge EXIT rc=$? COLLECT_DONE=$([ -f ../hiv/state/COLLECT_DONE ]&&echo Y||echo n)" >> ../logs/hiv_merge.out
