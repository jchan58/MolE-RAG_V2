#!/usr/bin/env bash
cd /work/hdd/bhrm/jchan5/exp70_multi/code
source env_common.sh hiv
export EXP70_PROP_CAP=200 EXP70_CLEAN_THREADS=48
echo "[$(date -u +%FT%TZ)] hiv PARALLEL cleanset START node=$(hostname -s)" >> ../logs/hiv_cleanset.out
/u/jchan5/miniconda3/bin/python3 _hiv_cleanset_parallel.py >> ../logs/hiv_cleanset.out 2>&1
echo "[$(date -u +%FT%TZ)] hiv cleanset EXIT rc=$? COLLECT_DONE=$([ -f ../hiv/state/COLLECT_DONE ]&&echo Y||echo n)" >> ../logs/hiv_cleanset.out
