#!/usr/bin/env bash
cd ${CODE:-$(cd "$(dirname "$0")" && pwd)}
source env_common.sh hiv
export EXP70_PROP_CAP=200 EXP70_MERGE_THREADS=16
echo "[$(date -u +%FT%TZ)] hiv merge START node=$(hostname -s) prop_cap=$EXP70_PROP_CAP threads=$EXP70_MERGE_THREADS" >> ../logs/hiv_merge.out
${PY:-python3} run_merge_once.py >> ../logs/hiv_merge.out 2>&1
echo "[$(date -u +%FT%TZ)] hiv merge EXIT rc=$? COLLECT_DONE=$([ -f ../hiv/state/COLLECT_DONE ]&&echo Y||echo n)" >> ../logs/hiv_merge.out
