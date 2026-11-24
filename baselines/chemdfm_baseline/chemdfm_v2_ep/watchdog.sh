#!/bin/bash
# Self-healing, resumable driver for the ChemDFM framework runs through the chemdfm-v2-14b endpoint.
# Datasets run SEQUENTIALLY (single-GPU endpoint => serialized throughput; W=4 keeps the GPU queue
# full). Each runner is resumable via its part files; if it dies the loop relaunches it (skipping
# done parts). Zero-byte/truncated parts are deleted before each relaunch. Heartbeat for a guardian.
set -u
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY=/u/jchan5/miniconda3/bin/python
MODEL=chemdfm_v2_14b
W=${W:-2}   # single-worker endpoint + 60s invoke limit: keep queue shallow (1 in-flight + ~1 queued)
BACE=/work/hdd/bhrm/jchan5/molerag_reasoning/BACE/framework_final
FWC=/work/hdd/bhrm/jchan5/molerag_reasoning/fw_multi/code
FWR=/work/hdd/bhrm/jchan5/molerag_reasoning/fw_multi/runs
EP=/work/hdd/bhrm/jchan5/mole-rag_naacl/chemdfm_baseline/chemdfm_v2_ep
LOG=$EP/run_logs
mkdir -p $LOG
HB=$LOG/heartbeat
WLOG=$LOG/watchdog.log

log(){ echo "[$(date '+%F %T')] $*" | tee -a $WLOG; }
beat(){ date +%s > $HB; }

clean_zero(){ find "$1" -type f -name '*.json' -size 0 -delete 2>/dev/null; }

# $1=parts_dir  $2=target  $3=label  $4..=command to run the resumable runner
drive(){
  local dir="$1" tgt="$2" lab="$3"; shift 3
  local tries=0
  while :; do
    local n; n=$(ls "$dir" 2>/dev/null | wc -l)
    beat
    if [ "$n" -ge "$tgt" ]; then log "$lab complete ($n/$tgt)"; return 0; fi
    tries=$((tries+1))
    if [ "$tries" -gt 60 ]; then log "$lab GAVE UP after $tries relaunches ($n/$tgt)"; return 1; fi
    clean_zero "$dir"
    log "$lab launch #$tries ($n/$tgt)"
    "$@"
    log "$lab runner exited ($(ls "$dir" 2>/dev/null | wc -l)/$tgt)"
    sleep 6
  done
}

log "WATCHDOG START pid=$$ model=$MODEL"
beat

drive "$BACE/${MODEL}_test"        152 BACE     bash -c "cd $BACE && $PY run_final.py test $W $MODEL >> $LOG/bace.log 2>&1"
drive "$FWR/freesolv__${MODEL}"     65 freesolv bash -c "cd $FWC && $PY run_fw_multi.py freesolv $MODEL $W >> $LOG/freesolv.log 2>&1"
drive "$FWR/esol__${MODEL}"        114 esol     bash -c "cd $FWC && $PY run_fw_multi.py esol $MODEL $W >> $LOG/esol.log 2>&1"
drive "$FWR/clintox__${MODEL}"     149 clintox  bash -c "cd $FWC && $PY run_fw_multi.py clintox $MODEL $W >> $LOG/clintox.log 2>&1"
drive "$FWR/sider__${MODEL}"       144 sider    bash -c "cd $FWC && $PY run_fw_multi.py sider $MODEL $W >> $LOG/sider.log 2>&1"
drive "$FWR/bbbp__${MODEL}"        205 bbbp     bash -c "cd $FWC && $PY run_fw_multi.py bbbp $MODEL $W >> $LOG/bbbp.log 2>&1"
drive "$FWR/lipo__${MODEL}"        420 lipo     bash -c "cd $FWC && $PY run_fw_multi.py lipo $MODEL $W >> $LOG/lipo.log 2>&1"

beat
log "ALL DATASETS DONE"
touch $LOG/ALL_DONE.flag
