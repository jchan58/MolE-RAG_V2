#!/usr/bin/env bash
# Collector resilience supervisor (the "don't drop" fix). Every 120s, for each dataset not yet
# COLLECT_DONE, if its COLLECT_PROGRESS has been stale > 10 min it checks REAL process liveness
# (run_collect.py with EXP70_DATASET=<ds>) on BOTH this node and dt-login02, and relaunches the
# collector (idempotent resume) ONLY when none is alive anywhere. It never relaunches a collector
# that is alive-but-quiet (e.g. mid-merge/_results_sync) -> no double-launch. Singleton via flock.
set -u
export AWS_DEFAULT_REGION=us-east-1
BASE=${EXP70_BASE:-$(cd "$(dirname "$0")/.." && pwd)/corpus_runs}; CODE=$BASE/code; PY=${PY:-python3}
OTHER_NODES="dt-login02 dt-login01 dt-login03"
exec 7>"$BASE/.collector_supervisor.lock"
if ! flock -n 7; then echo "supervisor already running; exit"; exit 0; fi

# prints pids of live run_collect.py whose EXP70_DATASET matches $1, on the given host ("" = local)
LIVE_SCRIPT='for p in $(pgrep -f "[r]un_collect\.py" 2>/dev/null); do d=$(tr "\000" "\n" </proc/$p/environ 2>/dev/null | grep "^EXP70_DATASET=" | cut -d= -f2); [ "$d" = "__DS__" ] && echo "$p"; done'
alive_for() {
  local ds="$1" host="$2" s="${LIVE_SCRIPT/__DS__/$1}"
  if [ -z "$host" ]; then bash -c "$s"; else timeout 18 ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "$s" 2>/dev/null; fi
}
age_s() { timeout 10 "$PY" -c "import json,datetime as dt
try:
 u=json.load(open('$1')).get('updated_at','')
 print(int((dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(u.replace('Z','+00:00'))).total_seconds()))
except Exception: print(99999)" 2>/dev/null; }

while true; do
  alldone=1
  for ds in lipo sider tox21 hiv; do
    [ -f "$BASE/$ds/state/COLLECT_DONE" ] && continue
    alldone=0
    age=$(age_s "$BASE/$ds/state/COLLECT_PROGRESS.json"); age=${age:-99999}
    [ "$age" -lt 600 ] && continue           # progressing recently -> healthy
    live="$(alive_for "$ds" "")"
    for h in $OTHER_NODES; do [ -z "$live" ] && live="$(alive_for "$ds" "$h")"; done
    if [ -z "$live" ]; then
      echo "[$(date -u +%H:%M:%S)] [$ds] dead (no live collector) + stale ${age}s + not COLLECT_DONE -> RELAUNCH"
      ( source "$CODE/env_common.sh" "$ds"
        setsid nohup "$PY" "$CODE/run_collect.py" >> "$EXP70_RUN_DIR/logs/collect_supervised_$(date +%H%M%S).log" 2>&1 < /dev/null & )
    else
      echo "[$(date -u +%H:%M:%S)] [$ds] stale ${age}s but collector ALIVE ($live) -> leave it"
    fi
  done
  [ "$alldone" = 1 ] && { echo "all datasets COLLECT_DONE; supervisor exiting"; break; }
  sleep 120
done
