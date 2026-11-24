#!/usr/bin/env bash
# Lightweight resilience heartbeat: every 60s write RESUME_STATE.json (liveness read from the
# LOCAL /work PROGRESS.json -> node-independent, fresh while the process lives) to /work + S3.
# It does NOT sync state/ to S3 -- the collector's own _sync_state() already mirrors state/, and
# syncing the giant state/locks/ dir here HUNG on slow Lustre and killed the heartbeat. Keep this
# loop tiny + hang-proof. Singleton via flock.
set -u
export AWS_DEFAULT_REGION=us-east-1
BASE=/work/hdd/bhrm/jchan5/exp70_multi; PY=/u/jchan5/miniconda3/bin/python3
mkdir -p "$BASE"
# NOTE: no flock singleton — flock is unreliable on Lustre (stale locks blocked relaunch). Two
# heartbeats would only write the same idempotent anchor (last-wins, no corruption), so running
# one is a launch-discipline matter, not a correctness one. Record pid for visibility.
echo $$ > "$BASE/.heartbeat.pid" 2>/dev/null || true

while true; do
  timeout 60 "$PY" - "$BASE" <<'PYE' > "$BASE/RESUME_STATE.json.tmp" 2>/dev/null
import sys,os,json,glob,datetime
base=sys.argv[1]; dses=["lipo","sider","tox21","hiv"]
now=datetime.datetime.now(datetime.timezone.utc)
def rj(p):
    try: return json.load(open(p))
    except Exception: return {}
def age(ts):
    try: return round((now-datetime.datetime.fromisoformat(ts.replace("Z","+00:00"))).total_seconds(),1)
    except Exception: return None
st={"updated_at":now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "owner":"exp70_multi agent (lipo/sider/hiv/tox21)",
    "liveness_note":"liveness = LOCAL /work PROGRESS.json freshness (progress_age_s<~300 = alive); node-independent. Only relaunch a bridge if PROGRESS stale > ~10 min. Collection/run_tail/structuring run on the node named below.",
    "compute":"HIV chunk/BM25 = SageMaker ml.c6i.24xlarge (SageMakerExecutionRole, S3-only). EC2 launch_hiv_cpu_ec2.sh = fallback. Judge/extract = Bedrock qwen3-32b on Delta. Query-gen = Opus 4.6.",
    "orchestration":{
      "small":"lipo/sider/tox21 -> run_tail.sh (login node).",
      "hiv":"hiv_watcher_sm.sh -> COLLECT_DONE submits SageMaker chunk/BM25 -> CPU_DOWNSTREAM_DONE -> Delta judge -> JUDGE_DONE.",
      "structuring":"structuring_watcher.sh on JUDGE_DONE -> Cohere rerank + extract + build_cards.py -> s3://molerag/{ds}_chunks/v2/cards/.",
      "resume_doc":"exp70_multi/RESUME_PLAYBOOK.md (+ s3)"},
    "datasets":{}}
for ds in dses:
    sdir=os.path.join(base,ds,"state"); sp=rj(os.path.join(sdir,"PROGRESS.json"))
    sent=sorted(os.path.basename(x) for x in glob.glob(os.path.join(sdir,"*")) if os.path.basename(x) in
          ("COLLECT_DONE","FULLTEXT_DONE","JUDGE_DONE","STRUCT_DONE","HIV_SM_SUBMITTED"))
    smj=""
    p=os.path.join(sdir,"HIV_SM_SUBMITTED")
    if os.path.exists(p):
        try: smj=open(p).read().strip()
        except Exception: pass
    a=age(sp.get("updated_at",""))
    st["datasets"][ds]={"stage":sp.get("stage"),"node":sp.get("node"),
      "progress_updated_at":sp.get("updated_at"),"progress_age_s":a,
      "collection_alive":(a is not None and a<300),
      "molecule_querygen_done":sp.get("molecule_querygen_done"),
      "searches_started":sp.get("searches_started"),"sentinels":sent,"sagemaker_job":smj}
print(json.dumps(st,indent=1))
PYE
  if [ -s "$BASE/RESUME_STATE.json.tmp" ]; then
    mv -f "$BASE/RESUME_STATE.json.tmp" "$BASE/RESUME_STATE.json"
    # dedicated prefix (lipo/sider/hiv/tox21) so the peer agent's exp70_multi/ writes can't clobber my anchor
    timeout 30 aws s3 cp "$BASE/RESUME_STATE.json" s3://molerag/exp70_lsht/RESUME_STATE.json --only-show-errors 2>/dev/null || true
  fi
  sleep 60
done
