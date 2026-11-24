"""Autonomous serialized orchestrator for the LlaSMol baseline (our only usable instance type is
ml.g5.12xlarge, quota 1; ml.g5.24xlarge quota is 0). Steps, all logged to orchestrate.log:
  1. Wait until ml.g5.12xlarge is free (no InProgress job on it other than ours).
  2. Launch a SMOKE job (8 units/task) to validate the full stack (base+LoRA merge, generation, LL scoring).
  3. Wait for the smoke job to reach a terminal state; validate it Completed with every task status 'ok'.
  4. If smoke is good: wait for the slot to free again, launch the FULL job, poll it to terminal, then
     run analysis/aggregate.py (writes RESULTS.md + llasmol_results.json).
  5. If smoke fails: stop and log (no full job, no fabricated numbers).
It polls SageMaker job states only (never stops anyone's job), so it does not self-match any ps|grep loop.
Launch:
  setsid env PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    AWS_DEFAULT_REGION=us-east-1 SAGEMAKER_SUPPRESS_V2_WARNING=1 <py> orchestrate.py >/dev/null 2>&1 </dev/null &
"""
import boto3, time, json, subprocess, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
INSTANCE = "ml.g5.12xlarge"
BUCKET = "molerag"
sm = boto3.client("sagemaker", region_name="us-east-1")
s3 = boto3.client("s3", region_name="us-east-1")
log = open(os.path.join(HERE, "orchestrate.log"), "a")
TASKS = "esol,lipo,bbbp,clintox,hiv,sider"
SUPPORTED = ["esol", "lipo", "bbbp", "clintox", "hiv", "sider"]


def w(m):
    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}\n"); log.flush()


def env():
    return dict(os.environ, PYTHONNOUSERSITE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", AWS_DEFAULT_REGION="us-east-1", SAGEMAKER_SUPPRESS_V2_WARNING="1")


def slot_busy(exclude=()):
    """True if an InProgress training job uses INSTANCE (excluding job names in `exclude`)."""
    try:
        for j in sm.list_training_jobs(StatusEquals="InProgress", MaxResults=100)["TrainingJobSummaries"]:
            n = j["TrainingJobName"]
            if n in exclude:
                continue
            d = sm.describe_training_job(TrainingJobName=n)
            if d["ResourceConfig"]["InstanceType"] == INSTANCE:
                return n
    except Exception as e:
        w(f"slot check error: {str(e)[:120]}")
    return None


def wait_for_slot(exclude=(), label=""):
    while True:
        b = slot_busy(exclude)
        if b is None:
            w(f"slot free {label}")
            return
        w(f"slot {INSTANCE} busy ({b}); waiting {label}")
        time.sleep(180)


def launch(smoke):
    r = subprocess.run([PY, "llasmol_launch.py", TASKS, INSTANCE, "1" if smoke else "0"],
                       env=env(), capture_output=True, text=True, cwd=HERE)
    job = None
    for line in (r.stdout + "\n" + r.stderr).splitlines():
        if line.startswith("JOB_NAME "):
            job = line.split()[1].strip()
    w(f"launch(smoke={smoke}) rc={r.returncode} job={job}\n  out:{r.stdout[-400:]}\n  err:{r.stderr[-400:]}")
    return r.returncode, job


def wait_terminal(job):
    last = None
    while True:
        try:
            st = sm.describe_training_job(TrainingJobName=job)["TrainingJobStatus"]
        except Exception as e:
            st = f"err:{str(e)[:60]}"
        if st != last:
            w(f"{job} -> {st}"); last = st
        if st in ("Completed", "Failed", "Stopped") or st.startswith("err:"):
            return st
        time.sleep(120)


def smoke_results_ok():
    ok = {}
    for t in SUPPORTED:
        try:
            o = s3.get_object(Bucket=BUCKET, Key=f"llasmol_baseline/output/incremental_smoke/{t}.json")
            r = json.loads(o["Body"].read())
            ok[t] = r.get("status")
        except Exception as e:
            ok[t] = f"missing:{str(e)[:40]}"
    w(f"smoke task statuses: {ok}")
    return all(v == "ok" for v in ok.values())


def aggregate():
    r = subprocess.run([PY, "../analysis/aggregate.py"], env=env(), capture_output=True, text=True, cwd=HERE)
    w(f"AGGREGATE rc={r.returncode}\n{r.stdout[-3000:]}\nERR:{r.stderr[-600:]}")


def main():
    w("=== orchestrator start ===")
    # Phase 1-3: smoke
    wait_for_slot(label="(for smoke)")
    rc, sjob = launch(smoke=True)
    if rc != 0 or not sjob:
        w("smoke launch failed; aborting."); return
    st = wait_terminal(sjob)
    if st != "Completed":
        w(f"smoke job {sjob} ended {st}; aborting (no full run, no fabricated numbers)."); return
    if not smoke_results_ok():
        w("smoke validation FAILED (a task did not reach status ok); aborting."); return
    w("smoke PASSED -> proceeding to full run")
    # Phase 4: full
    wait_for_slot(exclude=(sjob,), label="(for full)")
    rc, fjob = launch(smoke=False)
    if rc != 0 or not fjob:
        w("full launch failed; aborting."); return
    st = wait_terminal(fjob)
    w(f"full job {fjob} terminal: {st}")
    aggregate()
    w("=== orchestrator done ===")


if __name__ == "__main__":
    main()
