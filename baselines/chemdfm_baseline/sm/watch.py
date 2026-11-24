"""Detached watcher: poll all ChemDFM SageMaker jobs; re-aggregate whenever a job terminates and once
more when all are terminal. Writes watch.log. Launch via:
  setsid env PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    AWS_DEFAULT_REGION=us-east-1 SAGEMAKER_SUPPRESS_V2_WARNING=1 <py> watch.py >/dev/null 2>&1 </dev/null &
"""
import boto3, time, json, subprocess, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sm = boto3.client("sagemaker", region_name="us-east-1")
jobs = [r["job"] for r in json.load(open(os.path.join(HERE, "chemdfm_jobs.json")))]
log = open(os.path.join(HERE, "watch.log"), "a")


def w(m):
    log.write(f"{time.strftime('%H:%M:%S')} {m}\n"); log.flush()


def reaggregate(tag):
    env = dict(os.environ, PYTHONNOUSERSITE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", AWS_DEFAULT_REGION="us-east-1")
    r = subprocess.run([sys.executable, "../analysis/aggregate.py"], env=env,
                       capture_output=True, text=True, cwd=HERE)
    w(f"AGG[{tag}] rc={r.returncode}\n{r.stdout[-3000:]}\nERR:{r.stderr[-600:]}")


w(f"=== watch start; jobs={len(jobs)} ===")
last = {}
for _ in range(5000):
    allterm = True
    changed = False
    for j in jobs:
        try:
            d = sm.describe_training_job(TrainingJobName=j)
            st = d["TrainingJobStatus"]
            sec = d.get("SecondaryStatus", "")
        except Exception as e:
            st = f"err:{str(e)[:60]}"; sec = ""
        key = f"{st}/{sec}"
        if last.get(j) != key:
            w(f"{j.split('-2026')[0]} -> {st} ({sec})")
            last[j] = key
            if st in ("Completed", "Failed", "Stopped"):
                changed = True
        if st not in ("Completed", "Failed", "Stopped") and not st.startswith("err:"):
            allterm = False
    if changed:
        reaggregate("progress")
    if allterm:
        w("ALL TERMINAL")
        reaggregate("final")
        break
    time.sleep(120)
w("=== watch done ===")
