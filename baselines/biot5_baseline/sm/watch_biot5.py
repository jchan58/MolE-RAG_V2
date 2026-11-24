import boto3, time, json, subprocess, os, sys
sm = boto3.client("sagemaker", region_name="us-east-1")
HERE = os.path.dirname(os.path.abspath(__file__))
jobs = [r["job"] for r in json.load(open(os.path.join(HERE, "biot5_jobs.json"))) if r.get("smoke") != "1"]
log = open(os.path.join(HERE, "watch_biot5.log"), "a")
def w(m): log.write("%s %s\n" % (time.strftime("%H:%M:%S"), m)); log.flush()
w("=== watch start; jobs=%d" % len(jobs))
last = {}
while True:
    allterm = True
    for j in jobs:
        try:
            st = sm.describe_training_job(TrainingJobName=j)["TrainingJobStatus"]
        except Exception as e:
            st = "DESCRIBE_ERR"
        if last.get(j) != st:
            w("%s -> %s" % (j.split('-2026')[0], st)); last[j] = st
        if st not in ("Completed", "Failed", "Stopped", "DESCRIBE_ERR"):
            allterm = False
    if allterm:
        w("ALL TERMINAL"); break
    time.sleep(120)
env = dict(os.environ, PYTHONNOUSERSITE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
           MKL_NUM_THREADS="1", AWS_DEFAULT_REGION="us-east-1")
r = subprocess.run([sys.executable, os.path.join(HERE, "..", "analysis", "aggregate.py")],
                   env=env, capture_output=True, text=True)
w("AGG STDOUT:\n" + r.stdout[-4000:]); w("AGG ERR:" + r.stderr[-1000:])
w("=== re-aggregated ===")
