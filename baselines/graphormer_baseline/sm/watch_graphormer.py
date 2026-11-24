import boto3,time,json,subprocess,os,sys
sm=boto3.client("sagemaker",region_name="us-east-1")
jobs=[r["job"] for r in json.load(open("graphormer_jobs.json"))]
log=open("watch_graphormer.log","a")
def w(m): log.write("%s %s\n"%(time.strftime("%H:%M:%S"),m)); log.flush()
w("=== watch start; jobs=%d"%len(jobs))
last={}
while True:
    allterm=True
    for j in jobs:
        st=sm.describe_training_job(TrainingJobName=j)["TrainingJobStatus"]
        if last.get(j)!=st:
            w("%s -> %s"%(j.split('-2026')[0],st)); last[j]=st
        if st not in ("Completed","Failed","Stopped"): allterm=False
    if allterm:
        w("ALL TERMINAL"); break
    time.sleep(120)
env=dict(os.environ,PYTHONNOUSERSITE="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",AWS_DEFAULT_REGION="us-east-1")
r=subprocess.run([sys.executable,"../analysis/aggregate.py"],env=env,capture_output=True,text=True,cwd="../analysis")
w("AGG STDOUT:\n"+r.stdout[-4000:]); w("AGG ERR:"+r.stderr[-800:])
w("=== re-aggregated ===")
