import boto3,time,sys,json,os
jn=json.load(open("gimlet_jobs_cs.json"))[0]["job"]
sm=boto3.client("sagemaker",region_name="us-east-1")
log=open("monitor_cs.log","a")
def w(m):
    print(m,flush=True); log.write(m+"\n"); log.flush()
w("=== monitor start %s job=%s"%(time.strftime("%H:%M:%S"),jn))
last=None
while True:
    d=sm.describe_training_job(TrainingJobName=jn)
    st=d["TrainingJobStatus"]; sec=d.get("SecondaryStatus")
    if (st,sec)!=last:
        w("%s %s / %s"%(time.strftime("%H:%M:%S"),st,sec))
        last=(st,sec)
    if st in ("Completed","Failed","Stopped"):
        w("FINAL %s %s"%(st,d.get("FailureReason","")))
        w("MODEL_ARTIFACTS %s"%d.get("ModelArtifacts",{}).get("S3ModelArtifacts",""))
        break
    time.sleep(90)
