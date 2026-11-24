import boto3,time,sys
JOB=sys.argv[1]
sm=boto3.client("sagemaker",region_name="us-east-1"); logs=boto3.client("logs",region_name="us-east-1")
GROUP="/aws/sagemaker/TrainingJobs"
last_sec=None; tok=None; seen=0
def tail():
    global tok,seen
    try:
        strs=logs.describe_log_streams(logGroupName=GROUP,logStreamNamePrefix=JOB+"/",orderBy="LastEventTime",descending=True,limit=1).get("logStreams",[])
        if not strs: return
        sn=strs[0]["logStreamName"]
        kw=dict(logGroupName=GROUP,logStreamName=sn,startFromHead=True,limit=200)
        if tok: kw["nextToken"]=tok
        r=logs.get_log_events(**kw)
        for e in r["events"]:
            m=e["message"].rstrip()
            if any(k in m for k in ("===","-> ","ALL DONE","Error","error","Traceback","rc=","Epoch","Successfully installed")):
                print("  LOG:",m[:200],flush=True)
        tok=r.get("nextForwardToken")
    except Exception as ex: print("  (log tail err",str(ex)[:60],")",flush=True)
for i in range(240):
    d=sm.describe_training_job(TrainingJobName=JOB)
    st=d["TrainingJobStatus"]; sec=d.get("SecondaryStatus")
    if sec!=last_sec: print(f"[{time.strftime('%H:%M:%S')}] {st} / {sec}",flush=True); last_sec=sec
    if sec in ("Training","Uploading","Completed","Failed"): tail()
    if st in ("Completed","Failed","Stopped"):
        print("FINAL:",st,"| FailureReason:",d.get("FailureReason"),flush=True)
        bs=d.get("BillableTimeInSeconds"); print("BillableSeconds:",bs,flush=True); break
    time.sleep(90)
