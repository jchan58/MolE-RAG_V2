import boto3,time,json
sm=boto3.client("sagemaker",region_name="us-east-1"); logs=boto3.client("logs",region_name="us-east-1")
GROUP="/aws/sagemaker/TrainingJobs"
jobs=[j["job"] for j in json.load(open("shard_jobs.json"))]
last={j:None for j in jobs}; toks={j:None for j in jobs}
def tail(j):
    try:
        strs=logs.describe_log_streams(logGroupName=GROUP,logStreamNamePrefix=j+"/",orderBy="LastEventTime",descending=True,limit=1).get("logStreams",[])
        if not strs: return
        kw=dict(logGroupName=GROUP,logStreamName=strs[0]["logStreamName"],startFromHead=True,limit=300)
        if toks[j]: kw["nextToken"]=toks[j]
        r=logs.get_log_events(**kw)
        for e in r["events"]:
            m=e["message"].rstrip()
            if any(k in m for k in ("TASKS:","-> ","ALL DONE","Traceback","Error ","error:","rc=1","Successfully installed chemprop")):
                print(f"  [{j.split('-2026')[0]}] {m[:180]}",flush=True)
        toks[j]=r.get("nextForwardToken")
    except Exception: pass
for i in range(260):
    allterm=True
    for j in jobs:
        d=sm.describe_training_job(TrainingJobName=j); st=d["TrainingJobStatus"]; sec=d.get("SecondaryStatus")
        if sec!=last[j]: print(f"[{time.strftime('%H:%M:%S')}] {j.split('-2026')[0]:16s} {st}/{sec}",flush=True); last[j]=sec
        if sec in ("Training","Uploading","Completed","Failed"): tail(j)
        if st not in ("Completed","Failed","Stopped"): allterm=False
        elif st=="Failed": print(f"  !! {j} FAILED: {d.get('FailureReason')}",flush=True)
    if allterm: print("ALL TERMINAL",flush=True); break
    time.sleep(75)
