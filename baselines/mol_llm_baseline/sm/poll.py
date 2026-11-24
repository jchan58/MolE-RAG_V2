"""Poll a SageMaker job to terminal state; print status, failure reason, S3 result, and last log lines."""
import sys, time, json
import boto3

job = sys.argv[1]
result_key = sys.argv[2] if len(sys.argv) > 2 else None
sm = boto3.client("sagemaker", region_name="us-east-1")
logs = boto3.client("logs", region_name="us-east-1")
s3 = boto3.client("s3", region_name="us-east-1")

while True:
    d = sm.describe_training_job(TrainingJobName=job)
    st = d["TrainingJobStatus"]; sec = d.get("SecondaryStatus")
    if st in ("Completed", "Failed", "Stopped"):
        break
    time.sleep(30)

print("JOB", job, "->", st, sec, flush=True)
if d.get("FailureReason"):
    print("FAILURE:", d["FailureReason"][:500], flush=True)

# last log lines
try:
    grp = "/aws/sagemaker/TrainingJobs"
    streams = logs.describe_log_streams(logGroupName=grp, logStreamNamePrefix=job,
                                        orderBy="LastEventTime", descending=True, limit=1)
    if streams["logStreams"]:
        sn = streams["logStreams"][0]["logStreamName"]
        ev = logs.get_log_events(logGroupName=grp, logStreamName=sn, limit=60, startFromHead=False)
        print("---- last log lines ----", flush=True)
        for e in ev["events"]:
            print(e["message"], flush=True)
except Exception as e:
    print("log fetch error:", str(e)[:200], flush=True)

if result_key:
    try:
        o = s3.get_object(Bucket="molerag", Key=result_key)
        print("---- S3", result_key, "----", flush=True)
        print(o["Body"].read().decode(), flush=True)
    except Exception as e:
        print("no S3 result yet:", str(e)[:150], flush=True)
print("POLL DONE", flush=True)
