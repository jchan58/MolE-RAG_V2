"""Emit one line per newly-completed task (status+value) from S3 incremental, plus job-status
changes and terminal state. For use under the Monitor tool. Exits when the job is terminal."""
import time, json, sys
import boto3

JOB = sys.argv[1]
INCR = "mol_llm_baseline/output/incremental"
sm = boto3.client("sagemaker", region_name="us-east-1")
s3 = boto3.client("s3", region_name="us-east-1")
seen = set()
last_status = None

while True:
    try:
        st = sm.describe_training_job(TrainingJobName=JOB)["TrainingJobStatus"]
    except Exception:
        st = "describe_error"
    if st != last_status:
        print(f"JOB_STATUS {st}", flush=True)
        last_status = st
    try:
        r = s3.list_objects_v2(Bucket="molerag", Prefix=INCR + "/")
        for o in r.get("Contents", []):
            k = o["Key"]
            if not k.endswith(".json") or k in seen or "smoke" in k:
                continue
            seen.add(k)
            try:
                obj = json.loads(s3.get_object(Bucket="molerag", Key=k)["Body"].read())
                print(f"TASK_DONE {obj.get('task')} status={obj.get('status')} "
                      f"{obj.get('metric')}={obj.get('value')} valbest={obj.get('val_value')} "
                      f"ep={obj.get('best_epoch')}/{obj.get('epochs_done')} min={obj.get('minutes')}",
                      flush=True)
            except Exception as e:
                print(f"TASK_READ_ERR {k} {str(e)[:80]}", flush=True)
    except Exception as e:
        print(f"S3_LIST_ERR {str(e)[:80]}", flush=True)
    if st in ("Completed", "Failed", "Stopped"):
        print(f"TERMINAL {st}", flush=True)
        break
    time.sleep(90)
