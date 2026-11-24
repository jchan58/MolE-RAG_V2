import time, json, boto3
s3=boto3.client("s3",region_name="us-east-1"); sm=boto3.client("sagemaker",region_name="us-east-1")
job="mollm-full-2026-10-12-05-51-46-991"
B="molerag"; P="mol_llm_baseline/output/incremental"
def get(k):
    try: return json.loads(s3.get_object(Bucket=B,Key=k)["Body"].read())
    except Exception: return None
printed_diag=False
while True:
    d=get(f"{P}/jit_diag.json")
    if d and not printed_diag:
        print("JIT_DIAG:", json.dumps(d)); printed_diag=True
    r=get(f"{P}/bbbp_seed0.json")
    if r and "traceback" in r:
        print("BBBP_STATUS:", r.get("status","")[:120])
        print("BBBP_TRACEBACK:\n"+r.get("traceback",""))
        break
    if r and r.get("status")=="ok":
        print("BBBP_OK value=",r.get("value")); break
    st=sm.describe_training_job(TrainingJobName=job)["TrainingJobStatus"]
    if st in ("Failed","Stopped","Completed") and r is not None and "traceback" not in r and r.get("status")!="ok":
        # terminal but no traceback field -> print whatever we have
        print("TERMINAL",st,"bbbp:",json.dumps(r)[:300]); break
    if st in ("Failed","Stopped") and r is None:
        print("TERMINAL",st,"no bbbp result"); break
    time.sleep(30)
print("WAIT_DIAG DONE")
