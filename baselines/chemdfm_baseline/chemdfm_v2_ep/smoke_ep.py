"""Smoke-test the chemdfm-v2-14b endpoint: contract + latency. Run after InService."""
import boto3, json, time
from botocore.config import Config
SMRT = boto3.client("sagemaker-runtime", region_name="us-east-1",
                    config=Config(read_timeout=900, connect_timeout=20, retries={"max_attempts": 0}))
EP = "chemdfm-v2-14b"


def call(prompt, mt):
    t = time.time()
    r = SMRT.invoke_endpoint(EndpointName=EP, ContentType="application/json",
                             Body=json.dumps({"prompt": prompt, "max_new_tokens": mt}))
    out = json.loads(r["Body"].read())
    return out, time.time() - t


tests = [
    ("What is the molecular formula of aspirin? Answer in one line.", 64),
    ('Predict BACE-1 inhibition for SMILES CC(C)Cc1ccccc1. Output ONLY JSON: '
     '{"prediction":"active|inactive|uncertain","probability_active":0.0,"justification":"..."}', 400),
]
for i, (p, mt) in enumerate(tests):
    try:
        out, dt = call(p, mt)
        txt = out.get("generated_text", "")
        print(f"[test{i}] {dt:.1f}s err={out.get('error')}")
        print("  ->", repr(txt[:400]))
    except Exception as e:
        print(f"[test{i}] EXC {type(e).__name__}: {str(e)[:200]}")
