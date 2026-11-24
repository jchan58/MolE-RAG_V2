"""Launch the ChemDFM framework (conditions A + C) on BACE test as a single SageMaker training job.
Proven path: same role/instance pool as chemdfm_baseline (training-job quota, in-container HF download,
internet egress for Cohere). Restart-safe via per-molecule S3 parts under s3://molerag/chemdfm_fw/parts/.
"""
import sagemaker, boto3, json, os, sys
from pathlib import Path
from sagemaker.pytorch import PyTorch

ROLE="arn:aws:iam::833984135295:role/service-role/AmazonSageMaker-ExecutionRole-20260922T220415"
BUCKET="molerag"; HERE=os.path.dirname(os.path.abspath(__file__))
sess=sagemaker.Session(boto3.Session(region_name="us-east-1"))

def bedrock_token():
    for l in (Path.home()/".config/molrag/bedrock.env").read_text().splitlines():
        l=l.strip().removeprefix("export ")
        if l.startswith("AWS_BEARER"): return l.split("=",1)[1].strip().strip('"').strip("'")
    return ""
TOK=bedrock_token(); assert TOK, "no bedrock token"

FALLBACK=["ml.g6e.2xlarge","ml.g6e.8xlarge","ml.g6e.12xlarge"]
def make_est(inst):
    return PyTorch(
        entry_point="entry_fw.py", source_dir=HERE, role=ROLE,
        framework_version="2.3", py_version="py311",
        instance_type=inst, instance_count=1, volume_size=120,
        output_path=f"s3://{BUCKET}/chemdfm_fw/output",
        code_location=f"s3://{BUCKET}/chemdfm_fw/code",
        base_job_name="chemdfm-fw", max_run=10*3600,
        environment={
            "MODEL_NAME":"OpenDFM/ChemDFM-v1.0-13B","SPLIT":"test",
            "S3_BUCKET":BUCKET,"S3_PARTS":"chemdfm_fw/parts","BEDROCK_TOKEN":TOK,
            "MAX_NEW_TOKENS":"420","HF_HOME":"/tmp/hf","TOKENIZERS_PARALLELISM":"false",
            "HF_HUB_ENABLE_HF_TRANSFER":"0",
        },
        sagemaker_session=sess, disable_profiler=True,
    )
jn=None
for it in FALLBACK:
    try:
        est=make_est(it); est.fit(wait=False); jn=est.latest_training_job.name
        print("LAUNCHED",jn,"|",it,flush=True); break
    except Exception as e:
        print(f"  {it} failed to launch ({str(e)[:160]}); trying next",flush=True)
if jn is None: print("ALL instance types failed",flush=True); sys.exit(1)
reg=os.path.join(HERE,"fw_job.json"); json.dump({"job":jn},open(reg,"w")); print("registry ->",reg)
