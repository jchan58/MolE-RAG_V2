"""Launch the CPU staging job: download MolLM MoleculePrediction checkpoint zip from HF,
extract ckpt(s), upload to s3://molerag/mol_llm_baseline/ckpt/. CPU instance -> no GPU quota use."""
import sagemaker, boto3, json
from sagemaker.pytorch import PyTorch

ROLE = "arn:aws:iam::833984135295:role/service-role/AmazonSageMaker-ExecutionRole-20260922T220415"
BUCKET = "molerag"
sess = sagemaker.Session(boto3.Session(region_name="us-east-1"))

est = PyTorch(
    entry_point="stage_entry.py", source_dir="stage", role=ROLE,
    framework_version="1.12.1", py_version="py38",
    instance_type="ml.m5.4xlarge", instance_count=1, volume_size=120,
    output_path=f"s3://{BUCKET}/mol_llm_baseline/output",
    code_location=f"s3://{BUCKET}/mol_llm_baseline/code",
    base_job_name="mollm-stage", max_run=3 * 3600,
    environment={
        "S3_BUCKET": BUCKET, "S3_CKPT_PREFIX": "mol_llm_baseline/ckpt",
        "HF_HOME": "/tmp/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
    },
    sagemaker_session=sess, disable_profiler=True,
)
est.fit(wait=False)
jn = est.latest_training_job.name
json.dump({"job": jn, "instance": "ml.m5.4xlarge"}, open("stage_job.json", "w"), indent=2)
print("LAUNCHED stage job:", jn, flush=True)
