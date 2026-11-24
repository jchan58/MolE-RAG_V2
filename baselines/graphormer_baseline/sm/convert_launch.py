"""Launch the checkpoint-conversion SageMaker job (HF clefourrier -> GIMLET fairseq keys -> S3)."""
import sagemaker, boto3, json, os
from sagemaker.pytorch import PyTorch
ROLE = "arn:aws:iam::833984135295:role/service-role/AmazonSageMaker-ExecutionRole-20260922T220415"
BUCKET = "molerag"
REPO = "/work/hdd/bhrm/jchan5/mole-rag_naacl/graphormer_baseline/code/GIMLET"
HERE = os.path.dirname(os.path.abspath(__file__))
sess = sagemaker.Session(boto3.Session(region_name="us-east-1"))
est = PyTorch(entry_point="convert_entry.py", source_dir=REPO, role=ROLE,
              framework_version="1.9.1", py_version="py38",
              instance_type="ml.m5.2xlarge", instance_count=1, volume_size=40,
              output_path=f"s3://{BUCKET}/graphormer_baseline/convert_output",
              code_location=f"s3://{BUCKET}/graphormer_baseline/code",
              base_job_name="graphormer-convert", max_run=3600,
              environment={"HF_HOME": "/tmp/hf", "TOKENIZERS_PARALLELISM": "false"},
              sagemaker_session=sess, disable_profiler=True)
est.fit(wait=False)
name = est.latest_training_job.name
print("JOB_NAME", name, flush=True)
json.dump([{"job": name, "kind": "convert"}], open(os.path.join(HERE, "convert_job.json"), "w"))
