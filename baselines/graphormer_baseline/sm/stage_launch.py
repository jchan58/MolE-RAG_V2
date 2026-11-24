"""Launch the checkpoint-staging SageMaker job (CPU instance; Azure->S3)."""
import sagemaker, boto3, json, os
from sagemaker.pytorch import PyTorch
ROLE = "arn:aws:iam::833984135295:role/service-role/AmazonSageMaker-ExecutionRole-20260922T220415"
BUCKET = "molerag"
HERE = os.path.dirname(os.path.abspath(__file__))
sess = sagemaker.Session(boto3.Session(region_name="us-east-1"))
est = PyTorch(entry_point="stage_entry.py",
              source_dir=os.path.join(HERE, "stage_src"),
              role=ROLE, framework_version="1.9.1", py_version="py38",
              instance_type="ml.m5.xlarge", instance_count=1, volume_size=30,
              output_path=f"s3://{BUCKET}/graphormer_baseline/stage_output",
              code_location=f"s3://{BUCKET}/graphormer_baseline/code",
              base_job_name="graphormer-stage", max_run=3600,
              sagemaker_session=sess, disable_profiler=True)
est.fit(wait=False)
name = est.latest_training_job.name
print("JOB_NAME", name, flush=True)
json.dump([{"job": name, "kind": "stage"}], open(os.path.join(HERE, "stage_job.json"), "w"))
