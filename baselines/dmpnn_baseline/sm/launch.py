"""Launch the DMPNN baseline as a SageMaker PyTorch training job (one GPU instance runs all 27 runs).
Uploads data/ to S3, starts the job (async). Prints the job name. Run from sm/ dir."""
import sagemaker, boto3, sys
from sagemaker.pytorch import PyTorch
ROLE="arn:aws:iam::833984135295:role/service-role/AmazonSageMaker-ExecutionRole-20260922T220415"
BUCKET="molerag"
INSTANCE=sys.argv[1] if len(sys.argv)>1 else "ml.g5.2xlarge"
sess=sagemaker.Session(boto3.Session(region_name="us-east-1"))
# upload prepared data (combined.csv + test.csv + meta.json per task)
data_s3=sess.upload_data(path="../data", bucket=BUCKET, key_prefix="dmpnn_baseline/data")
print("data ->",data_s3,flush=True)
est=PyTorch(
    entry_point="entry.py", source_dir=".", role=ROLE,
    framework_version="2.3", py_version="py311",
    instance_type=INSTANCE, instance_count=1, volume_size=60,
    output_path=f"s3://{BUCKET}/dmpnn_baseline/output",
    code_location=f"s3://{BUCKET}/dmpnn_baseline/code",
    base_job_name="dmpnn-moleculenet", max_run=5*3600,
    environment={"SEEDS":"0,1,2","EPOCHS":"50"},
    sagemaker_session=sess, disable_profiler=True,
)
est.fit({"train": data_s3}, wait=False)
print("JOB_NAME",est.latest_training_job.name,flush=True)
print("OUTPUT s3://%s/dmpnn_baseline/output/%s/output/model.tar.gz"%(BUCKET,est.latest_training_job.name),flush=True)
