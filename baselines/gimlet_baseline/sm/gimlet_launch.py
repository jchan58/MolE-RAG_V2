"""Launch GIMLET zero-shot (on MolE-RAG test molecules) as a SageMaker training job using the
legacy PyTorch 1.9.1 py38 cu111 DLC + requirements reconstructing the old PyG/transformers stack."""
import sagemaker, boto3, sys
from sagemaker.pytorch import PyTorch
ROLE="arn:aws:iam::833984135295:role/service-role/AmazonSageMaker-ExecutionRole-20260922T220415"
BUCKET="molerag"
REPO="/workspace/mole-rag_naacl/gimlet_baseline/code/GIMLET"
INST=sys.argv[1] if len(sys.argv)>1 else "ml.g4dn.4xlarge"
TASKS=sys.argv[2] if len(sys.argv)>2 else ""   # "" = all 9
sess=sagemaker.Session(boto3.Session(region_name="us-east-1"))
est=PyTorch(entry_point="gimlet_entry.py", source_dir=REPO, role=ROLE,
    framework_version="1.9.1", py_version="py38",
    instance_type=INST, instance_count=1, volume_size=80,
    output_path=f"s3://{BUCKET}/gimlet_baseline/output",
    code_location=f"s3://{BUCKET}/gimlet_baseline/code",
    base_job_name="gimlet-zs", max_run=5*3600,
    environment={"TASKS":TASKS,"HF_HOME":"/tmp/hf","TOKENIZERS_PARALLELISM":"false"},
    sagemaker_session=sess, disable_profiler=True)
est.fit(wait=False)
print("JOB_NAME",est.latest_training_job.name,flush=True)
import json; json.dump([{"job":est.latest_training_job.name,"tasks":TASKS or "all","instance":INST}],open("gimlet_jobs.json","w"))
