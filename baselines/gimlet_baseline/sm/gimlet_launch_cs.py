"""Launch GIMLET clintox+sider (MolE-RAG test molecules) as a SageMaker job, using the
gimlet_clintox_sider_entry.py entry which extracts authentic per-subtask prompts from the HF
molecule_property_instruction dataset, injects them, then runs the scaffold branch with the
MolE-RAG test override. Finishes GIMLET 9/9."""
import sagemaker, boto3, sys, json
from sagemaker.pytorch import PyTorch
ROLE="arn:aws:iam::833984135295:role/service-role/AmazonSageMaker-ExecutionRole-20260922T220415"
BUCKET="molerag"
REPO="/work/hdd/bhrm/jchan5/mole-rag_naacl/gimlet_baseline/code/GIMLET"
INST=sys.argv[1] if len(sys.argv)>1 else "ml.g4dn.4xlarge"
TASKS=sys.argv[2] if len(sys.argv)>2 else "clintox,sider"
sess=sagemaker.Session(boto3.Session(region_name="us-east-1"))
est=PyTorch(entry_point="gimlet_clintox_sider_entry.py", source_dir=REPO, role=ROLE,
    framework_version="1.9.1", py_version="py38",
    instance_type=INST, instance_count=1, volume_size=80,
    output_path=f"s3://{BUCKET}/gimlet_baseline/output_cs",
    code_location=f"s3://{BUCKET}/gimlet_baseline/code_cs",
    base_job_name="gimlet-cs", max_run=5*3600,
    environment={"TASKS":TASKS,"HF_HOME":"/tmp/hf","TOKENIZERS_PARALLELISM":"false"},
    sagemaker_session=sess, disable_profiler=True)
est.fit(wait=False)
jn=est.latest_training_job.name
print("JOB_NAME",jn,flush=True)
json.dump([{"job":jn,"tasks":TASKS,"instance":INST}],open("gimlet_jobs_cs.json","w"))
