"""Launch GIN-from-scratch as parallel SageMaker jobs on instance types NOT used by the DMPNN run
(each type quota=1). Uploads the minimal MolE-RAG splits, shards the 9 tasks."""
import sagemaker, boto3, json
from sagemaker.pytorch import PyTorch
ROLE="arn:aws:iam::833984135295:role/service-role/AmazonSageMaker-ExecutionRole-20260922T220415"
BUCKET="molerag"
sess=sagemaker.Session(boto3.Session(region_name="us-east-1"))
data_s3=sess.upload_data(path="data_stage/moleculenet_property_scaffold", bucket=BUCKET,
                         key_prefix="gin_baseline/data/moleculenet_property_scaffold")
# channel points at the dir whose child is 'seed_0'; sm_gin.py appends nothing extra -> it expects
# SM_CHANNEL_TRAINING/moleculenet_property_scaffold, so upload under that name:
print("data ->",data_s3,flush=True)
SHARDS=[
 ("hiv","ml.g5.8xlarge"),
 ("tox21","ml.g6.4xlarge"),
 ("sider,lipo","ml.g6.xlarge"),
 ("esol,freesolv,bbbp,bace,clintox","ml.g5.xlarge"),
]
jobs=[]
for tasks,inst in SHARDS:
    tag=tasks.split(",")[0]
    est=PyTorch(entry_point="sm_gin.py", source_dir=".", role=ROLE,
        framework_version="2.4", py_version="py311",
        instance_type=inst, instance_count=1, volume_size=60,
        output_path=f"s3://{BUCKET}/gin_baseline/output",
        code_location=f"s3://{BUCKET}/gin_baseline/code",
        base_job_name=f"gin-{tag}", max_run=6*3600,
        environment={"TASKS":tasks},
        sagemaker_session=sess, disable_profiler=True)
    est.fit({"training": data_s3}, wait=False)
    jn=est.latest_training_job.name; jobs.append({"job":jn,"tasks":tasks,"instance":inst})
    print("LAUNCHED",jn,"|",tasks,"|",inst,flush=True)
json.dump(jobs,open("gin_jobs.json","w"),indent=2)
print("ALL LAUNCHED:",len(jobs))
