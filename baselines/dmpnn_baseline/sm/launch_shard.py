"""Launch the DMPNN baseline as several PARALLEL SageMaker jobs, one per distinct GPU instance type
(each type has quota 1, but ~28 types exist). Shards the 9 tasks to balance the long poles."""
import sagemaker, boto3, json, time
from sagemaker.pytorch import PyTorch
ROLE="arn:aws:iam::833984135295:role/service-role/AmazonSageMaker-ExecutionRole-20260922T220415"
BUCKET="molerag"
sess=sagemaker.Session(boto3.Session(region_name="us-east-1"))
data_s3=sess.upload_data(path="../data", bucket=BUCKET, key_prefix="dmpnn_baseline/data")
print("data ->",data_s3,flush=True)
# shard: each big task its own instance; small ones batched. (task-subset, instance-type)
SHARDS=[
 ("hiv","ml.g5.2xlarge"),
 ("tox21","ml.g4dn.2xlarge"),
 ("sider,lipo","ml.g6.2xlarge"),
 ("esol,freesolv,bbbp,bace,clintox","ml.g5.4xlarge"),
]
jobs=[]
for tasks,inst in SHARDS:
    tag=tasks.split(",")[0]
    est=PyTorch(entry_point="entry.py", source_dir=".", role=ROLE,
        framework_version="2.3", py_version="py311",
        instance_type=inst, instance_count=1, volume_size=60,
        output_path=f"s3://{BUCKET}/dmpnn_baseline/output",
        code_location=f"s3://{BUCKET}/dmpnn_baseline/code",
        base_job_name=f"dmpnn-{tag}", max_run=4*3600,
        environment={"SEEDS":"0,1,2","EPOCHS":"50","TASKS":tasks},
        sagemaker_session=sess, disable_profiler=True)
    est.fit({"train":data_s3}, wait=False)
    jn=est.latest_training_job.name; jobs.append({"job":jn,"tasks":tasks,"instance":inst})
    print("LAUNCHED",jn,"|",tasks,"|",inst,flush=True)
json.dump(jobs,open("shard_jobs.json","w"),indent=2)
print("ALL LAUNCHED:",len(jobs),"jobs")
