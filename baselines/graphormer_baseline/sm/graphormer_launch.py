"""Launch a Graphormer (PCQM4Mv2-pretrained) fine-tune SageMaker job.
Usage: python graphormer_launch.py <task[,task2,...]> [instance_type] [seeds] [smoke] [epochs]
The PCQM4Mv2 checkpoint is provided as the 'ckpt' S3 input channel."""
import sagemaker, boto3, sys, json, os
from sagemaker.pytorch import PyTorch
ROLE = "arn:aws:iam::833984135295:role/service-role/AmazonSageMaker-ExecutionRole-20260922T220415"
BUCKET = "molerag"
REPO = "/workspace/mole-rag_naacl/graphormer_baseline/code/GIMLET"
HERE = os.path.dirname(os.path.abspath(__file__))

task = sys.argv[1]
inst = sys.argv[2] if len(sys.argv) > 2 else "ml.g5.16xlarge"
seeds = sys.argv[3] if len(sys.argv) > 3 else "0,1,2"
smoke = sys.argv[4] if len(sys.argv) > 4 else "0"
epochs = sys.argv[5] if len(sys.argv) > 5 else ""

sess = sagemaker.Session(boto3.Session(region_name="us-east-1"))
env = {"TASKS": task, "SEEDS": seeds, "SMOKE": smoke,
       "HF_HOME": "/tmp/hf", "TOKENIZERS_PARALLELISM": "false"}
if epochs:
    env["EPOCHS"] = epochs
base = ("graphormer-" + task.replace(",", "-"))[:38]
est = PyTorch(entry_point="graphormer_entry.py", source_dir=REPO, role=ROLE,
              framework_version="1.9.1", py_version="py38",
              instance_type=inst, instance_count=1, volume_size=80,
              output_path=f"s3://{BUCKET}/graphormer_baseline/output",
              code_location=f"s3://{BUCKET}/graphormer_baseline/code",
              base_job_name=base, max_run=22 * 3600,
              environment=env, sagemaker_session=sess, disable_profiler=True)
est.fit(inputs={"ckpt": f"s3://{BUCKET}/graphormer_baseline/ckpt/"}, wait=False)
name = est.latest_training_job.name
print("JOB_NAME", name, flush=True)
# append to jobs registry
reg_path = os.path.join(HERE, "graphormer_jobs.json")
reg = json.load(open(reg_path)) if os.path.exists(reg_path) else []
reg.append({"job": name, "tasks": task, "instance": inst, "seeds": seeds, "smoke": smoke, "epochs": epochs})
json.dump(reg, open(reg_path, "w"), indent=2)
