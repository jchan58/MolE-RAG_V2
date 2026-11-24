"""Launch a BioT5 (biot5-base, fine-tuned) MoleculeNet SageMaker job for a batch of datasets.
Usage: python biot5_launch.py <groups csv> [instance_type] [seeds] [smoke]
  groups: comma list of datasets, e.g. "bbbp,bace,hiv" or "sider" or "tox21"
The whole BioT5 repo (code/BioT5) is the source_dir; requirements.txt at its root reconstructs the
stack. biot5-base weights are downloaded from HF inside the container. Data channel from S3."""
import sagemaker, boto3, sys, json, os
from sagemaker.pytorch import PyTorch

ROLE = "arn:aws:iam::833984135295:role/service-role/AmazonSageMaker-ExecutionRole-20260922T220415"
BUCKET = "molerag"
SRC = "/work/hdd/bhrm/jchan5/mole-rag_naacl/biot5_baseline/code/BioT5"
HERE = os.path.dirname(os.path.abspath(__file__))

groups = sys.argv[1]
inst = sys.argv[2] if len(sys.argv) > 2 else "ml.g5.2xlarge"
seeds = sys.argv[3] if len(sys.argv) > 3 else "0"
smoke = sys.argv[4] if len(sys.argv) > 4 else "0"
sl = sys.argv[5] if len(sys.argv) > 5 else ""   # optional "start:end" target slice

sess = sagemaker.Session(boto3.Session(region_name="us-east-1"))
env = {"GROUPS": groups, "SEEDS": seeds, "SMOKE": smoke,
       "HF_HOME": "/tmp/hf", "TOKENIZERS_PARALLELISM": "false",
       "S3_BUCKET": BUCKET, "S3_INCR": "biot5_baseline/output/incremental"}
if sl:
    env["SLICE"] = sl
tag = groups.replace(",", "-") + (("-" + sl.replace(":", "_")) if sl else "")
base = ("biot5-" + ("smoke-" if smoke == "1" else "") + tag)[:40]
est = PyTorch(entry_point="biot5/biot5_entry.py", source_dir=SRC, role=ROLE,
              framework_version="2.0.1", py_version="py310",
              instance_type=inst, instance_count=1, volume_size=100,
              output_path=f"s3://{BUCKET}/biot5_baseline/output",
              code_location=f"s3://{BUCKET}/biot5_baseline/code",
              base_job_name=base, max_run=36 * 3600,
              environment=env, sagemaker_session=sess, disable_profiler=True,
              keep_alive_period_in_seconds=0)
est.fit(inputs={"train": f"s3://{BUCKET}/biot5_baseline/data/"}, wait=False)
name = est.latest_training_job.name
print("JOB_NAME", name, flush=True)

reg_path = os.path.join(HERE, "biot5_jobs.json")
reg = json.load(open(reg_path)) if os.path.exists(reg_path) else []
reg.append({"job": name, "groups": groups, "instance": inst, "seeds": seeds, "smoke": smoke})
json.dump(reg, open(reg_path, "w"), indent=2)
