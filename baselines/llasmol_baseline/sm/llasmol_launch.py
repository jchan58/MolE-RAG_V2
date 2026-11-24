"""Launch the LlaSMol (Mistral-7B) released-checkpoint INFERENCE baseline on SageMaker.

LlaSMol is inference-only (no fine-tuning, no seeds), so a single job covers all 6 supported SMolInstruct
property-prediction tasks (esol,lipo,bbbp,clintox,hiv,sider). The whole LLM4Chem repo (code/LLM4Chem) is
the source_dir (so model.py/generation.py/utils/ import directly); requirements.txt at its root rebuilds
the stack; smol_templates/ bundles LlaSMol's verbatim SMolInstruct prompt templates. base Mistral-7B-v0.1
+ the LlaSMol LoRA adapter are downloaded from HF in-container (neither gated). Data channel from S3.

Quota = 1 per instance type; our pool = ml.g5.12xlarge, ml.g5.24xlarge (A10G 24GB). A 7B bf16 model fits
on ONE A10G, so CUDA_VISIBLE_DEVICES=0 pins the whole model to one GPU. Falls back across the two types.

Usage: python llasmol_launch.py [tasks_csv] [instance_type] [smoke]
"""
import sagemaker, boto3, sys, json, os
from sagemaker.pytorch import PyTorch

ROLE = "arn:aws:iam::833984135295:role/service-role/AmazonSageMaker-ExecutionRole-20260922T220415"
BUCKET = "molerag"
SRC = "/workspace/mole-rag_naacl/llasmol_baseline/code/LLM4Chem"
HERE = os.path.dirname(os.path.abspath(__file__))

tasks = sys.argv[1] if len(sys.argv) > 1 else "esol,lipo,bbbp,clintox,hiv,sider"
inst = sys.argv[2] if len(sys.argv) > 2 else "ml.g5.12xlarge"
smoke = sys.argv[3] if len(sys.argv) > 3 else "0"

sess = sagemaker.Session(boto3.Session(region_name="us-east-1"))

# MolE-RAG per-molecule test splits live in the dmpnn baseline; copy into OUR prefix only.
DATA_SRC = "/workspace/mole-rag_naacl/dmpnn_baseline/data"
data_s3 = sess.upload_data(path=DATA_SRC, bucket=BUCKET, key_prefix="llasmol_baseline/data")
print("data ->", data_s3, flush=True)

env = {
    "TASKS": tasks, "SMOKE": smoke,
    "MODEL_NAME": "osunlp/LlaSMol-Mistral-7B", "BASE_MODEL": "mistralai/Mistral-7B-v0.1",
    "BATCH_SIZE": "1", "MAX_IN": "512", "MAX_OUT": "20", "NUM_BEAMS": "4",
    "S3_BUCKET": BUCKET,
    "S3_INCR": "llasmol_baseline/output/incremental" + ("_smoke" if smoke == "1" else ""),
    "HF_HOME": "/opt/ml/hf_cache", "TOKENIZERS_PARALLELISM": "false",
    "CUDA_VISIBLE_DEVICES": "0",          # pin the 7B model to a single A10G
    "TORCHDYNAMO_DISABLE": "1",           # disable torch.compile (speed-only; no effect on outputs)
    "PYTHONNOUSERSITE": "1",
}
base = ("llasmol-" + ("smoke-" if smoke == "1" else "") + tasks.split(",")[0])[:40]
est = PyTorch(
    entry_point="naacl_entry.py", source_dir=SRC, role=ROLE,
    framework_version="2.3", py_version="py311",
    instance_type=inst, instance_count=1, volume_size=140,
    output_path=f"s3://{BUCKET}/llasmol_baseline/output",
    code_location=f"s3://{BUCKET}/llasmol_baseline/code",
    base_job_name=base, max_run=20 * 3600,
    environment=env, sagemaker_session=sess, disable_profiler=True,
    keep_alive_period_in_seconds=0,
)
est.fit(inputs={"train": data_s3}, wait=False)
name = est.latest_training_job.name
print("JOB_NAME", name, flush=True)

reg_path = os.path.join(HERE, "llasmol_jobs.json")
reg = json.load(open(reg_path)) if os.path.exists(reg_path) else []
reg.append({"job": name, "tasks": tasks, "instance": inst, "smoke": smoke})
json.dump(reg, open(reg_path, "w"), indent=2)
print("registered", name, flush=True)
