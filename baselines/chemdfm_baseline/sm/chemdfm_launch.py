"""Launch the ChemDFM-v1.0-13B (released checkpoint, zero-shot) baseline as parallel SageMaker jobs.

ChemDFM-13B (~26GB fp16) needs an L40S (48GB). Our quota pool = {ml.g6e.2xlarge, ml.g6e.8xlarge,
ml.g6e.12xlarge} (quota 1 per type). We use the two SINGLE-GPU types (2xlarge, 8xlarge) for two
balanced shards (~9.3k prompts each). The model is downloaded from the HF hub in-container
(HF_HOME=/tmp/hf). Inference only -- no training, single greedy pass, no seeds.

All S3 I/O stays under s3://molerag/chemdfm_baseline/. Restart-safe: entry.py streams per-task
results to s3://molerag/chemdfm_baseline/output/incremental/<task>.json and skips tasks already
done on resume.

Usage: python chemdfm_launch.py [only_tags_comma]   (default launches both shards)
"""
import sagemaker, boto3, json, sys, os
from sagemaker.pytorch import PyTorch

ROLE = "arn:aws:iam::833984135295:role/service-role/AmazonSageMaker-ExecutionRole-20260922T220415"
BUCKET = "molerag"
HERE = os.path.dirname(os.path.abspath(__file__))
sess = sagemaker.Session(boto3.Session(region_name="us-east-1"))

DATA_SRC = "/work/hdd/bhrm/jchan5/mole-rag_naacl/dmpnn_baseline/data"
data_s3 = sess.upload_data(path=DATA_SRC, bucket=BUCKET, key_prefix="chemdfm_baseline/data")
print("data ->", data_s3, flush=True)

# (tag, tasks, instance, max_run_hours, batch).  ~9.3k prompts per shard.
SHARDS = [
    ("tox21", "tox21", "ml.g6e.2xlarge", 10, "16"),                               # 783*12 = 9396 prompts
    ("mix", "hiv,sider,bbbp,bace,clintox,esol,freesolv,lipo", "ml.g6e.8xlarge", 10, "16"),  # ~9254
]
# Fallback order per shard if a type hits ResourceLimit/Capacity.
FALLBACK = ["ml.g6e.2xlarge", "ml.g6e.8xlarge", "ml.g6e.12xlarge"]

only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None

reg_path = os.path.join(HERE, "chemdfm_jobs.json")
jobs = json.load(open(reg_path)) if os.path.exists(reg_path) else []
launched = []


def make_est(tag, tasks, inst, hrs, batch):
    return PyTorch(
        entry_point="entry.py", source_dir=HERE, role=ROLE,
        framework_version="2.3", py_version="py311",
        instance_type=inst, instance_count=1, volume_size=120,
        output_path=f"s3://{BUCKET}/chemdfm_baseline/output",
        code_location=f"s3://{BUCKET}/chemdfm_baseline/code",
        base_job_name=f"chemdfm-{tag}"[:38], max_run=hrs * 3600,
        environment={
            "TASKS": tasks, "BATCH_SIZE": batch,
            "MODEL_NAME": "OpenDFM/ChemDFM-v1.0-13B",
            "MAX_IN": "1024", "CLS_NEW_TOKENS": "6", "REG_NEW_TOKENS": "24",
            "S3_BUCKET": BUCKET, "S3_INCR": "chemdfm_baseline/output/incremental",
            "RESUME": "1", "HF_HOME": "/tmp/hf", "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
        },
        sagemaker_session=sess, disable_profiler=True,
    )


for tag, tasks, inst, hrs, batch in SHARDS:
    if only and tag not in only:
        continue
    tried = [inst] + [x for x in FALLBACK if x != inst]
    jn = None
    for it in tried:
        try:
            est = make_est(tag, tasks, it, hrs, batch)
            est.fit({"train": data_s3}, wait=False)
            jn = est.latest_training_job.name
            inst = it
            break
        except Exception as e:
            print(f"  {tag}: {it} failed to launch ({str(e)[:160]}); trying next", flush=True)
    if jn is None:
        print(f"  {tag}: ALL instance types failed to launch", flush=True)
        continue
    rec = {"job": jn, "tag": tag, "tasks": tasks, "instance": inst, "batch": batch}
    jobs.append(rec); launched.append(rec)
    print("LAUNCHED", jn, "|", tasks, "|", inst, flush=True)

json.dump(jobs, open(reg_path, "w"), indent=2)
print("ALL LAUNCHED:", len(launched), "jobs (registry total:", len(jobs), ")")
