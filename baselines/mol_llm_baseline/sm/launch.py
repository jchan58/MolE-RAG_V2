"""Launch the MolLM (Gerstein Lab, 2D+3D) fine-tune as ONE SageMaker job running all 9 tasks
sequentially on one A10G GPU. MODE=smoke validates the environment first; MODE=full runs everything.

Instance: A10G (g5 family, sm_86) -- natively supported by the PyTorch 1.9.1 / cu111 DLC the bundled
fairseq/Transformer-M stack targets. Tries a smaller free g5 type, falls back to the sanctioned pool
type on ResourceLimitExceeded. Quota is 1 per instance type."""
import sys, json
import sagemaker, boto3
from sagemaker.pytorch import PyTorch
from botocore.exceptions import ClientError

ROLE = "arn:aws:iam::833984135295:role/service-role/AmazonSageMaker-ExecutionRole-20260922T220415"
BUCKET = "molerag"
sess = sagemaker.Session(boto3.Session(region_name="us-east-1"))

MODE = sys.argv[1] if len(sys.argv) > 1 else "smoke"
TASKS = sys.argv[2] if len(sys.argv) > 2 else ""
FW = sys.argv[3] if len(sys.argv) > 3 else "1.9.1"   # framework_version (torch)
PYV = {"1.9.1": "py38", "1.12.1": "py38", "2.0.1": "py310", "2.2.0": "py310"}.get(FW, "py38")
# Account quota is on the g6e family (L40S); the 48xl/5.48xl "pool" is actually 0.
# smoke: smallest free g6e; full: more vCPUs for conformer featurisation.
if MODE == "smoke":
    CANDIDATES = ["ml.g6e.4xlarge", "ml.g6e.2xlarge", "ml.g6e.12xlarge", "ml.g6e.16xlarge"]
else:
    CANDIDATES = ["ml.g6e.16xlarge", "ml.g6e.12xlarge", "ml.g6e.4xlarge"]

DATA_SRC = "/work/hdd/bhrm/jchan5/mole-rag_naacl/dmpnn_baseline/data"
data_s3 = sess.upload_data(path=DATA_SRC, bucket=BUCKET, key_prefix="mol_llm_baseline/data")
print("data ->", data_s3, flush=True)

env = {
    "MODE": MODE, "SEED": "0", "EPOCHS": "50", "BATCH_SIZE": "32", "PATIENCE": "15",
    "MAX_TASK_MIN": "180", "LR": "1e-4", "USE_3D": "1",
    "S3_BUCKET": BUCKET, "S3_INCR": "mol_llm_baseline/output/incremental",
    "S3_CKPT_PREFIX": "mol_llm_baseline/ckpt",
    "HF_HOME": "/tmp/hf", "OMP_NUM_THREADS": "4",
    # torch 1.9.1's @torch.jit.script `gaussian` (3D bias) gets TE-fused + nvrtc-compiled for the
    # L40S (sm_89), which the bundled nvrtc (cu111) cannot target. PYTORCH_JIT=0 makes jit.script a
    # no-op so it runs eagerly (precompiled kernels, identical numerics, no nvrtc).
    "PYTORCH_JIT": "0", "PYTORCH_TENSOREXPR": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256",
}
if TASKS:
    env["TASKS"] = TASKS

max_run = 2 * 3600 if MODE == "smoke" else 24 * 3600

last_err = None
for inst in CANDIDATES:
    try:
        est = PyTorch(
            entry_point="entry.py", source_dir=".", role=ROLE,
            framework_version=FW, py_version=PYV,
            instance_type=inst, instance_count=1, volume_size=200,
            output_path=f"s3://{BUCKET}/mol_llm_baseline/output",
            code_location=f"s3://{BUCKET}/mol_llm_baseline/code",
            base_job_name=f"mollm-{MODE}", max_run=max_run,
            environment=env, sagemaker_session=sess, disable_profiler=True,
        )
        est.fit({"train": data_s3}, wait=False)
        jn = est.latest_training_job.name
        rec = {"job": jn, "instance": inst, "mode": MODE, "tasks": TASKS}
        json.dump(rec, open(f"job_{MODE}.json", "w"), indent=2)
        print("LAUNCHED", jn, "|", inst, "| MODE=", MODE, "| TASKS=", TASKS or "ALL", flush=True)
        break
    except ClientError as e:
        last_err = e
        if "ResourceLimitExceeded" in str(e):
            print("ResourceLimitExceeded on", inst, "-> trying next", flush=True)
            continue
        raise
else:
    raise SystemExit(f"all instance types exhausted: {last_err}")
