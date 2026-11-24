#!/usr/bin/env bash
# SageMaker Processing entrypoint: heavy CPU downstream for a LARGE dataset (HIV).
# Download OA full text + chunk + BM25 prescreen on a big ml.c6i box, push passages +
# BM25 candidates to S3. The Bedrock qwen3-32b JUDGE stays on DELTA (token local) reading
# candidates from S3 -> NO credential transfer. SageMaker auto-terminates the job at exit.
#
# Mirrors launch_hiv_cpu_ec2.sh's user-data, but:
#   * heavy dirs live UNDER /opt/ml/processing (the attached EBS volume, VolumeSizeInGB),
#     NOT the small container root fs.
#   * no self-terminate / no 6h watchdog needed (SageMaker StoppingCondition handles it).
#   * runs with SageMakerExecutionRole (S3FullAccess) via the container credential chain.
set -ux
DS="${1:?usage: sm_chunk_bm25_entry.sh <dataset>}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
PY="$(command -v python3)"
WORK=/opt/ml/processing/work
CODE=/opt/ml/processing/input/code          # staged by ProcessingInput
# If code was not staged as an input, pull it from S3 (either path works).
if [ ! -f "$CODE/run_fulltext.py" ]; then
  CODE=/opt/ml/processing/code; mkdir -p "$CODE"
  aws s3 sync s3://molerag/exp70_chunks/multi_code/ "$CODE/" --only-show-errors
fi

# deps (sklearn image has numpy/scipy/boto3; add the rest + awscli at runtime)
pip install -q --no-input bm25s PyStemmer lxml rdkit requests awscli 2>&1 | tail -3 || true
command -v aws >/dev/null 2>&1 || { echo "FATAL: aws CLI unavailable after pip"; exit 2; }

NCPU="$(nproc 2>/dev/null || echo 32)"
export EXP70_DATASET="$DS"
export EXP70_RUN_DIR="$WORK/$DS"
export EXP70_BULK_ROOT="$WORK/bulk/chunks"
export EXP70_FILTER_BASE="$WORK/bulk"
export EXP70_BULK_DIR="$WORK/bulk_ftp"
export EXP70_CHUNK_PROCS="$NCPU"
export EXP70_READ_WORKERS="$NCPU"
export EXP70_RETRIEVE_THREADS="$NCPU"
mkdir -p "$EXP70_RUN_DIR/data" "$EXP70_RUN_DIR/state" "$EXP70_BULK_ROOT" "$EXP70_FILTER_BASE" "$EXP70_BULK_DIR"

# pull the collected registry + identities (produced by Delta collection/merge)
aws s3 sync "s3://molerag/${DS}_chunks/collect_state/data/"  "$EXP70_RUN_DIR/data/"  --only-show-errors
aws s3 sync "s3://molerag/${DS}_chunks/collect_state/state/" "$EXP70_RUN_DIR/state/" --only-show-errors

# heavy CPU downstream: download OA + chunk, then BM25 prescreen (each pushes its own
# outputs to S3, same as the Delta/EC2 path).
"$PY" "$CODE/run_fulltext.py"   || echo "run_fulltext nonzero"
"$PY" "$CODE/bm25_prescreen.py" || echo "bm25_prescreen nonzero"

# signal Delta that candidates are ready to judge
echo "$(date -u) sagemaker $(hostname)" > "$WORK/cpu_done.txt"
aws s3 cp "$WORK/cpu_done.txt" "s3://molerag/${DS}_chunks/CPU_DOWNSTREAM_DONE.txt" --only-show-errors
# also drop a status file into the processing output (auto-uploaded by SageMaker)
mkdir -p /opt/ml/processing/output
cp "$WORK/cpu_done.txt" /opt/ml/processing/output/status.txt 2>/dev/null || true
echo "SM_CHUNK_BM25_DONE $DS"
