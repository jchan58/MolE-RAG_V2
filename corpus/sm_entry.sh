#!/bin/bash
# SageMaker Processing entrypoint: heavy CPU download+chunk+BM25 for one dataset, off S3.
# Mirrors launch_hiv_cpu_ec2.sh's user-data body but runs as a managed SageMaker Processing
# job (auto-terminates, native S3 IO, SageMakerExecutionRole creds via container provider --
# NO credential transfer, role has S3FullAccess only). The Bedrock judge stays on Delta.
set -x
DS="${1:-hiv}"
export AWS_DEFAULT_REGION=us-east-1
# deps (sklearn image has python+numpy; add the rest at runtime -- confirmed by smoke test)
pip install -q --no-input awscli boto3 bm25s PyStemmer lxml rdkit requests 2>&1 | tail -3
hash -r
AWS=$(command -v aws); PY=$(command -v python3)
mkdir -p /opt/exp70/code
$AWS s3 sync s3://molerag/exp70_chunks/multi_code/ /opt/exp70/code/ --only-show-errors
export EXP70_DATASET=$DS
export EXP70_RUN_DIR=/opt/exp70/$DS
export EXP70_BULK_ROOT=/opt/bulk/chunks
export EXP70_FILTER_BASE=/opt/bulk
export EXP70_BULK_DIR=/opt/exp70/bulk_ftp        # not needed downstream (identity already done)
NP=$(nproc)
export EXP70_CHUNK_PROCS=$NP EXP70_READ_WORKERS=$NP EXP70_RETRIEVE_THREADS=$NP
mkdir -p $EXP70_RUN_DIR/data $EXP70_RUN_DIR/state $EXP70_BULK_ROOT $EXP70_FILTER_BASE /opt/ml/processing/output
# pull the collected registry + identities (produced by Delta collection/merge)
$AWS s3 sync s3://molerag/${DS}_chunks/collect_state/data/  $EXP70_RUN_DIR/data/  --only-show-errors
$AWS s3 sync s3://molerag/${DS}_chunks/collect_state/state/ $EXP70_RUN_DIR/state/ --only-show-errors
# heavy CPU downstream: download OA full text + chunk (passages tar -> S3), then BM25 prescreen
$PY /opt/exp70/code/run_fulltext.py   2>&1 | tail -40 || true
$PY /opt/exp70/code/bm25_prescreen.py 2>&1 | tail -40 || true
# signal Delta that candidates are ready to judge (both the output mount AND the root marker)
echo "$(date -u) nproc=$NP" > /opt/ml/processing/output/CPU_DOWNSTREAM_DONE.txt
$AWS s3 cp /opt/ml/processing/output/CPU_DOWNSTREAM_DONE.txt s3://molerag/${DS}_chunks/CPU_DOWNSTREAM_DONE.txt --only-show-errors
echo "SM_CHUNKBM25_DONE $DS"
