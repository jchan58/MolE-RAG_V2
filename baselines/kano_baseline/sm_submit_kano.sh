#!/bin/bash
# Package KANO (CMPNN) + our seed_0 data into a SageMaker sourcedir, upload to S3,
# and launch a SageMaker PyTorch-2.1 DLC training job via boto3 (light; no sagemaker SDK).
set -euo pipefail
BASE=/workspace/mole-rag_naacl/kano_baseline
KANO=/workspace/mole-rag_naacl/kano/KANO
BUILD=$BASE/sm_build
BUCKET=sagemaker-us-east-1-833984135295
PREFIX=kano_cmpnn
IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker
ROLE=arn:aws:iam::833984135295:role/SageMakerExecutionRole
PY=/home/user/miniconda3/bin/python   # for boto3 only

echo "[1/4] staging sourcedir -> $BUILD"
rm -rf "$BUILD"; mkdir -p "$BUILD"
# KANO code + assets needed at finetune time (exclude original data/ 39M and fig/)
cp -r "$KANO"/chemprop "$KANO"/initial "$KANO"/dumped "$KANO"/KGembedding "$BUILD"/
cp "$KANO"/train.py "$KANO"/predict.py "$BUILD"/ 2>/dev/null || true
cp "$BASE"/sm_entry.py "$BASE"/requirements.txt "$BUILD"/
# our prepared seed_0 wide CSVs -> sm_data/
cp -r "$BASE"/data "$BUILD"/sm_data
echo "  staged: $(du -sh "$BUILD" | cut -f1)"

echo "[2/4] tar + upload"
tar czf "$BASE/sourcedir.tar.gz" -C "$BUILD" .
echo "  tarball: $(du -sh "$BASE/sourcedir.tar.gz" | cut -f1)"
aws s3 cp "$BASE/sourcedir.tar.gz" "s3://$BUCKET/$PREFIX/sourcedir.tar.gz" --only-show-errors
S3SRC="s3://$BUCKET/$PREFIX/sourcedir.tar.gz"
S3OUT="s3://$BUCKET/$PREFIX/output/"
echo "  uploaded -> $S3SRC"

echo "[3/4] create training job (boto3)"
JOB="kano-cmpnn-$(date +%Y%m%d-%H%M%S)"
S3SRC="$S3SRC" S3OUT="$S3OUT" IMAGE="$IMAGE" ROLE="$ROLE" JOB="$JOB" "$PY" - <<'PYEOF'
import os, json, boto3
sm = boto3.client("sagemaker", region_name="us-east-1")
hp = {k: json.dumps(v) for k, v in {
    "sagemaker_program": "sm_entry.py",
    "sagemaker_submit_directory": os.environ["S3SRC"],
    "sagemaker_region": "us-east-1",
    "encoder": "CMPNN",
    "tag": "cmpnn",
    "ckpt": "./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl",
    "epochs": "100",
    "num_runs": "1",
}.items()}
sm.create_training_job(
    TrainingJobName=os.environ["JOB"],
    AlgorithmSpecification={"TrainingImage": os.environ["IMAGE"], "TrainingInputMode": "File"},
    RoleArn=os.environ["ROLE"],
    HyperParameters=hp,
    OutputDataConfig={"S3OutputPath": os.environ["S3OUT"]},
    ResourceConfig={"InstanceType": "ml.g5.12xlarge", "InstanceCount": 1, "VolumeSizeInGB": 100},
    StoppingCondition={"MaxRuntimeInSeconds": 36000},
    EnableManagedSpotTraining=False,
)
print("CREATED", os.environ["JOB"])
PYEOF

echo "[4/4] submitted. poll: aws sagemaker describe-training-job --training-job-name $JOB --query TrainingJobStatus"
echo "$JOB" > "$BASE/execution/sm_job_cmpnn.txt"
echo "SM_SUBMIT_DONE $JOB"
