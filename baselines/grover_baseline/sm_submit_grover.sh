#!/bin/bash
# Package GROVER-Large (GODE m_level fork) + grover_large.pt + our seed_0 data into a
# SageMaker sourcedir, upload to S3, launch a PyTorch-1.13 DLC training job via boto3.
# TASKS env controls scope ("bbbp" for a smoke test, "all" for the full 9).
set -euo pipefail
BASE=/work/hdd/bhrm/jchan5/mole-rag_naacl/grover_baseline
MLEVEL=/work/hdd/bhrm/jchan5/mole-rag_naacl/gode/Gode/m_level_pretrain_and_fine_tune
KANODATA=/work/hdd/bhrm/jchan5/mole-rag_naacl/kano_baseline/data   # reuse the wide CSVs
BUILD=$BASE/sm_build
BUCKET=sagemaker-us-east-1-833984135295
PREFIX=grover_large
IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker
ROLE=arn:aws:iam::833984135295:role/SageMakerExecutionRole
INSTANCE=${INSTANCE:-ml.g5.12xlarge}
TASKS=${TASKS:-all}
EPOCHS=${EPOCHS:-20}
PY=/u/jchan5/miniconda3/bin/python

echo "[1/4] staging sourcedir -> $BUILD"
rm -rf "$BUILD"; mkdir -p "$BUILD"
cp -r "$MLEVEL"/grover "$MLEVEL"/task "$MLEVEL"/scripts "$BUILD"/
cp "$MLEVEL"/main.py "$MLEVEL"/setup.py "$BUILD"/ 2>/dev/null || true
cp "$BASE"/sm_entry_grover.py "$BASE"/requirements.txt "$BUILD"/
cp "$BASE"/grover_large.pt "$BUILD"/
cp -r "$KANODATA" "$BUILD"/sm_data
echo "  staged: $(du -sh "$BUILD" | cut -f1)"

echo "[2/4] tar + upload"
tar czf "$BASE/sourcedir.tar.gz" -C "$BUILD" .
echo "  tarball: $(du -sh "$BASE/sourcedir.tar.gz" | cut -f1)"
aws s3 cp "$BASE/sourcedir.tar.gz" "s3://$BUCKET/$PREFIX/sourcedir.tar.gz" --only-show-errors
S3SRC="s3://$BUCKET/$PREFIX/sourcedir.tar.gz"
S3OUT="s3://$BUCKET/$PREFIX/output/"

echo "[3/4] create training job ($INSTANCE, tasks=$TASKS)"
JOB="grover-large-$(date +%Y%m%d-%H%M%S)"
S3SRC="$S3SRC" S3OUT="$S3OUT" IMAGE="$IMAGE" ROLE="$ROLE" JOB="$JOB" INSTANCE="$INSTANCE" TASKS="$TASKS" EPOCHS="$EPOCHS" "$PY" - <<'PYEOF'
import os, json, boto3
sm = boto3.client("sagemaker", region_name="us-east-1")
hp = {k: json.dumps(v) for k, v in {
    "sagemaker_program": "sm_entry_grover.py",
    "sagemaker_submit_directory": os.environ["S3SRC"],
    "sagemaker_region": "us-east-1",
    "tasks": os.environ["TASKS"],
    "epochs": os.environ["EPOCHS"],
}.items()}
sm.create_training_job(
    TrainingJobName=os.environ["JOB"],
    AlgorithmSpecification={"TrainingImage": os.environ["IMAGE"], "TrainingInputMode": "File"},
    RoleArn=os.environ["ROLE"], HyperParameters=hp,
    OutputDataConfig={"S3OutputPath": os.environ["S3OUT"]},
    ResourceConfig={"InstanceType": os.environ["INSTANCE"], "InstanceCount": 1, "VolumeSizeInGB": 120},
    StoppingCondition={"MaxRuntimeInSeconds": 50400})
print("CREATED", os.environ["JOB"], "on", os.environ["INSTANCE"], "tasks=", os.environ["TASKS"])
PYEOF
echo "$JOB" > "$BASE/execution/sm_job_grover.txt"
echo "[4/4] SM_SUBMIT_DONE $JOB"
