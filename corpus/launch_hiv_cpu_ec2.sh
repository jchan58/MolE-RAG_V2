#!/usr/bin/env bash
# CPU-EC2 heavy downstream for a LARGE dataset (HIV): download OA full text + chunk + BM25
# prescreen on a c6i.12xlarge (48 vCPU, standard on-demand vCPU quota — separate from the
# GPU quota). Pushes passages + BM25 candidates to S3. The Bedrock qwen3-32b JUDGE stays on
# DELTA (token local) reading candidates from S3 -> NO credential transfer to the box.
# Self-terminates when done. Usage: bash launch_hiv_cpu_ec2.sh [dataset=hiv] [itype=c6i.12xlarge]
set -u
export AWS_DEFAULT_REGION=us-east-1
DS="${1:-hiv}"; ITYPE="${2:-c6i.12xlarge}"
AMI="${EC2_AMI:-ami-012ba162b9cd2729c}"          # DLAMI (has python/awscli); any Ubuntu 22.04 works
SUBNET="${EC2_SUBNET:-subnet-07db442d5f467916a}"
PROFILE="${EC2_PROFILE:-exp70-ec2-s3}"

UD=$(cat <<EOF
#!/bin/bash
set -x
exec > /var/log/${DS}_cpu_boot.log 2>&1
# COST-SAFETY watchdog: hard self-terminate after 6h regardless (bug/hang guard)
( sleep 21600; T=\$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 120"); I=\$(curl -s -H "X-aws-ec2-metadata-token: \$T" http://169.254.169.254/latest/meta-data/instance-id); aws ec2 terminate-instances --region $AWS_DEFAULT_REGION --instance-ids \$I ) &
PY=\$(command -v python3)
pip install -q bm25s PyStemmer boto3 lxml rdkit requests 2>&1 | tail -2
mkdir -p /opt/exp70/code && aws s3 sync s3://molerag/exp70_chunks/multi_code/ /opt/exp70/code/ --only-show-errors
export EXP70_DATASET=$DS
export EXP70_RUN_DIR=/opt/exp70/$DS
export EXP70_BULK_ROOT=/opt/bulk/chunks
export EXP70_FILTER_BASE=/opt/bulk
export EXP70_BULK_DIR=/opt/exp70/bulk_ftp   # not needed downstream (identity already done)
export EXP70_CHUNK_PROCS=40 EXP70_READ_WORKERS=40 EXP70_RETRIEVE_THREADS=40
mkdir -p \$EXP70_RUN_DIR/data \$EXP70_RUN_DIR/state \$EXP70_BULK_ROOT \$EXP70_FILTER_BASE
# pull the collected registry + identities (produced by Delta collection/merge)
aws s3 sync s3://molerag/${DS}_chunks/collect_state/data/ \$EXP70_RUN_DIR/data/ --only-show-errors
aws s3 sync s3://molerag/${DS}_chunks/collect_state/state/ \$EXP70_RUN_DIR/state/ --only-show-errors
# heavy CPU downstream: download OA + chunk, then BM25 prescreen -> candidates to S3
\$PY /opt/exp70/code/run_fulltext.py   || true
\$PY /opt/exp70/code/bm25_prescreen.py || true
# signal Delta that candidates are ready to judge
echo "\$(date -u)" > /tmp/cpu_done.txt
aws s3 cp /tmp/cpu_done.txt s3://molerag/${DS}_chunks/CPU_DOWNSTREAM_DONE.txt --only-show-errors
# self-terminate
TOKEN=\$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
IID=\$(curl -s -H "X-aws-ec2-metadata-token: \$TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --region $AWS_DEFAULT_REGION --instance-ids \$IID
EOF
)
echo "$UD" > /tmp/${DS}_cpu_userdata.sh
aws ec2 run-instances --image-id "$AMI" --instance-type "$ITYPE" --subnet-id "$SUBNET" \
  --iam-instance-profile Name="$PROFILE" \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=500,VolumeType=gp3}' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=exp70-multi-${DS}-cpu}]" \
  --user-data file:///tmp/${DS}_cpu_userdata.sh \
  --query 'Instances[0].InstanceId' --output text
