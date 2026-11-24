#!/usr/bin/env bash
# TURNKEY MedCPT GPU launch on EC2 (run WHEN a dataset's kept chunks + judgments are in S3).
# Launches a GPU box with a self-configuring user-data that: installs deps, pulls code +
# the datasets' passages/judgments from S3, runs medcpt_rerank (BM25 500/q -> MedCPT ->
# K200/300/400), pushes structuring/ to S3, then self-terminates. NO Bedrock token needed
# (MedCPT is a local model) -> sidesteps the credential-transfer block; uses the S3-only
# instance profile exp70-ec2-s3.
#
# Usage: bash launch_medcpt_ec2.sh "<ds1> <ds2> ..."   (e.g. "sider lipo tox21")
# GPU: default p3.2xlarge (V100, P-quota, independent of the BBBP g5's G-quota). For HIV
# scale use g5.12xlarge after a quota bump (L-DB2E81BA->96) or once BBBP's g5 frees.
set -u
export AWS_DEFAULT_REGION=us-east-1
DSLIST="${1:?usage: launch_medcpt_ec2.sh \"sider lipo tox21\"}"
ITYPE="${EC2_ITYPE:-p3.2xlarge}"
AMI="${EC2_AMI:-ami-012ba162b9cd2729c}"          # DLAMI PyTorch 2.7 Ubuntu 22.04 (has CUDA/torch)
SUBNET="${EC2_SUBNET:-subnet-07db442d5f467916a}"
PROFILE="${EC2_PROFILE:-exp70-ec2-s3}"

UD=$(cat <<EOF
#!/bin/bash
set -x
exec > /var/log/medcpt_boot.log 2>&1
# COST-SAFETY watchdog: hard self-terminate after 3h no matter what (bug/hang guard)
( sleep 10800; T=\$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 120"); I=\$(curl -s -H "X-aws-ec2-metadata-token: \$T" http://169.254.169.254/latest/meta-data/instance-id); aws ec2 terminate-instances --region $AWS_DEFAULT_REGION --instance-ids \$I ) &
PY=\$(command -v python3)
pip install -q transformers bm25s PyStemmer boto3 accelerate 2>&1 | tail -2
mkdir -p /opt/exp70/code && cd /opt/exp70
aws s3 sync s3://molerag/exp70_chunks/multi_code/ /opt/exp70/code/ --only-show-errors
for DS in $DSLIST; do
  export EXP70_DATASET=\$DS
  export EXP70_RUN_DIR=/opt/exp70/\$DS
  export EXP70_FILTER_BASE=/opt/exp70/filter_\$DS
  mkdir -p \$EXP70_RUN_DIR/structuring \$EXP70_FILTER_BASE/chunks \$EXP70_FILTER_BASE/judgments_qwen
  # pull query registry + kept judgments + passages from S3
  aws s3 sync s3://molerag/\${DS}_chunks/collect_state/state/ \$EXP70_RUN_DIR/state/ --only-show-errors
  aws s3 cp s3://molerag/\${DS}_chunks/qwen_judgments/ \$EXP70_FILTER_BASE/judgments_qwen/ --recursive --only-show-errors
  aws s3 cp s3://molerag/\${DS}_chunks/passages/passages_\${DS}.tar.gz /tmp/p.tgz --only-show-errors && tar xzf /tmp/p.tgz -C \$EXP70_FILTER_BASE/chunks/
  \$PY /opt/exp70/code/query_registry.py || true
  \$PY /opt/exp70/code/medcpt_rerank.py || true
done
# self-terminate
TOKEN=\$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
IID=\$(curl -s -H "X-aws-ec2-metadata-token: \$TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --region $AWS_DEFAULT_REGION --instance-ids \$IID
EOF
)

echo "$UD" > /tmp/medcpt_userdata.sh
echo "launching $ITYPE for MedCPT over: $DSLIST"
aws ec2 run-instances \
  --image-id "$AMI" --instance-type "$ITYPE" --subnet-id "$SUBNET" \
  --iam-instance-profile Name="$PROFILE" \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=300,VolumeType=gp3}' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=exp70-multi-medcpt}]' \
  --user-data file:///tmp/medcpt_userdata.sh \
  --query 'Instances[0].InstanceId' --output text
