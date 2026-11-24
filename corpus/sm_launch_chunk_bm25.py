"""
Submit a SageMaker Processing job that does one dataset's heavy download+chunk+BM25 on a big
CPU box, off S3 (entrypoint = sm_entry.sh). Managed + auto-terminating + native S3 IO +
SageMakerExecutionRole (S3FullAccess only, no Bedrock) -> NO credential transfer. The Bedrock
judge runs separately on Delta. Prints the job name on the last line.
Usage: python sm_launch_chunk_bm25.py <dataset> [c6i.24xlarge] [volume_gb=1000]
"""
import sys, time, boto3

DS = sys.argv[1] if len(sys.argv) > 1 else "hiv"
ITYPE = "ml." + (sys.argv[2] if len(sys.argv) > 2 else "c6i.24xlarge")   # 96 vCPU; quota=2/type
VOL = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
REGION = "us-east-1"
IMG = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1"
ROLE = "arn:aws:iam::833984135295:role/SageMakerExecutionRole"   # S3FullAccess (smoke-verified)

sm = boto3.client("sagemaker", region_name=REGION)
JOB = "exp70-%s-chunkbm25-%d" % (DS, int(time.time()))
sm.create_processing_job(
    ProcessingJobName=JOB,
    RoleArn=ROLE,
    AppSpecification={"ImageUri": IMG,
                      "ContainerEntrypoint": ["bash", "/opt/ml/processing/input/code/sm_entry.sh", DS]},
    ProcessingInputs=[{"InputName": "code", "S3Input": {
        "S3Uri": "s3://molerag/exp70_sm/chunkbm25/code/",
        "LocalPath": "/opt/ml/processing/input/code",
        "S3DataType": "S3Prefix", "S3InputMode": "File"}}],
    ProcessingOutputConfig={"Outputs": [{"OutputName": "out", "S3Output": {
        "S3Uri": "s3://molerag/%s_chunks/sm_out/" % DS,
        "LocalPath": "/opt/ml/processing/output", "S3UploadMode": "EndOfJob"}}]},
    ProcessingResources={"ClusterConfig": {"InstanceType": ITYPE, "InstanceCount": 1,
                                           "VolumeSizeInGB": VOL}},
    StoppingCondition={"MaxRuntimeInSeconds": 21600},   # 6h cap
    Tags=[{"Key": "exp70", "Value": DS}],
)
print(JOB)
