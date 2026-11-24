"""SageMaker entry wrapper for the GIN-from-scratch baseline. Runs train_gin.py for a task subset
(TASKS env), reading MolE-RAG splits from the input channel, writing results to SM_MODEL_DIR."""
import os,sys,subprocess
sel=os.environ.get("TASKS","").strip()
allt=["freesolv","esol","lipo","bace","bbbp","clintox","sider","tox21","hiv"]
tasks=[t for t in (sel.split(",") if sel else allt)]
ch=os.environ.get("SM_CHANNEL_TRAINING","/opt/ml/input/data/training")
# channel dir contains seed_0/ directly (SageMaker syncs the S3 prefix CONTENTS); train_gin reads DATA_ROOT/seed_N/...
root=ch
out=os.environ.get("SM_MODEL_DIR","/opt/ml/model")
cmd=[sys.executable,"train_gin.py","--datasets",*tasks,"--seeds","0","1","2",
     "--data-root",root,"--out-dir",out]
print("RUN:"," ".join(cmd),flush=True)
sys.exit(subprocess.run(cmd).returncode)
