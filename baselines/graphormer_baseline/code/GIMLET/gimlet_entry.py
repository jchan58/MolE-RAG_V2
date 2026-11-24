"""SageMaker entry: GIMLET zero-shot on MolE-RAG MoleculeNet test molecules.
Runs downstream_test.py per task (with MOLERAG_TEST_JSON overriding the test split to our seed-0 test).
Results CSV -> SM_MODEL_DIR. One GPU instance; TASKS env can subset."""
import os,subprocess,sys
# apex in the DLC is ABI-built for torch 1.9.1; we pinned 1.9.0 -> uninstall so transformers T5 uses pure LayerNorm
try:
    subprocess.run([sys.executable,"-m","pip","uninstall","-y","apex"],check=False)
except Exception:
    pass
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get("SM_MODEL_DIR","/opt/ml/model"); os.makedirs(OUT,exist_ok=True)
os.environ["MOLERAG_TEST_JSON"]=os.path.join(HERE,"molerag_test_smiles.json")
resfile=os.path.join(OUT,"gimlet_results.csv")
sel=os.environ.get("TASKS","").strip()
allt=["bbbp","bace","clintox","sider","tox21","esol","freesolv","lipo","hiv"]
tasks=[t for t in (sel.split(",") if sel else allt)]
print("TASKS:",tasks,flush=True)
fails=[]
for ds in tasks:
    cmd=[sys.executable,"downstream_test.py","--zero_shot","--transformer_backbone","gimlet",
         "--model_name_or_path","haitengzhao/gimlet","--tokenizer_name","t5-small",
         "--dataset",ds,"--runseed","5","--batch_size","40","--grad_accum_step","1",
         "--transform_in_collator","--only_test","--output_result_to_file",resfile,
         "--not_retest_tasks_in_result_file"]
    print("=== RUN",ds,"===",flush=True)
    rc=subprocess.run(cmd,cwd=HERE).returncode
    print("  ",ds,"rc=",rc,flush=True)
    fails.append((ds,rc))
bad=[d for d,rc in fails if rc!=0]
print("ALL DONE; failed:",bad,flush=True)
import sys as _s
_s.exit(1 if len(bad)==len(tasks) and tasks else 0)
