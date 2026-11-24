"""SageMaker entry: finish GIMLET 9/9 by running clintox + sider on MolE-RAG test molecules.

GIMLET's released prompt JSON files have NO clintox/sider entries, so we extract the authentic
per-subtask instruction templates from the HF dataset haitengzhao/molecule_property_instruction
(the dataset GIMLET itself built, keyed by task_index == subtask index), inject them into
prompts/selected_prompt_downstream_task.json as {task: {"0":[...], "1":[...], ...}}, then run the
SAME scaffold-split branch used for the other 7 tasks with MOLERAG_TEST_JSON overriding the test
split to our seed-0 test molecules. One prompt (text[0]) per subtask -> matches the "selected"
single-prompt semantics and bounds runtime (sider = 27 subtasks)."""
import os,subprocess,sys,json

# apex in the DLC is ABI-built for torch 1.9.1; we pinned 1.9.0 -> uninstall so T5 uses pure LayerNorm
try:
    subprocess.run([sys.executable,"-m","pip","uninstall","-y","apex"],check=False)
except Exception:
    pass

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get("SM_MODEL_DIR","/opt/ml/model"); os.makedirs(OUT,exist_ok=True)
os.environ["MOLERAG_TEST_JSON"]=os.path.join(HERE,"molerag_test_smiles.json")
os.environ.setdefault("HF_HOME","/tmp/hf")

TASKS=[t for t in os.environ.get("TASKS","clintox,sider").split(",") if t.strip()]
print("TASKS:",TASKS,flush=True)

# ---- 1) Extract authentic per-subtask prompts from the HF instruction dataset ----
from datasets import load_dataset
dd=load_dataset("haitengzhao/molecule_property_instruction")
print("HF dataset splits:",list(dd.keys()),flush=True)

prompt_path=os.path.join(HERE,"prompts","selected_prompt_downstream_task.json")
import commentjson
with open(prompt_path) as f:
    allprompts=commentjson.load(f)

for ds in TASKS:
    if ds not in dd:
        print("[WARN] %s not a split key in HF dataset; skipping extraction"%ds,flush=True)
        continue
    sub=dd[ds]
    # one representative row per task_index is enough: the instruction text does not depend on the molecule.
    # to_pandas + drop_duplicates is far faster than row-by-row iteration over the full split.
    df=sub.to_pandas()[["task_index","text"]].drop_duplicates("task_index")
    per={}  # task_index(str) -> prompt string (text[0])
    for _,row in df.iterrows():
        ti=str(row["task_index"])
        txt=row["text"]
        if isinstance(txt,(list,tuple)) or hasattr(txt,"__len__") and not isinstance(txt,str):
            txt=list(txt)[0]
        per[ti]=txt
    # build {task: {"0":[prompt], "1":[prompt], ...}} in subtask-index order
    n=max(int(k) for k in per)+1
    entry={str(i):[per[str(i)]] for i in range(n) if str(i) in per}
    allprompts[ds]=entry
    print("[PROMPTS] %s: %d subtasks extracted"%(ds,len(entry)),flush=True)
    print("   e.g. subtask 0 ->",entry.get("0"),flush=True)

with open(prompt_path,"w") as f:
    json.dump(allprompts,f,indent=1)
print("[PROMPTS] merged clintox/sider into",prompt_path,flush=True)

# ---- 2) Run the scaffold-split branch (with MolE-RAG test override) for each task ----
resfile=os.path.join(OUT,"gimlet_results.csv")
fails=[]
for ds in TASKS:
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
sys.exit(1 if len(bad)==len(TASKS) and TASKS else 0)
