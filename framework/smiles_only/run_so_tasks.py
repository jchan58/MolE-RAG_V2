import csv,json,re,time,urllib.request,urllib.error,os,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
_HERE=Path(__file__).resolve().parent; _REPO=_HERE.parent.parent
sys.path.insert(0,str(_HERE.parent))  # framework/ for model_client
try:
    from dotenv import load_dotenv; load_dotenv(_REPO/'.env')
except Exception: pass
from model_client import load_models, chat as _chat
DATA_DIR=Path(os.environ.get('MOLERAG_DATA', _REPO/'data'/'moleculenet'/'seed_0'))
BASE=Path(os.environ.get('MOLERAG_RUNS', _REPO/'runs'))/'smiles_only'/'smilesonly_multitask'; BASE.mkdir(parents=True,exist_ok=True)
ROOT=str(DATA_DIR)
MODEL=sys.argv[1]; W=int(sys.argv[2]) if len(sys.argv)>2 else 4
TASKS=sys.argv[3].split(',') if len(sys.argv)>3 else ['freesolv','esol','bbbp','lipo','sider','hiv','tox21']
_MODELS=load_models(_REPO); MCFG=_MODELS[MODEL]
def call(prompt,mt=1200): return _chat(prompt,MCFG,mt)
def xj(raw):
    t=re.sub(r'<think>.*?</think>','',raw or '',flags=re.S).replace('```json','').replace('```','');m=re.search(r'\{.*\}',t,re.S)
    if not m:return None,'malformed'
    for c in (m.group(0),re.sub(r',(\s*[}\]])',r'\1',m.group(0))):
        try:return json.loads(c),'clean'
        except:pass
    return None,'malformed'
ENDPOINT={'p_np':'penetrates the blood-brain barrier (BBB+)','HIV_active':'is active at inhibiting HIV replication','CT_TOX':'failed clinical trials for toxicity reasons (clinically toxic)','FDA_APPROVED':'is an FDA-approved drug'}
def cls_prompt(smi,task,endpoint_desc):
    return (f"You are predicting a molecular property from SMILES only.\nTASK: whether the molecule tests POSITIVE for: {endpoint_desc}.\n"
            f"SMILES: {smi}\n\nactivity_score = probability the molecule is POSITIVE for this endpoint (0-1).\n"
            'Output ONLY JSON: {{"prediction":"positive|negative|uncertain","activity_score":0.0}}'.replace('{{','{').replace('}}','}'))
def reg_prompt(smi,prop):
    return (f"You are predicting a molecular property from SMILES only.\nPROPERTY: {prop}.\nSMILES: {smi}\n\n"
            'Output ONLY JSON: {"value": <number>}')
REGDESC={'logSolubility':'aqueous solubility logS (log mol/L, ESOL)','lipo':'lipophilicity logD at pH 7.4 (ChEMBL)','freesolv':'hydration free energy (kcal/mol, FreeSolv)'}
def endpoint_desc(task,tt,taskcol):
    if tt=='regression': return None
    if taskcol in ENDPOINT: return ENDPOINT[taskcol]
    if task=='tox21': return f"toxicity activity in the Tox21 '{taskcol}' assay"
    if task=='sider': return f"causes side effects in the category '{taskcol.replace('_',' ')}'"
    return f"the '{taskcol}' positive class"
tasks_done=0
for task in TASKS:
    rows=list(csv.DictReader(open(f"{ROOT}/{task}/{task}_test.csv")))
    tt=rows[0]['task_type']; od=BASE/f"{MODEL}__{task}"; od.mkdir(exist_ok=True)
    todo=[(i,r) for i,r in enumerate(rows) if not (od/f"{i:04d}.json").exists()]
    print(f"{MODEL} {task} ({tt}): {len(todo)} todo of {len(rows)}",flush=True)
    def work(ir):
        i,r=ir; smi=r['smiles']; tc=r.get('task','')
        if tt=='regression': p=reg_prompt(smi,REGDESC.get(tc,tc))
        else: p=cls_prompt(smi,task,endpoint_desc(task,tt,tc))
        raw,cs=call(p); j,st=xj(raw)
        json.dump({'task':task,'endpoint':tc,'idx':i,'tt':tt,'label':r['label'],'raw':raw,'parsed':j,'status':st,'call':cs},open(od/f"{i:04d}.json",'w'))
    with ThreadPoolExecutor(max_workers=W) as ex: list(ex.map(work,todo))
    print(f"{MODEL} {task} DONE",flush=True)
print(f"{MODEL} ALL TASKS DONE",flush=True)
