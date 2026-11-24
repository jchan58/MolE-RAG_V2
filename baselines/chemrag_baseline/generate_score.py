"""ChemRAG framework prediction on 9 MoleculeNet test sets with Qwen3-4B-Instruct-2507.
Uses ChemRAG's RAG prompt templates (Table 17 multi-choice for classification, Table 18 numerical for regression)
with top-5 RRF(BM25+Contriever) retrieved documents prepended.
Classification: P(positive) from forced-choice answer-token logprobs (A=Yes vs B=No) -> AUC.
Regression: parse the numerical answer -> RMSE.
Run from MAIN env (torch 2.12, GPU). Phases: --classify --regress --score"""
import os,sys,json,re,argparse,math,time
from pathlib import Path
T0=time.time(); BUDGET_MIN=52.0
def over_budget(): return (time.time()-T0)/60.0>BUDGET_MIN
def load_done(outp):
    d=set()
    if outp.exists():
        for l in open(outp):
            try: d.add(json.loads(l)["topic_id"])
            except Exception: pass   # tolerate a trailing partial line from a hard kill
    return d
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
HERE=Path(__file__).resolve().parent
MODEL="/work/hdd/bhrm/jchan5/models/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
QD=HERE/"queries"; RET=HERE/"retrieval"; OUT=HERE/"outputs"; OUT.mkdir(exist_ok=True)
CLS=["bbbp","bace","hiv","clintox","sider","tox21"]; REG=["esol","freesolv","lipo"]
# per-task question (Instruction) + positive-class meaning
QUESTION={
 "bbbp":"Does the molecule with SMILES {smi} penetrate the blood-brain barrier?",
 "bace":"Does the molecule with SMILES {smi} inhibit human beta-secretase 1 (BACE-1)?",
 "hiv":"Is the molecule with SMILES {smi} active at inhibiting HIV replication?",
 "clintox":"Is the molecule with SMILES {smi} clinically toxic (did it fail clinical trials for toxicity)?",
 "sider":"Is the molecule with SMILES {smi} associated with adverse drug reactions (side effects)?",
 "tox21":"Is the molecule with SMILES {smi} toxic (active in the Tox21 toxicity assays)?",
 "esol":"What is the aqueous solubility (log solubility, logS, in mol/L) of the molecule with SMILES {smi}?",
 "freesolv":"What is the hydration free energy (in kcal/mol) of the molecule with SMILES {smi}?",
 "lipo":"What is the lipophilicity (octanol-water distribution coefficient logD at pH 7.4) of the molecule with SMILES {smi}?",
}
def reference_block(docs):
    out=[]
    for j,d in enumerate(docs,1):
        txt=" ".join((d.get("contents") or "").split()[:400])  # cap each doc ~400 words
        out.append(f"[Document {j}] {txt}")
    return "\n".join(out) if out else "(no documents retrieved)"
def cls_prompt(ref,instr):
    return ("Answer the question based on the given document.\nOnly give me the answer and do not output any other words.\n"
            "The following are given documents.\n"+ref+"\nQuestion: "+instr+"\nChoices: A. Yes\nB. No\nMake prediction from the given choices.\nAnswer:")
def reg_prompt(ref,instr):
    return ("Answer the question based on the given document.\nConclude the answer by stating \"The answer is therefore [ANSWER]\"\n"
            "Only give me the answer and do not output any other words.\nThe following are given documents.\n"+ref+"\nQuestion: "+instr+"\nAnswer:")
def load_docs():
    d={}
    for l in open(RET/"fused_docs.jsonl"): r=json.loads(l); d[r["topic_id"]]=r["docs"]
    return d
def load_queries(task):
    return [json.loads(l) for l in open(QD/f"{task}_queries.jsonl")]
def build_model():
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="cuda").eval()
    return tok,model
def ids_for(tok,words):
    s=set()
    for w in words:
        for v in (w," "+w):
            t=tok.encode(v,add_special_tokens=False)
            if t: s.add(t[0])
    return sorted(s)
def classify(task,tok,model,docs,BATCH=16):
    outp=OUT/f"{task}_preds.jsonl"
    done=load_done(outp)
    qs=[q for q in load_queries(task) if q["qid"] not in done]
    if not qs: print(f"{task}: all done"); return
    A=ids_for(tok,["A","Yes","yes"]); B=ids_for(tok,["B","No","no"])
    prompts=[]
    for q in qs:
        ref=reference_block(docs.get(q["qid"],[])); instr=QUESTION[task].format(smi=q["smiles"])
        msgs=[{"role":"user","content":cls_prompt(ref,instr)}]
        prompts.append(tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True))
    f=open(outp,"a")
    for i in range(0,len(prompts),BATCH):
        if over_budget(): print(f"  {task}: time budget reached, exiting (resumable)",flush=True); f.close(); return
        enc=tok(prompts[i:i+BATCH],return_tensors="pt",padding=True,truncation=True,max_length=4096).to("cuda")
        with torch.no_grad(): lg=model(**enc).logits[:,-1,:].float()
        lp=torch.log_softmax(lg,dim=-1)
        pa=torch.logsumexp(lp[:,A],dim=-1); pb=torch.logsumexp(lp[:,B],dim=-1)
        ppos=torch.sigmoid(pa-pb).cpu().tolist()   # P(A)/(P(A)+P(B))
        for q,p in zip(qs[i:i+BATCH],ppos):
            f.write(json.dumps({"topic_id":q["qid"],"task":task,"label":float(q["label"]),"p_pos":p})+"\n")
        f.flush(); print(f"  {task} {min(i+BATCH,len(prompts))}/{len(prompts)}",flush=True)
    f.close()
NUM=re.compile(r'(-?\d+(?:\.\d+)?)')
def parse_num(txt):
    m=re.search(r'answer is therefore\s*\[?\s*(-?\d+(?:\.\d+)?)',txt,re.I)
    if m: return float(m.group(1))
    m=NUM.findall(txt); return float(m[-1]) if m else None
def regress(task,tok,model,docs,BATCH=16):
    outp=OUT/f"{task}_preds.jsonl"
    done=load_done(outp)
    qs=[q for q in load_queries(task) if q["qid"] not in done]
    if not qs: print(f"{task}: all done"); return
    prompts=[]
    for q in qs:
        ref=reference_block(docs.get(q["qid"],[])); instr=QUESTION[task].format(smi=q["smiles"])
        msgs=[{"role":"user","content":reg_prompt(ref,instr)}]
        prompts.append(tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True))
    f=open(outp,"a")
    for i in range(0,len(prompts),BATCH):
        if over_budget(): print(f"  {task}: time budget reached, exiting (resumable)",flush=True); f.close(); return
        enc=tok(prompts[i:i+BATCH],return_tensors="pt",padding=True,truncation=True,max_length=4096).to("cuda")
        with torch.no_grad():
            g=model.generate(**enc,max_new_tokens=64,do_sample=False,pad_token_id=tok.pad_token_id)
        txts=tok.batch_decode(g[:,enc["input_ids"].shape[1]:],skip_special_tokens=True)
        for q,t in zip(qs[i:i+BATCH],txts):
            f.write(json.dumps({"topic_id":q["qid"],"task":task,"label":float(q["label"]),"pred":parse_num(t),"text":t.strip()[:200]})+"\n")
        f.flush(); print(f"  {task} {min(i+BATCH,len(prompts))}/{len(prompts)}",flush=True)
    f.close()
def auc(scores,labels):
    pairs=sorted(zip(scores,labels)); pos=sum(labels); neg=len(labels)-pos
    if pos==0 or neg==0: return None
    # rank-based with tie handling
    order=sorted(range(len(scores)),key=lambda i:scores[i]); r=[0.0]*len(scores); i=0
    while i<len(order):
        j=i
        while j+1<len(order) and scores[order[j+1]]==scores[order[i]]: j+=1
        avg=(i+j)/2+1
        for k in range(i,j+1): r[order[k]]=avg
        i=j+1
    sr=sum(r[i] for i in range(len(scores)) if labels[i]==1)
    return (sr-pos*(pos+1)/2)/(pos*neg)
def score():
    res={}
    for task in CLS:
        p=OUT/f"{task}_preds.jsonl"
        if not p.exists(): continue
        rows=[json.loads(l) for l in open(p)]
        sc=[r["p_pos"] for r in rows]; lab=[int(r["label"]) for r in rows]
        a=auc(sc,lab)
        res[task]={"type":"classification","n":len(rows),"AUC":round(a,4) if a else None,"pos_rate":round(sum(lab)/len(lab),3)}
    for task in REG:
        p=OUT/f"{task}_preds.jsonl"
        if not p.exists(): continue
        rows=[json.loads(l) for l in open(p)]
        ok=[(r["pred"],r["label"]) for r in rows if r["pred"] is not None]
        if ok:
            rmse=(sum((pr-la)**2 for pr,la in ok)/len(ok))**0.5
            mae=sum(abs(pr-la) for pr,la in ok)/len(ok)
        else: rmse=mae=None
        res[task]={"type":"regression","n":len(rows),"parsed":len(ok),"RMSE":round(rmse,4) if rmse else None,"MAE":round(mae,4) if mae else None}
    json.dump(res,open(HERE/"analysis/results.json","w"),indent=2)
    print(json.dumps(res,indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--classify",action="store_true"); ap.add_argument("--regress",action="store_true"); ap.add_argument("--score",action="store_true")
    ap.add_argument("--tasks",default=""); a=ap.parse_args()
    if a.classify or a.regress:
        docs=load_docs(); tok,model=build_model()
        sel=a.tasks.split(",") if a.tasks else None
        if a.classify:
            for t in CLS:
                if not sel or t in sel: classify(t,tok,model,docs)
        if a.regress:
            for t in REG:
                if not sel or t in sel: regress(t,tok,model,docs)
    if a.score: score()
