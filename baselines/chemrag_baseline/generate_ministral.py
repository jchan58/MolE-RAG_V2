"""ChemRAG framework on 9 MoleculeNet test sets with Ministral-14B (Bedrock Converse).
Reuses the top-5 RRF(BM25+Contriever) docs (retrieval/fused_docs.jsonl). FORCED NUMERIC outputs:
 - classification: model outputs P(positive) in [0,1] (Bedrock gives no logprobs) -> AUC
 - regression: model outputs a numeric estimate, must NOT abstain -> RMSE
Parallel dispatch + backoff, skip-done. Outputs -> outputs_ministral/. Run on login node.
Phases: --generate [--tasks a,b] --score"""
import os,sys,json,re,argparse,time,urllib.request,urllib.error,random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent
QD=HERE/"queries"; RET=HERE/"retrieval"; OUT=HERE/"outputs_ministral"; OUT.mkdir(exist_ok=True)
(HERE/"analysis").mkdir(exist_ok=True)
MODEL="mistral.ministral-3-14b-instruct"
EP=f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{MODEL}/converse"
CLS=["bbbp","bace","hiv","clintox","sider","tox21"]; REG=["esol","freesolv","lipo"]
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
SYS="You are an expert medicinal chemist. Use the documents and your knowledge. Always commit to a numeric answer."
def tok():
    for l in (Path.home()/'.config/molrag/bedrock.env').read_text().splitlines():
        if l.startswith('AWS_BEARER_TOKEN_BEDROCK='): return l.split('=',1)[1].strip()
    raise RuntimeError("no token")
def call(prompt,maxtok=48):
    body={"system":[{"text":SYS}],"messages":[{"role":"user","content":[{"text":prompt}]}],
          "inferenceConfig":{"temperature":0,"maxTokens":maxtok}}
    last=None
    for a in range(8):
        try:
            req=urllib.request.Request(EP,data=json.dumps(body).encode(),method="POST",
                headers={"Authorization":"Bearer "+tok(),"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=90) as r: o=json.loads(r.read())
            return "".join(c.get("text","") for c in o["output"]["message"]["content"]).strip()
        except urllib.error.HTTPError as e:
            last=e
            if e.code in (429,500,502,503,529): time.sleep(min(40,1.5*(2**a))+random.uniform(0,1.5)); continue
            raise
        except (urllib.error.URLError,TimeoutError) as e:
            last=e; time.sleep(min(40,1.5*(2**a))+random.uniform(0,1.5)); continue
    raise last
def reference_block(docs):
    out=[]
    for j,d in enumerate(docs,1):
        out.append(f"[Document {j}] "+" ".join((d.get('contents') or '').split()[:400]))
    return "\n".join(out) if out else "(no documents retrieved)"
def cls_prompt(ref,instr):
    return ("Answer the question based on the given documents and your chemical knowledge.\n"
            "The following are given documents.\n"+ref+"\nQuestion: "+instr+"\n"
            "Respond with ONLY a single number between 0 and 1 giving the probability that the answer is yes "
            "(1 = certainly yes, 0 = certainly no). Output only the number, nothing else.\nAnswer:")
def reg_prompt(ref,instr):
    return ("Answer the question based on the given documents and your chemical knowledge.\n"
            "The following are given documents.\n"+ref+"\nQuestion: "+instr+"\n"
            "You MUST give your single best numerical estimate; never answer 'unknown'. "
            "Respond with ONLY the numerical value as a single number (for example: -3.2). Output only the number, nothing else.\nAnswer:")
NUM=re.compile(r'(-?\d+(?:\.\d+)?)')
def parse_prob(t):
    m=NUM.findall(t)
    if not m: return None
    v=float(m[0]); return max(0.0,min(1.0,v))
def parse_val(t):
    m=re.search(r'therefore\s*\[?\s*(-?\d+(?:\.\d+)?)',t,re.I)
    if m: return float(m.group(1))
    m=NUM.findall(t); return float(m[0]) if m else None  # terse output leads with the value
def load_docs():
    d={}
    for l in open(RET/"fused_docs.jsonl"): r=json.loads(l); d[r["topic_id"]]=r["docs"]
    return d
def one(args):
    task,q,docs,is_cls=args
    ref=reference_block(docs.get(q["qid"],[])); instr=QUESTION[task].format(smi=q["smiles"])
    if is_cls:
        txt=call(cls_prompt(ref,instr),48); v=parse_prob(txt)
        return {"topic_id":q["qid"],"task":task,"label":float(q["label"]),"p_pos":v,"text":txt[:120]}
    else:
        txt=call(reg_prompt(ref,instr),64); v=parse_val(txt)
        return {"topic_id":q["qid"],"task":task,"label":float(q["label"]),"pred":v,"text":txt[:160]}
def generate(tasks,workers=16):
    docs=load_docs()
    for task in tasks:
        is_cls=task in CLS; outp=OUT/f"{task}_preds.jsonl"
        done=set(json.loads(l)["topic_id"] for l in open(outp)) if outp.exists() else set()
        qs=[json.loads(l) for l in open(QD/f"{task}_queries.jsonl")]
        qs=[q for q in qs if q["qid"] not in done]
        if not qs: print(f"{task}: all done",flush=True); continue
        print(f"{task}: {len(qs)} to do ({'cls' if is_cls else 'reg'})",flush=True)
        f=open(outp,"a"); ok=err=0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs=[ex.submit(one,(task,q,docs,is_cls)) for q in qs]
            for i,fut in enumerate(as_completed(futs),1):
                try: rec=fut.result(); f.write(json.dumps(rec)+"\n"); f.flush(); ok+=1
                except Exception as e:
                    err+=1
                    if err<=3: print("  ERR",str(e)[:80],flush=True)
                if i%500==0: print(f"  {task} {i}/{len(qs)} ok={ok} err={err}",flush=True)
        f.close(); print(f"{task} done ok={ok} err={err}",flush=True)
def auc(sc,lab):
    pos=sum(lab); neg=len(lab)-pos
    if not pos or not neg: return None
    order=sorted(range(len(sc)),key=lambda i:sc[i]); r=[0.0]*len(sc); i=0
    while i<len(order):
        j=i
        while j+1<len(order) and sc[order[j+1]]==sc[order[i]]: j+=1
        a=(i+j)/2+1
        for k in range(i,j+1): r[order[k]]=a
        i=j+1
    return (sum(r[i] for i in range(len(sc)) if lab[i]==1)-pos*(pos+1)/2)/(pos*neg)
def score():
    res={}
    for task in CLS:
        p=OUT/f"{task}_preds.jsonl"
        if not p.exists(): continue
        rows=[json.loads(l) for l in open(p)]; ok=[r for r in rows if r.get("p_pos") is not None]
        sc=[r["p_pos"] for r in ok]; lab=[int(r["label"]) for r in ok]
        a=auc(sc,lab)
        res[task]={"type":"classification","n":len(rows),"scored":len(ok),"AUC":round(a,4) if a else None,"pos_rate":round(sum(lab)/len(lab),3) if lab else None}
    for task in REG:
        p=OUT/f"{task}_preds.jsonl"
        if not p.exists(): continue
        rows=[json.loads(l) for l in open(p)]; ok=[(r["pred"],r["label"]) for r in rows if r.get("pred") is not None]
        abstain=sum(1 for r in rows if 'unknown' in (r.get('text','').lower()))
        if ok:
            rmse=(sum((a-b)**2 for a,b in ok)/len(ok))**0.5; mae=sum(abs(a-b) for a,b in ok)/len(ok)
        else: rmse=mae=None
        res[task]={"type":"regression","n":len(rows),"numeric":len(ok),"abstain":abstain,"RMSE":round(rmse,4) if rmse else None,"MAE":round(mae,4) if mae else None}
    json.dump(res,open(HERE/"analysis/results_ministral.json","w"),indent=2)
    print(json.dumps(res,indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--generate",action="store_true"); ap.add_argument("--score",action="store_true")
    ap.add_argument("--tasks",default=""); ap.add_argument("--workers",type=int,default=16); a=ap.parse_args()
    if a.generate:
        tasks=a.tasks.split(",") if a.tasks else (REG+CLS)
        generate(tasks,a.workers)
    if a.score: score()
