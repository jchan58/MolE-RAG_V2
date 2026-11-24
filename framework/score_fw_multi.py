"""Score fw_multi runs per dataset x model x arm (A / B_final / C).
regression (esol,freesolv,lipo): RMSE + MAE (lower better). classification (bbbp): ROC-AUC + Brier + acc.
multilabel (sider): mean per-endpoint ROC-AUC. Paired bootstrap C-A (and B-A) on the headline metric.
Missing/unparseable predictions excluded (never imputed). Usage: score_fw_multi.py [ds] [model]"""
import json,glob,sys,math,random,os
from pathlib import Path
_REPO=Path(__file__).resolve().parent.parent
RUNS=Path(os.environ.get('MOLERAG_RUNS', _REPO/'runs'))
random.seed(0)
TT={"esol":"regression","freesolv":"regression","lipo":"regression","bbbp":"classification","hiv":"classification","sider":"multilabel","clintox":"multilabel","tox21":"multilabel"}
def auc(pairs):
    pos=[s for s,l in pairs if l==1]; neg=[s for s,l in pairs if l==0]
    if not pos or not neg: return None
    c=sum((s>t)+0.5*(s==t) for s in pos for t in neg); return c/(len(pos)*len(neg))
import re as _re
def _num(v):
    if v is None: return None
    try: return float(v)
    except: pass
    m=_re.search(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', str(v));
    return float(m.group()) if m else None
def getpred(parsed,tt):
    if not parsed: return None
    if tt=="regression": return _num(parsed.get("prediction_value"))
    if tt=="classification": return _num(parsed.get("probability_positive"))
    e=parsed.get("endpoints"); return e if isinstance(e,dict) else None
def arm_pred(d,arm,tt):
    if arm=="A": return getpred((d.get("A") or {}).get("parsed"),tt)
    if arm=="B": return getpred(d.get("B_final"),tt)
    if arm=="C": return getpred((d.get("C") or {}).get("final"),tt)
def load(ds,model):
    rows=[]
    for f in glob.glob(str(RUNS/f"{ds}__{model}"/"*.json")):
        try: d=json.load(open(f))
        except: continue
        rows.append(d)
    return rows
def score_reg(rows,arm):
    xs=[]
    for d in rows:
        p=arm_pred(d,arm,"regression"); y=d.get("target_SEALED")
        if p is None or y is None: continue
        xs.append((p,float(y)))
    if not xs: return {"n":0}
    rmse=math.sqrt(sum((p-y)**2 for p,y in xs)/len(xs)); mae=sum(abs(p-y) for p,y in xs)/len(xs)
    return {"n":len(xs),"miss":len(rows)-len(xs),"rmse":round(rmse,3),"mae":round(mae,3)}
def score_cls(rows,arm):
    xs=[]
    for d in rows:
        p=arm_pred(d,arm,"classification"); y=d.get("target_SEALED")
        if p is None or y is None: continue
        xs.append((p,int(y)))
    if not xs: return {"n":0}
    a=auc(xs); brier=sum((p-y)**2 for p,y in xs)/len(xs); acc=sum((p>=0.5)==(y==1) for p,y in xs)/len(xs)
    return {"n":len(xs),"miss":len(rows)-len(xs),"auc":round(a,3) if a is not None else None,"brier":round(brier,3),"acc":round(acc,3)}
def score_ml(rows,arm):
    # per-endpoint AUC then mean
    byep={}
    n_used=0
    for d in rows:
        p=arm_pred(d,arm,"multilabel"); y=d.get("target_SEALED")
        if not isinstance(p,dict) or not isinstance(y,dict): continue
        n_used+=1
        for ep,lab in y.items():
            try: pr=float(p.get(ep))
            except: continue
            byep.setdefault(ep,[]).append((pr,int(lab)))
    aucs={ep:auc(v) for ep,v in byep.items()}
    valid=[a for a in aucs.values() if a is not None]
    return {"n":n_used,"miss":len(rows)-n_used,"mean_auc":round(sum(valid)/len(valid),3) if valid else None,"n_endpoints_scored":len(valid)}
def paired_boot(rows,tt,B=2000):
    """paired bootstrap of the headline metric difference C-A and B-A over molecules."""
    def headline(sample,arm):
        if tt=="regression":
            xs=[(arm_pred(d,arm,tt),float(d["target_SEALED"])) for d in sample if arm_pred(d,arm,tt) is not None and d.get("target_SEALED") is not None]
            return math.sqrt(sum((p-y)**2 for p,y in xs)/len(xs)) if xs else None
        if tt=="classification":
            xs=[(arm_pred(d,arm,tt),int(d["target_SEALED"])) for d in sample if arm_pred(d,arm,tt) is not None and d.get("target_SEALED") is not None]
            return auc(xs)
        # multilabel: mean per-endpoint auc
        byep={}
        for d in sample:
            p=arm_pred(d,arm,tt); y=d.get("target_SEALED")
            if not isinstance(p,dict) or not isinstance(y,dict): continue
            for ep,lab in y.items():
                try: pr=float(p.get(ep))
                except: continue
                byep.setdefault(ep,[]).append((pr,int(lab)))
        vals=[auc(v) for v in byep.values() if auc(v) is not None]
        return sum(vals)/len(vals) if vals else None
    both=[d for d in rows if arm_pred(d,"A",tt) is not None and arm_pred(d,"C",tt) is not None]
    if len(both)<5: return None
    idx=list(range(len(both))); dC=[]; dB=[]
    for _ in range(B):
        samp=[both[random.choice(idx)] for _ in idx]
        a=headline(samp,"A"); c=headline(samp,"C"); b=headline(samp,"B")
        if a is not None and c is not None: dC.append(c-a)
        if a is not None and b is not None: dB.append(b-a)
    def ci(x): x=sorted(x); return (round(x[int(.025*len(x))],3),round(x[int(.975*len(x))],3)) if x else None
    a0=headline(both,"A"); c0=headline(both,"C"); b0=headline(both,"B")
    return {"n_paired":len(both),"A":round(a0,3) if a0 else None,"B":round(b0,3) if b0 else None,"C":round(c0,3) if c0 else None,
            "dC_minus_A":round(c0-a0,3) if (a0 and c0) else None,"dC_CI":ci(dC),
            "dB_minus_A":round(b0-a0,3) if (a0 and b0) else None,"dB_CI":ci(dB)}

DSS=[sys.argv[1]] if len(sys.argv)>1 else ["bbbp","esol","freesolv","lipo","sider"]
MODS=[sys.argv[2]] if len(sys.argv)>2 else ["ministral_3_14b","qwen3_32b","opus_4_6","gpt5mini"]
report={}
for ds in DSS:
    tt=TT[ds]; report[ds]={}
    print(f"\n=== {ds} ({tt}) ===")
    hm={"regression":"rmse","classification":"auc","multilabel":"mean_auc"}[tt]
    for m in MODS:
        rows=load(ds,m)
        if not rows: print(f"  {m:16} (no parts)"); continue
        sc={"A":None,"B":None,"C":None}
        for arm in ("A","B","C"):
            sc[arm]= {"regression":score_reg,"classification":score_cls,"multilabel":score_ml}[tt](rows,arm)
        pb=paired_boot(rows,tt)
        report[ds][m]={"n_parts":len(rows),"A":sc["A"],"B":sc["B"],"C":sc["C"],"paired":pb}
        line=f"  {m:16} n={len(rows):4} | A {hm}={sc['A'].get(hm)} | B {hm}={sc['B'].get(hm)} | C {hm}={sc['C'].get(hm)}"
        if pb: line+=f" | C-A={pb['dC_minus_A']} CI{pb['dC_CI']}"
        print(line)
out=_REPO/'scores'/"SCORES_fw_multi.json"; out.parent.mkdir(parents=True,exist_ok=True)
json.dump(report,open(out,"w"),indent=1); print("\nsaved",out)
