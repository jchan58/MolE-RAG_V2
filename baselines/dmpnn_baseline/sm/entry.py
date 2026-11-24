"""SageMaker entry point: DMPNN (Chemprop v2) trained from scratch on MoleculeNet MolE-RAG splits.
Runs all (task x seed) sequentially on the instance GPU. Data mounted at SM_CHANNEL_TRAIN; results
(per-run result.json + summary.json) written to SM_MODEL_DIR (-> uploaded to S3 by SageMaker)."""
import os,sys,json,subprocess,glob,time,csv,math,shutil
DATA=os.environ.get("SM_CHANNEL_TRAIN","/opt/ml/input/data/train")
OUTD=os.environ.get("SM_MODEL_DIR","/opt/ml/model")
SEEDS=[int(x) for x in os.environ.get("SEEDS","0,1,2").split(",")]
EPOCHS=int(os.environ.get("EPOCHS","50"))
META=json.load(open(f"{DATA}/meta.json"))
CP=shutil.which("chemprop") or "chemprop"
def auc(scores,labels):
    pairs=[(s,l) for s,l in zip(scores,labels) if l is not None and s is not None]
    pos=[s for s,l in pairs if l==1]; neg=[s for s,l in pairs if l==0]
    if not pos or not neg: return None
    allp=sorted(pairs,key=lambda x:x[0]); n=len(allp); r=[0.0]*n; i=0; vals=[s for s,_ in allp]
    while i<n:
        j=i
        while j+1<n and vals[j+1]==vals[i]: j+=1
        a=(i+j)/2+1
        for k in range(i,j+1): r[k]=a
        i=j+1
    sr=sum(r[k] for k in range(n) if allp[k][1]==1); return (sr-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def score_task(task,preds_csv):
    m=META[task]; targets=m["targets"]; typ=m["task_type"]
    test_rows=list(csv.DictReader(open(f"{DATA}/{task}/test.csv")))
    P=list(csv.DictReader(open(preds_csv)))
    if len(P)!=len(test_rows): raise RuntimeError(f"{task}: preds {len(P)} vs test {len(test_rows)}")
    def pcol(name): return name if name in P[0] else (name+"_pred" if P and name+"_pred" in P[0] else None)
    per={}
    for t in targets:
        c=pcol(t); yt=[]; yp=[]
        for tr,pr in zip(test_rows,P):
            lv=tr.get(t,""); pv=pr.get(c,"") if c else ""
            if lv==""or pv=="": continue
            yt.append(float(lv)); yp.append(float(pv))
        if not yt: per[t]=None; continue
        per[t]=math.sqrt(sum((a-b)**2 for a,b in zip(yp,yt))/len(yt)) if typ=="regression" else auc(yp,[int(round(v)) for v in yt])
    vals=[v for v in per.values() if v is not None]
    return {"metric":"RMSE" if typ=="regression" else "AUC","value":(sum(vals)/len(vals) if vals else None),"per_target":per,"n_scored_targets":len(vals)}
def run_one(task,seed):
    outdir=f"{OUTD}/runs/{task}/seed{seed}"; os.makedirs(outdir,exist_ok=True)
    m=META[task]; typ=m["task_type"]; metric="rmse" if typ=="regression" else "roc"
    tr=[CP,"train","--data-path",f"{DATA}/{task}/combined.csv","--smiles-columns","smiles",
        "--splits-column","split","--task-type",typ,"--metrics",metric,"--output-dir",outdir,
        "--epochs",str(EPOCHS),"--pytorch-seed",str(seed),"--num-workers","0",
        "--accelerator","gpu","--devices","1"]
    rc=subprocess.run(tr,stdout=open(f"{outdir}/train.log","w"),stderr=subprocess.STDOUT).returncode
    if rc!=0: return {"status":f"train_fail rc={rc}"}
    cks=glob.glob(f"{outdir}/**/best*.pt",recursive=True)+glob.glob(f"{outdir}/**/best*.ckpt",recursive=True)+glob.glob(f"{outdir}/**/*.ckpt",recursive=True)+glob.glob(f"{outdir}/**/*.pt",recursive=True)
    if not cks: return {"status":"no_ckpt"}
    ckpt=sorted(cks,key=lambda p:("best" not in os.path.basename(p),p))[0]
    preds=f"{outdir}/test_preds.csv"
    pr=[CP,"predict","--test-path",f"{DATA}/{task}/test.csv","--smiles-columns","smiles",
        "--model-path",ckpt,"--preds-path",preds,"--num-workers","0","--accelerator","gpu","--devices","1"]
    rc=subprocess.run(pr,stdout=open(f"{outdir}/predict.log","w"),stderr=subprocess.STDOUT).returncode
    if rc!=0 or not os.path.exists(preds): return {"status":f"predict_fail rc={rc}"}
    sc=score_task(task,preds); sc.update({"task":task,"seed":seed,"status":"ok","ckpt":os.path.basename(ckpt)})
    json.dump(sc,open(f"{outdir}/result.json","w"),indent=2)
    # keep model dir small: drop heavy checkpoints from the uploaded artifact
    for f in glob.glob(f"{outdir}/**/*.ckpt",recursive=True)+glob.glob(f"{outdir}/**/*.pt",recursive=True):
        try: os.remove(f)
        except: pass
    return sc
def main():
    t0=time.time(); allres=[]
    sel=os.environ.get("TASKS","").strip()
    tasks=[t for t in META if (not sel or t in sel.split(","))]
    print("TASKS:",tasks,flush=True)
    for task in tasks:
        for s in SEEDS:
            print(f"=== {task} seed{s} ===",flush=True)
            r=run_one(task,s); allres.append(r)
            print(f"  -> {r.get('status')} {r.get('metric')}={r.get('value')} ({(time.time()-t0)/60:.1f}min)",flush=True)
    json.dump(allres,open(f"{OUTD}/summary.json","w"),indent=2)
    print("ALL DONE",flush=True)
if __name__=="__main__": main()
