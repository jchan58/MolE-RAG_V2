"""DMPNN (trained from scratch, Chemprop v2) on MoleculeNet MolE-RAG splits.
Per task x seed: train from scratch (best checkpoint by validation), predict on test, score
AUC (classification, mean over targets) / RMSE (regression). Resumable (skip-done via result.json),
time-budgeted so it chains across 1hr interactive GPU jobs. Env: cp_dmpnn.
Usage: run_dmpnn.py [--tasks a,b] [--seeds 0,1,2] [--epochs 50] [--minutes 52] [--accelerator gpu]"""
import os,sys,json,subprocess,glob,time,argparse,csv,math
ROOT="/work/hdd/bhrm/jchan5/mole-rag_naacl/dmpnn_baseline"
CP="/u/jchan5/miniconda3/envs/cp_dmpnn/bin/chemprop"
META=json.load(open(f"{ROOT}/data/meta.json"))
def auc(scores,labels):
    pairs=[(s,l) for s,l in zip(scores,labels) if l is not None and s is not None]
    pos=[s for s,l in pairs if l==1]; neg=[s for s,l in pairs if l==0]
    if not pos or not neg: return None
    allp=sorted(pairs,key=lambda x:x[0]); n=len(allp); r=[0.0]*n; i=0
    vals=[s for s,_ in allp]
    while i<n:
        j=i
        while j+1<n and vals[j+1]==vals[i]: j+=1
        a=(i+j)/2+1
        for k in range(i,j+1): r[k]=a
        i=j+1
    sr=sum(r[k] for k in range(n) if allp[k][1]==1); np_=len(pos); nn=len(neg)
    return (sr-np_*(np_+1)/2)/(np_*nn)
def score_task(task,preds_csv):
    m=META[task]; targets=m["targets"]; typ=m["task_type"]
    # test labels (combined.csv rows with split==test, in file order) == predict order (chemprop keeps order)
    test_rows=[r for r in csv.DictReader(open(f"{ROOT}/data/{task}/test.csv"))]
    P=list(csv.DictReader(open(preds_csv)))
    assert len(P)==len(test_rows),f"{task}: preds {len(P)} vs test {len(test_rows)}"
    def col(rowset,name):  # tolerant column fetch
        return name if name in rowset[0] else (name+"_pred" if name+"_pred" in rowset[0] else None)
    per={}
    for t in targets:
        yt=[]; yp=[]
        pcol=col(P,t)
        for tr,pr in zip(test_rows,P):
            lv=tr.get(t,""); pv=pr.get(pcol,"")
            if lv=="" or pv=="" : continue
            yt.append(float(lv)); yp.append(float(pv))
        if not yt: per[t]=None; continue
        if typ=="regression":
            per[t]=math.sqrt(sum((a-b)**2 for a,b in zip(yp,yt))/len(yt))
        else:
            per[t]=auc(yp,[int(round(v)) for v in yt])
    vals=[v for v in per.values() if v is not None]
    agg=sum(vals)/len(vals) if vals else None
    return {"metric":"RMSE" if typ=="regression" else "AUC","value":agg,"per_target":per,"n_targets":len(vals)}
def run_one(task,seed,epochs,accel):
    outdir=f"{ROOT}/runs/{task}/seed{seed}"; os.makedirs(outdir,exist_ok=True)
    res_f=f"{outdir}/result.json"
    if os.path.exists(res_f): return "skip"
    m=META[task]; typ=m["task_type"]; metric="rmse" if typ=="regression" else "roc"
    env=dict(os.environ,OMP_NUM_THREADS="8",OPENBLAS_NUM_THREADS="8",MKL_NUM_THREADS="8",PYTHONNOUSERSITE="1")
    # TRAIN (best model selected on validation split)
    tr=[CP,"train","--data-path",f"{ROOT}/data/{task}/combined.csv","--smiles-columns","smiles",
        "--splits-column","split","--task-type",typ,"--metrics",metric,
        "--output-dir",outdir,"--epochs",str(epochs),"--pytorch-seed",str(seed),
        "--num-workers","0","--save-smiles-splits"]
    if accel=="gpu": tr+=["--accelerator","gpu","--devices","1"]
    else: tr+=["--accelerator","cpu"]
    rc=subprocess.run(tr,env=env,stdout=open(f"{outdir}/train.log","w"),stderr=subprocess.STDOUT).returncode
    if rc!=0: return f"train_fail rc={rc}"
    # find best checkpoint
    cks=glob.glob(f"{outdir}/**/best*.pt",recursive=True)+glob.glob(f"{outdir}/**/best*.ckpt",recursive=True)+glob.glob(f"{outdir}/model_*/*.pt",recursive=True)+glob.glob(f"{outdir}/**/*.ckpt",recursive=True)
    if not cks: return "no_ckpt"
    ckpt=sorted(cks,key=lambda p:("best" not in p,p))[0]
    # PREDICT on test
    preds=f"{outdir}/test_preds.csv"
    pr=[CP,"predict","--test-path",f"{ROOT}/data/{task}/test.csv","--smiles-columns","smiles",
        "--model-path",ckpt,"--preds-path",preds,"--num-workers","0"]
    if accel=="gpu": pr+=["--accelerator","gpu","--devices","1"]
    else: pr+=["--accelerator","cpu"]
    rc=subprocess.run(pr,env=env,stdout=open(f"{outdir}/predict.log","w"),stderr=subprocess.STDOUT).returncode
    if rc!=0 or not os.path.exists(preds): return f"predict_fail rc={rc}"
    sc=score_task(task,preds); sc.update({"task":task,"seed":seed,"ckpt":ckpt})
    json.dump(sc,open(res_f,"w"),indent=2)
    return f"ok {sc['metric']}={sc['value']}"
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tasks",default=""); ap.add_argument("--seeds",default="0,1,2")
    ap.add_argument("--epochs",type=int,default=50); ap.add_argument("--minutes",type=float,default=52)
    ap.add_argument("--accelerator",default="gpu"); a=ap.parse_args()
    tasks=a.tasks.split(",") if a.tasks else list(META)
    seeds=[int(x) for x in a.seeds.split(",")]
    t0=time.time()
    for task in tasks:
        for s in seeds:
            if (time.time()-t0)/60>a.minutes: print("TIME BUDGET; stop",flush=True); return
            print(f"=== {task} seed{s} ===",flush=True)
            r=run_one(task,s,a.epochs,a.accelerator); print(f"  -> {r}",flush=True)
if __name__=="__main__": main()
