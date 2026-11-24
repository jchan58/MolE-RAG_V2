import json,glob,math,collections,os
MT='smilesonly_multitask'; MODELS=['qwen3_32b','ministral_3_14b','gpt5mini','opus_4_6']
CLS9={'bace','bbbp','hiv','sider','tox21','clintox'}; REG={'esol','lipo','freesolv'}
MULTI={'tox21','sider','clintox'}  # per-endpoint mean
def auc(p):
    pos=[s for s,l in p if l==1];neg=[s for s,l in p if l==0]
    return (sum((s>t)+0.5*(s==t) for s in pos for t in neg)/(len(pos)*len(neg))) if pos and neg else None
def read_dir(d,sfx=''):
    out=[]
    for f in glob.glob(d+'/*.json'):
        e=json.load(open(f))
        if sfx and not os.path.basename(f).startswith(sfx): continue
        out.append(e)
    return out
def score(model,task):
    # bace lives in smilesonly_<model>/test_*.json ; others in smilesonly_multitask/<model>__<task>
    if task=='bace':
        rows=read_dir(f'smilesonly_{model.replace("3_32b","").replace("_3_14b","").replace("_4_6","") if False else model if os.path.isdir("smilesonly_"+model) else model}','test_')
        d=f'smilesonly_{"qwen" if model=="qwen3_32b" else model}'
        rows=[json.load(open(f)) for f in glob.glob(d+'/test_*.json')]
    else:
        d=f'{MT}/{model}__{task}'
        if not os.path.isdir(d): return None
        rows=[json.load(open(f)) for f in glob.glob(d+'/*.json')]
    if not rows: return None
    clean=[r for r in rows if r.get('status')=='clean' and (r.get('parsed') or {})]
    miss=len(rows)-len(clean)
    if task in REG:
        vals=[]
        for r in clean:
            v=(r['parsed'] or {}).get('value')
            try:vals.append((float(v),float(r['label'])))
            except:miss+=1
        if not vals:return None
        return {'metric':'RMSE','value':round(math.sqrt(sum((a-b)**2 for a,b in vals)/len(vals)),3),'n':len(vals),'miss':miss}
    # classification
    def prob(r):
        s=(r['parsed'] or {}).get('activity_score')
        try:return float(s)
        except:return None
    if task in MULTI:
        byep=collections.defaultdict(list)
        for r in clean:
            p=prob(r); 
            if p is None: continue
            byep[r.get('endpoint','_')].append((p,int(float(r['label']))))
        aucs=[auc(v) for v in byep.values()]; aucs=[a for a in aucs if a is not None]
        return {'metric':'AUC(mean-endpoint)','value':round(sum(aucs)/len(aucs),3) if aucs else None,'endpoints':len(aucs),'n':sum(len(v) for v in byep.values()),'miss':miss}
    pr=[(prob(r),int(float(r['label']))) for r in clean if prob(r) is not None]
    a=auc(pr); return {'metric':'AUC','value':round(a,3) if a else None,'n':len(pr),'miss':miss}
TASKS=['bace','bbbp','clintox','hiv','sider','tox21','esol','freesolv','lipo']
tbl={}
for t in TASKS:
    tbl[t]={}
    for m in MODELS:
        r=score(m,t)
        if r: tbl[t][m]=r
json.dump(tbl,open('FULL_SMILES_TABLE.json','w'),indent=1)
print("DONE")
