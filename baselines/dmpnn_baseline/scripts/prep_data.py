"""Convert MolE-RAG MoleculeNet scaffold splits -> Chemprop v2 CSVs.
Single-task (esol/freesolv/lipo/bbbp/bace/hiv): columns [smiles, <target>].
Multitask (tox21/sider/clintox): pivot same-SMILES rows (keyed by 'task' = subtask) into one row/molecule
with one column per subtask; missing (mol,subtask) left blank (Chemprop masks missing targets)."""
import csv,os,json
BASE="/work/hdd/bhrm/jchan5/MolE-RAG/data/moleculenet_property_scaffold/seed_0"
OUT="/work/hdd/bhrm/jchan5/mole-rag_naacl/dmpnn_baseline/data"
REG={"esol","freesolv","lipo"}
TASKS=["esol","freesolv","lipo","bbbp","bace","hiv","clintox","sider","tox21"]
meta={}
for t in TASKS:
    os.makedirs(f"{OUT}/{t}",exist_ok=True)
    # discover subtasks from train (ordered, stable)
    subtasks=[]; seen=set()
    for r in csv.DictReader(open(f"{BASE}/{t}/{t}_train.csv")):
        if r['task'] not in seen: seen.add(r['task']); subtasks.append(r['task'])
    multitask=len(subtasks)>1
    cols=["smiles"]+subtasks if multitask else ["smiles",subtasks[0]]
    nrows={}
    for sp in ["train","valid","test"]:
        rows=list(csv.DictReader(open(f"{BASE}/{t}/{t}_{sp}.csv")))
        if multitask:
            piv={}  # smiles -> {subtask: label}
            order=[]
            for r in rows:
                s=r['smiles']
                if s not in piv: piv[s]={}; order.append(s)
                piv[s][r['task']]=r['label']
            with open(f"{OUT}/{t}/{sp}.csv","w",newline="") as f:
                w=csv.writer(f); w.writerow(cols)
                for s in order:
                    w.writerow([s]+[piv[s].get(st,"") for st in subtasks])
            nrows[sp]=len(order)
        else:
            with open(f"{OUT}/{t}/{sp}.csv","w",newline="") as f:
                w=csv.writer(f); w.writerow(cols)
                for r in rows: w.writerow([r['smiles'],r['label']])
            nrows[sp]=len(rows)
    meta[t]={"task_type":"regression" if t in REG else "classification","n_targets":len(subtasks),
             "targets":subtasks,"rows":nrows,"multitask":multitask}
    print(f"{t:9s} {'REG' if t in REG else 'CLS'} targets={len(subtasks)} rows(tr/va/te)={nrows['train']}/{nrows['valid']}/{nrows['test']}")
json.dump(meta,open(f"{OUT}/meta.json","w"),indent=2)
print("meta.json written")
