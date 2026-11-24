#!/usr/bin/env python3
"""Convert MoleculeNet seed_0 scaffold splits (long format) -> KANO/chemprop WIDE CSVs.

Our splits: smiles,label,task,dataset,task_type,metric,split  (one row per molecule x endpoint).
chemprop wants: smiles,<endpoint1>,<endpoint2>,...  (one row per molecule; empty = missing label).

Writes kano_baseline/data/<task>/{train,val,test}.csv and a tasks.json manifest
(dataset_type + metric + endpoint columns). Single-task sets get one label column.
This lets KANO finetune on EXACTLY our seed_0 partition via --separate_val_path/--separate_test_path.
"""
import csv, json
from collections import OrderedDict
from pathlib import Path

SRC = Path("/work/hdd/bhrm/jchan5/MolE-RAG/data/moleculenet_property_scaffold/seed_0")
DEST = Path("/work/hdd/bhrm/jchan5/mole-rag_naacl/kano_baseline/data")
CLS = ["bbbp", "bace", "hiv", "clintox", "sider", "tox21"]
REG = ["esol", "freesolv", "lipo"]


def load_long(path):
    rows = list(csv.DictReader(open(path)))
    endpoints = []  # preserve first-seen order
    seen = set()
    for r in rows:
        ep = r.get("task") or "y"
        if ep not in seen:
            seen.add(ep); endpoints.append(ep)
    return rows, endpoints


def pivot(rows, endpoints):
    # molecule order = first appearance; cell = label string ('' if absent)
    mols = OrderedDict()  # smiles -> {endpoint: label}
    for r in rows:
        smi = r["smiles"].strip()
        lab = r.get("label")
        if lab is None or str(lab).strip().lower() in ("", "nan", "none"):
            continue
        ep = r.get("task") or "y"
        mols.setdefault(smi, {})[ep] = str(lab).strip()
    return mols


def write_wide(mols, endpoints, out):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles"] + endpoints)
        for smi, d in mols.items():
            w.writerow([smi] + [d.get(ep, "") for ep in endpoints])
    return len(mols)


def main():
    manifest = {}
    for t in CLS + REG:
        dtype = "classification" if t in CLS else "regression"
        metric = "auc" if t in CLS else "rmse"
        # endpoints from the UNION across splits (train defines the set; all share it)
        tr_rows, eps = load_long(SRC / t / f"{t}_train.csv")
        d = DEST / t; d.mkdir(parents=True, exist_ok=True)
        counts = {}
        for split, fn in [("train", "train"), ("val", "valid"), ("test", "test")]:
            rows, _ = load_long(SRC / t / f"{t}_{fn}.csv")
            mols = pivot(rows, eps)
            counts[split] = write_wide(mols, eps, d / f"{split}.csv")
        manifest[t] = {"dataset_type": dtype, "metric": metric,
                       "n_endpoints": len(eps), "endpoints": eps, "counts": counts}
        print(f"{t:<10} {dtype:<14} endpoints={len(eps):<3} "
              f"train/val/test={counts['train']}/{counts['val']}/{counts['test']}", flush=True)
    json.dump(manifest, open(DEST.parent / "tasks.json", "w"), indent=1)
    print("wrote tasks.json")


if __name__ == "__main__":
    main()
