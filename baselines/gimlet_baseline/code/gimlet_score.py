#!/usr/bin/env python
"""Score GIMLET zero-shot predictions -> analysis/results.json (chemrag conventions)."""
import os, sys, json, csv, math, argparse
import numpy as np
from sklearn.metrics import roc_auc_score

DATA_ROOT = '/work/hdd/bhrm/jchan5/MolE-RAG/data/moleculenet_property_scaffold/seed_0'
REG_TASKS = {'esol', 'freesolv', 'lipo'}
ALL_TASKS = ['bbbp', 'bace', 'hiv', 'clintox', 'sider', 'tox21', 'esol', 'freesolv', 'lipo']
MULTITASK = {'tox21', 'sider', 'clintox'}


def read_csv_meta(task):
    """rowidx -> (endpoint, label_float); and ordered list of valid labels."""
    fn = os.path.join(DATA_ROOT, task, '%s_test.csv' % task)
    meta = {}
    labels = []
    with open(fn) as f:
        for i, row in enumerate(csv.DictReader(f)):
            lab = row['label'].strip()
            if lab == '' or lab.lower() in ('nan', 'none'):
                continue
            try:
                y = float(lab)
            except ValueError:
                continue
            if math.isnan(y):
                continue
            meta[i] = (row['task'], y)
            labels.append(y)
    return meta, labels


def load_preds(path):
    preds = {}
    if not os.path.exists(path):
        return preds
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rowidx = int(r['topic_id'].rsplit('_', 1)[1])
            preds[rowidx] = r
    return preds


def score_task(task, out_dir):
    meta, labels = read_csv_meta(task)
    n = len(meta)
    preds = load_preds(os.path.join(out_dir, '%s_preds.jsonl' % task))

    if task in REG_TASKS:
        ys, ps = [], []
        numeric = 0
        for rowidx, (ep, y) in meta.items():
            r = preds.get(rowidx)
            if r is None:
                continue
            p = r.get('pred')
            if p is None or (isinstance(p, float) and math.isnan(p)):
                continue
            ys.append(y); ps.append(float(p)); numeric += 1
        ys = np.array(ys); ps = np.array(ps)
        res = {"type": "regression", "n": n, "scored": len(preds), "numeric": numeric,
               "abstain": len(preds) - numeric}
        if numeric > 0:
            res["RMSE"] = round(float(np.sqrt(np.mean((ys - ps) ** 2))), 4)
            res["MAE"] = round(float(np.mean(np.abs(ys - ps))), 4)
        alllab = np.array(labels)
        res["mean_predictor_RMSE"] = round(float(np.std(alllab)), 4)
        return res

    # classification
    ys, ps, eps = [], [], []
    for rowidx, (ep, y) in meta.items():
        r = preds.get(rowidx)
        if r is None:
            continue
        pp = r.get('p_pos')
        if pp is None or (isinstance(pp, float) and math.isnan(pp)):
            continue
        ys.append(int(round(y))); ps.append(float(pp)); eps.append(ep)
    ys = np.array(ys); ps = np.array(ps); eps = np.array(eps)
    res = {"type": "classification", "n": n, "scored": len(ys)}
    pos_rate = float(np.mean(ys)) if len(ys) else float('nan')
    res["pos_rate"] = round(pos_rate, 4)
    if len(np.unique(ys)) > 1:
        res["AUC"] = round(float(roc_auc_score(ys, ps)), 4)
        res["acc@0.5"] = round(float(np.mean((ps >= 0.5).astype(int) == ys)), 4)
    if task in MULTITASK:
        aucs = {}
        for ep in sorted(set(eps.tolist())):
            m = eps == ep
            yy = ys[m]
            if len(np.unique(yy)) > 1:
                aucs[ep] = round(float(roc_auc_score(yy, ps[m])), 4)
        res["n_endpoints"] = len(set(eps.tolist()))
        res["n_endpoints_scored"] = len(aucs)
        res["macro_AUC"] = round(float(np.mean(list(aucs.values()))), 4) if aucs else None
        res["per_endpoint_AUC"] = aucs
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='/work/hdd/bhrm/jchan5/mole-rag_naacl/gimlet_baseline/outputs')
    ap.add_argument('--analysis_dir', default='/work/hdd/bhrm/jchan5/mole-rag_naacl/gimlet_baseline/analysis')
    ap.add_argument('--tasks', nargs='+', default=ALL_TASKS)
    args = ap.parse_args()
    os.makedirs(args.analysis_dir, exist_ok=True)
    results = {}
    for task in args.tasks:
        try:
            results[task] = score_task(task, args.out_dir)
            print(task, json.dumps(results[task]))
        except Exception as e:
            print("ERROR scoring", task, repr(e))
    with open(os.path.join(args.analysis_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("WROTE", os.path.join(args.analysis_dir, 'results.json'))


if __name__ == '__main__':
    main()
