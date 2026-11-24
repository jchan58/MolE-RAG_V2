#!/usr/bin/env python3
"""Re-score the MolE-RAG C (and A) point estimates for tox21 + hiv from
fw_multi/runs, using the exact math of score_fw_multi.py but with robust
parallel file reads and no bootstrap (point estimates only; Table 2 needs just
the C value). Does NOT touch SCORES_fw_multi.json.
"""
import json, glob, math, re
from concurrent.futures import ThreadPoolExecutor

import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
RUNS = _os.environ.get("MOLERAG_RUNS", _os.path.join(_REPO, "runs"))
OUT = _os.path.join(_REPO, "scores", "molerag_C_tox21_hiv.json")
MODELS = ["qwen3_32b", "ministral_3_14b", "gpt5mini", "opus_4_6"]
TT = {"tox21": "multilabel", "hiv": "classification"}
_dec = json.JSONDecoder()


def _num(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        m = re.search(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", str(v))
        return float(m.group()) if m else None


def getpred(parsed, tt):
    if not parsed:
        return None
    if tt == "classification":
        return _num(parsed.get("probability_positive"))
    e = parsed.get("endpoints")
    return e if isinstance(e, dict) else None


def arm_pred(d, arm, tt):
    if arm == "A":
        return getpred((d.get("A") or {}).get("parsed"), tt)
    if arm == "C":
        return getpred((d.get("C") or {}).get("final"), tt)


def auc(pairs):
    pos = [s for s, l in pairs if l == 1]
    neg = [s for s, l in pairs if l == 0]
    if not pos or not neg:
        return None
    return sum((s > t) + 0.5 * (s == t) for s in pos for t in neg) / (len(pos) * len(neg))


def read_first(f):
    try:
        o, _ = _dec.raw_decode(open(f).read().lstrip())
        return o
    except Exception:
        return None


def load(ds, model):
    files = glob.glob(f"{RUNS}/{ds}__{model}/*.json")
    with ThreadPoolExecutor(max_workers=32) as ex:
        rows = [r for r in ex.map(read_first, files) if r is not None]
    return rows, len(files)


def score(ds, model):
    tt = TT[ds]
    rows, nfiles = load(ds, model)
    bad = nfiles - len(rows)
    out = {"n_files": nfiles, "unparsed": bad}
    for arm in ("A", "C"):
        if tt == "classification":
            xs = []
            for d in rows:
                p = arm_pred(d, arm, tt); y = d.get("target_SEALED")
                if p is None or y is None:
                    continue
                xs.append((p, int(y)))
            a = auc(xs)
            out[arm] = {"n": len(xs), "miss": len(rows) - len(xs),
                        "auc": round(a, 3) if a is not None else None}
        else:  # multilabel macro
            byep = {}; n_used = 0
            for d in rows:
                p = arm_pred(d, arm, tt); y = d.get("target_SEALED")
                if not isinstance(p, dict) or not isinstance(y, dict):
                    continue
                n_used += 1
                for ep, lab in y.items():
                    if lab is None:
                        continue
                    try:
                        pr = float(p.get(ep))
                    except (TypeError, ValueError):
                        continue
                    byep.setdefault(ep, []).append((pr, int(float(lab))))
            valid = [a for a in (auc(v) for v in byep.values()) if a is not None]
            out[arm] = {"n": n_used, "miss": len(rows) - n_used,
                        "mean_auc": round(sum(valid) / len(valid), 3) if valid else None,
                        "n_endpoints_scored": len(valid)}
    return out


report = {}
for ds in ("tox21", "hiv"):
    report[ds] = {}
    for m in MODELS:
        report[ds][m] = score(ds, m)
        hm = "mean_auc" if TT[ds] == "multilabel" else "auc"
        c = report[ds][m]["C"]; a = report[ds][m]["A"]
        print(f"{ds:6} {m:16} files={report[ds][m]['n_files']:5} "
              f"A {hm}={a.get(hm)} C {hm}={c.get(hm)} (C miss={c['miss']})")
json.dump(report, open(OUT, "w"), indent=1)
print("WROTE", OUT)
print("DONE")
