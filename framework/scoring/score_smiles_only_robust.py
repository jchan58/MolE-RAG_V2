#!/usr/bin/env python3
"""Robust + parallel re-implementation of BACE/test_eval/score_full_table.py.

Same scoring math (Mann-Whitney AUC, RMSE, per-endpoint macro) but (1) parses
the first JSON object per file via raw_decode so a double-written/trailing-data
file does not crash the run, and (2) reads the ~66k prediction files with a
thread pool. Writes FULL_SMILES_TABLE.json + a small parse-health report.
"""
import json, glob, math, collections, os
from concurrent.futures import ThreadPoolExecutor

import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
BASE = _os.environ.get("MOLERAG_SO_BASE", _os.path.join(_REPO, "runs", "smiles_only"))
OUT = _os.path.join(_REPO, "scores", "smiles_only_scores.json")
MODELS = ["qwen3_32b", "ministral_3_14b", "gpt5mini", "opus_4_6"]
TASKS = ["bace", "bbbp", "clintox", "hiv", "sider", "tox21", "esol", "freesolv", "lipo"]
REG = {"esol", "lipo", "freesolv"}
MULTI = {"tox21", "sider", "clintox"}
_dec = json.JSONDecoder()

bad = collections.Counter()      # (model,task) -> malformed files
extra = collections.Counter()    # (model,task) -> files with trailing extra data (recovered)


def read_first(path):
    """Return the first JSON object in a file, tolerating trailing extra data."""
    try:
        s = open(path).read()
    except Exception:
        return None, "unreadable"
    try:
        obj, end = _dec.raw_decode(s.lstrip())
        return obj, ("extra" if s.strip()[end - len(s.strip()):].strip() else "ok")
    except Exception:
        # fall back to first non-empty line (JSONL-style)
        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line), "line"
            except Exception:
                break
        return None, "bad"


def dir_for(model, task):
    if task == "bace":
        return os.path.join(BASE, "smilesonly_" + ("qwen" if model == "qwen3_32b" else model))
    return os.path.join(BASE, "smilesonly_multitask", f"{model}__{task}")


def load_rows(model, task):
    d = dir_for(model, task)
    if not os.path.isdir(d):
        return None
    files = glob.glob(d + ("/test_*.json" if task == "bace" else "/*.json"))
    rows = []
    with ThreadPoolExecutor(max_workers=32) as ex:
        for obj, how in ex.map(read_first, files):
            if obj is None:
                bad[(model, task)] += 1
                continue
            if how == "extra":
                extra[(model, task)] += 1
            rows.append(obj)
    return rows


def auc(pairs):
    pos = [s for s, l in pairs if l == 1]
    neg = [s for s, l in pairs if l == 0]
    if not pos or not neg:
        return None
    return sum((s > t) + 0.5 * (s == t) for s in pos for t in neg) / (len(pos) * len(neg))


def prob(r):
    s = (r.get("parsed") or {}).get("activity_score")
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def score(model, task):
    rows = load_rows(model, task)
    if not rows:
        return None
    clean = [r for r in rows if r.get("status") == "clean" and (r.get("parsed") or {})]
    miss = len(rows) - len(clean)
    if task in REG:
        vals = []
        for r in clean:
            try:
                vals.append((float((r["parsed"] or {}).get("value")), float(r["label"])))
            except (TypeError, ValueError):
                miss += 1
        if not vals:
            return None
        rmse = math.sqrt(sum((a - b) ** 2 for a, b in vals) / len(vals))
        return {"metric": "RMSE", "value": round(rmse, 3), "n": len(vals), "miss": miss}
    if task in MULTI:
        byep = collections.defaultdict(list)
        for r in clean:
            p = prob(r)
            if p is None:
                continue
            byep[r.get("endpoint", "_")].append((p, int(float(r["label"]))))
        aucs = [a for a in (auc(v) for v in byep.values()) if a is not None]
        return {"metric": "AUC(mean-endpoint)",
                "value": round(sum(aucs) / len(aucs), 3) if aucs else None,
                "endpoints": len(aucs), "n": sum(len(v) for v in byep.values()), "miss": miss}
    pr = [(prob(r), int(float(r["label"]))) for r in clean if prob(r) is not None]
    a = auc(pr)
    return {"metric": "AUC", "value": round(a, 3) if a else None, "n": len(pr), "miss": miss}


tbl = {}
for t in TASKS:
    tbl[t] = {}
    for m in MODELS:
        r = score(m, t)
        if r:
            tbl[t][m] = r
json.dump(tbl, open(OUT, "w"), indent=1)
total_bad = sum(bad.values()); total_extra = sum(extra.values())
print(f"WROTE {OUT}")
print(f"PARSE HEALTH: {total_extra} files had trailing extra-data (recovered first object); "
      f"{total_bad} unrecoverable (skipped).")
if total_bad:
    print("  unrecoverable by cell:", dict(bad))
if total_extra:
    print("  extra-data by cell:", dict(extra))
print("DONE")
