"""Score the BACE framework_final run for chemdfm_v2_14b (A vs C). Mirrors scratchpad/score_final.py
logic (AUC from probability_active, Brier, acc, abstain, paired bootstrap C-A) but scoped to the
chemdfm model and tolerant of a truncated part file. Output: SCORES_C_chemdfm.json next to parts."""
import json, glob, random
from pathlib import Path
BASE = Path("/workspace/molerag_reasoning/BACE/framework_final")
MODELS = ["chemdfm_v2_14b"]
random.seed(0)


def auc(pairs):
    pos = [s for s, l in pairs if l == 1]; neg = [s for s, l in pairs if l == 0]
    if not pos or not neg: return None
    c = sum((s > t) + 0.5 * (s == t) for s in pos for t in neg); return c / (len(pos) * len(neg))


def getprob(d):
    try: return float(d.get("probability_active"))
    except: return None


def predlabel(parsed, prob):
    p = (parsed or {}).get("prediction")
    if p in ("active", "inactive", "uncertain"): return p
    if prob is None: return None
    return "active" if prob >= 0.5 else "inactive"


def load(model):
    rows = {}
    for f in glob.glob(str(BASE / f"{model}_test" / "*.json")):
        try: d = json.load(open(f))
        except Exception: continue
        idx = d["idx"]; lab = int(d["label_SEALED"])
        A = d["A_diff"].get("parsed") or {}; C = (d["C"].get("final") or {})
        pa = getprob(A); pc = getprob(C)
        rows[idx] = {"lab": lab, "A": pa, "C": pc, "Apred": predlabel(A, pa), "Cpred": predlabel(C, pc),
                     "nrounds": len(d["C"].get("rounds", [])), "rec": d}
    return rows


def cond_metrics(rows, key):
    sc = [(r[key], r["lab"]) for r in rows.values() if r[key] is not None]
    miss = sum(1 for r in rows.values() if r[key] is None)
    a = auc(sc)
    brier = sum((s - l) ** 2 for s, l in sc) / len(sc) if sc else None
    acc_t = sum((s >= 0.5) == (l == 1) for s, l in sc) / len(sc) if sc else None
    predkey = {"A": "Apred", "C": "Cpred"}[key]
    comm = [(r[predkey], r["lab"]) for r in rows.values() if r[predkey] in ("active", "inactive")]
    acc_c = sum((p == "active") == (l == 1) for p, l in comm) / len(comm) if comm else None
    abst = sum(1 for r in rows.values() if r[predkey] == "uncertain")
    return {"n": len(sc), "miss": miss, "auc": round(a, 3) if a is not None else None,
            "brier": round(brier, 3) if brier is not None else None,
            "acc@0.5": round(acc_t, 3) if acc_t is not None else None,
            "acc_committed": round(acc_c, 3) if acc_c is not None else None, "abstain": abst, "committed": len(comm)}


def paired_boot(rows, B=2000):
    both = [(r["A"], r["C"], r["lab"]) for r in rows.values() if r["A"] is not None and r["C"] is not None]
    if len(both) < 5: return None
    idx = list(range(len(both))); dA = []
    for _ in range(B):
        samp = [both[random.choice(idx)] for _ in idx]
        aA = auc([(a, l) for a, c, l in samp]); aC = auc([(c, l) for a, c, l in samp])
        if aA is not None and aC is not None: dA.append(aC - aA)
    def ci(x): x = sorted(x); return (round(x[int(.025 * len(x))], 3), round(x[int(.975 * len(x))], 3))
    aA0 = auc([(a, l) for a, c, l in both]); aC0 = auc([(c, l) for a, c, l in both])
    return {"n_paired": len(both), "A_auc": round(aA0, 3), "C_auc": round(aC0, 3),
            "dAUC_C_minus_A": round(aC0 - aA0, 3), "dAUC_CI": ci(dA) if dA else None}


report = {}
print(f"{'model':16} {'cond':4} {'n':>4} {'miss':>4} {'AUC':>6} {'Brier':>6} {'acc@.5':>7} {'accCom':>7} {'abst':>4}")
for m in MODELS:
    if not glob.glob(str(BASE / f"{m}_test" / "*.json")): print(f"{m}: (no parts yet)"); continue
    rows = load(m); report[m] = {"n_files": len(rows)}
    for k in ("A", "C"):
        met = cond_metrics(rows, k); report[m][k] = met
        print(f"{m:16} {k:4} {met['n']:>4} {met['miss']:>4} {str(met['auc']):>6} {str(met['brier']):>6} {str(met['acc@0.5']):>7} {str(met['acc_committed']):>7} {met['abstain']:>4}")
    report[m]["paired_C_minus_A"] = paired_boot(rows)
    print("    paired C-A:", report[m]["paired_C_minus_A"])
out = Path("/workspace/mole-rag_naacl/chemdfm_baseline/chemdfm_v2_ep/SCORES_C_chemdfm.json")
json.dump(report, open(out, "w"), indent=1); print("\nsaved", out)
