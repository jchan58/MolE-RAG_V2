"""
SCOPE-CUT hiv clean set to a fixed PMCID BUDGET so chunk/judge/extract fit a ~2h window.
Uses the already-built molecule_pmcids.json (no 253k re-read). Strategy: keep ALL property
PMCIDs, then add molecules SMALLEST-first (most specific literature), up to TOPK pmcids each,
until the clean set reaches BUDGET. Smallest-first maximises how MANY molecules get a card.
Overwrites data/clean_pmcids.txt + data/clean_set_summary.json.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

BUDGET = int(os.environ.get("EXP70_CLEAN_BUDGET", "30000"))
TOPK = int(os.environ.get("EXP70_CLEAN_TOPK", "3"))


def _sk(x):
    return int(x) if str(x).isdigit() else x


def main():
    prop = [str(p) for p in (C.read_json(C.P("data", "property_task_pmcids.json"), {}) or {}).get("pmcids", [])]
    mol = C.read_json(C.P("data", "molecule_pmcids.json"), {}) or {}
    print("loaded property=%d molecules=%d budget=%d topk=%d" % (len(prop), len(mol), BUDGET, TOPK), flush=True)

    clean = set(prop)
    # molecules smallest-first (skip zero), add up to TOPK each. Build a REDUCED
    # molecule_pmcids.json so chunk_download (which iterates molecule_pmcids.json, NOT
    # clean_pmcids.txt) only downloads the budget molecules.
    mids = sorted(mol.keys(), key=lambda m: mol[m].get("n_pmcids", 0))
    reduced = {}
    n_mol_used = 0
    for m in mids:
        pm = mol[m].get("pmcids", []) or []
        if not pm:
            continue
        if len(clean) >= BUDGET:
            break
        keep = [str(p) for p in pm[:TOPK]]
        for p in keep:
            clean.add(p)
        info = dict(mol[m])
        info["pmcids"] = keep
        info["n_pmcids"] = len(keep)
        info["scopecut_from_n_pmcids"] = mol[m].get("n_pmcids", len(pm))
        reduced[m] = info
        n_mol_used += 1
    clean_sorted = sorted(clean, key=_sk)

    # back up the full molecule map once, then write the reduced one in its place
    full_path = C.P("data", "molecule_pmcids.json")
    bak = C.P("data", "molecule_pmcids.full.json")
    if not os.path.exists(bak):
        try:
            import shutil; shutil.copy2(full_path, bak)
        except Exception as e:
            print("WARN backup: %r" % e, flush=True)
    C.write_json(full_path, reduced)

    C.atomic_write_text(C.P("data", "clean_pmcids.txt"), "\n".join(clean_sorted) + "\n", fsync=False)
    summ = dict(computed_at=C.now_iso(), builder="budget_scopecut",
                budget=BUDGET, topk=TOPK,
                property_pmcids=len(prop), molecules_total=len(mol),
                molecules_used=n_mol_used, clean_total_pmcids=len(clean_sorted),
                reduced_molecule_map=True, full_backup="molecule_pmcids.full.json")
    C.write_json(C.P("data", "clean_set_summary.json"), summ)
    print("SCOPE-CUT CLEAN SET: %d pmcids (property=%d, %d molecules @ top-%d); reduced molecule_pmcids.json written (full backed up)"
          % (len(clean_sorted), len(prop), n_mol_used, TOPK), flush=True)


if __name__ == "__main__":
    main()
