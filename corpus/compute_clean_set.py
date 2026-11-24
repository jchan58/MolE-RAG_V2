"""
Compute the CLEAN full-text-processing set (user scope): property PMCIDs + PMCIDs
from molecule/synonym searches whose retrieved count is small (<= CLEAN_MAX, i.e.
reported_count <= 1000 -> not broad-match noise). Light + low-memory: uses the
pmcids.json FILE SIZE as a cheap pre-filter so mega-molecule id lists (millions of
entries) are never loaded.
Writes data/clean_pmcids.txt + data/clean_set_summary.json.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CLEAN_MAX = 1000
SIZE_CAP = 22000  # bytes: ~ >1000 PMCIDs (each ~8 digits + quotes/comma) -> skip


def _sk(x):
    return int(x) if str(x).isdigit() else x


def run():
    import augment as AUG
    import search as S
    main_items = S.search_items()
    aug_items = AUG.search_items_aug()
    # which sids are molecule/synonym searches
    mol_sids = set()
    for d in (main_items, aug_items):
        for sid, it in d.items():
            qt = it.get("query_types") or []
            if "molecule" in qt or "synonym_augmentation" in qt:
                mol_sids.add(sid)

    # property PMCIDs come from property_task_pmcids.json, which merge.py already caps to the
    # top-PROP_CAP per property search (the broad-topic scope cap lives in merge, where the
    # per-search results are available; the SageMaker box only has data/, not results/search/).
    prop = set(str(p) for p in (C.read_json(C.P("data", "property_task_pmcids.json"), {}) or {}).get("pmcids", []))
    clean = set(prop)
    base = C.P("results", "search")
    n_clean = 0
    scanned = 0
    for sid in mol_sids:
        pj = os.path.join(base, sid, "pmcids.json")
        try:
            if os.path.getsize(pj) > SIZE_CAP:
                continue  # broad search -> not clean
        except OSError:
            continue
        pm = C.read_json(pj, []) or []
        scanned += 1
        if len(pm) <= CLEAN_MAX:
            n_clean += 1
            clean.update(str(p) for p in pm)
    clean_sorted = sorted(clean, key=_sk)
    C.atomic_write_text(C.P("data", "clean_pmcids.txt"), "\n".join(clean_sorted) + "\n", fsync=False)
    summ = dict(computed_at=C.now_iso(), clean_max_reported=CLEAN_MAX,
                property_pmcids=len(prop), molecule_searches_total=len(mol_sids),
                clean_molecule_searches=n_clean, searches_scanned=scanned,
                clean_total_pmcids=len(clean_sorted))
    C.write_json(C.P("data", "clean_set_summary.json"), summ)
    C.log("INFO", "clean_set", "CLEAN SET: %d pmcids (property=%d, %d clean molecule searches)"
          % (len(clean_sorted), len(prop), n_clean))
    return summ


if __name__ == "__main__":
    run()
