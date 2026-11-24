"""
FAST PARALLEL clean-set builder for hiv (takes the slow serial identity merge OFF the
critical path). Reads property + molecule/aug search results directly from results/search/
with a thread pool, writes:
  data/property_task_pmcids.json  (complete, PROP_CAP top-K per property search)
  data/clean_pmcids.txt + data/clean_set_summary.json
  state/COLLECT_DONE
so hiv_watcher_sm can sync data/ to S3 and submit the SageMaker chunk/BM25 job NOW,
in parallel with the (still-running) full registry merge that cards need later.
Read-only w.r.t. search results; only ADDS/overwrites the files above (idempotent).
"""
import os, sys, json, time
import concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import datasets as DS
import search as S
try:
    import augment as AUG
    aug_items = AUG.search_items_aug()
except Exception:
    aug_items = {}

PROP_CAP = int(os.environ.get("EXP70_PROP_CAP", "200"))
CLEAN_MAX = 1000
SIZE_CAP = 22000
NTHR = int(os.environ.get("EXP70_CLEAN_THREADS", "48"))
BASE = C.P("results", "search")


def _sk(x):
    return int(x) if str(x).isdigit() else x


def _pj(sid):
    return os.path.join(BASE, sid, "pmcids.json")


def _read_prop(sid):
    try:
        pm = C.read_json(_pj(sid), []) or []
    except Exception:
        pm = []
    if len(pm) > PROP_CAP:
        pm = pm[:PROP_CAP]
    return [str(p) for p in pm]


def _read_mol(sid):
    # size pre-filter: broad searches (>SIZE_CAP bytes) are noise -> skip without loading
    try:
        if os.path.getsize(_pj(sid)) > SIZE_CAP:
            return None
    except OSError:
        return None
    try:
        pm = C.read_json(_pj(sid), []) or []
    except Exception:
        return None
    if len(pm) > CLEAN_MAX:
        return None
    return [str(p) for p in pm]


def main():
    t0 = time.time()
    main_items = S.search_items()
    prop_sids, prop_search_ids = set(), set()
    mol_sids = set()
    for d in (main_items, aug_items):
        for sid, it in d.items():
            qt = it.get("query_types") or []
            if "property" in qt:
                prop_sids.add(sid); prop_search_ids.add(sid)
            if "molecule" in qt or "synonym_augmentation" in qt:
                mol_sids.add(sid)
    print("items: property_sids=%d molecule_sids=%d (nthr=%d)" % (len(prop_sids), len(mol_sids), NTHR), flush=True)

    # PROPERTY (parallel)
    prop_pmcids = set()
    with cf.ThreadPoolExecutor(max_workers=NTHR) as ex:
        for lst in ex.map(_read_prop, prop_sids):
            prop_pmcids.update(lst)
    prop_sorted = sorted(prop_pmcids, key=_sk)
    C.write_json(C.P("data", "property_task_pmcids.json"), dict(
        task="%s_property" % DS.CFG["name"], found_by="property",
        n_search_queries=len(prop_search_ids),
        search_ids=sorted(prop_search_ids),
        n_pmcids=len(prop_sorted), pmcids=prop_sorted))
    print("property_task_pmcids.json: %d queries, %d pmcids (%.0fs)" % (len(prop_search_ids), len(prop_sorted), time.time()-t0), flush=True)

    # MOLECULE (parallel) -> clean union
    clean = set(prop_pmcids)
    n_clean = scanned = 0
    with cf.ThreadPoolExecutor(max_workers=NTHR) as ex:
        for res in ex.map(_read_mol, mol_sids):
            if res is None:
                continue
            scanned += 1; n_clean += 1
            clean.update(res)
    clean_sorted = sorted(clean, key=_sk)
    C.atomic_write_text(C.P("data", "clean_pmcids.txt"), "\n".join(clean_sorted) + "\n", fsync=False)
    summ = dict(computed_at=C.now_iso(), clean_max_reported=CLEAN_MAX,
                property_pmcids=len(prop_sorted), molecule_searches_total=len(mol_sids),
                clean_molecule_searches=n_clean, searches_scanned=scanned,
                clean_total_pmcids=len(clean_sorted), builder="parallel_fast",
                elapsed_s=round(time.time()-t0, 1))
    C.write_json(C.P("data", "clean_set_summary.json"), summ)
    print("CLEAN SET: %d pmcids (property=%d, %d clean molecule searches) in %.0fs"
          % (len(clean_sorted), len(prop_sorted), n_clean, time.time()-t0), flush=True)

    # unblock SageMaker
    C.atomic_write_text(C.P("state", "COLLECT_DONE"), C.now_iso())
    print("COLLECT_DONE written -> hiv_watcher_sm will sync data/ + submit SageMaker", flush=True)


if __name__ == "__main__":
    main()
