"""
Lightweight augmentation finalize (avoids the slow full 8.85M re-merge):
 1. Union synonym-augmentation PMCIDs into molecule_pmcids.json per molecule with
    found_by + synonym_aug_pmcids provenance (keeps all existing).
 2. Extend data/registry_pmcids.txt with any genuinely-new PMCIDs.
 3. Write state/AUG_RESULTS.json (counts).
 4. Compute the CLEAN full-text-processing set (user scope for the ~1h target):
    property PMCIDs + PMCIDs from molecule/synonym searches with reported_count <= 1000.
    -> data/clean_pmcids.txt + data/clean_set_summary.json. (ALL PMCIDs stay in the
    index files; this set only limits download/chunk/judge.)
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import augment as AUG
import search as S

CLEAN_MAX_REPORTED = 1000


def _sk(x):
    return int(x) if str(x).isdigit() else x


def run():
    reg_path = C.P("data", "registry_pmcids.txt")
    reg = set()
    with open(reg_path) as f:
        for line in f:
            s = line.strip()
            if s:
                reg.add(s)
    before = len(reg)
    C.log("INFO", "aug_finalize", "loaded registry set: %d" % before)

    # augmentation per-molecule pmcids
    aug_items = AUG.search_items_aug()
    mol_aug = {}
    all_aug = set()
    for sid, it in aug_items.items():
        pm = C.read_json(C.P("results", "search", sid, "pmcids.json"), []) or []
        pm = [str(p) for p in pm]
        all_aug.update(pm)
        for m in it.get("molecule_ids", []):
            mol_aug.setdefault(m, set()).update(pm)
    new_aug = all_aug - reg
    C.log("INFO", "aug_finalize", "aug pmcids: %d total, %d NEW to registry" % (len(all_aug), len(new_aug)))

    # update molecule_pmcids.json
    mp = C.read_json(C.P("data", "molecule_pmcids.json"), {}) or {}
    gained = 0
    for m, augset in mol_aug.items():
        e = mp.get(m)
        if e is None:
            e = dict(cid=None, split=None, found_by="synonym_augmentation", n_pmcids=0, pmcids=[])
            mp[m] = e
        existing = set(str(p) for p in e.get("pmcids", []))
        newly = augset - existing
        if newly:
            gained += 1
        allp = sorted(existing | augset, key=_sk)
        e["pmcids"] = allp
        e["n_pmcids"] = len(allp)
        e["synonym_aug_pmcids"] = sorted(newly, key=_sk)
        e["n_synonym_aug_only"] = len(newly)
        if newly:
            e["found_by"] = ("molecule+synonym_augmentation" if existing else "synonym_augmentation")
    C.write_json(C.P("data", "molecule_pmcids.json"), mp)
    C.log("INFO", "aug_finalize", "molecule_pmcids updated; %d molecules gained new PMCIDs" % gained)

    # extend registry
    if new_aug:
        reg |= new_aug
        allsorted = sorted(reg, key=_sk)
        C.atomic_write_text(reg_path, "\n".join(allsorted) + "\n", fsync=False)

    results = dict(computed_at=C.now_iso(),
                   augmentation_searches=len(aug_items),
                   aug_pmcids_total=len(all_aug),
                   total_unique_pmcids_before=before,
                   total_unique_pmcids_after=len(reg),
                   new_unique_pmcids=len(new_aug),
                   molecules_that_gained_pmcids=gained,
                   note=("most augmentation PMCIDs were already in the (very broad) registry; "
                         "the main gain is additional per-molecule associations"))
    C.write_json(C.P("state", "AUG_RESULTS.json"), results)
    C.log("INFO", "aug_finalize", "AUG_RESULTS: %s" % json.dumps(results))

    # ---- clean set ----
    prop = set(str(p) for p in (C.read_json(C.P("data", "property_task_pmcids.json"), {}) or {}).get("pmcids", []))
    clean = set(prop)
    items_main = S.search_items()
    base = C.P("results", "search")
    n_clean_searches = 0
    for sid in os.listdir(base):
        mpath = os.path.join(base, sid, "manifest.json")
        m = C.read_json(mpath)
        if not m:
            continue
        qtypes = m.get("query_types") or (items_main.get(sid, {}) or {}).get("query_types") \
            or (aug_items.get(sid, {}) or {}).get("query_types") or []
        if not ("molecule" in qtypes or "synonym_augmentation" in qtypes):
            continue
        rc = m.get("reported_count_total")
        if rc is None or rc > CLEAN_MAX_REPORTED:
            continue
        n_clean_searches += 1
        for p in (C.read_json(os.path.join(base, sid, "pmcids.json"), []) or []):
            clean.add(str(p))
    clean_sorted = sorted(clean, key=_sk)
    C.atomic_write_text(C.P("data", "clean_pmcids.txt"), "\n".join(clean_sorted) + "\n", fsync=False)
    summ = dict(computed_at=C.now_iso(), clean_max_reported=CLEAN_MAX_REPORTED,
                property_pmcids=len(prop), clean_molecule_searches=n_clean_searches,
                clean_total_pmcids=len(clean_sorted))
    C.write_json(C.P("data", "clean_set_summary.json"), summ)
    C.log("INFO", "aug_finalize", "CLEAN SET: %d pmcids (property=%d, %d clean molecule searches)"
          % (len(clean_sorted), len(prop), n_clean_searches))
    return results, summ


if __name__ == "__main__":
    run()
