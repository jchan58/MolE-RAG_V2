"""
Stage 9 — merge search results into the article registry (MEMORY-LIGHT).

Union of property + molecule search results, deduped by PMCID. The over-broad molecule
searches for these datasets return millions of PMCIDs each (lipo alone: ~9.3M molecule-only
PMCIDs), so the old per-PMCID aggregation dict (one dict with 4 sets PER pmcid over ~10M
pmcids) needed 18-37 GB and got OOM-killed under the shared 40 GB user cgroup.

This version NEVER builds that per-pmcid dict. It keeps only a few flat PMCID sets
(property / molecule / synonym-augmentation) and derives the found_by_* counts as set sizes
and intersections. Broad molecule/aug searches are BOUNDED exactly like compute_clean_set.py:
a cheap pmcids.json FILE-SIZE pre-filter (> SIZE_CAP bytes) skips them WITHOUT ever loading
the mega id-list, and a post-load len(pmcids) > CLEAN_MAX guard backs it up. Property
searches are always kept in full. The resulting registry therefore has
registry_scope = "clean_bounded": total_unique_pmcids reflects the clean-ish set (the full
~10M set was unused downstream), not the broad-match 10M.

The heavy, download-unused articles.jsonl + query_article_matches.jsonl are NOT written
unless EXP70_MERGE_WRITE_ARTICLES=1 (they are consumed by nothing in the chunk/BM25/judge/
cards path — grep-confirmed); the pointer files the downstream DOES read are still written:
  data/molecule_pmcids.json        — {mol_id: {cid, split, pmcids, found_by, ...}}  (bounded)
  data/property_task_pmcids.json   — shared property-task PMCIDs
  data/registry_pmcids.txt         — sorted unique PMCIDs of the clean-bounded registry

build_registry()'s return value + data/registry_summary.json keep the SAME key schema
(run_collect / progress / report read them), plus two honest new fields:
  n_molecule_searches_broad_skipped, registry_scope.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import datasets as DS

# Bounding thresholds (match compute_clean_set.py so the registry == the clean fetch scope).
CLEAN_MAX = int(os.environ.get("EXP70_CLEAN_MAX", "1000"))      # skip molecule/aug search if >this many PMCIDs
SIZE_CAP = int(os.environ.get("EXP70_CLEAN_SIZE_CAP", "22000"))  # bytes: >1000 PMCIDs -> broad, never load
PROP_CAP = int(os.environ.get("EXP70_PROP_CAP", "200"))         # top-K PMCIDs kept per property search (scope cap)


def shard_of(pmcid):
    s = str(pmcid)
    return s[-3:].zfill(3)


def _search_dirs():
    base = C.P("results", "search")
    for sid in sorted(os.listdir(base)):
        d = os.path.join(base, sid)
        if os.path.isdir(d):
            yield sid, d


def build_registry():
    import search as S
    items = S.search_items()
    # additively include synonym-augmentation searches (if that pass has run)
    try:
        import augment as AUG
        items.update(AUG.search_items_aug())
    except Exception:
        pass

    write_articles = os.environ.get("EXP70_MERGE_WRITE_ARTICLES", "0") == "1"

    # FLAT bounded PMCID sets (no per-pmcid dict) — this is the whole memory fix.
    prop_pmcids = set()         # from property searches (kept in full)
    prop_search_ids = set()
    mol_nonaug_pmcids = set()   # from non-broad molecule (non-aug) searches
    aug_pmcids = set()          # from non-broad synonym-augmentation searches
    mol_pmcids = {}             # mol_id -> set (bounded) — needed for molecule_pmcids.json
    mol_pmcids_aug = {}         # mol_id -> set (aug only, bounded)
    # Only populated when EXP70_MERGE_WRITE_ARTICLES=1 (bounded, so never the 10M dict).
    agg = {} if write_articles else None

    n_searches = 0
    n_broad_skipped = 0
    incomplete_searches = []

    # PARALLEL reads: tens of thousands of tiny manifest.json/pmcids.json on slow/over-quota
    # Lustre; reading serially stalls for hours (esp. HIV, 41k+ searches). The reader decides
    # PER SEARCH whether to load pmcids.json at all: property -> always; molecule/aug -> only
    # if the pmcids.json file is small (<= SIZE_CAP), so the broad mega-lists never hit RAM.
    import concurrent.futures as _cf

    def _classify(sid_d):
        sid, d = sid_d
        manifest = C.read_json(os.path.join(d, "manifest.json"))
        if not manifest or not manifest.get("complete"):
            return (sid, manifest, "incomplete", None)
        types = (items.get(sid, {}) or {}).get("query_types", [])
        pj = os.path.join(d, "pmcids.json")
        if "property" in types:
            return (sid, manifest, "read", C.read_json(pj, []))
        # molecule / synonym_augmentation: cheap FILE-SIZE pre-filter (never load broad lists)
        try:
            if os.path.getsize(pj) > SIZE_CAP:
                return (sid, manifest, "broad", None)
        except OSError:
            return (sid, manifest, "broad", None)
        return (sid, manifest, "read", C.read_json(pj, []))

    _dirs = list(_search_dirs())
    _nthr = int(os.environ.get("EXP70_MERGE_THREADS", "16"))
    with _cf.ThreadPoolExecutor(max_workers=_nthr) as _ex:
        for sid, manifest, action, pmcids in _ex.map(_classify, _dirs):
            if action == "incomplete":
                if manifest:
                    incomplete_searches.append(sid)
                continue
            n_searches += 1
            item = items.get(sid, {})
            qids = item.get("query_ids", [])
            mids = item.get("molecule_ids", [])
            types = item.get("query_types", [])
            is_prop = "property" in types
            is_aug = "synonym_augmentation" in types
            if is_prop:
                prop_search_ids.add(sid)
            if action == "broad":
                n_broad_skipped += 1
                continue
            # action == "read": double-check the bound for molecule/aug (never bound property)
            if not is_prop and len(pmcids) > CLEAN_MAX:
                n_broad_skipped += 1
                continue
            # PROPERTY scope cap: keep only the top-K PMCIDs per property search. Broad property
            # topics (e.g. sider ADRs) otherwise push property_task_pmcids.json to ~9M -> infeasible
            # to download/chunk. Top-K bounds it; BM25 prescreen + rerank select relevant passages.
            if is_prop and len(pmcids) > PROP_CAP:
                pmcids = pmcids[:PROP_CAP]
            for p in pmcids:
                if is_prop:
                    prop_pmcids.add(p)
                if mids:
                    if is_aug:
                        aug_pmcids.add(p)
                    else:
                        mol_nonaug_pmcids.add(p)
                    for m in mids:
                        mol_pmcids.setdefault(m, set()).add(p)
                        if is_aug:
                            mol_pmcids_aug.setdefault(m, set()).add(p)
                if agg is not None:
                    a = agg.get(p)
                    if a is None:
                        a = dict(pmcid=p, search_ids=set(), query_ids=set(),
                                 molecule_ids=set(), types=set())
                        agg[p] = a
                    a["search_ids"].add(sid)
                    a["query_ids"].update(qids)
                    a["molecule_ids"].update(mids)
                    a["types"].update(types)

    def _sk(x):
        return int(x) if str(x).isdigit() else x

    # optional (default OFF): the download-unused articles.jsonl + query_article_matches.jsonl
    if agg is not None:
        def found_by(types):
            srcs = []
            if "property" in types:
                srcs.append("property")
            if "molecule" in types:
                srcs.append("molecule")
            if "synonym_augmentation" in types:
                srcs.append("synonym_augmentation")
            if len(srcs) == 1:
                return srcs[0]
            if set(srcs) == {"property", "molecule"}:
                return "both"
            return "+".join(srcs) if srcs else "unknown"

        with open(C.P("data", "articles.jsonl"), "w") as fa, \
             open(C.P("data", "query_article_matches.jsonl"), "w") as fm:
            for p in sorted(agg, key=_sk):
                a = agg[p]
                fb = found_by(a["types"])
                fa.write(json.dumps(dict(pmcid=p, pmcid_full="PMC%s" % p, found_by=fb,
                                         n_searches=len(a["search_ids"]),
                                         molecule_ids=sorted(a["molecule_ids"]),
                                         search_ids=sorted(a["search_ids"]),
                                         query_ids=sorted(a["query_ids"]),
                                         candidate_molecule_association=sorted(a["molecule_ids"]),
                                         association_note=("molecule associations are CANDIDATE "
                                                           "query matches, not confirmed identity")),
                                    ensure_ascii=False) + "\n")
                fm.write(json.dumps(dict(pmcid=p, found_by=fb,
                                         query_ids=sorted(a["query_ids"]),
                                         search_ids=sorted(a["search_ids"]),
                                         molecule_ids=sorted(a["molecule_ids"])),
                                    ensure_ascii=False) + "\n")

    # per-molecule mapping (molecule_pmcids.json) — on BOUNDED sets, so memory-safe
    ident = {}
    pdir = C.P("results", "pubchem")
    if os.path.isdir(pdir):
        for fn in os.listdir(pdir):
            if fn.endswith(".json"):
                r = C.read_json(os.path.join(pdir, fn))
                if r and r.get("mol_id"):
                    ident[r["mol_id"]] = r

    mol_map = {}
    n_mol_gained_from_aug = 0
    for mid in sorted(set(mol_pmcids.keys()) | set(ident.keys())):
        allp = mol_pmcids.get(mid, set())
        augp = mol_pmcids_aug.get(mid, set())
        aug_only = augp - (allp - augp)   # PMCIDs that came ONLY via synonym-augmentation
        if aug_only:
            n_mol_gained_from_aug += 1
        idrec = ident.get(mid, {})
        fb = "molecule"
        if augp:
            fb = "molecule+synonym_augmentation" if (allp - augp) else "synonym_augmentation"
        mol_map[mid] = dict(
            cid=idrec.get("cid"),
            match_type=idrec.get("match_type"),
            preferred_name=idrec.get("preferred_name"),
            split=idrec.get("split"),
            found_by=fb,
            n_pmcids=len(allp),
            overlap_with_property=sorted(allp & prop_pmcids, key=_sk),
            pmcids=sorted(allp, key=_sk),
            synonym_aug_pmcids=sorted(aug_only, key=_sk),
            n_synonym_aug_only=len(aug_only),
        )
    C.write_json(C.P("data", "molecule_pmcids.json"), mol_map)

    prop_sorted = sorted(prop_pmcids, key=_sk)
    C.write_json(C.P("data", "property_task_pmcids.json"), dict(
        task="%s_property" % DS.CFG["name"], found_by="property",
        n_search_queries=len(prop_search_ids),
        search_ids=sorted(prop_search_ids),
        n_pmcids=len(prop_sorted), pmcids=prop_sorted))

    # registry pmcid list (clean-bounded) — drives metadata/download/oa_gate
    mol_union = mol_nonaug_pmcids | aug_pmcids
    all_set = prop_pmcids | mol_union
    all_pmcids = sorted(all_set, key=_sk)
    C.atomic_write_text(C.P("data", "registry_pmcids.txt"),
                        "\n".join(str(p) for p in all_pmcids) + ("\n" if all_pmcids else ""))

    both = prop_pmcids & mol_union
    aug_only_pmcids = aug_pmcids - prop_pmcids - mol_nonaug_pmcids
    summary = dict(
        built_at=C.now_iso(), n_searches_complete=n_searches,
        n_searches_incomplete=len(incomplete_searches),
        incomplete_search_ids=incomplete_searches[:50],
        total_unique_pmcids=len(all_pmcids),
        found_by_property=len(prop_pmcids),
        found_by_molecule=len(mol_union),
        found_by_both=len(both),
        found_by_property_only=len(prop_pmcids - mol_union),
        found_by_molecule_only=len(mol_union - prop_pmcids),
        found_by_synonym_augmentation=len(aug_pmcids),
        found_only_by_synonym_augmentation=len(aug_only_pmcids),
        n_molecules_gained_from_aug=n_mol_gained_from_aug,
        n_molecules_with_pmcids=sum(1 for m in mol_map if mol_map[m]["n_pmcids"] > 0),
        # new, honest accounting of the memory-light bounding:
        n_molecule_searches_broad_skipped=n_broad_skipped,
        registry_scope="clean_bounded",
    )
    C.write_json(C.P("data", "registry_summary.json"), summary)
    C.log("INFO", "merge", "registry[clean_bounded]: %d unique PMCIDs (property=%d molecule=%d "
          "both=%d; %d broad molecule searches skipped)"
          % (summary["total_unique_pmcids"], summary["found_by_property"],
             summary["found_by_molecule"], summary["found_by_both"], n_broad_skipped))
    return summary


def registry_pmcids():
    path = C.P("data", "registry_pmcids.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]
