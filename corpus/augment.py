"""
Synonym augmentation — ADDITIVE full-PubChem-synonym search pass.

The curated per-molecule queries (gpt + a 40-synonym supply cap) used only a
fraction of each molecule's PubChem synonyms. This pass searches EVERY remaining
synonym so no PubChem synonym is missed.

Source of truth: data/cid_synonyms_full.jsonl ({mol_id, cid, synonyms_full:[...], n})
extracted from bulk_ftp/CID-Synonym-filtered.gz (the complete set, not the truncated
cache). For each molecule, the missing terms = synonyms_full whose normalized form
(lowercase, alphanumerics only) does NOT already appear in that molecule's existing
query text. Missing terms are quoted, OR'd, and batched (<=BATCH_TERMS per search) to
respect PMC limits; default [All Fields], same as the existing molecule queries.

Outputs: data/queries_synonym_aug.jsonl, data/search_items_aug.jsonl,
state/AUG_QUERY_STATS.json, data/molecule_identities_full.jsonl (identity records
enriched with the full synonym list, so no information is lost at the storage layer).
"""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import search as S

BATCH_TERMS = 120
MEGA_THRESHOLD = 200


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _clean(term):
    t = term.replace('"', " ").strip()        # PMC quoted phrase can't hold a literal "
    return re.sub(r"\s+", " ", t)


def _existing_text_by_mol():
    exist = {}
    with open(C.P("data", "queries.jsonl")) as f:
        for line in f:
            q = json.loads(line)
            for m in q.get("molecule_ids", []):
                exist.setdefault(m, []).append(q.get("query_text", ""))
    return {m: norm(" ".join(v)) for m, v in exist.items()}


def _full_synonyms():
    d = {}
    path = C.P("data", "cid_synonyms_full.jsonl")
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            d[r["mol_id"]] = (r.get("cid"), r.get("synonyms_full") or [])
    return d


def ready():
    """Both inputs present and the extraction finished (stats written)."""
    return (os.path.exists(C.P("data", "cid_synonyms_full.jsonl")) and
            os.path.exists(C.P("data", "cid_synonyms_full.stats.json")))


def build_aug_queries():
    exist = _existing_text_by_mol()
    full = _full_synonyms()
    queries = []
    items = {}
    stats = dict(generated_at=C.now_iso(), molecules_total=len(full),
                 molecules_with_missing=0, total_missing_terms=0,
                 total_synonyms_full=0, n_searches=0, mega_molecules=[])
    for mid, (cid, syns) in sorted(full.items()):
        et = exist.get(mid, "")
        seen = set()
        missing = []
        for s in syns:
            c = _clean(s)
            n = norm(c)
            if not n or n in seen:
                continue
            seen.add(n)
            if n not in et:
                missing.append(c)
        stats["total_synonyms_full"] += len(syns)
        if not missing:
            continue
        stats["molecules_with_missing"] += 1
        stats["total_missing_terms"] += len(missing)
        mega = len(missing) > MEGA_THRESHOLD
        if mega:
            stats["mega_molecules"].append([mid, len(missing)])
        for bi in range(0, len(missing), BATCH_TERMS):
            batch = missing[bi:bi + BATCH_TERMS]
            qtext = "(" + " OR ".join('"%s"' % t for t in batch) + ")"
            v = S.validate_query(qtext)
            sid = S.search_id_for(qtext)
            qid = "%s#aug%d" % (mid, bi // BATCH_TERMS)
            queries.append(dict(
                query_id=qid, query_type="synonym_augmentation", query_text=qtext,
                purpose="full-PubChem-synonym coverage (synonyms dropped by curated queries)",
                coverage_description="exact synonym phrases not present in existing queries",
                molecule_ids=[mid], pubchem_cids=([cid] if cid else []),
                identifiers_used=batch, model="deterministic_synonym_batch",
                prompt_version=C.PROMPT_VERSION, generated_at=C.now_iso(),
                response_path=None, validation=v, search_id=sid, mega=mega))
            it = items.get(sid)
            if it is None:
                it = dict(search_id=sid, query_text=qtext, query_ids=[], molecule_ids=[],
                          query_types=["synonym_augmentation"], pubchem_cids=[],
                          valid=v["valid"], validation=v)
                items[sid] = it
            it["query_ids"].append(qid)
            if mid not in it["molecule_ids"]:
                it["molecule_ids"].append(mid)
            if cid and cid not in it["pubchem_cids"]:
                it["pubchem_cids"].append(cid)
    stats["n_searches"] = len(items)
    with open(C.P("data", "queries_synonym_aug.jsonl"), "w") as f:
        for r in queries:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(C.P("data", "search_items_aug.jsonl"), "w") as f:
        for sid in sorted(items):
            f.write(json.dumps(items[sid], ensure_ascii=False) + "\n")
    C.write_json(C.P("state", "AUG_QUERY_STATS.json"), stats)
    C.log("INFO", "augment",
          "aug queries: %d searches from %d missing terms across %d molecules (%d mega)"
          % (len(items), stats["total_missing_terms"], stats["molecules_with_missing"],
             len(stats["mega_molecules"])))
    return stats


def search_items_aug():
    items = {}
    path = C.P("data", "search_items_aug.jsonl")
    if not os.path.exists(path):
        return items
    with open(path) as f:
        for line in f:
            it = json.loads(line)
            items[it["search_id"]] = it
    return items


def write_full_synonym_identities():
    """Enrich molecule_identities with the full synonym list (no info loss)."""
    full = {}
    with open(C.P("data", "cid_synonyms_full.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            full[r["mol_id"]] = r.get("synonyms_full") or []
    out = C.P("data", "molecule_identities_full.jsonl")
    n = 0
    with open(C.P("data", "molecule_identities.jsonl")) as fin, open(out, "w") as fo:
        for line in fin:
            r = json.loads(line)
            fs = full.get(r["mol_id"])
            if fs is not None:
                r["synonyms_full"] = fs
                r["n_synonyms_full"] = len(fs)
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    C.log("INFO", "augment", "wrote molecule_identities_full.jsonl (%d records)" % n)
    return n
