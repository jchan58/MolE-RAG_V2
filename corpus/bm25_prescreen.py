"""
exp70 BM25 pre-screen (NO LLM spend). Builds a temporary BM25 index over ALL
paragraph+table passages (node-local /tmp), then retrieves candidate passages via
two routes and writes a deduped candidate set with full provenance. STOPS at the
candidate set + recall sample-check; the Opus 4.6 judge is a separate, gated step.

Routes:
  PROPERTY  : expanded topic queries (4 topics), top-K per query string.
  MOLECULE  : for each of the 1975 molecules, EVERY verified alias
              (preferred_name + iupac_name + synonyms + good_syns + synonyms_full
               + dev_codes), each alias searched SEPARATELY (top-k per alias),
              merged per-molecule, top-K per molecule. Aliases are globally
              de-duplicated (queried once, mapped back to all owning molecules).

Outputs under /tmp/exp70_filter/bm25/ (+ synced to s3://molerag/exp70_chunks/candidates/):
  candidates_passages.jsonl   full passage record + provenance for each candidate
  candidate_provenance.jsonl  per (passage, route, topic/alias, molecule_id, score)
  prescreen_summary.json      query terms, K values, counts, reduction, timings
  recall_sample.jsonl         sampled non-candidate passages for recall eyeballing
"""
import os, sys, json, time, glob, pickle, gzip, re, threading, subprocess
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import datasets as DS

BASE = os.environ.get("EXP70_FILTER_BASE", "/tmp/exp70_filter")
MOL_TREE = os.path.join(BASE, "chunks", "passages")
PROP_TREE = os.path.join(BASE, "chunks_prop", "passages")
OUT = os.path.join(BASE, "bm25")
IDX = os.path.join(OUT, "index")
PROG = os.path.join(OUT, "PRESCREEN_PROGRESS.json")
# per-dataset identities (full synonyms preferred, else base identities)
_IDF = C.P("data", "molecule_identities_full.jsonl")
IDENT = _IDF if os.path.exists(_IDF) else C.P("data", "molecule_identities.jsonl")
S3_DEST = os.environ.get("EXP70_CAND_S3", "s3://molerag/%s_chunks/candidates/" % DS.CFG["name"])

# Tunables (env-overridable) -------------------------------------------------
K_PROP = int(os.environ.get("EXP70_K_PROP", "2000"))      # top-K per property query string
K_MOL_ALIAS = int(os.environ.get("EXP70_K_MOL_ALIAS", "20"))  # top-k per alias query
K_MOL_MOL = int(os.environ.get("EXP70_K_MOL_MOL", "40"))  # top-K per molecule after merge
TEXT_CAP = int(os.environ.get("EXP70_TOK_CAP", "20000"))  # cap chars fed to tokenizer (huge tables)
READ_WORKERS = int(os.environ.get("EXP70_READ_WORKERS", "64"))
NT = int(os.environ.get("EXP70_RETRIEVE_THREADS", str(max(1, (os.cpu_count() or 8)))))
MIN_ALIAS_LEN = 3

# Expanded PROPERTY topic queries (REPORTED verbatim before any spend). Per-dataset,
# sourced from datasets.py so the prescreen topic set matches the dataset's property.
PROPERTY_TOPICS = DS.property_topics()


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


_GROUP_CODE = {(): 0, ("molecule",): 1, ("property",): 2, ("molecule", "property"): 3}


def _read_shard(files):
    """Worker: read a list of passage files -> list of (pid,pmcid,grp,ptype,text)."""
    out = []
    for f in files:
        try:
            with open(f) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    pid = d.get("passage_id")
                    if not pid:
                        continue
                    g = tuple(sorted(d.get("document_groups") or []))
                    grp = _GROUP_CODE.get(g, 0)
                    ptype = 1 if d.get("passage_type") == "table" else 0
                    out.append((pid, d.get("pmcid"), grp, ptype, (d.get("text") or "")[:TEXT_CAP]))
        except Exception:
            continue
    return out


def build_pmcid_path_map():
    """pmcid -> source file path (prefer MOL_TREE; it holds the article's full passage set)."""
    m = {}
    for tree in (PROP_TREE, MOL_TREE):   # MOL_TREE last so it wins
        for f in glob.glob(os.path.join(tree, "*", "*.jsonl")):
            m[os.path.basename(f)[:-6]] = f
    return m


def build_corpus():
    files = glob.glob(os.path.join(MOL_TREE, "*", "*.jsonl")) + \
            glob.glob(os.path.join(PROP_TREE, "*", "*.jsonl"))
    _cap = int(os.environ.get("EXP70_MAX_FILES", "0"))
    if _cap:
        files = files[:_cap]
        log("SMOKE: capped to %d files" % _cap)
    log("corpus: %d passage files across both trees" % len(files))
    # shard files for workers
    nsh = READ_WORKERS * 4
    shards = [files[i::nsh] for i in range(nsh)]
    pid_seen = set()
    pids, pmcids, grp, ptype, texts = [], [], [], [], []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=READ_WORKERS) as ex:
        for res in ex.map(_read_shard, shards):
            for (pid, pmcid, g, pt, tx) in res:
                if pid in pid_seen:
                    continue
                pid_seen.add(pid)
                pids.append(pid); pmcids.append(pmcid)
                grp.append(g); ptype.append(pt); texts.append(tx)
            done += 1
            if done % 32 == 0:
                log("  corpus read %d/%d shards, %d passages, %.0fs" % (done, nsh, len(pids), time.time() - t0))
    log("corpus built: %d unique passages in %.0fs" % (len(pids), time.time() - t0))
    return pids, np.array(pmcids, dtype=object), np.array(grp, dtype=np.int8), np.array(ptype, dtype=np.int8), texts


def tokenize_corpus(texts):
    import bm25s, Stemmer
    st = Stemmer.Stemmer("english")
    t0 = time.time()
    log("tokenizing %d passages (single-thread, ~12min for 8M)..." % len(texts))
    toks = bm25s.tokenize(texts, stopwords="en", stemmer=st, return_ids=False, show_progress=True)
    log("tokenized in %.0fs" % (time.time() - t0))
    return toks


def build_index(toks):
    import bm25s
    t0 = time.time()
    r = bm25s.BM25()
    r.index(toks, show_progress=True)
    log("BM25 index built in %.0fs" % (time.time() - t0))
    return r


ARRAYS = os.path.join(OUT, "arrays.pkl")


def save_index(r, pids, pmcids, grp, ptype):
    """Persist index + passage arrays so retrieval can be re-run without re-tokenizing."""
    try:
        r.save(IDX)
        with open(ARRAYS, "wb") as f:
            pickle.dump(dict(pids=pids, pmcids=pmcids, grp=grp, ptype=ptype), f, protocol=4)
        log("saved index -> %s and arrays -> %s" % (IDX, ARRAYS))
    except Exception as e:
        log("save_index failed (non-fatal): %r" % e)


def load_saved():
    """Load persisted index (mmap) + arrays for a retrieval-only rerun. Returns
    (r, pids, pmcids, grp, ptype) or None if not present."""
    import bm25s
    if not (os.path.isdir(IDX) and os.path.exists(ARRAYS)):
        return None
    t0 = time.time()
    r = bm25s.BM25.load(IDX, mmap=True, load_corpus=False)
    with open(ARRAYS, "rb") as f:
        a = pickle.load(f)
    log("loaded saved index + arrays in %.0fs (%d passages)" % (time.time() - t0, len(a["pids"])))
    return r, a["pids"], a["pmcids"], a["grp"], a["ptype"]


# ---- alias hygiene (fixes: semicolon mega-queries + encoding mojibake) -----
_MOJI_MARKERS = ("inverted exclamation", "exclamation mark", "i<<", "<<", ">>", "|",
                 "�", "cedilla", "acute accent", "grave accent", "diaeresis",
                 "circumflex accent")
# Ultra-generic names that would match nearly everything (belt; DF-filter is the main guard).
GENERIC_NAMES = {
    "water", "ethanol", "methanol", "acetone", "benzene", "toluene", "chloroform",
    "glucose", "sucrose", "fructose", "lactose", "oxygen", "nitrogen", "hydrogen",
    "carbon dioxide", "acid", "base", "salt", "alcohol", "sugar", "oil", "gas", "ice",
    "dye", "drug", "compound", "control", "sample", "standard", "buffer", "saline",
    "medium", "solvent", "air", "protein", "peptide", "lipid", "amino acid", "dmso",
    "vitamin", "vitamin a", "vitamin b", "vitamin c", "vitamin d", "vitamin e", "vitamin k",
    "sodium chloride", "pbs", "nacl", "urea", "glycerol", "starch", "cellulose",
}


def _clean_alias(a):
    """Return a cleaned alias or None (drops mojibake / too-short / junk)."""
    if not a:
        return None
    a = a.strip().strip("…").strip()   # strip ellipsis
    if any(ord(c) < 32 for c in a):         # control chars
        return None
    low = a.lower()
    for m in _MOJI_MARKERS:                 # mojibake signatures
        if m in low:
            return None
    alnum = sum(1 for c in a if c.isalnum())
    if alnum < MIN_ALIAS_LEN:
        return None
    ok = sum(1 for c in a if c.isalnum() or c in " -(),.+/'[]")
    if len(a) and ok / len(a) < 0.6:        # too many weird chars
        return None
    return a


def _split_and_clean(raw):
    """Split a raw alias field on ';' / newlines into SEPARATE cleaned aliases."""
    out = []
    for p in re.split(r"[;\n\r]+", raw):
        c = _clean_alias(p)
        if c:
            out.append(c)
    return out


def load_molecule_aliases():
    """Return (mols list of dict, alias_map: alias_lower -> list[mol_idx]).
    Each raw alias field is split on ';'/newlines so a long synonym list is NEVER one
    query; mojibake/junk dropped; ultra-generic names removed (DF-filter adds more)."""
    mols = []
    alias_map = {}
    for line in open(IDENT):
        d = json.loads(line)
        idx = len(mols)
        aset = set()
        for f in ("preferred_name", "iupac_name"):
            v = d.get(f)
            if v and isinstance(v, str):
                aset.update(_split_and_clean(v))
        for f in ("synonyms", "good_syns", "synonyms_full", "dev_codes"):
            for v in (d.get(f) or []):
                if v and isinstance(v, str):
                    aset.update(_split_and_clean(v))
        aset = {a for a in aset if a.lower() not in GENERIC_NAMES}
        mols.append(dict(mol_id=d.get("mol_id"), preferred_name=d.get("preferred_name"),
                         iupac_name=d.get("iupac_name"), canonical_smiles=d.get("canonical_smiles"),
                         inchikey=d.get("inchikey_rdkit"), n_aliases=len(aset)))
        for a in aset:
            alias_map.setdefault(a.lower(), []).append(idx)
    return mols, alias_map


def retrieve_property(r, npass):
    """top-K per query string; return dict passage_idx -> list of (topic, qstr, score)."""
    import bm25s, Stemmer
    st = Stemmer.Stemmer("english")
    qstrings, qmeta = [], []
    for topic, qs in PROPERTY_TOPICS.items():
        for q in qs:
            qstrings.append(q); qmeta.append((topic, q))
    qtok = bm25s.tokenize(qstrings, stopwords="en", stemmer=st, return_ids=False, show_progress=False)
    k = min(K_PROP, npass)
    t0 = time.time()
    res, sco = r.retrieve(qtok, k=k, n_threads=NT, show_progress=True, return_as="tuple")
    log("property: %d queries x k=%d in %.0fs" % (len(qstrings), k, time.time() - t0))
    prov = {}
    for qi in range(len(qstrings)):
        topic, q = qmeta[qi]
        for j in range(res.shape[1]):
            di = int(res[qi, j]); s = float(sco[qi, j])
            if s <= 0:
                continue
            prov.setdefault(di, []).append(("property:" + topic, q, round(s, 3)))
    return prov, qmeta


DF_MAX_FRAC = float(os.environ.get("EXP70_DF_MAX_FRAC", "0.02"))  # alias dropped if its RAREST token is in > this fraction of docs


def _token_df(r):
    """Return (vocab_dict, df_array, num_docs) from the bm25s CSC index."""
    sc = r.scores
    df = np.diff(np.asarray(sc["indptr"]))
    return r.vocab_dict, df, int(sc["num_docs"])


def retrieve_molecule(r, mols, alias_map, npass):
    """Query each unique CLEANED alias SEPARATELY (top-k), map hits back to owning
    molecules, keep top-K per molecule by score. Generic aliases (rarest token common
    in the corpus) are dropped before querying. Returns
    dict passage_idx -> list of (molecule_id, alias, score), plus a stats dict."""
    import bm25s, Stemmer
    st = Stemmer.Stemmer("english")
    aliases = list(alias_map.keys())
    atoks = bm25s.tokenize(aliases, stopwords="en", stemmer=st, return_ids=False, show_progress=False)
    vocab, df, ndoc = _token_df(r)
    df_max = DF_MAX_FRAC * ndoc
    kept_idx, dropped_generic, dropped_empty = [], 0, 0
    for i, toks in enumerate(atoks):
        ids = [vocab[t] for t in toks if t in vocab]
        if not ids:
            dropped_empty += 1; continue           # no corpus token -> can't match
        if min(df[j] for j in ids) > df_max:
            dropped_generic += 1; continue          # even rarest token is generic
        kept_idx.append(i)
    log("molecule: %d unique aliases; kept %d, dropped %d generic + %d empty/OOV (df_max=%.0f docs)"
        % (len(aliases), len(kept_idx), dropped_generic, dropped_empty, df_max))
    kept_aliases = [aliases[i] for i in kept_idx]
    kept_toks = [atoks[i] for i in kept_idx]
    k = min(K_MOL_ALIAS, npass)
    t0 = time.time()
    res, sco = r.retrieve(kept_toks, k=k, n_threads=NT, show_progress=True, return_as="tuple")
    log("molecule: %d alias-queries x k=%d retrieved in %.0fs" % (len(kept_aliases), k, time.time() - t0))
    permol = {}
    for ai, alias in enumerate(kept_aliases):
        owners = alias_map[alias]
        row_d = res[ai]; row_s = sco[ai]
        for j in range(row_d.shape[0]):
            s = float(row_s[j])
            if s <= 0:
                continue
            di = int(row_d[j])
            for m in owners:
                dd = permol.setdefault(m, {})
                cur = dd.get(di)
                if cur is None or s > cur[0]:
                    dd[di] = (s, alias)
    prov = {}
    for m, dd in permol.items():
        top = sorted(dd.items(), key=lambda kv: -kv[1][0])[:K_MOL_MOL]
        mid = mols[m]["mol_id"]
        for di, (s, alias) in top:
            prov.setdefault(di, []).append((mid, alias, round(s, 3)))
    stats = dict(unique_aliases=len(aliases), queried=len(kept_aliases),
                 dropped_generic=dropped_generic, dropped_empty=dropped_empty,
                 df_max_docs=round(df_max), df_max_frac=DF_MAX_FRAC)
    return prov, stats


def main():
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    _write_json(PROG, dict(stage="start", at=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    texts = None
    loaded = load_saved() if os.environ.get("EXP70_LOAD_INDEX") == "1" else None
    if loaded is not None:
        r, pids, pmcids, grp, ptype = loaded
        npass = len(pids)
        log("RETRIEVE-ONLY mode: reusing saved index (%d passages)" % npass)
    else:
        pids, pmcids, grp, ptype, texts = build_corpus()
        npass = len(pids)
        _write_json(PROG, dict(stage="corpus_done", n_passages=npass,
                               elapsed_s=round(time.time() - t_start)))
        toks = tokenize_corpus(texts)
        r = build_index(toks)
        del toks
        save_index(r, pids, pmcids, grp, ptype)
    _write_json(PROG, dict(stage="index_done", n_passages=npass,
                           elapsed_s=round(time.time() - t_start)))

    prop_prov, qmeta = retrieve_property(r, npass)
    mols, alias_map = load_molecule_aliases()
    total_alias_slots = sum(len(v) for v in alias_map.values())
    mol_prov, mol_stats = retrieve_molecule(r, mols, alias_map, npass)

    # merge candidate passage indices
    cand = set(prop_prov) | set(mol_prov)
    log("candidates: property=%d molecule=%d union=%d (of %d passages)" %
        (len(prop_prov), len(mol_prov), len(cand), npass))

    # provenance + counts
    prop_cand = set(prop_prov); mol_cand = set(mol_prov)
    per_topic = {}
    for di, lst in prop_prov.items():
        for (topic, q, s) in lst:
            per_topic[topic] = per_topic.get(topic, 0) + 1
    # per-topic counts = number of candidate passages touched by that topic (passages may repeat across topics)
    per_mol_counts = {}
    for di, lst in mol_prov.items():
        for (mid, alias, s) in lst:
            per_mol_counts[mid] = per_mol_counts.get(mid, 0) + 1

    # materialize candidate passage records by re-reading their source files (full metadata for review/judge)
    cand_pmcids = set(str(pmcids[di]) for di in cand)
    log("materializing %d candidate passages from %d source articles" % (len(cand), len(cand_pmcids)))
    want_pids = set(pids[di] for di in cand)
    pid_to_prov = {}
    for di in cand:
        pid = pids[di]
        pid_to_prov[pid] = dict(
            property=[dict(topic=t.split(":", 1)[1], query=q, score=s) for (t, q, s) in prop_prov.get(di, [])],
            molecule=[dict(molecule_id=m, alias=a, score=s) for (m, a, s) in mol_prov.get(di, [])],
        )
    path_map = build_pmcid_path_map()
    _materialize(cand_pmcids, want_pids, pid_to_prov, path_map)

    # recall sample-check: sample non-candidate passages (with text snippet) for eyeballing
    rng = np.random.default_rng(70)
    noncand = np.array([i for i in range(npass) if i not in cand])
    samp = rng.choice(noncand, size=min(400, len(noncand)), replace=False) if len(noncand) else np.array([], dtype=int)
    _recall_sample(samp, pids, pmcids, grp, texts)

    summary = dict(
        experiment="exp70_bm25_prescreen",
        built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        total_elapsed_s=round(time.time() - t_start),
        n_passages_indexed=npass,
        text_cap_chars=TEXT_CAP,
        property_topics={t: qs for t, qs in PROPERTY_TOPICS.items()},
        property_K_per_query=K_PROP,
        molecule_policy=dict(
            fields=["preferred_name", "iupac_name", "synonyms", "good_syns", "synonyms_full", "dev_codes"],
            split_on=["semicolon", "newline"], min_alias_len=MIN_ALIAS_LEN,
            mojibake_cleaning=True, generic_name_stoplist=len(GENERIC_NAMES),
            df_generic_filter=mol_stats,
            n_molecules=len(mols),
            n_unique_aliases=len(alias_map),
            n_aliases_queried=mol_stats["queried"],
            total_alias_slots=total_alias_slots,
            k_per_alias=K_MOL_ALIAS, K_per_molecule=K_MOL_MOL,
        ),
        candidates=dict(
            property_passages=len(prop_cand),
            molecule_passages=len(mol_cand),
            union_passages=len(cand),
            both_routes=len(prop_cand & mol_cand),
            source_articles=len(cand_pmcids),
            reduction_ratio=round(npass / max(len(cand), 1), 1),
        ),
        per_property_topic_passage_hits=per_topic,
        molecules_with_hits=len(per_mol_counts),
        molecules_with_zero_hits=len(mols) - len(per_mol_counts),
    )
    _write_json(os.path.join(OUT, "prescreen_summary.json"), summary)
    _write_json(PROG, dict(stage="done", elapsed_s=round(time.time() - t_start),
                           union_candidates=len(cand)))
    log("SUMMARY: %s" % json.dumps(summary["candidates"]))
    _sync_s3()
    log("DONE in %.0fs. candidates union=%d" % (time.time() - t_start, len(cand)))


def _materialize(cand_pmcids, want_pids, pid_to_prov, path_map):
    """Re-read source files for candidate articles; write full records for candidate passages."""
    outp = os.path.join(OUT, "candidates_passages.jsonl")
    provp = os.path.join(OUT, "candidate_provenance.jsonl")
    fo = open(outp, "w"); fp = open(provp, "w")
    n = 0
    for pmcid in cand_pmcids:
        cand_file = path_map.get(pmcid)
        if not cand_file:
            continue
        try:
            with open(cand_file) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    pid = d.get("passage_id")
                    if pid not in want_pids:
                        continue
                    prov = pid_to_prov.get(pid, {})
                    rec = dict(
                        passage_id=pid, pmcid=d.get("pmcid"), passage_type=d.get("passage_type"),
                        document_groups=d.get("document_groups"), section_path=d.get("section_path"),
                        heading=d.get("heading"), article_title=d.get("article_title"),
                        journal=d.get("journal"), doi=d.get("doi"), local_index=d.get("local_index"),
                        parent_table_id=d.get("parent_table_id"),
                        molecule_associations=d.get("molecule_associations"),
                        text=d.get("text"),
                        retrieval=dict(property=prov.get("property", []), molecule=prov.get("molecule", [])),
                    )
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fp.write(json.dumps(dict(passage_id=pid, pmcid=d.get("pmcid"),
                                             property=prov.get("property", []),
                                             molecule=prov.get("molecule", []))) + "\n")
                    n += 1
        except Exception:
            continue
    fo.close(); fp.close()
    log("materialized %d candidate passage records -> %s" % (n, outp))


def _recall_sample(samp, pids, pmcids, grp, texts):
    """Write a sample of non-candidate passages (with text snippet when available)."""
    import json as _j
    outp = os.path.join(OUT, "recall_sample.jsonl")
    with open(outp, "w") as f:
        for di in samp.tolist():
            rec = dict(passage_id=pids[di], pmcid=str(pmcids[di]), group_code=int(grp[di]))
            if texts is not None:
                rec["text"] = (texts[di] or "")[:600]
            f.write(_j.dumps(rec, ensure_ascii=False) + "\n")
    log("wrote %d non-candidate sample passages -> %s" % (len(samp), outp))


def _sync_s3():
    try:
        subprocess.run(["aws", "s3", "sync", OUT, S3_DEST, "--only-show-errors",
                        "--exclude", "index/*"], timeout=1200, check=False)
        log("synced candidates -> %s" % S3_DEST)
    except Exception as e:
        log("s3 sync failed: %r" % e)


if __name__ == "__main__":
    main()
