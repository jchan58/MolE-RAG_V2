"""
Structuring steps 4-9 — BM25 candidate pool (<=1000/query over KEPT passages) ->
MedCPT cross-encoder rerank (GPU, once) -> nested K200/K400/K600/K800/K1000 shortlists.

Runs on a GPU box. Reads the dataset's KEPT passages (judge retention keep/review joined
to passage text) + the query registry, retrieves up to K_BM25 (default 1000) BM25
candidates per query, reranks (query, passage) pairs with ncbi/MedCPT-Cross-Encoder, ranks
WITHIN each query, and writes rankings + nested shortlists at every band in
card_schema.KBANDS. Pushes everything to S3. No LLM API / no Bedrock token needed.

The band set lives in card_schema.KBANDS (single source of truth; cohere_rerank.py imports
KBANDS/shortlists from here, so the two rerankers can never disagree). A short candidate
pool (e.g. a sparse query) simply yields fewer entries in each band — top-k slicing never
fabricates passages, and deeper bands legitimately report 0 when the pool is shallow.

Env: EXP70_DATASET, EXP70_FILTER_BASE (kept passages), EXP70_RUN_DIR (query registry),
     EXP70_STRUCT_S3 (default s3://molerag/<ds>_chunks/structuring/), EXP70_K_BM25.
"""
import os, sys, json, glob, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import datasets as DS
import card_schema as CS   # canonical KBANDS (shared by all structuring stages)

BASE = os.environ.get("EXP70_FILTER_BASE", "/tmp/exp70_%s" % DS.CFG["name"])
PASS_TREE = os.path.join(BASE, "chunks", "passages")
JOUT = os.path.join(BASE, "judgments_qwen")
# CANONICAL structuring output dir = RUN_DIR/structuring (on /work, persistent). extract.py,
# build_indexes.py and build_cards.py all read+write structuring outputs HERE, so the rerankers
# MUST write shortlists here too (the old BASE/structuring put shortlists on a different node's
# /tmp -> extract read an empty RUN_DIR/structuring -> empty union -> hollow cards).
OUT = C.P("structuring")
S3 = os.environ.get("EXP70_STRUCT_S3", "s3://molerag/%s_chunks/structuring/" % DS.CFG["name"])
# S3 home of the judge output + the chunked-passages tar (staged into FILTER_BASE on a fresh
# node). The JUDGE scored the CHUNK TREE (passages_<ds>.tar.gz -> BASE/chunks/passages), so the
# retention passage_ids are chunk-tree ids and the chunk tree is the authoritative text source
# (NOT candidates_passages.jsonl, which is the bm25 prescreen subset with disjoint ids).
S3_CHUNKS = "s3://molerag/%s_chunks/" % DS.CFG["name"]
PASS_TAR = os.path.join(BASE, "passages_%s.tar.gz" % DS.CFG["name"])
# Candidate-pool depth over KEPT passages. Must be >= max(KBANDS) so the deepest band (K1000
# for the property/task route, whose kept-pool is deep) can actually fill; a shallow pool
# just yields fewer per band (no fabrication).
K_BM25 = int(os.environ.get("EXP70_K_BM25", "1000"))
KBANDS = CS.KBANDS
MEDCPT_MODEL = os.environ.get("EXP70_MEDCPT", "ncbi/MedCPT-Cross-Encoder")
RERANK_BATCH = int(os.environ.get("EXP70_RERANK_BATCH", "64"))


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def _ensure_inputs_staged():
    """A fresh/clean structuring node has neither the judge retention nor the chunked passages
    in FILTER_BASE; pull them from S3. Retention ~tens of MB (sync). Passages tar ~1.6GB ->
    extracted to BASE/chunks/passages (= PASS_TREE), the judge's own text source. Idempotent;
    the tar is deleted after a successful extract to spare node-local disk."""
    ret_dir = os.path.join(JOUT, "retention")
    if not glob.glob(os.path.join(ret_dir, "*", "*.jsonl")):
        os.makedirs(ret_dir, exist_ok=True)
        subprocess.run(["aws", "s3", "sync", S3_CHUNKS + "qwen_judgments/retention/", ret_dir,
                        "--only-show-errors"], timeout=1800, check=False)
    if not glob.glob(os.path.join(PASS_TREE, "*", "*.jsonl")):
        chunks_dir = os.path.join(BASE, "chunks")   # tar's top-level is passages/ -> lands at PASS_TREE
        os.makedirs(chunks_dir, exist_ok=True)
        if not (os.path.exists(PASS_TAR) and os.path.getsize(PASS_TAR) > 0):
            subprocess.run(["aws", "s3", "cp",
                            S3_CHUNKS + "passages/passages_%s.tar.gz" % DS.CFG["name"],
                            PASS_TAR, "--only-show-errors"], timeout=7200, check=False)
        rc = subprocess.run(["tar", "xzf", PASS_TAR, "-C", chunks_dir], timeout=7200, check=False)
        if rc.returncode == 0 and glob.glob(os.path.join(PASS_TREE, "*", "*.jsonl")):
            try:
                os.remove(PASS_TAR)   # reclaim ~1.6GB of node-local disk once extracted
            except OSError:
                pass
    return ret_dir


def _iter_tree_passages():
    """Yield each chunk-tree passage record (text + metadata), streamed file-by-file."""
    for fp in glob.glob(os.path.join(PASS_TREE, "*", "*.jsonl")):
        with open(fp) as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass


def candidate_passage_map(want_ids):
    """{passage_id: full passage record} for the requested ids, from the chunk tree (staged
    from the S3 passages tar if a fresh node has none). Shared by extract.py so extraction
    reads the SAME text source the judge scored."""
    _ensure_inputs_staged()
    want = set(want_ids)
    m = {}
    for d in _iter_tree_passages():
        pid = d.get("passage_id")
        if pid in want and pid not in m:
            m[pid] = d
    return m


def load_kept_passages():
    """Return {passage_id: {text, pmcid, article_title, section_path, document_groups,
    molecule_associations, retention}} for the judge-retained (keep/review) passages. Stages
    the judge retention + the chunked-passages tar from S3 into FILTER_BASE first, so it works
    on a fresh structuring node that never saw the collection/judge node's /tmp."""
    ret_dir = _ensure_inputs_staged()
    kept_ids = {}   # passage_id -> list of (group, decision)
    for fp in glob.glob(os.path.join(ret_dir, "*", "*.jsonl")):
        with open(fp) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("decision") in ("keep", "review"):
                    kept_ids.setdefault(r["passage_id"], []).append((r.get("group"), r.get("decision")))
    log("kept passage_ids (keep/review): %d" % len(kept_ids))
    passages = {}
    for d in _iter_tree_passages():
        pid = d.get("passage_id")
        if pid in kept_ids and pid not in passages:
            passages[pid] = dict(passage_id=pid, text=d.get("text") or "", pmcid=d.get("pmcid"),
                                 article_title=d.get("article_title"), section_path=d.get("section_path"),
                                 document_groups=d.get("document_groups"),
                                 molecule_associations=d.get("molecule_associations"),
                                 retention=kept_ids[pid])
    log("kept passages with text: %d" % len(passages))
    return passages


def bm25_candidates(passages, queries):
    """For each query (bm25_query), top-K_BM25 kept passages. Returns {query_id: [pid,...]}."""
    import bm25s, Stemmer
    st = Stemmer.Stemmer("english")
    pids = list(passages.keys())
    texts = [passages[p]["text"][:20000] for p in pids]
    toks = bm25s.tokenize(texts, stopwords="en", stemmer=st, show_progress=False)
    r = bm25s.BM25(); r.index(toks, show_progress=False)
    qtok = bm25s.tokenize([q["bm25_query"] for q in queries], stopwords="en", stemmer=st, show_progress=False)
    k = min(K_BM25, len(pids))
    res, sco = r.retrieve(qtok, k=k, show_progress=False, return_as="tuple")
    out = {}
    for qi, q in enumerate(queries):
        cand = [(pids[int(res[qi, j])], float(sco[qi, j])) for j in range(res.shape[1]) if sco[qi, j] > 0]
        out[q["query_id"]] = cand
    return out


def medcpt_rerank(passages, queries, cand):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("loading MedCPT on %s" % dev)
    tok = AutoTokenizer.from_pretrained(MEDCPT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MEDCPT_MODEL).to(dev).eval()
    rankings = {}
    for q in queries:
        qid = q["query_id"]; sem = q["semantic_query"]
        pairs = cand.get(qid, [])
        if not pairs:
            rankings[qid] = []
            continue
        scores = []
        for i in range(0, len(pairs), RERANK_BATCH):
            chunk = pairs[i:i + RERANK_BATCH]
            arts = []
            for pid, _bs in chunk:
                p = passages[pid]
                title = (p.get("article_title") or "")[:200]
                arts.append("%s. %s" % (title, p["text"][:1500]))
            enc = tok([[sem, a] for a in arts], truncation=True, padding=True,
                      max_length=512, return_tensors="pt").to(dev)
            with torch.no_grad():
                logits = model(**enc).logits.squeeze(-1)
            scores.extend(logits.tolist())
        ranked = sorted(zip([p for p, _ in pairs], scores,
                            [bs for _, bs in pairs]), key=lambda x: -x[1])
        rankings[qid] = [dict(passage_id=p, medcpt_score=round(s, 4), medcpt_rank=i + 1, bm25_score=round(bs, 3))
                         for i, (p, s, bs) in enumerate(ranked)]
    return rankings


def shortlists(rankings):
    """Nested shortlists per query (one list per band in KBANDS) + merged unique passage
    sets per band. Top-k slicing (ranked[:k]) means a short pool yields fewer entries and
    never fabricates passages; deeper bands report only what the pool actually reached."""
    out = {}
    merged = {k: set() for k in KBANDS}
    for qid, ranked in rankings.items():
        out[qid] = {}
        for k in KBANDS:
            ids = [r["passage_id"] for r in ranked[:k]]
            out[qid]["K%d" % k] = ids
            merged[k].update(ids)
    return out, {("K%d" % k): sorted(v) for k, v in merged.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    # query registry (task route) from RUN_DIR
    qreg_path = C.P("structuring", "query_registry.jsonl")
    if not os.path.exists(qreg_path):
        import query_registry as QR
        QR.build()
    queries = [json.loads(l) for l in open(qreg_path) if l.strip()]
    log("task-route queries: %d" % len(queries))
    passages = load_kept_passages()
    if not passages:
        log("NO kept passages yet (judge not done?) — aborting rerank"); return
    cand = bm25_candidates(passages, queries)
    log("BM25 pools built (<=%d/query)" % K_BM25)
    rankings = medcpt_rerank(passages, queries, cand)
    sl, merged = shortlists(rankings)
    # write outputs
    with open(os.path.join(OUT, "medcpt_rankings.jsonl"), "w") as f:
        for qid, ranked in rankings.items():
            f.write(json.dumps(dict(query_id=qid, ranked=ranked), ensure_ascii=False) + "\n")
    C.write_json(os.path.join(OUT, "shortlists.json"), sl)
    summary = dict(dataset=DS.CFG["name"], built_at=C.now_iso(), model=MEDCPT_MODEL,
                   n_queries=len(queries), n_kept_passages=len(passages), k_bm25=K_BM25,
                   unique_passages_per_band={k: len(v) for k, v in merged.items()},
                   elapsed_s=round(time.time() - t0))
    C.write_json(os.path.join(OUT, "rerank_summary.json"), summary)
    subprocess.run(["aws", "s3", "sync", OUT, S3, "--only-show-errors"], timeout=1800, check=False)
    log("DONE rerank: %s" % json.dumps(summary["unique_passages_per_band"]))


if __name__ == "__main__":
    main()
