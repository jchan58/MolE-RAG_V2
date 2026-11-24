"""
Structuring steps 4-9, GPU-FREE — BM25 candidate pool (<=K_BM25, default 1000, per query
over KEPT passages) -> Cohere embeddings (Bedrock cohere.embed-english-v3) -> COSINE rerank
to full pool depth -> nested K200/K400/K600/K800/K1000 shortlists. Runs on Delta (Bedrock
token local). Drop-in replacement for medcpt_rerank.py; the shortlist ranking column is
labeled "cohere_cosine".

The band set (KBANDS) and the candidate-pool helpers (bm25_candidates / shortlists / K_BM25)
are imported from medcpt_rerank, which in turn takes KBANDS from card_schema — so the cohere
and MedCPT rerankers share ONE band definition and can never drift. A short pool just yields
fewer entries per band (no fabrication); the K1000 union is what extract.py consumes.

Embeddings are cached per passage_id (model+version) under OUT/embed_cache + synced to S3.
Env: EXP70_DATASET, EXP70_FILTER_BASE, EXP70_RUN_DIR, EXP70_STRUCT_S3, EXP70_K_BM25.
"""
import os, sys, json, time, math, glob, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import datasets as DS
import medcpt_rerank as MR     # reuse load_kept_passages / bm25_candidates / shortlists
import cohere_embed as CE

OUT = MR.OUT
S3 = MR.S3
KBANDS = MR.KBANDS
# embed_cache can be ~100s of MB -> keep it on node-local /tmp (FILTER_BASE), NOT on the
# over-quota /work RUN_DIR where OUT now lives. It is synced to S3 separately.
EMB_CACHE = os.path.join(MR.BASE, "embed_cache")
RANKER = "cohere_cosine"
# selected per dataset in main() (v3 primary / v4 fallback — NEVER mixed within a dataset)
_MODEL = CE.MODEL_V3
_DIM = CE.MODEL_DIM[CE.MODEL_V3]


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def _mtag():
    return _MODEL.replace(":", "_").replace(".", "_")


def _cache_path(pid):
    # namespace cache by MODEL so a v4 fallback never reuses v3 vectors (different space)
    return os.path.join(EMB_CACHE, _mtag(), pid[:3] if len(pid) >= 3 else "x", pid + ".json")


def _load_cached(pids):
    out = {}
    for p in pids:
        cp = _cache_path(p)
        if os.path.exists(cp):
            try:
                out[p] = json.load(open(cp))["v"]
            except Exception:
                pass
    return out


def _save_cached(pid, vec):
    cp = _cache_path(pid)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    tmp = cp + ".tmp"
    json.dump({"model": _MODEL, "dim": _DIM, "v": vec}, open(tmp, "w"))
    os.replace(tmp, cp)


def embed_passages(passages, need_ids):
    """Embed (search_document) the candidate passages that aren't cached yet."""
    cached = _load_cached(need_ids)
    todo = [p for p in need_ids if p not in cached]
    log("passage embeddings: %d cached, %d to embed" % (len(cached), len(todo)))
    texts = []
    for p in todo:
        pa = passages[p]
        _t = pa.get("article_title")
        title = (_t if isinstance(_t, str) else "")[:150]   # some passages carry a dict/non-str title
        _txt = pa.get("text")
        _txt = _txt if isinstance(_txt, str) else ("" if _txt is None else str(_txt))
        texts.append(("%s. %s" % (title, _txt))[:2000])
    vecs = CE.embed_many(texts, "search_document", model=_MODEL, log=log)
    for p, v in zip(todo, vecs):
        if v and len(v) == _DIM:
            cached[p] = v
            _save_cached(p, v)
    return cached


def _cos(a, b):
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return s / (na * nb) if na and nb else 0.0


def main():
    global _MODEL, _DIM
    os.makedirs(OUT, exist_ok=True); os.makedirs(EMB_CACHE, exist_ok=True)
    t0 = time.time()
    # pick ONE embedding model for this dataset (v3 primary, v4 fallback if v3 throttles)
    _MODEL, _DIM = CE.select_model()
    log("embedding model for this dataset: %s (dim=%d)" % (_MODEL, _DIM))
    qreg = C.P("structuring", "query_registry.jsonl")
    if not os.path.exists(qreg):
        import query_registry as QR; QR.build()
    queries = [json.loads(l) for l in open(qreg) if l.strip()]
    passages = MR.load_kept_passages()
    if not passages:
        # Distinguish "judge not done yet" (legit, exit 0) from "judge DONE but we loaded
        # nothing" (a staging/plumbing bug). In the latter case HARD-FAIL so the structuring
        # chain stops and no empty shortlists / hollow STRUCT_DONE are produced.
        judged = bool(glob.glob(os.path.join(MR.JOUT, "retention", "*", "*.jsonl")))
        if not judged:
            try:
                r = subprocess.run(["aws", "s3", "ls", MR.S3_CHUNKS + "qwen_judgments/retention/"],
                                   capture_output=True, text=True, timeout=60)
                judged = bool(r.stdout.strip())
            except Exception:
                judged = False
        if judged:
            raise SystemExit("STRUCTURING INPUT ERROR: judge retention exists but 0 kept passages "
                             "loaded (staging/plumbing bug). Refusing to emit empty shortlists.")
        log("no kept passages and no judge output yet — nothing to rerank"); return
    cand = MR.bm25_candidates(passages, queries)
    log("BM25 pools built (<=%d/query)" % MR.K_BM25)

    # embed all candidate passages (union) + each query's semantic_query WITH THE SAME MODEL
    need = sorted({pid for c in cand.values() for (pid, _s) in c})
    pemb = embed_passages(passages, need)
    qvecs = CE.embed_many([q["semantic_query"] for q in queries], "search_query", model=_MODEL, log=log)

    rankings = {}
    for qi, q in enumerate(queries):
        qid = q["query_id"]; qv = qvecs[qi]
        pairs = cand.get(qid, [])
        if not qv or not pairs:
            rankings[qid] = []; continue
        scored = []
        for pid, bs in pairs:
            v = pemb.get(pid)
            if v is None:
                continue
            scored.append((pid, _cos(qv, v), bs))
        scored.sort(key=lambda x: -x[1])
        rankings[qid] = [dict(passage_id=p, cohere_cosine=round(s, 4), rank=i + 1,
                              bm25_score=round(bs, 3), ranker=RANKER,
                              embed_model=_MODEL, dim=_DIM)
                         for i, (p, s, bs) in enumerate(scored)]

    sl, merged = MR.shortlists(rankings)
    with open(os.path.join(OUT, "rankings.jsonl"), "w") as f:
        for qid, ranked in rankings.items():
            f.write(json.dumps(dict(query_id=qid, ranker=RANKER, embed_model=_MODEL, dim=_DIM,
                                    ranked=ranked), ensure_ascii=False) + "\n")
    C.write_json(os.path.join(OUT, "shortlists.json"), sl)
    summary = dict(dataset=DS.CFG["name"], built_at=C.now_iso(), ranker=RANKER,
                   embed_model=_MODEL, embedding_dim=_DIM,
                   n_queries=len(queries), n_kept_passages=len(passages), k_bm25=MR.K_BM25,
                   unique_passages_per_band={k: len(v) for k, v in merged.items()},
                   elapsed_s=round(time.time() - t0))
    C.write_json(os.path.join(OUT, "rerank_summary.json"), summary)
    subprocess.run(["aws", "s3", "sync", OUT, S3, "--only-show-errors", "--exclude", "embed_cache/*"],
                   timeout=1800, check=False)
    subprocess.run(["aws", "s3", "sync", EMB_CACHE, S3 + "embed_cache/", "--only-show-errors"],
                   timeout=1800, check=False)
    log("DONE cohere rerank: %s" % json.dumps(summary["unique_passages_per_band"]))


if __name__ == "__main__":
    main()
