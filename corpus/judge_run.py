"""
Judging driver — concurrent, batched gpt-5.4-mini relevance judging over chunked
passages. Property-group and molecule-group judgments run together. Resumable
(per-article done markers), idempotent, full-coverage (every paragraph + table).

Scales by processes: `judge_run.py <nshards> <shard>` processes only articles with
(int(pmcid) % nshards == shard); launch N processes to multiply throughput. Each
process uses a large thread pool over articles; the central OpenAI token bucket
(state/rate_overrides.json "openai" ~450/s) caps the global request rate.

Outputs: data/judge_config.json, task_judgments.jsonl, molecule_judgments.jsonl,
retention_decisions.jsonl, context_dependencies.jsonl, processing_manifest.jsonl
(all sharded per-article under judgments/), raw_judge_responses/, usage ledger,
state/JUDGE_PROGRESS.json.
"""
import os, sys, json, time, threading, glob
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import merge as M
import chunker as CH
import judge as J

# Smaller thread stacks so many shard-processes x threads fit under the node's
# memory/pids limits (5120 x 8MB stacks was ~40GB -> fork EAGAIN).
try:
    threading.stack_size(512 * 1024)
except (ValueError, RuntimeError):
    pass
THREADS = int(os.environ.get("EXP70_JUDGE_THREADS", "48"))  # per process
_ctr = dict(articles=0, prop_passages=0, mol_judgments=0, ctx_expansions=0,
            keep=0, review=0, exclude=0, pending=0)
_lock = threading.Lock()
_t0 = time.time()


def _jpath(group, pmcid):
    return C.P("judgments", group, M.shard_of(pmcid), "%s.jsonl" % pmcid)


def _done_marker(pmcid):
    return C.P("judgments", "done", M.shard_of(pmcid), "%s.done" % pmcid)


def _article_done(pmcid):
    return os.path.exists(_done_marker(pmcid))


def _load_passages(pmcid):
    path = CH.passages_path(pmcid)
    out = []
    try:
        with open(path) as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    except FileNotFoundError:
        return []
    return out


def _context_for(passage, by_index, passages):
    """Neighbour paragraphs (local_index +/-2) + parent table caption."""
    li = passage.get("local_index")
    ctx = []
    for d in (-2, -1, 1, 2):
        nb = by_index.get(li + d)
        if nb:
            ctx.append(nb["text"][:1500])
    if passage.get("parent_table_id"):
        for p in passages:
            if p.get("passage_type") == "table" and (p.get("table") or {}).get("caption"):
                ctx.append("TABLE CAPTION: " + p["table"]["caption"][:500])
                break
    return "\n---\n".join(ctx)[:5000]


def _expand_and_rejudge(kind, passage, target, by_index, passages, raw_key):
    """Up to MAX_CTX_ROUNDS context-augmented single re-judgements. Returns updated verdict."""
    updated = None
    ctx = _context_for(passage, by_index, passages)
    if not ctx:
        return None
    for rnd in range(J.MAX_CTX_ROUNDS):
        if kind == "property":
            user = dict(passages=[{"passage_id": passage["passage_id"],
                                   "text": passage["text"][:6000], "surrounding_context": ctx}])
            vmap = J.judge_property_batch(
                [{"passage_id": passage["passage_id"],
                  "text": passage["text"][:6000] + "\n\n[SURROUNDING CONTEXT]\n" + ctx}],
                "%s.ctx%d" % (raw_key, rnd))
        else:
            vmap = J.judge_molecule_batch(
                [{"passage_id": passage["passage_id"], "target": target,
                  "text": passage["text"][:6000] + "\n\n[SURROUNDING CONTEXT]\n" + ctx}],
                "%s.ctx%d" % (raw_key, rnd))
        v = vmap.get(passage["passage_id"])
        if v:
            updated = v
            if not v.get("needs_context"):
                break
    return updated


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def judge_article(pmcid):
    if _article_done(pmcid):
        return "cached"
    # Cross-node claim (O_EXCL on shared /scratch, with lease) so ANY number of
    # nodes can run nshards=1 over the whole clean set with no double-judging and
    # full coverage — no pre-assigned shard ranges needed.
    lk = C.ItemLock("judge_article", pmcid)
    if not lk.acquire():
        return "claimed"
    try:
        if _article_done(pmcid):
            return "cached"
        return _judge_article_inner(pmcid)
    finally:
        lk.release()


def _judge_article_inner(pmcid):
    passages = _load_passages(pmcid)
    if not passages:
        _mark_done(pmcid, 0, 0)
        return "empty"
    by_index = {p.get("local_index"): p for p in passages}

    prop_passages = [p for p in passages if "property" in (p.get("document_groups") or [])]
    # molecule items keyed by passage_id@molecule_id
    mol_items = []
    for p in passages:
        if "molecule" in (p.get("document_groups") or []):
            for a in (p.get("molecule_associations") or []):
                tgt = dict(molecule_id=a.get("molecule_id"),
                           preferred_name=a.get("preferred_name"),
                           aliases=(a.get("aliases") or [])[:20])
                mol_items.append(dict(jid="%s@%s" % (p["passage_id"], a.get("molecule_id")),
                                      passage=p, target=tgt))

    prop_recs, mol_recs, retention, ctxdeps = [], [], [], []

    # ---- property judging ----
    for bi, batch in enumerate(_batched(prop_passages, J.BATCH)):
        raw_key = "%s_prop_%d" % (pmcid, bi)
        vmap = J.judge_property_batch(batch, raw_key)
        # 1:1 validation + one retry for missing
        missing = [p for p in batch if p["passage_id"] not in vmap]
        if missing:
            vmap2 = J.judge_property_batch(missing, raw_key + "_retry")
            vmap.update(vmap2)
        for p in batch:
            v = vmap.get(p["passage_id"])
            v0 = v
            if v and v.get("needs_context"):
                upd = _expand_and_rejudge("property", p, None, by_index, passages, raw_key + "_" + p["passage_id"][-6:])
                if upd:
                    ctxdeps.append(dict(passage_id=p["passage_id"], judgment="property",
                                        initial=v0, updated=upd, rounds_used=True))
                    v = upd
            dec = J.retention_property(v)
            rec = dict(passage_id=p["passage_id"], pmcid=pmcid, group="property",
                       prompt_version=J.JUDGE_PROMPT_VERSION, model=J.JUDGE_MODEL,
                       verdict=v, initial_verdict=(v0 if v0 is not v else None),
                       retention=dec, judged_at=C.now_iso())
            prop_recs.append(rec)
            retention.append(dict(passage_id=p["passage_id"], group="property", decision=dec))
            with _lock:
                _ctr["prop_passages"] += 1; _ctr[dec] = _ctr.get(dec, 0) + 1

    # ---- molecule judging ----
    for bi, batch in enumerate(_batched(mol_items, J.BATCH)):
        raw_key = "%s_mol_%d" % (pmcid, bi)
        items = [{"passage_id": it["jid"], "text": it["passage"]["text"], "target": it["target"]} for it in batch]
        vmap = J.judge_molecule_batch(items, raw_key)
        missing = [it for it in items if it["passage_id"] not in vmap]
        if missing:
            vmap.update(J.judge_molecule_batch(missing, raw_key + "_retry"))
        for it in batch:
            v = vmap.get(it["jid"]); v0 = v
            if v and v.get("needs_context"):
                upd = _expand_and_rejudge("molecule", it["passage"], it["target"], by_index, passages,
                                          raw_key + "_" + it["jid"][-8:])
                if upd:
                    ctxdeps.append(dict(passage_id=it["passage"]["passage_id"], molecule_id=it["target"]["molecule_id"],
                                        judgment="molecule", initial=v0, updated=upd, rounds_used=True))
                    v = upd
            dec = J.retention_molecule(v)
            rec = dict(passage_id=it["passage"]["passage_id"], pmcid=pmcid, group="molecule",
                       molecule_id=it["target"]["molecule_id"],
                       prompt_version=J.JUDGE_PROMPT_VERSION, model=J.JUDGE_MODEL,
                       verdict=v, initial_verdict=(v0 if v0 is not v else None),
                       retention=dec, judged_at=C.now_iso())
            mol_recs.append(rec)
            retention.append(dict(passage_id=it["passage"]["passage_id"], group="molecule",
                                  molecule_id=it["target"]["molecule_id"], decision=dec))
            with _lock:
                _ctr["mol_judgments"] += 1; _ctr[dec] = _ctr.get(dec, 0) + 1
                if v0 is not v:
                    _ctr["ctx_expansions"] += 1

    # write per-article outputs (fsync-light, atomic)
    if prop_recs:
        C.atomic_write_text(_jpath("property", pmcid),
                            "\n".join(json.dumps(r, ensure_ascii=False) for r in prop_recs) + "\n", fsync=False)
    if mol_recs:
        C.atomic_write_text(_jpath("molecule", pmcid),
                            "\n".join(json.dumps(r, ensure_ascii=False) for r in mol_recs) + "\n", fsync=False)
    if retention:
        C.atomic_write_text(_jpath("retention", pmcid),
                            "\n".join(json.dumps(r, ensure_ascii=False) for r in retention) + "\n", fsync=False)
    if ctxdeps:
        C.atomic_write_text(_jpath("context", pmcid),
                            "\n".join(json.dumps(r, ensure_ascii=False) for r in ctxdeps) + "\n", fsync=False)
    _mark_done(pmcid, len(prop_recs), len(mol_recs))
    with _lock:
        _ctr["articles"] += 1
    return "ok"


def _mark_done(pmcid, nprop, nmol):
    C.write_json(_done_marker(pmcid), dict(pmcid=pmcid, n_property=nprop, n_molecule=nmol,
                                           done_at=C.now_iso()), fsync=False)


def _write_progress(nshards, shard):
    with _lock:
        c = dict(_ctr)
    el = max(time.time() - _t0, 1e-6)
    c.update(updated_at=C.now_iso(), shard="%d/%d" % (shard, nshards),
             elapsed_s=round(el, 1),
             passages_per_s=round((c["prop_passages"] + c["mol_judgments"]) / el, 1),
             articles_per_s=round(c["articles"] / el, 2),
             openai_rate_setting=C.service_rate("openai"))
    # per-shard file (shards don't clobber each other); aggregate computed separately
    os.makedirs(C.P("state", "judge_progress"), exist_ok=True)
    C.write_json(C.P("state", "judge_progress", "shard_%02d.json" % shard), c, fsync=False)


def _clean_set():
    p = C.P("data", "clean_pmcids.txt")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return set(x.strip() for x in f if x.strip())

def enumerate_chunked(nshards, shard):
    base = C.P("passages")
    clean = _clean_set()   # judge only the clean full-text set when present
    for sh in os.listdir(base):
        d = os.path.join(base, sh)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".jsonl"):
                pmcid = fn[:-6]
                if clean is not None and pmcid not in clean:
                    continue
                if nshards <= 1 or (pmcid.isdigit() and int(pmcid) % nshards == shard):
                    yield pmcid


def run(nshards=1, shard=0):
    C.ensure_dirs()
    for g in ("property", "molecule", "retention", "context", "done"):
        os.makedirs(C.P("judgments", g), exist_ok=True)
    os.makedirs(C.P("raw_judge_responses"), exist_ok=True)
    if shard == 0:
        C.write_json(C.P("data", "judge_config.json"), J.judge_config())
    C.atomic_write_text(C.P("state", "judge_run_%d.pid" % shard), str(os.getpid()))
    C.log("INFO", "judge_run", "shard %d/%d starting with %d threads (openai=%s/s)"
          % (shard, nshards, THREADS, C.service_rate("openai")))

    last = [time.time()]
    stop_marker = C.P("state", "JUDGE_STOP")
    def submit_loop():
        idle = 0
        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            while True:
                pend = [p for p in enumerate_chunked(nshards, shard) if not _article_done(p)]
                if not pend:
                    # keep-alive: wait for newly-chunked articles until a stop signal
                    if os.path.exists(stop_marker):
                        return
                    idle += 1
                    _write_progress(nshards, shard)
                    time.sleep(20)
                    continue
                idle = 0
                futs = [ex.submit(_guarded, pmcid) for pmcid in pend]
                for _f in futs:
                    _f.result()
                    if time.time() - last[0] > 20:
                        _write_progress(nshards, shard); last[0] = time.time()
    submit_loop()
    _write_progress(nshards, shard)
    C.log("INFO", "judge_run", "shard %d/%d done: %s" % (shard, nshards, json.dumps(dict(_ctr))))


def _guarded(pmcid):
    try:
        return judge_article(pmcid)
    except Exception as e:
        C.log("ERROR", "judge_run", "article %s: %r" % (pmcid, e))
        return "err"


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sh = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run(ns, sh)
