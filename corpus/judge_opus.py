"""
exp70 relevance filter — Claude Opus 4.6 (Bedrock) judge driver.

Runs the full two-group filter over the chunk corpus pulled to NODE-LOCAL /tmp
(scratch is over quota). Reuses the frozen exp70 prompts/retention/schema from
judge.py; calls Bedrock Opus (bedrock_llm) with multi-region round-robin + backoff.
Batched (BATCH passages/structured-call, strict 1:1 passage_id validation),
context-expansion (<=2 rounds) on needs_context before any exclude, deterministic
retention. Per-article done markers (resumable). Outputs under /tmp; periodically
`aws s3 sync`-ed to s3://molerag/exp70_chunks/opus_judgments/ so results survive /tmp
wipes. Progress -> /tmp/exp70_filter/JUDGE_PROGRESS.json.

Shardable: judge_opus.py <nshards> <shard> for multi-process scaling.
"""
import os, sys, json, time, threading, subprocess, glob
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import merge as M
import judge as J
# Backend is swappable via EXP70_BACKEND (all expose the same converse/throttle/endpoint API):
#   claude    -> bedrock_llm      (Claude fan-out; here restricted to Opus 4.5 for the 10k slice)
#   qwen      -> bedrock_qwen     (Qwen3-32B, 3 regions)
#   deepseek  -> bedrock_deepseek (DeepSeek-V3.2, 3 regions)
#   mistral   -> bedrock_mistral  (Mistral-Large-3 675B)
#   ministral -> bedrock_ministral(Ministral-3 14B; weakest -> pure noise)
# Optional CLAUDE FALLBACK (_FB) re-judges items a cheap backend cannot resolve cleanly; it
# costs Claude $ so it is OFF by default (set EXP70_CLAUDE_FALLBACK=1 to enable). Residual
# unresolved items are recorded 'pending', never silently dropped.
import importlib
_BACKENDS = {"claude": "bedrock_llm", "qwen": "bedrock_qwen", "deepseek": "bedrock_deepseek",
             "mistral": "bedrock_mistral", "ministral": "bedrock_ministral",
             "vllm": "vllm_llm"}
_BK = os.environ.get("EXP70_BACKEND", "claude").lower()
B = importlib.import_module(_BACKENDS.get(_BK, "bedrock_llm"))
_FB = None
if os.environ.get("EXP70_CLAUDE_FALLBACK", "0") == "1" and _BK != "claude":
    import bedrock_llm as _FB

BASE = os.environ.get("EXP70_FILTER_BASE", "/tmp/exp70_filter")
CHUNKS = os.path.join(BASE, "chunks")
# --- PER-STREAM config (multi-backend split) --------------------------------
# Each stream judges a DISJOINT partition of the 222k corpus into its OWN output
# dir + done-dir + S3 prefix, so corpora never mix and each resumes independently.
# STREAM defaults to the backend name.
STREAM = os.environ.get("EXP70_STREAM") or _BK
JOUT = os.environ.get("EXP70_JOUT") or os.path.join(BASE, "judgments_%s" % STREAM)
RAW = os.path.join(BASE, "raw_judge_responses_%s" % STREAM)
PROGRESS = os.path.join(BASE, "JUDGE_PROGRESS_%s.json" % STREAM)
USAGE = os.path.join(BASE, "judge_usage_%s.jsonl" % STREAM)
import datasets as DS
S3_SYNC_DEST = os.environ.get("EXP70_S3_DEST") or ("s3://molerag/%s_chunks/%s_judgments/" % (DS.CFG["name"], STREAM))
# Whitelist of in-scope PMCIDs for THIS stream (the partition). None => whole corpus.
ASSIGN_FILE = os.environ.get("EXP70_ASSIGN") or None

THREADS = int(os.environ.get("EXP70_JUDGE_THREADS", "72"))     # concurrent GROUPS per shard
GROUP = int(os.environ.get("EXP70_GROUP", "6"))                # articles concatenated per batching unit (cross-article packing)
BATCH = int(os.environ.get("EXP70_BATCH", "300"))             # HARD cap passages per structured call (packed ACROSS articles)
CHAR_BUDGET = int(os.environ.get("EXP70_CHARBUDGET", "220000"))  # ~55k input tokens/call; packs ~250 avg passages per call
MAXTOK = int(os.environ.get("EXP70_MAXTOK", "6000"))          # decision-only output for up to ~300 short label entries
TEXT_CAP = 3000
SPLIT_DEPTH = 6     # recursive split-retry depth for coverage misses before giving up (-> pending)
SYNC_SEC = int(os.environ.get("EXP70_SYNC_SEC", "90"))   # S3 done-marker/result sync cadence (s); lost unsynced work twice to node restarts -> keep tight
SAVE_RAW = os.environ.get("EXP70_SAVE_RAW", "1") == "1"   # off for bulk run (avoids ~650k tiny /tmp files)
CLEAN_PMCIDS_FILE = C.P("data", "clean_pmcids.txt")
_CLEAN = None       # set of in-scope PMCIDs (None => no scoping)

_ctr = dict(articles=0, calls=0, prop=0, mol=0, ctx=0, keep=0, review=0, exclude=0,
            pending=0, in_tokens=0, out_tokens=0, parse_fail=0, throttle=0,
            failed=0, fallback=0, fb_calls=0)
_lock = threading.Lock()
_t0 = time.time()


def passages_path(pmcid):
    return os.path.join(CHUNKS, "passages", M.shard_of(pmcid), "%s.jsonl" % pmcid)


def _jpath(group, pmcid):
    return os.path.join(JOUT, group, M.shard_of(pmcid), "%s.jsonl" % pmcid)


def _done_marker(pmcid):
    return os.path.join(JOUT, "done", M.shard_of(pmcid), "%s.done" % pmcid)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _record_usage(kind, usage, meta):
    if not usage:
        return
    model = (meta or {}).get("model"); region = (meta or {}).get("region")
    cr = usage.get("cacheReadInputTokens", usage.get("cacheReadInputTokenCount", 0)) or 0
    cw = usage.get("cacheWriteInputTokens", usage.get("cacheWriteInputTokenCount", 0)) or 0
    with _lock:
        _ctr["in_tokens"] += usage.get("inputTokens", 0) or 0
        _ctr["out_tokens"] += usage.get("outputTokens", 0) or 0
        _ctr["cache_read"] = _ctr.get("cache_read", 0) + cr
        _ctr["cache_write"] = _ctr.get("cache_write", 0) + cw
    try:
        with open(USAGE, "a") as f:
            f.write(json.dumps(dict(ts=C.now_iso(), kind=kind, model=model, region=region,
                                    input=usage.get("inputTokens"), output=usage.get("outputTokens"),
                                    cache_read=cr, cache_write=cw)) + "\n")
    except Exception:
        pass


def _call(system, user_obj, raw_key, backend=None):
    """Returns (raw_content_or_None, model_id). model_id tags which model judged this call.
    backend=None -> primary (B); pass _FB to route through the Claude fallback."""
    bk = backend or B
    is_fb = backend is not None and backend is not B
    ok, content, usage, meta = bk.converse(system, json.dumps(user_obj, ensure_ascii=False), max_tokens=MAXTOK)
    model = (meta or {}).get("model")
    err = (meta or {}).get("error")
    with _lock:
        _ctr["calls"] += 1
        if is_fb:
            _ctr["fb_calls"] += 1
        if err and "throttl" in str(err):
            _ctr["throttle"] += 1
    _record_usage("judge", usage, meta)
    if SAVE_RAW:
        try:
            _write(os.path.join(RAW, M.shard_of(raw_key.split("_")[0]) if raw_key[0].isdigit() else "x",
                                raw_key + ".json"),
                   json.dumps(dict(ok=ok, meta=meta, usage=usage, content=content), ensure_ascii=False))
        except Exception:
            pass
    return (content if ok else None), model


def judge_property_batch(passages, raw_key, backend=None):
    """MINIMAL decision-only. Returns {passage_id: (verdict, model_id)}."""
    user = {"passages": {p["passage_id"]: p["text"][:TEXT_CAP] for p in passages}}
    content, model = _call(J.PROPERTY_SYSTEM_MIN, user, raw_key, backend=backend)
    vmap = J.parse_property_map(content)
    if vmap is None:
        with _lock:
            _ctr["parse_fail"] += 1
        vmap = {}
    return {pid: (v, model) for pid, v in vmap.items()}


def judge_molecule_batch(items, raw_key, backend=None):
    """MINIMAL decision-only. items carry passage_id (the per-(passage,molecule) jid). Returns {id: (verdict, model_id)}."""
    user = {"items": {it["passage_id"]: {"target": it["target"], "text": it["text"][:TEXT_CAP]} for it in items}}
    content, model = _call(J.MOLECULE_SYSTEM_MIN, user, raw_key, backend=backend)
    vmap = J.parse_molecule_map(content)
    if vmap is None:
        with _lock:
            _ctr["parse_fail"] += 1
        vmap = {}
    return {pid: (v, model) for pid, v in vmap.items()}


def _ctx_for(passage, by_index, passages):
    li = passage.get("local_index")
    parts = []
    for d in (-2, -1, 1, 2):
        nb = by_index.get(li + d)
        if nb:
            parts.append(nb["text"][:1200])
    if passage.get("parent_table_id"):
        for p in passages:
            if p.get("passage_type") == "table" and (p.get("table") or {}).get("caption"):
                parts.append("TABLE CAPTION: " + p["table"]["caption"][:400]); break
    return "\n---\n".join(parts)[:4500]


def _expand(kind, passage, target, by_index, passages, raw_key):
    """Returns (updated_verdict, model_id) or (None, None)."""
    ctx = _ctx_for(passage, by_index, passages)
    if not ctx:
        return None, None
    updated, umodel = None, None
    for rnd in range(J.MAX_CTX_ROUNDS):
        aug = passage["text"][:TEXT_CAP] + "\n\n[SURROUNDING CONTEXT]\n" + ctx
        if kind == "property":
            vm = judge_property_batch([{"passage_id": passage["passage_id"], "text": aug}], "%s.ctx%d" % (raw_key, rnd))
        else:
            vm = judge_molecule_batch([{"passage_id": passage["passage_id"], "target": target, "text": aug}],
                                      "%s.ctx%d" % (raw_key, rnd))
        vt = vm.get(passage["passage_id"])
        if vt:
            updated, umodel = vt
            if not updated.get("needs_context"):
                break
        with _lock:
            _ctr["ctx"] += 1
    return updated, umodel


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ------------------------------------------------------------------ #
# CROSS-ARTICLE batcher: pack ~BATCH passages from MANY small articles #
# into one structured call (char-budget bounded), strict 1:1 coverage  #
# with recursive split-retry, route verdicts back per article.         #
# ------------------------------------------------------------------ #
def _unit_cost(u):
    """Approx serialized chars this unit contributes to a call payload: capped text PLUS
    the molecule target metadata (ids + up to 20 aliases) — the latter was previously
    uncounted and pushed molecule batches past Qwen's 32k context (-> HTTP 400)."""
    tl = len(u["text"][:TEXT_CAP])
    tgt = u.get("target")
    if tgt:
        tl += len(json.dumps(tgt, ensure_ascii=False))
    return tl + 40  # per-item JSON punctuation/key overhead


def _pack(units):
    """Yield batches bounded by BATCH count and CHAR_BUDGET cumulative serialized size."""
    batch, chars = [], 0
    for u in units:
        tl = _unit_cost(u)
        if batch and (len(batch) >= BATCH or chars + tl > CHAR_BUDGET):
            yield batch
            batch, chars = [], 0
        batch.append(u)
        chars += tl
    if batch:
        yield batch


def _dispatch(kind, batch, rk, depth=0, backend=None):
    """ONE structured call over `batch` (units with id/text[/target]); recursive split-retry
    on coverage misses keyed by the per-item id. Returns {id: (verdict, model)}. Any id still
    missing after SPLIT_DEPTH splits is left absent (the caller routes it to the Claude
    fallback, or records 'pending' if even that fails -- never silently dropped)."""
    if kind == "property":
        vmap = judge_property_batch([{"passage_id": u["id"], "text": u["text"]} for u in batch], rk, backend=backend)
    else:
        vmap = judge_molecule_batch([{"passage_id": u["id"], "text": u["text"], "target": u["target"]}
                                     for u in batch], rk, backend=backend)
    miss = [u for u in batch if u["id"] not in vmap]
    if miss and depth < SPLIT_DEPTH:
        if len(batch) == 1:
            vmap.update(_dispatch(kind, batch, rk + "r", depth + 1, backend=backend))   # retry lone poison item
        else:
            mid = max(len(miss) // 2, 1)
            vmap.update(_dispatch(kind, miss[:mid], rk + "a", depth + 1, backend=backend))
            if miss[mid:]:
                vmap.update(_dispatch(kind, miss[mid:], rk + "b", depth + 1, backend=backend))
    return vmap


def _load_passages(pmcid):
    try:
        with open(passages_path(pmcid)) as f:
            return [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return None


def judge_group(pmcids):
    """Load GROUP articles, pack their property passages and molecule (passage,molecule) items
    into fat cross-article batches (~BATCH/call, char-budget bounded), dispatch with split-retry,
    then route each item's verdict back to its own article and finalize (retention + records +
    done marker). An article is finalized only after ALL its property + molecule items resolve."""
    arts = {}
    prop_units, mol_units = [], []
    for pmcid in pmcids:
        if os.path.exists(_done_marker(pmcid)):
            continue
        passages = _load_passages(pmcid)
        if passages is None:
            continue  # passages file missing (not yet pulled) — leave for a later pass
        if not passages:
            _write(_done_marker(pmcid), "{}")
            with _lock:
                _ctr["articles"] += 1
            continue
        a = dict(pmcid=pmcid, prop=[], mol=[])
        arts[pmcid] = a
        for p in passages:
            groups = p.get("document_groups") or []
            if "property" in groups:
                pid = p["passage_id"]
                a["prop"].append(pid)
                prop_units.append(dict(id=pid, text=p.get("text") or "", pmcid=pmcid))
            if "molecule" in groups:
                for assoc in (p.get("molecule_associations") or []):
                    mid = assoc.get("molecule_id")
                    jid = "%s@%s" % (p["passage_id"], mid)
                    a["mol"].append((jid, p["passage_id"], mid))
                    mol_units.append(dict(id=jid, text=p.get("text") or "", pmcid=pmcid,
                                          target=dict(molecule_id=mid,
                                                      preferred_name=assoc.get("preferred_name"),
                                                      aliases=(assoc.get("aliases") or [])[:20])))
    if not arts:
        return "none"
    gkey = pmcids[0]
    vmap = {}
    for bi, batch in enumerate(_pack(prop_units)):
        vmap.update(_dispatch("property", batch, "%s_prop_%d" % (gkey, bi)))
    for bi, batch in enumerate(_pack(mol_units)):
        vmap.update(_dispatch("molecule", batch, "%s_mol_%d" % (gkey, bi)))
    # CLAUDE FALLBACK: re-judge items Qwen could not resolve cleanly (coverage misses /
    # malformed JSON surviving split-retry) with the cheap Claude pool so NOTHING is left
    # unjudged. Verdicts carry the Claude model id -> judge_model provenance is automatic.
    if _FB is not None:
        miss_prop = [u for u in prop_units if u["id"] not in vmap]
        miss_mol = [u for u in mol_units if u["id"] not in vmap]
        nmiss = len(miss_prop) + len(miss_mol)
        if nmiss:
            with _lock:
                _ctr["failed"] += nmiss
            for bi, batch in enumerate(_pack(miss_prop)):
                vmap.update(_dispatch("property", batch, "%s_propFB_%d" % (gkey, bi), backend=_FB))
            for bi, batch in enumerate(_pack(miss_mol)):
                vmap.update(_dispatch("molecule", batch, "%s_molFB_%d" % (gkey, bi), backend=_FB))
            resolved = sum(1 for u in (miss_prop + miss_mol) if u["id"] in vmap)
            with _lock:
                _ctr["fallback"] += resolved
    for pmcid, a in arts.items():
        prop_recs, mol_recs, retn = [], [], []
        for pid in a["prop"]:
            vt = vmap.get(pid)
            v, model = (vt if vt else (None, None))
            dec = J.retention_property(v)
            prop_recs.append(dict(passage_id=pid, pmcid=pmcid, group="property",
                                  judge_model=model, prompt_version=J.JUDGE_PROMPT_VERSION,
                                  verdict=v, retention=dec, judged_at=C.now_iso()))
            retn.append(dict(passage_id=pid, group="property", decision=dec))
            with _lock:
                _ctr["prop"] += 1
                _ctr[dec] = _ctr.get(dec, 0) + 1
        for (jid, pid, mid) in a["mol"]:
            vt = vmap.get(jid)
            v, model = (vt if vt else (None, None))
            dec = J.retention_molecule(v)
            mol_recs.append(dict(passage_id=pid, pmcid=pmcid, group="molecule", molecule_id=mid,
                                 judge_model=model, prompt_version=J.JUDGE_PROMPT_VERSION,
                                 verdict=v, retention=dec, judged_at=C.now_iso()))
            retn.append(dict(passage_id=pid, group="molecule", molecule_id=mid, decision=dec))
            with _lock:
                _ctr["mol"] += 1
                _ctr[dec] = _ctr.get(dec, 0) + 1
        if prop_recs:
            _write(_jpath("property", pmcid), "\n".join(json.dumps(r, ensure_ascii=False) for r in prop_recs) + "\n")
        if mol_recs:
            _write(_jpath("molecule", pmcid), "\n".join(json.dumps(r, ensure_ascii=False) for r in mol_recs) + "\n")
        if retn:
            _write(_jpath("retention", pmcid), "\n".join(json.dumps(r, ensure_ascii=False) for r in retn) + "\n")
        _write(_done_marker(pmcid), json.dumps(dict(n_property=len(prop_recs), n_molecule=len(mol_recs),
                                                     done_at=C.now_iso())))
        with _lock:
            _ctr["articles"] += 1
    return "ok"


def judge_article(pmcid):
    if os.path.exists(_done_marker(pmcid)):
        return "cached"
    try:
        with open(passages_path(pmcid)) as f:
            passages = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return "missing"
    if not passages:
        _write(_done_marker(pmcid), "{}")
        return "empty"
    by_index = {p.get("local_index"): p for p in passages}
    prop_p = [p for p in passages if "property" in (p.get("document_groups") or [])]
    mol_items = []
    for p in passages:
        if "molecule" in (p.get("document_groups") or []):
            for a in (p.get("molecule_associations") or []):
                mol_items.append(dict(jid="%s@%s" % (p["passage_id"], a.get("molecule_id")), passage=p,
                                      target=dict(molecule_id=a.get("molecule_id"),
                                                  preferred_name=a.get("preferred_name"),
                                                  aliases=(a.get("aliases") or [])[:20])))
    prop_recs, mol_recs, retn, ctxdeps = [], [], [], []

    for bi, batch in enumerate(_batched(prop_p, BATCH)):
        rk = "%s_prop_%d" % (pmcid, bi)
        vmap = judge_property_batch(batch, rk)
        miss = [p for p in batch if p["passage_id"] not in vmap]
        if miss:
            vmap.update(judge_property_batch(miss, rk + "_r"))
        for p in batch:
            vt = vmap.get(p["passage_id"]); v, model = (vt if vt else (None, None)); v0 = v
            if v and v.get("needs_context"):
                up, um = _expand("property", p, None, by_index, passages, rk + p["passage_id"][-6:])
                if up:
                    ctxdeps.append(dict(passage_id=p["passage_id"], judgment="property", initial=v0, updated=up))
                    v = up; model = um or model
            dec = J.retention_property(v)
            prop_recs.append(dict(passage_id=p["passage_id"], pmcid=pmcid, group="property",
                                  judge_model=model, prompt_version=J.JUDGE_PROMPT_VERSION,
                                  verdict=v, initial_verdict=(v0 if v0 is not v else None),
                                  retention=dec, judged_at=C.now_iso()))
            retn.append(dict(passage_id=p["passage_id"], group="property", decision=dec))
            with _lock:
                _ctr["prop"] += 1; _ctr[dec] = _ctr.get(dec, 0) + 1

    for bi, batch in enumerate(_batched(mol_items, BATCH)):
        rk = "%s_mol_%d" % (pmcid, bi)
        items = [{"passage_id": it["jid"], "text": it["passage"]["text"], "target": it["target"]} for it in batch]
        vmap = judge_molecule_batch(items, rk)
        miss = [it for it in items if it["passage_id"] not in vmap]
        if miss:
            vmap.update(judge_molecule_batch(miss, rk + "_r"))
        for it in batch:
            vt = vmap.get(it["jid"]); v, model = (vt if vt else (None, None)); v0 = v
            if v and v.get("needs_context"):
                up, um = _expand("molecule", it["passage"], it["target"], by_index, passages, rk + it["jid"][-8:])
                if up:
                    ctxdeps.append(dict(passage_id=it["passage"]["passage_id"], molecule_id=it["target"]["molecule_id"],
                                        judgment="molecule", initial=v0, updated=up))
                    v = up; model = um or model
            dec = J.retention_molecule(v)
            mol_recs.append(dict(passage_id=it["passage"]["passage_id"], pmcid=pmcid, group="molecule",
                                 molecule_id=it["target"]["molecule_id"], judge_model=model,
                                 prompt_version=J.JUDGE_PROMPT_VERSION, verdict=v,
                                 initial_verdict=(v0 if v0 is not v else None), retention=dec, judged_at=C.now_iso()))
            retn.append(dict(passage_id=it["passage"]["passage_id"], group="molecule",
                             molecule_id=it["target"]["molecule_id"], decision=dec))
            with _lock:
                _ctr["mol"] += 1; _ctr[dec] = _ctr.get(dec, 0) + 1

    if prop_recs:
        _write(_jpath("property", pmcid), "\n".join(json.dumps(r, ensure_ascii=False) for r in prop_recs) + "\n")
    if mol_recs:
        _write(_jpath("molecule", pmcid), "\n".join(json.dumps(r, ensure_ascii=False) for r in mol_recs) + "\n")
    if retn:
        _write(_jpath("retention", pmcid), "\n".join(json.dumps(r, ensure_ascii=False) for r in retn) + "\n")
    if ctxdeps:
        _write(_jpath("context", pmcid), "\n".join(json.dumps(r, ensure_ascii=False) for r in ctxdeps) + "\n")
    _write(_done_marker(pmcid), json.dumps(dict(n_property=len(prop_recs), n_molecule=len(mol_recs),
                                                done_at=C.now_iso())))
    with _lock:
        _ctr["articles"] += 1
    return "ok"


def enumerate_chunked(nshards, shard):
    # Assignment mode: iterate THIS stream's partition directly (every assign PMCID is in the
    # chunked corpus, so no filesystem scan is needed) -> fast startup + exact scoping.
    if _CLEAN is not None:
        for pmcid in _CLEAN:
            if nshards <= 1 or (pmcid.isdigit() and int(pmcid) % nshards == shard):
                yield pmcid
        return
    base = os.path.join(CHUNKS, "passages")
    if not os.path.isdir(base):
        return
    for sh in os.listdir(base):
        d = os.path.join(base, sh)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".jsonl"):
                pmcid = fn[:-6]
                if nshards <= 1 or (pmcid.isdigit() and int(pmcid) % nshards == shard):
                    yield pmcid


ARTICLES_TOTAL = int(os.environ.get("EXP70_ARTICLES_TOTAL", "222222"))  # this stream's denominator
UNIFIED = os.path.join(BASE, "PROGRESS_%s.json" % STREAM)   # per-stream; a combiner merges both


_win = {"t": _t0, "calls": 0, "judged": 0, "articles": 0}


def _progress(nshards, shard, total):
    """Lightweight PER-SHARD progress (no global FS walk; the unified writer aggregates).
    Rates are LIVE (windowed over the interval since the previous _progress call, ~15s),
    not cumulative-since-start, so the headline numbers reflect the warm steady state."""
    with _lock:
        c = dict(_ctr)
    now = time.time()
    el = max(now - _t0, 1e-6)
    judged = c["prop"] + c["mol"]
    dt = max(now - _win["t"], 1e-6)
    live_pps = (judged - _win["judged"]) / dt
    live_cps = (c["calls"] - _win["calls"]) / dt
    live_aps = (c["articles"] - _win["articles"]) / dt
    _win.update(t=now, calls=c["calls"], judged=judged, articles=c["articles"])
    c.update(updated_at=C.now_iso(), shard="%d/%d" % (shard, nshards),
             elapsed_s=round(el, 1), articles_total=ARTICLES_TOTAL,
             articles_done_this_shard=c["articles"],
             # LIVE windowed rates (headline); *_avg = cumulative since process start
             passages_per_s_this_shard=round(live_pps, 1), calls_per_s_this_shard=round(live_cps, 2),
             articles_per_s_this_shard=round(live_aps, 3),
             passages_per_s_avg=round(judged / el, 1), calls_per_s_avg=round(c["calls"] / el, 2),
             throttle_events=B.throttle_stats().get("count", 0),
             endpoint_stats=B.endpoint_stats())
    _write(PROGRESS, json.dumps(c, indent=2))


def _count_done():
    """Cheap recursive count of per-article .done markers on node-local /tmp."""
    n = 0
    try:
        for sh in os.scandir(os.path.join(JOUT, "done")):
            if sh.is_dir():
                with os.scandir(sh.path) as it:
                    for e in it:
                        if e.name.endswith(".done"):
                            n += 1
    except FileNotFoundError:
        pass
    return n


def _unified_progress():
    """Aggregate all per-shard JUDGE_PROGRESS_*.json + global done markers -> PROGRESS.json."""
    agg = dict(keep=0, review=0, exclude=0, pending=0, prop=0, mol=0, calls=0,
               parse_fail=0, throttle=0, articles=0, ctx=0, in_tokens=0, out_tokens=0,
               failed=0, fallback=0, fb_calls=0)
    max_el = 1e-6
    live_pps = live_cps = live_aps = 0.0   # sum of per-shard LIVE (windowed) rates
    per_shard = []
    for f in glob.glob(os.path.join(BASE, "JUDGE_PROGRESS_%s_*.json" % STREAM)):
        try:
            if time.time() - os.path.getmtime(f) > 150:
                continue  # stale shard file from a prior run config -> ignore (no double-count)
            d = json.load(open(f))
        except Exception:
            continue
        for k in agg:
            agg[k] += d.get(k, 0) or 0
        max_el = max(max_el, d.get("elapsed_s", 0) or 0)
        live_pps += d.get("passages_per_s_this_shard", 0) or 0
        live_cps += d.get("calls_per_s_this_shard", 0) or 0
        live_aps += d.get("articles_per_s_this_shard", 0) or 0
        per_shard.append(dict(shard=d.get("shard"), articles=d.get("articles_done_this_shard"),
                              passages_per_s=d.get("passages_per_s_this_shard"),
                              calls_per_s=d.get("calls_per_s_this_shard"),
                              updated_at=d.get("updated_at")))
    done = _count_done()
    judged = agg["prop"] + agg["mol"]
    remaining = max(ARTICLES_TOTAL - done, 0)
    tot_tokens = agg["in_tokens"] + agg["out_tokens"]
    # headline rates = LIVE windowed (sum across shards); ETA from live article rate.
    out = dict(
        updated_at=C.now_iso(), stream=STREAM, scope=("partition" if _CLEAN is not None else "full_222k"),
        backend=getattr(B, "BACKEND_NAME", B.__name__), model=B.MODEL,
        articles_done=done, articles_total=ARTICLES_TOTAL, pct=round(100.0 * done / ARTICLES_TOTAL, 2),
        articles_per_s=round(live_aps, 3), articles_per_s_avg=round(agg["articles"] / max_el, 3),
        passages_judged=judged, passages_per_s=round(live_pps, 1), passages_per_s_avg=round(judged / max_el, 1),
        calls=agg["calls"], calls_per_s=round(live_cps, 2), calls_per_s_avg=round(agg["calls"] / max_el, 2),
        keep=agg["keep"], review=agg["review"], exclude=agg["exclude"], pending=agg["pending"],
        parse_fail=agg["parse_fail"], context_expansions=agg["ctx"],
        qwen_failed_count=agg["failed"], claude_fallback_count=agg["fallback"],
        claude_fallback_calls=agg["fb_calls"],
        prompt_tokens=agg["in_tokens"], completion_tokens=agg["out_tokens"], total_tokens=tot_tokens,
        tokens_per_article=(round(tot_tokens / agg["articles"], 1) if agg["articles"] else None),
        tokens_per_s=round(tot_tokens / max_el, 0), tpm_estimate=round(tot_tokens / max_el * 60.0, 0),
        throttle_events=B.throttle_stats().get("count", 0), endpoint_stats=B.endpoint_stats(),
        eta_min=(round(remaining / live_aps / 60.0, 1) if live_aps > 1e-6 else None),
        n_shards=len(per_shard), per_shard=per_shard)
    _write(UNIFIED, json.dumps(out, indent=2))
    return out


def _unified_loop():
    """Shard-0 daemon: refresh unified PROGRESS.json and push it to S3 periodically."""
    i = 0
    while True:
        try:
            _unified_progress()
            if i % 6 == 0:  # ~every 2 min
                subprocess.run(["aws", "s3", "cp", UNIFIED, S3_SYNC_DEST + "PROGRESS.json",
                                "--only-show-errors"], timeout=120, check=False)
        except Exception as e:
            C.log("WARN", "judge_opus", "unified progress failed: %r" % e)
        i += 1
        time.sleep(20)


_sync_busy = {"on": False}


def _s3_sync():
    # Guard: never let two syncs overlap (as done/ grows a sync can exceed the interval;
    # overlapping `aws s3 sync` would pile up threads and thrash). Skip if one is running.
    with _lock:
        if _sync_busy["on"]:
            return
        _sync_busy["on"] = True
    try:
        subprocess.run(["aws", "s3", "sync", JOUT, S3_SYNC_DEST, "--only-show-errors"],
                       timeout=600, check=False)
        cfg = os.path.join(BASE, "judge_config.json")
        if os.path.exists(cfg):
            subprocess.run(["aws", "s3", "cp", cfg, S3_SYNC_DEST + "judge_config.json",
                            "--only-show-errors"], timeout=120, check=False)
        if os.path.exists(UNIFIED):
            subprocess.run(["aws", "s3", "cp", UNIFIED, S3_SYNC_DEST + "PROGRESS.json",
                            "--only-show-errors"], timeout=120, check=False)
    except Exception as e:
        C.log("WARN", "judge_opus", "s3 sync failed: %r" % e)
    finally:
        with _lock:
            _sync_busy["on"] = False


def run(nshards=1, shard=0):
    global PROGRESS, _CLEAN
    PROGRESS = os.path.join(BASE, "JUDGE_PROGRESS_%s_%d.json" % (STREAM, shard))
    for g in ("property", "molecule", "retention", "context", "done"):
        os.makedirs(os.path.join(JOUT, g), exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    # SCOPE: this stream's assignment partition (EXP70_ASSIGN). Falls back to the clean-set
    # toggle, else the full corpus. The partition files are disjoint across streams.
    if ASSIGN_FILE and os.path.exists(ASSIGN_FILE):
        with open(ASSIGN_FILE) as f:
            _CLEAN = set(l.strip() for l in f if l.strip())
        print("SCOPE: assignment partition %s = %d PMCIDs" % (ASSIGN_FILE, len(_CLEAN)), flush=True)
    elif os.environ.get("EXP70_SCOPE_CLEAN") == "1" and os.path.exists(CLEAN_PMCIDS_FILE):
        with open(CLEAN_PMCIDS_FILE) as f:
            _CLEAN = set(l.strip() for l in f if l.strip())
        print("SCOPE: clean set = %d PMCIDs (broad-noise articles skipped)" % len(_CLEAN), flush=True)
    else:
        _CLEAN = None
        print("SCOPE: FULL corpus (no restriction)", flush=True)
    if shard == 0:
        cfg = J.judge_config()
        cfg["backend"] = getattr(B, "BACKEND_NAME", B.__name__)
        cfg["model"] = B.MODEL; cfg["models"] = B.MODEL_WEIGHTS; cfg["regions"] = B.REGIONS
        cfg["batch_size"] = BATCH; cfg["max_tokens"] = MAXTOK
        cfg["batching"] = "cross_article"; cfg["group_articles"] = GROUP
        cfg["char_budget_per_call"] = CHAR_BUDGET; cfg["text_cap"] = TEXT_CAP
        cfg["split_retry_depth"] = SPLIT_DEPTH
        _re = getattr(B, "REASONING_EFFORT", None)
        if _re:
            cfg["reasoning_effort"] = _re
        cfg["output_mode"] = "minimal_decision_only"
        cfg["property_system"] = J.PROPERTY_SYSTEM_MIN; cfg["molecule_system"] = J.MOLECULE_SYSTEM_MIN
        cfg["model_note"] = ("%d models x %d regions = %d endpoints, weighted toward Opus 4.5 + the two "
                             "Sonnets; MINIMAL decision-only output (label per passage_id, no spans/"
                             "reasons/CoT); evidence deferred to a cheap 2nd pass over KEPT passages; "
                             "each judgment record tagged judge_model"
                             % (len(B.MODEL_WEIGHTS), len(B.REGIONS), len(B.MODEL_WEIGHTS) * len(B.REGIONS)))
        cfg["scope"] = "full_222k" if _CLEAN is None else ("clean_%d" % len(_CLEAN))
        _write(os.path.join(BASE, "judge_config.json"), json.dumps(cfg, indent=2))
        threading.Thread(target=_unified_loop, daemon=True).start()
    print("judge_opus shard %d/%d: %d models x %d regions, %d threads, GROUP=%d, BATCH<=%d, "
          "char_budget=%d, maxtok=%d (cross-article packing)"
          % (shard, nshards, len(B.MODEL_WEIGHTS), len(B.REGIONS), THREADS, GROUP, BATCH,
             CHAR_BUDGET, MAXTOK), flush=True)
    last_sync = last_prog = time.time()
    while True:
        pend = [p for p in enumerate_chunked(nshards, shard) if not os.path.exists(_done_marker(p))]
        total = _ctr["articles"] + len(pend)
        if not pend:
            # nothing pending now; if the pull is still going, wait for more, else finish
            if os.path.exists(os.path.join(BASE, "PULL_DONE.txt")):
                break
            time.sleep(15); continue
        groups = [pend[i:i + GROUP] for i in range(0, len(pend), GROUP)]
        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            futs = [ex.submit(_guarded, g) for g in groups]
            for _f in futs:
                _f.result()
                now = time.time()
                if now - last_prog > 15:
                    _progress(nshards, shard, total); last_prog = now
                if shard == 0 and now - last_sync > SYNC_SEC:
                    threading.Thread(target=_s3_sync, daemon=True).start(); last_sync = now
        _progress(nshards, shard, total)
    _progress(nshards, shard, _ctr["articles"])
    if shard == 0:
        _s3_sync()
    print("judge_opus shard %d/%d DONE: %s" % (shard, nshards, json.dumps(dict(_ctr))), flush=True)


def _guarded(group):
    try:
        return judge_group(group)
    except Exception as e:
        C.log("ERROR", "judge_opus", "group head=%s n=%d: %r" % (group[0] if group else "-", len(group), e))
        return "err"


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sh = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run(ns, sh)
