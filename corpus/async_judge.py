"""
ASYNC judging driver (aiohttp) — single process, thousands of concurrent in-flight
requests, in-process ~450/s token bucket, to actually saturate the OpenAI ceiling
(threads can't reach the needed ~6k in-flight with a ~13s reasoning latency).

Reuses judge.py prompts/parsing/retention and the chunker passage files. Judges only
the clean set (data/clean_pmcids.txt). Batched (15 passages/call), strict 1:1
passage_id validation (+retry), context expansion on needs_context (<=2 rounds),
deterministic retention. Per-article outputs + raw responses + usage; resumable via
per-article done markers.

Run: python async_judge.py [nshards] [shard]   (optional sharding to add processes)
"""
import os, sys, json, time, asyncio, hashlib, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import merge as M
import chunker as CH
import judge as J
import aiohttp

RATE = 450.0                 # calls/s (in-process)
ARTICLE_WORKERS = 1200       # concurrent article coroutines
CONNECTOR_LIMIT = 6000
MAX_RETRIES = 5

_ctr = dict(articles=0, calls=0, prop=0, mol=0, ctx=0,
            keep=0, review=0, exclude=0, pending=0, err429=0, errors=0)
_t0 = time.time()


class Bucket:
    def __init__(self, rate):
        self.rate = rate; self.tok = rate; self.ts = time.monotonic()
    async def acquire(self):
        while True:
            now = time.monotonic()
            self.tok = min(self.rate, self.tok + (now - self.ts) * self.rate); self.ts = now
            if self.tok >= 1.0:
                self.tok -= 1.0; return
            await asyncio.sleep((1.0 - self.tok) / self.rate)


def _jpath(group, pmcid):
    return C.P("judgments", group, M.shard_of(pmcid), "%s.jsonl" % pmcid)


def _done_marker(pmcid):
    return C.P("judgments", "done", M.shard_of(pmcid), "%s.done" % pmcid)


def _load_passages(pmcid):
    out = []
    try:
        with open(CH.passages_path(pmcid)) as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    except FileNotFoundError:
        pass
    return out


async def _call(session, bucket, system, user_obj, raws):
    body = json.dumps({"model": J.JUDGE_MODEL,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)}],
                       "max_completion_tokens": J.MCT}).encode()
    key = C.openai_key()
    for attempt in range(MAX_RETRIES + 1):
        await bucket.acquire()
        try:
            async with session.post(C.ENDPOINTS["openai"], data=body,
                                    headers={"Authorization": "Bearer " + key,
                                             "Content-Type": "application/json"}) as r:
                if r.status == 200:
                    j = await r.json()
                    with_usage(j.get("usage"))
                    _ctr["calls"] += 1
                    content = (j.get("choices") or [{}])[0].get("message", {}).get("content")
                    raws.append({"usage": j.get("usage"), "content": content})
                    return J._parse_verdicts(content)
                if r.status == 429:
                    _ctr["err429"] += 1
                    ra = r.headers.get("Retry-After")
                    await asyncio.sleep(float(ra) if ra and ra.replace('.', '', 1).isdigit() else min(30, 2 ** attempt))
                    continue
                if r.status >= 500:
                    await asyncio.sleep(min(30, 2 ** attempt)); continue
                txt = await r.text()
                raws.append({"error": "http %s" % r.status, "body": txt[:300]})
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(min(20, 2 ** attempt)); continue
            _ctr["errors"] += 1
            raws.append({"error": repr(e)})
            return None
    return None


_usage = dict(prompt=0, completion=0, total=0, n=0)
def with_usage(u):
    if u:
        _usage["prompt"] += u.get("prompt_tokens") or 0
        _usage["completion"] += u.get("completion_tokens") or 0
        _usage["total"] += u.get("total_tokens") or 0
        _usage["n"] += 1


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _context_for(passage, by_index, passages):
    li = passage.get("local_index")
    ctx = []
    for d in (-2, -1, 1, 2):
        nb = by_index.get(li + d)
        if nb:
            ctx.append(nb["text"][:1500])
    return "\n---\n".join(ctx)[:4000]


def _persist(pmcid, prop_recs, mol_recs, retention, ctxdeps, raws):
    # All blocking Lustre I/O in ONE thread hop so it never blocks the event loop.
    for grp, recs in (("property", prop_recs), ("molecule", mol_recs),
                      ("retention", retention), ("context", ctxdeps)):
        if recs:
            C.atomic_write_text(_jpath(grp, pmcid),
                                "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", fsync=False)
    if raws:
        C.atomic_write_text(C.P("raw_judge_responses", M.shard_of(pmcid), "%s.json" % pmcid),
                            json.dumps(raws, ensure_ascii=False), fsync=False)
    C.write_json(_done_marker(pmcid), dict(pmcid=pmcid, n_property=len(prop_recs),
                                           n_molecule=len(mol_recs), done_at=C.now_iso()), fsync=False)


async def judge_article(session, bucket, pmcid):
    # done-marker pre-filtered at queue build; file I/O offloaded to threads.
    passages = await asyncio.to_thread(_load_passages, pmcid)
    if not passages:
        await asyncio.to_thread(C.write_json, _done_marker(pmcid),
                                dict(pmcid=pmcid, n=0, done_at=C.now_iso()), False)
        _ctr["articles"] += 1
        return
    by_index = {p.get("local_index"): p for p in passages}
    raws = []
    prop_recs, mol_recs, retention, ctxdeps = [], [], [], []

    prop_pass = [p for p in passages if "property" in (p.get("document_groups") or [])]
    mol_items = []
    for p in passages:
        if "molecule" in (p.get("document_groups") or []):
            for a in (p.get("molecule_associations") or []):
                mol_items.append((p, dict(molecule_id=a.get("molecule_id"),
                                          preferred_name=a.get("preferred_name"),
                                          aliases=(a.get("aliases") or [])[:20])))

    async def do_prop(batch):
        user = dict(passages=[{"passage_id": p["passage_id"], "text": p["text"][:6000]} for p in batch])
        vs = await _call(session, bucket, J.PROPERTY_SYSTEM, user, raws) or []
        vmap = {v["passage_id"]: v for v in vs if isinstance(v, dict) and v.get("passage_id")}
        for p in batch:
            v = vmap.get(p["passage_id"]); v0 = v
            if v and v.get("needs_context"):
                ctx = _context_for(p, by_index, passages)
                if ctx:
                    u2 = dict(passages=[{"passage_id": p["passage_id"],
                                         "text": p["text"][:6000] + "\n\n[CONTEXT]\n" + ctx}])
                    vs2 = await _call(session, bucket, J.PROPERTY_SYSTEM, u2, raws) or []
                    v2 = {x["passage_id"]: x for x in vs2 if isinstance(x, dict) and x.get("passage_id")}.get(p["passage_id"])
                    if v2:
                        ctxdeps.append(dict(passage_id=p["passage_id"], judgment="property", initial=v0, updated=v2))
                        v = v2; _ctr["ctx"] += 1
            dec = J.retention_property(v)
            prop_recs.append(dict(passage_id=p["passage_id"], pmcid=pmcid, group="property",
                                  verdict=v, initial_verdict=(v0 if v0 is not v else None),
                                  retention=dec, model=J.JUDGE_MODEL, prompt_version=J.JUDGE_PROMPT_VERSION))
            retention.append(dict(passage_id=p["passage_id"], group="property", decision=dec))
            _ctr["prop"] += 1; _ctr[dec] = _ctr.get(dec, 0) + 1

    async def do_mol(batch):
        user = dict(items=[{"passage_id": "%s@%s" % (p["passage_id"], t["molecule_id"]),
                            "target": t, "text": p["text"][:6000]} for (p, t) in batch])
        vs = await _call(session, bucket, J.MOLECULE_SYSTEM, user, raws) or []
        vmap = {v["passage_id"]: v for v in vs if isinstance(v, dict) and v.get("passage_id")}
        for (p, t) in batch:
            jid = "%s@%s" % (p["passage_id"], t["molecule_id"])
            v = vmap.get(jid); v0 = v
            if v and v.get("needs_context"):
                ctx = _context_for(p, by_index, passages)
                if ctx:
                    u2 = dict(items=[{"passage_id": jid, "target": t,
                                      "text": p["text"][:6000] + "\n\n[CONTEXT]\n" + ctx}])
                    vs2 = await _call(session, bucket, J.MOLECULE_SYSTEM, u2, raws) or []
                    v2 = {x["passage_id"]: x for x in vs2 if isinstance(x, dict) and x.get("passage_id")}.get(jid)
                    if v2:
                        ctxdeps.append(dict(passage_id=p["passage_id"], molecule_id=t["molecule_id"],
                                            judgment="molecule", initial=v0, updated=v2))
                        v = v2; _ctr["ctx"] += 1
            dec = J.retention_molecule(v)
            mol_recs.append(dict(passage_id=p["passage_id"], pmcid=pmcid, group="molecule",
                                 molecule_id=t["molecule_id"], verdict=v,
                                 initial_verdict=(v0 if v0 is not v else None), retention=dec,
                                 model=J.JUDGE_MODEL, prompt_version=J.JUDGE_PROMPT_VERSION))
            retention.append(dict(passage_id=p["passage_id"], group="molecule",
                                  molecule_id=t["molecule_id"], decision=dec))
            _ctr["mol"] += 1; _ctr[dec] = _ctr.get(dec, 0) + 1

    tasks = [do_prop(b) for b in _batched(prop_pass, J.BATCH)] + \
            [do_mol(b) for b in _batched(mol_items, J.BATCH)]
    if tasks:
        await asyncio.gather(*tasks)

    await asyncio.to_thread(_persist, pmcid, prop_recs, mol_recs, retention, ctxdeps, raws)
    _ctr["articles"] += 1


def _clean_set():
    with open(C.P("data", "clean_pmcids.txt")) as f:
        return set(x.strip() for x in f if x.strip())


def enumerate_articles(nshards, shard):
    clean = _clean_set()
    base = C.P("passages")
    for sh in os.listdir(base):
        d = os.path.join(base, sh)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".jsonl"):
                pmcid = fn[:-6]
                if pmcid in clean and (nshards <= 1 or (pmcid.isdigit() and int(pmcid) % nshards == shard)):
                    yield pmcid


async def _progress_loop(nshards, shard):
    while True:
        await asyncio.sleep(15)
        el = max(time.time() - _t0, 1e-6)
        C.write_json(C.P("state", "JUDGE_PROGRESS.json"),
                     dict(updated_at=C.now_iso(), mode="async", shard="%d/%d" % (shard, nshards),
                          **{k: _ctr[k] for k in _ctr},
                          calls_per_s=round(_ctr["calls"] / el, 1),
                          passages_per_s=round((_ctr["prop"] + _ctr["mol"]) / el, 1),
                          tokens_per_call=(round(_usage["total"] / _usage["n"]) if _usage["n"] else 0),
                          total_tokens=_usage["total"], elapsed_s=round(el, 1)))


async def main(nshards, shard):
    import concurrent.futures
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=192))
    C.ensure_dirs()
    for g in ("property", "molecule", "retention", "context", "done"):
        os.makedirs(C.P("judgments", g), exist_ok=True)
    os.makedirs(C.P("raw_judge_responses"), exist_ok=True)
    if shard == 0:
        C.write_json(C.P("data", "judge_config.json"), J.judge_config())
    C.atomic_write_text(C.P("state", "async_judge_%d.pid" % shard), str(os.getpid()))
    bucket = Bucket(RATE)
    conn = aiohttp.TCPConnector(limit=CONNECTOR_LIMIT, limit_per_host=CONNECTOR_LIMIT, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=180)
    prog = asyncio.ensure_future(_progress_loop(nshards, shard))
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        q = asyncio.Queue()
        for pmcid in enumerate_articles(nshards, shard):
            if not os.path.exists(_done_marker(pmcid)):
                q.put_nowait(pmcid)
        C.log("INFO", "async_judge", "shard %d/%d: %d articles to judge, %d workers, rate=%s/s"
              % (shard, nshards, q.qsize(), ARTICLE_WORKERS, RATE))

        async def worker():
            while True:
                try:
                    pmcid = q.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await judge_article(session, bucket, pmcid)
                except Exception as e:
                    _ctr["errors"] += 1
                    C.log("ERROR", "async_judge", "%s: %r" % (pmcid, e))
                finally:
                    q.task_done()
        await asyncio.gather(*[worker() for _ in range(ARTICLE_WORKERS)])
    prog.cancel()
    el = max(time.time() - _t0, 1e-6)
    C.log("INFO", "async_judge", "shard %d/%d DONE: articles=%d calls=%d (%.1f/s) passages=%d (%.0f/s)"
          % (shard, nshards, _ctr["articles"], _ctr["calls"], _ctr["calls"] / el,
             _ctr["prop"] + _ctr["mol"], (_ctr["prop"] + _ctr["mol"]) / el))


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sh = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    asyncio.run(main(ns, sh))
