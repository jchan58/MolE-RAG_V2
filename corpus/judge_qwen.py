"""
exp70 relevance judge — LOCAL Qwen3-8B via vLLM (OpenAI-compatible endpoint), no
Bedrock throttle. Same exp70 two-group filter logic/schema as judge.py (reused):
property->BBBP relevance; molecule->substantive-molecule-info (molecule_id + aliases
embedded per passage); batched structured JSON with strict 1:1 passage_id validation;
context-expansion (<=2 rounds) on needs_context; deterministic retention.

Runs on a GPU compute node: chunks on node-local disk, outputs node-local + synced to
s3://molerag/exp70_chunks/qwen_judgments/. Resumable per-article done markers. A single
PROGRESS.json (articles_done/222222, passages/s, keep/exclude, ETA) is updated often.

Modes:  judge_qwen.py gate [n]      -> quality gate: judge a sample, compare vs Opus
        judge_qwen.py full [nshards] [shard]
"""
import os, sys, json, re, time, glob, threading, subprocess, random
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import merge as M
import judge as J
import requests

BASE = os.environ.get("EXP70_FILTER_BASE", "/tmp/exp70_filter")
CHUNKS = os.path.join(BASE, "chunks")
JOUT = os.path.join(BASE, "qwen_judgments")
RAW = os.path.join(BASE, "qwen_raw")
PROGRESS = os.path.join(BASE, "QWEN_PROGRESS.json")
OPUS_JUDG = os.path.join(BASE, "judgments")  # Opus judgments (for the gate comparison)
S3_DEST = "s3://molerag/exp70_chunks/qwen_judgments/"

QWEN_URL = os.environ.get("QWEN_URL", "http://127.0.0.1:8000/v1/chat/completions")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B")
ARTICLES_TOTAL = 222222
THREADS = int(os.environ.get("QWEN_THREADS", "192"))
BATCH = 30
MAXTOK = 10000
TEXT_CAP = 3500

_S = requests.Session()
_S.mount("http://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=512))
_ctr = dict(articles=0, calls=0, prop=0, mol=0, ctx=0, keep=0, review=0, exclude=0,
            pending=0, parse_fail=0, http_err=0)
_lock = threading.Lock()
_t0 = time.time()


def vllm_chat(system, user_text, max_tokens=MAXTOK):
    body = json.dumps({
        "model": QWEN_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_text}],
        "temperature": 0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    for attempt in range(5):
        try:
            r = _S.post(QWEN_URL, data=body, timeout=600, headers={"Content-Type": "application/json"})
        except requests.exceptions.RequestException:
            time.sleep(1 + attempt); continue
        if r.status_code == 200:
            try:
                return r.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                return None
        time.sleep(1 + attempt)
    with _lock:
        _ctr["http_err"] += 1
    return None


def _strip_think(t):
    if not t:
        return t
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    return t.strip()


def _call(system, user_obj, raw_key):
    content = vllm_chat(system, json.dumps(user_obj, ensure_ascii=False))
    with _lock:
        _ctr["calls"] += 1
    try:
        os.makedirs(os.path.join(RAW, M.shard_of(raw_key.split("_")[0]) if raw_key[0].isdigit() else "x"), exist_ok=True)
        with open(os.path.join(RAW, (M.shard_of(raw_key.split("_")[0]) if raw_key[0].isdigit() else "x"),
                                raw_key + ".json"), "w") as f:
            json.dump({"content": content}, f)
    except Exception:
        pass
    v = J._parse_verdicts(_strip_think(content))
    if v is None:
        with _lock:
            _ctr["parse_fail"] += 1
    return v


def jp_batch(passages, rk):
    user = dict(passages=[{"passage_id": p["passage_id"], "text": p["text"][:TEXT_CAP]} for p in passages])
    return {v["passage_id"]: v for v in (_call(J.PROPERTY_SYSTEM, user, rk) or [])
            if isinstance(v, dict) and v.get("passage_id")}


def jm_batch(items, rk):
    user = dict(items=[{"passage_id": it["passage_id"], "target": it["target"], "text": it["text"][:TEXT_CAP]} for it in items])
    return {v["passage_id"]: v for v in (_call(J.MOLECULE_SYSTEM, user, rk) or [])
            if isinstance(v, dict) and v.get("passage_id")}


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def passages_path(pmcid):
    return os.path.join(CHUNKS, "passages", M.shard_of(pmcid), "%s.jsonl" % pmcid)


def _jpath(g, pmcid):
    return os.path.join(JOUT, g, M.shard_of(pmcid), "%s.jsonl" % pmcid)


def _done(pmcid):
    return os.path.join(JOUT, "done", M.shard_of(pmcid), "%s.done" % pmcid)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    open(tmp, "w").write(text)
    os.replace(tmp, path)


def _load_passages(pmcid):
    try:
        return [json.loads(l) for l in open(passages_path(pmcid)) if l.strip()]
    except FileNotFoundError:
        return []


def _ctx(passage, by_index, passages):
    li = passage.get("local_index"); parts = []
    for d in (-2, -1, 1, 2):
        nb = by_index.get(li + d)
        if nb:
            parts.append(nb["text"][:1200])
    return "\n---\n".join(parts)[:4000]


def _expand(kind, passage, target, by_index, passages, rk):
    ctx = _ctx(passage, by_index, passages)
    if not ctx:
        return None
    up = None
    for rnd in range(J.MAX_CTX_ROUNDS):
        aug = passage["text"][:TEXT_CAP] + "\n\n[SURROUNDING CONTEXT]\n" + ctx
        if kind == "property":
            vm = jp_batch([{"passage_id": passage["passage_id"], "text": aug}], "%s.ctx%d" % (rk, rnd))
        else:
            vm = jm_batch([{"passage_id": passage["passage_id"], "target": target, "text": aug}], "%s.ctx%d" % (rk, rnd))
        v = vm.get(passage["passage_id"])
        if v:
            up = v
            if not v.get("needs_context"):
                break
        with _lock:
            _ctr["ctx"] += 1
    return up


def judge_article(pmcid):
    if os.path.exists(_done(pmcid)):
        return "cached"
    passages = _load_passages(pmcid)
    if not passages:
        _write(_done(pmcid), "{}"); return "empty"
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
        rk = "%s_p%d" % (pmcid, bi)
        vmap = jp_batch(batch, rk)
        miss = [p for p in batch if p["passage_id"] not in vmap]
        if miss:
            vmap.update(jp_batch(miss, rk + "_r"))
        for p in batch:
            v = vmap.get(p["passage_id"]); v0 = v
            if v and v.get("needs_context"):
                u = _expand("property", p, None, by_index, passages, rk + p["passage_id"][-6:])
                if u:
                    ctxdeps.append(dict(passage_id=p["passage_id"], judgment="property", initial=v0, updated=u)); v = u
            dec = J.retention_property(v)
            prop_recs.append(dict(passage_id=p["passage_id"], pmcid=pmcid, group="property", model=QWEN_MODEL,
                                  verdict=v, initial_verdict=(v0 if v0 is not v else None), retention=dec, judged_at=C.now_iso()))
            retn.append(dict(passage_id=p["passage_id"], group="property", decision=dec))
            with _lock:
                _ctr["prop"] += 1; _ctr[dec] = _ctr.get(dec, 0) + 1
    for bi, batch in enumerate(_batched(mol_items, BATCH)):
        rk = "%s_m%d" % (pmcid, bi)
        items = [{"passage_id": it["jid"], "text": it["passage"]["text"], "target": it["target"]} for it in batch]
        vmap = jm_batch(items, rk)
        miss = [it for it in items if it["passage_id"] not in vmap]
        if miss:
            vmap.update(jm_batch(miss, rk + "_r"))
        for it in batch:
            v = vmap.get(it["jid"]); v0 = v
            if v and v.get("needs_context"):
                u = _expand("molecule", it["passage"], it["target"], by_index, passages, rk + it["jid"][-8:])
                if u:
                    ctxdeps.append(dict(passage_id=it["passage"]["passage_id"], molecule_id=it["target"]["molecule_id"],
                                        judgment="molecule", initial=v0, updated=u)); v = u
            dec = J.retention_molecule(v)
            mol_recs.append(dict(passage_id=it["passage"]["passage_id"], pmcid=pmcid, group="molecule",
                                 molecule_id=it["target"]["molecule_id"], model=QWEN_MODEL, verdict=v,
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
    _write(_done(pmcid), json.dumps(dict(n_property=len(prop_recs), n_molecule=len(mol_recs), done_at=C.now_iso())))
    with _lock:
        _ctr["articles"] += 1
    return "ok"


def enumerate_chunked(nshards, shard):
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


def write_progress():
    with _lock:
        c = dict(_ctr)
    el = max(time.time() - _t0, 1e-6)
    judged = c["prop"] + c["mol"]
    done = len(glob.glob(os.path.join(JOUT, "done", "*", "*.done")))
    pps = judged / el
    c.update(updated_at=C.now_iso(), articles_done=done, articles_total=ARTICLES_TOTAL,
             passages_judged=judged, passages_per_s=round(pps, 1),
             calls_per_s=round(c["calls"] / el, 2),
             valid_json_rate=round(1 - c["parse_fail"] / max(c["calls"], 1), 3),
             eta_hours=(round((ARTICLES_TOTAL - done) / max(done / el, 1e-9) / 3600, 2) if done else None),
             model=QWEN_MODEL)
    _write(PROGRESS, json.dumps(c, indent=2))


# ---------------- quality gate ----------------
def quality_gate(n=400):
    """Sample passages already judged by Opus, re-judge with Qwen, compare."""
    # gather Opus judgments: passage_id -> (group, retention, molecule_id)
    opus = {}
    for g in ("property", "molecule"):
        for f in glob.glob(os.path.join(OPUS_JUDG, g, "*", "*.jsonl")):
            for l in open(f):
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                key = r["passage_id"] + ("@" + r.get("molecule_id", "") if g == "molecule" else "")
                opus[(g, r["passage_id"], r.get("molecule_id"))] = r.get("retention")
            if len(opus) > 20000:
                break
        if len(opus) > 20000:
            break
    # pick a mix of pmcids that Opus judged
    pmcids = sorted(set(k[1].split("@")[0] if False else None for k in []))  # placeholder
    done_pmcids = [os.path.basename(p)[:-5] for p in glob.glob(os.path.join(OPUS_JUDG, "done", "*", "*.done"))]
    random.shuffle(done_pmcids)
    agree = total = valid_calls = total_calls = 0
    bydec = {"agree": 0, "disagree": 0}
    gate_recs = []
    for pmcid in done_pmcids:
        if total >= n:
            break
        passages = _load_passages(pmcid)
        if not passages:
            continue
        # property
        prop_p = [p for p in passages if "property" in (p.get("document_groups") or [])]
        if prop_p:
            total_calls += 1
            vm = jp_batch(prop_p[:BATCH], "gate_%s_p" % pmcid)
            if vm:
                valid_calls += 1
            for p in prop_p[:BATCH]:
                ov = opus.get(("property", p["passage_id"], None))
                if ov is None:
                    continue
                qdec = J.retention_property(vm.get(p["passage_id"]))
                a = (_binz(qdec) == _binz(ov))
                agree += a; total += 1; bydec["agree" if a else "disagree"] += 1
                gate_recs.append(dict(pid=p["passage_id"], group="property", opus=ov, qwen=qdec, agree=a))
                if total >= n:
                    break
        mol_items = []
        for p in passages:
            if "molecule" in (p.get("document_groups") or []):
                for a2 in (p.get("molecule_associations") or []):
                    mol_items.append((p, a2))
        if mol_items and total < n:
            total_calls += 1
            items = [{"passage_id": "%s@%s" % (p["passage_id"], a2.get("molecule_id")), "text": p["text"],
                      "target": dict(molecule_id=a2.get("molecule_id"), preferred_name=a2.get("preferred_name"),
                                     aliases=(a2.get("aliases") or [])[:20])} for p, a2 in mol_items[:BATCH]]
            vm = jm_batch(items, "gate_%s_m" % pmcid)
            if vm:
                valid_calls += 1
            for p, a2 in mol_items[:BATCH]:
                ov = opus.get(("molecule", p["passage_id"], a2.get("molecule_id")))
                if ov is None:
                    continue
                qdec = J.retention_molecule(vm.get("%s@%s" % (p["passage_id"], a2.get("molecule_id"))))
                a = (_binz(qdec) == _binz(ov))
                agree += a; total += 1; bydec["agree" if a else "disagree"] += 1
                gate_recs.append(dict(pid=p["passage_id"], group="molecule", molecule_id=a2.get("molecule_id"),
                                      opus=ov, qwen=qdec, agree=a))
                if total >= n:
                    break
    res = dict(model=QWEN_MODEL, sampled_passages=total, calls=total_calls,
               valid_json_call_rate=round(valid_calls / max(total_calls, 1), 3),
               parse_fail=_ctr["parse_fail"], http_err=_ctr["http_err"],
               keep_exclude_agreement_vs_opus=round(agree / max(total, 1), 3),
               agree=bydec["agree"], disagree=bydec["disagree"], at=C.now_iso())
    _write(os.path.join(BASE, "QWEN_GATE.json"), json.dumps(res, indent=2))
    _write(os.path.join(BASE, "QWEN_GATE_detail.jsonl"),
           "\n".join(json.dumps(r) for r in gate_recs))
    print("GATE:", json.dumps(res), flush=True)
    return res


def _binz(dec):
    # collapse retention to keep-ish vs exclude-ish for agreement
    return "keep" if dec in ("keep", "review", "pending") else "exclude"


def _s3_sync():
    try:
        subprocess.run(["aws", "s3", "sync", JOUT, S3_DEST, "--only-show-errors"], timeout=900, check=False)
        subprocess.run(["aws", "s3", "cp", PROGRESS, S3_DEST + "PROGRESS.json", "--only-show-errors"], timeout=120, check=False)
    except Exception:
        pass


def run_full(nshards=1, shard=0):
    for g in ("property", "molecule", "retention", "context", "done"):
        os.makedirs(os.path.join(JOUT, g), exist_ok=True)
    cfg = J.judge_config(); cfg["backend"] = "vllm_qwen"; cfg["model"] = QWEN_MODEL; cfg["batch_size"] = BATCH
    _write(os.path.join(BASE, "qwen_judge_config.json"), json.dumps(cfg, indent=2))
    print("judge_qwen FULL shard %d/%d model=%s threads=%d batch=%d" % (shard, nshards, QWEN_MODEL, THREADS, BATCH), flush=True)
    last_sync = last_prog = time.time()
    while True:
        pend = [p for p in enumerate_chunked(nshards, shard) if not os.path.exists(_done(p))]
        if not pend:
            break
        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            futs = [ex.submit(_guarded, p) for p in pend]
            for _f in futs:
                _f.result()
                now = time.time()
                if now - last_prog > 10:
                    write_progress(); last_prog = now
                if shard == 0 and now - last_sync > 300:
                    threading.Thread(target=_s3_sync, daemon=True).start(); last_sync = now
        write_progress()
    write_progress(); _s3_sync()
    print("judge_qwen FULL done: %s" % json.dumps(dict(_ctr)), flush=True)


def _guarded(pmcid):
    try:
        return judge_article(pmcid)
    except Exception as e:
        C.log("ERROR", "judge_qwen", "article %s: %r" % (pmcid, e)); return "err"


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if mode == "gate":
        quality_gate(int(sys.argv[2]) if len(sys.argv) > 2 else 400)
    else:
        run_full(int(sys.argv[2]) if len(sys.argv) > 2 else 1,
                 int(sys.argv[3]) if len(sys.argv) > 3 else 0)
