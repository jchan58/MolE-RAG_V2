"""
Structuring steps 11-13 — MULTI-MODEL BEDROCK FLEET STRUCTURED-EVIDENCE EXTRACTION over the
largest-band (K1000) union (runs on DELTA; Bedrock token is local -> no credential transfer
to any box). The band set is card_schema.KBANDS; the union key is derived from max(KBANDS),
so extraction tracks the rerankers' deepest band automatically.

THROUGHPUT: a single Bedrock model (qwen3-32b) tops out near ~1.7 passages/s regardless of
thread count (its own concurrency ceiling). This extractor instead runs a FLEET of capable
models, each with its OWN worker pool (one-pool-per-backend), all pulling passages from ONE
shared queue, so aggregate concurrency is the SUM across models (target ~3-5 passages/s).
Default fleet (capable; NO weak ministral): qwen3-32b + opus-4-5/opus-4-6 + deepseek-v3 +
mistral-large. ONLY the model that extracts each passage changes — the prompt, output schema,
exact-substring quote validation, per-passage done-markers/resumability and the lock-guarded
atomic append to evidence_records.jsonl are IDENTICAL for every model, so evidence quality
stays comparable (tox21/hiv evidence is a model mix; lipo/sider used qwen3-32b+opus).

Reads the largest-band union shortlist + kept passages (text+metadata) + the frozen 2-level
attribute vocab, and emits one record per supported finding with normalized
attribute_path = "<attribute_group>.<attribute>" (+ attribute_label_original verbatim).
Cached by passage; validates that each supporting_quote is a substring of the source passage.

Output: RUN_DIR/structuring/evidence_records.jsonl (+ S3).
Env: EXP70_DATASET, EXP70_FILTER_BASE, EXP70_RUN_DIR;
     EXP70_EXTRACT_FLEET (csv of qwen,opus,deepseek,mistral), EXP70_EXTRACT_POOL_DEFAULT,
     EXP70_EXTRACT_POOL_<NAME> (per-model pool size), EXP70_EXTRACT_OPUS_MODELS.
"""
import os, sys, json, glob, time, threading, subprocess
from queue import Queue, Empty
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import datasets as DS
import card_schema as CS   # v2 finding-record enrichment (pure helpers; no heavy deps)
import judge as J   # reuses _loads_obj  (does NOT import any bedrock backend)
import importlib

BASE = os.environ.get("EXP70_FILTER_BASE", "/tmp/exp70_%s" % DS.CFG["name"])
PASS_TREE = os.path.join(BASE, "chunks", "passages")
OUT = C.P("structuring")
DONE = os.path.join(OUT, "extract_done")
S3 = os.environ.get("EXP70_STRUCT_S3", "s3://molerag/%s_chunks/structuring/" % DS.CFG["name"])
# ---- FLEET config (one worker pool per backend; shared passage queue) ----
# Members map to backend modules; each backend round-robins its OWN model(s) x regions, so
# pools don't share a single-model concurrency ceiling. Per-model pool size kept modest
# (24-48) so total Bedrock load is spread, not spiked; each backend has per-endpoint backoff.
FLEET_SPEC = os.environ.get("EXP70_EXTRACT_FLEET", "qwen,opus,deepseek,mistral")
POOL_DEFAULT = int(os.environ.get("EXP70_EXTRACT_POOL_DEFAULT",
                                  os.environ.get("EXP70_EXTRACT_THREADS", "32")))
# the fleet's "opus" member = bedrock_llm pinned to the capable opus models (NOT sonnet/haiku)
OPUS_MODELS = os.environ.get("EXP70_EXTRACT_OPUS_MODELS",
                             "us.anthropic.claude-opus-4-6-v1,"
                             "us.anthropic.claude-opus-4-5-20251101-v1:0")
_BACKEND_MODULE = {"qwen": "bedrock_qwen", "deepseek": "bedrock_deepseek",
                   "mistral": "bedrock_mistral"}
MAXTOK = int(os.environ.get("EXP70_EXTRACT_MAXTOK", "4000"))
TEXT_CAP = 3500
# Optional cap on the number of union passages to extract (0 = no cap). Used to VERIFY the
# plumbing cheaply without a full multi-thousand-call Bedrock run; the deterministic sorted
# head is taken so a capped run is reproducible. Production leaves this at 0.
EXTRACT_MAX = int(os.environ.get("EXP70_EXTRACT_MAX", "0"))
PROMPT_VERSION = "exp70-multi-extract-v1"

# Extraction runs ONCE over the LARGEST-band (K1000) union; this key is derived from
# card_schema.KBANDS so it tracks the rerankers automatically (no hardcoded band).
_UNION_KEY = "K%d" % max(CS.KBANDS)

_VOCAB = DS.attribute_vocab()


def _system():
    return (
        "You extract STRUCTURED EVIDENCE from a single biomedical passage about %s. You are "
        "given the passage text, and the ENTITY SCOPE: prediction tasks and/or compounds the "
        "passage is associated with. Extract EVERY supported finding as a separate record. "
        "Use ONLY the passage's own text; do not infer unstated values. For each finding emit a "
        "record with EXACTLY these fields: entity_type ('prediction_task' or 'compound'); "
        "entity_id (a task_id or compound_id from the scope); entity_mention (the exact text "
        "naming it, or '' for a task); attribute_path (ONE of the controlled paths below, or "
        "'<group>.other' / 'other.other' if none fit); attribute_label_original (the paper's "
        "verbatim wording for the attribute); evidence_type (one of: 'experimental result', "
        "'computational prediction', 'mechanism', 'general relationship', 'method'); finding "
        "(concise statement); value (number/string or null); unit (or null); conditions (assay/"
        "pH/cell-line/etc or null); qualifiers (hedging/negation/etc or null); supporting_quote "
        "(an EXACT substring copied from the passage); identity_status ('resolved' if the scope "
        "gives a verified id, 'article-local' if only an article-local label like 'compound 7', "
        "'unresolved' otherwise). "
        "CONTROLLED attribute_path values (group.attribute) for COMPOUND records: %s. For "
        "TASK records: %s. "
        "Return ONLY a JSON object {\"records\":[ ... ]} with one object per finding (empty list "
        "if the passage supports none). No prose, no <think>."
        % (DS.pretty(), ", ".join(_VOCAB["compound_paths"]), ", ".join(_VOCAB["task_paths"]))
    )


_ctr = dict(passages=0, records=0, calls=0, parse_fail=0, quote_fail=0)
_by_model = {}          # model_id -> dict(passages, records) : which model produced what
_lock = threading.Lock()
_fout_lock = threading.Lock()   # guards the SINGLE evidence_records.jsonl append across all pools


def _import_member(name):
    """Import a fleet member's backend module. 'opus'/'claude'/'llm' -> bedrock_llm pinned to
    the capable opus models via EXP70_CLAUDE_MODELS (read at import; nothing imports bedrock_llm
    before this, so the pin takes effect)."""
    name = name.lower()
    try:
        if name in ("opus", "claude", "llm"):
            os.environ["EXP70_CLAUDE_MODELS"] = OPUS_MODELS
            return importlib.import_module("bedrock_llm")
        mod = _BACKEND_MODULE.get(name)
        return importlib.import_module(mod) if mod else None
    except Exception as e:
        C.log("WARN", "extract", "fleet member '%s' unavailable, skipping: %r" % (name, e))
        return None


def build_fleet():
    """[(name, backend_module, pool_size)] for the configured fleet (skips members that fail
    to import). Falls back to a single qwen pool if none are available."""
    members = []
    for name in [s.strip() for s in FLEET_SPEC.split(",") if s.strip()]:
        n = int(os.environ.get("EXP70_EXTRACT_POOL_%s" % name.upper(), str(POOL_DEFAULT)))
        if n <= 0:
            continue
        mod = _import_member(name)
        if mod is not None:
            members.append((name, mod, n))
    if not members:
        members = [("qwen", importlib.import_module("bedrock_qwen"), POOL_DEFAULT)]
    return members


def _passage_text(pid_map, pid):
    return pid_map.get(pid)


def _load_pid_map(want_ids):
    """Passage records for the extraction union. Delegates to medcpt_rerank, which stages the
    candidate passages from S3 into FILTER_BASE if a fresh structuring node has none (so this
    no longer depends on the collection/judge node's /tmp chunk tree being present)."""
    import medcpt_rerank as MR
    return MR.candidate_passage_map(want_ids)


def _largest_band_union():
    """Union of passages across every query's LARGEST-band shortlist (K1000). Bands are
    nested so the top band is the superset; extraction is done ONCE over this union and the
    smaller-band cards are built by shortlist membership without re-extracting."""
    sl = C.read_json(os.path.join(OUT, "shortlists.json"), {}) or {}
    perq_task = {}   # pid -> set(task_id) from task-route retrieval
    union = set()
    for qid, bands in sl.items():
        for pid in bands.get(_UNION_KEY, []):
            union.add(pid)
            perq_task.setdefault(pid, set())
    # map query_id -> task_id via registry
    qmap = {}
    for l in open(C.P("structuring", "query_registry.jsonl")):
        r = json.loads(l); qmap[r["query_id"]] = r.get("task_id")
    for qid, bands in sl.items():
        tid = qmap.get(qid)
        if tid:
            for pid in bands.get(_UNION_KEY, []):
                perq_task[pid].add(tid)
    return union, perq_task


def _scope(passage, task_ids):
    tasks = sorted(task_ids or [])
    comps = []
    for a in (passage.get("molecule_associations") or []):
        comps.append(dict(compound_id=a.get("molecule_id"),
                          mention=a.get("preferred_name") or (a.get("aliases") or [None])[0]))
    return dict(prediction_tasks=tasks, compounds=comps)


def _enrich_record(r, pid, passage, meta):
    """ADDITIVELY upgrade a raw extraction record to the v2 `finding record` (CARD_SPEC s4).

    Only adds fields; never removes or rewrites the existing ones (attribute_path,
    attribute_label_original, value, unit, conditions, qualifiers, supporting_quote,
    identity_status, passage_id, pmcid, judge_model_extract, prompt_version,
    quote_validated all stay). The retrieval rank_band / rerank_score / query_id can only
    be known once the shortlists+rankings tables are joined, so they are left NULL here and
    filled by build_cards.py; matched_alias (cheap) is filled now.
    """
    et = r.get("entity_type")
    ap = r.get("attribute_path") or "other.other"                 # may be 2- OR 3-level
    attribute_id, group, _matched = CS.normalize_attr_path(ap, et, _VOCAB)
    r.setdefault("group", group)
    r.setdefault("attribute_label", CS.humanize_attr(attribute_id))
    r.setdefault("value_type", CS.infer_value_type(r.get("value")))
    if "direction" not in r:
        r["direction"] = r.get("direction")                       # model-supplied or None
    article_id = passage.get("pmcid") or r.get("pmcid")
    r["article_id"] = article_id
    r.setdefault("source_location", r.get("source_location") or passage.get("section_path"))
    r["source_url"] = CS.pmc_url(article_id)
    r.setdefault("retrieval", dict(matched_alias=(r.get("entity_mention") or None)
                                   if et == "compound" else None,
                                   query_id=None, bm25_rank=None,
                                   rerank_score=None, rank_band=None))
    r.setdefault("contradicts", [])                               # populated at card build
    r["finding_id"] = CS.finding_id(r.get("entity_id"), attribute_id, pid,
                                    r.get("supporting_quote"), r.get("value"))
    r["extraction"] = dict(model=(meta or {}).get("model"),
                           prompt_version=CS.EXTRACT_PROMPT_VERSION,
                           record_id=CS.record_id(pid, r.get("entity_id"), ap,
                                                  r.get("supporting_quote"),
                                                  r.get("value"), r.get("unit")))
    return r


def _dmark(pid):
    return os.path.join(DONE, pid + ".done")


def extract_one(pid, passage, task_ids, fout, backend):
    """Extract one passage with the given fleet backend. Prompt / schema / quote validation /
    done-marker / append are IDENTICAL regardless of backend — only the model differs."""
    if os.path.exists(_dmark(pid)):
        return "skip"
    text = (passage.get("text") or "")[:TEXT_CAP]
    scope = _scope(passage, task_ids)
    user = json.dumps(dict(passage_id=pid, text=text, entity_scope=scope), ensure_ascii=False)
    ok, content, usage, meta = backend.converse(_system(), user, max_tokens=MAXTOK)
    model = (meta or {}).get("model")
    with _lock:
        _ctr["calls"] += 1
    if not ok:
        # API failure AFTER the backend's own retries/backoff -> do NOT write a done-marker;
        # let the fleet re-queue this passage so another (possibly different) model retries it.
        return "fail"
    recs = []
    obj = J._loads_obj(content)
    if isinstance(obj, dict) and isinstance(obj.get("records"), list):
        for r in obj["records"]:
            if not isinstance(r, dict):
                continue
            q = r.get("supporting_quote") or ""
            if q and q[:60] not in text:        # validate quote is from the passage
                with _lock:
                    _ctr["quote_fail"] += 1
                r["quote_validated"] = False
            else:
                r["quote_validated"] = True
            r["passage_id"] = pid
            r["pmcid"] = passage.get("pmcid")
            r["judge_model_extract"] = model
            r["prompt_version"] = PROMPT_VERSION
            _enrich_record(r, pid, passage, meta)   # additive: v2 finding-record fields
            recs.append(r)
    else:
        with _lock:
            _ctr["parse_fail"] += 1
    # SINGLE evidence_records.jsonl shared by all model pools -> guard the append with its own
    # lock + flush so concurrent writes never interleave a line (each line a complete JSON).
    with _fout_lock:
        for r in recs:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        fout.flush()
    with _lock:
        _ctr["records"] += len(recs)
        _ctr["passages"] += 1
        bm = _by_model.setdefault(model, dict(passages=0, records=0))
        bm["passages"] += 1
        bm["records"] += len(recs)
    C.atomic_write_text(_dmark(pid), C.now_iso(), fsync=False)
    return "ok"


MAX_FLEET_RETRY = int(os.environ.get("EXP70_EXTRACT_FLEET_RETRY", "3"))
_attempts = {}


def _bump_attempt(pid):
    with _lock:
        _attempts[pid] = _attempts.get(pid, 0) + 1
        return _attempts[pid]


def run_fleet(pids, pid_map, perq_task, fout):
    """Drain `pids` across the fleet: one worker pool per backend, all pulling from ONE shared
    queue (dynamic load-balancing -> faster models process more). Returns elapsed seconds."""
    fleet = build_fleet()
    C.log("INFO", "extract", "fleet: %s | total workers: %d"
          % (", ".join("%s[%s]x%d" % (nm, getattr(mod, "BACKEND_NAME", nm), n)
                       for nm, mod, n in fleet),
             sum(n for _, _, n in fleet)))
    q = Queue()
    for pid in pids:
        q.put(pid)
    t0 = time.time()
    stop = threading.Event()

    def worker(backend):
        while True:
            pid = q.get()
            try:
                if pid is None:
                    return
                status = extract_one(pid, pid_map[pid], perq_task.get(pid), fout, backend)
                if status == "fail":
                    if _bump_attempt(pid) < MAX_FLEET_RETRY:
                        q.put(pid)               # retry on (likely) a different model
                    else:
                        C.atomic_write_text(_dmark(pid), C.now_iso(), fsync=False)  # give up
                        C.log("WARN", "extract", "pid %s gave up after %d fleet retries"
                              % (pid, MAX_FLEET_RETRY))
            except Exception as e:
                C.log("ERROR", "extract", "pid %s on %s failed: %r"
                      % (pid, getattr(backend, "BACKEND_NAME", "?"), e))
            finally:
                q.task_done()

    threads = []
    for _name, mod, n in fleet:
        for _ in range(n):
            t = threading.Thread(target=worker, args=(mod,), daemon=True)
            t.start()
            threads.append(t)

    def ticker():
        while not stop.wait(20):
            done = _ctr["passages"]
            C.log("INFO", "extract", "%d/%d passages, %d records, %.2f/s | by_model=%s"
                  % (done, len(pids), _ctr["records"], done / max(time.time() - t0, 1e-6),
                     json.dumps({m: v["passages"] for m, v in _by_model.items()})))
    tk = threading.Thread(target=ticker, daemon=True)
    tk.start()

    q.join()                       # all items (incl. re-queued retries) processed
    stop.set()
    for _ in threads:
        q.put(None)                # release blocked workers
    for t in threads:
        t.join(timeout=10)
    return time.time() - t0, fleet


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DONE, exist_ok=True)
    union, perq_task = _largest_band_union()
    if not union:
        C.log("WARN", "extract", "no %s union (rerank shortlists missing) — aborting" % _UNION_KEY)
        return
    if EXTRACT_MAX > 0 and len(union) > EXTRACT_MAX:
        union = set(sorted(union)[:EXTRACT_MAX])
        C.log("INFO", "extract", "EXP70_EXTRACT_MAX -> capped union to %d passages" % len(union))
    pid_map = _load_pid_map(union)
    pids = [pid for pid in sorted(union) if pid in pid_map]
    C.log("INFO", "extract", "%s union passages: %d (of %d; %d had no staged text)"
          % (_UNION_KEY, len(pids), len(union), len(union) - len(pids)))

    fout = open(os.path.join(OUT, "evidence_records.jsonl"), "a")
    elapsed, fleet = run_fleet(pids, pid_map, perq_task, fout)
    fout.close()

    rate = _ctr["passages"] / max(elapsed, 1e-6)
    summ = dict(dataset=DS.CFG["name"], built_at=C.now_iso(), prompt_version=PROMPT_VERSION,
                vocab_version=_VOCAB["version"],
                fleet=[dict(name=nm, backend=getattr(mod, "BACKEND_NAME", nm), pool=n)
                       for nm, mod, n in fleet],
                elapsed_s=round(elapsed, 1), aggregate_passages_per_s=round(rate, 3),
                by_model=_by_model, **_ctr)
    C.write_json(os.path.join(OUT, "extract_summary.json"), summ)
    subprocess.run(["aws", "s3", "sync", OUT, S3, "--only-show-errors"], timeout=1800, check=False)
    C.log("INFO", "extract", "EXTRACT DONE: %.2f passages/s | %s | by_model=%s"
          % (rate, json.dumps(_ctr), json.dumps(_by_model)))


if __name__ == "__main__":
    main()
