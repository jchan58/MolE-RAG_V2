"""
Stages 5-8 — query validation, PMC ESearch submission, exhaustive retrieval of
EVERY matching PMCID (with subdivision past the 10,000 window), and the search
manifest.

Query validation (stage 5) is done in CODE without corrupting chemical names
(parentheses/commas inside quoted names are respected). Duplicate query strings
are executed ONCE (keyed by search_id) but all query/molecule associations are kept.

Retrieval (stage 7): ESearch exposes only the first 10,000 UIDs, so any query whose
reported count exceeds a safe threshold is subdivided recursively by publication
date ranges ([PDAT]) via deterministic binary splitting. The original Boolean is
kept intact in every subquery. Terminal single-day partitions that still exceed the
window trigger a History-server rescue (ESummary paging beyond 10k). Unique counts
are reconciled against reported counts; any residual gap also triggers the rescue.
Incomplete partitions are marked explicitly; nothing is silently truncated.

Per-search state: results/search/<search_id>/manifest.json + leaf_<key>.json
"""
import os, sys, json, re, datetime, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

EARLIEST = datetime.date(1500, 1, 1)
LATEST = datetime.date.today()

# SPEED LEVER (deadline mode): when EXP70_SEARCH_CAP>0, retrieve only up to CAP PMCIDs
# per query in a SINGLE ESearch page and skip ALL date subdivision. BM25 prescreen only
# needs enough candidates, not an exhaustive crawl — this makes each query exactly ~1
# NCBI request, maximizing searches/s under the shared 10/s cap.
SEARCH_CAP = int(os.environ.get("EXP70_SEARCH_CAP", "0"))
# cap molecule queries per molecule (0 = all); deadline mode sets ~2.
_MOL_QCAP = int(os.environ.get("EXP70_MOL_QCAP", "0"))

KNOWN_FIELDS = {
    "PDAT", "EDAT", "MHDA", "CRDT", "AUTH", "AU", "FAU", "TIAB", "TI", "TIABKW",
    "AB", "MESH", "MH", "JOUR", "TA", "LANG", "LA", "PTYP", "PT", "AFFL", "AD",
    "GRNT", "DP", "PMID", "PMCID", "DOI", "AID", "WORD", "ALL", "SO", "VOL", "ISS",
    "PG", "UID", "FILT", "SB", "BOOK", "ISBN", "PUBN", "CN", "IR", "OT",
}
BOOL_OPS = {"AND", "OR", "NOT"}
MAX_TERM_LEN = 7000


# ---------------- validation ----------------
def _mask_quotes(q):
    """Return (quotes_balanced, masked_text) where each quoted "..." span is
    replaced by a single placeholder OPERAND token (Q). Blanking quoted spans to
    spaces would make two operators that surround a quoted operand look adjacent
    (e.g. '"A" OR "B"' -> 'OR OR'), causing false doubled-operator errors; a
    placeholder operand preserves the real token structure for the checks below,
    while still removing any parentheses/operators that live inside quotes."""
    out = []
    in_q = False
    nq = 0
    for ch in q:
        if ch == '"':
            nq += 1
            if not in_q:
                out.append(" Q ")   # one operand placeholder per quoted span
            in_q = not in_q
        else:
            out.append(" " if in_q else ch)
    return (nq % 2 == 0), "".join(out)


def validate_query(q):
    reasons = []
    warnings = []
    if not q or not q.strip():
        return dict(valid=False, reasons=["empty_query"], warnings=[])
    if len(q) > MAX_TERM_LEN:
        warnings.append("query_long_%d_chars_submitted_via_post" % len(q))
    balanced_q, masked = _mask_quotes(q)
    if not balanced_q:
        reasons.append("unbalanced_double_quotes")
    # paren balance on the quote-masked text (names inside quotes ignored)
    depth = 0
    for ch in masked:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                reasons.append("unbalanced_parentheses")
                break
    if depth > 0:
        reasons.append("unbalanced_parentheses")
    # empty clauses / dangling operators (on masked text)
    if re.search(r"\(\s*\)", masked):
        reasons.append("empty_parentheses")
    toks = re.findall(r"[A-Za-z]+|\(|\)", masked)
    # doubled operators / operator at boundaries
    prev = None
    for i, t in enumerate(toks):
        tu = t.upper()
        if tu in BOOL_OPS:
            if i == 0 or i == len(toks) - 1:
                warnings.append("operator_at_boundary:%s" % tu)
            # NOT may legitimately follow AND/OR (exclusion, e.g. "X AND NOT Y");
            # any other adjacency of two binary operators is invalid.
            if prev in BOOL_OPS and not (tu == "NOT" and prev in ("AND", "OR")):
                reasons.append("doubled_operator:%s_%s" % (prev, tu))
        prev = tu
    # unknown field tags (outside quotes)
    for tag in re.findall(r"\[([^\[\]]+)\]", masked):
        base = tag.strip().upper().split(":")[0]
        if base and base not in KNOWN_FIELDS and not base.startswith("PDAT"):
            warnings.append("unknown_field_tag:%s" % tag.strip())
    return dict(valid=(len(reasons) == 0), reasons=reasons, warnings=warnings)


# ---------------- queries collation ----------------
def search_id_for(query_text):
    norm = re.sub(r"\s+", " ", query_text.strip())
    return "q" + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def collate_queries():
    """Build data/queries.jsonl and data/search_items.jsonl from querygen results.
    Dedups identical query strings into search_items while preserving associations."""
    import querygen as QG
    queries = []            # per (owner,query) association records
    search_items = {}       # search_id -> aggregated item

    def add(owner_type, owner_id, cid_list, resp_path, qobjs):
        for n, q in enumerate(qobjs):
            qtext = q["query"]
            v = validate_query(qtext)
            sid = search_id_for(qtext)
            qid = "%s#%d" % (owner_id, n)
            rec = dict(query_id=qid, query_type=owner_type, query_text=qtext,
                       purpose=q.get("purpose", ""), coverage_description=q.get("coverage", ""),
                       molecule_ids=([owner_id] if owner_type == "molecule" else []),
                       pubchem_cids=cid_list, identifiers_used=q.get("identifiers_used", []),
                       model=C.OPENAI_MODEL, prompt_version=C.PROMPT_VERSION,
                       generated_at=C.now_iso(), response_path=resp_path,
                       validation=v, search_id=sid)
            queries.append(rec)
            it = search_items.get(sid)
            if it is None:
                it = dict(search_id=sid, query_text=qtext, query_ids=[], molecule_ids=[],
                          query_types=[], pubchem_cids=[], valid=v["valid"],
                          validation=v)
                search_items[sid] = it
            it["query_ids"].append(qid)
            if owner_type == "molecule" and owner_id not in it["molecule_ids"]:
                it["molecule_ids"].append(owner_id)
            if owner_type not in it["query_types"]:
                it["query_types"].append(owner_type)
            for c in cid_list:
                if c not in it["pubchem_cids"]:
                    it["pubchem_cids"].append(c)

    # property (PER-TASK: one query set per task_id)
    pdir = C.P("results", "querygen_property")
    if os.path.isdir(pdir):
        for fn in sorted(os.listdir(pdir)):
            if not fn.endswith(".json"):
                continue
            prop = C.read_json(os.path.join(pdir, fn))
            if not prop:
                continue
            owner = prop.get("task_id") or "property"
            add("property", owner, [], os.path.join(pdir, fn), prop.get("queries", []))
    # molecules — PARALLELIZE the 41k per-molecule reads (I/O-bound on slow/over-quota
    # Lustre; serial read here was the same ptlrpc_set_wait wall as querygen), then aggregate
    # SERIALLY (add() mutates the shared queries/search_items dicts -> keep single-threaded).
    import concurrent.futures as _cf
    d = C.P("results", "querygen_molecule")
    try:
        mfiles = sorted(fn for fn in os.listdir(d) if fn.endswith(".json"))
    except FileNotFoundError:
        mfiles = []

    def _read_rec(fn):
        return fn, C.read_json(os.path.join(d, fn))

    nthr = max(1, int(os.environ.get("EXP70_SCAN_THREADS", "48")))
    with _cf.ThreadPoolExecutor(max_workers=nthr) as ex:
        read_results = list(ex.map(_read_rec, mfiles))
    for fn, rec in read_results:
        if not rec or rec.get("skipped"):
            continue
        cid = rec.get("cid")
        _qs = rec.get("queries", [])
        if _MOL_QCAP > 0:
            _qs = _qs[:_MOL_QCAP]   # deadline: cap queries per molecule (fewer NCBI searches)
        add("molecule", rec["mol_id"], ([cid] if cid else []),
            os.path.join(d, fn), _qs)

    with open(C.P("data", "queries.jsonl"), "w") as f:
        for r in queries:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(C.P("data", "search_items.jsonl"), "w") as f:
        for sid in sorted(search_items):
            f.write(json.dumps(search_items[sid], ensure_ascii=False) + "\n")
    C.log("INFO", "search", "collated %d query associations into %d unique searches"
          % (len(queries), len(search_items)))
    return queries, search_items


def search_items():
    items = {}
    path = C.P("data", "search_items.jsonl")
    if not os.path.exists(path):
        return items
    with open(path) as f:
        for line in f:
            it = json.loads(line)
            items[it["search_id"]] = it
    return items


def querygen_stats():
    """Compute + persist query-count stats the moment query generation completes.
    Written to state/QUERYGEN_STATS.json and data/querygen_stats.json."""
    import statistics
    qpath = C.P("data", "queries.jsonl")
    n_prop = n_mol = 0
    per_mol = {}
    valid = invalid = 0
    with open(qpath) as f:
        for line in f:
            q = json.loads(line)
            if q.get("validation", {}).get("valid"):
                valid += 1
            else:
                invalid += 1
            if q["query_type"] == "property":
                n_prop += 1
            else:
                n_mol += 1
                for m in q.get("molecule_ids", []):
                    per_mol[m] = per_mol.get(m, 0) + 1
    items = search_items()
    distinct = len(items)
    distinct_valid = sum(1 for it in items.values() if it.get("valid"))
    counts = sorted(per_mol.values())
    dist = dict(n_molecules_with_queries=len(counts),
                min=(counts[0] if counts else 0),
                median=(statistics.median(counts) if counts else 0),
                max=(counts[-1] if counts else 0),
                mean=(round(statistics.mean(counts), 2) if counts else 0))
    stats = dict(generated_at=C.now_iso(),
                 property_queries=n_prop,
                 molecule_query_associations=n_mol,
                 total_query_associations=n_prop + n_mol,
                 valid_associations=valid, invalid_associations=invalid,
                 distinct_searches_after_dedup=distinct,
                 distinct_valid_searches=distinct_valid,
                 dedup_savings=(n_prop + n_mol) - distinct,
                 queries_per_molecule=dist)
    C.write_json(C.P("state", "QUERYGEN_STATS.json"), stats)
    C.write_json(C.P("data", "querygen_stats.json"), stats)
    C.log("INFO", "querygen_stats",
          "QUERY COUNTS: property=%d molecule_assoc=%d distinct_searches=%d "
          "(valid=%d) per_molecule[min/med/max]=%s/%s/%s"
          % (n_prop, n_mol, distinct, distinct_valid,
             dist["min"], dist["median"], dist["max"]))
    return stats


# ---------------- ESearch primitives ----------------
def _parse_esearchresult(j):
    er = j.get("esearchresult", {}) if isinstance(j, dict) else {}
    try:
        count = int(er.get("count", "0"))
    except (TypeError, ValueError):
        count = 0
    return dict(count=count, retmax=er.get("retmax"), retstart=er.get("retstart"),
                idlist=er.get("idlist", []) or [],
                querytranslation=er.get("querytranslation"),
                translationstack=er.get("translationstack"),
                warninglist=er.get("warninglist"), errorlist=er.get("errorlist"),
                webenv=er.get("webenv"), querykey=er.get("querykey"),
                raw_error=er.get("ERROR"))


def esearch(term, retstart=0, retmax=0, usehistory=False, webenv=None, query_key=None):
    params = dict(db="pmc", term=term, retmode="json",
                  retstart=retstart, retmax=retmax)
    if usehistory:
        params["usehistory"] = "y"
    if webenv:
        params["WebEnv"] = webenv
        params["query_key"] = query_key
    r = C.ncbi_post("esearch", params, want="text")
    if not r.ok:
        return None, dict(error=r.error, status=r.status,
                          body=(r.text or "")[:400])
    try:
        j = json.loads(r.text)
    except json.JSONDecodeError as e:
        return None, dict(error="json-decode: %r" % e, status=r.status,
                          body=(r.text or "")[:400])
    return _parse_esearchresult(j), None


def _retrieve_direct(term, first=None):
    """Retrieve all UIDs for a term known to be under the window. Returns dict.
    `first` may be a pre-fetched esearch result (count + first page) to avoid a
    redundant count call."""
    ids = []
    if first is None:
        first, err = esearch(term, 0, C.SEARCH_PAGE)
        if err:
            return dict(ok=False, error=err, ids=[], reported=None, warnings=None,
                        translation=None)
    count = first["count"]
    ids.extend(first["idlist"])
    retstart = C.SEARCH_PAGE
    pages = 1
    while len(ids) < count and retstart < 10000:
        page, perr = esearch(term, retstart, C.SEARCH_PAGE)
        if perr:
            return dict(ok=False, error=perr, ids=ids, reported=count,
                        warnings=first["warninglist"], translation=first["querytranslation"],
                        pages=pages, incomplete=True)
        if not page["idlist"]:
            break
        ids.extend(page["idlist"])
        retstart += C.SEARCH_PAGE
        pages += 1
    return dict(ok=True, ids=ids, reported=count, pages=pages,
                warnings=first["warninglist"], errors=first["errorlist"],
                translation=first["querytranslation"],
                incomplete=(len(set(ids)) < count and count <= 10000 and count > 0 and len(ids) < count))


def _history_rescue(base_term):
    """Retrieve ALL UIDs for a term via the History server + ESummary paging
    (works beyond the 10,000 ESearch window). Returns (ids, reported, ok, err)."""
    res, err = esearch(base_term, 0, 0, usehistory=True)
    if err or not res or not res.get("webenv"):
        return [], (res["count"] if res else None), False, (err or "no_webenv")
    webenv, qk, count = res["webenv"], res["querykey"], res["count"]
    ids = []
    retstart = 0
    while retstart < count:
        params = dict(db="pmc", retmode="json", version="2.0",
                      WebEnv=webenv, query_key=qk,
                      retstart=retstart, retmax=C.SUMMARY_BATCH)
        r = C.ncbi_get("esummary", params, want="text")
        if not r.ok:
            return ids, count, False, ("esummary %s" % r.error)
        try:
            j = json.loads(r.text)
        except json.JSONDecodeError as e:
            return ids, count, False, ("esummary json %r" % e)
        uids = (j.get("result", {}) or {}).get("uids", []) or []
        if not uids:
            break
        ids.extend(uids)
        retstart += C.SUMMARY_BATCH
    return ids, count, True, None


# ---------------- subdivision ----------------
def _fmt(d):
    return "%04d/%02d/%02d" % (d.year, d.month, d.day)


def _dated_term(base_term, lo, hi):
    return '(%s) AND ("%s"[PDAT] : "%s"[PDAT])' % (base_term, _fmt(lo), _fmt(hi))


class SearchRunner:
    def __init__(self, item):
        self.item = item
        self.sid = item["search_id"]
        self.base_term = item["query_text"]
        self.dir = C.P("results", "search", self.sid)
        os.makedirs(self.dir, exist_ok=True)
        self.manifest_path = os.path.join(self.dir, "manifest.json")
        self.manifest = C.read_json(self.manifest_path) or {}
        self.partitions = self.manifest.get("partitions", [])
        self.pcount = 0

    def _leaf_path(self, key):
        return os.path.join(self.dir, "leaf_%s.json" % key)

    def _leaf_done(self, key):
        return os.path.exists(self._leaf_path(key))

    def _save_leaf(self, key, ids, meta):
        # fsync=False: leaves/manifests are regenerable & idempotent on resume;
        # skipping the Lustre sync avoids per-call cl_sync_io_wait stalls.
        C.write_json(self._leaf_path(key), dict(ids=sorted(set(ids)),
                                                n=len(set(ids)), **meta), fsync=False)

    def _save_manifest(self, **extra):
        self.manifest.update(extra)
        self.manifest["partitions"] = self.partitions
        self.manifest["updated_at"] = C.now_iso()
        C.write_json(self.manifest_path, self.manifest, fsync=False)

    def _add_partition(self, rec):
        # replace if same partition_id exists
        self.partitions = [p for p in self.partitions if p["partition_id"] != rec["partition_id"]]
        self.partitions.append(rec)

    def _range_key(self, lo, hi):
        return "%s_%s" % (_fmt(lo).replace("/", ""), _fmt(hi).replace("/", ""))

    def _collect_range(self, lo, hi, parent_id, depth):
        """Recursively retrieve a date range, subdividing when over threshold."""
        key = self._range_key(lo, hi)
        pid = "%s:%s" % (self.sid, key)
        if self._leaf_done(key):
            return  # resumable: this leaf already retrieved
        term = _dated_term(self.base_term, lo, hi)
        res, err = esearch(term, 0, C.SEARCH_PAGE)  # count + first page in one call
        if err:
            self._add_partition(dict(partition_id=pid, parent_id=parent_id,
                                     range=[_fmt(lo), _fmt(hi)], submitted_query=term,
                                     reported_count=None, retrieved_unique=0,
                                     state="service-failure", retry_count=0,
                                     warnings=None, errors=err, depth=depth))
            self._save_manifest()
            return
        count = res["count"]
        if count == 0:
            self._save_leaf(key, [], dict(range=[_fmt(lo), _fmt(hi)], reported=0,
                                          state="complete-zero"))
            self._add_partition(dict(partition_id=pid, parent_id=parent_id,
                                     range=[_fmt(lo), _fmt(hi)], submitted_query=term,
                                     reported_count=0, retrieved_unique=0,
                                     state="complete-zero", retry_count=0,
                                     warnings=res["warninglist"], errors=res["errorlist"],
                                     depth=depth))
            self._save_manifest()
            return
        if count <= C.PARTITION_THRESHOLD:
            rr = _retrieve_direct(term, first=res)  # reuse the count+page1 fetch
            ids = rr["ids"]
            state = "complete-with-results"
            if not rr.get("ok") or rr.get("incomplete"):
                state = "incomplete"
            self._save_leaf(key, ids, dict(range=[_fmt(lo), _fmt(hi)], reported=count,
                                           state=state, pages=rr.get("pages")))
            self._add_partition(dict(partition_id=pid, parent_id=parent_id,
                                     range=[_fmt(lo), _fmt(hi)], submitted_query=term,
                                     reported_count=count, retrieved_unique=len(set(ids)),
                                     state=state, retry_count=0,
                                     warnings=rr.get("warnings"), errors=rr.get("errors"),
                                     depth=depth))
            self.pcount += 1
            if self.pcount % 10 == 0:
                self._save_manifest()
            return
        # over threshold: subdivide
        if lo >= hi:
            # single day still over window -> History rescue so we never truncate
            ids, reported, ok, herr = _history_rescue(term)
            state = "complete-with-results" if ok else "incomplete"
            self._save_leaf(key, ids, dict(range=[_fmt(lo), _fmt(hi)], reported=count,
                                           state=state, method="history_rescue_single_day"))
            self._add_partition(dict(partition_id=pid, parent_id=parent_id,
                                     range=[_fmt(lo), _fmt(hi)], submitted_query=term,
                                     reported_count=count, retrieved_unique=len(set(ids)),
                                     state=state, retry_count=0, method="history_rescue",
                                     warnings=None, errors=herr, depth=depth))
            self._save_manifest()
            return
        # binary split on the date ordinal
        self._add_partition(dict(partition_id=pid, parent_id=parent_id,
                                 range=[_fmt(lo), _fmt(hi)], submitted_query=term,
                                 reported_count=count, retrieved_unique=0,
                                 state="internal-split", retry_count=0,
                                 warnings=None, errors=None, depth=depth))
        self._save_manifest()
        lo_ord, hi_ord = lo.toordinal(), hi.toordinal()
        mid = datetime.date.fromordinal((lo_ord + hi_ord) // 2)
        mid_next = datetime.date.fromordinal(mid.toordinal() + 1)
        self._collect_range(lo, mid, pid, depth + 1)
        self._collect_range(mid_next, hi, pid, depth + 1)

    def _union_ids(self):
        ids = set()
        for fn in os.listdir(self.dir):
            if fn.startswith("leaf_") and fn.endswith(".json"):
                leaf = C.read_json(os.path.join(self.dir, fn)) or {}
                ids.update(leaf.get("ids", []))
        return ids

    def run(self):
        if self.manifest.get("complete"):
            return self.manifest
        started = self.manifest.get("submitted_at") or C.now_iso()
        if not self.item.get("valid", True):
            self._save_manifest(submitted_at=started, completion_status="invalid-query",
                                validation=self.item.get("validation"),
                                reported_count_total=0, retrieved_unique_total=0,
                                pmcids_count=0, strategy="none", complete=True,
                                finished_at=C.now_iso())
            # also write empty pmcids file for collation
            C.write_json(os.path.join(self.dir, "pmcids.json"), [], fsync=False)
            return self.manifest

        # Top-level: fetch count AND first page in one call (saves a call for the
        # many small molecule queries that don't need subdivision).
        _page = (min(SEARCH_CAP, C.SEARCH_PAGE) if SEARCH_CAP > 0 else C.SEARCH_PAGE)
        top, err = esearch(self.base_term, 0, _page)
        if err:
            attempts = self.manifest.get("top_attempts", 0) + 1
            # bounded retry: after MAX_RETRIES persistent top-level failures, finalize
            final = attempts >= C.MAX_RETRIES
            self._save_manifest(submitted_at=started,
                                completion_status="service-failure",
                                top_error=err, top_attempts=attempts, complete=final)
            if final:
                C.write_json(os.path.join(self.dir, "pmcids.json"), [], fsync=False)
            return self.manifest
        total = top["count"]
        top_rec = dict(reported_count=total, querytranslation=top["querytranslation"],
                       warninglist=top["warninglist"], errorlist=top["errorlist"],
                       translation_altered=bool(top["warninglist"] or (top["errorlist"] and (
                           (top["errorlist"].get("phrasesnotfound") or []) or
                           (top["errorlist"].get("fieldsnotfound") or [])))))
        self.manifest["top_level"] = top_rec
        self.manifest["submitted_query"] = self.base_term
        self.manifest["query_ids"] = self.item.get("query_ids")
        self.manifest["molecule_ids"] = self.item.get("molecule_ids")
        self.manifest["query_types"] = self.item.get("query_types")
        # Raw search-response provenance (also the live progress signal).
        C.write_json(C.P("raw_search_responses", "%s.json" % self.sid),
                     dict(search_id=self.sid, submitted_query=self.base_term,
                          esearchresult=top_rec, retrieved_at=C.now_iso()), fsync=False)

        if total == 0:
            C.write_json(os.path.join(self.dir, "pmcids.json"), [], fsync=False)
            self._save_manifest(submitted_at=started, strategy="direct",
                                reported_count_total=0, retrieved_unique_total=0,
                                pmcids_count=0, completion_status="complete-zero",
                                complete=True, finished_at=C.now_iso())
            return self.manifest

        # Synonym-augmentation searches are NOT subdivided: a 120-term synonym
        # OR-chain can contain a common/noise term matching millions of articles,
        # and full [PDAT] subdivision would take ~130 calls/search (hours) while
        # attaching huge noise sets to the molecule. For synonym->molecule
        # association the first-10k window is sufficient; over-window searches are
        # retrieved up to 10k and marked complete-capped-10k (reported vs retrieved
        # recorded for transparency).
        no_subdivide = (SEARCH_CAP > 0) or ("synonym_augmentation" in (self.item.get("query_types") or []))
        if total <= C.PARTITION_THRESHOLD or no_subdivide:
            if SEARCH_CAP > 0:
                # deadline mode: single page already fetched; take up to CAP, no paging/subdivision
                rr = dict(ok=True, ids=(top["idlist"] or [])[:SEARCH_CAP], reported=total, pages=1,
                          warnings=top["warninglist"], errors=top["errorlist"])
            else:
                rr = _retrieve_direct(self.base_term, first=top)  # reuse count+page1; caps at 10k window
            ids = rr["ids"]
            capped = total > len(ids)
            state = ("complete-capped" if capped else "complete-with-results")
            if not rr.get("ok"):
                state = "incomplete"
            self._save_leaf("direct", ids, dict(range=["all", "all"], reported=total,
                                                 state=state, pages=rr.get("pages")))
            self.partitions = [dict(partition_id="%s:direct" % self.sid, parent_id=None,
                                    range=["all", "all"], submitted_query=self.base_term,
                                    reported_count=total, retrieved_unique=len(set(ids)),
                                    state=state, retry_count=0, warnings=rr.get("warnings"),
                                    errors=rr.get("errors"), depth=0)]
            strategy = ("direct-capped" if capped else "direct")
        else:
            self.manifest["strategy"] = "subdivided"
            self._save_manifest(submitted_at=started, strategy="subdivided")
            self._collect_range(EARLIEST, LATEST, None, 0)
            strategy = "subdivided"

        uni = self._union_ids()
        # Reconciliation vs reported total.
        recon = dict(reported_total=total, retrieved_unique=len(uni),
                     gap=total - len(uni))
        completion = "complete-with-results"
        if no_subdivide and total > len(uni):
            # Intentional cap (synonym batch OR deadline SEARCH_CAP); do NOT history-rescue.
            completion = "complete-capped-10k"
            recon["explanation"] = ("search intentionally capped (no subdivision / depth cap); "
                                    "enough candidates for BM25 prescreen")
        # If a gap remains (missing-PDAT / boundary records), History rescue the whole query.
        elif len(uni) < total:
            rescue_ids, rep, ok, herr = _history_rescue(self.base_term)
            before = len(uni)
            uni.update(rescue_ids)
            recon["history_rescue"] = dict(ok=ok, error=herr, added=len(uni) - before,
                                           reported=rep)
            if len(uni) < total:
                completion = "count-discrepancy"
                recon["explanation"] = ("retrieved_unique < reported_count after "
                                        "subdivision and history rescue; residual gap "
                                        "likely records lacking a parseable [PDAT] or "
                                        "transient count drift")
            self._save_leaf("history_rescue", rescue_ids,
                            dict(range=["all", "all"], reported=total,
                                 state=("complete-with-results" if ok else "incomplete"),
                                 method="history_rescue_reconcile"))
        # any partition incomplete / failed -> overall incomplete
        states = [p["state"] for p in self.partitions]
        if any(s in ("incomplete", "service-failure") for s in states):
            if completion == "complete-with-results":
                completion = "incomplete"

        pmcids = sorted(uni)
        C.write_json(os.path.join(self.dir, "pmcids.json"), pmcids, fsync=False)
        self._save_manifest(submitted_at=started, strategy=strategy,
                            reported_count_total=total,
                            retrieved_unique_total=len(pmcids),
                            pmcids_count=len(pmcids), reconciliation=recon,
                            completion_status=completion,
                            complete=(completion in ("complete-with-results", "complete-zero",
                                                      "complete-capped-10k", "count-discrepancy")),
                            finished_at=C.now_iso())
        C.record_usage("search_done", search_id=self.sid, reported=total,
                       retrieved=len(pmcids), status=completion)
        return self.manifest


def result_dir(sid):
    return C.P("results", "search", sid)


def is_done(sid):
    # Cheap: pmcids.json is written only at a terminal (complete) state, so a stat
    # suffices and avoids reading+parsing the (possibly large) manifest on every
    # feeder scan across thousands of searches.
    return os.path.exists(os.path.join(result_dir(sid), "pmcids.json"))


def process_item(sid, items):
    if is_done(sid):
        return "cached"
    item = items[sid]
    m = SearchRunner(item).run()
    return m.get("completion_status", "unknown")
