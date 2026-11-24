"""
Stage 10 — metadata via ESummary (db=pmc, batched) + any downloaded article XML.

Bibliographic / identifier / access metadata comes from ESummary (JSON). Abstract,
keywords, language, license/rights and relationship (correction/retraction) info are
parsed from the full-text XML when it has been downloaded. Nothing is LLM-inferred;
missing values are null/empty. Conflicting dates are preserved with their source.
Metadata is retained even when no full text is available. Metadata failures are
recorded separately from download failures.

Items are deterministic batches over data/registry_pmcids.txt (sorted).
Per-PMCID result: results/metadata/<shard>/<pmcid>.json
Batch marker:     results/metadata_batches/batch_<i>.json
Raw ESummary:     raw_metadata/esummary_batch_<i>.json
"""
import os, sys, json
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import merge as M

BATCH = C.SUMMARY_BATCH


def _batches():
    pm = M.registry_pmcids()
    return [pm[i:i + BATCH] for i in range(0, len(pm), BATCH)]


def n_batches():
    pm = M.registry_pmcids()
    return (len(pm) + BATCH - 1) // BATCH


def batch_ids(i):
    pm = M.registry_pmcids()
    return pm[i * BATCH:(i + 1) * BATCH]


def _meta_path(pmcid):
    return C.P("results", "metadata", M.shard_of(pmcid), "%s.json" % pmcid)


def _batch_marker(i):
    return C.P("results", "metadata_batches", "batch_%06d.json" % i)


def is_done(i):
    return C.read_json(_batch_marker(i)) is not None


def _parse_xml_meta(pmcid):
    """Best-effort extraction of abstract/keywords/language/license/relationships
    from a downloaded full-text XML, if present."""
    import download as D
    path = D.xml_path(pmcid)
    if not os.path.exists(path):
        return dict(xml_available=False)
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return dict(xml_available=True, xml_parse_error=True)
    root = tree.getroot()
    art = root.find(".//article")
    if art is None:
        return dict(xml_available=True, article_node=False)
    # abstract
    abstract = None
    ab = art.find(".//front//abstract")
    if ab is not None:
        abstract = " ".join("".join(p.itertext()).strip() for p in ab.iter() if p.tag == "p") \
            or "".join(ab.itertext()).strip()
    # keywords
    kwds = [k.text.strip() for k in art.findall(".//kwd") if k.text]
    # language
    lang = art.get("{http://www.w3.org/XML/1998/namespace}lang")
    # license / rights
    lic = None
    lic_node = art.find(".//permissions/license")
    if lic_node is not None:
        lic = dict(license_type=lic_node.get("license-type"),
                   href=lic_node.get("{http://www.w3.org/1999/xlink}href"),
                   text=" ".join("".join(lic_node.itertext()).split())[:1000])
    copyright_stmt = None
    cs = art.find(".//permissions/copyright-statement")
    if cs is not None:
        copyright_stmt = "".join(cs.itertext()).strip()
    # relationships (corrections / retractions) via related-article
    rels = []
    for ra in art.findall(".//related-article"):
        rels.append(dict(related_article_type=ra.get("related-article-type"),
                         href=ra.get("{http://www.w3.org/1999/xlink}href"),
                         ext_link_type=ra.get("ext-link-type")))
    has_body = art.find(".//body") is not None
    return dict(xml_available=True, article_node=True, has_body=has_body,
                abstract=abstract, keywords=kwds, language_xml=lang,
                license=lic, copyright_statement=copyright_stmt,
                relationships=rels)


def _parse_docsum(ds):
    def arts(idtype):
        out = []
        for aid in ds.get("articleids", []) or []:
            if aid.get("idtype") == idtype and aid.get("value"):
                out.append(aid["value"])
        return out
    doi = arts("doi")
    pmid = arts("pmid")
    pmcid_ids = arts("pmcid") or arts("pmc")
    return dict(
        esummary_title=ds.get("title"),
        source_journal=ds.get("fulljournalname") or ds.get("source"),
        journal_abbrev=ds.get("source"),
        authors=[a.get("name") for a in (ds.get("authors") or []) if a.get("name")],
        pubdate=ds.get("pubdate"), epubdate=ds.get("epubdate"),
        sortpubdate=ds.get("sortpubdate"),
        volume=ds.get("volume"), issue=ds.get("issue"), pages=ds.get("pages"),
        doi=(doi[0] if doi else None), all_doi=doi,
        pmid=(pmid[0] if pmid else None),
        pmcid=(pmcid_ids[0] if pmcid_ids else None),
        all_article_ids=ds.get("articleids"),
        pubtype=ds.get("pubtype"), lang=ds.get("lang"),
        sortfirstauthor=ds.get("sortfirstauthor"),
        dates=dict(pubdate=ds.get("pubdate"), epubdate=ds.get("epubdate"),
                   sortpubdate=ds.get("sortpubdate"),
                   printpubdate=ds.get("printpubdate")),
    )


def has_meta(pmcid):
    return os.path.exists(_meta_path(pmcid))


def _summarize(ids):
    """ESummary a list of PMCIDs and write per-PMCID metadata files (idempotent by
    file existence). Returns (n_written, n_failed)."""
    if not ids:
        return 0, 0
    params = dict(db="pmc", id=",".join(str(x) for x in ids), retmode="json", version="2.0")
    r = C.ncbi_post("esummary", params, want="text")
    if not r.ok:
        return 0, len(ids)
    try:
        result = (json.loads(r.text).get("result", {}) or {})
    except json.JSONDecodeError:
        return 0, len(ids)
    n = 0
    for p in ids:
        ds = result.get(str(p))
        rec = dict(pmcid=p, pmcid_full="PMC%s" % p, fetched_at=C.now_iso())
        if ds is None or (isinstance(ds, dict) and ds.get("error")):
            rec["metadata_status"] = "no-docsum"
            if isinstance(ds, dict):
                rec["error"] = ds.get("error")
        else:
            rec["metadata_status"] = "success"
            rec.update(_parse_docsum(ds))
        rec.update(_parse_xml_meta(p))
        C.write_json(_meta_path(p), rec, fsync=False)
        n += 1
    return n, 0


def process_pmcids(pmcids):
    """Fetch metadata for a set of PMCIDs that lack a metadata file (global dedup
    via per-PMCID file existence). Batched ESummary. Returns n_fetched."""
    todo = [p for p in pmcids if not has_meta(p)]
    fetched = 0
    for i in range(0, len(todo), BATCH):
        n, _f = _summarize(todo[i:i + BATCH])
        fetched += n
    return fetched


def process_batch(i):
    if is_done(i):
        return "cached"
    ids = batch_ids(i)
    if not ids:
        C.write_json(_batch_marker(i), dict(n=0, done_at=C.now_iso()))
        return "empty"
    params = dict(db="pmc", id=",".join(str(x) for x in ids),
                  retmode="json", version="2.0")
    r = C.ncbi_post("esummary", params, want="text")
    raw_path = C.P("raw_metadata", "esummary_batch_%06d.json" % i)
    C.atomic_write_text(raw_path, r.text or (r.error or ""))
    if not r.ok:
        # metadata-service failure for the batch: record, do NOT mark done (retry)
        C.log("WARN", "metadata", "batch %d esummary failed: %s" % (i, r.error))
        side = _batch_marker(i).replace(".json", ".attempts")
        prev = 0
        try:
            prev = int(open(side).read().strip() or "0")
        except Exception:
            prev = 0
        prev += 1
        C.atomic_write_text(side, str(prev))
        if prev >= C.MAX_RETRIES:
            # exhausted: write per-pmcid metadata-failure records and finalize
            for p in ids:
                rec = dict(pmcid=p, pmcid_full="PMC%s" % p, metadata_status="service-failure",
                           error=r.error, raw_response_path=raw_path, fetched_at=C.now_iso())
                rec.update(_parse_xml_meta(p))
                C.write_json(_meta_path(p), rec)
            C.write_json(_batch_marker(i), dict(n=len(ids), status="service-failure-exhausted",
                                                done_at=C.now_iso()))
            return "service-failure-exhausted"
        return "retry"
    try:
        j = json.loads(r.text)
    except json.JSONDecodeError as e:
        C.log("WARN", "metadata", "batch %d esummary json error %r" % (i, e))
        return "retry"
    result = j.get("result", {}) or {}
    for p in ids:
        ds = result.get(str(p))
        rec = dict(pmcid=p, pmcid_full="PMC%s" % p, fetched_at=C.now_iso(),
                   raw_response_path=raw_path)
        if ds is None or (isinstance(ds, dict) and ds.get("error")):
            rec["metadata_status"] = "no-docsum"
            if isinstance(ds, dict):
                rec["error"] = ds.get("error")
        else:
            rec["metadata_status"] = "success"
            rec.update(_parse_docsum(ds))
        rec.update(_parse_xml_meta(p))
        C.write_json(_meta_path(p), rec)
    C.write_json(_batch_marker(i), dict(n=len(ids), status="success", done_at=C.now_iso()))
    C.record_usage("metadata_batch", batch=i, n=len(ids))
    return "success"
