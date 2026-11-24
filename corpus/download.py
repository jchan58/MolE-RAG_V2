"""
Stage 11 (FINAL design) — full-text retrieval via the PMC Open Data S3 Cloud
Service (sanctioned, OFF the E-utilities 10/s cap), with EFetch as a sanctioned
fallback only.

For each OA PMCID (membership determined by oa_gate from S3): GET the JATS .xml from
s3://pmc-oa-opendata, verify identity (PMCID match), parse license + has_body, save
raw XML unchanged (atomic), and record path/size/checksum/source/route/license. If the
S3 GET fails or the identity does not match, fall back to EFetch (db=pmc, retmode=xml).
Non-OA PMCIDs are metadata-only (full_text_available=false) and are NOT fetched here;
they are represented in the download manifest at collation time.

No relevance filtering before download. Atomic writes; validated files preserved on
resume (idempotent: skip if a terminal per-PMCID result already exists).

Items: data/download_targets.txt (the in-S3 OA PMCIDs).
Per-PMCID result: results/download/<shard>/<pmcid>.json
Full text:        fulltext_xml/<shard>/<pmcid>.xml
"""
import os, sys, re
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import merge as M
import oa_gate as OA
import s3util as S3

MAX_DL_ATTEMPTS = C.MAX_RETRIES


def xml_path(pmcid):
    return C.BP("fulltext_xml", M.shard_of(pmcid), "%s.xml" % pmcid)


def _res_path(pmcid):
    return C.P("results", "download", M.shard_of(pmcid), "%s.json" % pmcid)


def item_ids():
    return OA.download_targets()


def is_done(pmcid):
    r = C.read_json(_res_path(pmcid))
    if r is None:
        return False
    st = r.get("status")
    if st in ("success", "success_no_body", "unavailable", "identity-mismatch",
              "invalid-payload"):
        return True
    # temporary failures: done only once attempts are exhausted
    return r.get("attempts", 0) >= MAX_DL_ATTEMPTS


def _parse_article(data):
    """Return (identity_pmcid, has_body, license_info, ok, parse_error)."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None, None, None, False, True
    art = root if root.tag == "article" else root.find(".//article")
    if art is None:
        return None, None, None, False, False
    pid = None
    for aid in art.findall(".//article-id"):
        if (aid.get("pub-id-type") or "").lower() in ("pmcid", "pmc") and aid.text:
            pid = re.sub(r"^PMC", "", aid.text.strip(), flags=re.I)
            break
    has_body = art.find(".//body") is not None
    lic = None
    ln = art.find(".//permissions/license")
    if ln is not None:
        lic = dict(license_type=ln.get("license-type"),
                   href=ln.get("{http://www.w3.org/1999/xlink}href"),
                   text=" ".join("".join(ln.itertext()).split())[:300])
    return pid, has_body, lic, True, False


def _efetch_single(pmcid):
    params = dict(db="pmc", id=str(pmcid), retmode="xml")
    r = C.http_request("ncbi", "POST", C.ENDPOINTS["efetch"],
                       data=C.ncbi_params(params), want="bytes")
    if not r.ok or not r.content:
        return None, r.error
    return r.content, None


def _write(pmcid, **fields):
    rec = dict(pmcid=pmcid, pmcid_full="PMC%s" % pmcid, recorded_at=C.now_iso())
    rec.update(fields)
    C.write_json(_res_path(pmcid), rec, fsync=False)


def _save_and_record(pmcid, data, route, source_url, license_fallback=None):
    pid, has_body, lic, ok, parse_err = _parse_article(data)
    if parse_err or not ok:
        _write(pmcid, status="invalid-payload", attempts=MAX_DL_ATTEMPTS,
               full_text_available=False, retrieval_route=route,
               source_url=source_url, error="unparseable / no <article>")
        return "invalid-payload"
    if pid and str(pid) != str(pmcid):
        _write(pmcid, status="identity-mismatch", attempts=MAX_DL_ATTEMPTS,
               full_text_available=False, retrieval_route=route,
               source_url=source_url, identity_match=False, returned_pmcid=pid)
        return "identity-mismatch"
    xp = xml_path(pmcid)
    C.atomic_write_bytes(xp, data, fsync=False)
    sz = os.path.getsize(xp)
    chk = C.sha256_file(xp)
    _write(pmcid, status=("success" if has_body else "success_no_body"),
           full_text_available=bool(has_body), has_body=bool(has_body),
           identity_match=True, xml_path=xp, bytes=sz, sha256=chk,
           retrieval_route=route, source_url=source_url, payload_verified=True,
           license=(lic or license_fallback))
    return "success" if has_body else "success_no_body"


def process_item(pmcid):
    if is_done(pmcid):
        return "cached"
    key = OA.xml_key_for(pmcid)
    err = None
    if key:
        data, err = S3.get_object_bytes(key)
        if data is not None:
            url = "s3://%s/%s" % (S3.BUCKET, key)
            res = _save_and_record(pmcid, data, "s3_cloud_service", url)
            if res in ("success", "success_no_body"):
                C.record_usage("download_s3", pmcid=pmcid, status=res)
                return res
            # identity-mismatch / invalid -> try EFetch fallback below
    # --- EFetch fallback (sanctioned) ---
    fb, ferr = _efetch_single(pmcid)
    if fb is not None:
        url = "%s?db=pmc&id=%s&retmode=xml" % (C.ENDPOINTS["efetch"], pmcid)
        res = _save_and_record(pmcid, fb, "efetch_fallback", url)
        C.record_usage("download_efetch_fallback", pmcid=pmcid, status=res)
        return res
    # both failed: temporary failure, count attempts for bounded retry
    prev = (C.read_json(_res_path(pmcid)) or {}).get("attempts", 0) + 1
    _write(pmcid, status="temporary-failure", attempts=prev,
           full_text_available=False, retrieval_route="s3_cloud_service",
           error="s3: %s | efetch: %s" % (err, ferr))
    return "temporary-failure"
