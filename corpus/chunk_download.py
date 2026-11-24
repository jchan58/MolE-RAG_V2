"""
Chunked, paced, per-entity full-text download + metadata (user-directed; replaces
the single 8.1M bulk pass).

ORDER: (1) property task first (its PMCIDs -> OA subset), then (2) molecules in
small concurrent batches, sorted ASCENDING by n_pmcids (smallest first) so the
~1,700 clean molecules finish fast and the mega-breadth (generic-term-noise)
molecules land at the end.

PER ENTITY: oa_gate each PMCID (S3 LIST) -> download OA XML (S3 GET) -> ESummary
metadata -> write a per-entity manifest.

GUARDS:
 * GLOBAL dedup: each unique PMCID is oa_gated/downloaded/metadata'd exactly once
   (per-PMCID result files are the authority; an in-memory set avoids re-stat).
   Per-molecule JSON (molecule_pmcids.json) still lists ALL its PMCIDs as pointers.
 * Lustre-safe: sharded output dirs (download.py / oa_gate.py shard by PMCID), all
   fsync-light + atomic rename, bounded S3 concurrency.
 * Resumable: skip done entities (entity manifest present) and done PMCIDs (is_done).
 * HARD CHECKPOINT: molecules with n_pmcids > MEGA_LIMIT are NOT processed; the run
   stops at that boundary and reports them pending a go-ahead.
 * Progress -> state/CHUNK_PROGRESS.json every batch.
"""
import os, sys, json, time, threading
from concurrent.futures import ThreadPoolExecutor, wait
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import oa_gate as OA
import download as DL
import metadata as MD

S3_THREADS = 48
MEGA_LIMIT = 50000      # molecules above this many PMCIDs await an explicit go-ahead
MOL_BATCH = 16          # molecules processed concurrently per batch

PROGRESS = lambda: C.P("state", "CHUNK_PROGRESS.json")
GLOBAL_DONE = set()
_lock = threading.Lock()
_ctr = dict(downloads_succeeded=0, xml_bytes=0, oa_in_s3=0)


def _entity_marker(eid):
    return C.P("results", "entity_manifests", eid.replace("/", "_") + ".json")


def _entity_done(eid):
    return os.path.exists(_entity_marker(eid))


def _process_pmcid(pmcid):
    try:
        if not OA.is_done(pmcid):
            OA.process_item(pmcid)
        r = C.read_json(OA._res_path(pmcid))
        if r and r.get("in_s3"):
            with _lock:
                _ctr["oa_in_s3"] += 1
            if not DL.is_done(pmcid):
                DL.process_item(pmcid)
            dr = C.read_json(DL._res_path(pmcid))
            if dr and dr.get("status") in ("success", "success_no_body"):
                with _lock:
                    _ctr["downloads_succeeded"] += 1
                    _ctr["xml_bytes"] += dr.get("bytes") or 0
    except Exception as e:
        C.log("WARN", "chunk", "pmcid %s: %r" % (pmcid, e))


def _process_group(entities, pool):
    """entities: list of (entity_id, pmcids). S3-process all new PMCIDs, then
    metadata, then per-entity manifests."""
    seen = set()
    todo = []
    for eid, pmcids in entities:
        for p in pmcids:
            if p in seen:
                continue
            seen.add(p)
            if p not in GLOBAL_DONE:
                todo.append(p)
    if todo:
        futs = [pool.submit(_process_pmcid, p) for p in todo]
        wait(futs)
    with _lock:
        GLOBAL_DONE.update(seen)
    # metadata for this group's PMCIDs (global dedup via per-PMCID file existence)
    MD.process_pmcids(list(seen))
    # per-entity manifests
    for eid, pmcids in entities:
        oa = dl = byt = 0
        for p in pmcids:
            dr = C.read_json(DL._res_path(p))
            if dr and dr.get("status") in ("success", "success_no_body"):
                dl += 1
                byt += dr.get("bytes") or 0
            orr = C.read_json(OA._res_path(p))
            if orr and orr.get("in_s3"):
                oa += 1
        C.write_json(_entity_marker(eid), dict(entity=eid, n_pmcids=len(pmcids),
                     n_in_s3=oa, n_downloaded=dl, xml_bytes=byt,
                     done_at=C.now_iso()), fsync=False)


def _write_progress(phase, entities_completed, pending_mega, note=""):
    C.write_json(PROGRESS(), dict(
        updated_at=C.now_iso(), phase=phase, entities_completed=entities_completed,
        unique_pmcids_seen=len(GLOBAL_DONE),
        downloads_succeeded_this_run=_ctr["downloads_succeeded"],
        xml_bytes_this_run=_ctr["xml_bytes"], xml_human=C.human(_ctr["xml_bytes"]),
        oa_in_s3_seen=_ctr["oa_in_s3"],
        n_pending_mega=len(pending_mega),
        pending_mega_molecules=pending_mega, note=note))


def run():
    C.ensure_dirs()
    os.makedirs(C.P("results", "entity_manifests"), exist_ok=True)
    C.atomic_write_text(C.P("state", "chunk.pid"), str(os.getpid()))
    C.log("INFO", "chunk", "=== chunked per-entity download starting (pid %d, %d S3 threads) ==="
          % (os.getpid(), S3_THREADS))
    pool = ThreadPoolExecutor(max_workers=S3_THREADS)
    completed = 0

    # Phase 1: property task
    prop = C.read_json(C.P("data", "property_task_pmcids.json"), {}) or {}
    prop_pmcids = prop.get("pmcids", [])
    C.log("INFO", "chunk", "PHASE 1 property task: %d PMCIDs" % len(prop_pmcids))
    if not _entity_done("property_task"):
        _process_group([("property_task", prop_pmcids)], pool)
    completed += 1
    _write_progress("property_done", completed, [], "property task: OA-gated + downloaded + metadata")
    C.log("INFO", "chunk", "property task done: unique_seen=%d downloaded_this_run=%d oa_in_s3=%d"
          % (len(GLOBAL_DONE), _ctr["downloads_succeeded"], _ctr["oa_in_s3"]))

    # Phase 2: molecules ascending by n_pmcids
    mol = C.read_json(C.P("data", "molecule_pmcids.json"), {}) or {}
    items = sorted(mol.items(), key=lambda kv: kv[1].get("n_pmcids", 0))
    pending_mega = [[m, info.get("n_pmcids", 0)] for m, info in items
                    if info.get("n_pmcids", 0) > MEGA_LIMIT]
    normal = [(m, info) for m, info in items if info.get("n_pmcids", 0) <= MEGA_LIMIT]
    C.log("INFO", "chunk", "PHASE 2: %d normal molecules (<=%d PMCIDs), %d mega pending checkpoint"
          % (len(normal), MEGA_LIMIT, len(pending_mega)))

    batch = []
    for m, info in normal:
        if _entity_done(m):
            completed += 1
            continue
        batch.append((m, info.get("pmcids", [])))
        if len(batch) >= MOL_BATCH:
            _process_group(batch, pool)
            completed += len(batch)
            batch = []
            _write_progress("molecules", completed, pending_mega)
    if batch:
        _process_group(batch, pool)
        completed += len(batch)
    pool.shutdown(wait=True)

    if pending_mega:
        _write_progress("STOP_mega_checkpoint", completed, pending_mega,
                        "HARD CHECKPOINT: %d mega molecules (n_pmcids>%d) NOT downloaded; "
                        "awaiting go-ahead (largest: %s)" % (
                            len(pending_mega), MEGA_LIMIT,
                            sorted(pending_mega, key=lambda x: -x[1])[:5]))
        C.write_json(C.P("state", "CHUNK_MEGA_CHECKPOINT.json"),
                     dict(created_at=C.now_iso(), mega_limit=MEGA_LIMIT,
                          pending=sorted(pending_mega, key=lambda x: -x[1])))
        C.log("INFO", "chunk", "HARD CHECKPOINT reached: %d mega molecules pending decision. STOP."
              % len(pending_mega))
    else:
        _write_progress("done", completed, [], "all entities downloaded")
        C.atomic_write_text(C.P("state", "DONE"), C.now_iso())
        C.log("INFO", "chunk", "chunked download fully complete")


if __name__ == "__main__":
    run()
