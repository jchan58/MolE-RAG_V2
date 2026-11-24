"""
Download (S3 OA-gate + GET) the CLEAN full-text set (data/clean_pmcids.txt) that is
not already downloaded. Threaded, off the E-utilities cap, idempotent (skips
already-done PMCIDs). Metadata (ESummary) for the clean set too.
"""
import os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, wait
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import oa_gate as OA
import download as DL
import metadata as MD

S3_THREADS = 48
_ctr = dict(done=0, downloaded=0, in_s3=0, bytes=0)
_lock = threading.Lock()


def _one(pmcid):
    try:
        if not OA.is_done(pmcid):
            OA.process_item(pmcid)
        r = C.read_json(OA._res_path(pmcid))
        if r and r.get("in_s3"):
            with _lock:
                _ctr["in_s3"] += 1
            if not DL.is_done(pmcid):
                DL.process_item(pmcid)
            dr = C.read_json(DL._res_path(pmcid))
            if dr and dr.get("status") in ("success", "success_no_body"):
                with _lock:
                    _ctr["downloaded"] += 1
                    _ctr["bytes"] += dr.get("bytes") or 0
    except Exception as e:
        C.log("WARN", "clean_dl", "pmcid %s: %r" % (pmcid, e))
    finally:
        with _lock:
            _ctr["done"] += 1


def run():
    C.ensure_dirs()
    C.atomic_write_text(C.P("state", "clean_download.pid"), str(os.getpid()))
    with open(C.P("data", "clean_pmcids.txt")) as f:
        clean = [x.strip() for x in f if x.strip()]
    todo = [p for p in clean if not DL.is_done(p)]
    C.log("INFO", "clean_dl", "clean set %d; %d need download/oa_gate on %d threads"
          % (len(clean), len(todo), S3_THREADS))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=S3_THREADS) as ex:
        futs = [ex.submit(_one, p) for p in todo]
        last = time.time()
        for _ in as_completed_safe(futs):
            if time.time() - last > 20:
                _progress(len(todo), t0)
                last = time.time()
    # metadata for the clean set (ESummary, batched, dedup via file existence)
    C.log("INFO", "clean_dl", "metadata for clean set (%d pmcids)" % len(clean))
    MD.process_pmcids(clean)
    _progress(len(todo), t0, final=True)
    C.log("INFO", "clean_dl", "clean download done: %s" % _ctr)


def as_completed_safe(futs):
    from concurrent.futures import as_completed
    for f in as_completed(futs):
        try:
            f.result()
        except Exception:
            pass
        yield f


def _progress(total, t0, final=False):
    with _lock:
        c = dict(_ctr)
    el = max(time.time() - t0, 1e-6)
    C.write_json(C.P("state", "CLEAN_DL_PROGRESS.json"),
                 dict(updated_at=C.now_iso(), total_todo=total, done=c["done"],
                      in_s3=c["in_s3"], downloaded=c["downloaded"],
                      xml_bytes=c["bytes"], xml_human=C.human(c["bytes"]),
                      rate_per_s=round(c["done"] / el, 1),
                      phase=("done" if final else "downloading")))


if __name__ == "__main__":
    run()
