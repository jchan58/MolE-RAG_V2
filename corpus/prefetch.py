"""
S3 prefetcher — overlaps the off-cap PMC Open Data S3 work (OA membership LIST +
full-text GET) with the still-running, NCBI-rate-capped SEARCH stage.

Standalone + idempotent: harvests PMCIDs from already-COMPLETED searches
(results/search/<sid>/pmcids.json) and runs the same per-PMCID oa_gate + download
functions the driver would run later (is_done-guarded, keyed per PMCID). The
driver's later oa_gate/download stages then find most items done and finalize fast.

Uses a bounded work queue + N worker threads (so the huge backlog doesn't flood
memory, and the is_done checks + S3 calls happen in the worker threads, not serially
in the feeder). S3 is off the E-utilities cap, so this does not slow search.
"""
import os, sys, time, threading
try:
    import queue as _queue
except ImportError:
    import Queue as _queue
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import oa_gate as OA
import download as DL

N_THREADS = 48
SCAN_INTERVAL = 8


def _completed_search_dirs():
    """Dirs whose search is complete — detected by the presence of pmcids.json
    (a cheap stat, not a manifest read)."""
    base = C.P("results", "search")
    if not os.path.isdir(base):
        return []
    out = []
    for sid in os.listdir(base):
        if os.path.exists(os.path.join(base, sid, "pmcids.json")):
            out.append(os.path.join(base, sid))
    return out


def process_pmcid(pmcid):
    try:
        if not OA.is_done(pmcid):
            OA.process_item(pmcid)
        r = C.read_json(OA._res_path(pmcid))
        if r and r.get("in_s3") and not DL.is_done(pmcid):
            DL.process_item(pmcid)
    except Exception as e:
        C.log("WARN", "prefetch", "pmcid %s: %r" % (pmcid, e))


def run():
    C.ensure_dirs()
    C.atomic_write_text(C.P("state", "prefetch.pid"), str(os.getpid()))
    C.log("INFO", "prefetch", "S3 prefetcher start (pid %d, %d threads, s3 rate=%s)"
          % (os.getpid(), N_THREADS, C.service_rate("s3")))
    q = _queue.Queue(maxsize=N_THREADS * 8)
    stop = threading.Event()
    counters = {"done": 0, "lock": threading.Lock()}

    def worker():
        while True:
            p = q.get()
            try:
                if p is None:
                    return
                process_pmcid(p)
                with counters["lock"]:
                    counters["done"] += 1
            finally:
                q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(N_THREADS)]
    for t in threads:
        t.start()

    harvested = set()
    seen = set()
    # Run through the WHOLE pipeline (incl. the synonym-augmentation searches),
    # harvesting PMCIDs from every completed search dir (main + aug), until DONE.
    done_marker = C.P("state", "DONE")
    try:
        while not stop.is_set():
            new = 0
            for d in _completed_search_dirs():
                if d in harvested:
                    continue
                harvested.add(d)
                for p in (C.read_json(os.path.join(d, "pmcids.json"), []) or []):
                    if p not in seen:
                        seen.add(p)
                        q.put(p)  # blocks on backpressure
                        new += 1
            if new:
                C.log("INFO", "prefetch", "queued %d new PMCIDs (seen=%d done=%d qsize=%d)"
                      % (new, len(seen), counters["done"], q.qsize()))
            if os.path.exists(done_marker) and new == 0:
                # search finished and nothing new harvested -> drain and exit
                if not any(d not in harvested for d in _completed_search_dirs()):
                    break
            time.sleep(SCAN_INTERVAL)
        q.join()
    finally:
        for _ in threads:
            q.put(None)
        for t in threads:
            t.join(timeout=20)
    C.log("INFO", "prefetch", "S3 prefetcher done (seen=%d processed=%d)"
          % (len(seen), counters["done"]))


if __name__ == "__main__":
    run()
