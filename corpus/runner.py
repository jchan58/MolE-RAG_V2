"""
Shared stage-draining loop used by both the driver and external workers.

A stage has: an item-id generator, a process(item) function, and a done(item)
predicate. Items are claimed with an atomic O_EXCL lock (lease + heartbeat); the
authoritative done-signal is the per-item result file, so claim races are harmless
(idempotent writes). Multiple processes sharing RUN_DIR can drain the same stage
concurrently; a per-process feeder/consumer thread pool adds intra-process
concurrency so a few processes saturate each service's central rate cap.
"""
import os, sys, time, threading, traceback, random
try:
    import queue as _queue
except ImportError:  # pragma: no cover
    import Queue as _queue
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# Parallelizable stages that workers may join. (Identity resolution is now an
# offline driver-only step, not a per-item HTTP stage.)
PARALLEL_STAGES = ["querygen_molecule", "search", "search_aug", "oa_gate", "metadata", "download"]

# Default per-process thread concurrency per stage (IO-bound network work).
# PubChem is capped at a few req/s and throttles bursts hard, so keep it low
# (concurrency gives ~no speedup past the global cap and only triggers 429s).
DEFAULT_THREADS = dict(pubchem=2, querygen_molecule=16, search=12, search_aug=8,
                       oa_gate=32, metadata=4, download=32)


def stage_spec(stage):
    """Return (items_fn, process_fn, done_fn) for a parallelizable stage."""
    if stage == "pubchem":
        import pubchem as PC
        structs = PC.structures_by_id()
        return (PC.item_ids, lambda it: PC.process_item(it, structs), PC.is_done)
    if stage == "querygen_molecule":
        import querygen as QG
        return (QG.molecule_item_ids, QG.generate_molecule_queries, QG.molecule_done)
    if stage == "search":
        import search as S
        items = S.search_items()
        return (lambda: list(items.keys()), lambda it: S.process_item(it, items), S.is_done)
    if stage == "search_aug":
        import search as S, augment as AUG
        items = AUG.search_items_aug()
        return (lambda: list(items.keys()), lambda it: S.process_item(it, items), S.is_done)
    if stage == "oa_gate":
        import oa_gate as OA
        return (OA.item_ids, OA.process_item, OA.is_done)
    if stage == "metadata":
        import metadata as MD
        return (lambda: list(range(MD.n_batches())), MD.process_batch, MD.is_done)
    if stage == "download":
        import download as DL
        return (DL.item_ids, DL.process_item, DL.is_done)
    raise ValueError("unknown parallel stage %s" % stage)


def stage_done_set_fn(stage):
    """Optional CHEAP done-SET provider (one directory listing) for a stage, so (re)start
    revalidation avoids a per-item done() stat/read over tens of thousands of files on slow
    Lustre. Returns a callable -> set(str item_id) of already-done items, or None to fall
    back to the per-item done() predicate."""
    if stage == "querygen_molecule":
        import querygen as QG
        return QG.molecule_done_ids
    return None


def _compute_remaining(items_fn, done_fn, done_set_fn):
    """The not-yet-done items. Uses a one-listing done-SET when a stage provides one
    (querygen_molecule: 41k markers -> a single readdir); otherwise parallelizes the
    per-item done() check for large item sets (I/O-bound) so a scan is seconds, not hours."""
    items = items_fn()
    if done_set_fn is not None:
        done = done_set_fn()
        return [it for it in items if str(it) not in done]
    if len(items) > 2000:
        import concurrent.futures as _cf
        nthr = max(1, int(os.environ.get("EXP70_SCAN_THREADS", "48")))
        with _cf.ThreadPoolExecutor(max_workers=nthr) as ex:
            flags = list(ex.map(done_fn, items))
        return [it for it, dn in zip(items, flags) if not dn]
    return [it for it in items if not done_fn(it)]


class Heartbeat(threading.Thread):
    def __init__(self, lock, interval=60):
        super().__init__(daemon=True)
        self.lock = lock
        self.interval = interval
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(self.interval):
            self.lock.heartbeat()

    def stop(self):
        self._stop.set()


def _process_one(stage, it, process_fn, done_fn, counters, clock):
    # No per-item heartbeat thread: under many consumer threads those threads got
    # GIL-starved and failed to reap, leaking into the hundreds and choking the
    # process. We rely on the lease (LEASE_SECONDS) instead — items complete well
    # under it, and the rare long-running item that gets re-claimed is processed
    # idempotently (per-leaf/per-item checkpoints), so no work is lost.
    if done_fn(it):
        return
    lk = C.ItemLock(stage, str(it))
    if not lk.acquire():
        return
    try:
        if done_fn(it):
            return
        process_fn(it)
        with counters["lock"]:
            counters["processed"] += 1
    except Exception as e:
        C.log("ERROR", stage, "item %s failed: %r | %s"
              % (it, e, traceback.format_exc().splitlines()[-1]))
        _record_item_error(stage, it, e)
    finally:
        lk.release()


def drain_stage(stage, n_threads=None, tick=None, tick_interval=60,
                exit_when_empty=True, max_idle_polls=None):
    """Drain a stage with a feeder (one scanner) + N consumer threads.

    Rounds: the feeder scans the remaining (not-done) items, enqueues them, waits
    for the round to finish, then rescans. Terminates when a scan finds nothing
    remaining (driver mode) or when the stage is marked complete / idle polls
    exhausted (worker mode)."""
    items_fn, process_fn, done_fn = stage_spec(stage)
    done_set_fn = stage_done_set_fn(stage)
    if n_threads is None:
        n_threads = DEFAULT_THREADS.get(stage, 1)
    n_threads = max(1, int(n_threads))
    counters = {"processed": 0, "lock": threading.Lock()}
    clock = {"last_tick": time.time()}
    q = _queue.Queue(maxsize=n_threads * 4)
    stop = threading.Event()
    complete_marker = C.P("state", "stage_%s.complete" % stage)
    done_marker = C.P("state", "DONE")

    def consumer():
        while True:
            it = q.get()
            try:
                if it is None:
                    return
                _process_one(stage, it, process_fn, done_fn, counters, clock)
            finally:
                q.task_done()

    consumers = [threading.Thread(target=consumer, daemon=True) for _ in range(n_threads)]
    for t in consumers:
        t.start()

    idle = 0
    try:
        while not stop.is_set():
            remaining = _compute_remaining(items_fn, done_fn, done_set_fn)
            if not remaining:
                if exit_when_empty:
                    break
                if os.path.exists(complete_marker) or os.path.exists(done_marker):
                    break
                idle += 1
                if max_idle_polls and idle >= max_idle_polls:
                    break
                time.sleep(3)
                continue
            idle = 0
            before = counters["processed"]
            for it in remaining:
                # backpressure: q.put blocks when full; tick while we wait
                while True:
                    try:
                        q.put(it, timeout=5)
                        break
                    except _queue.Full:
                        _maybe_tick(tick, tick_interval, clock)
                _maybe_tick(tick, tick_interval, clock)
            q.join()  # wait for this round to finish
            _maybe_tick(tick, tick_interval, clock)
            if exit_when_empty and counters["processed"] == before:
                # a full round made no progress and items remain claimed by others
                time.sleep(2)
                # re-check: if still nothing done by us and nothing remaining, loop exits above
    finally:
        for _ in consumers:
            q.put(None)
        for t in consumers:
            t.join(timeout=30)
    return counters["processed"]


def _maybe_tick(tick, interval, clock):
    if not tick:
        return
    now = time.time()
    if now - clock["last_tick"] >= interval:
        try:
            tick()
        except Exception:
            pass
        clock["last_tick"] = now


def _record_item_error(stage, item, exc):
    d = C.P("state", "errors")
    os.makedirs(d, exist_ok=True)
    try:
        import fcntl, json
        with open(os.path.join(d, "%s.jsonl" % stage), "a") as f:
            fcntl.lockf(f, fcntl.LOCK_EX)
            f.write(json.dumps(dict(ts=C.now_iso(), item=str(item), error=repr(exc))) + "\n")
            fcntl.lockf(f, fcntl.LOCK_UN)
    except Exception:
        pass
