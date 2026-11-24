"""
exp70-multi FRONT-HALF driver (per dataset, resumable, S3-checkpointed).

Stages: ingest -> identity (offline) -> per-task property querygen (Opus) ->
per-molecule querygen (Opus, parallel) -> collate+stats -> PMC ESearch (parallel,
NCBI 10/s shared) -> merge registry. STOPS at the registry (PMCID sets). The
full-text download -> chunk -> BM25 -> judge tail is launched separately.

Everything small+critical lives on /work RUN_DIR (survives login-node restarts);
this driver also mirrors data/+state/ to S3 periodically so a wiped node can resume.
Idempotent: re-running skips completed items (per-item result files / done markers).

Env: EXP70_DATASET, EXP70_RUN_DIR, EXP70_RL_DIR (shared NCBI bucket), EXP70_BULK_DIR,
     EXP70_CLAUDE_MODELS (Opus pool for querygen).
"""
import os, sys, time, json, subprocess, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import datasets as DS
import runner as R

STATE_S3 = os.environ.get("EXP70_STATE_S3",
                          "s3://molerag/%s_chunks/collect_state/" % DS.CFG["name"])
_last_sync = [0.0]
_sync_lock = threading.Lock()


def _sync_state(force=False):
    now = time.time()
    with _sync_lock:
        if not force and now - _last_sync[0] < 90:
            return
        _last_sync[0] = now
    # Mirror the resumable orchestration (data/ + state/ + results/, NOT bulk) to S3.
    for sub in ("data", "state"):
        src = C.P(sub)
        if os.path.isdir(src):
            subprocess.run(["aws", "s3", "sync", src, STATE_S3 + sub + "/",
                            "--only-show-errors", "--exclude", "*.tmp*"],
                           timeout=600, check=False)


def _results_sync():
    # results/ (per-item querygen + search PMCID lists) can be large; sync less often.
    src = C.P("results")
    if os.path.isdir(src):
        subprocess.run(["aws", "s3", "sync", src, STATE_S3 + "results/",
                        "--only-show-errors", "--exclude", "*.tmp*"],
                       timeout=1800, check=False)


def _progress(**kw):
    p = C.read_json(C.P("state", "COLLECT_PROGRESS.json"), {}) or {}
    p.update(kw); p["updated_at"] = C.now_iso(); p["dataset"] = DS.CFG["name"]
    C.write_json(C.P("state", "COLLECT_PROGRESS.json"), p)


def main():
    C.ensure_dirs()
    C.atomic_write_text(C.P("state", "collect.pid"), str(os.getpid()))
    C.log("INFO", "collect", "=== exp70-multi collect dataset=%s run_dir=%s ===" % (DS.CFG["name"], C.RUN_DIR))

    # Stage 0: ingest
    _progress(phase="ingest")
    if C.read_json(C.P("data", "ingest_summary.json")) is None:
        import ingest as IG
        IG.run()
    _sync_state(force=True)

    # Stage 1: identity (offline)
    _progress(phase="identity")
    import identity as ID
    if not ID.is_done():
        ID.resolve_all()
    st = C.read_json(C.P("state", "IDENTITY_STATS.json"), {})
    C.log("INFO", "collect", "IDENTITY: %s" % json.dumps(st))
    _sync_state(force=True)

    # Stage 2: per-task property querygen (Opus)
    _progress(phase="querygen_property")
    import querygen as QG
    QG.generate_property_queries()
    _sync_state(force=True)

    # Stage 3: per-molecule querygen (Opus 4.6, high concurrency — not throttled)
    _progress(phase="querygen_molecule")
    qg_threads = int(os.environ.get("EXP70_QG_THREADS", "64"))
    C.log("INFO", "collect", "molecule querygen drain (%d threads)" % qg_threads)
    R.drain_stage("querygen_molecule", n_threads=qg_threads,
                  tick=lambda: (_sync_state(), _progress(phase="querygen_molecule")))
    _sync_state(force=True)

    # Stage 4: collate queries + stats
    _progress(phase="collate")
    import search as S
    S.collate_queries()
    stats = S.querygen_stats()
    C.log("INFO", "collect", "QUERYGEN_STATS: %s" % json.dumps(stats))
    _sync_state(force=True)

    # Stage 5: PMC ESearch (parallel; NCBI 10/s bucket SHARED across all dataset jobs)
    _progress(phase="search")
    s_threads = int(os.environ.get("EXP70_SEARCH_THREADS", "8"))
    C.log("INFO", "collect", "search drain (%d threads, shared 10/s bucket)" % s_threads)
    R.drain_stage("search", n_threads=s_threads,
                  tick=lambda: (_sync_state(), _progress(phase="search")))
    _sync_state(force=True)

    # Stage 6: merge registry
    _progress(phase="merge")
    import merge as M
    summary = M.build_registry()
    C.write_json(C.P("data", "registry_summary.json"), summary)
    C.log("INFO", "collect", "REGISTRY: %s" % json.dumps(summary))
    _progress(phase="collect_done", registry=summary)
    _sync_state(force=True)
    _results_sync()
    C.atomic_write_text(C.P("state", "COLLECT_DONE"), C.now_iso())
    C.log("INFO", "collect", "=== collect COMPLETE dataset=%s ===" % DS.CFG["name"])


if __name__ == "__main__":
    main()
