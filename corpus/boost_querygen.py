"""
Additive querygen_molecule ACCELERATOR — a second cooperative worker that drains the
per-molecule Opus query-gen stage using the SAME on-disk locks as the running run_collect,
so it never double-processes and needs NO kill/restart of the live collector. It touches
ONLY querygen_molecule (never search/merge -> no NCBI-bucket or merge contention). Kill this
process alone to stop boosting. Resumable/idempotent (per-molecule done markers + locks).
Env via env_common.sh <ds>; EXP70_QG_THREADS sets this worker's concurrency.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import datasets as DS
import runner as R

N = int(os.environ.get("EXP70_QG_THREADS", "96"))
C.log("INFO", "boost", "BOOST querygen_molecule dataset=%s threads=%d (cooperative via on-disk locks; querygen only)"
      % (DS.CFG["name"], N))
R.drain_stage("querygen_molecule", n_threads=N)
C.log("INFO", "boost", "BOOST exiting (querygen_molecule fully drained / no claimable work left)")
