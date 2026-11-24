"""
Standalone fast merge: run the (now parallel-read) build_registry for the active dataset,
write registry_summary.json, and touch COLLECT_DONE. Used to finish lipo/sider whose collectors
were grinding the OLD serial merge. Idempotent (build_registry rebuilds from search results).
Env via env_common.sh <ds>; EXP70_MERGE_THREADS sets read concurrency.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import merge as M
import common as C

t0 = time.time()
summary = M.build_registry()
C.write_json(C.P("data", "registry_summary.json"), summary)
C.atomic_write_text(C.P("state", "COLLECT_DONE"), C.now_iso())
n = summary.get("total_unique_pmcids") if isinstance(summary, dict) else summary
print("MERGE_DONE dataset=%s elapsed=%.0fs unique_pmcids=%s COLLECT_DONE written"
      % (os.environ.get("EXP70_DATASET"), time.time() - t0, n))
