#!/usr/bin/env bash
# Quick per-dataset status snapshot for exp70-multi collection.
set -u
B=${EXP70_BASE:-$(cd "$(dirname "$0")/.." && pwd)/corpus_runs}
PY=${PY:-python3}
for DS in sider lipo tox21 hiv; do
  R="$B/$DS"
  [ -d "$R" ] || { echo "$DS: not started"; continue; }
  "$PY" - "$DS" "$R" <<'PY'
import json, os, sys
ds, R = sys.argv[1], sys.argv[2]
def n(p):
    try: return len(os.listdir(os.path.join(R, p)))
    except OSError: return 0
prog = {}
try: prog = json.load(open(os.path.join(R,"state","COLLECT_PROGRESS.json")))
except Exception: pass
idn = {}
try: idn = json.load(open(os.path.join(R,"state","IDENTITY_STATS.json")))
except Exception: pass
def done(m): return os.path.exists(os.path.join(R,"state",m))
reg = {}
try: reg = json.load(open(os.path.join(R,"data","registry_summary.json")))
except Exception: pass
print("%-6s phase=%-18s ident=%s/%s(syn%s,par%s) propQ=%s molQ=%s search=%s | COLLECT_DONE=%s reg_pmcids=%s" % (
  ds, prog.get("phase","-"),
  idn.get("n_resolved","-"), idn.get("n_structures","-"),
  idn.get("n_with_synonyms","-"), idn.get("n_with_parent_cid","-"),
  n("results/querygen_property"), n("results/querygen_molecule"), n("results/search"),
  done("COLLECT_DONE"), reg.get("total_unique_pmcids","-")))
PY
done
