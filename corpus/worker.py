"""
exp70 worker — an additional process that helps drain a rate-bound parallel stage.

Usage:
    python3 worker.py <stage>      # pubchem | querygen_molecule | search | metadata | download
    python3 worker.py auto         # follow state/current_stage, help whichever is active

Launch as many as you like; they share RUN_DIR with the driver. The central on-disk
token buckets keep total NCBI/PubChem/OpenAI rates within the global caps regardless
of how many workers run, so extra workers overlap CPU/IO (XML parsing, file writes,
PubChem vs NCBI concurrency) but never exceed the service limits.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import runner as R


def register():
    d = C.P("state", "workers")
    os.makedirs(d, exist_ok=True)
    C.atomic_write_text(os.path.join(d, "%d.worker" % os.getpid()),
                        "%s pid=%d\n" % (C.now_iso(), os.getpid()))


def run_one(stage):
    C.log("INFO", "worker", "pid %d draining stage %s" % (os.getpid(), stage))
    R.drain_stage(stage, exit_when_empty=False)


def run_auto():
    C.log("INFO", "worker", "pid %d auto-following current_stage" % os.getpid())
    while True:
        if os.path.exists(C.P("state", "DONE")):
            C.log("INFO", "worker", "DONE marker present; exiting")
            return
        stage = None
        try:
            with open(C.P("state", "current_stage")) as f:
                stage = f.read().strip()
        except FileNotFoundError:
            pass
        if stage in R.PARALLEL_STAGES and not os.path.exists(
                C.P("state", "stage_%s.complete" % stage)):
            R.drain_stage(stage, exit_when_empty=False, max_idle_polls=5)
        else:
            time.sleep(5)


if __name__ == "__main__":
    C.ensure_dirs()
    register()
    arg = sys.argv[1] if len(sys.argv) > 1 else "auto"
    if arg == "auto":
        run_auto()
    elif arg in R.PARALLEL_STAGES:
        run_one(arg)
    else:
        print("unknown stage: %s (choices: auto, %s)" % (arg, ", ".join(R.PARALLEL_STAGES)))
        sys.exit(2)
