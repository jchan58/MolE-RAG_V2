"""
exp70-multi FULL-TEXT + CHUNK stage (per dataset, resumable, bulk on /tmp + S3).

Runs AFTER run_collect (registry present). Steps:
  1. compute_clean_set  -> data/clean_pmcids.txt (property PMCIDs + small molecule searches)
  2. chunk_download     -> OA-gate + download OA XML + ESummary metadata (per-entity, resumable)
  3. chunk_run          -> parse JATS -> passages (clean-restricted), on BULK_ROOT=/tmp
  4. tar passages -> S3 (s3://molerag/<ds>_chunks/) so they survive /tmp wipes

Bulk (fulltext_xml / passages / extracted_text) lives under EXP70_BULK_ROOT (node-local
/tmp); only the clean-restricted passages are tarred to S3. Everything is idempotent.
"""
import os, sys, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import datasets as DS

CHUNK_S3 = os.environ.get("EXP70_CHUNK_S3", "s3://molerag/%s_chunks/" % DS.CFG["name"])


def _tar_to_s3():
    pdir = C.BP("passages")
    if not os.path.isdir(pdir):
        return
    tarp = os.path.join(C.BULK_ROOT, "passages_%s.tar.gz" % DS.CFG["name"])
    C.log("INFO", "fulltext", "tar passages -> %s" % tarp)
    subprocess.run(["tar", "czf", tarp, "-C", C.BULK_ROOT, "passages"], timeout=7200, check=False)
    subprocess.run(["aws", "s3", "cp", tarp, CHUNK_S3 + "passages/", "--only-show-errors"],
                   timeout=7200, check=False)
    # also sync extracted_text metadata index (small)
    subprocess.run(["aws", "s3", "sync", C.P("data"), CHUNK_S3 + "collect_state/data/",
                    "--only-show-errors", "--exclude", "*.tmp*"], timeout=1800, check=False)


def main():
    C.ensure_dirs()
    os.makedirs(C.BULK_ROOT, exist_ok=True)
    C.atomic_write_text(C.P("state", "fulltext.pid"), str(os.getpid()))
    C.log("INFO", "fulltext", "=== fulltext+chunk dataset=%s bulk_root=%s ===" % (DS.CFG["name"], C.BULK_ROOT))

    # 1. clean set (best-effort; if it fails, chunk_run falls back to chunking all downloaded)
    if not os.path.exists(C.P("data", "clean_pmcids.txt")):
        try:
            import compute_clean_set as CS
            CS.run()
        except Exception as e:
            C.log("WARN", "fulltext", "compute_clean_set failed (%r); proceeding without clean restriction" % e)

    # 2. download OA full text + metadata (resumable, MEGA_LIMIT checkpoint)
    import chunk_download as CD
    CD.run()

    # 3. parse to passages (clean-restricted), on /tmp
    import chunk_run as CR
    CR.run()

    # 4. offload passages to S3 + signal the judge there is nothing more to pull
    _tar_to_s3()
    try:
        open(os.path.join(C.BULK_ROOT.rsplit("/chunks", 1)[0], "PULL_DONE.txt"), "w").write(C.now_iso())
    except Exception:
        pass
    C.atomic_write_text(C.P("state", "FULLTEXT_DONE"), C.now_iso())
    C.log("INFO", "fulltext", "=== fulltext+chunk COMPLETE dataset=%s ===" % DS.CFG["name"])


if __name__ == "__main__":
    main()
