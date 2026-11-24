"""
Per-dataset S3 PROGRESS.json writer (survives node death). Reads the cheap state
files each stage writes (no walking of huge dirs beyond guarded top-level counts) and
pushes s3://molerag/<ds>_chunks/PROGRESS.json. Call periodically from the sbatch.
"""
import os, sys, json, time, socket, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import datasets as DS

S3 = os.environ.get("EXP70_PROGRESS_S3", "s3://molerag/%s_chunks/PROGRESS.json" % DS.CFG["name"])


def _count(path, cap=200000):
    try:
        n = 0
        with os.scandir(path) as it:
            for _ in it:
                n += 1
                if n >= cap:
                    break
        return n
    except FileNotFoundError:
        return 0


def gather():
    g = C.read_json(C.P("state", "IDENTITY_STATS.json"), {}) or {}
    qg = C.read_json(C.P("state", "QUERYGEN_STATS.json"), {}) or {}
    reg = C.read_json(C.P("data", "registry_summary.json"), {}) or {}
    cc = C.read_json(C.P("state", "CHUNK_COUNTS.json"), {}) or {}
    cp = C.read_json(C.P("state", "COLLECT_PROGRESS.json"), {}) or {}
    # judge progress (bedrock qwen stream) lives on the node-local filter base
    fb = os.environ.get("EXP70_FILTER_BASE", "/tmp/exp70_%s" % DS.CFG["name"])
    jp = {}
    try:
        jp = json.load(open(os.path.join(fb, "PROGRESS_qwen.json")))
    except Exception:
        pass
    bm = {}
    try:
        bm = json.load(open(os.path.join(fb, "bm25", "prescreen_summary.json")))
    except Exception:
        pass
    out = dict(
        dataset=DS.CFG["name"], updated_at=C.now_iso(), node=socket.gethostname(),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        stage=cp.get("phase"),
        collect_done=os.path.exists(C.P("state", "COLLECT_DONE")),
        fulltext_done=os.path.exists(C.P("state", "FULLTEXT_DONE")),
        identity=dict(n_structures=g.get("n_structures"), n_resolved=g.get("n_resolved"),
                      n_with_cid=g.get("n_with_cid"), n_with_synonyms=g.get("n_with_synonyms"),
                      median_synonyms=g.get("median_synonyms_per_molecule"),
                      n_name_less=g.get("n_name_less_smiles_only"),
                      n_with_parent_cid=g.get("n_with_parent_cid")),
        queries=dict(property_queries=qg.get("property_queries"),
                     molecule_associations=qg.get("molecule_query_associations"),
                     distinct_searches=qg.get("distinct_searches_after_dedup")),
        property_tasks_done=_count(C.P("results", "querygen_property")),
        molecule_querygen_done=_count(C.P("results", "querygen_molecule")),
        searches_started=_count(C.P("results", "search")),
        registry=dict(total_unique_pmcids=reg.get("total_unique_pmcids"),
                      found_by_property=reg.get("found_by_property"),
                      found_by_molecule=reg.get("found_by_molecule")),
        articles_chunked=cc.get("articles_parsed"), total_passages=cc.get("total_passages"),
        bm25_candidates=(bm.get("candidates") or {}).get("union_passages"),
        judge=dict(articles_done=jp.get("articles_done"), passages_judged=jp.get("passages_judged"),
                   keep=jp.get("keep"), review=jp.get("review"), exclude=jp.get("exclude"),
                   calls_per_s=jp.get("calls_per_s")),
    )
    return out


def push():
    out = gather()
    local = C.P("state", "PROGRESS.json")
    C.write_json(local, out)
    subprocess.run(["aws", "s3", "cp", local, S3, "--only-show-errors"], timeout=120, check=False)
    return out


if __name__ == "__main__":
    print(json.dumps(push(), indent=1))
