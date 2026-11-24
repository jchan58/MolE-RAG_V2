"""
exp70 driver — end-to-end, resumable orchestration.

Runs every stage in order and can complete the whole pipeline by itself; external
`worker.py` processes accelerate the rate-bound parallel stages by claiming items
from the same RUN_DIR. Idempotent: re-running skips completed model calls, searches,
and downloads. Designed to run detached (setsid+nohup).
"""
import os, sys, json, time, datetime, signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import runner as R

PROPERTY_INSTRUCTION = None  # filled from querygen to avoid duplication


def set_progress(**kw):
    prog = C.read_json(C.P("state", "progress.json"), {}) or {}
    prog.update(kw)
    prog["updated_at"] = C.now_iso()
    C.write_json(C.P("state", "progress.json"), prog)


def set_stage(stage):
    C.atomic_write_text(C.P("state", "current_stage"), stage)
    set_progress(phase=stage)


def mark_stage_complete(stage):
    C.atomic_write_text(C.P("state", "stage_%s.complete" % stage), C.now_iso())


def write_run_config():
    import querygen as QG
    rid = "exp70_%s" % datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    existing = C.read_json(C.P("run_config.json"))
    if existing and existing.get("run_id"):
        rid = existing["run_id"]  # preserve across resume
    cfg = dict(
        run_id=rid,
        created_at=(existing.get("created_at") if existing else C.now_iso()),
        updated_at=C.now_iso(),
        run_dir=C.RUN_DIR,
        model=C.OPENAI_MODEL,
        model_params=dict(max_completion_tokens=C.OPENAI_MAX_COMPLETION_TOKENS,
                          temperature="OMITTED (reasoning model)",
                          endpoint=C.ENDPOINTS["openai"], api="chat-completions",
                          max_completion_tokens_ladder=QG.MCT_LADDER),
        prompt_version=C.PROMPT_VERSION,
        instructions=dict(property=QG.PROPERTY_INSTRUCTION,
                          molecule=QG.MOLECULE_INSTRUCTION,
                          output_format_system=QG.SYSTEM_FORMAT),
        input=dict(
            reference_seed="seed_0",
            source_csvs=["data/moleculenet/seed_0/bbbp/bbbp_%s.csv" % s for s in ("train", "valid", "test")],
            id_column="(none; SMILES is the only identifier)",
            smiles_column="smiles", label_column="label (preserved; EXCLUDED from query inputs)",
            split_source="seed_0 train/valid/test membership",
            seed_ambiguity_resolution=(
                "7 scaffold-split seeds exist (0,1,2,3,4,900,901). Seeds 0-4 are "
                "different split assignments of the SAME molecule set; seed_900/901 "
                "swap ~205 different molecules (verified by RDKit canonicalization). "
                "The full BBBP set = union of any one seed's train+valid+test. We use "
                "seed_0 as the single reference (its union is the full dataset) and "
                "DELIBERATELY EXCLUDE the seed_900/901 variant. All 2039 seed_0 rows "
                "are processed; identity resolution + query generation run on the "
                "unique canonical structures, with mappings back to all original rows "
                "and splits retained. No raw MoleculeNet BBBP.csv (num/name/p_np) "
                "exists under MolE-RAG; data/bbbp is a symlink to seed_0/bbbp."),
            normalization=dict(engine="RDKit 2025.09.2",
                               canonicalization="MolToSmiles isomericSmiles=True (stereo preserved)",
                               dedup_key="canonical isomeric SMILES",
                               salts_stereoisomers="NOT merged",
                               invalid_handling="recorded, not dropped"),
        ),
        identity_resolution=dict(
            mode="OFFLINE (live PubChem PUG-REST 429-blocks systematic access)",
            source1="local MolE-RAG caches (synonyms_cache.json, llm_filtered_synonyms.json) "
                    "matched by exact canonical / desalted / InChIKey-skeleton",
            source2="PubChem bulk FTP flat files (CID-InChI-Key, CID-Title, CID-IUPAC, "
                    "CID-Synonym-filtered) streamed offline; InChIKey->CID then CID->names",
            bulk_dir=C.P("bulk_ftp"),
            salts_stereoisomers="target identity keeps original structure; desalt/skeleton "
                                "matches only borrow names, mapping recorded",
            unresolved="kept with a deterministic InChIKey structure-only query"),
        download_fulltext=dict(
            route="PMC Open Data S3 Cloud Service (s3://pmc-oa-opendata, anonymous UNSIGNED)",
            off_eutils_cap=True, fallback="EFetch (sanctioned)",
            s3_per_second_softcap=C.RATE_S3),
        endpoints=C.ENDPOINTS,
        timeouts=dict(http_seconds=C.HTTP_TIMEOUT),
        rate_limits=dict(
            ncbi_per_second_with_key=C.RATE_NCBI_FULL,
            ncbi_per_second_fallback=C.RATE_NCBI_FALLBACK,
            ncbi_key_source="~/.config/molrag/ncbi.env (read at call time, never logged)",
            pubchem_per_second=C.RATE_PUBCHEM,
            openai_per_second_softcap=C.RATE_OPENAI,
            enforcement="central on-disk fcntl token buckets, global across all workers"),
        retry_policy=dict(max_retries=C.MAX_RETRIES, backoff_base=C.BACKOFF_BASE,
                          backoff_max=C.BACKOFF_MAX, respect_retry_after=True,
                          lease_seconds=C.LEASE_SECONDS),
        search_params=dict(partition_threshold=C.PARTITION_THRESHOLD,
                           esearch_page=C.SEARCH_PAGE, subdivision="binary [PDAT] date ranges",
                           earliest_date="1500/01/01", history_rescue=True),
        batch_sizes=dict(esummary=C.SUMMARY_BATCH, efetch=20),
        secrets=dict(openai="~/.config/molrag/openai.env (OPENAI_API_KEY)",
                     policy="read at call time; NEVER logged/saved/echoed"),
        output_paths=dict(data=C.DIRS["data"], results=C.DIRS["results"],
                          raw_pubchem=C.DIRS["raw_pubchem"], raw_model=C.DIRS["raw_model"],
                          raw_search=C.DIRS["raw_search"], raw_metadata=C.DIRS["raw_metadata"],
                          fulltext=C.DIRS["fulltext"], logs=C.DIRS["logs"],
                          report=C.P("collection_report.md")),
    )
    C.write_json(C.P("run_config.json"), cfg)
    return cfg


import threading as _threading
_last_report = [0.0]
_report_lock = _threading.Lock()

def _do_report():
    # Only one report at a time; skip if one is already running.
    if not _report_lock.acquire(blocking=False):
        return
    try:
        import report as RP
        RP.write_report()
    except Exception as e:
        C.log("WARN", "report", "periodic report failed: %r" % e)
    finally:
        _report_lock.release()

def _cheap_progress():
    """A fast progress snapshot (no file reads / no huge globs) — the full
    collation report reads 200k+ files on Lustre and must NOT run mid-stage."""
    try:
        import os as _os
        prog = dict(phase=C.read_json(C.P("state", "progress.json"), {}).get("phase"),
                    searches_with_topcall=len(_os.listdir(C.P("raw_search_responses"))),
                    searches_started=len(_os.listdir(C.P("results", "search"))),
                    updated_at=C.now_iso())
        C.write_json(C.P("state", "progress_snapshot.json"), prog)
    except Exception:
        pass

def periodic_report(force=False, block=False):
    """The full collation report globs/reads 200k+ files on Lustre (minutes) and
    thrashes IO+GIL, starving the search consumers — so it runs ONLY at the end
    (block=True). Mid-run calls just drop a cheap progress snapshot."""
    if block:
        _do_report()
        _last_report[0] = time.time()
        return
    _cheap_progress()


def run():
    C.ensure_dirs()
    C.atomic_write_text(C.P("state", "driver.pid"), str(os.getpid()))
    write_run_config()
    C.log("INFO", "driver", "=== exp70 driver starting (pid %d) ===" % os.getpid())

    # Stage: ingest (driver-only)
    set_stage("ingest")
    if C.read_json(C.P("data", "ingest_summary.json")) is None:
        import ingest as IG
        IG.run()
    write_run_config()  # refresh with input hashes available
    periodic_report(force=True)

    # Stage: identity resolution (OFFLINE: local caches + PubChem bulk FTP).
    # Driver-only (local file streaming + one-time bulk download, not per-item HTTP).
    set_stage("identity")
    C.log("INFO", "driver", "stage identity: OFFLINE resolution (caches + bulk FTP)")
    import identity as ID
    if not ID.is_done():
        ID.resolve_all()
    mark_stage_complete("identity")
    periodic_report(force=True)

    # Stage: property queries (driver-only, once)
    set_stage("querygen_property")
    import querygen as QG
    QG.generate_property_queries()
    mark_stage_complete("querygen_property")

    # Stage: molecule queries (parallel)
    set_stage("querygen_molecule")
    C.log("INFO", "driver", "stage querygen_molecule")
    R.drain_stage("querygen_molecule", tick=periodic_report)
    mark_stage_complete("querygen_molecule")
    periodic_report(force=True)

    # Stage: collate queries + validate (driver-only) + emit query-count stats
    set_stage("collate_queries")
    import search as S
    S.collate_queries()
    stats = S.querygen_stats()
    C.log("INFO", "driver", "QUERYGEN COMPLETE -> state/QUERYGEN_STATS.json: %s"
          % json.dumps(stats))
    periodic_report(force=True)

    # Stage: search (parallel)
    set_stage("search")
    C.log("INFO", "driver", "stage search: PMC ESearch + subdivision")
    R.drain_stage("search", tick=periodic_report)
    mark_stage_complete("search")
    periodic_report(force=True)

    # Stage: merge registry (driver-only)
    set_stage("merge")
    import merge as M
    pre = M.build_registry()
    C.write_json(C.P("data", "registry_summary_pre_aug.json"), pre)
    periodic_report(force=True)

    # Stage: synonym augmentation (ADDITIVE full-PubChem-synonym search) — chained
    # right after search+merge, BEFORE the heavy download/metadata, so those run
    # once over the combined PMCID set. Narrow name queries -> near the 10/s cap.
    set_stage("augment")
    import augment as AUG
    if AUG.ready():
        if not os.path.exists(C.P("data", "search_items_aug.jsonl")):
            AUG.build_aug_queries()
        AUG.write_full_synonym_identities()
        set_stage("search_aug")
        C.log("INFO", "driver", "stage search_aug: full-synonym augmentation searches")
        R.drain_stage("search_aug", tick=periodic_report)
        mark_stage_complete("search_aug")
        post = M.build_registry()   # re-merge to union augmentation PMCIDs + provenance
        aug_q = C.read_json(C.P("state", "AUG_QUERY_STATS.json"), {})
        C.write_json(C.P("state", "AUG_RESULTS.json"), dict(
            computed_at=C.now_iso(),
            augmentation_searches=aug_q.get("n_searches"),
            missing_terms=aug_q.get("total_missing_terms"),
            molecules_augmented=aug_q.get("molecules_with_missing"),
            total_unique_pmcids_before=pre.get("total_unique_pmcids"),
            total_unique_pmcids_after=post.get("total_unique_pmcids"),
            new_unique_pmcids=(post.get("total_unique_pmcids", 0) - pre.get("total_unique_pmcids", 0)),
            pmcids_found_by_synonym_augmentation=post.get("found_by_synonym_augmentation"),
            pmcids_found_ONLY_by_synonym_augmentation=post.get("found_only_by_synonym_augmentation"),
            molecules_that_gained_pmcids=post.get("n_molecules_gained_from_aug")))
        C.log("INFO", "driver", "AUGMENTATION done: %s"
              % json.dumps(C.read_json(C.P("state", "AUG_RESULTS.json"), {})))
        periodic_report(force=True)
    else:
        C.log("WARN", "driver", "augment not ready (cid_synonyms_full missing); skipping")

    # Stage: OA-gate via S3 Cloud Service (parallel, off the E-utilities cap)
    set_stage("oa_gate")
    C.log("INFO", "driver", "stage oa_gate: S3 OA membership")
    R.drain_stage("oa_gate", tick=periodic_report)
    import oa_gate as OA
    OA.collate()
    mark_stage_complete("oa_gate")
    periodic_report(force=True)

    # Stage: download OA full text via S3 Cloud Service (parallel, off-cap).
    # Runs before metadata so ESummary metadata can enrich from the downloaded XML
    # (abstract/keywords/license/relationships).
    set_stage("download")
    C.log("INFO", "driver", "stage download: S3 OA full-text XML")
    R.drain_stage("download", tick=periodic_report)
    mark_stage_complete("download")
    periodic_report(force=True)

    # Stage: metadata (parallel, E-utilities ESummary + downloaded-XML enrichment)
    set_stage("metadata")
    C.log("INFO", "driver", "stage metadata: ESummary")
    R.drain_stage("metadata", tick=periodic_report)
    mark_stage_complete("metadata")

    # Final report (synchronous — must finish before we mark DONE)
    set_stage("report")
    periodic_report(block=True)
    C.atomic_write_text(C.P("state", "DONE"), C.now_iso())
    set_progress(phase="done")
    C.log("INFO", "driver", "=== exp70 driver COMPLETE ===")


if __name__ == "__main__":
    run()
