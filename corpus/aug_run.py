"""
Augmentation orchestrator — resumes the pipeline at MERGE (search is already
complete) and runs: merge -> synonym augmentation searches -> re-merge ->
oa_gate -> download -> metadata -> DONE. Avoids re-traversing the cached
identity/querygen/search stages (slow Lustre rescans). Reuses the driver's
stage helpers and the idempotent, tmpfs-rate-limited machinery; workers in auto
mode follow the stages it sets. Detached/resumable.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import driver as DRV
import merge as M
import augment as AUG
import runner as R
import oa_gate as OA


def run():
    C.ensure_dirs()
    C.atomic_write_text(C.P("state", "driver.pid"), str(os.getpid()))
    C.log("INFO", "aug_run", "=== augmentation orchestrator starting (pid %d) ===" % os.getpid())

    # Main merge over the completed search (one-time, ~8 min over 8.1M). On a
    # restart it is cached: reuse the saved pre-aug summary + existing registry.
    DRV.set_stage("merge")
    pre = C.read_json(C.P("data", "registry_summary_pre_aug.json"))
    if pre and os.path.exists(C.P("data", "registry_pmcids.txt")):
        C.log("INFO", "aug_run", "pre-aug merge cached (skip): %s unique PMCIDs"
              % pre.get("total_unique_pmcids"))
    else:
        pre = M.build_registry()
        C.write_json(C.P("data", "registry_summary_pre_aug.json"), pre)
        C.log("INFO", "aug_run", "pre-aug registry: %s unique PMCIDs" % pre.get("total_unique_pmcids"))
    DRV.periodic_report(force=True)

    # Synonym augmentation searches (chained before the heavy download/metadata).
    DRV.set_stage("augment")
    if AUG.ready():
        if not os.path.exists(C.P("data", "search_items_aug.jsonl")):
            AUG.build_aug_queries()
        AUG.write_full_synonym_identities()
        DRV.set_stage("search_aug")
        C.log("INFO", "aug_run", "stage search_aug: full-synonym augmentation searches")
        R.drain_stage("search_aug", tick=DRV.periodic_report)
        DRV.mark_stage_complete("search_aug")
        post = M.build_registry()  # re-merge: union aug PMCIDs + provenance
        aug_q = C.read_json(C.P("state", "AUG_QUERY_STATS.json"), {})
        results = dict(
            computed_at=C.now_iso(),
            augmentation_searches=aug_q.get("n_searches"),
            missing_terms=aug_q.get("total_missing_terms"),
            molecules_augmented=aug_q.get("molecules_with_missing"),
            total_unique_pmcids_before=pre.get("total_unique_pmcids"),
            total_unique_pmcids_after=post.get("total_unique_pmcids"),
            new_unique_pmcids=(post.get("total_unique_pmcids", 0) - pre.get("total_unique_pmcids", 0)),
            pmcids_found_by_synonym_augmentation=post.get("found_by_synonym_augmentation"),
            pmcids_found_ONLY_by_synonym_augmentation=post.get("found_only_by_synonym_augmentation"),
            molecules_that_gained_pmcids=post.get("n_molecules_gained_from_aug"))
        C.write_json(C.P("state", "AUG_RESULTS.json"), results)
        C.log("INFO", "aug_run", "AUGMENTATION done: %s" % json.dumps(results))
        DRV.periodic_report(force=True)
    else:
        C.log("WARN", "aug_run", "augment not ready; skipping augmentation")

    # Full-text + metadata: CHUNKED per-entity (property first, then molecules
    # ascending by n_pmcids), NOT a single 8.1M bulk pass. Mega molecules (>50k
    # PMCIDs) hit a hard checkpoint and await a go-ahead. (User-directed pacing.)
    DRV.set_stage("chunk_download")
    import chunk_download as CD
    CD.run()
    C.log("INFO", "aug_run", "=== augmentation + chunked download phase returned ===")


if __name__ == "__main__":
    run()
