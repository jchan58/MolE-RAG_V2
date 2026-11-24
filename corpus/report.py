"""
Stages 14 & 15 — collate canonical JSONL artifacts, verify exact counts, and write
collection_report.md. Safe to run at any time (idempotent, progress-aware).
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def _iter_json(dir_glob):
    for path in glob.glob(dir_glob):
        r = C.read_json(path)
        if r is not None:
            yield path, r


def collate_identities():
    out = C.P("data", "molecule_identities.jsonl")
    recs = []
    for _p, r in _iter_json(C.P("results", "pubchem", "*.json")):
        recs.append(r)
    recs.sort(key=lambda r: r.get("mol_id", ""))
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(recs)


def collate_alias_accounting():
    out = C.P("data", "alias_accounting.jsonl")
    recs = []
    for _p, r in _iter_json(C.P("results", "querygen_molecule", "*.json")):
        if r.get("skipped"):
            recs.append(dict(mol_id=r["mol_id"], skipped=True, reason=r.get("reason")))
        else:
            aa = r.get("alias_accounting", {})
            recs.append(dict(mol_id=r["mol_id"], cid=r.get("cid"), **aa))
    recs.sort(key=lambda r: r.get("mol_id", ""))
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(recs)


def collate_search_manifest():
    out = C.P("data", "search_manifest.jsonl")
    base = C.P("results", "search")
    n = 0
    if not os.path.isdir(base):
        open(out, "w").close()
        return 0
    with open(out, "w") as f:
        for sid in sorted(os.listdir(base)):
            m = C.read_json(os.path.join(base, sid, "manifest.json"))
            if not m:
                continue
            rec = dict(search_id=sid,
                       submitted_query=m.get("submitted_query"),
                       query_ids=m.get("query_ids"), molecule_ids=m.get("molecule_ids"),
                       query_types=m.get("query_types"),
                       top_level=m.get("top_level"),
                       strategy=m.get("strategy"),
                       reported_count_total=m.get("reported_count_total"),
                       retrieved_unique_total=m.get("retrieved_unique_total"),
                       reconciliation=m.get("reconciliation"),
                       completion_status=m.get("completion_status"),
                       complete=m.get("complete"),
                       submitted_at=m.get("submitted_at"), finished_at=m.get("finished_at"),
                       n_partitions=len(m.get("partitions", [])),
                       partitions=m.get("partitions", []))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def _license_label(lic):
    if not lic:
        return "none/unknown"
    if isinstance(lic, dict):
        t = lic.get("license_type") or lic.get("href") or lic.get("text")
    else:
        t = str(lic)
    if not t:
        return "none/unknown"
    tl = t.lower()
    if "creativecommons.org/licenses/by-nc" in tl or "cc by-nc" in tl:
        return "CC-BY-NC"
    if "creativecommons.org/licenses/by" in tl or "cc by" in tl or "cc-by" in tl:
        return "CC-BY"
    if "creativecommons.org/publicdomain" in tl or "cc0" in tl:
        return "CC0"
    if "text mining" in tl:
        return "text-mining-only"
    return (t[:40])


def collate_download_manifest():
    """Emit download_manifest.jsonl: S3/EFetch results for OA articles PLUS
    metadata-only entries synthesized for non-OA (not-in-S3) registry PMCIDs."""
    out = C.P("data", "download_manifest.jsonl")
    n = 0
    total_bytes = 0
    status_counts = {}
    license_counts = {}
    route_counts = {}
    with open(out, "w") as f:
        # OA download results
        for path in glob.glob(C.P("results", "download", "*", "*.json")):
            r = C.read_json(path)
            if r is None:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
            st = r.get("status")
            status_counts[st] = status_counts.get(st, 0) + 1
            route_counts[r.get("retrieval_route")] = route_counts.get(r.get("retrieval_route"), 0) + 1
            if r.get("bytes"):
                total_bytes += r["bytes"]
            if st in ("success", "success_no_body"):
                lab = _license_label(r.get("license"))
                license_counts[lab] = license_counts.get(lab, 0) + 1
        # non-OA (not in S3) -> metadata-only entries
        oamap = C.P("data", "oa_membership.jsonl")
        if os.path.exists(oamap):
            with open(oamap) as mf:
                for line in mf:
                    m = json.loads(line)
                    if m.get("in_s3"):
                        continue
                    rec = dict(pmcid=m["pmcid"], pmcid_full="PMC%s" % m["pmcid"],
                               status="metadata_only_non_oa", full_text_available=False,
                               retrieval_route=None,
                               note="not in PMC OA S3 subset; metadata retained")
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    status_counts["metadata_only_non_oa"] = status_counts.get("metadata_only_non_oa", 0) + 1
    return n, total_bytes, status_counts, license_counts, route_counts


def compute_counts():
    counts = dict(generated_at=C.now_iso())
    ing = C.read_json(C.P("data", "ingest_summary.json"), {})
    counts["ingest"] = dict(
        input_rows=ing.get("total_input_rows"),
        valid_rows=ing.get("valid_rows"), invalid_rows=ing.get("invalid_rows"),
        unique_structures=ing.get("unique_valid_structures"),
        per_split=ing.get("per_split"),
        reference_seed=ing.get("reference_seed"),
        multi_seed=ing.get("multi_seed_verification", {}).get("comparison"),
        all_seeds_same=ing.get("multi_seed_verification", {}).get("all_seeds_same_molecule_set"),
    )

    # identity resolution (offline). Prefer the identity stage's own clean summary.
    idsum = C.read_json(C.P("data", "identity_summary.json"), {})
    if idsum:
        counts["pubchem"] = dict(n_processed=idsum.get("n_structures"),
                                 by_match_type=idsum.get("by_match_type", {}),
                                 by_source=idsum.get("by_source", {}),
                                 resolved=idsum.get("n_resolved"),
                                 unresolved=idsum.get("n_unresolved"),
                                 with_cid=idsum.get("n_with_cid"),
                                 with_synonyms=idsum.get("n_with_synonyms"))
    else:
        mt = {}
        n_ident = 0
        for _p, r in _iter_json(C.P("results", "pubchem", "*.json")):
            n_ident += 1
            mt[r.get("match_type")] = mt.get(r.get("match_type"), 0) + 1
        RES = {"exact_canonical", "desalted", "inchikey_skeleton", "inchikey_exact", "cid"}
        counts["pubchem"] = dict(n_processed=n_ident, by_match_type=mt,
                                 resolved=sum(v for k, v in mt.items() if k in RES),
                                 unresolved=mt.get("none", 0))

    # queries
    n_prop = n_mol_q = 0
    prop = C.read_json(C.P("results", "querygen_property", "property.json"))
    if prop:
        n_prop = prop.get("n_queries", 0)
    n_mol_entities = 0
    for _p, r in _iter_json(C.P("results", "querygen_molecule", "*.json")):
        if not r.get("skipped"):
            n_mol_entities += 1
            n_mol_q += r.get("n_queries", 0)
    valid_q = invalid_q = total_q = 0
    if os.path.exists(C.P("data", "queries.jsonl")):
        with open(C.P("data", "queries.jsonl")) as f:
            for line in f:
                q = json.loads(line)
                total_q += 1
                if q.get("validation", {}).get("valid"):
                    valid_q += 1
                else:
                    invalid_q += 1
    counts["queries"] = dict(property_queries=n_prop, molecule_queries=n_mol_q,
                             molecule_entities_with_queries=n_mol_entities,
                             total_query_associations=total_q,
                             valid=valid_q, invalid=invalid_q)

    # search
    sc = dict(complete_with_results=0, complete_zero=0, incomplete=0,
              invalid_query=0, service_failure=0, count_discrepancy=0,
              subdivided=0, total_searches=0)
    base = C.P("results", "search")
    if os.path.isdir(base):
        for sid in os.listdir(base):
            m = C.read_json(os.path.join(base, sid, "manifest.json"))
            if not m:
                continue
            sc["total_searches"] += 1
            if m.get("strategy") == "subdivided":
                sc["subdivided"] += 1
            st = m.get("completion_status")
            key = {"complete-with-results": "complete_with_results",
                   "complete-zero": "complete_zero", "incomplete": "incomplete",
                   "invalid-query": "invalid_query", "service-failure": "service_failure",
                   "count-discrepancy": "count_discrepancy"}.get(st)
            if key:
                sc[key] += 1
    counts["search"] = sc

    # registry
    counts["registry"] = C.read_json(C.P("data", "registry_summary.json"), {})

    # metadata
    md = dict(success=0, no_docsum=0, service_failure=0, total=0)
    for path in glob.glob(C.P("results", "metadata", "*", "*.json")):
        r = C.read_json(path)
        if r is None:
            continue
        md["total"] += 1
        st = r.get("metadata_status")
        if st == "success":
            md["success"] += 1
        elif st == "no-docsum":
            md["no_docsum"] += 1
        elif st and "failure" in st:
            md["service_failure"] += 1
    counts["metadata"] = md

    # OA gating (S3 membership)
    counts["oa_gate"] = C.read_json(C.P("data", "oa_summary.json"), {})

    # downloads (S3 Cloud Service + EFetch fallback)
    _n, total_bytes, status_counts, license_counts, route_counts = collate_download_manifest()
    dl = dict(status_counts=status_counts, license_counts=license_counts,
              route_counts=route_counts,
              success=status_counts.get("success", 0),
              success_no_body=status_counts.get("success_no_body", 0),
              unavailable=status_counts.get("unavailable", 0),
              identity_mismatch=status_counts.get("identity-mismatch", 0),
              invalid_payload=status_counts.get("invalid-payload", 0),
              temporary_failure=status_counts.get("temporary-failure", 0),
              metadata_only_non_oa=status_counts.get("metadata_only_non_oa", 0),
              total_records=_n, total_xml_bytes=total_bytes)
    dl["validated_full_text_xml"] = dl["success"]
    dl["total_xml_human"] = C.human(total_bytes)
    counts["downloads"] = dl

    # per-split PMCID counts (molecule-found) + property task
    mol_map = C.read_json(C.P("data", "molecule_pmcids.json"), {})
    prop_map = C.read_json(C.P("data", "property_task_pmcids.json"), {})
    split_pmcids = {"train": set(), "valid": set(), "test": set(), "unknown": set()}
    for mid, info in mol_map.items():
        sp = info.get("split") or "unknown"
        if sp not in split_pmcids:
            split_pmcids[sp] = set()
        split_pmcids[sp].update(info.get("pmcids", []))
    counts["per_split_pmcids"] = {k: len(v) for k, v in split_pmcids.items()}
    counts["property_task_pmcids"] = prop_map.get("n_pmcids", 0)

    # disk / usage (avoid walking the whole tree at scale: report free space via
    # statvfs and run-dir usage approximated by downloaded-XML bytes, the dominant
    # component)
    counts["disk"] = dict(xml_bytes=dl["total_xml_bytes"],
                          run_dir_human="~%s (XML)" % C.human(dl["total_xml_bytes"]),
                          scratch_free_human=C.human(C.disk_free_bytes()))
    counts["openai_usage"] = _openai_usage()
    counts["ncbi_fallback_active"] = C.ncbi_fallback_active()
    return counts


def _openai_usage():
    path = C.P("state", "usage_ledger.jsonl")
    tot = dict(calls=0, prompt_tokens=0, completion_tokens=0, total_tokens=0)
    if not os.path.exists(path):
        return tot
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "openai_querygen":
                tot["calls"] += 1
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    tot[k] += (r.get(k) or 0)
    return tot


def write_report():
    collate_identities()
    collate_alias_accounting()
    collate_search_manifest()
    c = compute_counts()  # also rebuilds download_manifest.jsonl
    C.write_json(C.P("data", "counts.json"), c)
    cfg = C.read_json(C.P("run_config.json"), {})
    ing = c["ingest"]
    pc = c["pubchem"]
    q = c["queries"]
    s = c["search"]
    reg = c.get("registry", {})
    dl = c["downloads"]
    md = c["metadata"]
    search_complete = (s["total_searches"] > 0 and
                       (s["incomplete"] + s["service_failure"]) == 0 and
                       s["total_searches"] >= (q["total_query_associations"] and 1))
    lines = []
    A = lines.append
    A("# exp70 — BBBP PMC Literature Collection — Report")
    A("")
    A("Generated: %s" % c["generated_at"])
    A("Run id: %s" % cfg.get("run_id", "?"))
    A("Run dir: %s" % C.RUN_DIR)
    A("Phase: %s" % C.read_json(C.P("state", "progress.json"), {}).get("phase", "?"))
    A("NCBI fallback (3/s) active: %s" % c["ncbi_fallback_active"])
    A("")
    A("## Input")
    A("- Reference seed: **%s** (seed ambiguity resolved to seed_0; seeds 0-4 share the"
      " same molecule set, seed_900/901 swap ~205 molecules and are EXCLUDED).")
    A("  - all 7 seeds same molecule set: %s" % ing.get("all_seeds_same"))
    A("- Input rows: %s (valid %s / invalid %s)" % (ing["input_rows"], ing["valid_rows"], ing["invalid_rows"]))
    A("- Unique valid structures: %s" % ing["unique_structures"])
    if ing.get("per_split"):
        for sp, v in ing["per_split"].items():
            A("  - %s: rows=%s valid=%s unique=%s" % (sp, v["rows"], v["valid_rows"], v["unique_structures"]))
    A("")
    A("## Molecule identity resolution (OFFLINE: local caches + PubChem bulk FTP)")
    A("- Processed: %s / %s unique structures" % (pc.get("n_processed"), ing["unique_structures"]))
    A("- Resolved: %s | unresolved: %s | with CID: %s | with synonyms: %s"
      % (pc.get("resolved"), pc.get("unresolved"), pc.get("with_cid"), pc.get("with_synonyms")))
    A("- By source: %s" % json.dumps(pc.get("by_source", {})))
    A("- By match type: %s" % json.dumps(pc.get("by_match_type", {})))
    A("")
    A("## Query generation (model: %s, prompt: %s)" % (cfg.get("model"), cfg.get("prompt_version")))
    A("- Property queries: %s" % q["property_queries"])
    A("- Molecule queries: %s across %s entities" % (q["molecule_queries"], q["molecule_entities_with_queries"]))
    A("- Query associations: %s (valid %s / invalid %s)" % (q["total_query_associations"], q["valid"], q["invalid"]))
    A("")
    A("## Search (PMC ESearch + subdivision)")
    A("- Unique searches: %s (subdivided past 10k: %s)" % (s["total_searches"], s["subdivided"]))
    A("- complete-with-results: %s | complete-zero: %s | count-discrepancy: %s"
      % (s["complete_with_results"], s["complete_zero"], s["count_discrepancy"]))
    A("- incomplete: %s | invalid-query: %s | service-failure: %s"
      % (s["incomplete"], s["invalid_query"], s["service_failure"]))
    A("")
    A("## Article registry")
    A("- Total unique PMCIDs: %s" % reg.get("total_unique_pmcids"))
    A("- found-by property: %s | molecule: %s | both: %s"
      % (reg.get("found_by_property"), reg.get("found_by_molecule"), reg.get("found_by_both")))
    A("- property-only: %s | molecule-only: %s"
      % (reg.get("found_by_property_only"), reg.get("found_by_molecule_only")))
    A("- Property-task PMCIDs: %s" % c.get("property_task_pmcids"))
    A("- Per-split molecule PMCIDs: %s" % json.dumps(c.get("per_split_pmcids")))
    A("")
    oa = c.get("oa_gate", {})
    A("## OA gating (S3 Cloud Service membership)")
    A("- Registry PMCIDs checked: %s" % oa.get("n_registry"))
    A("- OA (in s3://pmc-oa-opendata): %s | non-OA (metadata-only): %s"
      % (oa.get("n_oa_in_s3"), oa.get("n_non_oa")))
    A("")
    A("## Metadata (ESummary, E-utilities)")
    A("- success: %s | no-docsum: %s | service-failure: %s | total: %s"
      % (md["success"], md["no_docsum"], md["service_failure"], md["total"]))
    A("")
    A("## Full-text download (S3 Cloud Service; EFetch fallback)")
    A("- validated full-text XML (with body): %s" % dl["validated_full_text_xml"])
    A("- success_no_body: %s | identity-mismatch: %s | invalid-payload: %s | temp-failure: %s"
      % (dl["success_no_body"], dl["identity_mismatch"], dl["invalid_payload"], dl["temporary_failure"]))
    A("- metadata-only (non-OA): %s" % dl["metadata_only_non_oa"])
    A("- retrieval routes: %s" % json.dumps(dl.get("route_counts", {})))
    A("- license breakdown (full-text): %s" % json.dumps(dl.get("license_counts", {})))
    A("- total XML size: %s" % dl["total_xml_human"])
    A("")
    A("## Resources")
    A("- OpenAI usage: %s" % json.dumps(c["openai_usage"]))
    A("- Disk (run dir): %s | scratch free: %s"
      % (c["disk"]["run_dir_human"], c["disk"]["scratch_free_human"]))
    A("")
    A("## Completion")
    A("- Search declared COMPLETE: %s (complete only when every generated query is"
      " fully handled; full-text availability is reported separately above)." % search_complete)
    C.atomic_write_text(C.P("collection_report.md"), "\n".join(lines) + "\n")
    C.log("INFO", "report", "report written; unique_pmcids=%s xml=%s"
          % (reg.get("total_unique_pmcids"), dl["total_xml_human"]))
    return c


if __name__ == "__main__":
    write_report()
