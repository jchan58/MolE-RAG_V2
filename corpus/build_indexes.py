"""
Structuring steps 10, 14, 15 — build the THREE evidence indexes, the
200/400/600/800/1000 shortlist comparison, and the passage-preview artifact (no pause —
run end to end). Delta-side, no GPU. Joins compound identity metadata from
molecule_identities_full.jsonl.

Inputs (RUN_DIR/structuring/): evidence_records.jsonl, shortlists.json, medcpt_rankings.jsonl,
query_registry.jsonl. Outputs: task_index.json, compound_index.json, preview.json,
comparison_200_400_600_800_1000.json. Pushed to S3.

Band set = card_schema.KBANDS (single source of truth; same bands the rerankers produce).
"""
import os, sys, json, glob, collections, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import datasets as DS
import card_schema as CS   # canonical KBANDS

OUT = C.P("structuring")
S3 = os.environ.get("EXP70_STRUCT_S3", "s3://molerag/%s_chunks/structuring/" % DS.CFG["name"])
KBANDS = CS.KBANDS
_COMPARISON_NAME = "comparison_%s.json" % "_".join(str(k) for k in KBANDS)
_FIND_FIELDS = ["evidence_type", "finding", "value", "unit", "conditions", "qualifiers",
                "supporting_quote", "passage_id", "pmcid", "identity_status",
                "attribute_label_original", "quote_validated"]


def _identity_map():
    m = {}
    p = C.P("data", "molecule_identities_full.jsonl")
    if not os.path.exists(p):
        return m
    for line in open(p):
        d = json.loads(line)
        m[d["mol_id"]] = dict(
            compound_id=d["mol_id"], preferred_name=d.get("preferred_name"),
            iupac_name=d.get("iupac_name"), synonyms=d.get("synonyms_full") or d.get("synonyms"),
            dev_codes=d.get("dev_codes"), canonical_smiles=d.get("canonical_smiles"),
            inchikey=d.get("inchikey_rdkit"), cid=d.get("cid"),
            parent_cid=d.get("parent_cid"), parent_name=d.get("parent_name"),
            chemical_class=d.get("chemical_class"), class_source=d.get("class_source"),
            identity_status=d.get("identity_status"))
    return m


def _records():
    p = os.path.join(OUT, "evidence_records.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def _band_membership():
    """passage_id -> smallest band it appears in (bands nested 200<400<600<800<1000)."""
    sl = C.read_json(os.path.join(OUT, "shortlists.json"), {}) or {}
    band = {}
    for qid, bands in sl.items():
        for k in KBANDS:
            for pid in bands.get("K%d" % k, []):
                if pid not in band or k < band[pid]:
                    band[pid] = k
    return band


def _finding(r):
    return {k: r.get(k) for k in _FIND_FIELDS}


def build_indexes(records):
    task_idx = collections.defaultdict(lambda: collections.defaultdict(list))
    comp_idx = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        ap = r.get("attribute_path") or "other.other"
        if r.get("entity_type") == "prediction_task":
            task_idx[r.get("entity_id")][ap].append(_finding(r))
        elif r.get("entity_type") == "compound":
            comp_idx[r.get("entity_id")][ap].append(_finding(r))
    return task_idx, comp_idx


def main():
    os.makedirs(OUT, exist_ok=True)
    records = _records()
    if not records:
        C.log("WARN", "indexes", "no evidence_records yet — aborting"); return
    idmap = _identity_map()
    band = _band_membership()

    task_idx, comp_idx = build_indexes(records)
    # task index
    task_out = {tid: {"task_id": tid, "attributes": {ap: f for ap, f in attrs.items()}}
                for tid, attrs in task_idx.items()}
    C.write_json(os.path.join(OUT, "task_index.json"), task_out)
    # compound index WITH identity metadata joined
    comp_out = {}
    for cid, attrs in comp_idx.items():
        comp_out[cid] = dict(identity=idmap.get(cid, {"compound_id": cid, "identity_status": "unknown"}),
                             attributes={ap: f for ap, f in attrs.items()})
    C.write_json(os.path.join(OUT, "compound_index.json"), comp_out)

    # shortlist comparison across all KBANDS (reuse the largest-band records filtered by
    # band membership; extraction was done ONCE over the K-max union, so no re-extraction)
    comp = {}
    seen_findings = set()
    for k in KBANDS:
        recs_k = [r for r in records if band.get(r.get("passage_id"), 999) <= k]
        passages_k = {r.get("passage_id") for r in recs_k}
        arts_k = {r.get("pmcid") for r in recs_k}
        tcov = {r.get("entity_id") for r in recs_k if r.get("entity_type") == "prediction_task"}
        ccov = {r.get("entity_id") for r in recs_k if r.get("entity_type") == "compound"}
        fkeys = {(r.get("entity_id"), r.get("attribute_path"), (r.get("supporting_quote") or "")[:80]) for r in recs_k}
        new = len(fkeys - seen_findings); seen_findings |= fkeys
        comp["K%d" % k] = dict(records=len(recs_k), unique_passages=len(passages_k),
                               unique_articles=len(arts_k), supported_findings=len(fkeys),
                               new_findings_vs_smaller_band=new,
                               task_attribute_coverage=len(tcov), compound_coverage=len(ccov))
    C.write_json(os.path.join(OUT, _COMPARISON_NAME),
                 dict(dataset=DS.CFG["name"], built_at=C.now_iso(), bands=comp,
                      note="more rows != better quality; choose shortlist size on DEV not test"))

    # passage preview (organized by entity, with rank band labels)
    rankmap = {}
    rp = os.path.join(OUT, "rankings.jsonl")
    if not os.path.exists(rp):
        rp = os.path.join(OUT, "medcpt_rankings.jsonl")
    if os.path.exists(rp):
        for l in open(rp):
            d = json.loads(l)
            for e in d.get("ranked", []):
                rankmap.setdefault(e["passage_id"], e)
    bands_legend, _prev = {}, None
    for lab, k in zip(CS.BAND_LABELS, KBANDS):
        bands_legend[lab] = ("in K%d (tightest shortlist)" % k if _prev is None
                             else "in K%d but not K%d" % (k, _prev))
        _prev = k
    preview = dict(dataset=DS.CFG["name"], built_at=C.now_iso(),
                   n_task_entities=len(task_out), n_compound_entities=len(comp_out),
                   bands_legend=bands_legend,
                   sample_records=records[:50])
    C.write_json(os.path.join(OUT, "preview.json"), preview)

    summary = dict(dataset=DS.CFG["name"], built_at=C.now_iso(), n_records=len(records),
                   n_task_entities=len(task_out), n_compound_entities=len(comp_out),
                   comparison=comp)
    C.write_json(os.path.join(OUT, "indexes_summary.json"), summary)
    subprocess.run(["aws", "s3", "sync", OUT, S3, "--only-show-errors"], timeout=1800, check=False)
    C.log("INFO", "indexes", "INDEXES DONE: %s" % json.dumps(summary["comparison"]))


if __name__ == "__main__":
    main()
