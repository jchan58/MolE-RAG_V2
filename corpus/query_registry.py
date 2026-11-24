"""
Structuring step 2 — VERSIONED QUERY REGISTRY (task route + compound route).

Task route: for each prediction task, a semantic question (for MedCPT reranking) + a
BM25 term string (candidate retrieval), derived from datasets.py per-task bio/topics.
Compound route: reusable templates instantiated per compound at rerank time (the
matched alias is substituted; see medcpt_rerank). No GPU, no data dependency for the
task route. Writes RUN_DIR/structuring/query_registry.jsonl (+ S3).

Fields: query_id, route(task|compound), task_id, compound_id, topic_id, semantic_query,
bm25_query, query_version.
"""
import os, sys, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import datasets as DS

QUERY_VERSION = "exp70-multi-qreg-v1"

# Reusable compound-attribute templates ({alias} substituted per compound at rerank).
# Chosen to match each dataset's attribute groups (datasets.attribute_vocab()).
_COMPOUND_TEMPLATES = {
    "lipo": [
        ("logp_logd", "What is the measured logP, logD or octanol-water partition coefficient of {alias}?"),
        ("ionization", "What is the pKa, ionization state, or fraction ionized of {alias}?"),
        ("physchem", "What physicochemical properties (TPSA, molecular weight, H-bond donors/acceptors) are reported for {alias}?"),
        ("permeability", "What membrane permeability (Caco-2, PAMPA, MDCK) or solubility is reported for {alias}?"),
        ("measurement", "How was the lipophilicity of {alias} measured (shake-flask, chromatographic, computational)?"),
    ],
    "hiv": [
        ("antiviral", "What anti-HIV antiviral activity, EC50/IC50 potency, or percent inhibition is reported for {alias}?"),
        ("mechanism", "What is the anti-HIV mechanism or target class (reverse transcriptase, integrase, protease, entry/fusion) of {alias}?"),
        ("selectivity", "What cytotoxicity (CC50) or selectivity index is reported for {alias}?"),
        ("resistance", "What HIV drug-resistance mutations or cross-resistance are associated with {alias}?"),
        ("admet", "What logP, permeability, or metabolic stability is reported for {alias}?"),
    ],
    "tox21": [
        ("nr_activity", "What nuclear-receptor (AR, ER, AhR, aromatase, PPAR-gamma) agonism or antagonism activity is reported for {alias}?"),
        ("stress", "What cellular stress-response (ARE/Nrf2, heat-shock, mitochondrial, p53, ATAD5) activity is reported for {alias}?"),
        ("ac50", "What AC50 or assay activity (active/inactive) in toxicity assays is reported for {alias}?"),
        ("structure", "What structural alerts or toxicophores are noted for {alias}?"),
    ],
    "sider": [
        ("adr_organ", "What adverse drug reactions or organ-system toxicities are reported for {alias}?"),
        ("severity", "Does {alias} have a black-box warning, contraindication, or notable adverse-event frequency?"),
        ("mechanism", "What is the mechanism of {alias}'s adverse effects (on-target vs off-target)?"),
        ("biomarker", "What safety biomarkers (ALT/AST, QT, creatinine) are associated with {alias}?"),
        ("pharmacology", "What is the drug class, target/MOA, or indication of {alias}?"),
    ],
}


def _qid(route, key):
    return "q_%s_%s" % (route, hashlib.sha1(key.encode()).hexdigest()[:12])


def build_task_registry():
    """Task-route queries (one per task; semantic question + BM25 terms)."""
    rows = []
    for tid, task, bio, topics in DS.tasks():
        semantic = ("What experimental results, measurements, mechanisms, or structure-activity "
                    "relationships describe %s?" % bio)
        bm25 = " ".join(topics)
        rows.append(dict(query_id=_qid("task", tid), route="task", task_id=tid,
                         compound_id=None, topic_id=tid, semantic_query=semantic,
                         bm25_query=bm25, query_version=QUERY_VERSION))
    return rows


def compound_templates():
    return _COMPOUND_TEMPLATES.get(DS.CFG["name"], _COMPOUND_TEMPLATES["lipo"])


def build():
    C.ensure_dirs()
    out_dir = C.P("structuring")
    os.makedirs(out_dir, exist_ok=True)
    task_rows = build_task_registry()
    path = os.path.join(out_dir, "query_registry.jsonl")
    with open(path, "w") as f:
        for r in task_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # compound templates stored separately (instantiated per matched compound at rerank)
    tmpl = [dict(topic_id=t, template=tpl, route="compound", query_version=QUERY_VERSION)
            for (t, tpl) in compound_templates()]
    C.write_json(os.path.join(out_dir, "compound_templates.json"),
                 dict(version=QUERY_VERSION, dataset=DS.CFG["name"], templates=tmpl))
    C.log("INFO", "query_registry", "task-route queries=%d compound-templates=%d -> %s"
          % (len(task_rows), len(tmpl), path))
    return dict(n_task_queries=len(task_rows), n_compound_templates=len(tmpl), path=path)


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
