"""
exp70-multi — v2 IDENTITY-CARD builder (CARD_SPEC v2, generalized per-dataset).

Implements CARD_SPEC sections 2,4,5,6,7,8,9 for the FOUR datasets lipo / sider / hiv /
tox21 (BBBP is a DIFFERENT agent's dataset and is NEVER written here). The active dataset
is selected at import time via EXP70_DATASET (same as datasets.py); everything
dataset/task-specific is driven by datasets.py (attribute_vocab(), tasks(), csv_path()).

This is an ADDITIVE sibling of build_indexes.py (which is left untouched): build_indexes.py
produces the legacy task_index/compound_index; build_cards.py produces the v2 cards.

Two card types:
  * COMPOUND card — one per molecule: deterministic IDENTITY layer (always present, even
    with zero literature) + RDKit computed descriptors + per-task dataset LABELS + the
    EVIDENCE layer (validated findings organized by attribute_id).
  * TASK card — one per prediction task (lipo=1, hiv=1, tox21=12, sider=27): identity +
    dataset block + task.* evidence.

Inputs  ({run_dir}/structuring/ unless EXP70_STRUCT_DIR overrides):
  evidence_records.jsonl, shortlists.json, rankings.jsonl (or medcpt_rankings.jsonl),
  query_registry.jsonl.  Identity: {run_dir}/data/molecule_identities_full.jsonl.
  Labels: datasets.csv_path(split) (long-format CSV, joined via identity row_ids).

Outputs (local /work mirror {cards_dir}, then `aws s3 sync` to the PER-DATASET path
  s3://molerag/<ds>_chunks/v2/cards/):
  tasks/<task_id>.json  compounds/<compound_id>.json  all_cards.jsonl  findings.jsonl
  index.json  validation_report.json  render/<compound_id>.md (optional)

Env knobs (all optional; sensible defaults):
  EXP70_STRUCT_DIR      structuring dir            (default {run_dir}/structuring)
  EXP70_IDENTITY_FILE   identity jsonl             (default {run_dir}/data/molecule_identities_full.jsonl)
  EXP70_CARDS_DIR       local output dir           (default {struct_dir}/cards)
  EXP70_CARDS_S3        s3 destination             (default s3://molerag/<ds>_chunks/v2/cards/)
  EXP70_CARDS_NO_S3     '1' to skip the s3 sync    (default off)
  EXP70_CARDS_COMPOUNDS 'all'|'evidence'           (default 'all' = identity card per molecule)
  EXP70_CARDS_ONLY      comma mol_ids to restrict  (default: no restriction)
  EXP70_CARDS_RENDER    '1'|'0' human .md render   (default '1')
  EXP70_PASSAGES_FILE   jsonl for quote re-check   (default: none; relies on quote_validated)
"""
import os, sys, json, csv, glob, math, collections, subprocess, statistics

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import datasets as DS
import card_schema as CS

PIPELINE_VERSION = "v2"
VOCAB = DS.attribute_vocab()

STRUCT_DIR = os.environ.get("EXP70_STRUCT_DIR") or C.P("structuring")
IDENTITY_FILE = os.environ.get("EXP70_IDENTITY_FILE") or C.P("data", "molecule_identities_full.jsonl")
CARDS_DIR = os.environ.get("EXP70_CARDS_DIR") or os.path.join(STRUCT_DIR, "cards")
CARDS_S3 = os.environ.get("EXP70_CARDS_S3") or ("s3://molerag/%s_chunks/v2/cards/" % DS.CFG["name"])
NO_S3 = os.environ.get("EXP70_CARDS_NO_S3", "0") == "1"
# Default to 'evidence' for the giant hiv set (41k molecules): writing an identity-only card
# per molecule to slow Lustre + S3 is wasteful when most have no literature. The full identity
# of every molecule remains in molecule_identities_full.jsonl; set EXP70_CARDS_COMPOUNDS=all to override.
COMPOUND_SCOPE = os.environ.get("EXP70_CARDS_COMPOUNDS",
                                "evidence" if DS.CFG["name"] == "hiv" else "all").lower()
ONLY = set(x for x in (os.environ.get("EXP70_CARDS_ONLY", "").split(",")) if x.strip())
RENDER = os.environ.get("EXP70_CARDS_RENDER", "1") == "1"
PASSAGES_FILE = os.environ.get("EXP70_PASSAGES_FILE")


# ====================================================================== loaders
def _read_jsonl(path):
    out = []
    if path and os.path.exists(path):
        for line in open(path):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _struct_path(name):
    """A structuring file, preferring EXP70_STRUCT_DIR, else the node-local filter base."""
    p = os.path.join(STRUCT_DIR, name)
    if os.path.exists(p):
        return p
    fb = os.path.join(os.environ.get("EXP70_FILTER_BASE", "/tmp/exp70_%s" % DS.CFG["name"]),
                      "structuring", name)
    return fb if os.path.exists(fb) else p


def load_records():
    return _read_jsonl(_struct_path("evidence_records.jsonl"))


def load_shortlists():
    return C.read_json(_struct_path("shortlists.json"), {}) or {}


def _judge_inputs_exist():
    """True if the judge/prescreen produced output for this dataset (locally staged or in S3).
    Used by the HOLLOW-CARD guard: if these exist but there are 0 evidence_records, structuring
    silently failed to load the kept passages and we must NOT finalize identity-only cards."""
    base = os.environ.get("EXP70_FILTER_BASE", "/tmp/exp70_%s" % DS.CFG["name"])
    if glob.glob(os.path.join(base, "judgments_qwen", "retention", "*", "*.jsonl")):
        return True
    if os.path.exists(os.path.join(base, "candidates_passages.jsonl")):
        return True
    try:
        r = subprocess.run(["aws", "s3", "ls",
                            "s3://molerag/%s_chunks/qwen_judgments/retention/" % DS.CFG["name"]],
                           capture_output=True, text=True, timeout=60)
        return bool(r.stdout.strip())
    except Exception:
        return False


def load_query_registry():
    """query_id -> task_id (task route only; compound templates are not separately reranked)."""
    q2t = {}
    for r in _read_jsonl(_struct_path("query_registry.jsonl")):
        q2t[r.get("query_id")] = r.get("task_id")
    return q2t


def load_rankings():
    """(query_id, passage_id) -> {rerank_score, rank, bm25_score}. Handles both the cohere
    (`cohere_cosine`,`rank`) and MedCPT (`medcpt_score`,`medcpt_rank`) ranking schemas."""
    path = _struct_path("rankings.jsonl")
    if not os.path.exists(path):
        path = _struct_path("medcpt_rankings.jsonl")
    out = {}
    for d in _read_jsonl(path):
        qid = d.get("query_id")
        for e in d.get("ranked", []):
            pid = e.get("passage_id")
            score = e.get("cohere_cosine")
            if score is None:
                score = e.get("medcpt_score")
            rank = e.get("rank", e.get("medcpt_rank"))
            out[(qid, pid)] = dict(rerank_score=score, rank=rank, bm25_score=e.get("bm25_score"))
    return out


def band_membership(shortlists):
    """passage_id -> smallest band K it appears in (across all task queries). Bands come
    from card_schema.KBANDS so they never drift from what the rerankers emit."""
    band = {}
    for _qid, bands in shortlists.items():
        for k in CS.KBANDS:
            for pid in bands.get("K%d" % k, []):
                if pid not in band or k < band[pid]:
                    band[pid] = k
    return band


_UNION_BAND = "K%d" % max(CS.KBANDS)   # largest band = superset (nested); the extraction union


def pid_to_queries(shortlists):
    """passage_id -> list of query_ids whose LARGEST-band (K1000) shortlist contains it.
    Uses the nested top band so passages that only reach a deep band (601-1000) still join
    to their query for retrieval provenance."""
    out = collections.defaultdict(list)
    for qid, bands in shortlists.items():
        for pid in bands.get(_UNION_BAND, []):
            out[pid].append(qid)
    return out


# ====================================================================== identity + labels
def load_identities():
    m = {}
    if not os.path.exists(IDENTITY_FILE):
        C.log("WARN", "cards", "identity file missing: %s" % IDENTITY_FILE)
        return m
    for line in open(IDENTITY_FILE):
        if line.strip():
            d = json.loads(line)
            m[d["mol_id"]] = d
    return m


_LABEL_CACHE = {"splits": None, "rows": {}, "targets": {}}


def _load_label_rows():
    """Read each split CSV once into a 0-indexed list of its data rows, and accumulate the
    per-task target domain (regression range / classification classes). Matches the identity
    row_ids convention seed_<seed>:<split>:<0-based-row-index-within-that-split-csv>."""
    if _LABEL_CACHE["splits"] is not None:
        return
    rows_by_split = {}
    targets = {}   # task_id -> dict(type, task_type, metric, values=set/min/max)
    for split in DS.SPLITS:
        path = DS.csv_path(split)
        if not os.path.exists(path):
            rows_by_split[split] = []
            continue
        rows = []
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
                task = row.get("task")
                if task is None:
                    continue
                tid = DS.task_id(task)
                t = targets.setdefault(tid, dict(task=task, task_type=row.get("task_type"),
                                                 metric=row.get("metric"),
                                                 classes=set(), vmin=None, vmax=None))
                lab = row.get("label")
                fv = CS.as_float(lab)
                if fv is not None:
                    t["vmin"] = fv if t["vmin"] is None else min(t["vmin"], fv)
                    t["vmax"] = fv if t["vmax"] is None else max(t["vmax"], fv)
                    t["classes"].add(("%g" % fv))
                elif lab is not None:
                    t["classes"].add(str(lab))
        rows_by_split[split] = rows
    _LABEL_CACHE["splits"] = DS.SPLITS
    _LABEL_CACHE["rows"] = rows_by_split
    _LABEL_CACHE["targets"] = targets


def labels_for(identity):
    """identity.labels: task_id -> {value, source:'dataset', split, seed}. Joined EXACTLY via
    row_ids (seed:split:idx) into the per-split CSVs — robust to SMILES canonicalization."""
    _load_label_rows()
    out = {}
    for rid in (identity.get("row_ids") or []):
        parts = str(rid).split(":")
        if len(parts) != 3:
            continue
        seed_s, split, idx_s = parts
        try:
            idx = int(idx_s)
            seed = int(seed_s.replace("seed_", "")) if seed_s.startswith("seed_") else 0
        except ValueError:
            continue
        rows = _LABEL_CACHE["rows"].get(split) or []
        if not (0 <= idx < len(rows)):
            continue
        row = rows[idx]
        task = row.get("task")
        if not task:
            continue
        tid = DS.task_id(task)
        lab = row.get("label")
        fv = CS.as_float(lab)
        out[tid] = dict(value=(fv if fv is not None else lab), source="dataset",
                        split=row.get("split") or split, seed=seed)
    return out


def task_target(tid):
    """target block for a task card: regression -> range; classification -> classes."""
    _load_label_rows()
    t = _LABEL_CACHE["targets"].get(tid)
    if not t:
        return dict(type="unknown")
    ttype = (t.get("task_type") or "").lower()
    if ttype.startswith("regress"):
        return dict(type="regression",
                    range=[t.get("vmin"), t.get("vmax")],
                    metric=t.get("metric"))
    classes = sorted(t.get("classes") or [])
    return dict(type=("binary" if len(classes) <= 2 else "multiclass"),
                classes=classes, metric=t.get("metric"))


def task_dataset_block(tid):
    """dataset provenance block (n_rows/n_unique/splits/seed) for a task card."""
    _load_label_rows()
    n_rows = 0
    smis = set()
    splits = []
    for split in DS.SPLITS:
        for row in (_LABEL_CACHE["rows"].get(split) or []):
            if DS.task_id(row.get("task") or "") == tid:
                n_rows += 1
                if row.get("smiles"):
                    smis.add(row["smiles"])
                if split not in splits:
                    splits.append(split)
    return dict(name="MoleculeNet %s" % DS.CFG["name"], task_in_dataset=tid.split(":", 1)[-1],
                n_rows=n_rows, n_unique_structures=len(smis), splits=splits, seed=0,
                source="moleculenet_property_scaffold/seed_0")


# ====================================================================== RDKit descriptors
_RD = {"ok": None}


def _rdkit():
    if _RD["ok"] is None:
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
            RDLogger.DisableLog("rdApp.*")
            _RD.update(ok=True, Chem=Chem, Descriptors=Descriptors,
                       Crippen=Crippen, rdMolDescriptors=rdMolDescriptors)
        except Exception as e:                           # pragma: no cover
            _RD.update(ok=False, err=str(e))
            C.log("WARN", "cards", "RDKit unavailable: %s" % e)
    return _RD


# functional-group SMARTS for a simple pKa-free charge-class heuristic
_ACID_SMARTS = ["C(=O)[OH]", "C(=O)[O-]", "S(=O)(=O)[OH]", "S(=O)(=O)[O-]",
                "P(=O)([OH])[OH]", "[OH]c1ccccc1[N+](=O)[O-]", "c1cc(=O)[nH]c(=O)[nH]1"]
_BASE_SMARTS = ["[NX3;H2,H1,H0;!$(NC=O);!$(N=*);!$([N+]);!$(Nc)]",  # aliphatic amine
                "[nX3;H0;$(n1cc[nH]c1)]", "[NX3;$(N=C-N)]"]         # imidazole-like / amidine


def charge_class(mol):
    rd = _rdkit()
    if not rd["ok"] or mol is None:
        return None
    Chem = rd["Chem"]
    has_acid = any(mol.HasSubstructMatch(Chem.MolFromSmarts(s))
                   for s in _ACID_SMARTS if Chem.MolFromSmarts(s) is not None)
    has_base = any(mol.HasSubstructMatch(Chem.MolFromSmarts(s))
                   for s in _BASE_SMARTS if Chem.MolFromSmarts(s) is not None)
    if has_acid and has_base:
        return "zwitterion"
    if has_acid:
        return "acid"
    if has_base:
        return "base"
    return "neutral"


def computed_descriptors(smiles):
    rd = _rdkit()
    if not rd["ok"] or not smiles:
        return {}
    mol = rd["Chem"].MolFromSmiles(smiles)
    if mol is None:
        return {}
    D, Cr, RD = rd["Descriptors"], rd["Crippen"], rd["rdMolDescriptors"]
    def v(x, u, **extra):
        d = dict(value=x, unit=u, source="computed")
        d.update(extra)
        return d
    return {
        "physchem.mw": v(round(D.MolWt(mol), 2), "g/mol"),
        "physchem.tpsa": v(round(RD.CalcTPSA(mol), 2), "A^2"),
        "physchem.logp": v(round(Cr.MolLogP(mol), 2), "log", method="Crippen"),
        "physchem.hbd": v(RD.CalcNumHBD(mol), "count"),
        "physchem.hba": v(RD.CalcNumHBA(mol), "count"),
        "physchem.rotatable_bonds": v(RD.CalcNumRotatableBonds(mol), "count"),
        "physchem.aromatic_rings": v(RD.CalcNumAromaticRings(mol), "count"),
        "ionization.charge_class": dict(value=charge_class(mol), source="computed",
                                        method="functional-group heuristic"),
    }


# ====================================================================== identity block
def _aliases(identity):
    out = []
    seen = set()
    prov = identity.get("identity_source") or "pubchem"

    def add(alias, atype, ambiguous=False):
        if not alias:
            return
        key = alias.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        out.append(dict(alias=alias, type=atype, provenance=prov, ambiguous=bool(ambiguous)))

    add(identity.get("preferred_name"), "generic")
    for c in (identity.get("dev_codes") or []):
        add(c, "code")
    syns = identity.get("synonyms_full") or identity.get("synonyms") or []
    dev = set(identity.get("dev_codes") or [])
    for s in syns:
        add(s, "code" if s in dev else "synonym")
    return out


def _parent_relationship(identity):
    cid = identity.get("cid")
    pcid = identity.get("parent_cid")
    if pcid is None or cid is None or str(pcid) == str(cid):
        return "self"
    if identity.get("desalt_mapping"):          # a desalt mapping implies a salt/mixture parent
        dm = identity["desalt_mapping"]
        if isinstance(dm, dict) and dm.get("type") == "mixture":
            return "mixture_component"
    return "salt_of"                            # structure differs from its PubChem parent


def identity_block(cid_, identity):
    smi_can = identity.get("canonical_smiles")
    pub = identity.get("pubchem_properties") or {}
    classes = identity.get("chemical_class") or []   # [] if class_source flags none — NEVER invent
    blk = dict(
        compound_id=cid_,
        canonical_name=identity.get("preferred_name"),
        iupac_name=identity.get("iupac_name"),
        smiles_canonical=smi_can,
        smiles_isomeric=pub.get("SMILES") or smi_can,
        inchikey=identity.get("inchikey_rdkit") or pub.get("InChIKey"),
        molecular_formula=pub.get("MolecularFormula"),
        pubchem_cid=identity.get("cid"),
        parent=dict(cid=identity.get("parent_cid"), name=identity.get("parent_name"),
                    relationship=_parent_relationship(identity)),
        chemical_classes=list(classes),
        chemical_class_source=identity.get("class_source"),
        aliases=_aliases(identity),
        identity_status=identity.get("identity_status"),
        labels=labels_for(identity),
    )
    return blk


# ====================================================================== findings
def _as_conditions(c):
    if isinstance(c, dict):
        return {k: v for k, v in c.items() if v is not None}
    if c in (None, "", "null", "none", "None"):
        return {}
    return {"raw": c}


def _as_list(q):
    if isinstance(q, list):
        return [x for x in q if x is not None]
    if q in (None, "", "null", "none"):
        return []
    return [q]


class Retriever(object):
    def __init__(self, shortlists):
        self.band = band_membership(shortlists)
        self.pid2q = pid_to_queries(shortlists)
        self.ranks = load_rankings()
        self.q2t = load_query_registry()

    def for_finding(self, r):
        pid = r.get("passage_id")
        et = r.get("entity_type")
        eid = r.get("entity_id")
        cand = self.pid2q.get(pid, [])
        if et == "prediction_task":
            own = [q for q in cand if self.q2t.get(q) == eid]
            pool = own or cand
        else:
            pool = cand
        best = None
        for q in pool:
            info = self.ranks.get((q, pid))
            if info is None:
                continue
            sc = info.get("rerank_score")
            if best is None or (sc is not None and (best[1].get("rerank_score") or -1e9) < sc):
                best = (q, info)
        band = CS.band_label(self.band.get(pid))
        ret = dict(matched_alias=(r.get("entity_mention") or None) if et == "compound" else None,
                   query_id=best[0] if best else None,
                   bm25_rank=(best[1].get("rank") if best else None),
                   rerank_score=(best[1].get("rerank_score") if best else None),
                   rank_band=band)
        if best:
            ret["bm25_score"] = best[1].get("bm25_score")
        return ret


def make_finding(r, retriever):
    et = r.get("entity_type")
    ap = r.get("attribute_path") or "other.other"
    attribute_id, group, _m = CS.normalize_attr_path(ap, et, VOCAB)
    pid = r.get("passage_id")
    value = r.get("value")
    article_id = r.get("article_id") or r.get("pmcid")
    ret = retriever.for_finding(r)
    ev_canon = CS.canon_evidence(r.get("evidence_type"))
    return dict(
        finding_id=CS.finding_id(r.get("entity_id"), attribute_id, pid,
                                 r.get("supporting_quote"), value),
        entity_type=et, entity_id=r.get("entity_id"),
        attribute_id=attribute_id, group=group,
        attribute_label=r.get("attribute_label") or CS.humanize_attr(attribute_id),
        attribute_label_original=r.get("attribute_label_original"),
        evidence_type=ev_canon, evidence_type_original=r.get("evidence_type"),
        value=value, unit=r.get("unit"),
        value_type=r.get("value_type") or CS.infer_value_type(value),
        qualifiers=_as_list(r.get("qualifiers")),
        conditions=_as_conditions(r.get("conditions")),
        direction=r.get("direction"),
        finding=r.get("finding"),
        supporting_quote=r.get("supporting_quote"),
        passage_id=pid, article_id=article_id,
        source_location=r.get("source_location"),
        source_url=r.get("source_url") or CS.pmc_url(article_id),
        retrieval=ret,
        identity_status=r.get("identity_status"),
        contradicts=[],
        quote_validated=r.get("quote_validated"),
        extraction=r.get("extraction") or dict(model=r.get("judge_model_extract"),
                                               prompt_version=CS.EXTRACT_PROMPT_VERSION,
                                               record_id=CS.record_id(pid, r.get("entity_id"),
                                                   ap, r.get("supporting_quote"),
                                                   value, r.get("unit"))),
    )


# ---- contradictions (spec s6: incompatible value/direction under COMPARABLE conditions) ----
_COND_KEYS = ("species", "system", "assay", "ph", "cell_line", "in_vivo", "raw")
_INC = {"increase", "increases", "increased", "higher", "up", "positive", "rise"}
_DEC = {"decrease", "decreases", "decreased", "lower", "down", "negative", "fall"}


def _cond_sig(conditions):
    return tuple(sorted((k, str(conditions.get(k))) for k in _COND_KEYS if conditions.get(k) is not None))


def _dir(d):
    if not d:
        return None
    s = str(d).strip().lower()
    if s in _INC:
        return "+"
    if s in _DEC:
        return "-"
    return None


def _incompatible(a, b):
    da, db = _dir(a.get("direction")), _dir(b.get("direction"))
    if da and db and da != db:
        return True
    va, vb = a.get("value"), b.get("value")
    vta, vtb = a.get("value_type"), b.get("value_type")
    if vta in ("boolean", "categorical") and vtb in ("boolean", "categorical"):
        na, nb = _norm_cat(va), _norm_cat(vb)
        if na is not None and nb is not None and na != nb:
            return True
    # numeric: flag only a NET-DIRECTION disagreement (opposite sign); plain spread is not a
    # contradiction (measurement variation stays "consistent", per spec example).
    if vta == "numeric" and vtb == "numeric" and CS.is_number(va) and CS.is_number(vb):
        fa, fb = CS.as_float(va), CS.as_float(vb)
        if fa is not None and fb is not None and abs(fa) > 1e-6 and abs(fb) > 1e-6:
            if (fa > 0) != (fb > 0):
                return True
    return False


def _norm_cat(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    truthy = {"true", "yes", "active", "positive", "substrate", "present", "1", "1.0"}
    falsy = {"false", "no", "inactive", "negative", "non-substrate", "nonsubstrate", "absent", "0", "0.0"}
    if s in truthy:
        return "T"
    if s in falsy:
        return "F"
    return s


def detect_contradictions(findings):
    by_sig = collections.defaultdict(list)
    for f in findings:
        by_sig[_cond_sig(f["conditions"])].append(f)
    for _sig, fs in by_sig.items():
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                if _incompatible(fs[i], fs[j]):
                    if fs[j]["finding_id"] not in fs[i]["contradicts"]:
                        fs[i]["contradicts"].append(fs[j]["finding_id"])
                    if fs[i]["finding_id"] not in fs[j]["contradicts"]:
                        fs[j]["contradicts"].append(fs[i]["finding_id"])


def _finding_sort_key(f):
    return (CS.band_order(f["retrieval"].get("rank_band")),
            -(f["retrieval"].get("rerank_score") or -1e9),
            -CS.evidence_strength(f["evidence_type"]))


def summarize_attribute(findings):
    has_contra = any(f["contradicts"] for f in findings)
    nums = [CS.as_float(f["value"]) for f in findings
            if f["value_type"] == "numeric" and CS.is_number(f["value"])]
    nums = [x for x in nums if x is not None]
    summ = dict(n_findings=len(findings),
                has_contradiction=has_contra,
                agreement=("mixed" if has_contra else "consistent"))
    if nums:
        summ["value_range"] = [min(nums), max(nums)]
        summ["consensus"] = round(statistics.median(nums), 4)
    else:
        vals = [f["value"] for f in findings if f["value"] is not None]
        if vals:
            cnt = collections.Counter(_norm_cat(v) or str(v) for v in vals)
            summ["value"] = cnt.most_common(1)[0][0]
        dirs = [_dir(f.get("direction")) for f in findings if _dir(f.get("direction"))]
        if dirs:
            summ["direction_consensus"] = collections.Counter(dirs).most_common(1)[0][0]
    return summ


def build_attributes(records, retriever):
    """Group this entity's records by attribute_id -> attribute block (spec s5)."""
    by_attr = collections.defaultdict(list)
    for r in records:
        f = make_finding(r, retriever)
        by_attr[f["attribute_id"]].append(f)
    attributes = {}
    for aid, fs in by_attr.items():
        detect_contradictions(fs)
        fs.sort(key=_finding_sort_key)
        passages = {f["passage_id"] for f in fs}
        articles = {f["article_id"] for f in fs}
        attributes[aid] = dict(
            group=aid.split(".")[0], attribute_id=aid,
            label=fs[0]["attribute_label"] or CS.humanize_attr(aid),
            n_findings=len(fs), n_passages=len(passages), n_articles=len(articles),
            summary=summarize_attribute(fs), findings=fs)
    return attributes


# ====================================================================== coverage
def _schema_paths(entity_type):
    return VOCAB["task_paths"] if entity_type in ("prediction_task", "task") else VOCAB["compound_paths"]


def coverage_block(attributes, entity_type, shortlists_band):
    present = sorted(attributes.keys())
    present_two = set()
    for a in present:
        k = CS.schema_key(a, entity_type, VOCAB)
        if k:
            present_two.add(k)
    groups_present = sorted({a.split(".", 1)[0] for a in present})
    schema = _schema_paths(entity_type)
    absent = sorted(p for p in schema if p not in present_two and not p.endswith(".other"))
    findings = [f for a in attributes.values() for f in a["findings"]]
    passages = {f["passage_id"] for f in findings}
    articles = {f["article_id"] for f in findings}
    short = {("k%d" % k): 0 for k in CS.KBANDS}   # cumulative: k<K> = passages with band <= K
    unresolved = sum(1 for f in findings if f.get("identity_status") == "unresolved")
    art_local = sum(1 for f in findings if f.get("identity_status") == "article-local")
    for pid in passages:
        b = shortlists_band.get(pid)
        if b is None:
            continue
        for k in CS.KBANDS:
            if b <= k:
                short["k%d" % k] += 1
    return dict(
        attributes_present=present, groups_present=groups_present, absent=absent,
        counts=dict(findings=len(findings), passages=len(passages), articles=len(articles),
                    bm25_candidates=None, shortlisted=short),
        flags=dict(unresolved_mentions=unresolved, article_local_mentions=art_local,
                   shallow_cutoff=False))


def _provenance(attributes):
    qids, aliases = set(), set()
    for a in attributes.values():
        for f in a["findings"]:
            ret = f.get("retrieval") or {}
            if ret.get("query_id"):
                qids.add(ret["query_id"])
            if ret.get("matched_alias"):
                aliases.add(ret["matched_alias"])
    return dict(query_ids=sorted(qids), aliases_used=sorted(aliases))


# ====================================================================== card assembly
def _models():
    return dict(embed=os.environ.get("EXP70_EMBED_MODEL", "cohere.embed-english-v3"),
                extract=os.environ.get("EXP70_EXTRACT_MODEL", "qwen3-32b (bedrock)"))


def compound_card(cid_, identity, records, retriever, band):
    attributes = build_attributes(records, retriever)
    idblk = identity_block(cid_, identity)
    return dict(
        card_id="compound:%s" % cid_, card_type="compound",
        schema_version=CS.CARD_SCHEMA_VERSION, pipeline_version=PIPELINE_VERSION,
        dataset=DS.CFG["name"], generated_at=C.now_iso(), models=_models(),
        identity=idblk,
        computed_descriptors=computed_descriptors(idblk.get("smiles_canonical")),
        attributes=attributes,
        coverage=coverage_block(attributes, "compound", band),
        provenance=_provenance(attributes))


def task_card(tid, task, bio, records, retriever, band):
    attributes = build_attributes(records, retriever)
    return dict(
        card_id="task:%s" % tid, card_type="task",
        schema_version=CS.CARD_SCHEMA_VERSION, pipeline_version=PIPELINE_VERSION,
        dataset=DS.CFG["name"], generated_at=C.now_iso(), models=_models(),
        identity=dict(task_id=tid, name=task, description=bio,
                      target=task_target(tid), dataset=task_dataset_block(tid)),
        attributes=attributes,
        coverage=coverage_block(attributes, "task", band),
        provenance=_provenance(attributes))


# ====================================================================== findings.jsonl
_FLAT_FIELDS = ["entity_type", "entity_id", "attribute_id", "group", "value", "unit",
                "value_type", "evidence_type", "conditions", "qualifiers", "direction",
                "identity_status", "passage_id", "article_id", "rank_band",
                "supporting_quote", "finding_id", "contradicts"]


def flatten_finding(f):
    row = {k: f.get(k) for k in _FLAT_FIELDS if k != "rank_band"}
    row["rank_band"] = (f.get("retrieval") or {}).get("rank_band")
    return row


# ====================================================================== render
def render_md(card):
    i = card["identity"]
    L = []
    if card["card_type"] == "compound":
        L.append("# %s  (`%s`)" % (i.get("canonical_name") or i["compound_id"], card["card_id"]))
        L.append("")
        L.append("- SMILES: `%s`" % (i.get("smiles_canonical") or ""))
        L.append("- InChIKey: `%s`  |  CID: %s  |  formula: %s"
                 % (i.get("inchikey"), i.get("pubchem_cid"), i.get("molecular_formula")))
        if i.get("chemical_classes"):
            L.append("- Classes: %s" % ", ".join(i["chemical_classes"]))
        cd = card.get("computed_descriptors") or {}
        if cd:
            L.append("- Computed: MW %s, TPSA %s, cLogP %s, charge %s" % (
                (cd.get("physchem.mw") or {}).get("value"),
                (cd.get("physchem.tpsa") or {}).get("value"),
                (cd.get("physchem.logp") or {}).get("value"),
                (cd.get("ionization.charge_class") or {}).get("value")))
        if i.get("labels"):
            L.append("- Dataset labels: " + "; ".join(
                "%s=%s (%s)" % (t, v.get("value"), v.get("split")) for t, v in i["labels"].items()))
    else:
        L.append("# Task: %s  (`%s`)" % (i.get("name"), card["card_id"]))
        L.append("")
        L.append("- %s" % (i.get("description") or ""))
        tgt = i.get("target") or {}
        L.append("- Target: %s %s" % (tgt.get("type"),
                                      tgt.get("classes") or tgt.get("range") or ""))
    L.append("")
    L.append("## Evidence (%d attributes)" % len(card["attributes"]))
    for aid in card["coverage"]["attributes_present"]:
        a = card["attributes"][aid]
        L.append("### %s — %s  (%d findings%s)"
                 % (aid, a["label"], a["n_findings"],
                    ", CONTRADICTION" if a["summary"].get("has_contradiction") else ""))
        for f in a["findings"][:5]:
            L.append("- [%s] %s %s%s — \"%s\" (%s, band %s)" % (
                f["evidence_type"], f.get("value"), f.get("unit") or "",
                (" dir=%s" % f["direction"]) if f.get("direction") else "",
                (f.get("supporting_quote") or "")[:140], f.get("article_id"),
                (f.get("retrieval") or {}).get("rank_band")))
    return "\n".join(L) + "\n"


# ====================================================================== validation (spec s8)
def load_passages_for(pids):
    """Load ONLY referenced passages for quote re-validation.

    Default source is EXP70_PASSAGES_FILE (a small jsonl). The node-local passage STORE is
    only walked when EXP70_VALIDATE_QUOTES=1 is set explicitly, because that store can be
    large and Lustre is slow/over-quota (memory rule: never walk the passage store blindly).
    When neither is available, validation falls back to the per-record `quote_validated`
    flag that extract.py already set at extraction time."""
    text = {}
    if PASSAGES_FILE and os.path.exists(PASSAGES_FILE):
        for line in open(PASSAGES_FILE):
            if line.strip():
                d = json.loads(line)
                if d.get("passage_id") in pids:
                    text[d["passage_id"]] = d.get("text") or ""
        return text
    if os.environ.get("EXP70_VALIDATE_QUOTES") != "1":
        return text
    tree = os.path.join(os.environ.get("EXP70_FILTER_BASE", "/tmp/exp70_%s" % DS.CFG["name"]),
                        "chunks", "passages")
    if os.path.isdir(tree):
        for f in glob.glob(os.path.join(tree, "*", "*.jsonl")):
            for line in open(f):
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("passage_id") in pids:
                    text[d["passage_id"]] = d.get("text") or ""
    return text


def _norm_ws(s):
    return " ".join((s or "").split())


def validate_cards(cards, registry_ids, passages=None):
    """Run CARD_SPEC s8 checks over emitted cards. Returns a list of violation dicts."""
    viol = []
    passages = passages or {}
    for card in cards:
        et = "task" if card["card_type"] == "task" else "compound"
        eid = card["identity"].get("task_id") or card["identity"].get("compound_id")
        if eid not in registry_ids:
            viol.append(dict(card=card["card_id"], kind="entity_not_in_registry", entity=eid))
        for aid, a in card["attributes"].items():
            if not CS.is_schema_valid(aid, et, VOCAB):
                viol.append(dict(card=card["card_id"], kind="attribute_not_in_schema", attribute_id=aid))
            for f in a["findings"]:
                fid = f.get("finding_id")
                for req in ("evidence_type", "value_type", "identity_status"):
                    if f.get(req) in (None, ""):
                        viol.append(dict(card=card["card_id"], finding=fid,
                                         kind="missing_required_field", field=req))
                if f.get("retrieval") is None:
                    viol.append(dict(card=card["card_id"], finding=fid, kind="missing_retrieval"))
                if not f.get("article_id") or not f.get("passage_id"):
                    viol.append(dict(card=card["card_id"], finding=fid, kind="missing_passage_ref"))
                q = f.get("supporting_quote")
                pid = f.get("passage_id")
                if q and pid in passages:
                    if _norm_ws(q)[:200] not in _norm_ws(passages[pid]):
                        viol.append(dict(card=card["card_id"], finding=fid,
                                         kind="quote_not_substring", passage_id=pid))
                elif q and f.get("quote_validated") is False:
                    viol.append(dict(card=card["card_id"], finding=fid, kind="quote_flagged_invalid"))
        # computed vs literature never silently merged: computed descriptors must be source=computed
        for k, d in (card.get("computed_descriptors") or {}).items():
            if isinstance(d, dict) and d.get("source") != "computed":
                viol.append(dict(card=card["card_id"], kind="computed_not_flagged", key=k))
    return viol


# ====================================================================== driver
def _write(path, obj):
    C.atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2), fsync=False)


def main():
    if DS.CFG["name"] == "bbbp" or "exp70_chunks/v2/cards" in CARDS_S3:
        raise SystemExit("REFUSING: build_cards.py never writes the BBBP agent's path "
                         "(s3://molerag/exp70_chunks/v2/cards/). dataset=%s s3=%s"
                         % (DS.CFG["name"], CARDS_S3))
    os.makedirs(os.path.join(CARDS_DIR, "tasks"), exist_ok=True)
    os.makedirs(os.path.join(CARDS_DIR, "compounds"), exist_ok=True)
    if RENDER:
        os.makedirs(os.path.join(CARDS_DIR, "render"), exist_ok=True)

    records = load_records()
    shortlists = load_shortlists()
    # HOLLOW-CARD GUARD: if the judge/prescreen produced output but we have 0 evidence records,
    # structuring failed to load the kept passages (staging/plumbing bug). Fail LOUDLY so the
    # caller does NOT mark STRUCT_DONE and finalize identity-only (hollow) cards. (0 records with
    # NO judge output is a legitimate not-collected-yet state and is allowed through.)
    if not records and _judge_inputs_exist():
        raise SystemExit(
            "REFUSING to finalize HOLLOW cards: 0 evidence_records but the judge/candidate "
            "inputs EXIST for dataset=%s. Structuring did not load the kept passages; fix the "
            "rerank/extract staging and re-run. NOT writing cards or STRUCT_DONE." % DS.CFG["name"])
    retriever = Retriever(shortlists)
    band = retriever.band
    identities = load_identities()
    C.log("INFO", "cards", "records=%d  identities=%d  tasks=%d  dataset=%s"
          % (len(records), len(identities), len(DS.tasks()), DS.CFG["name"]))

    comp_records = collections.defaultdict(list)
    task_records = collections.defaultdict(list)
    for r in records:
        if r.get("entity_type") == "compound" and r.get("entity_id"):
            comp_records[r["entity_id"]].append(r)
        elif r.get("entity_type") == "prediction_task" and r.get("entity_id"):
            task_records[r["entity_id"]].append(r)

    # which compounds get a card: 'all' (identity card per molecule, even zero-literature)
    # or 'evidence' (only molecules with >=1 finding). EXP70_CARDS_ONLY restricts further.
    if COMPOUND_SCOPE == "evidence":
        comp_ids = set(comp_records)
    else:
        comp_ids = set(identities) | set(comp_records)
    if ONLY:
        comp_ids &= ONLY
    comp_ids = sorted(comp_ids)

    all_cards_path = os.path.join(CARDS_DIR, "all_cards.jsonl")
    findings_path = os.path.join(CARDS_DIR, "findings.jsonl")
    fall = open(all_cards_path, "w")
    ffind = open(findings_path, "w")
    index = dict(dataset=DS.CFG["name"], schema_version=CS.CARD_SCHEMA_VERSION,
                 generated_at=C.now_iso(), pipeline_version=PIPELINE_VERSION,
                 structuring_dir=STRUCT_DIR, s3=CARDS_S3,
                 tasks={}, compounds={},
                 totals=dict(n_tasks=0, n_compounds=0, n_findings=0))
    emitted = []
    registry_ids = set(identities) | set(t[0] for t in DS.tasks())
    n_find_total = 0

    # ---- TASK cards ----
    for tid, task, bio, _topics in DS.tasks():
        card = task_card(tid, task, bio, task_records.get(tid, []), retriever, band)
        _write(os.path.join(CARDS_DIR, "tasks", "%s.json" % task), card)
        fall.write(json.dumps(card, ensure_ascii=False) + "\n")
        for a in card["attributes"].values():
            for f in a["findings"]:
                ffind.write(json.dumps(flatten_finding(f), ensure_ascii=False) + "\n")
                n_find_total += 1
        if RENDER:
            C.atomic_write_text(os.path.join(CARDS_DIR, "render", "%s.md" % task),
                                render_md(card), fsync=False)
        index["tasks"][tid] = dict(card_id=card["card_id"], path="tasks/%s.json" % task,
                                   name=task, coverage=dict(
                                       attributes_present=card["coverage"]["attributes_present"],
                                       groups_present=card["coverage"]["groups_present"],
                                       n_absent=len(card["coverage"]["absent"])),
                                   counts=card["coverage"]["counts"])
        emitted.append(card)

    # ---- COMPOUND cards ----
    n_comp = 0
    for cid_ in comp_ids:
        identity = identities.get(cid_, dict(mol_id=cid_, identity_status="unknown"))
        card = compound_card(cid_, identity, comp_records.get(cid_, []), retriever, band)
        _write(os.path.join(CARDS_DIR, "compounds", "%s.json" % cid_), card)
        fall.write(json.dumps(card, ensure_ascii=False) + "\n")
        nf = 0
        for a in card["attributes"].values():
            for f in a["findings"]:
                ffind.write(json.dumps(flatten_finding(f), ensure_ascii=False) + "\n")
                n_find_total += 1
                nf += 1
        if RENDER and nf:
            C.atomic_write_text(os.path.join(CARDS_DIR, "render", "%s.md" % cid_),
                                render_md(card), fsync=False)
        index["compounds"][cid_] = dict(card_id=card["card_id"], path="compounds/%s.json" % cid_,
                                        canonical_name=card["identity"].get("canonical_name"),
                                        coverage=dict(
                                            attributes_present=card["coverage"]["attributes_present"],
                                            groups_present=card["coverage"]["groups_present"],
                                            n_absent=len(card["coverage"]["absent"])),
                                        counts=card["coverage"]["counts"])
        emitted.append(card)
        n_comp += 1

    fall.close()
    ffind.close()
    index["totals"] = dict(n_tasks=len(DS.tasks()), n_compounds=n_comp, n_findings=n_find_total)
    _write(os.path.join(CARDS_DIR, "index.json"), index)

    # ---- validation (spec s8) ----
    all_pids = {f["passage_id"] for c in emitted for a in c["attributes"].values()
                for f in a["findings"] if f.get("passage_id")}
    passages = load_passages_for(all_pids) if all_pids else {}
    viol = validate_cards(emitted, registry_ids, passages)
    _write(os.path.join(CARDS_DIR, "validation_report.json"),
           dict(dataset=DS.CFG["name"], generated_at=C.now_iso(),
                n_cards=len(emitted), n_findings=n_find_total,
                n_passages_checked=len(passages), n_violations=len(viol),
                violations=viol[:500]))
    C.log("INFO", "cards", "built %d task + %d compound cards, %d findings, %d violations"
          % (len(DS.tasks()), n_comp, n_find_total, len(viol)))

    # ---- S3 sync to the PER-DATASET path ----
    if not NO_S3:
        rc = subprocess.run(["aws", "s3", "sync", CARDS_DIR, CARDS_S3, "--only-show-errors"],
                            timeout=1800, check=False)
        C.log("INFO", "cards", "s3 sync -> %s (rc=%s)" % (CARDS_S3, rc.returncode))
    else:
        C.log("INFO", "cards", "EXP70_CARDS_NO_S3=1 -> skipped s3 sync (local: %s)" % CARDS_DIR)
    return index


if __name__ == "__main__":
    main()
