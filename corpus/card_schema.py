"""
exp70-multi v2 Identity-Card — shared PURE helpers (no heavy deps: hashlib + re only).

Imported by BOTH extract.py (to additively enrich each emitted finding record to the
v2 `finding record` shape, CARD_SPEC s4) and build_cards.py (to assemble + validate
cards). Keeping these deterministic helpers in one module guarantees the finding fields
extract.py writes and the ones build_cards.py derives can never drift apart.

Nothing here touches the network, Bedrock, RDKit, S3 or the filesystem — it is safe to
import from the running extraction path (no credential/import side effects).
"""
import hashlib
import re

CARD_SCHEMA_VERSION = "v2"
EXTRACT_PROMPT_VERSION = "ext.v2"      # record-format tag for finding.extraction.prompt_version

# --------------------------------------------------------------------------- evidence type
# Canonicalize the extractor's verbose evidence_type strings (extract.py asks for
# 'experimental result' / 'computational prediction' / 'mechanism' / 'general relationship'
# / 'method') to the spec s4 tokens, with a strength ranking (experimental strongest) used
# for within-attribute finding ordering (spec s7 step 4).
_EVIDENCE_CANON = {
    "experimental result": "experimental", "experimental": "experimental",
    "experiment": "experimental", "measured": "experimental",
    "computational prediction": "computational", "computational": "computational",
    "computation": "computational", "prediction": "computational", "predicted": "computational",
    "mechanism": "mechanism", "mechanistic": "mechanism",
    "general relationship": "general_relationship", "general_relationship": "general_relationship",
    "relationship": "general_relationship",
    "method": "method", "methods": "method", "protocol": "method",
}
EVIDENCE_STRENGTH = {"experimental": 5, "computational": 4, "mechanism": 3,
                     "general_relationship": 2, "method": 1, "other": 0}


def canon_evidence(ev):
    if not ev:
        return "other"
    return _EVIDENCE_CANON.get(str(ev).strip().lower(), "other")


def evidence_strength(ev_canon):
    return EVIDENCE_STRENGTH.get(ev_canon, 0)


# --------------------------------------------------------------------------- rank bands
# CANONICAL nested shortlist band edges = the SINGLE SOURCE OF TRUTH for the 5 bands.
# Imported by medcpt_rerank.py (hence cohere_rerank.py via MR), build_indexes.py,
# build_cards.py and extract.py so the band set can never drift between stages. The
# shortlists are nested: K200 subset K400 subset K600 subset K800 subset K1000; the
# LARGEST band (K1000) union is what extraction consumes.
KBANDS = [200, 400, 600, 800, 1000]


def _derive_band_labels(kbands):
    """Contiguous rank-band labels from the band edges: [200,400,...] -> ['1-200','201-400',...]."""
    labels, prev = [], 0
    for k in kbands:
        labels.append("%d-%d" % (prev + 1, k))
        prev = k
    return labels


BAND_LABELS = _derive_band_labels(KBANDS)   # ["1-200","201-400","401-600","601-800","801-1000"]
_BAND_ORDER = {lab: i for i, lab in enumerate(BAND_LABELS)}
_BAND_ORDER[None] = len(BAND_LABELS)


def band_order(band):
    """Sort key: smaller = better (retrieved in a tighter shortlist)."""
    return _BAND_ORDER.get(band, len(BAND_LABELS))


def band_label(smallest_k):
    """Smallest shortlist K a passage appears in -> rank-band label (spec s4). Derived from
    KBANDS so adding/removing a band here propagates everywhere."""
    if smallest_k is None:
        return None
    for i, k in enumerate(KBANDS):
        if smallest_k <= k:
            return BAND_LABELS[i]
    return None


# --------------------------------------------------------------------------- stable ids
def _h(*parts):
    s = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def finding_id(entity_id, attribute_id, passage_id, quote, value):
    return "f_" + _h(entity_id, attribute_id, passage_id, (quote or "")[:120], value)[:16]


def record_id(passage_id, entity_id, attribute_path, quote, value, unit):
    return "r_" + _h(passage_id, entity_id, attribute_path, (quote or "")[:120], value, unit)[:16]


# --------------------------------------------------------------------------- value typing
_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_BOOL_TOKENS = {"true", "false", "yes", "no"}
_CATEGORICAL_TOKENS = {"active", "inactive", "positive", "negative", "substrate",
                       "non-substrate", "nonsubstrate", "present", "absent", "agonist",
                       "antagonist", "acid", "base", "neutral", "zwitterion"}


def infer_value_type(value):
    """numeric | boolean | categorical | text (spec s3). Deterministic; null -> text."""
    if value is None:
        return "text"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "numeric"
    s = str(value).strip()
    if not s:
        return "text"
    if _NUM_RE.match(s.replace(",", "")):
        return "numeric"
    low = s.lower()
    if low in _BOOL_TOKENS:
        return "boolean"
    if low in _CATEGORICAL_TOKENS:
        return "categorical"
    # a short single-token value with no spaces reads as a category label (e.g. "MDR1-MDCK")
    if len(s) <= 40 and " " not in s:
        return "categorical"
    return "text"


def is_number(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if value is None:
        return False
    return bool(_NUM_RE.match(str(value).strip().replace(",", "")))


def as_float(value, default=None):
    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- source url
def pmc_url(pmcid):
    """PMC article URL from a pmcid (accepts 'PMC1234567' or a bare numeric accession)."""
    if not pmcid:
        return None
    s = str(pmcid).strip()
    if not s:
        return None
    if s.upper().startswith("PMC"):
        acc = s.upper()
    elif s.isdigit():
        acc = "PMC" + s
    else:
        acc = s
    return "https://www.ncbi.nlm.nih.gov/pmc/articles/%s/" % acc


# --------------------------------------------------------------------------- attribute keys
def humanize_attr(attribute_id):
    """A readable label for an attribute_id's non-group portion (not a value, just display).
    Dot-safe: only the leading group segment is stripped, so a dotted attribute name such as
    'logD7.4' is kept intact."""
    tail = attribute_id.split(".", 1)[1] if "." in attribute_id else attribute_id
    return tail.replace("_", " ").strip() or attribute_id


def _vocab_sets(vocab, entity_type):
    key = "task_paths" if entity_type in ("prediction_task", "task") else "compound_paths"
    paths = set(vocab.get(key) or [])
    # Group = the segment before the FIRST dot. Group names never contain a dot; attribute
    # names CAN (e.g. "lipophilicity.logD7.4", "ionization.pct_ionized_pH7.4"), so we must
    # never re-derive the 2-level boundary positionally — always match against `paths`.
    groups = set(p.split(".", 1)[0] for p in paths if "." in p)
    return paths, groups


def schema_key(attribute_id, entity_type, vocab):
    """The canonical 2-level schema path an attribute_id belongs to, or None.

    Handles dotted attribute names + optional 3-level subkeys by longest-match against the
    controlled `paths` set rather than by positional splitting."""
    if attribute_id is None:
        return None
    paths, _g = _vocab_sets(vocab, entity_type)
    if attribute_id in paths:
        return attribute_id
    best = None
    for p in paths:
        if attribute_id.startswith(p + ".") and (best is None or len(p) > len(best)):
            best = p
    return best


def normalize_attr_path(attribute_path, entity_type, vocab):
    """Normalize a raw attribute_path against datasets.attribute_vocab() for this dataset.

    Returns (attribute_id, group, matched_schema: bool).

    * A 2-level 'group.attribute' that is in the schema is kept verbatim (attribute names
      may themselves contain dots, e.g. 'lipophilicity.logD7.4').
    * An OPTIONAL 3-level 'group.attribute.subkey' whose 'group.attribute' part is a schema
      key is kept whole (the subkey is a refinement; group roll-ups still work by prefix,
      and it still counts as 'in schema' for validation s8).
    * Anything else routes to '<group>.other' if the group is known, else 'other.other';
      the caller keeps attribute_label_original (the paper's verbatim wording).
    """
    ap = (attribute_path or "other.other").strip() or "other.other"
    paths, groups = _vocab_sets(vocab, entity_type)
    group = ap.split(".", 1)[0] or "other"
    if ap in paths:
        return ap, group, True
    if schema_key(ap, entity_type, vocab) is not None:   # 3-level refinement of a schema key
        return ap, group, True
    if group in groups:
        return group + ".other", group, False
    return "other.other", "other", False


def schema_two_level(attribute_id, entity_type=None, vocab=None):
    """Canonical 2-level schema path for an attribute_id. With vocab, uses longest-match
    (dot-safe); without it, falls back to the naive first-two-segments (legacy callers)."""
    if vocab is not None:
        k = schema_key(attribute_id, entity_type, vocab)
        if k is not None:
            return k
    parts = attribute_id.split(".")
    if len(parts) >= 2:
        return parts[0] + "." + parts[1]
    return attribute_id


def is_schema_valid(attribute_id, entity_type, vocab):
    """True if attribute_id is a schema key (dot-safe, incl. 3-level refinements) or 'other'."""
    if attribute_id is None:
        return False
    if attribute_id == "other.other" or attribute_id.endswith(".other"):
        return True
    return schema_key(attribute_id, entity_type, vocab) is not None
