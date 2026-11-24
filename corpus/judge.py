"""
LLM relevance judging (gpt-5.4-mini, structured output, BATCHED).

Two judgments, run per article over its chunked passages:
  * PROPERTY passages -> "Is this passage relevant to BBBP?" (broad usefulness).
  * MOLECULE passages -> "Does this passage contain SUBSTANTIVE info about THE
    specified target molecule?" The target molecule_id + verified aliases are
    supplied (from collection provenance); the judge still verifies the passage
    actually discusses that molecule (identity_match) and the info level.

Batching: up to BATCH passages per structured-output call; the model returns one
independent verdict per passage_id; a strict 1:1 mapping is validated (missing
verdicts are retried, then marked uncertain). Each verdict must cite evidence only
from its own passage.

Context expansion (refinement 6): when a verdict is needs_context (or uncertain
leaning to exclude), neighboring paragraphs / the parent table are loaded and the
passage is RE-judged singly (up to 2 rounds). Both initial and updated verdicts are
saved; only after context fails is uncertain/exclude final.

Deterministic retention -> keep | review | exclude | pending.
Raw model responses + token usage preserved.
"""
import os, sys, json, re, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import merge as M
import chunker as CH
import datasets as DS

JUDGE_MODEL = "local-or-bedrock"
JUDGE_PROMPT_VERSION = "exp70-multi-judge-v1"
BATCH = 15
MAX_CTX_ROUNDS = 2
MCT = 6000

# Per-dataset PROPERTY relevance prompts (from datasets.py). MOLECULE prompts below
# are dataset-agnostic (substantive info about THE target molecule).
PROPERTY_SYSTEM = DS.judge_property_system()

MOLECULE_SYSTEM = (
    "You are a biomedical relevance filter. Each supplied item has a passage and a TARGET "
    "molecule (its id, preferred name, and aliases). For EACH item: (1) identity_match — does "
    "the passage actually discuss THAT specific molecule? 'supported' if it clearly refers to that "
    "compound, 'ambiguous' if a matched name could denote something else or is unclear, 'absent' if "
    "the molecule is not discussed. (2) info_level — 'substantive' if the passage gives real "
    "information/data/claims about that molecule (properties, behavior, BBB, pharmacology, "
    "structure-activity, etc.), 'mention_only' if merely named without substantive info, 'none' if "
    "not present, 'uncertain' if unclear. Use ONLY the passage's own text; aliases are to RECOGNIZE "
    "the molecule, not to assume relevance. Return ONLY JSON: {\"verdicts\":[{\"passage_id\":<str>,"
    "\"identity_match\":\"supported|ambiguous|absent\",\"info_level\":\"substantive|mention_only|none|"
    "uncertain\",\"needs_context\":<bool>,\"evidence_spans\":[<short quotes from THIS passage>],"
    "\"reason\":<str>}]}. Exactly one verdict per supplied passage_id."
)


# ---------------- minimal decision-only prompts (output-token lever) ----------------
# The hot-path judge emits ONLY the decision label per passage_id (no evidence_spans,
# no reason, no needs_context, no chain-of-thought). This cuts generated tokens ~10-40x.
# Evidence spans / reasons are regenerated in a cheap SECOND PASS over KEPT passages only.
PROPERTY_SYSTEM_MIN = DS.judge_property_system_min()

MOLECULE_SYSTEM_MIN = (
    "You are a biomedical relevance filter. You receive a JSON object 'items' mapping an id -> "
    "{\"target\":{molecule_id, preferred_name, aliases}, \"text\": passage}. For EACH item decide, "
    "using ONLY the passage text: (A) identity_match - does the passage actually discuss THAT "
    "specific molecule? supported | ambiguous | absent. (B) info_level - does it give substantive "
    "info/data/claims about that molecule (properties, behavior, BBB, pharmacology, SAR)? "
    "substantive | mention_only | none | uncertain. Aliases are only to RECOGNIZE the molecule, "
    "not to assume relevance. "
    "Output ONLY a single JSON object mapping EACH id to the string \"<identity_match>|<info_level>\". "
    "Example: {\"p1@123\":\"supported|substantive\",\"p2@45\":\"absent|none\"}. "
    "Include every supplied id exactly once and no other keys. "
    "Do NOT output explanations, reasons, evidence, extra fields, or any <think> text. "
    "Output the JSON object and nothing else."
)


def _loads_obj(content):
    """Parse the model's single JSON object (strip code fences and any <think>...</think>)."""
    if not content:
        return None
    t = content.strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t); t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


_PROP_LABELS = {"relevant": "relevant", "irrelevant": "not_relevant",
                "not_relevant": "not_relevant", "not relevant": "not_relevant",
                "uncertain": "uncertain"}
_ID_LABELS = {"supported", "ambiguous", "absent"}
_IL_LABELS = {"substantive", "mention_only", "none", "uncertain"}


def parse_property_map(content):
    """Compact property map -> {passage_id: {"relevance": <canonical>}}. None on parse failure."""
    obj = _loads_obj(content)
    if not isinstance(obj, dict):
        return None
    out = {}
    for pid, lab in obj.items():
        if not isinstance(lab, str):
            continue
        out[str(pid)] = {"relevance": _PROP_LABELS.get(lab.strip().lower().replace("-", "_"), "uncertain")}
    return out


def parse_molecule_map(content):
    """Compact molecule map -> {id: {"identity_match","info_level"}}. None on parse failure."""
    obj = _loads_obj(content)
    if not isinstance(obj, dict):
        return None
    out = {}
    for pid, lab in obj.items():
        if not isinstance(lab, str):
            continue
        parts = [p.strip().lower().replace(" ", "_") for p in re.split(r"[|/,;:]", lab)]
        im = next((p for p in parts if p in _ID_LABELS), "ambiguous")
        il = next((p for p in parts if p in _IL_LABELS), "uncertain")
        out[str(pid)] = {"identity_match": im, "info_level": il}
    return out


def judge_config():
    return dict(
        model=JUDGE_MODEL, prompt_version=JUDGE_PROMPT_VERSION, batch_size=BATCH,
        max_completion_tokens=MCT, max_context_rounds=MAX_CTX_ROUNDS,
        endpoint=C.ENDPOINTS["openai"], chunker_version=CH.CHUNKER_VERSION,
        property_system=PROPERTY_SYSTEM, molecule_system=MOLECULE_SYSTEM,
        retention_policy=dict(
            property={"relevant": "keep", "uncertain": "review", "not_relevant": "exclude"},
            molecule={"supported+substantive": "keep", "supported+mention_only": "review",
                      "absent": "exclude", "none": "exclude",
                      "ambiguous": "review", "uncertain": "review",
                      "unresolved_needs_context": "pending"}),
        created_at=C.now_iso())


def _parse_verdicts(content):
    if not content:
        return None
    t = content.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t); t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if isinstance(obj, dict) and isinstance(obj.get("verdicts"), list):
        return obj["verdicts"]
    if isinstance(obj, list):
        return obj
    return None


def _raw_path(key):
    return C.P("raw_judge_responses", M.shard_of(key.split("_")[0]) if key[0].isdigit() else "misc",
               key + ".json")


def _call(system, user_obj, raw_key):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)}]
    ok, content, raw, usage, note = C.openai_chat(messages, max_completion_tokens=MCT, model=JUDGE_MODEL)
    try:
        C.write_json(C.P("raw_judge_responses", "%s.json" % raw_key),
                     dict(ok=ok, note=note, usage=usage, content=content), )
    except Exception:
        pass
    if usage:
        C.record_usage("judge_call", raw_key=raw_key,
                       prompt_tokens=usage.get("prompt_tokens"),
                       completion_tokens=usage.get("completion_tokens"),
                       total_tokens=usage.get("total_tokens"))
    if not ok:
        return None, note
    return _parse_verdicts(content), note


def judge_property_batch(passages, raw_key):
    """passages: list of dicts with passage_id + text. Returns {passage_id: verdict}."""
    user = dict(passages=[{"passage_id": p["passage_id"], "text": p["text"][:6000]} for p in passages])
    verdicts, note = _call(PROPERTY_SYSTEM, user, raw_key)
    out = {}
    if verdicts:
        for v in verdicts:
            if isinstance(v, dict) and v.get("passage_id"):
                out[v["passage_id"]] = v
    return out


def judge_molecule_batch(items, raw_key):
    """items: list of dicts {passage_id, text, target:{molecule_id, preferred_name, aliases}}."""
    user = dict(items=[{"passage_id": it["passage_id"], "target": it["target"],
                        "text": it["text"][:6000]} for it in items])
    verdicts, note = _call(MOLECULE_SYSTEM, user, raw_key)
    out = {}
    if verdicts:
        for v in verdicts:
            if isinstance(v, dict) and v.get("passage_id"):
                out[v["passage_id"]] = v
    return out


# ---------------- retention ----------------
def retention_property(v):
    if not v:
        return "pending"
    return {"relevant": "keep", "uncertain": "review", "not_relevant": "exclude",
            "irrelevant": "exclude"}.get(v.get("relevance"), "review")


def retention_molecule(v):
    if not v:
        return "pending"
    im, il = v.get("identity_match"), v.get("info_level")
    if im == "supported" and il == "substantive":
        return "keep"
    if im == "supported" and il == "mention_only":
        return "review"
    if im == "absent" or il == "none":
        return "exclude"
    if im == "ambiguous" or il == "uncertain":
        return "review"
    return "review"
