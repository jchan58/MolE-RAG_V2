"""
Cohere embeddings via Bedrock (invoke-model, Bearer-token auth) — replaces the GPU
MedCPT cross-encoder for semantic ranking. Model cohere.embed-english-v3 (BARE id, NOT
us.* profile), 1024-dim. Multi-region round-robin + backoff. No GPU, no EC2.

API: POST .../model/cohere.embed-english-v3/invoke with body
  {"texts":[...<=96...],"input_type":"search_document"|"search_query","embedding_types":["float"]}
Returns 1024-dim float vectors.
"""
import os, sys, json, time, random, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import requests

MODEL_V3 = "cohere.embed-english-v3"   # 1024-dim (primary)
MODEL_V4 = "cohere.embed-v4:0"         # 1536-dim (fallback; DIFFERENT vector space, NOT mixable with v3)
MODEL_DIM = {MODEL_V3: 1024, MODEL_V4: 1536}
MODEL = MODEL_V3
REGIONS = ["us-east-1", "us-west-2", "us-east-2"]
BEDROCK_ENV = os.path.expanduser("~/.config/molrag/bedrock.env")
MAX_BATCH = 96
MAX_RETRIES = 6
_SESSION = requests.Session()
_SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=128))
_rr = {"i": 0}
_lock = threading.Lock()
_TOK = {"v": None}


def _token():
    if _TOK["v"] is None:
        _TOK["v"] = C._load_env_var(BEDROCK_ENV, "AWS_BEARER_TOKEN_BEDROCK")
        if not _TOK["v"]:
            raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK missing")
    return _TOK["v"]


V4_REGIONS = ["us-east-1"]   # cohere.embed-v4:0 confirmed in us-east-1; pin fallback there


def _region(model=None):
    regs = V4_REGIONS if model == MODEL_V4 else REGIONS
    with _lock:
        r = regs[_rr["i"] % len(regs)]; _rr["i"] += 1
    return r


def embed(texts, input_type, model=None, max_retries=MAX_RETRIES):
    """Embed <=96 texts with the given Cohere model. Returns list of float vectors (dim
    depends on model: v3=1024, v4=1536) or None on failure."""
    model = model or MODEL
    body = json.dumps({"texts": texts, "input_type": input_type,
                       "embedding_types": ["float"]}).encode()
    for attempt in range(max_retries + 1):
        region = _region(model)
        url = "https://bedrock-runtime.%s.amazonaws.com/model/%s/invoke" % (region, model)
        try:
            resp = _SESSION.post(url, data=body, timeout=120,
                                 headers={"Authorization": "Bearer " + _token(),
                                          "Content-Type": "application/json",
                                          "Accept": "application/json"})
        except requests.exceptions.RequestException:
            if attempt < max_retries:
                time.sleep(min(20, 2 ** attempt) + random.uniform(0, 0.5)); continue
            return None
        if resp.status_code == 200:
            j = resp.json()
            emb = j.get("embeddings")
            if isinstance(emb, dict):
                return emb.get("float")
            return emb
        if resp.status_code in (429, 503) and attempt < max_retries:
            time.sleep(min(20, 2 ** attempt) + random.uniform(0, 0.5)); continue
        if resp.status_code in (500, 502, 504) and attempt < max_retries:
            time.sleep(min(20, 2 ** attempt)); continue
        return None
    return None


def embed_many(texts, input_type, model=None, log=None):
    """Embed an arbitrary number of texts in <=96 batches with ONE model. Returns list
    aligned to texts (None for failed batches)."""
    out = [None] * len(texts)
    for i in range(0, len(texts), MAX_BATCH):
        batch = texts[i:i + MAX_BATCH]
        vecs = embed(batch, input_type, model=model)
        if vecs and len(vecs) == len(batch):
            for j, v in enumerate(vecs):
                out[i + j] = v
        if log and (i // MAX_BATCH) % 20 == 0:
            log("embedded %d/%d (%s, %s)" % (i + len(batch), len(texts), input_type, model or MODEL))
    return out


def select_model():
    """Pick the embedding model for a dataset: EXP70_COHERE_MODEL override, else v3 if a
    probe succeeds, else v4 fallback. Returns (model, dim). Use ONE model per dataset/index."""
    ov = os.environ.get("EXP70_COHERE_MODEL")
    if ov in MODEL_DIM:
        return ov, MODEL_DIM[ov]
    v = embed(["probe"], "search_query", model=MODEL_V3)
    if v and len(v[0]) == MODEL_DIM[MODEL_V3]:
        return MODEL_V3, MODEL_DIM[MODEL_V3]
    v = embed(["probe"], "search_query", model=MODEL_V4)
    if v and len(v[0]) == MODEL_DIM[MODEL_V4]:
        return MODEL_V4, MODEL_DIM[MODEL_V4]
    return MODEL_V3, MODEL_DIM[MODEL_V3]   # last resort


if __name__ == "__main__":
    v = embed(["blood brain barrier penetration"], "search_query")
    print("ok" if v and len(v[0]) == 1024 else "FAIL", "dim", (len(v[0]) if v else None))
