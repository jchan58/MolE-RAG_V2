"""
Bedrock Qwen3-32B backend (Converse API, Bearer-token auth) — drop-in replacement for
bedrock_llm with the SAME interface (converse / throttle_stats / endpoint_stats / MODEL /
MODEL_WEIGHTS / REGIONS). Dense 32B, serverless on Bedrock, far cheaper than Claude.

Model id is the BARE id `qwen.qwen3-32b-v1:0` (no us.* inference-profile prefix). Available
in us-east-1, us-west-2, us-east-2 (NOT us-west-1 -> HTTP 400, excluded). Each region is an
independent throttle bucket; weighted round-robin + per-endpoint adaptive backoff.

Thinking suppression: Qwen3 defaults to a <think> chain which would bloat/ truncate the
decision-only JSON. We append the Qwen3 soft switch "/no_think" to the system prompt AND
request thinking disabled via additionalModelRequestFields; the judge parser also strips any
residual <think>...</think>.

Token: AWS_BEARER_TOKEN_BEDROCK from ~/.config/molrag/bedrock.env (read ONCE, cached, never
logged). temperature=0, structured JSON expected in the single text block.
"""
import os, sys, json, time, random, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import requests

BACKEND_NAME = "bedrock_qwen"
REGIONS = ["us-east-1", "us-west-2", "us-east-2"]   # us-west-1 rejects qwen (HTTP 400)
MODEL = "qwen.qwen3-32b-v1:0"
MODEL_WEIGHTS = {MODEL: 1}                           # single model; 3 regions = 3 endpoints
_ENDPOINTS = [(m, r) for m in MODEL_WEIGHTS for r in REGIONS]
_WPOOL = [(m, r) for (m, r) in _ENDPOINTS for _ in range(MODEL_WEIGHTS[m])]
BEDROCK_ENV = os.path.expanduser("~/.config/molrag/bedrock.env")
MAX_RETRIES = 8
BACKOFF_BASE = 0.5
BACKOFF_MAX = 20.0
NO_THINK = os.environ.get("EXP70_QWEN_NOTHINK", "1") == "1"

_SESSION = requests.Session()
_SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_connections=96, pool_maxsize=512))
_rr_lock = threading.Lock()
_cooldown = {ep: 0.0 for ep in _ENDPOINTS}
_ep_stats = {ep: {"calls": 0, "throttle": 0} for ep in _ENDPOINTS}
_throttle_events = {"count": 0}
_tlock = threading.Lock()

_TOKEN_CACHE = {"v": None}
_token_lock = threading.Lock()


def _token():
    v = _TOKEN_CACHE["v"]
    if v is not None:
        return v
    with _token_lock:
        if _TOKEN_CACHE["v"] is None:
            t = C._load_env_var(BEDROCK_ENV, "AWS_BEARER_TOKEN_BEDROCK")
            if not t:
                raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK not found in %s" % BEDROCK_ENV)
            _TOKEN_CACHE["v"] = t
    return _TOKEN_CACHE["v"]


def _pick_endpoint():
    now = time.time()
    with _rr_lock:
        avail = [ep for ep in _WPOOL if _cooldown[ep] <= now]
        if avail:
            return random.choice(avail)
        return min(_ENDPOINTS, key=lambda k: _cooldown[k])


def throttle_stats():
    with _tlock:
        return dict(_throttle_events)


def endpoint_stats():
    with _tlock:
        return {"qwen3-32b@%s" % r: dict(v) for (m, r), v in _ep_stats.items()}


def converse(system, user_text, max_tokens=5000, max_retries=MAX_RETRIES):
    """Returns (ok, content_text, usage, meta={"region","model"} | {"error"}). Weighted
    multi-region dispatch with per-endpoint backoff. Qwen thinking suppressed."""
    sys_text = system + (" /no_think" if NO_THINK else "")
    payload = {
        "system": [{"text": sys_text}],
        "messages": [{"role": "user", "content": [{"text": user_text}]}],
        "inferenceConfig": {"temperature": 0, "maxTokens": max_tokens},
    }
    body = json.dumps(payload).encode()
    for attempt in range(max_retries + 1):
        model, region = _pick_endpoint()
        url = "https://bedrock-runtime.%s.amazonaws.com/model/%s/converse" % (region, model)
        try:
            resp = _SESSION.post(url, data=body, timeout=180,
                                 headers={"Authorization": "Bearer " + _token(),
                                          "Content-Type": "application/json"})
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                _backoff(attempt); continue
            return (False, None, None, {"error": "network: %r" % e})
        sc = resp.status_code
        with _tlock:
            _ep_stats[(model, region)]["calls"] += 1
        if sc == 200:
            j = resp.json()
            try:
                txt = j["output"]["message"]["content"][0]["text"]
            except (KeyError, IndexError):
                txt = None
            return (True, txt, j.get("usage"), {"region": region, "model": model})
        if sc in (429, 503) or "ThrottlingException" in (resp.text or "") or "TooManyRequests" in (resp.text or ""):
            with _tlock:
                _throttle_events["count"] += 1
                _ep_stats[(model, region)]["throttle"] += 1
            _cooldown[(model, region)] = time.time() + min(BACKOFF_MAX, BACKOFF_BASE * (2 ** attempt))
            if attempt < max_retries:
                _backoff(attempt, resp.headers.get("Retry-After")); continue
            return (False, None, None, {"error": "throttled %s" % sc, "model": model})
        if sc in (500, 502, 504):
            if attempt < max_retries:
                _backoff(attempt); continue
        return (False, None, None, {"error": "http %s: %s" % (sc, (resp.text or "")[:200]), "model": model})
    return (False, None, None, {"error": "exhausted"})


def _backoff(attempt, retry_after=None):
    if retry_after:
        try:
            d = float(retry_after)
        except ValueError:
            d = BACKOFF_BASE * (2 ** attempt)
    else:
        d = BACKOFF_BASE * (2 ** attempt)
    time.sleep(min(d, BACKOFF_MAX) + random.uniform(0, 0.5))
