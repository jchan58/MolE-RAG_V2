"""
Bedrock Claude Opus 4.6 backend (Converse API, Bearer-token auth) with MULTI-REGION
round-robin fan-out and adaptive exponential backoff on throttling (429/503/
ThrottlingException). Each region has its own throttle quota, so round-robin
multiplies effective throughput.

Token: AWS_BEARER_TOKEN_BEDROCK from ~/.config/molrag/bedrock.env (read at call time,
never logged). temperature=0, structured JSON expected in the single text block.
"""
import os, sys, json, time, random, threading, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import requests

# 5 models x 4 US regions = 20 endpoints, each its OWN throttle quota. Weighted toward
# Haiku 4.5 (fastest + headroom) and Opus 4.5 (biggest quota). All are us.* inference
# profiles (bare ids reject on-demand).
REGIONS = ["us-east-1", "us-west-2", "us-east-2", "us-west-1"]
# COST-AWARE weighting: Sonnet 4.5/4.6 ($3/$15 per M) carry the bulk (strong judge, ~40%
# cheaper than Opus); Opus ($5/$25) reserved for a small quality share; Haiku ($0.8/$4) a
# little cheap throughput headroom. judge_model is tagged on every record so the split is
# auditable. Override with EXP70_CLAUDE_MODELS to restrict the pool.
MODEL_WEIGHTS = {
    "us.anthropic.claude-sonnet-4-6": 6,                   # bulk judge ($3/$15)
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": 6,     # bulk judge ($3/$15)
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": 2,      # cheap throughput headroom ($0.8/$4)
    "us.anthropic.claude-opus-4-5-20251101-v1:0": 2,       # quality reserve ($5/$25)
    "us.anthropic.claude-opus-4-6-v1": 1,                  # quality reserve (modest quota)
}
MODEL = "us.anthropic.claude-sonnet-4-6"   # primary (for config display)
CACHE = os.environ.get("EXP70_CACHE", "1") == "1"   # prompt-cache the stable system prefix
# Optional model-subset filter (used when bedrock_llm serves as the cheap CLAUDE FALLBACK
# behind the Qwen judge: e.g. EXP70_CLAUDE_MODELS="us.anthropic.claude-sonnet-4-6,
# us.anthropic.claude-haiku-4-5-20251001-v1:0").
_ALLOW = os.environ.get("EXP70_CLAUDE_MODELS")
if _ALLOW:
    _allow = set(s.strip() for s in _ALLOW.split(",") if s.strip())
    _filtered = {m: w for m, w in MODEL_WEIGHTS.items() if m in _allow}
    if _filtered:
        MODEL_WEIGHTS = _filtered
        MODEL = next(iter(MODEL_WEIGHTS))
_ENDPOINTS = [(m, r) for m in MODEL_WEIGHTS for r in REGIONS]
_WPOOL = [(m, r) for (m, r) in _ENDPOINTS for _ in range(MODEL_WEIGHTS[m])]
BEDROCK_ENV = os.path.expanduser("~/.config/molrag/bedrock.env")
MAX_RETRIES = 8
BACKOFF_BASE = 1.0
BACKOFF_MAX = 30.0

_SESSION = requests.Session()
_SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_connections=128, pool_maxsize=768))
_rr_lock = threading.Lock()
_cooldown = {ep: 0.0 for ep in _ENDPOINTS}          # per-endpoint cooldown-until
_ep_stats = {ep: {"calls": 0, "throttle": 0} for ep in _ENDPOINTS}
_throttle_events = {"count": 0}
_tlock = threading.Lock()


_TOKEN_CACHE = {"v": None}
_token_lock = threading.Lock()
def _token():
    # Cache the token: reading it from home-NFS on EVERY call serialized dispatch
    # (GIL + NFS latency under 1000+ threads) and was the real concurrency ceiling.
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
        return min(_ENDPOINTS, key=lambda k: _cooldown[k])  # all cooling: soonest-free


def throttle_stats():
    with _tlock:
        return dict(_throttle_events)


def endpoint_stats():
    with _tlock:
        return {"%s@%s" % (m.split('.')[-1][:18], r): dict(v) for (m, r), v in _ep_stats.items()}


def converse(system, user_text, max_tokens=8000, max_retries=MAX_RETRIES):
    """Returns (ok, content_text, usage, meta) where meta={"region","model"} on success
    (or {"error":...}). Weighted multi-endpoint dispatch; per-endpoint backoff."""
    # Prompt caching: a cachePoint after the (stable, identical-across-calls) system prompt
    # makes Bedrock charge the instructions+schema as cache_read (~10% price) after the first
    # call within the ~5-min TTL. The volatile passage payload stays in the user turn (last).
    sysblk = [{"text": system}]
    if CACHE:
        sysblk.append({"cachePoint": {"type": "default"}})
    body = json.dumps({
        "system": sysblk,
        "messages": [{"role": "user", "content": [{"text": user_text}]}],
        "inferenceConfig": {"temperature": 0, "maxTokens": max_tokens},
    }).encode()
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
    time.sleep(min(d, BACKOFF_MAX) + random.uniform(0, 0.75))
