"""
OpenAI gpt-5-nano judge backend (chat-completions). DROP-IN replacement for the
Bedrock backend (bedrock_llm): exposes the same surface the judge driver uses --
  converse(system, user_text, max_tokens) -> (ok, content, usage, meta)
  throttle_stats(), endpoint_stats(), MODEL_WEIGHTS, REGIONS, MODEL
so judge_opus.py only has to swap the import.

gpt-5-nano is a REASONING model: we send max_completion_tokens (NOT max_tokens) and
send NO temperature. It burns a little hidden reasoning budget even on trivial calls,
so max_completion_tokens must leave headroom for reasoning + the minimal JSON map.
reasoning_effort='low' keeps that hidden burn small so minimal-output batches return
promptly and stay far under the 180M TPM ceiling (500 req/s RPM ceiling).

Key: OPENAI_API_KEY read AT CALL TIME from OPENAI_JUDGE_ENV (node-local /tmp), NEVER
logged / echoed / written to reports.
"""
import os, sys, json, time, random, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import requests

MODEL = os.environ.get("EXP70_OPENAI_MODEL", "gpt-5-mini")
REASONING_EFFORT = os.environ.get("EXP70_REASONING_EFFORT", "low")
# Single OpenAI endpoint; kept as a 1-entry "weights"/"regions" map so the driver's
# config-dump + progress code (written for the multi-endpoint Bedrock backend) still works.
MODEL_WEIGHTS = {MODEL: 1}
REGIONS = ["openai"]
OPENAI_JUDGE_ENV = os.environ.get("OPENAI_JUDGE_ENV", "/tmp/exp70_filter/openai_judge.env")
URL = "https://api.openai.com/v1/chat/completions"
MAX_RETRIES = 8
BACKOFF_BASE = 0.5
BACKOFF_MAX = 20.0

_SESSION = requests.Session()
_SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_connections=256, pool_maxsize=1024))
_tlock = threading.Lock()
_throttle_events = {"count": 0}
_stats = {"calls": 0, "throttle": 0, "empty": 0, "ok": 0}


def _key():
    k = C._load_env_var(OPENAI_JUDGE_ENV, "OPENAI_API_KEY")
    if not k:
        raise RuntimeError("OPENAI_API_KEY not found in %s" % OPENAI_JUDGE_ENV)
    return k


def throttle_stats():
    with _tlock:
        return dict(_throttle_events)


def endpoint_stats():
    with _tlock:
        return {"%s@openai" % MODEL: dict(_stats)}


def converse(system, user_text, max_tokens=6000, max_retries=MAX_RETRIES):
    """Returns (ok, content_text, usage, meta). usage uses Bedrock-style keys
    inputTokens/outputTokens so the driver's accounting is unchanged."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user_text}],
        "max_completion_tokens": max_tokens,
        "reasoning_effort": REASONING_EFFORT,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode()
    for attempt in range(max_retries + 1):
        try:
            resp = _SESSION.post(URL, data=body, timeout=180,
                                 headers={"Authorization": "Bearer " + _key(),
                                          "Content-Type": "application/json"})
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                _backoff(attempt); continue
            return (False, None, None, {"error": "network: %r" % e})
        sc = resp.status_code
        with _tlock:
            _stats["calls"] += 1
        if sc == 200:
            j = resp.json()
            try:
                txt = j["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                txt = None
            u = j.get("usage") or {}
            usage = {"inputTokens": u.get("prompt_tokens"), "outputTokens": u.get("completion_tokens")}
            with _tlock:
                if txt:
                    _stats["ok"] += 1
                else:
                    _stats["empty"] += 1
            return (bool(txt), txt, usage, {"region": "openai", "model": MODEL})
        if sc in (429, 503):
            with _tlock:
                _throttle_events["count"] += 1
                _stats["throttle"] += 1
            if attempt < max_retries:
                _backoff(attempt, resp.headers.get("Retry-After")); continue
            return (False, None, None, {"error": "throttled %s" % sc, "model": MODEL})
        if sc in (500, 502, 504):
            if attempt < max_retries:
                _backoff(attempt); continue
        return (False, None, None, {"error": "http %s: %s" % (sc, (resp.text or "")[:200]), "model": MODEL})
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
