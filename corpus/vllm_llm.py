"""
Local vLLM (OpenAI-compatible) backend for the exp70 judge/extraction, exposing the
SAME interface as bedrock_llm (converse / throttle_stats / endpoint_stats / MODEL /
MODEL_WEIGHTS / REGIONS) so judge_opus.py can use it via EXP70_BACKEND=vllm.

Points at a vLLM server serving a local Qwen model with an OpenAI-compatible API
(continuous batching -> throttle-free high concurrency). No auth, no AWS. Config:
  EXP70_VLLM_URL    base url (default http://localhost:8000/v1)
  EXP70_VLLM_MODEL  served model id (default the Qwen3-8B path)
"""
import os, sys, json, time, random, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import requests

BACKEND_NAME = "vllm_local"
URL = os.environ.get("EXP70_VLLM_URL", "http://localhost:8000/v1").rstrip("/")
MODEL = os.environ.get("EXP70_VLLM_MODEL", "Qwen/Qwen3-8B")
# API-compat shims so judge_opus's config logging works unchanged.
MODEL_WEIGHTS = {MODEL: 1}
REGIONS = ["local"]
REASONING_EFFORT = None
# Qwen3 is a hybrid-reasoning model; the judge wants decision-only output, so disable
# thinking via the chat_template kwarg (vLLM passes it through) and strip any <think>.
NO_THINK = os.environ.get("EXP70_VLLM_NOTHINK", "1") == "1"
MAX_RETRIES = int(os.environ.get("EXP70_VLLM_RETRIES", "5"))
TIMEOUT = int(os.environ.get("EXP70_VLLM_TIMEOUT", "300"))

_SESSION = requests.Session()
_SESSION.mount("http://", requests.adapters.HTTPAdapter(pool_connections=256, pool_maxsize=1024))
_lock = threading.Lock()
_stats = {"calls": 0, "throttle": 0, "errors": 0}


def throttle_stats():
    with _lock:
        return {"count": _stats["throttle"]}


def endpoint_stats():
    with _lock:
        return {"%s@local" % os.path.basename(MODEL)[:18]: dict(calls=_stats["calls"],
                                                                errors=_stats["errors"])}


def converse(system, user_text, max_tokens=6000, max_retries=MAX_RETRIES):
    """OpenAI /chat/completions against the local vLLM server. Returns
    (ok, content_text, usage_like, meta) with usage mapped to Bedrock-style keys."""
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user_text}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if NO_THINK:
        # vLLM forwards chat_template_kwargs to the Qwen3 template (enable_thinking=False).
        body["chat_template_kwargs"] = {"enable_thinking": False}
    data = json.dumps(body).encode()
    for attempt in range(max_retries + 1):
        try:
            resp = _SESSION.post(URL + "/chat/completions", data=data, timeout=TIMEOUT,
                                 headers={"Content-Type": "application/json"})
        except requests.exceptions.RequestException as e:
            with _lock:
                _stats["errors"] += 1
            if attempt < max_retries:
                time.sleep(min(20.0, 1.0 * (2 ** attempt)) + random.uniform(0, 0.5)); continue
            return (False, None, None, {"error": "network: %r" % e})
        sc = resp.status_code
        with _lock:
            _stats["calls"] += 1
        if sc == 200:
            try:
                j = resp.json()
                txt = j["choices"][0]["message"]["content"]
                u = j.get("usage", {}) or {}
                usage = {"inputTokens": u.get("prompt_tokens"),
                         "outputTokens": u.get("completion_tokens"),
                         "totalTokens": u.get("total_tokens")}
            except (KeyError, IndexError, ValueError) as e:
                return (False, None, None, {"error": "parse: %r" % e})
            return (True, txt, usage, {"region": "local", "model": MODEL})
        if sc in (429, 503):
            with _lock:
                _stats["throttle"] += 1
            if attempt < max_retries:
                time.sleep(min(20.0, 1.0 * (2 ** attempt)) + random.uniform(0, 0.5)); continue
            return (False, None, None, {"error": "busy %s" % sc})
        if sc in (500, 502, 504) and attempt < max_retries:
            time.sleep(min(20.0, 1.0 * (2 ** attempt))); continue
        return (False, None, None, {"error": "http %s: %s" % (sc, (resp.text or "")[:200])})
    return (False, None, None, {"error": "exhausted"})


def health():
    try:
        r = _SESSION.get(URL + "/models", timeout=10)
        return r.status_code == 200, (r.text or "")[:300]
    except requests.exceptions.RequestException as e:
        return False, repr(e)


if __name__ == "__main__":
    ok, info = health()
    print("vLLM health:", ok, info)
    if ok:
        t0 = time.time()
        ok, txt, usage, meta = converse("Reply with a single JSON object {\"ok\":true} and nothing else.",
                                        "ping", max_tokens=50)
        print("converse:", ok, "%.2fs" % (time.time() - t0), repr(txt)[:200], usage, meta)
