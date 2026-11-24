"""Provider-agnostic model client.

The framework talks to models through the OpenAI Chat Completions / Embeddings
wire format, so you can point it at ANY provider that speaks that API — OpenAI,
a local vLLM / TGI server, an inference gateway, a cloud provider's
OpenAI-compatible endpoint, etc. You choose the models.

To reproduce the paper's numbers you must serve the SAME models we used
(see models.json): Qwen3-32B, Ministral-3-14B, GPT-5-mini, Claude Opus 4.6, and
the same embedding model for evidence retrieval. Any endpoint that serves those
model ids will do.

Configuration lives in `models.json` at the repo root: per logical model name,
a `base_url`, the provider's `model` id, and the name of the env var holding the
API key (`api_key_env`). Keys go in `.env`.
"""
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path


def load_models(repo):
    return json.load(open(Path(repo) / "models.json"))


def _key(cfg):
    return os.environ.get(cfg.get("api_key_env", "MOLERAG_API_KEY"), "").strip()


def _post(url, body, key, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def chat(prompt, cfg, max_tokens=1600):
    """One user-turn chat completion. Returns (text, status)."""
    if not cfg.get("base_url"):
        raise RuntimeError(f"models.json: no base_url set for model '{cfg.get('model')}'")
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    body = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}]}
    body[cfg.get("token_param", "max_tokens")] = max_tokens
    if cfg.get("temperature") is not None:   # some reasoning models reject temperature
        body["temperature"] = cfg["temperature"]
    key = _key(cfg)
    for a in range(8):
        try:
            r = _post(url, body, key, 180)
            return r["choices"][0]["message"]["content"], "ok"
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(min(4 + 4 * a, 45)); continue
            return f"[HTTP{e.code}]", "http"
        except Exception:
            time.sleep(min(3 + 3 * a, 30))
    return "[ERR]", "err"


def embed(texts, cfg):
    """Embed one string or a list. Returns a list of vectors, or None on failure."""
    if not cfg.get("base_url"):
        raise RuntimeError("models.json: no base_url set for 'embedder'")
    url = cfg["base_url"].rstrip("/") + "/embeddings"
    one = not isinstance(texts, list)
    body = {"model": cfg["model"], "input": [texts] if one else texts}
    if cfg.get("extra"):
        body.update(cfg["extra"])
    key = _key(cfg)
    for a in range(6):
        try:
            r = _post(url, body, key, 90)
            return [d["embedding"] for d in r["data"]]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(4 + 4 * a); continue
            return None
        except Exception:
            time.sleep(3 + 3 * a)
    return None
