"""SageMaker real-time inference handler for OpenDFM/ChemDFM-v2.0-14B (Qwen2.5-14B-based chat model).

Downloads + loads the HF checkpoint in-container (HF_HOME=/tmp/hf), bf16 (fp16 fallback),
device_map="auto", trust_remote_code=True. Greedy chat generation; returns ONLY the newly
generated text. Request: {"prompt": "...", "max_new_tokens": <int, default 512>}.
Response: {"generated_text": "..."} (application/json)."""
import os, json, traceback
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_NAME", "OpenDFM/ChemDFM-v2.0-14B")


def _eos_ids(tok):
    ids = []
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    for t in ("<|im_end|>", "<|endoftext|>"):
        try:
            i = tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != getattr(tok, "unk_token_id", None):
                ids.append(i)
        except Exception:
            pass
    out = []
    for i in ids:
        if i not in out:
            out.append(i)
    return out or None


def model_fn(model_dir):
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[model_fn] loading {MODEL_ID} dtype={dtype} cuda={torch.cuda.is_available()}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    ).eval()
    eos = _eos_ids(tok)
    print(f"[model_fn] loaded; eos_token_ids={eos} pad={tok.pad_token_id}", flush=True)
    return {"model": model, "tok": tok, "eos": eos}


def input_fn(request_body, request_content_type="application/json"):
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    if request_content_type and "json" in request_content_type:
        d = json.loads(request_body)
    else:
        d = {"prompt": str(request_body)}
    if "prompt" not in d:
        d["prompt"] = ""
    return d


def predict_fn(data, bundle):
    model = bundle["model"]; tok = bundle["tok"]
    prompt = data.get("prompt", "")
    try:
        mnt = int(data.get("max_new_tokens", 512))
    except Exception:
        mnt = 512
    mnt = max(1, min(mnt, 4096))
    try:
        messages = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        dev = next(model.parameters()).device
        input_ids = enc["input_ids"].to(dev)
        attn = enc.get("attention_mask")
        attn = attn.to(dev) if attn is not None else None
        gkw = dict(max_new_tokens=mnt, do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id)
        if bundle.get("eos") is not None:
            gkw["eos_token_id"] = bundle["eos"]
        if attn is not None:
            gkw["attention_mask"] = attn
        with torch.no_grad():
            gen = model.generate(input_ids, **gkw)
        new_tokens = gen[0][input_ids.shape[1]:]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        return {"generated_text": text}
    except Exception as e:
        return {"generated_text": "", "error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[:800]}


def output_fn(prediction, response_content_type="application/json"):
    return json.dumps(prediction), "application/json"
