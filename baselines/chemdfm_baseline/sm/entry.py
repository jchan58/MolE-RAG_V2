"""SageMaker entry point: ChemDFM-v1.0-13B (released checkpoint) ZERO-SHOT inference on the
MoleculeNet MolE-RAG seed-0 test splits.

ChemDFM (Zhao et al., Cell Rep. Phys. Sci. 2025) is a dialogue foundation model for chemistry built
on LLaMA-1-13B. We load the released merged checkpoint `OpenDFM/ChemDFM-v1.0-13B` (6 safetensors
shards, LlamaForCausalLM, fp16) from the HF hub in-container and run a SINGLE GREEDY pass per
(molecule, endpoint) prompt. No fine-tuning, no seeds.

PROMPTS (chemdfm_templates.json, shipped in source_dir):
  * bbbp, bace, hiv, clintox (2 ep), tox21 (12 ep): the EXACT ChemLLMBench property-prediction
    prompts released in the ChemDFM GitHub repo (verbatim prefix/suffix extracted from their
    jsonl eval files), wrapped in ChemDFM's dialogue format
    `[Round 0]\nHuman: {question}\nAssistant:` (per the repo's eval README).
  * sider (27 ep), esol/freesolv/lipo (regression): ChemDFM released NO prompt for these tasks;
    we use ChemDFM-STYLE wording (same "You are an expert chemist..." instruction shell) in the
    same dialogue format. Clearly flagged in provenance.

SMILES are canonicalized with rdkit (Chem.CanonSmiles, useChiral=True) as the repo README
recommends; on failure the raw SMILES is used (counted).

SCORING:
  * classification -> ROC-AUC from the probability of the positive answer token: one greedy
    forward/generate, softmax of the FIRST generated-step logits over the {Yes, No} answer-token
    ids (P(Yes) == P(positive), since every template is phrased so Yes==label 1). We ALSO parse the
    generated text to a hard Yes/No label and report accuracy (ChemDFM's own string-match scoring).
    Multitask (clintox/tox21/sider): one prompt per present endpoint; AUC per endpoint then averaged
    (missing labels never become rows == masked; endpoints with a single test class are skipped).
  * regression -> greedy generate, parse the first float from the generated text, impute the TRAIN
    task-mean for unparseable outputs (counted), RMSE in original units.

Per-task result JSON is written to SM_MODEL_DIR/runs/<task>.json AND streamed to
s3://<bucket>/chemdfm_baseline/output/incremental/<task>.json so completed tasks survive a failed or
pre-empted job (restart-safe: a task already present as status==ok in S3 is skipped on resume).
"""
import os, sys, json, time, csv, math, re

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from transformers import LlamaTokenizer, LlamaForCausalLM

# ------------------------------------------------------------------ config
DATA = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
OUTD = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
CODE = os.environ.get("SM_MODULE_DIR_LOCAL", os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = os.environ.get("MODEL_NAME", "OpenDFM/ChemDFM-v1.0-13B")
BATCH = int(os.environ.get("BATCH_SIZE", "16"))
MAX_IN = int(os.environ.get("MAX_IN", "768"))
CLS_NEW_TOKENS = int(os.environ.get("CLS_NEW_TOKENS", "6"))
REG_NEW_TOKENS = int(os.environ.get("REG_NEW_TOKENS", "24"))
S3_BUCKET = os.environ.get("S3_BUCKET", "molerag")
S3_INCR = os.environ.get("S3_INCR", "chemdfm_baseline/output/incremental")
RESUME = os.environ.get("RESUME", "1") == "1"

META = json.load(open(f"{DATA}/meta.json"))
TEMPLATES = json.load(open(os.path.join(CODE, "chemdfm_templates.json")))

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def canon(smiles):
    try:
        c = Chem.CanonSmiles(smiles, useChiral=True)
        if c:
            return c, True
    except Exception:
        pass
    return smiles, False


def read_rows(task, split):
    with open(f"{DATA}/{task}/{split}.csv") as fh:
        return list(csv.DictReader(fh))


def wrap(question):
    # ChemDFM dialogue format (repo eval README): text already contains the SMILES + cue.
    return f"[Round 0]\nHuman: {question.strip()}\nAssistant:"


def build_prompt(tpl_ep, smi):
    return wrap(tpl_ep["prefix"] + smi + tpl_ep["suffix"])


# ----------------------------------------------------------------- S3 helpers
def s3_put(key, obj):
    try:
        import boto3
        boto3.client("s3", region_name="us-east-1").put_object(
            Bucket=S3_BUCKET, Key=key, Body=json.dumps(obj, indent=2).encode())
        print(f"  uploaded s3://{S3_BUCKET}/{key}", flush=True)
    except Exception as e:
        print(f"  (s3 upload failed: {str(e)[:160]})", flush=True)


def s3_get(key):
    try:
        import boto3
        o = boto3.client("s3", region_name="us-east-1").get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(o["Body"].read())
    except Exception:
        return None


# ----------------------------------------------------------------- model
def load_model():
    print(f"loading tokenizer+model {MODEL_NAME} ...", flush=True)
    t0 = time.time()
    tok = LlamaTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only generation needs left padding
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                             device_map="auto")
    model.eval()
    print(f"  model loaded in {time.time()-t0:.0f}s; dtype fp16; device_map auto", flush=True)
    return tok, model


def answer_ids(tok, word):
    ids = set()
    for v in (word, " " + word):
        enc = tok.encode(v, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return sorted(ids)


# ----------------------------------------------------------------- scoring
# ONE molecule per forward pass (batch_size=1) with per-step GPU cleanup, so KV-cache /
# allocator fragmentation never accumulates across the ~9k generations of a long shard.
def _encode(tok, prompt, device):
    """Tokenize a single prompt; truncate (from the LEFT, preserving the trailing answer cue)
    only if it exceeds MAX_IN. Returns (enc_on_device, was_truncated)."""
    full = tok(prompt, return_tensors="pt", truncation=False)
    n = full["input_ids"].shape[1]
    if n > MAX_IN:
        full = {k: v[:, -MAX_IN:] for k, v in full.items()}   # keep the last MAX_IN tokens (cue+tail)
        trunc = True
    else:
        trunc = False
    enc = {k: v.to(device) for k, v in full.items()}
    return enc, trunc


@torch.no_grad()
def classify(tok, model, prompts, yes_ids, no_ids, device):
    """Return (P(yes) array, hard-label array {1,0,-1}, n_truncated). batch_size=1."""
    probs = np.zeros(len(prompts), dtype=np.float64)
    hard = np.full(len(prompts), -1, dtype=np.int64)
    yes_t = torch.tensor(yes_ids, device=device)
    no_t = torch.tensor(no_ids, device=device)
    n_trunc = 0
    cuda = device.startswith("cuda")
    for i, prompt in enumerate(prompts):
        enc, tr = _encode(tok, prompt, device)
        n_trunc += int(tr)
        out = model.generate(**enc, max_new_tokens=CLS_NEW_TOKENS, do_sample=False, num_beams=1,
                             output_scores=True, return_dict_in_generate=True,
                             pad_token_id=tok.pad_token_id)
        step0 = out.scores[0].float()[0]                    # [vocab]
        lp = step0[yes_t].max()
        ln = step0[no_t].max()
        probs[i] = torch.softmax(torch.stack([lp, ln]), dim=-1)[0].item()
        gen = out.sequences[0, enc["input_ids"].shape[1]:]
        sl = tok.decode(gen, skip_special_tokens=True).strip().lower()
        if sl.startswith("yes"):
            hard[i] = 1
        elif sl.startswith("no"):
            hard[i] = 0
        del enc, out, step0, gen, lp, ln
        if cuda:
            torch.cuda.empty_cache()
    return probs, hard, n_trunc


@torch.no_grad()
def regress(tok, model, prompts, device, impute):
    """Return (preds, n_unparsed, n_truncated). batch_size=1."""
    preds = np.zeros(len(prompts), dtype=np.float64)
    n_unparsed = 0
    n_trunc = 0
    cuda = device.startswith("cuda")
    for i, prompt in enumerate(prompts):
        enc, tr = _encode(tok, prompt, device)
        n_trunc += int(tr)
        gen = model.generate(**enc, max_new_tokens=REG_NEW_TOKENS, do_sample=False, num_beams=1,
                             pad_token_id=tok.pad_token_id)
        new = gen[0, enc["input_ids"].shape[1]:]
        s = tok.decode(new, skip_special_tokens=True)
        m = _FLOAT_RE.search(s)
        val = None
        if m:
            try:
                val = float(m.group())
            except ValueError:
                val = None
        if val is None:
            val = impute; n_unparsed += 1
        preds[i] = val
        del enc, gen, new
        if cuda:
            torch.cuda.empty_cache()
    return preds, n_unparsed, n_trunc


# ----------------------------------------------------------------- per task
def run_classification(task, tok, model, device):
    meta = META[task]
    eps = TEMPLATES[task]["endpoints"]          # {col: {prefix,suffix,...}}
    rows = read_rows(task, "test")
    # canonicalize once per molecule
    smis = []
    n_canon_fail = 0
    for r in rows:
        c, ok = canon(r["smiles"])
        smis.append(c)
        if not ok:
            n_canon_fail += 1
    # build (prompt, endpoint, y) rows over present labels
    prompts, ep_idx, ys = [], [], []
    ep_names = list(eps.keys())
    for col in ep_names:
        tpl = eps[col]
        for i, r in enumerate(rows):
            v = r.get(col, "")
            if v is None or v == "":
                continue
            prompts.append(build_prompt(tpl, smis[i]))
            ep_idx.append(col)
            ys.append(int(float(v)))
    yes_ids = answer_ids(tok, eps[ep_names[0]].get("positive_token", "Yes"))
    no_ids = answer_ids(tok, "No")
    print(f"  {task}: {len(rows)} molecules, {len(ep_names)} endpoints, {len(prompts)} prompts; "
          f"canon_fail={n_canon_fail}; yes_ids={yes_ids} no_ids={no_ids}", flush=True)
    probs, hard, n_trunc = classify(tok, model, prompts, yes_ids, no_ids, device)
    ys = np.array(ys)
    # per-endpoint AUC + accuracy
    per_auc = {}
    accs = []
    for col in ep_names:
        idx = [k for k in range(len(ep_idx)) if ep_idx[k] == col]
        if not idx:
            continue
        y = ys[idx]
        p = probs[idx]
        h = hard[idx]
        valid = h >= 0
        if valid.sum() > 0:
            accs.append(float((h[valid] == y[valid]).mean()))
        if y.min() != y.max():
            per_auc[col] = float(roc_auc_score(y, p))
    mean_auc = float(np.mean(list(per_auc.values()))) if per_auc else None
    n_unparsed = int((hard < 0).sum())
    return {
        "task": task, "status": "ok", "task_type": "classification", "metric": "AUC",
        "value": mean_auc, "accuracy_mean_over_endpoints": (float(np.mean(accs)) if accs else None),
        "n_molecules": len(rows), "n_endpoints": len(ep_names),
        "n_endpoints_scored": len(per_auc), "per_endpoint_auc": per_auc,
        "n_prompts": len(prompts), "n_canon_fail": n_canon_fail,
        "n_hardlabel_unparsed": n_unparsed, "n_prompts_truncated": n_trunc,
        "scoring": "P(positive answer token) via first greedy-step {Yes,No} softmax; AUC over endpoints",
        "prompt_provenance": TEMPLATES[task]["provenance"], "model": MODEL_NAME,
    }


def run_regression(task, tok, model, device):
    meta = META[task]
    col = meta["targets"][0]
    tpl = TEMPLATES[task]["endpoints"][col]
    tr = read_rows(task, "train")
    tr_y = [float(r[col]) for r in tr if r.get(col) not in (None, "")]
    impute = float(np.mean(tr_y))
    rows = read_rows(task, "test")
    prompts, ys = [], []
    n_canon_fail = 0
    for r in rows:
        v = r.get(col, "")
        if v is None or v == "":
            continue
        c, ok = canon(r["smiles"])
        if not ok:
            n_canon_fail += 1
        prompts.append(build_prompt(tpl, c))
        ys.append(float(v))
    print(f"  {task}: {len(prompts)} test molecules; train-mean impute={impute:.4f}; "
          f"canon_fail={n_canon_fail}", flush=True)
    preds, n_unparsed, n_trunc = regress(tok, model, prompts, device, impute)
    y = np.array(ys)
    rmse = float(math.sqrt(np.mean((preds - y) ** 2)))
    return {
        "task": task, "status": "ok", "task_type": "regression", "metric": "RMSE",
        "value": rmse, "n_molecules": len(rows), "n_prompts": len(prompts),
        "n_unparseable": int(n_unparsed), "n_prompts_truncated": n_trunc,
        "impute_train_mean": impute,
        "n_canon_fail": n_canon_fail,
        "scoring": "greedy generate, parse first float, impute train-mean if unparseable, RMSE",
        "prompt_provenance": TEMPLATES[task]["provenance"], "model": MODEL_NAME,
    }


def main():
    t0 = time.time()
    sel = os.environ.get("TASKS", "").strip()
    tasks = [t for t in META if (not sel or t in [s.strip() for s in sel.split(",")])]
    print("ChemDFM-13B zero-shot | TASKS:", tasks, "| MODEL:", MODEL_NAME,
          "| BATCH:", BATCH, flush=True)
    os.makedirs(f"{OUTD}/runs", exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
          "| n_gpu", torch.cuda.device_count(), flush=True)

    tok, model = load_model()
    allres = []
    failures = 0
    for task in tasks:
        incr_key = f"{S3_INCR}/{task}.json"
        if RESUME:
            prev = s3_get(incr_key)
            if prev and prev.get("status") == "ok" and prev.get("value") is not None:
                print(f"=== {task}: already done (resume), skipping ===", flush=True)
                allres.append(prev)
                json.dump(prev, open(f"{OUTD}/runs/{task}.json", "w"), indent=2)
                continue
        print(f"=== {task} ===", flush=True)
        try:
            if META[task]["task_type"] == "classification":
                r = run_classification(task, tok, model, device)
            else:
                r = run_regression(task, tok, model, device)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = {"task": task, "status": f"exception:{str(e)[:300]}"}
        if r.get("status") != "ok":
            failures += 1
        r["elapsed_min_at_finish"] = round((time.time() - t0) / 60, 1)
        allres.append(r)
        json.dump(r, open(f"{OUTD}/runs/{task}.json", "w"), indent=2)
        s3_put(incr_key, r)
        print(f"  -> {r.get('status')} {r.get('metric')}={r.get('value')} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    json.dump(allres, open(f"{OUTD}/summary.json", "w"), indent=2)
    print("ALL DONE failures=", failures, "| total_min=", round((time.time()-t0)/60, 1), flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
