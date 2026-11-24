#!/usr/bin/env python3
"""Standalone runner: ChemDFM-v2.0-14B (OpenDFM) ZERO-SHOT inference on the 9 MoleculeNet
MolE-RAG seed-0 scaffold-split TEST sets. Designed to run on a multi-GPU box (e.g. 4x A10G
on SageMaker) with `device_map="auto"` (14B fp16 ~28GB spans ~2 GPUs).

v2 is Qwen2-based (Qwen2ForCausalLM), so we use the model's OWN chat template
(tok.apply_chat_template) instead of ChemDFM-v1's `[Round 0]\\nHuman:...` LLaMA dialogue format.
Everything else matches the v1 ChemDFM baseline exactly (same ChemLLMBench prompts in
chemdfm_templates.json, same scoring):
  classification -> ROC-AUC from P(positive answer token): softmax of the FIRST greedy-step
    logits over {Yes,No} ids (templates phrased so Yes==label 1); macro-AUC over endpoints.
  regression -> greedy generate, parse first float, impute train-mean if unparseable; RMSE.

Local layout (this package): ./data/meta.json, ./data/<task>/{train,test}.csv,
./chemdfm_templates.json. Writes ./results/<task>.json (resume: a task with status==ok is skipped).

Run:  python run_chemdfm_v2.py            # all 9 tasks
      python run_chemdfm_v2.py tox21 bbbp # subset
"""
import os, sys, json, time, csv, math, re
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUTD = os.path.join(HERE, "results"); os.makedirs(OUTD, exist_ok=True)
MODEL_NAME = os.environ.get("MODEL_NAME", "OpenDFM/ChemDFM-v2.0-14B")
MAX_IN = int(os.environ.get("MAX_IN", "1024"))
CLS_NEW_TOKENS = int(os.environ.get("CLS_NEW_TOKENS", "6"))
REG_NEW_TOKENS = int(os.environ.get("REG_NEW_TOKENS", "24"))
META = json.load(open(f"{DATA}/meta.json"))
TEMPLATES = json.load(open(os.path.join(HERE, "chemdfm_templates.json")))
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
TOK = None


def canon(s):
    try:
        c = Chem.CanonSmiles(s, useChiral=True)
        if c:
            return c, True
    except Exception:
        pass
    return s, False


def read_rows(task, split):
    with open(f"{DATA}/{task}/{split}.csv") as fh:
        return list(csv.DictReader(fh))


def build_prompt(tpl_ep, smi):
    return tpl_ep["prefix"] + smi + tpl_ep["suffix"]   # raw question; chat template applied at encode


def _encode(prompt, device):
    text = TOK.apply_chat_template([{"role": "user", "content": prompt.strip()}],
                                   tokenize=False, add_generation_prompt=True)
    full = TOK(text, return_tensors="pt", truncation=False)
    n = full["input_ids"].shape[1]
    tr = n > MAX_IN
    if tr:
        full = {k: v[:, -MAX_IN:] for k, v in full.items()}
    return {k: v.to(device) for k, v in full.items()}, tr


def answer_ids(word):
    ids = set()
    for v in (word, " " + word):
        enc = TOK.encode(v, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return sorted(ids)


@torch.no_grad()
def classify(model, prompts, yes_ids, no_ids, device):
    probs = np.zeros(len(prompts)); hard = np.full(len(prompts), -1, dtype=np.int64); n_trunc = 0
    yes_t = torch.tensor(yes_ids, device=device); no_t = torch.tensor(no_ids, device=device)
    for i, p in enumerate(prompts):
        enc, tr = _encode(p, device); n_trunc += int(tr)
        out = model.generate(**enc, max_new_tokens=CLS_NEW_TOKENS, do_sample=False, num_beams=1,
                             output_scores=True, return_dict_in_generate=True, pad_token_id=TOK.pad_token_id)
        s0 = out.scores[0].float()[0]
        lp = s0[yes_t].max(); ln = s0[no_t].max()
        probs[i] = torch.softmax(torch.stack([lp, ln]), -1)[0].item()
        sl = TOK.decode(out.sequences[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        hard[i] = 1 if sl.startswith("yes") else (0 if sl.startswith("no") else -1)
        del enc, out, s0
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(prompts)}", flush=True)
    return probs, hard, n_trunc


@torch.no_grad()
def regress(model, prompts, device, impute):
    preds = np.zeros(len(prompts)); n_un = 0; n_trunc = 0
    for i, p in enumerate(prompts):
        enc, tr = _encode(p, device); n_trunc += int(tr)
        gen = model.generate(**enc, max_new_tokens=REG_NEW_TOKENS, do_sample=False, num_beams=1, pad_token_id=TOK.pad_token_id)
        s = TOK.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        m = _FLOAT_RE.search(s); val = None
        if m:
            try: val = float(m.group())
            except ValueError: val = None
        if val is None: val = impute; n_un += 1
        preds[i] = val
        del enc, gen
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(prompts)}", flush=True)
    return preds, n_un, n_trunc


def run_classification(task, model, device):
    eps = TEMPLATES[task]["endpoints"]; rows = read_rows(task, "test")
    smis = []; ncf = 0
    for r in rows:
        c, ok = canon(r["smiles"]); smis.append(c); ncf += (not ok)
    prompts, ep_idx, ys = [], [], []; ep_names = list(eps.keys())
    for col in ep_names:
        for i, r in enumerate(rows):
            v = r.get(col, "")
            if v in (None, ""):
                continue
            prompts.append(build_prompt(eps[col], smis[i])); ep_idx.append(col); ys.append(int(float(v)))
    yes_ids = answer_ids(eps[ep_names[0]].get("positive_token", "Yes")); no_ids = answer_ids("No")
    print(f"  {task}: {len(rows)} mols, {len(ep_names)} ep, {len(prompts)} prompts; yes={yes_ids} no={no_ids}", flush=True)
    probs, hard, nt = classify(model, prompts, yes_ids, no_ids, device); ys = np.array(ys)
    per_auc = {}; accs = []
    for col in ep_names:
        idx = [k for k in range(len(ep_idx)) if ep_idx[k] == col]
        if not idx: continue
        y = ys[idx]; p = probs[idx]; h = hard[idx]; valid = h >= 0
        if valid.sum(): accs.append(float((h[valid] == y[valid]).mean()))
        if y.min() != y.max(): per_auc[col] = float(roc_auc_score(y, p))
    return {"task": task, "status": "ok", "task_type": "classification", "metric": "AUC",
            "value": (float(np.mean(list(per_auc.values()))) if per_auc else None),
            "accuracy_mean_over_endpoints": (float(np.mean(accs)) if accs else None),
            "n_molecules": len(rows), "n_endpoints": len(ep_names), "n_endpoints_scored": len(per_auc),
            "per_endpoint_auc": per_auc, "n_prompts": len(prompts), "n_canon_fail": ncf,
            "n_hardlabel_unparsed": int((hard < 0).sum()), "n_prompts_truncated": nt,
            "prompt_provenance": TEMPLATES[task]["provenance"], "model": MODEL_NAME}


def run_regression(task, model, device):
    col = META[task]["targets"][0]; tpl = TEMPLATES[task]["endpoints"][col]
    tr_y = [float(r[col]) for r in read_rows(task, "train") if r.get(col) not in (None, "")]
    impute = float(np.mean(tr_y)); rows = read_rows(task, "test")
    prompts, ys = [], []; ncf = 0
    for r in rows:
        v = r.get(col, "")
        if v in (None, ""): continue
        c, ok = canon(r["smiles"]); ncf += (not ok)
        prompts.append(build_prompt(tpl, c)); ys.append(float(v))
    print(f"  {task}: {len(prompts)} test mols; impute={impute:.4f}", flush=True)
    preds, nun, nt = regress(model, prompts, device, impute); y = np.array(ys)
    return {"task": task, "status": "ok", "task_type": "regression", "metric": "RMSE",
            "value": float(math.sqrt(np.mean((preds - y) ** 2))), "n_molecules": len(rows),
            "n_prompts": len(prompts), "n_unparseable": int(nun), "n_prompts_truncated": nt,
            "impute_train_mean": impute, "n_canon_fail": ncf,
            "prompt_provenance": TEMPLATES[task]["provenance"], "model": MODEL_NAME}


def main():
    global TOK
    sel = [a for a in sys.argv[1:] if not a.startswith("-")]
    tasks = [t for t in META if (not sel or t in sel)]
    print(f"ChemDFM-v2.0-14B zero-shot | tasks={tasks} | model={MODEL_NAME}", flush=True)
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} n_gpu={torch.cuda.device_count()}", flush=True)
    t0 = time.time()
    TOK = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if TOK.pad_token_id is None:
        TOK.pad_token = TOK.eos_token
    TOK.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                                 device_map="auto", trust_remote_code=True).eval()
    print(f"  model loaded in {time.time()-t0:.0f}s", flush=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    summary = {}
    for task in tasks:
        rp = os.path.join(OUTD, f"{task}.json")
        if os.path.exists(rp):
            r = json.load(open(rp))
            if r.get("status") == "ok":
                print(f"[skip] {task} already done ({r['metric']}={r['value']})", flush=True)
                summary[task] = r; continue
        print(f"\n===== {task} ({META[task]['task_type']}) =====", flush=True)
        tt = time.time()
        res = (run_classification if META[task]["task_type"] == "classification" else run_regression)(task, model, device)
        res["seconds"] = round(time.time() - tt, 1)
        json.dump(res, open(rp, "w"), indent=2); summary[task] = res
        print(f"[{task}] {res['metric']}={res['value']} ({res['seconds']}s)", flush=True)
    json.dump(summary, open(os.path.join(OUTD, "_summary.json"), "w"), indent=2)
    print("\n=== ChemDFM-v2.0-14B SUMMARY ===", flush=True)
    for t, r in summary.items():
        print(f"  {t:<9} {r['metric']}={r['value']}", flush=True)
    print("CHEMDFM_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
