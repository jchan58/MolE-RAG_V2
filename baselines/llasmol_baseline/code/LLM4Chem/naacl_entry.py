"""SageMaker entry point: LlaSMol (Mistral-7B) RELEASED-CHECKPOINT INFERENCE on the MolE-RAG
NAACL MoleculeNet test sets.

LlaSMol-Mistral-7B = a LoRA adapter (osunlp/LlaSMol-Mistral-7B, modules_to_save=[]) on the openly
available base mistralai/Mistral-7B-v0.1, instruction-tuned on SMolInstruct. This script loads the
released checkpoint (base + adapter, merged) and runs a SINGLE deterministic inference pass (LlaSMol's
own property-prediction generation config: num_beams=4, num_return_sequences=1, max_new_tokens=20;
NO fine-tuning, NO seeds) on OUR test molecules.

Only the 6 MoleculeNet tasks in SMolInstruct's property-prediction subset are supported:
    esol, lipo (regression)   bbbp, clintox, hiv, sider (classification)
FreeSolv, BACE, Tox21 are NOT in SMolInstruct -> reported as unsupported (no fabricated numbers).

Faithfulness:
  * Prompts are LlaSMol's EXACT verbatim SMolInstruct templates (bundled in smol_templates/), with the
    SMILES wrapped in the dataset's core tag (<SMILES> ... </SMILES>). Each test molecule is assigned a
    template by cycling through the task's full verbatim template set (index mod n_templates), mirroring
    SMolInstruct's multi-template test construction.
  * SMILES are canonicalized with LlaSMol's own utils.smiles_canonicalization.canonicalize_molecule_smiles
    (the exact function used to build SMolInstruct), so our molecules match the training distribution.
  * The chat wrapping (<s>[INST] ... [/INST]), generation, and tag-based answer extraction reuse the
    repo's own generation.py / extract_prediction.py / config.TASK_TAGS unchanged.
  * clintox: SMolInstruct asks "Is X toxic?" -> empirically equals the MoleculeNet CT_TOX endpoint
    (verified 16/16 on overlapping molecules); evaluated against CT_TOX only.
  * sider: SMolInstruct asks one organ-system question per (molecule, endpoint); templates cover 20 of
    the 27 MoleculeNet SIDER endpoints. Mean AUC is taken over those 20 supported endpoints (the other
    7 have no LlaSMol template and are masked out and reported).

Scoring (two views):
  * classification ROC-AUC (headline, suite-comparable): probability of the positive ("Yes") answer via
    sequence log-likelihood of the two canonical boolean responses, softmax -> P(Yes). AUC per endpoint,
    mean over endpoints.
  * classification native (LlaSMol's own scoring): calculate_boolean_metrics (ROC-AUC/accuracy/F1 on the
    HARD generated Yes/No label, dropping no-answer/invalid) -- copied verbatim from utils/metrics.py.
  * regression RMSE (headline, suite-comparable): parse the <NUMBER>, impute our TRAIN-set target mean for
    unparseable / no-answer, RMSE over all test molecules. Also reports LlaSMol-native RMSE
    (calculate_number_metrics, which drops unparseable) + the unparseable count.

Per-task raw outputs stream to SM_MODEL_DIR/runs/<task>.jsonl and to S3 incrementally; final per-task
result -> SM_MODEL_DIR/results.jsonl + summary.json. Exits nonzero if any *supported* task fails.
"""
import os, sys, json, time, csv, math

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ---- reuse the LlaSMol repo's own code (source_dir is the repo root) ----
from config import TASK_TAGS
from generation import LlaSMolGeneration, canonicalize_smiles_in_text
from utils.chat_generation import generate_chat
from utils.smiles_canonicalization import canonicalize_molecule_smiles
from extract_prediction import extract_answer_part

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ config
DATA = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
OUTD = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
MODEL_NAME = os.environ.get("MODEL_NAME", "osunlp/LlaSMol-Mistral-7B")
BASE_MODEL = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-v0.1")
BATCH = int(os.environ.get("BATCH_SIZE", "16"))          # LlaSMol PP default
MAX_IN = int(os.environ.get("MAX_IN", "512"))            # LlaSMol DEFAULT_MAX_INPUT_TOKENS
MAX_OUT = int(os.environ.get("MAX_OUT", "20"))           # LlaSMol PP default
NUM_BEAMS = int(os.environ.get("NUM_BEAMS", "4"))        # LlaSMol PP default
SMOKE = os.environ.get("SMOKE", "0") == "1"
SMOKE_N = int(os.environ.get("SMOKE_N", "8"))
S3_BUCKET = os.environ.get("S3_BUCKET", "molerag")
S3_INCR = os.environ.get("S3_INCR", "llasmol_baseline/output/incremental")

TPL_DIR = os.path.join(HERE, "smol_templates")

# supported SMolInstruct property-prediction tasks -> our dataset config
# (our label column(s); for single-endpoint cls the one column; sider handled specially)
SUPPORTED = {
    "esol":    {"type": "regression",     "col": "logSolubility"},
    "lipo":    {"type": "regression",     "col": "lipo"},
    "bbbp":    {"type": "classification", "col": "p_np"},
    "hiv":     {"type": "classification", "col": "HIV_active"},
    "clintox": {"type": "classification", "col": "CT_TOX"},   # "Is X toxic?" == CT_TOX (verified 16/16)
    "sider":   {"type": "classification", "col": None},       # per-endpoint; see SIDER_* below
}
UNSUPPORTED = {
    "freesolv": "not in SMolInstruct (SMolInstruct property-prediction regression = ESOL + Lipophilicity only)",
    "bace":     "not in SMolInstruct property-prediction subset",
    "tox21":    "not in SMolInstruct property-prediction subset",
}
TASK_ORDER = ["bbbp", "bace", "hiv", "clintox", "sider", "tox21", "esol", "freesolv", "lipo"]


def s3_put(key, obj):
    try:
        import boto3
        boto3.client("s3", region_name="us-east-1").put_object(
            Bucket=S3_BUCKET, Key=key, Body=json.dumps(obj, indent=2, default=str).encode())
        print(f"  uploaded s3://{S3_BUCKET}/{key}", flush=True)
    except Exception as e:
        print(f"  (s3 upload failed: {str(e)[:160]})", flush=True)


def s3_get_result(task):
    """Fetch an already-completed per-task result from S3 incremental (for task-level resume)."""
    try:
        import boto3
        o = boto3.client("s3", region_name="us-east-1").get_object(
            Bucket=S3_BUCKET, Key=f"{S3_INCR}/{task}.json")
        return json.loads(o["Body"].read())
    except Exception:
        return None


# ================= LlaSMol scoring, copied VERBATIM from utils/metrics.py =================
def calculate_number_metrics(pred_text_list, gold_text_list):
    assert len(pred_text_list) == len(gold_text_list)
    num_all = len(pred_text_list)
    metrics = {}
    metrics['num_all'] = num_all
    num_no_answer = 0
    num_invalid = 0
    new_pred_text_list, new_gold_text_list = [], []
    for (pred_item, gold_item) in zip(pred_text_list, gold_text_list):
        if pred_item is None:
            num_no_answer += 1
            continue
        assert len(pred_item) == 1
        assert len(gold_item) == 1
        pred_item = pred_item[0]
        gold_item = gold_item[0]
        if pred_item == '':
            num_no_answer += 1
            continue
        try:
            pred_item = float(pred_item)
        except (SyntaxError, ValueError):
            num_invalid += 1
            continue
        gold_item = float(gold_item)
        new_pred_text_list.append(pred_item)
        new_gold_text_list.append(gold_item)
    new_pred_text_list = np.array(new_pred_text_list)
    new_gold_text_list = np.array(new_gold_text_list)
    score = np.sqrt(((new_pred_text_list - new_gold_text_list) ** 2).mean()) if len(new_pred_text_list) else None
    metrics['num_no_answer'] = num_no_answer
    metrics['num_invalid'] = num_invalid
    metrics['RMSE'] = float(score) if score is not None else None
    return metrics


def calculate_boolean_metrics(pred_text_list, gold_text_list):
    assert len(pred_text_list) == len(gold_text_list)
    num_all = len(pred_text_list)
    metrics = {}
    metrics['num_all'] = num_all
    num_no_answer = 0
    num_invalid = 0
    num_correct = 0
    new_pred_text_list, new_gold_text_list = [], []
    for (pred_item, gold_item) in zip(pred_text_list, gold_text_list):
        if pred_item is None or pred_item == '':
            num_no_answer += 1
            continue
        assert len(pred_item) == 1
        assert len(gold_item) == 1
        pred_item = pred_item[0].strip().lower()
        gold_item = gold_item[0].strip().lower()
        if pred_item == '':
            num_no_answer += 1
            continue
        if pred_item not in ('yes', 'no'):
            num_invalid += 1
            continue
        pred_item = 1 if pred_item == 'yes' else 0
        gold_item = 1 if gold_item == 'yes' else 0
        new_pred_text_list.append(pred_item)
        new_gold_text_list.append(gold_item)
        if gold_item == pred_item:
            num_correct += 1
    metrics['num_no_answer'] = num_no_answer
    metrics['num_invalid'] = num_invalid
    metrics['num_correct'] = num_correct
    metrics['num_scored'] = len(new_gold_text_list)
    new_gold_text_list = np.array(new_gold_text_list)
    new_pred_text_list = np.array(new_pred_text_list)
    if len(new_gold_text_list) == 0 or new_gold_text_list.min() == new_gold_text_list.max():
        metrics['roc_auc_score'] = None
    else:
        metrics['roc_auc_score'] = float(roc_auc_score(new_gold_text_list, new_pred_text_list))
    if len(new_gold_text_list):
        metrics['accuracy'] = float(num_correct) / float(len(new_gold_text_list))
        try:
            metrics['f1_score'] = float(f1_score(new_gold_text_list, new_pred_text_list))
            metrics['precision'] = float(precision_score(new_gold_text_list, new_pred_text_list, zero_division=0))
            metrics['recall'] = float(recall_score(new_gold_text_list, new_pred_text_list, zero_division=0))
        except Exception:
            metrics['f1_score'] = metrics['precision'] = metrics['recall'] = None
    return metrics
# =========================================================================================


def load_templates(task):
    tk = "property_prediction-" + task
    tmpl = json.load(open(os.path.join(TPL_DIR, "template", tk + ".json")))
    core = json.load(open(os.path.join(TPL_DIR, "core_tag", tk + ".json")))
    return tmpl, core


def build_query(input_template, core_tag_left, core_tag_right, smiles):
    """Reproduce SMolInstruct's input exactly: template['input'] with <INPUT> replaced by the
    core-tag-wrapped (canonicalized) SMILES."""
    input_data_str = "%s %s %s" % (core_tag_left, smiles, core_tag_right)
    return input_template.replace("<INPUT>", input_data_str)


def read_rows(task):
    with open(os.path.join(DATA, task, "test.csv")) as fh:
        return list(csv.DictReader(fh))


def train_target_mean(task, col):
    ys = []
    with open(os.path.join(DATA, task, "train.csv")) as fh:
        for r in csv.DictReader(fh):
            v = r.get(col, "")
            if v not in ("", None):
                ys.append(float(v))
    return float(np.mean(ys)) if ys else 0.0


# ---- sequence log-likelihood scoring for P(Yes) ----
@torch.no_grad()
def seq_logprob(model, tok, prompt_text, full_text, device):
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    f_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    lcp = 0
    while lcp < len(p_ids) and lcp < len(f_ids) and p_ids[lcp] == f_ids[lcp]:
        lcp += 1
    if lcp >= len(f_ids):
        return None, 0
    ids = torch.tensor([f_ids], device=device)
    logits = model(ids).logits[0]                       # [T, V]
    logp = torch.log_softmax(logits.float(), dim=-1)
    total = 0.0
    ntok = 0
    for pos in range(max(lcp, 1), len(f_ids)):
        total += logp[pos - 1, f_ids[pos]].item()
        ntok += 1
    return total, ntok


def p_yes_for(generator, query, device):
    """P(Yes) via softmax of the sequence log-likelihoods of the two canonical boolean responses.
    Uses the SAME prompt construction as LlaSMolGeneration.create_sample (canonicalize off: SMILES are
    pre-canonicalized when the query is built)."""
    # SMILES is already canonicalized when the query was built, and generation runs with
    # canonicalize_smiles=False, so use the query verbatim -> LL prompt == generation prompt exactly.
    real_text = query
    chat = generate_chat(real_text, output_text=None)
    prompt_text = generator.prompter.generate_prompt(chat)        # '<s>[INST] ... [/INST]'
    # assistant turn formatting is ' %s</s>' (see utils/general_prompter.get_chat_content)
    full_yes = prompt_text + " <BOOLEAN> Yes </BOOLEAN></s>"
    full_no = prompt_text + " <BOOLEAN> No </BOOLEAN></s>"
    ll_yes, ny = seq_logprob(generator.model, generator.tokenizer, prompt_text, full_yes, device)
    ll_no, nn_ = seq_logprob(generator.model, generator.tokenizer, prompt_text, full_no, device)
    if ll_yes is None or ll_no is None:
        return None, (ny, nn_)
    m = max(ll_yes, ll_no)
    py = math.exp(ll_yes - m)
    pn = math.exp(ll_no - m)
    return py / (py + pn), (ny, nn_)


# ---- build eval units (one dict per model query) ----
def build_units(task, cfg):
    tmpl, core = load_templates(task)
    ctl, ctr = core["input"][0], core["input"][1]
    rows = read_rows(task)
    units = []
    n_bad_smiles = 0
    if task == "sider":
        # endpoint (organ system) -> list of templates whose Target matches
        by_target = {}
        for t in tmpl:
            by_target.setdefault(t["Target"], []).append(t)
        # our 27 columns; organ name = col.replace('_',' '); keep only those with a template
        meta = json.load(open(os.path.join(DATA, "meta.json")))["sider"]
        cols = meta["targets"]
        supported_cols = [c for c in cols if c.replace("_", " ") in by_target]
        # per molecule, canonicalize once
        mol_canon = {}
        for r in rows:
            c = canonicalize_molecule_smiles(r["smiles"], return_none_for_error=True)
            mol_canon[id(r)] = c
            if c is None:
                n_bad_smiles += 1
        for ei, col in enumerate(supported_cols):
            organ = col.replace("_", " ")
            group = by_target[organ]
            j = 0
            for r in rows:
                v = r.get(col, "")
                if v in ("", None):
                    continue                          # masked (missing label)
                c = mol_canon[id(r)]
                if c is None:
                    continue
                itpl = group[j % len(group)]["input"]
                j += 1
                units.append({
                    "endpoint": ei, "endpoint_name": col, "smiles_raw": r["smiles"],
                    "query": build_query(itpl, ctl, ctr, c),
                    "gold": "Yes" if int(float(v)) == 1 else "No",
                    "gold_bin": int(float(v)),
                })
        meta_extra = {"supported_endpoints": supported_cols,
                      "n_supported_endpoints": len(supported_cols),
                      "n_total_endpoints": len(cols),
                      "unsupported_endpoints": [c for c in cols if c not in supported_cols]}
        return units, n_bad_smiles, meta_extra

    col = cfg["col"]
    n = len(tmpl)
    for i, r in enumerate(rows):
        v = r.get(col, "")
        if v in ("", None):
            continue
        c = canonicalize_molecule_smiles(r["smiles"], return_none_for_error=True)
        if c is None:
            n_bad_smiles += 1
            # still record a unit with no query -> counts as no-answer
            if cfg["type"] == "classification":
                units.append({"endpoint": 0, "endpoint_name": col, "smiles_raw": r["smiles"],
                              "query": None, "gold": "Yes" if int(float(v)) == 1 else "No",
                              "gold_bin": int(float(v))})
            else:
                units.append({"endpoint": 0, "endpoint_name": col, "smiles_raw": r["smiles"],
                              "query": None, "gold": str(v), "gold_val": float(v)})
            continue
        itpl = tmpl[i % n]["input"]
        if cfg["type"] == "classification":
            units.append({"endpoint": 0, "endpoint_name": col, "smiles_raw": r["smiles"],
                          "query": build_query(itpl, ctl, ctr, c),
                          "gold": "Yes" if int(float(v)) == 1 else "No", "gold_bin": int(float(v))})
        else:
            units.append({"endpoint": 0, "endpoint_name": col, "smiles_raw": r["smiles"],
                          "query": build_query(itpl, ctl, ctr, c),
                          "gold": str(v), "gold_val": float(v)})
    return units, n_bad_smiles, {}


def run_task(generator, task, device):
    cfg = SUPPORTED[task]
    units, n_bad, meta_extra = build_units(task, cfg)
    if SMOKE:
        units = units[:SMOKE_N]
    n_mol = len(read_rows(task))
    print(f"  {task}: {len(units)} eval units ({n_mol} test molecules, {n_bad} unparseable SMILES)", flush=True)

    left_tag, right_tag = TASK_TAGS["property_prediction-" + task]
    is_cls = cfg["type"] == "classification"
    runs_fp = os.path.join(OUTD, "runs", task + ".jsonl")

    # --- within-task resume: reuse any per-unit records already written (restart-safe) ---
    done = {}
    if os.path.isfile(runs_fp):
        for line in open(runs_fp):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                done[d["uid"]] = d
            except Exception:
                pass
        print(f"    {task}: resuming, {len(done)} units already recorded", flush=True)

    fout = open(runs_fp, "a")
    t0 = time.time()
    # ONE molecule at a time: num_beams=4 within a single example is faithful but 24GB cannot hold
    # beam x batch; batch_size=1 + empty_cache() between molecules bounds memory (see coordinator note).
    for i, u in enumerate(units):
        uid = f"{u['endpoint']}:{u['smiles_raw']}"
        if uid in done:
            continue
        if u["query"] is None:
            out = None
        else:
            gen = generator.generate(
                [u["query"]], batch_size=1, max_input_tokens=MAX_IN, max_new_tokens=MAX_OUT,
                canonicalize_smiles=False, print_out=False,
                num_beams=NUM_BEAMS, num_return_sequences=1,
            )
            out = gen[0]["output"]                      # list[str] or None (too long)
        if out is None:
            hard_pred = ""                              # no answer (too long or no query)
        else:
            hard_pred = extract_answer_part(out, left_tag, right_tag, mode="tag")[0]
        rec = {"uid": uid, "endpoint": u["endpoint"], "endpoint_name": u["endpoint_name"],
               "smiles_raw": u["smiles_raw"], "gold": u["gold"],
               "output": out, "hard_pred": hard_pred}
        if is_cls:
            rec["gold_bin"] = u["gold_bin"]
            if u["query"] is not None and out is not None:
                py, (ny, nn_) = p_yes_for(generator, u["query"], device)
                rec["p_yes"] = py
            else:
                rec["p_yes"] = None
        else:
            rec["gold_val"] = u["gold_val"]
        fout.write(json.dumps(rec, default=str) + "\n")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()                    # between molecules
        if (i + 1) % 200 == 0:
            print(f"    {task} {i+1}/{len(units)} ({(time.time()-t0)/60:.1f} min)", flush=True)
            fout.flush()
    fout.close()

    # ---------------- scoring ----------------
    records = [json.loads(l) for l in open(runs_fp)]
    res = {"task": task, "task_type": cfg["type"], "status": "ok",
           "n_test_molecules": n_mol, "n_units": len(records),
           "n_unparseable_smiles": n_bad,
           "generation": {"num_beams": NUM_BEAMS, "num_return_sequences": 1,
                          "max_new_tokens": MAX_OUT, "max_input_tokens": MAX_IN, "do_sample": False},
           "prompt_provenance": "SMolInstruct verbatim templates (bundled), template_id = unit_index mod n_templates"}
    res.update(meta_extra)

    if is_cls:
        # group by endpoint
        eps = sorted(set(r["endpoint"] for r in records))
        prob_aucs, native = [], []
        per_ep = []
        for ep in eps:
            sub = [r for r in records if r["endpoint"] == ep]
            name = sub[0]["endpoint_name"]
            # probability AUC (over units with a p_yes and valid label)
            ys = [r["gold_bin"] for r in sub if r.get("p_yes") is not None]
            ps = [r["p_yes"] for r in sub if r.get("p_yes") is not None]
            pauc = None
            if len(ys) and min(ys) != max(ys):
                pauc = float(roc_auc_score(ys, ps))
            # native hard-label metrics
            pred_list = [[r["hard_pred"]] for r in sub]
            gold_list = [[r["gold"]] for r in sub]
            nat = calculate_boolean_metrics(pred_list, gold_list)
            per_ep.append({"endpoint": name, "prob_auc": pauc, "native": nat,
                           "n_pos": int(sum(ys)) if ys else 0, "n": len(sub)})
            if pauc is not None:
                prob_aucs.append(pauc)
            if nat.get("roc_auc_score") is not None:
                native.append(nat["roc_auc_score"])
        res["metric"] = "ROC-AUC"
        res["value"] = float(np.mean(prob_aucs)) if prob_aucs else None          # headline (prob-based)
        res["prob_auc_mean"] = res["value"]
        res["native_hardlabel_auc_mean"] = float(np.mean(native)) if native else None
        res["n_endpoints_scored"] = len(prob_aucs)
        res["per_endpoint"] = per_ep
        # aggregate native counts
        res["native_total_no_answer"] = sum(e["native"]["num_no_answer"] for e in per_ep)
        res["native_total_invalid"] = sum(e["native"]["num_invalid"] for e in per_ep)
        res["native_total_correct"] = sum(e["native"]["num_correct"] for e in per_ep)
        res["native_total_scored"] = sum(e["native"]["num_scored"] for e in per_ep)
        if res["native_total_scored"]:
            res["native_accuracy"] = res["native_total_correct"] / res["native_total_scored"]
    else:
        pred_list = [([r["hard_pred"]] if r["hard_pred"] != "" else None) for r in records]
        gold_list = [[str(r["gold_val"])] for r in records]
        native = calculate_number_metrics(pred_list, gold_list)
        res["native_rmse"] = native["RMSE"]
        res["native_num_no_answer"] = native["num_no_answer"]
        res["native_num_invalid"] = native["num_invalid"]
        # suite RMSE: impute train mean for no-answer/unparseable, over ALL molecules
        mu = train_target_mean(task, cfg["col"])
        preds, golds = [], []
        for r in records:
            g = float(r["gold_val"])
            p = None
            if r["hard_pred"] != "":
                try:
                    p = float(r["hard_pred"])
                except ValueError:
                    p = None
            preds.append(p if p is not None else mu)
            golds.append(g)
        preds = np.array(preds); golds = np.array(golds)
        res["metric"] = "RMSE"
        res["value"] = float(np.sqrt(((preds - golds) ** 2).mean()))             # headline (imputed)
        res["rmse_imputed_mean"] = res["value"]
        res["imputed_train_mean"] = mu
        res["n_imputed"] = native["num_no_answer"] + native["num_invalid"]
    return res


def main():
    t0 = time.time()
    sel = os.environ.get("TASKS", "").strip()
    req = [s.strip() for s in sel.split(",") if s.strip()] if sel else list(SUPPORTED.keys())
    tasks = [t for t in req if t in SUPPORTED]
    skipped = [t for t in req if t in UNSUPPORTED]
    print("LlaSMol inference | MODEL:", MODEL_NAME, "| BASE:", BASE_MODEL, flush=True)
    print("TASKS:", tasks, "| UNSUPPORTED requested:", skipped, "| SMOKE:", SMOKE, flush=True)
    os.makedirs(os.path.join(OUTD, "runs"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model ...", flush=True)
    generator = LlaSMolGeneration(model_name=MODEL_NAME, base_model=BASE_MODEL, device=device)
    # LlaSMol's model.py sets tokenizer.pad_token='<pad>' (resolves to the unk id for Mistral, which is
    # masked out at generation time). Safety no-op: only if that leaves pad_token_id None, fall back to eos
    # so tokenizer.pad() cannot crash -- never triggers on the normal path, does not change outputs.
    if generator.tokenizer.pad_token_id is None:
        generator.tokenizer.pad_token = generator.tokenizer.eos_token
        generator.model.config.pad_token_id = generator.tokenizer.eos_token_id
        print("  (applied pad_token=eos safeguard)", flush=True)
    print("pad_token_id =", generator.tokenizer.pad_token_id, flush=True)
    print("Model loaded (%.1f min)" % ((time.time() - t0) / 60), flush=True)

    resfp = open(os.path.join(OUTD, "results.jsonl"), "w")
    allres = []
    failures = 0
    resume = os.environ.get("RESUME", "1") == "1" and not SMOKE
    for task in tasks:
        print(f"=== {task} ===", flush=True)
        if resume:
            prev = s3_get_result(task)
            if prev is not None and prev.get("status") == "ok" and prev.get("value") is not None:
                print(f"  (resume: {task} already complete in S3 -> {prev.get('metric')}={prev.get('value')}; skip)",
                      flush=True)
                allres.append(prev)
                resfp.write(json.dumps(prev, default=str) + "\n"); resfp.flush()
                continue
        try:
            r = run_task(generator, task, device)
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {"task": task, "status": f"exception:{str(e)[:300]}", "task_type": SUPPORTED[task]["type"]}
            failures += 1
        allres.append(r)
        resfp.write(json.dumps(r, default=str) + "\n"); resfp.flush()
        s3_put(f"{S3_INCR}/{task}.json", r)
        print(f"  -> {task} {r.get('metric')}={r.get('value')} ({(time.time()-t0)/60:.1f} min)", flush=True)
    resfp.close()

    summary = {"model": "LlaSMol (Mistral-7B)", "supported_tasks": tasks,
               "unsupported_tasks": {t: UNSUPPORTED[t] for t in UNSUPPORTED},
               "results": allres, "smoke": SMOKE}
    json.dump(summary, open(os.path.join(OUTD, "summary.json"), "w"), indent=2, default=str)
    print("ALL DONE failures=", failures, flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
