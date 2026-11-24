"""SageMaker entry point: BioT5 (biot5-base, fine-tuned) on MoleculeNet, MolE-RAG seed-0 splits.

Faithful reproduction of BioT5's MoleculeNet molnet pipeline (Pei et al., NeurIPS 2023):
  * Each (sub)task is an independent fine-tune of the pretrained `QizhiPei/biot5-base` T5 model,
    exactly as the released repo does (one split dir per endpoint under data/splits/molnet/*).
  * Input = molecule SELFIES wrapped `Molecule: <bom>...<eom>`, built with the EXACT BioT5
    Natural-Instructions template (DataCollatorForNI, add_task_definition=True, 0 in-context
    examples) and BioT5's own per-dataset task Definition strings (pulled verbatim from the
    QizhiPei/BioT5_finetune_dataset molnet task JSONs).  Only the molecules/labels are swapped
    for the MolE-RAG seed-0 splits.
  * Model/tokenizer/data-collator/optimizer(AdamW)/cosine-schedule/loss are BioT5's own
    `biot5/utils` functions, imported unchanged.
  * Classification ROC-AUC is BioT5's exact scoring: prob = logit("Yes")/(logit("Yes")+logit("No"))
    from the first generated-token scores (vocab ids 2163="Yes", 465="No"), then sklearn roc_auc.
  * Multitask datasets (clintox/tox21/sider): one independent fine-tune per endpoint (BioT5 native);
    missing labels are simply absent from that endpoint's task; dataset score = mean ROC-AUC over
    endpoints whose test split has both classes.
  * Regression (esol/freesolv/lipo) is NOT part of BioT5's molnet pipeline (it only emits Yes/No).
    We add a minimal, clearly-labelled regression variant: the model generates the (train-
    standardized) numeric value as text, we parse it and de-standardize, metric = RMSE.

We run to a per-dataset optimizer-step budget with eval every N steps and select the checkpoint
with best validation metric (best-val -> corresponding test), which is the standard MoleculeNet
protocol; BioT5's shipped molnet.yaml total_steps=50000 is a generic cap, far more than needed for
these small splits, so we cap steps and early-stop on validation to stay in budget while keeping
every numerical component (optimizer, lr, schedule, data template, scoring) identical.

Env: GROUPS (comma list of datasets), SEEDS (default "0"), SMOKE ("1" = tiny),
     plus optional per-run overrides. Data channel at SM_CHANNEL_TRAIN, biot5-base via the
     `ckpt` channel (SM_CHANNEL_CKPT) if present else downloaded from HF.
Results stream to SM_MODEL_DIR/results.jsonl and to s3://<bucket>/biot5_baseline/output/incremental/.
"""
import os, sys, json, time, re, math, traceback, random, types

HERE = os.path.dirname(os.path.abspath(__file__))     # .../code/BioT5/biot5
REPO = os.path.dirname(HERE)                           # .../code/BioT5
sys.path.insert(0, HERE)                               # so `import utils` resolves to biot5/utils
# BioT5's logging_utils does `import neptune` at top (only used when logging.neptune=True, which we
# keep False) -> shim it so we don't need the heavy neptune package in the container. The shim must
# carry a non-None __spec__ or transformers' `importlib.util.find_spec("neptune")` raises
# "neptune.__spec__ is None" (it then finds no installed version and treats neptune as unavailable).
import importlib.machinery as _ilm
if "neptune" not in sys.modules:
    _nep = types.ModuleType("neptune")
    _nep.__spec__ = _ilm.ModuleSpec("neptune", loader=None)
    sys.modules["neptune"] = _nep

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from omegaconf import OmegaConf, open_dict

DATA = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
CKPT_CH = os.environ.get("SM_CHANNEL_CKPT", "/opt/ml/input/data/ckpt")
OUTD = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
GROUPS = [g for g in os.environ.get("GROUPS", "bbbp").split(",") if g]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
SMOKE = os.environ.get("SMOKE", "0") == "1"
# optional "start:end" python slice applied to each dataset's target list, so a large multitask
# dataset (e.g. sider's 27 endpoints) can be split across several SageMaker jobs/instance types.
SLICE = os.environ.get("SLICE", "")
S3_BUCKET = os.environ.get("S3_BUCKET", "molerag")
S3_INCR = os.environ.get("S3_INCR", "biot5_baseline/output/incremental")
MODEL_NAME = os.environ.get("BASE_MODEL", "google/t5-v1_1-base")
BIOT5_REPO = os.environ.get("BIOT5_REPO", "QizhiPei/biot5-base")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
META = json.load(open(os.path.join(DATA, "meta.json")))
DEFS_CLASS = json.load(open(os.path.join(DATA, "definitions_class.json")))
NI_META = json.load(open(os.path.join(DATA, "ni_meta_template.json")))

# minimal, clearly-labelled regression definitions (BioT5 is Yes/No-only natively)
DEFS_REG = {
 "esol": "Molecule property prediction task (a regression task) for the ESOL dataset. The ESOL "
         "dataset is a collection of water solubility data (log solubility in mols per litre) for "
         "common organic small molecules. Your job is to predict the log solubility value for the "
         "given molecule.",
 "freesolv": "Molecule property prediction task (a regression task) for the FreeSolv dataset. The "
         "FreeSolv dataset provides experimental hydration free energies of small molecules in "
         "water. Your job is to predict the hydration free energy value for the given molecule.",
 "lipo": "Molecule property prediction task (a regression task) for the Lipophilicity dataset. The "
         "Lipophilicity dataset provides experimental octanol/water distribution coefficients "
         "(logD at pH 7.4). Your job is to predict the lipophilicity value for the given molecule.",
}

# per-dataset optimizer-step budget (steps, eval_every, warmup, patience-in-evals)
BUDGET = {
 "bbbp":     (800, 40, 40, 8),
 "bace":     (800, 40, 40, 8),
 "clintox":  (800, 40, 40, 8),
 "sider":    (600, 40, 30, 7),
 "tox21":    (1200, 60, 60, 8),
 "hiv":      (2000, 100, 100, 8),
 "esol":     (1200, 60, 60, 8),
 "freesolv": (1000, 50, 50, 8),
 "lipo":     (1500, 75, 75, 8),
}
if SMOKE:
    BUDGET = {k: (20, 10, 4, 3) for k in BUDGET}

YES_ID, NO_ID = 2163, 465  # BioT5's hardcoded "Yes"/"No" first-token ids in t5-v1_1-base vocab


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def s3_put(key, obj):
    try:
        import boto3
        boto3.client("s3", region_name="us-east-1").put_object(
            Bucket=S3_BUCKET, Key=key, Body=json.dumps(obj).encode())
    except Exception as e:
        log("WARN s3_put failed", key, e)


# ---------------------------------------------------------------------------
# SELFIES encoding (BioT5 uses the `selfies` lib; molecules that fail are skipped & counted)
# ---------------------------------------------------------------------------
def build_selfies_encoder():
    import selfies as sf
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    def encode(smiles):
        try:
            m = Chem.MolFromSmiles(smiles)
            if m is None:
                return None
            can = Chem.MolToSmiles(m)
            s = sf.encoder(can)
            if not s:
                return None
            return s
        except Exception:
            return None
    return encode, sf.__version__


# ---------------------------------------------------------------------------
# write one endpoint's NI-format task JSONs + split txts, swapping in OUR molecules
# ---------------------------------------------------------------------------
def read_csv(path):
    import csv
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r]
    return header, rows


def make_ni_endpoint(work, ds, target, definition, encode, task_type, stats):
    """Create data/splits/molnet/<tag>/ and data/tasks/<tag>_{train,valid,test}.json.
    Returns (data_dir, info) or None if empty. `stats` collects selfies counts."""
    tag = re.sub(r"[^a-z0-9]+", "_", f"{ds}_{target}".lower()).strip("_")
    tasks_dir = os.path.join(work, "data", "tasks")
    split_dir = os.path.join(work, "data", "splits", "molnet", tag)
    os.makedirs(tasks_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    # regression: standardize targets on TRAIN
    mean = std = None
    if task_type == "regression":
        header, rows = read_csv(os.path.join(DATA, ds, "train.csv"))
        ci = header.index(target)
        ys = [float(r[ci]) for r in rows if r[ci] != ""]
        mean = float(np.mean(ys)); std = float(np.std(ys)) or 1.0

    split_files = {"train": "train.csv", "valid": "valid.csv", "test": "test.csv"}
    counts = {}
    for split, fn in split_files.items():
        header, rows = read_csv(os.path.join(DATA, ds, fn))
        si = header.index("smiles")
        ci = header.index(target)
        instances = []
        n_skip_sf = 0
        n_pos = n_neg = 0
        for k, row in enumerate(rows):
            lab = row[ci]
            if lab == "" or lab is None:   # missing label -> absent from this endpoint
                continue
            selfies = encode(row[si])
            if selfies is None:
                n_skip_sf += 1
                continue
            inp = f"Molecule: <bom>{selfies}<eom>"
            if task_type == "classification":
                y = int(float(lab))
                out = ["Yes."] if y == 1 else ["No."]
                if y == 1: n_pos += 1
                else: n_neg += 1
            else:
                z = (float(lab) - mean) / std
                out = [f"{z:.4f}"]
            instances.append({"id": f"{tag}-{split}-{k}", "input": inp, "output": out})
        counts[split] = {"n": len(instances), "skipped_selfies": n_skip_sf,
                         "n_pos": n_pos, "n_neg": n_neg}
        task_name = f"molerag_{tag}_{split}"
        taskobj = dict(NI_META)
        taskobj["Definition"] = [definition]
        taskobj["Instances"] = instances
        json.dump(taskobj, open(os.path.join(tasks_dir, task_name + ".json"), "w"))
        splitfile = {"train": "train_tasks.txt", "valid": "validation_tasks.txt",
                     "test": "test_tasks.txt"}[split]
        open(os.path.join(split_dir, splitfile), "w").write(task_name + "\n")

    stats[tag] = counts
    info = {"tag": tag, "counts": counts, "mean": mean, "std": std}
    if counts["train"]["n"] == 0 or counts["test"]["n"] == 0:
        return None
    return split_dir, info


# ---------------------------------------------------------------------------
# build the OmegaConf args BioT5's utils expect
# ---------------------------------------------------------------------------
def build_args(split_dir, task_type, seed, steps, warmup, ckpt_path):
    max_target_len = 8 if task_type == "classification" else 16
    cfg = {
        "mode": "ft",
        "device": "gpu",
        "eval_only": False, "predict_only": False,
        "seed": seed,
        "model": {"name": MODEL_NAME, "checkpoint_path": ckpt_path, "dropout": 0.0,
                  "random_init": False, "compile": False},
        "data": {"input_length": 512, "max_seq_len": 512, "max_target_len": max_target_len,
                 "max_num_instances_per_task": 10_000_000,
                 "add_task_name": False, "add_task_definition": True,
                 "num_pos_examples": 0, "num_neg_examples": 0,
                 "add_explanation": False, "tk_instruct": False,
                 "exec_file_path": os.path.join(HERE, "utils", "ni_dataset.py"),
                 "data_dir": split_dir, "task_dir": os.path.join(os.path.dirname(
                     os.path.dirname(os.path.dirname(split_dir))), "tasks"),
                 "num_workers": 2},
        "optim": {"name": "adamw", "base_lr": 1e-3, "batch_size": 128, "epochs": -1,
                  "total_steps": steps, "warmup_steps": warmup, "lr_scheduler": "cosine",
                  "weight_decay": 0.0, "grad_clip": 0.0, "grad_acc": 8,
                  "test_bsz_multi": 1, "final_cosine": 1e-5},
        "eval": {"every_steps": 1000, "steps": 100000},
        "pred": {"every_steps": 1000},
        "checkpoint": {"every_steps": 10 ** 9},
        "logging": {"neptune": False, "every_steps": 100, "grad_l2": False, "weights_l2": False},
        "molecule_dict": "dict/selfies_dict.txt",
    }
    return OmegaConf.create(cfg)


class _L:  # logger stub for get_lr_scheduler / get_model
    def log_message(self, m): log(m)


def decode_labels(tokenizer, labels):
    labels = labels.clone()
    labels[labels == -100] = tokenizer.pad_token_id
    txt = tokenizer.batch_decode(labels, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)
    return [t.strip() for t in txt]


@torch.no_grad()
def evaluate(model, dl, tokenizer, task_type, max_target_len, std=None):
    model.eval()
    ys, ps = [], []
    tot_loss, nb = 0.0, 0
    for batch in dl:
        batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["attention_mask"] = (batch["input_ids"] != 0)
        # loss (for selection fallback)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        tot_loss += float(out.loss.item()); nb += 1
        gen = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             max_length=max_target_len, generation_config=model.generation_config,
                             return_dict_in_generate=True, output_scores=True)
        scores = gen.scores
        refs = decode_labels(tokenizer, batch["labels"])
        n = batch["input_ids"].shape[0]
        if task_type == "classification":
            for i in range(n):
                p = (scores[0][i][YES_ID] / (scores[0][i][YES_ID] + scores[0][i][NO_ID])).item()
                ps.append(p)
                ys.append(1 if refs[i] == "Yes." else 0)
        else:
            preds = decode_labels(tokenizer, gen.sequences)
            for i in range(n):
                zt = _parse_float(refs[i])
                zp = _parse_float(preds[i])
                if zp is None:
                    zp = 0.0  # fall back to train mean (standardized 0)
                ys.append(zt * std); ps.append(zp * std)
    val_loss = tot_loss / max(nb, 1)
    if task_type == "classification":
        if len(set(ys)) < 2:
            return {"auroc": None, "loss": val_loss, "n": len(ys)}
        return {"auroc": float(roc_auc_score(ys, ps)), "loss": val_loss, "n": len(ys)}
    else:
        rmse = float(np.sqrt(np.mean((np.array(ps) - np.array(ys)) ** 2)))
        return {"rmse": rmse, "loss": val_loss, "n": len(ys)}


def _parse_float(s):
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    try:
        return float(m.group()) if m else None
    except Exception:
        return None


def finetune_endpoint(args, task_type, steps, eval_every, patience, std):
    from accelerate.utils import set_seed
    from utils import (get_tokenizer, get_model, get_config, get_optimizer,
                       get_lr_scheduler, get_dataloaders)
    set_seed(int(args.seed))
    logger = _L()
    tokenizer = get_tokenizer(args)
    config = get_config(args)
    model = get_model(args, config, tokenizer, logger).to(DEVICE)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, logger)
    train_dl, val_dl, test_dl = get_dataloaders(tokenizer, config, args)
    mtl = args.data.max_target_len

    metric_key = "auroc" if task_type == "classification" else "rmse"
    better = (lambda a, b: a > b) if task_type == "classification" else (lambda a, b: a < b)
    best_sel = None       # selection score (val metric, or -val_loss fallback)
    best_val = None
    best_test = None
    best_step = 0
    no_improve = 0
    grad_acc = int(args.optim.grad_acc)
    step = 0
    model.train()
    stop = False
    t0 = time.time()
    while step < steps and not stop:
        optimizer.zero_grad(set_to_none=True)
        for bid, batch in enumerate(train_dl, start=1):
            batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch["attention_mask"] = (batch["input_ids"] != 0)
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
            (out.loss / grad_acc).backward()
            if bid % grad_acc == 0:
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % eval_every == 0 or step >= steps:
                    v = evaluate(model, val_dl, tokenizer, task_type, mtl, std)
                    t = evaluate(model, test_dl, tokenizer, task_type, mtl, std)
                    sel = v[metric_key] if v[metric_key] is not None else -v["loss"]
                    log(f"  step {step}/{steps} val={v} test={t} sel={sel:.4f}")
                    if best_sel is None or better(sel, best_sel) or (
                            v[metric_key] is None and sel > best_sel):
                        best_sel = sel; best_val = v; best_test = t; best_step = step
                        no_improve = 0
                    else:
                        no_improve += 1
                    model.train()
                    if no_improve >= patience:
                        stop = True
                        break
                if step >= steps:
                    stop = True
                    break
    dur = time.time() - t0
    return {"best_step": best_step, "val": best_val, "test": best_test,
            "metric": metric_key, "train_seconds": round(dur, 1)}


def resolve_ckpt():
    # prefer staged ckpt channel; else download from HF
    for cand in [os.path.join(CKPT_CH, "pytorch_model.bin")]:
        if os.path.exists(cand):
            log("using staged ckpt", cand)
            return cand
    from huggingface_hub import hf_hub_download
    log("downloading biot5-base from HF...")
    p = hf_hub_download(repo_id=BIOT5_REPO, filename="pytorch_model.bin",
                        cache_dir=os.environ.get("HF_HOME", "/tmp/hf"))
    log("downloaded", p)
    return p


def main():
    os.makedirs(OUTD, exist_ok=True)
    encode, sf_ver = build_selfies_encoder()
    log("selfies version", sf_ver, "| device", DEVICE, "| groups", GROUPS, "| seeds", SEEDS,
        "| SMOKE", SMOKE)
    ckpt = resolve_ckpt()
    work = "/tmp/biot5_work"
    os.makedirs(work, exist_ok=True)

    # sanity-check BioT5's hardcoded Yes/No first-token ids against the reconstructed tokenizer
    try:
        from utils import get_tokenizer
        tok = get_tokenizer(OmegaConf.create(
            {"model": {"name": MODEL_NAME}, "molecule_dict": "dict/selfies_dict.txt"}))
        yid = tok("Yes", add_special_tokens=False).input_ids[0]
        nid = tok("No", add_special_tokens=False).input_ids[0]
        log(f"token-id check: len(tokenizer)={len(tok)} 'Yes'->{yid} (hard {YES_ID}) "
            f"'No'->{nid} (hard {NO_ID})")
        if (yid, nid) != (YES_ID, NO_ID):
            log("WARNING: tokenizer Yes/No ids differ from BioT5 constants; using actual ids")
            globals()["YES_ID"], globals()["NO_ID"] = yid, nid
    except Exception as e:
        log("token-id check failed (continuing with hardcoded ids):", repr(e))

    results_path = os.path.join(OUTD, "results.jsonl")
    rf = open(results_path, "a")

    def emit(rec):
        rf.write(json.dumps(rec) + "\n"); rf.flush()
        key = f"{S3_INCR}/{rec.get('dataset')}_{rec.get('kind')}_{rec.get('target','ds')}_seed{rec.get('seed')}.json"
        s3_put(key, rec)

    any_fail = False
    for ds in GROUPS:
        if ds not in META:
            log("SKIP unknown dataset", ds); continue
        info = META[ds]
        ttype = info["task_type"]
        targets = info["targets"]
        if SLICE:
            a, b = SLICE.split(":")
            targets = targets[int(a) if a else None:int(b) if b else None]
        if SMOKE and len(targets) > 2:
            targets = targets[:2]
        for seed in SEEDS:
            ep_results = []
            for target in targets:
                definition = (DEFS_CLASS[ds][target] if ttype == "classification"
                              else DEFS_REG[ds])
                log(f"=== {ds} / {target} / seed {seed} ({ttype}) ===")
                stats = {}
                try:
                    made = make_ni_endpoint(work, ds, target, definition, encode, ttype, stats)
                    if made is None:
                        log("  empty endpoint, skipping", ds, target)
                        continue
                    split_dir, epinfo = made
                    steps, eval_every, warmup, patience = BUDGET[ds]
                    args = build_args(split_dir, ttype, seed, steps, warmup, ckpt)
                    r = finetune_endpoint(args, ttype, steps, eval_every, patience,
                                          epinfo.get("std"))
                    rec = {"kind": "endpoint", "dataset": ds, "target": target, "seed": seed,
                           "task_type": ttype, "metric": r["metric"],
                           "test": r["test"], "val": r["val"], "best_step": r["best_step"],
                           "counts": epinfo["counts"], "train_seconds": r["train_seconds"],
                           "budget": {"steps": steps, "eval_every": eval_every,
                                      "warmup": warmup, "patience": patience},
                           "base_lr": 1e-3, "grad_acc": 8, "eff_batch": 128,
                           "base_model": BIOT5_REPO, "selfies_version": sf_ver}
                    emit(rec)
                    ep_results.append(rec)
                except Exception as e:
                    any_fail = True
                    log("  ERROR endpoint", ds, target, repr(e))
                    traceback.print_exc()
                    emit({"kind": "endpoint_error", "dataset": ds, "target": target,
                          "seed": seed, "error": repr(e), "stats": stats})
            # dataset-level aggregate over endpoints
            if ep_results:
                mk = ep_results[0]["metric"]
                if ttype == "classification":
                    vals = [e["test"]["auroc"] for e in ep_results
                            if e["test"] and e["test"].get("auroc") is not None]
                    excluded = [e["target"] for e in ep_results
                                if not e["test"] or e["test"].get("auroc") is None]
                else:
                    vals = [e["test"]["rmse"] for e in ep_results if e["test"]]
                    excluded = []
                if vals:
                    agg = {"kind": "dataset", "dataset": ds, "seed": seed, "task_type": ttype,
                           "metric": mk, "value": float(np.mean(vals)),
                           "n_endpoints": len(ep_results), "n_scored": len(vals),
                           "excluded_endpoints": excluded,
                           "per_endpoint": {e["target"]: (e["test"].get(mk) if e["test"] else None)
                                            for e in ep_results},
                           "base_model": BIOT5_REPO, "selfies_version": sf_ver}
                    emit(agg)
                    log("DATASET", ds, "seed", seed, "->", mk, agg["value"],
                        "over", len(vals), "endpoints")
    rf.close()
    if any_fail and not os.environ.get("ALLOW_PARTIAL"):
        log("one or more endpoints failed")
        sys.exit(1)
    log("DONE")


if __name__ == "__main__":
    main()
