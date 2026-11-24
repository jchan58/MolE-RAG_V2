"""SageMaker entry: MolLM (Gerstein Lab, 2D+3D) fine-tuned per MoleculeNet task on MolE-RAG splits.

Faithful to the released repo (downstream/MoleculePrediction): the released MolLM molecular graph
encoder (Transformer-M over OGB 2D features + RDKit ETKDG 3D conformers) initialised from the
released pretrained checkpoint, with a fresh prediction head, trained end-to-end. Unlike the repo's
finetune.py (which selects the epoch by TEST score), we select the best checkpoint by VALIDATION and
report its TEST score -- matching the other MolE-RAG supervised baselines (DMPNN/MolCLR/Graphormer).

Classification: masked BCE over present labels (missing masked from loss AND metric), metric = mean
ROC-AUC over scorable endpoints. Regression: train-set target standardisation, metric = RMSE.
All 9 tasks run sequentially on one GPU; per-task result -> SM_MODEL_DIR + live S3 incremental
(restart-safe: a task whose ok result already exists in S3 is skipped). One seed (0).
"""
import os, sys, json, time, csv, math
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

DATA = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
OUTD = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
MODE = os.environ.get("MODE", "full").lower()
SEED = int(os.environ.get("SEED", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "50"))
BATCH = int(os.environ.get("BATCH_SIZE", "32"))
PATIENCE = int(os.environ.get("PATIENCE", "15"))
MAX_TASK_MIN = float(os.environ.get("MAX_TASK_MIN", "180"))
LR = float(os.environ.get("LR", "1e-4"))
FEAT_PROCS = int(os.environ.get("FEAT_PROCS", str(max(1, (os.cpu_count() or 8) - 2))))
USE_3D = os.environ.get("USE_3D", "1") == "1"
S3_BUCKET = os.environ.get("S3_BUCKET", "molerag")
S3_INCR = os.environ.get("S3_INCR", "mol_llm_baseline/output/incremental")
S3_CKPT_PREFIX = os.environ.get("S3_CKPT_PREFIX", "mol_llm_baseline/ckpt")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def raise_fd_limit():
    """Each multiprocessing.Pool uses ~2 pipe FDs per worker; featurising a large task in many chunks
    can exhaust the default soft limit (1024) -> '[Errno 24] Too many open files'. Raise soft to hard."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard if hard != resource.RLIM_INFINITY else 1048576
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        print(f"  RLIMIT_NOFILE {soft} -> {resource.getrlimit(resource.RLIMIT_NOFILE)[0]}", flush=True)
    except Exception as e:
        print(f"  (raise_fd_limit failed: {str(e)[:80]})", flush=True)


def disable_jit_fuser():
    """torch 1.9.1's TensorExpr/profiling JIT fuser nvrtc-compiles fused CUDA kernels at runtime;
    its bundled nvrtc (cu111) cannot target the L40S (sm_89) -> 'invalid value for --gpu-architecture'.
    Disable GPU fusion so ops run unfused with precompiled kernels (identical numerics).
    Returns a dict of which torch._C JIT APIs exist/applied (for diagnosis)."""
    diag = {}
    for fn, arg in [("_jit_set_profiling_executor", False), ("_jit_set_profiling_mode", False),
                    ("_jit_set_texpr_fuser_enabled", False), ("_jit_set_nvfuser_enabled", False),
                    ("_jit_override_can_fuse_on_gpu", False), ("_jit_override_can_fuse_on_cpu", False),
                    ("_debug_set_fusion_group_inlining", False)]:
        try:
            getattr(torch._C, fn)(arg)
            diag[fn] = "ok"
        except AttributeError:
            diag[fn] = "missing"
        except Exception as e:
            diag[fn] = "err:" + str(e)[:60]
    try:
        torch._C._llvm_enabled()
    except Exception:
        pass
    print("  JIT fuser disable attempt:", diag, flush=True)
    return diag


def s3():
    import boto3
    return boto3.client("s3", region_name="us-east-1")


def s3_put(key, obj):
    try:
        s3().put_object(Bucket=S3_BUCKET, Key=key,
                        Body=json.dumps(obj, indent=2).encode())
        print("  uploaded s3://%s/%s" % (S3_BUCKET, key), flush=True)
    except Exception as e:
        print("  (s3 put failed: %s)" % str(e)[:160], flush=True)


def s3_get(key):
    try:
        o = s3().get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(o["Body"].read())
    except Exception:
        return None


def resolve_ckpt():
    """Download the pretrained graph-encoder checkpoint from S3 (staged by stage job)."""
    local = "/tmp/mollm_ckpt.ckpt"
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local
    summ = s3_get(f"{S3_CKPT_PREFIX}/stage_summary.json")
    cands = []
    if summ and summ.get("ckpts"):
        cands = [c["basename"] for c in summ["ckpts"]]
    # prefer the checkpoint the repo's scripts load, then 507, then any
    order = sorted(cands, key=lambda b: (0 if "394" in b else (1 if "507" in b else 2), b))
    if not order:
        order = ["model-epoch=394.ckpt", "model-epoch=507.ckpt"]
    cli = s3()
    for b in order:
        try:
            print("  downloading ckpt s3://%s/%s/%s" % (S3_BUCKET, S3_CKPT_PREFIX, b), flush=True)
            cli.download_file(S3_BUCKET, f"{S3_CKPT_PREFIX}/{b}", local)
            print("  ckpt ->", local, os.path.getsize(local), "bytes", flush=True)
            return local
        except Exception as e:
            print("  ckpt %s not available: %s" % (b, str(e)[:120]), flush=True)
    raise RuntimeError("no pretrained checkpoint could be downloaded from S3")


# ---------------------------------------------------------------- data
def read_rows(task, split, targets):
    path = f"{DATA}/{task}/{split}.csv"
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            y = []
            for t in targets:
                v = r.get(t, "")
                y.append(float(v) if (v is not None and v != "") else float("nan"))
            rows.append((r["smiles"], y))
    return rows


def _feat_worker(arg):
    """Returns the featurised item as a dict of NUMPY arrays (not torch tensors). Returning torch
    tensors through the Pool pipe triggers PyTorch's file-descriptor tensor sharing (one FD per
    tensor) -> '[Errno 24] Too many open files' for big sets like HIV (~41k mols x ~8 tensors).
    Numpy arrays pickle by value, so no FDs are shared; the main process rebuilds torch tensors."""
    import mollm_common as mc
    smiles, y = arg
    try:
        it = mc.mol_to_item(smiles, y)
        if it is None:
            return None
        it.edge_input = it.edge_input[:, :, :5, :].contiguous()  # cap multi-hop dim
        return {k: (v.detach().cpu().numpy() if hasattr(v, "numpy") else v) for k, v in it.items()}
    except Exception:
        return None


def _np_to_item(d):
    """Rebuild an AttrDict of torch tensors from the numpy dict a worker returned."""
    import mollm_common as mc
    it = mc.AttrDict()
    for k, v in d.items():
        it[k] = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
    return it


def _featurize_set(sub_rows, procs, stall=60.0):
    """Featurise a (chunk-sized) list with Pool.imap_unordered + next(timeout): results are consumed
    in completion order (no serial stall); if no new result arrives within `stall` seconds (all
    remaining molecules are hung, or a worker crashed and stopped the result stream) we break and keep
    what finished. Pool.terminate() hard-kills any lingering workers. Dropped = hung/crashed/parse."""
    import multiprocessing as mp
    items = []
    pool = mp.Pool(processes=procs)
    it = pool.imap_unordered(_feat_worker, sub_rows)
    got, n = 0, len(sub_rows)
    while got < n:
        try:
            r = it.next(timeout=stall)
        except mp.TimeoutError:
            break          # no completion within stall -> remaining hung / result stream stalled
        except StopIteration:
            break
        except Exception:
            break          # worker crash / broken pool -> keep what we have
        got += 1
        if r is not None:
            items.append(_np_to_item(r))
    try:
        pool.terminate(); pool.join()
    except Exception:
        pass
    return items


def featurize(rows, procs):
    """3D-conformer featurisation robust to RDKit hangs AND worker crashes. Rows are processed in
    CHUNKS, each via a fresh Pool, so one crashing/hanging molecule only disrupts its chunk (losing
    at most the in-flight molecules there). Dropped molecules are reported and skipped."""
    if procs <= 1:
        out = []
        for r in rows:
            try:
                d = _feat_worker(r)
                out.append(_np_to_item(d) if d is not None else None)
            except Exception:
                out.append(None)
        return [x for x in out if x is not None]
    chunk = int(os.environ.get("FEAT_CHUNK", "2000"))
    items = []
    for start in range(0, len(rows), chunk):
        sub = rows[start:start + chunk]
        items.extend(_featurize_set(sub, min(procs, len(sub))))
    print(f"  featurize: {len(items)} ok, {len(rows) - len(items)} dropped (crash/hang/parse) "
          f"of {len(rows)}", flush=True)
    return items


NODE_BUDGET = int(os.environ.get("NODE_BUDGET", str(24 * 128 * 128)))  # bounds count * maxN^2 per batch


def make_batches(items, batch, shuffle, rng=None):
    """Node-budget batching: the Transformer-M attention bias / edge_input are O(N^2) per graph, so
    a single large molecule at batch=32 OOMs a 48 GB GPU. We cap each batch so (count * maxN^2) stays
    under NODE_BUDGET (and count <= batch), so a giant molecule lands in a tiny batch -- no molecule
    is dropped. Order within an (optionally shuffled) pass groups by the running max to pack well."""
    idx = list(range(len(items)))
    if shuffle:
        (rng or np.random).shuffle(idx)
    cur, cur_maxn = [], 0
    for j in idx:
        n = int(items[j].x.size(0))
        nm = n if n > cur_maxn else cur_maxn
        if cur and (len(cur) >= batch or (len(cur) + 1) * nm * nm > NODE_BUDGET):
            yield [items[k] for k in cur]
            cur, cur_maxn, nm = [], 0, n
        cur.append(j)
        cur_maxn = nm
    if cur:
        yield [items[k] for k in cur]


def to_device(d, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}


# ---------------------------------------------------------------- metrics
def mean_auc(logits, y):
    prob = 1.0 / (1.0 + np.exp(-logits))
    aucs = []
    for t in range(y.shape[1]):
        m = np.isfinite(y[:, t])
        yt, pt = y[m, t], prob[m, t]
        if yt.size == 0 or yt.min() == yt.max():
            continue
        aucs.append(roc_auc_score(yt.astype(int), pt))
    return (float(np.mean(aucs)) if aucs else None), aucs


def mean_rmse(pred, y):
    rmses = []
    for t in range(y.shape[1]):
        m = np.isfinite(y[:, t])
        if m.sum() == 0:
            continue
        rmses.append(math.sqrt(float(np.mean((pred[m, t] - y[m, t]) ** 2))))
    return (float(np.mean(rmses)) if rmses else None), rmses


def evaluate(model, items, device, T, task_type, norm):
    import mollm_common as mc
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for b in make_batches(items, BATCH, shuffle=False):
            subs = [b]
            while subs:
                sb = subs.pop(0)
                try:
                    batch = to_device(mc.custom_collate(sb), device)
                    out = model(batch).detach().cpu().numpy()
                    if task_type == "regression" and norm is not None:
                        out = out * norm[1] + norm[0]
                    yb = batch["y"].detach().cpu().numpy().reshape(-1, T)
                    P.append(out); Y.append(yb)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and len(sb) > 1:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        mid = len(sb) // 2          # split and retry smaller
                        subs = [sb[:mid], sb[mid:]] + subs
                        continue
                    raise
    model.train()
    return np.concatenate(P), np.concatenate(Y)


# ---------------------------------------------------------------- one task
def run_task(task, meta, ckpt_path, device):
    import mollm_common as mc
    m = meta[task]; targets = m["targets"]; T = len(targets); task_type = m["task_type"]
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    t0 = time.time()

    tr_rows = read_rows(task, "train", targets)
    va_rows = read_rows(task, "valid", targets)
    te_rows = read_rows(task, "test", targets)
    print(f"  rows train/valid/test = {len(tr_rows)}/{len(va_rows)}/{len(te_rows)} T={T}; "
          f"featurising with {FEAT_PROCS} procs...", flush=True)
    tr = featurize(tr_rows, FEAT_PROCS)
    va = featurize(va_rows, FEAT_PROCS)
    te = featurize(te_rows, FEAT_PROCS)
    print(f"  featurised train/valid/test = {len(tr)}/{len(va)}/{len(te)} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)

    norm = None
    if task_type == "regression":
        ys = np.array([float(it.y[0]) for it in tr if np.isfinite(float(it.y[0]))])
        mu = float(ys.mean()); sd = float(ys.std()) or 1.0
        norm = (mu, sd)
        print(f"  target norm mu={mu:.4f} sd={sd:.4f}", flush=True)

    model = mc.MolLMPredictor(num_tasks=T, use_3d=USE_3D).to(device)
    n_enc, n_miss, n_unexp = model.from_pretrained(ckpt_path, map_location="cpu")
    opt = torch.optim.Adam([
        {"params": model.transformer.parameters(), "lr": LR},
        {"params": model.graph_pred_head.parameters(), "lr": LR},
    ], lr=LR)

    best_val = -1e18 if task_type == "classification" else 1e18
    best_state, best_epoch, since, epochs_done = None, -1, 0, 0
    rng = np.random.RandomState(SEED)
    for ep in range(EPOCHS):
        model.train()
        oom = 0
        for b in make_batches(tr, BATCH, shuffle=True, rng=rng):
            if len(b) < 2:   # BatchNorm in the head needs >=2 samples
                continue
            try:
                batch = to_device(mc.custom_collate(b), device)
                opt.zero_grad()
                out = model(batch)
                y = batch["y"].reshape(out.shape)
                mask = torch.isfinite(y).float()
                yf = torch.nan_to_num(y, nan=0.0)
                if task_type == "classification":
                    per = F.binary_cross_entropy_with_logits(out, yf, reduction="none")
                else:
                    tgt = (yf - norm[0]) / norm[1]
                    per = (out - tgt) ** 2
                loss = (per * mask).sum() / mask.sum().clamp(min=1)
                loss.backward(); opt.step()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom += 1
                    opt.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise
        if oom:
            print(f"  (ep{ep}: skipped {oom} OOM batches)", flush=True)
        epochs_done = ep + 1

        P, Y = evaluate(model, va, device, T, task_type, norm)
        if task_type == "classification":
            vmet, _ = mean_auc(P, Y); improved = (vmet is not None and vmet > best_val)
        else:
            vmet, _ = mean_rmse(P, Y); improved = (vmet is not None and vmet < best_val)
        if improved:
            best_val = vmet; best_epoch = ep; since = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        mins = (time.time() - t0) / 60
        print(f"  ep{ep:03d} val={vmet} best={best_val:.4f}@{best_epoch} ({mins:.1f}min)", flush=True)
        if since >= PATIENCE:
            print(f"  early stop ep{ep} (patience {PATIENCE})", flush=True); break
        if mins >= MAX_TASK_MIN:
            print(f"  time cap {MAX_TASK_MIN}min reached at ep{ep}", flush=True); break

    if best_state is None:
        return {"task": task, "seed": SEED, "status": "no_valid_checkpoint"}
    model.load_state_dict(best_state)
    P, Y = evaluate(model, te, device, T, task_type, norm)
    if task_type == "classification":
        tval, per = mean_auc(P, Y); metric = "AUC"
    else:
        tval, per = mean_rmse(P, Y); metric = "RMSE"
    return {"task": task, "seed": SEED, "status": "ok", "metric": metric, "value": tval,
            "val_value": best_val, "best_epoch": best_epoch, "epochs_done": epochs_done,
            "n_targets_scored": len(per), "per_target": per, "task_type": task_type,
            "n_train": len(tr), "n_valid": len(va), "n_test": len(te),
            "enc_tensors": n_enc, "missing_keys": n_miss, "unexpected_keys": n_unexp,
            "use_3d": USE_3D, "minutes": round((time.time() - t0) / 60, 1)}


# ---------------------------------------------------------------- smoke
def smoke(meta, ckpt_path, device):
    import mollm_common as mc
    print("=== SMOKE: build + import + load ckpt + 1 fwd/bwd ===", flush=True)
    rows = read_rows("bbbp", "train", meta["bbbp"]["targets"])[:16]
    items = featurize(rows, 2)
    print("  featurised", len(items), "mols; keys:", sorted(items[0].keys()), flush=True)
    model = mc.MolLMPredictor(num_tasks=1, use_3d=USE_3D).to(device)
    n_enc, n_miss, n_unexp = model.from_pretrained(ckpt_path, map_location="cpu")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    # run several steps so the JIT TensorExpr fuser (which needs a few iters to kick in) is exercised
    loss = None
    for step in range(6):
        batch = to_device(mc.custom_collate(items), device)
        out = model(batch)
        y = batch["y"].reshape(out.shape)
        mask = torch.isfinite(y).float()
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(out, torch.nan_to_num(y), reduction="none") * mask).sum() / mask.sum().clamp(min=1)
        loss.backward(); opt.step()
        print("  smoke step", step, "loss", float(loss), flush=True)
    res = {"status": "smoke_ok", "out_shape": list(out.shape), "loss": float(loss),
           "enc_tensors": n_enc, "missing_keys": n_miss, "unexpected_keys": n_unexp,
           "n_params": sum(p.numel() for p in model.parameters()),
           "cuda": torch.cuda.is_available(),
           "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
           "torch": torch.__version__}
    print("SMOKE RESULT:", json.dumps(res), flush=True)
    os.makedirs(OUTD, exist_ok=True)
    json.dump(res, open(os.path.join(OUTD, "smoke.json"), "w"), indent=2)
    s3_put(f"{S3_INCR}/smoke.json", res)
    return res


def self_check(meta, ckpt_path, device):
    """Fast in-job validation that mimics real training (several optimiser steps over >1 batch
    shape) so the job fails fast if the JIT/nvrtc fusion issue is not fully suppressed, before any
    long featurisation."""
    import mollm_common as mc
    print("=== SELF-CHECK: real-training-like GPU steps (fuser path) ===", flush=True)
    rows = read_rows("bbbp", "train", meta["bbbp"]["targets"])[:48]
    items = featurize(rows, 4)
    model = mc.MolLMPredictor(num_tasks=1, use_3d=USE_3D).to(device)
    model.from_pretrained(ckpt_path, map_location="cpu")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    # repeat the SAME batch shape many times: this is what triggers the profiling-executor
    # specialisation + fusion that broke real training (a few mixed-shape steps did NOT).
    for step, bs in enumerate([32, 32, 32, 32, 32, 32, 16, 16, 8]):
        b = items[:bs]
        batch = to_device(mc.custom_collate(b), device)
        out = model(batch)
        y = batch["y"].reshape(out.shape)
        mask = torch.isfinite(y).float()
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(out, torch.nan_to_num(y), reduction="none") * mask).sum() / mask.sum().clamp(min=1)
        loss.backward(); opt.step()
        print(f"  self-check step{step} bs={bs} loss={float(loss):.4f}", flush=True)
    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  SELF-CHECK PASSED", flush=True)


# ---------------------------------------------------------------- main
TASK_ORDER = ["bbbp", "bace", "clintox", "sider", "esol", "freesolv", "tox21", "lipo", "hiv"]


def main():
    meta = json.load(open(f"{DATA}/meta.json"))
    raise_fd_limit()
    diag = disable_jit_fuser()
    if torch.cuda.is_available():
        diag["device_cap"] = list(torch.cuda.get_device_capability(0))
        diag["device_name"] = torch.cuda.get_device_name(0)
    diag["torch"] = torch.__version__
    diag["PYTORCH_JIT"] = os.environ.get("PYTORCH_JIT")
    s3_put(f"{S3_INCR}/jit_diag.json", diag)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("MODE=%s device=%s torch=%s cuda=%s" %
          (MODE, device, torch.__version__, torch.cuda.is_available()), flush=True)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    ckpt_path = resolve_ckpt()

    if MODE == "smoke":
        smoke(meta, ckpt_path, device)
        return

    self_check(meta, ckpt_path, device)

    sel = os.environ.get("TASKS", "").strip()
    tasks = [t for t in TASK_ORDER if (not sel or t in [s.strip() for s in sel.split(",")])]
    print("TASKS:", tasks, flush=True)
    os.makedirs(f"{OUTD}/runs", exist_ok=True)
    allres, failures = [], 0
    for task in tasks:
        prev = s3_get(f"{S3_INCR}/{task}_seed{SEED}.json")
        if prev and prev.get("status") == "ok" and prev.get("value") is not None:
            print(f"=== {task}: already done in S3 (value={prev['value']}), skipping ===", flush=True)
            allres.append(prev); continue
        print(f"=== {task} seed{SEED} ===", flush=True)
        try:
            r = run_task(task, meta, ckpt_path, device)
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {"task": task, "seed": SEED, "status": "exception:" + str(e)[:200],
                 "traceback": traceback.format_exc()[-2500:]}
        if r.get("status") != "ok":
            failures += 1
        allres.append(r)
        json.dump(r, open(f"{OUTD}/runs/{task}_seed{SEED}.json", "w"), indent=2)
        s3_put(f"{S3_INCR}/{task}_seed{SEED}.json", r)
        print(f"  -> {r.get('status')} {r.get('metric')}={r.get('value')}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    json.dump(allres, open(f"{OUTD}/summary.json", "w"), indent=2)
    print("ALL DONE failures=", failures, flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
