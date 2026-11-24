#!/usr/bin/env python3
"""
GIN (trained from scratch) supervised baseline for MoleculeNet, comparable to MolE-RAG.

- Reuses the pretrain-gnns GIN architecture + featurization (gin_model.py, featurize.py), random init.
- One model per dataset. Multitask datasets (clintox/tox21/sider) use multiple output heads and
  mask missing labels in the loss and in evaluation.
- Trains on our scaffold-split TRAIN, selects hyperparameters + best checkpoint on VALID, and
  evaluates on exactly the MolE-RAG TEST molecules (same seed-0 split).
- Repeats over >=3 initialization seeds; reports mean +/- std.

GIN accesses the training labels through parameter updates; MolE-RAG accesses the same labels
through retrieved analogues. We report that distinction explicitly.

Usage:
  python train_gin.py --datasets bbbp --seeds 0 1 2
  python train_gin.py --datasets bbbp bace clintox hiv tox21 sider esol freesolv lipo --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from rdkit import Chem, RDLogger

from featurize import smiles_to_data
from gin_model import GNN_graphpred

RDLogger.DisableLog("rdApp.*")

import os
# Data root is overridable (Delta default here; on SageMaker point at the input channel via
# --data-root or $GIN_DATA_ROOT, e.g. /opt/ml/input/data/training/moleculenet_property_scaffold).
DATA_ROOT = Path(os.environ.get(
    "GIN_DATA_ROOT",
    "/workspace/MolE-RAG/data/moleculenet_property_scaffold"))

DATASET_TASK_TYPE = {
    "bbbp": "classification", "bace": "classification", "hiv": "classification",
    "clintox": "classification", "tox21": "classification", "sider": "classification",
    "esol": "regression", "freesolv": "regression", "lipo": "regression",
}

# Architecture defaults (pretrain-gnns finetune.py)
NUM_LAYER = 5
EMB_DIM = 300
JK = "last"
GRAPH_POOLING = "mean"

# Per-dataset train budget overrides (bigger sets -> larger batch, fewer epochs)
DATASET_OVERRIDES = {
    "hiv":   {"batch_size": 256, "epochs": 60},
    "tox21": {"batch_size": 128, "epochs": 100},
    "sider": {"batch_size": 64,  "epochs": 100},
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def canonical(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m, isomericSmiles=True) if m is not None else None


def load_split(dataset, split, endpoints, split_seed):
    """Return (data_list, invalid_rows). Each Data carries y[num_tasks] and mask[num_tasks]."""
    path = DATA_ROOT / f"seed_{split_seed}" / dataset / f"{dataset}_{split}.csv"
    df = pd.read_csv(path)
    ep_index = {e: i for i, e in enumerate(endpoints)}
    nt = len(endpoints)

    # pivot long -> per-molecule label vector + mask (keyed by canonical, stereo-preserving SMILES)
    mols = {}   # can_smiles -> {"y": [nt], "mask": [nt], "orig": smiles}
    order = []
    for row in df.itertuples(index=False):
        smi = str(getattr(row, "smiles")).strip()
        ep = str(getattr(row, "task")).strip()
        try:
            lab = float(getattr(row, "label"))
        except (TypeError, ValueError):
            continue
        can = canonical(smi)
        if can is None:
            continue
        if can not in mols:
            mols[can] = {"y": [np.nan] * nt, "mask": [0.0] * nt, "orig": smi}
            order.append(can)
        if ep in ep_index and np.isnan(mols[can]["y"][ep_index[ep]]):
            mols[can]["y"][ep_index[ep]] = lab
            mols[can]["mask"][ep_index[ep]] = 1.0

    data_list, invalid = [], []
    for can in order:
        rec = mols[can]
        data, reason = smiles_to_data(can)
        if data is None:
            invalid.append({"dataset": dataset, "split": split, "smiles": rec["orig"], "reason": reason})
            continue
        y = np.nan_to_num(np.array(rec["y"], dtype=np.float32), nan=0.0)
        data.y = torch.tensor(y, dtype=torch.float).view(1, nt)
        data.mask = torch.tensor(rec["mask"], dtype=torch.float).view(1, nt)
        data.smiles = can
        data_list.append(data)
    return data_list, invalid


def build_dataset(dataset, split_seed):
    """Load all splits; derive endpoints from the train CSV `task` column."""
    train_path = DATA_ROOT / f"seed_{split_seed}" / dataset / f"{dataset}_train.csv"
    endpoints = sorted(pd.read_csv(train_path, usecols=["task"])["task"].astype(str).str.strip().unique())
    invalid_all = []
    splits = {}
    for s in ("train", "valid", "test"):
        dl, inv = load_split(dataset, s, endpoints, split_seed)
        splits[s] = dl
        invalid_all.extend(inv)
    return splits, endpoints, invalid_all


# ----------------------------------------------------------------------------
# Train / eval
# ----------------------------------------------------------------------------
def masked_cls_loss(logits, y, mask):
    per = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (per * mask).sum() / mask.sum().clamp(min=1.0)


def masked_reg_loss(pred, y, mask):
    per = (pred - y) ** 2
    return (per * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    outs, ys, masks = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        outs.append(out.cpu()); ys.append(batch.y.cpu()); masks.append(batch.mask.cpu())
    return torch.cat(outs), torch.cat(ys), torch.cat(masks)


def eval_classification(model, loader, device, endpoints):
    logits, y, mask = predict(model, loader, device)
    probs = torch.sigmoid(logits).numpy()
    y = y.numpy(); mask = mask.numpy()
    aucs, per_ep = [], {}
    for i, ep in enumerate(endpoints):
        m = mask[:, i] > 0
        yt = y[m, i]
        if m.sum() < 2 or len(set(yt.round().astype(int))) < 2:
            per_ep[ep] = None
            continue
        a = float(roc_auc_score(yt, probs[m, i]))
        per_ep[ep] = a; aucs.append(a)
    macro = float(np.mean(aucs)) if aucs else None
    return macro, per_ep, len(aucs)


def eval_regression(model, loader, device, y_mean, y_std):
    logits, y, mask = predict(model, loader, device)
    m = (mask[:, 0] > 0).numpy()
    pred = logits.numpy()[m, 0] * y_std + y_mean          # invert standardization
    true = y.numpy()[m, 0] * y_std + y_mean
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    return rmse, mae, int(m.sum())


def train_one(dataset, task_type, splits, endpoints, hp, seed, device, y_mean, y_std):
    """Train one model; return best-val checkpoint's val+test metrics (+ test predictions)."""
    torch.manual_seed(seed); np.random.seed(seed)
    nt = len(endpoints)
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(splits["train"], batch_size=hp["batch_size"], shuffle=True, generator=g)
    valid_loader = DataLoader(splits["valid"], batch_size=256, shuffle=False)
    test_loader = DataLoader(splits["test"], batch_size=256, shuffle=False)

    model = GNN_graphpred(NUM_LAYER, EMB_DIM, nt, JK=JK, drop_ratio=hp["dropout"],
                          graph_pooling=GRAPH_POOLING, gnn_type="gin").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=0.0)

    best = {"val": None, "epoch": -1, "test": None, "test_aux": None, "per_ep": None, "n_eval": None}
    better = (lambda a, b: a > b) if task_type == "classification" else (lambda a, b: a < b)

    for epoch in range(hp["epochs"]):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch)
            if task_type == "classification":
                loss = masked_cls_loss(out, batch.y, batch.mask)
            else:
                loss = masked_reg_loss(out, batch.y, batch.mask)
            opt.zero_grad(); loss.backward(); opt.step()

        if task_type == "classification":
            val_m, _, _ = eval_classification(model, valid_loader, device, endpoints)
            val_for_sel = val_m if val_m is not None else -1e9
        else:
            val_m, _, _ = eval_regression(model, valid_loader, device, y_mean, y_std)
            val_for_sel = val_m if val_m is not None else 1e9

        if best["val"] is None or better(val_for_sel, best["val"]):
            best["val"] = val_for_sel; best["epoch"] = epoch
            if task_type == "classification":
                t_macro, t_per, t_n = eval_classification(model, test_loader, device, endpoints)
                best.update({"test": t_macro, "per_ep": t_per, "n_eval": t_n})
            else:
                t_rmse, t_mae, t_n = eval_regression(model, test_loader, device, y_mean, y_std)
                best.update({"test": t_rmse, "test_aux": {"mae": t_mae}, "n_eval": t_n})
    return best


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run_dataset(dataset, args, device, out_dir):
    t0 = time.time()
    task_type = DATASET_TASK_TYPE[dataset]
    ov = DATASET_OVERRIDES.get(dataset, {})
    epochs = args.epochs or ov.get("epochs", 100)
    batch_size = args.batch_size or ov.get("batch_size", 128)

    splits, endpoints, invalid = build_dataset(dataset, args.split_seed)

    # regression target standardization (fit on observed train labels)
    y_mean = y_std = None
    if task_type == "regression":
        tr = np.array([d.y[0, 0].item() for d in splits["train"] if d.mask[0, 0] > 0])
        y_mean, y_std = float(tr.mean()), float(tr.std() + 1e-8)
        for s in splits:
            for d in splits[s]:
                d.y[0, 0] = (d.y[0, 0] - y_mean) / y_std   # standardized target

    hp_grid = [{"lr": lr, "dropout": args.dropout, "epochs": epochs, "batch_size": batch_size}
               for lr in args.lrs]

    results = {"dataset": dataset, "task_type": task_type, "endpoints": endpoints,
               "n_endpoints": len(endpoints), "n_invalid": len(invalid),
               "n_train": len(splits["train"]), "n_valid": len(splits["valid"]),
               "n_test": len(splits["test"]), "epochs": epochs, "batch_size": batch_size,
               "hp_grid": hp_grid, "per_hp": [], "init_seeds": args.seeds}

    best_hp, best_hp_val = None, None
    better_mean = (lambda a, b: a > b) if task_type == "classification" else (lambda a, b: a < b)

    for hp in hp_grid:
        seed_bests = []
        for seed in args.seeds:
            b = train_one(dataset, task_type, splits, endpoints, hp, seed, device, y_mean, y_std)
            seed_bests.append(b)
        vals = [b["val"] for b in seed_bests]
        tests = [b["test"] for b in seed_bests if b["test"] is not None]
        mean_val = float(np.mean(vals))
        entry = {"hp": hp, "val_per_seed": vals, "mean_val": mean_val,
                 "test_per_seed": [b["test"] for b in seed_bests],
                 "best_epoch_per_seed": [b["epoch"] for b in seed_bests],
                 "test_mean": float(np.mean(tests)) if tests else None,
                 "test_std": float(np.std(tests)) if tests else None,
                 "seed_bests": seed_bests}
        results["per_hp"].append({k: v for k, v in entry.items() if k != "seed_bests"})
        if best_hp_val is None or better_mean(mean_val, best_hp_val):
            best_hp_val = mean_val; best_hp = entry

    # headline from selected hp
    sel = best_hp
    results["selected_hp"] = sel["hp"]
    results["test_mean"] = sel["test_mean"]
    results["test_std"] = sel["test_std"]
    results["test_per_seed"] = sel["test_per_seed"]
    results["selected_mean_val"] = sel["mean_val"]
    if task_type == "classification":
        results["metric"] = "macro_roc_auc"
        results["per_endpoint_roc_auc_seed0"] = sel["seed_bests"][0]["per_ep"]
        results["n_endpoints_evaluable"] = sel["seed_bests"][0]["n_eval"]
    else:
        results["metric"] = "rmse"
        results["test_mae_per_seed"] = [b["test_aux"]["mae"] if b["test_aux"] else None
                                        for b in sel["seed_bests"]]
        results["y_mean"], results["y_std"] = y_mean, y_std
    results["coverage"] = results["n_test"] and (1.0 - len(
        [r for r in invalid if r["split"] == "test"]) / (results["n_test"] + len(
        [r for r in invalid if r["split"] == "test"])))
    results["runtime_s"] = round(time.time() - t0, 1)

    ddir = out_dir / dataset
    ddir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(invalid).to_csv(ddir / "invalid_smiles.csv", index=False)
    with open(ddir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_TASK_TYPE.keys()))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="initialization seeds")
    ap.add_argument("--split-seed", type=int, default=0, help="scaffold-split seed (MolE-RAG uses 0)")
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 1e-4])
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=0, help="0 -> per-dataset default/override")
    ap.add_argument("--batch-size", type=int, default=0, help="0 -> per-dataset default/override")
    ap.add_argument("--out-dir", default="/workspace/mole-rag_naacl/gin/seed_0")
    ap.add_argument("--data-root", default=None, help="override moleculenet_property_scaffold dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.data_root:
        global DATA_ROOT
        DATA_ROOT = Path(args.data_root)

    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    config = {"method": "GIN (trained from scratch)", "arch": "pretrain-gnns GIN (ported to PyG 2.x)",
              "num_layer": NUM_LAYER, "emb_dim": EMB_DIM, "JK": JK, "graph_pooling": GRAPH_POOLING,
              "optimizer": "Adam", "weight_decay": 0.0, "dropout": args.dropout, "lrs": args.lrs,
              "init_seeds": args.seeds, "split_seed": args.split_seed, "device": str(device),
              "python": platform.python_version(), "torch": torch.__version__}
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"device={device}  datasets={args.datasets}  init_seeds={args.seeds}  lrs={args.lrs}")
    summary = []
    for d in args.datasets:
        r = run_dataset(d, args, device, out_dir)
        mean, std = r["test_mean"], r["test_std"]
        ms = f"{mean:.4f}+/-{std:.4f}" if mean is not None else "n/a"
        print(f"{d:<9} {r['metric']:<14} {ms:<18} sel_lr={r['selected_hp']['lr']} "
              f"n_ep={r['n_endpoints']} t={r['runtime_s']}s")
        summary.append({"dataset": d, "task_type": r["task_type"], "metric": r["metric"],
                        "test_mean": mean, "test_std": std, "selected_lr": r["selected_hp"]["lr"],
                        "n_endpoints": r["n_endpoints"], "runtime_s": r["runtime_s"]})
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
