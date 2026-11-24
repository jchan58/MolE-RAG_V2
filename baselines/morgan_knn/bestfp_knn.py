#!/usr/bin/env python3
"""
Best-fingerprint KNN baseline for MoleculeNet property prediction.

A no-LLM nearest-neighbour baseline that mirrors the Morgan-KNN baseline in every
respect *except* the fingerprint: instead of a single fixed Morgan fingerprint for
all datasets, each dataset uses the **same structural-retrieval fingerprint that
MolE-RAG deploys for that dataset** (the per-task `DATASET_BEST_FP` map, fixed
k=5). This turns the KNN baseline into a like-for-like structural comparison with
MolE-RAG: "with the exact fingerprint MolE-RAG retrieves neighbours with, and no
LLM, how far does nearest-neighbour voting get you?"

The per-dataset fingerprint identities come from MolE-RAG's deployed structural
map (selected by the paper's gpt-4o-mini validation sweep; see the Molecular
Fingerprints appendix). The RDKit builders below reproduce the exact sweep
configurations (2048-bit hashed where applicable, RDKit-default chirality=False;
MACCS is 167 keyed bits).

Method configuration (frozen, applied consistently to every dataset):
    Fingerprint         per-dataset best FP (DATASET_BEST_FP, see below)
    Similarity          Tanimoto
    Neighbour weighting  Uniform (unweighted)
    k                   5  (MolE-RAG deploys fixed k=5 structural pools)
    Reference database  Training split only

Everything else -- splitting, canonicalization, tie-breaks, k-selection logic,
metrics, outputs -- is identical to morgan_knn_k5.py so the two baselines differ
only in the fingerprint.

Usage:
    python bestfp_knn.py                         # all 9 tasks, seed 0
    python bestfp_knn.py --datasets bbbp esol
    python bestfp_knn.py --selftest              # correctness checks only

Data root resolves from $KNN_DATA_ROOT, else the repo's data/moleculenet.
Outputs land in --out-dir (default: this script's results_bestfp/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import rdkit
import sklearn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")

# ----------------------------------------------------------------------------
# Paths (portable: env override, else repo-relative)
# ----------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]  # baselines/morgan_knn/ -> repo root
DATA_ROOT = Path(os.environ.get("KNN_DATA_ROOT", _REPO_ROOT / "data" / "moleculenet"))

# ----------------------------------------------------------------------------
# Fingerprints: exact MolE-RAG structural-retrieval configurations
# (from the Molecular Fingerprints appendix fingerprint_configurations.csv)
# ----------------------------------------------------------------------------
def _fp_ecfp2(mol):   # Morgan radius 1, 2048 bits (ECFP diameter-2 naming)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=2048)

def _fp_fcfp2(mol):   # Feature-Morgan radius 1, 2048 bits
    return AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=2048, useFeatures=True)

def _fp_maccs(mol):   # 167 keyed bits (bit 0 unused)
    return MACCSkeys.GenMACCSKeys(mol)

def _fp_rdkit(mol):   # RDKit topological (Daylight-like), default maxPath=7, 2048 bits
    return Chem.RDKFingerprint(mol)

def _fp_toptor(mol):  # Hashed topological torsion, 2048 bits
    return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)

def _fp_atompair(mol):  # Hashed atom pair, 2048 bits
    return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)

FP_BUILDERS = {
    "ECFP2": _fp_ecfp2,
    "FCFP2": _fp_fcfp2,
    "MACCS": _fp_maccs,
    "RDKit": _fp_rdkit,
    "Topological Torsion": _fp_toptor,
    "Atom Pair": _fp_atompair,
}

# Exact RDKit call per fingerprint, for provenance in config.json.
FP_RDKIT_CALL = {
    "ECFP2": "AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=2048)",
    "FCFP2": "AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=2048, useFeatures=True)",
    "MACCS": "MACCSkeys.GenMACCSKeys(mol)  # 167 keyed bits",
    "RDKit": "Chem.RDKFingerprint(mol)  # default maxPath=7, 2048 bits",
    "Topological Torsion": "rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)",
    "Atom Pair": "rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)",
}

# MolE-RAG's deployed per-dataset structural-retrieval fingerprint (DATASET_BEST_FP).
DATASET_BEST_FP = {
    "bace": "Topological Torsion",
    "bbbp": "ECFP2",
    "clintox": "FCFP2",
    "esol": "Atom Pair",
    "freesolv": "MACCS",
    "hiv": "RDKit",
    "lipo": "Atom Pair",
    "sider": "FCFP2",
    "tox21": "Atom Pair",
}

# Per-dataset neighbourhood size k, from the paper's fingerprint-selection table
# (the validation-selected k that accompanies each best fingerprint).
DATASET_BEST_K = {
    "bace": 5,
    "bbbp": 3,
    "clintox": 5,
    "esol": 5,
    "freesolv": 3,
    "hiv": 5,
    "lipo": 5,
    "sider": 5,
    "tox21": 5,
}

# ----------------------------------------------------------------------------
# Frozen method configuration
# ----------------------------------------------------------------------------
METHOD_NAME = "BestFP-KNN"
MAX_K = 5  # retrieve up to 5 neighbours; the per-dataset k selects how many are used
FALLBACK_K = 5

DATASET_TASK_TYPE = {
    "bbbp": "classification",
    "bace": "classification",
    "clintox": "classification",
    "hiv": "classification",
    "tox21": "classification",
    "sider": "classification",
    "esol": "regression",
    "freesolv": "regression",
    "lipo": "regression",
}
ALL_DATASETS = list(DATASET_TASK_TYPE.keys())


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass
class MolRecord:
    molecule_id: str
    orig_smiles: str
    canonical_smiles: str
    split: str
    endpoint: str
    label: float
    label_is_observed: bool
    row_order: int
    fp: object = field(default=None, repr=False)


@dataclass
class RunConfig:
    method_name: str = METHOD_NAME
    fingerprint_per_dataset: dict = field(default_factory=lambda: dict(DATASET_BEST_FP))
    fingerprint_rdkit_call: dict = field(default_factory=lambda: dict(FP_RDKIT_CALL))
    k_per_dataset: dict = field(default_factory=lambda: dict(DATASET_BEST_K))
    fingerprint_source: str = (
        "MolE-RAG deployed structural-retrieval map DATASET_BEST_FP "
        "(rebuild_fewshot_pools.py), selected by the paper's gpt-4o-mini validation "
        "sweep; see the Molecular Fingerprints appendix. k is the paper's "
        "validation-selected neighbourhood size per dataset."
    )
    similarity: str = "Tanimoto"
    neighbor_weighting: str = "uniform"
    max_retrieved: int = MAX_K
    reference_database: str = "training split only"
    tie_break: str = "ascending reference molecule_id, then source-row order (numpy stable sort)"
    preprocessing: str = (
        "RDKit MolFromSmiles only; no new salt stripping / neutralization / "
        "tautomer normalization (matches make_scaffold_splits.py)"
    )
    python_version: str = platform.python_version()
    rdkit_version: str = rdkit.__version__
    sklearn_version: str = sklearn.__version__
    numpy_version: str = np.__version__
    pandas_version: str = pd.__version__


# ----------------------------------------------------------------------------
# Fingerprints (cached by (fp_name, canonical SMILES))
# ----------------------------------------------------------------------------
_FP_CACHE: dict[tuple, object] = {}


def canonical_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def make_fingerprint(can_smiles: str, fp_name: str) -> Optional[object]:
    key = (fp_name, can_smiles)
    if key in _FP_CACHE:
        return _FP_CACHE[key]
    mol = Chem.MolFromSmiles(can_smiles)
    fp = FP_BUILDERS[fp_name](mol) if mol is not None else None
    _FP_CACHE[key] = fp
    return fp


# ----------------------------------------------------------------------------
# Loading & validation
# ----------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(dataset: str, seed: int):
    fp_name = DATASET_BEST_FP[dataset]
    folder = DATA_ROOT / f"seed_{seed}" / dataset
    hashes = {}
    structure_to_id: dict[str, str] = {}
    structure_splits: dict[str, set] = {}
    next_id = [0]

    def get_id(can: str) -> str:
        if can not in structure_to_id:
            structure_to_id[can] = f"{dataset}_{next_id[0]:06d}"
            next_id[0] += 1
            structure_splits[can] = set()
        return structure_to_id[can]

    records_by_split: dict[str, list[MolRecord]] = {}
    invalid_rows = []
    endpoints: set[str] = set()

    for split in ("train", "valid", "test"):
        path = folder / f"{dataset}_{split}.csv"
        try:
            rel_path = str(path.relative_to(_REPO_ROOT))
        except ValueError:
            rel_path = path.name
        hashes[split] = {"path": rel_path, "sha256": file_sha256(path)}
        df = pd.read_csv(path)
        recs: list[MolRecord] = []
        for row_order, row in enumerate(df.itertuples(index=False)):
            smi = str(getattr(row, "smiles")).strip()
            endpoint = str(getattr(row, "task")).strip()
            raw_label = getattr(row, "label")
            endpoints.add(endpoint)

            can = canonical_smiles(smi)
            if can is None:
                invalid_rows.append({"dataset": dataset, "split": split,
                                     "row_order": row_order, "smiles": smi,
                                     "reason": "RDKit could not parse SMILES"})
                continue

            mid = get_id(can)
            structure_splits[can].add(split)

            observed = raw_label is not None and not (isinstance(raw_label, float) and np.isnan(raw_label))
            try:
                label_val = float(raw_label)
            except (TypeError, ValueError):
                label_val = np.nan
                observed = False

            recs.append(MolRecord(
                molecule_id=mid, orig_smiles=smi, canonical_smiles=can,
                split=split, endpoint=endpoint, label=label_val,
                label_is_observed=observed, row_order=row_order,
                fp=make_fingerprint(can, fp_name),
            ))
        records_by_split[split] = recs

    duplicate_report = []
    for can, splits in structure_splits.items():
        if len(splits) > 1:
            duplicate_report.append({
                "dataset": dataset,
                "molecule_id": structure_to_id[can],
                "canonical_smiles": can,
                "splits_present": "|".join(sorted(splits)),
            })

    return records_by_split, sorted(endpoints), invalid_rows, duplicate_report, hashes, fp_name


# ----------------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------------
def build_reference_pool(train_recs: list[MolRecord], endpoint: str):
    refs = [r for r in train_recs
            if r.endpoint == endpoint and r.label_is_observed and r.fp is not None]
    refs.sort(key=lambda r: (r.molecule_id, r.row_order))
    return refs


def retrieve(query_fp, refs: list[MolRecord], max_k: int = MAX_K):
    if not refs or query_fp is None:
        return []
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, [r.fp for r in refs]))
    order = np.argsort(-sims, kind="stable")[:max_k]
    return [(refs[i], float(sims[i])) for i in order]


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def classify_score(neighbors, k: int) -> Optional[float]:
    labs = [int(round(r.label)) for r, _ in neighbors[:k]]
    if not labs:
        return None
    return sum(labs) / len(labs)


def classify_hard(neighbors, k: int, train_majority: int) -> Optional[int]:
    labs = [int(round(r.label)) for r, _ in neighbors[:k]]
    if not labs:
        return None
    counts = Counter(labs)
    top = counts.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    return train_majority


def regress_value(neighbors, k: int) -> Optional[float]:
    vals = [r.label for r, _ in neighbors[:k]]
    if not vals:
        return None
    return float(np.mean(vals))


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def rmse(preds, trues) -> float:
    p, t = np.asarray(preds, float), np.asarray(trues, float)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(preds, trues) -> float:
    p, t = np.asarray(preds, float), np.asarray(trues, float)
    return float(np.mean(np.abs(p - t)))


def safe_auc(y_true, y_score) -> Optional[float]:
    if len(set(int(round(y)) for y in y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


# ----------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------
def neighbours_for_split(query_recs, refs):
    out = []
    for q in query_recs:
        if q.fp is None:
            out.append((q, None))
        else:
            out.append((q, retrieve(q.fp, refs)))
    return out


def endpoint_auc_at_k(query_neighbours, k, train_majority):
    y_true, y_score = [], []
    for q, neigh in query_neighbours:
        if neigh is None or not neigh:
            continue
        s = classify_score(neigh, k)
        if s is None:
            continue
        y_true.append(int(round(q.label)))
        y_score.append(s)
    if len(y_true) < 2:
        return None
    return safe_auc(y_true, y_score)


def endpoint_rmse_at_k(query_neighbours, k):
    preds, trues = [], []
    for q, neigh in query_neighbours:
        if neigh is None or not neigh:
            continue
        v = regress_value(neigh, k)
        if v is None:
            continue
        preds.append(v)
        trues.append(q.label)
    if len(preds) < 2:
        return None
    return rmse(preds, trues)


# ----------------------------------------------------------------------------
# Per-dataset driver
# ----------------------------------------------------------------------------
def run_dataset(dataset: str, seed: int, out_dir: Path):
    t0 = time.time()
    task_type = DATASET_TASK_TYPE[dataset]
    records, endpoints, invalid_rows, dup_report, hashes, fp_name = load_dataset(dataset, seed)
    t_load = time.time() - t0

    train, valid, test = records["train"], records["valid"], records["test"]

    def by_endpoint(recs):
        d: dict[str, list[MolRecord]] = {e: [] for e in endpoints}
        for r in recs:
            d[r.endpoint].append(r)
        return d

    train_e, valid_e, test_e = by_endpoint(train), by_endpoint(valid), by_endpoint(test)

    train_majority = {}
    for e in endpoints:
        labs = [int(round(r.label)) for r in train_e[e] if r.label_is_observed]
        if labs:
            c = Counter(labs)
            top = c.most_common()
            train_majority[e] = top[0][0] if (len(top) == 1 or top[0][1] > top[1][1]) else 0
        else:
            train_majority[e] = 0

    t1 = time.time()
    refs_e = {e: build_reference_pool(train, e) for e in endpoints}
    valid_neigh = {e: neighbours_for_split(valid_e[e], refs_e[e]) for e in endpoints}
    test_neigh = {e: neighbours_for_split(test_e[e], refs_e[e]) for e in endpoints}
    t_retrieve = time.time() - t1

    val_rows = []
    val_per_ep = []
    k_list = [DATASET_BEST_K[dataset]]  # paper's per-dataset neighbourhood size

    if task_type == "classification":
        macro_by_k = {}
        for k in k_list:
            ep_aucs = []
            for e in endpoints:
                auc = endpoint_auc_at_k(valid_neigh[e], k, train_majority[e])
                val_per_ep.append({"dataset": dataset, "endpoint": e, "k": k,
                                   "metric": "roc_auc", "value": auc})
                if auc is not None:
                    ep_aucs.append(auc)
            macro = float(np.mean(ep_aucs)) if ep_aucs else None
            macro_by_k[k] = (macro, len(ep_aucs))
            val_rows.append({"dataset": dataset, "k": k, "metric": "macro_roc_auc",
                             "value": macro, "n_endpoints_evaluable": len(ep_aucs)})
        defined = {k: v for k, (v, _) in macro_by_k.items() if v is not None}
        if defined:
            best = max(defined.values())
            selected_k = min(k for k, v in defined.items() if v == best)
            selection_note = "max macro validation ROC-AUC (tie -> smaller k)"
        else:
            selected_k = FALLBACK_K
            selection_note = f"fallback k={FALLBACK_K}: no endpoint had a defined validation AUC"
    else:
        rmse_by_k = {}
        for k in k_list:
            r = endpoint_rmse_at_k(valid_neigh[endpoints[0]], k)
            rmse_by_k[k] = r
            val_rows.append({"dataset": dataset, "k": k, "metric": "rmse",
                             "value": r, "n_endpoints_evaluable": 1 if r is not None else 0})
        defined = {k: v for k, v in rmse_by_k.items() if v is not None}
        if defined:
            best = min(defined.values())
            selected_k = min(k for k, v in defined.items() if v == best)
            selection_note = "min validation RMSE (tie -> smaller k)"
        else:
            selected_k = FALLBACK_K
            selection_note = f"fallback k={FALLBACK_K}: validation RMSE undefined"

    pred_rows = []
    neigh_rows = []
    n_invalid_test = sum(1 for _, n in sum(test_neigh.values(), []) if n is None)
    n_failures = 0

    per_endpoint_metrics = {}
    reg_preds, reg_trues = [], []

    for e in endpoints:
        y_true, y_score = [], []
        for q, neigh in test_neigh[e]:
            if neigh is None:
                status = "invalid_structure"
                score_or_val, pred_class, max_sim, mean_sim, eff_k = (None, None, None, None, 0)
            elif not neigh:
                status = "no_eligible_references"
                score_or_val, pred_class, max_sim, mean_sim, eff_k = (None, None, None, None, 0)
                n_failures += 1
            else:
                used = neigh[:selected_k]
                eff_k = len(used)
                sims = [s for _, s in used]
                max_sim, mean_sim = float(max(sims)), float(np.mean(sims))
                status = "ok" if max_sim > 0 else "ok_all_zero_similarity"
                if task_type == "classification":
                    score_or_val = classify_score(neigh, selected_k)
                    pred_class = classify_hard(neigh, selected_k, train_majority[e])
                else:
                    score_or_val = regress_value(neigh, selected_k)
                    pred_class = None
                for rank, (rr, ss) in enumerate(used, start=1):
                    neigh_rows.append({
                        "dataset": dataset, "split_id": f"seed{seed}",
                        "query_molecule_id": q.molecule_id, "endpoint": e,
                        "neighbor_rank": rank, "reference_molecule_id": rr.molecule_id,
                        "tanimoto_similarity": round(ss, 6), "reference_label": rr.label,
                    })
                if task_type == "classification":
                    y_true.append(int(round(q.label)))
                    y_score.append(score_or_val)
                else:
                    reg_preds.append(score_or_val)
                    reg_trues.append(q.label)

            pred_rows.append({
                "dataset": dataset, "split_id": f"seed{seed}", "molecule_id": q.molecule_id,
                "endpoint": e, "selected_k": selected_k, "effective_k": eff_k,
                "prediction_score_or_value": (None if score_or_val is None else round(score_or_val, 6)),
                "predicted_class": pred_class,
                "maximum_similarity": (None if max_sim is None else round(max_sim, 6)),
                "mean_similarity": (None if mean_sim is None else round(mean_sim, 6)),
                "status": status,
            })

        if task_type == "classification":
            auc = safe_auc(y_true, y_score) if len(y_true) >= 2 else None
            per_endpoint_metrics[e] = {"roc_auc": auc, "n_eval": len(y_true)}

    n_targets = sum(len(test_e[e]) for e in endpoints)
    n_predicted = sum(1 for r in pred_rows if r["prediction_score_or_value"] is not None)
    coverage = n_predicted / n_targets if n_targets else 0.0

    metrics = {
        "dataset": dataset, "seed": seed, "task_type": task_type,
        "fingerprint": fp_name, "fingerprint_rdkit_call": FP_RDKIT_CALL[fp_name],
        "selected_k": selected_k, "k_selection": selection_note,
        "n_endpoints": len(endpoints),
        "n_targets": n_targets, "n_predicted": n_predicted, "coverage": round(coverage, 4),
        "n_invalid_test_structures": n_invalid_test, "n_failed_predictions": n_failures,
    }
    if task_type == "classification":
        evaluable = {e: m["roc_auc"] for e, m in per_endpoint_metrics.items() if m["roc_auc"] is not None}
        undefined = [e for e, m in per_endpoint_metrics.items() if m["roc_auc"] is None]
        macro = float(np.mean(list(evaluable.values()))) if evaluable else None
        metrics.update({
            "macro_roc_auc": (None if macro is None else round(macro, 4)),
            "n_endpoints_evaluable": len(evaluable),
            "n_endpoints_undefined": len(undefined),
            "undefined_endpoints": undefined,
            "per_endpoint_roc_auc": {e: (None if v is None else round(v, 4))
                                     for e, v in ((e, per_endpoint_metrics[e]["roc_auc"]) for e in endpoints)},
        })
        headline = ("macro_roc_auc", metrics["macro_roc_auc"])
    else:
        metrics.update({
            "rmse": round(rmse(reg_preds, reg_trues), 4) if reg_preds else None,
            "mae": round(mae(reg_preds, reg_trues), 4) if reg_preds else None,
        })
        headline = ("rmse", metrics["rmse"])

    runtime = {"load_s": round(t_load, 2), "retrieve_s": round(t_retrieve, 2),
               "total_s": round(time.time() - t0, 2)}
    metrics["runtime_s"] = runtime

    ddir = out_dir / dataset
    ddir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_rows).to_csv(ddir / "predictions.csv", index=False)
    pd.DataFrame(neigh_rows).to_csv(ddir / "neighbors.csv", index=False)
    pd.DataFrame(val_rows).to_csv(ddir / "validation_scores.csv", index=False)
    if val_per_ep:
        pd.DataFrame(val_per_ep).to_csv(ddir / "validation_scores_per_endpoint.csv", index=False)
    pd.DataFrame(invalid_rows).to_csv(ddir / "invalid_smiles.csv", index=False)
    pd.DataFrame(dup_report).to_csv(ddir / "duplicates.csv", index=False)
    with open(ddir / "metrics.json", "w") as f:
        json.dump({**metrics, "hashes": hashes}, f, indent=2)

    return metrics, headline


# ----------------------------------------------------------------------------
# Correctness checks
# ----------------------------------------------------------------------------
def run_selftests() -> dict:
    results = {}

    def ok(name, cond):
        results[name] = bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    neigh = [(MolRecord("a", "", "", "train", "e", 1.0, True, 0), 0.9),
             (MolRecord("b", "", "", "train", "e", 1.0, True, 1), 0.8),
             (MolRecord("c", "", "", "train", "e", 0.0, True, 2), 0.7),
             (MolRecord("d", "", "", "train", "e", 1.0, True, 3), 0.6)]
    ok("classification_score_1101_is_0.75", classify_score(neigh, 4) == 0.75)

    rneigh = [(MolRecord("a", "", "", "train", "e", 2.0, True, 0), 0.5),
              (MolRecord("b", "", "", "train", "e", 4.0, True, 1), 0.4)]
    ok("regression_mean_2_4_is_3.0", regress_value(rneigh, 2) == 3.0)

    tie2 = [(MolRecord("a", "", "", "train", "e", 1.0, True, 0), 0.5),
            (MolRecord("b", "", "", "train", "e", 0.0, True, 1), 0.4)]
    ok("tie_vote_uses_train_majority", classify_hard(tie2, 2, train_majority=1) == 1)
    ok("tie_vote_train_tie_predicts_0", classify_hard(tie2, 2, train_majority=0) == 0)

    ok("effective_k_short_pool", classify_score(neigh, 10) == 0.75 and len(neigh) == 4)

    # tie-break + Tanimoto=1 on identical structure, for every fingerprint family
    for fp_name in FP_BUILDERS:
        refs = [MolRecord("z_02", "", "C", "train", "e", 1.0, True, 1, fp=make_fingerprint("C", fp_name)),
                MolRecord("z_01", "", "C", "train", "e", 0.0, True, 0, fp=make_fingerprint("C", fp_name))]
        refs.sort(key=lambda r: (r.molecule_id, r.row_order))
        q = make_fingerprint("C", fp_name)
        ranked = retrieve(q, refs, max_k=2)
        ok(f"[{fp_name}] tie_break_by_molecule_id",
           [r.molecule_id for r, _ in ranked] == ["z_01", "z_02"])
        ok(f"[{fp_name}] identical_structure_sim_1.0", abs(ranked[0][1] - 1.0) < 1e-9)

    all_pass = all(results.values())
    print(f"  => {sum(results.values())}/{len(results)} checks passed")
    return {"all_pass": all_pass, "checks": results}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Best-fingerprint KNN MoleculeNet baseline")
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out-dir", default=str(_HERE.parent / "results_bestfp"))
    ap.add_argument("--selftest", action="store_true", help="run correctness checks and exit")
    args = ap.parse_args()

    print(f"=== {METHOD_NAME} correctness checks ===")
    selftest = run_selftests()
    if not selftest["all_pass"]:
        print("Self-tests failed; aborting before evaluation.")
        sys.exit(1)
    if args.selftest:
        return

    cfg = RunConfig()
    print(f"\n=== {METHOD_NAME} config ===")
    print(f"  data_root: {DATA_ROOT}")
    for ds in args.datasets:
        print(f"  {ds:<10} -> {DATASET_BEST_FP[ds]}")

    summary = []
    for seed in args.seeds:
        out_dir = Path(args.out_dir) / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)
        with open(out_dir / "selftest.json", "w") as f:
            json.dump(selftest, f, indent=2)

        print(f"\n=== Evaluating seed {seed} ===")
        print(f"{'Dataset':<10} {'FP':<20} {'Type':<14} {'k':>3} {'Metric':>14} {'Value':>9} "
              f"{'Cover':>6} {'Time(s)':>8}")
        print("-" * 92)
        for dataset in args.datasets:
            metrics, (mname, mval) = run_dataset(dataset, seed, out_dir)
            summary.append({
                "dataset": dataset, "seed": seed, "task_type": metrics["task_type"],
                "fingerprint": metrics["fingerprint"],
                "selected_k": metrics["selected_k"], "metric": mname,
                "value": mval, "coverage": metrics["coverage"],
                "runtime_s": metrics["runtime_s"]["total_s"],
                "n_endpoints": metrics["n_endpoints"],
                "n_endpoints_undefined": metrics.get("n_endpoints_undefined", 0),
            })
            vstr = "n/a" if mval is None else f"{mval:.4f}"
            print(f"{dataset:<10} {metrics['fingerprint']:<20} {metrics['task_type']:<14} "
                  f"{metrics['selected_k']:>3} {mname:>14} {vstr:>9} {metrics['coverage']:>6.3f} "
                  f"{metrics['runtime_s']['total_s']:>8.1f}")

        sdf = pd.DataFrame([s for s in summary if s["seed"] == seed])
        sdf.to_csv(out_dir / "summary.csv", index=False)

        lines = [f"# {METHOD_NAME} baseline (MolE-RAG per-task best fingerprint) — seed {seed}", "",
                 "| Dataset | Fingerprint | Task type | k | Test metric | Value | Coverage | Runtime (s) |",
                 "|---|---|---|---|---|---|---|---|"]
        for s in [s for s in summary if s["seed"] == seed]:
            vstr = "n/a" if s["value"] is None else f"{s['value']:.4f}"
            lines.append(f"| {s['dataset']} | {s['fingerprint']} | {s['task_type']} | {s['selected_k']} | "
                         f"{s['metric']} | {vstr} | {s['coverage']:.3f} | {s['runtime_s']:.1f} |")
        (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
        print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
