# KNN baselines (no-LLM nearest-neighbour)

Two training-free, no-LLM nearest-neighbour baselines for the 9 MoleculeNet
tasks. Both predict a test molecule's property from the labels of its
Tanimoto-nearest **training** molecules — uniform (unweighted) vote for
classification, uniform mean for regression. Morgan-KNN uses a fixed **k = 5**;
BestFP-KNN uses the paper's validation-selected neighbourhood size per dataset
(**k = 3** for BBBP and FreeSolv, **k = 5** otherwise). They answer: *how far
does molecular similarity plus known training labels get you, with no LLM
reasoning at all?*

The two baselines differ in **one thing only — the fingerprint**:

| Script | Fingerprint | Notes |
|---|---|---|
| `morgan_knn_k5.py` | Fixed binary Morgan (radius 2, 2048 bits, chirality on) for every dataset | The paper's KNN row |
| `bestfp_knn.py` | Each dataset's **MolE-RAG structural-retrieval fingerprint** (`DATASET_BEST_FP`) | Like-for-like with MolE-RAG's retrieval |

`morgan_knn_baseline.py` is the same method as `morgan_knn_k5.py` but sweeps
`k ∈ {1,2,4,8,16,32}` and selects k per dataset on validation; the paper uses the
fixed-k=5 variant.

Everything else — scaffold splits, RDKit canonicalization, training-only
reference pool, deterministic tie-breaks (ascending reference id, stable sort),
metrics (macro ROC-AUC / RMSE), and outputs — is identical across the scripts, so
the two baselines are directly comparable and differ only in the fingerprint.

## Per-dataset best fingerprint (`bestfp_knn.py`)

These are MolE-RAG's deployed structural-retrieval fingerprints (the
`DATASET_BEST_FP` map in `rebuild_fewshot_pools.py`), selected by the paper's
gpt-4o-mini validation sweep — see the Molecular Fingerprints appendix. The RDKit
builders reproduce the exact sweep configurations (2048-bit hashed where
applicable, RDKit-default chirality = False; MACCS = 167 keyed bits).

| Dataset | Fingerprint | k | Dataset | Fingerprint | k |
|---|---|---|---|---|---|
| bace | Topological Torsion | 5 | hiv | RDKit | 5 |
| bbbp | ECFP2 (Morgan r=1) | 3 | lipo | Atom Pair | 5 |
| clintox | FCFP2 (feature-Morgan r=1) | 5 | sider | FCFP2 | 5 |
| esol | Atom Pair | 5 | tox21 | Atom Pair | 5 |
| freesolv | MACCS | 3 | | | |

## Run

```bash
# uses ../../data/moleculenet/seed_0 by default (override with $KNN_DATA_ROOT)
python morgan_knn_k5.py          # Morgan-KNN, k=5  -> results_morgan_k5/
python bestfp_knn.py             # BestFP-KNN, k=5  -> results_bestfp/
python morgan_knn_baseline.py    # Morgan-KNN, k swept {1..32} -> results_morgan_sweep/
python bestfp_knn.py --selftest  # correctness checks only
```

Only `rdkit`, `scikit-learn`, `numpy`, `pandas` are needed (all in the repo
`requirements.txt`). Runs on a single CPU in a few minutes (hiv/tox21 dominate).
Fully deterministic: re-running reproduces the tables below exactly.

## Results (seed-0 scaffold test)

Classification = macro ROC-AUC (↑ better); regression = RMSE (↓ better).
Morgan-KNN is fixed k=5; BestFP-KNN uses the paper's per-dataset (fingerprint, k).

| Dataset | Metric | Morgan-KNN (k=5) | BestFP-KNN | FP | k | Δ (BestFP − Morgan) |
|---|---|---|---|---|---|---|
| bbbp | ROC-AUC | 0.8728 | **0.8777** | ECFP2 | 3 | +0.0049 |
| bace | ROC-AUC | 0.7751 | **0.8033** | Topological Torsion | 5 | +0.0282 |
| clintox | ROC-AUC | **0.7496** | 0.6556 | FCFP2 | 5 | −0.0940 |
| tox21 | ROC-AUC | 0.7001 | **0.7337** | Atom Pair | 5 | +0.0336 |
| sider | ROC-AUC | **0.6358** | 0.6119 | FCFP2 | 5 | −0.0239 |
| hiv | ROC-AUC | **0.8010** | 0.7876 | RDKit | 5 | −0.0134 |
| esol | RMSE | 1.2373 | **1.0033** | Atom Pair | 5 | −0.2340 |
| freesolv | RMSE | 4.5243 | **2.7960** | MACCS | 3 | −1.7283 |
| lipo | RMSE | 1.0428 | **0.9362** | Atom Pair | 5 | −0.1066 |

**Reading it.** Switching the KNN fingerprint to MolE-RAG's per-task choice helps
all three regression tasks (clearly for FreeSolv) and three of six classification
tasks (bbbp, bace, tox21), but *hurts* clintox, sider, and hiv. This is expected:
those fingerprints were chosen to maximise an **LLM's** few-shot validation
accuracy, not a nearest-neighbour vote, so they are not guaranteed to be the best
fingerprint for pure KNN. The comparison is still the fair one to report — it is
KNN using exactly the structural similarity MolE-RAG itself retrieves with.

## Outputs

Per baseline, under `results_*/seed_0/`:

- `summary.csv`, `summary.md` — headline metric per dataset.
- `<dataset>/metrics.json` — full metrics + input file SHA-256 hashes + the
  fingerprint used.
- `<dataset>/predictions.csv` — per-molecule prediction, neighbour-similarity
  stats, status.
- `<dataset>/neighbors.csv` — the k retrieved neighbours (id + Tanimoto + label)
  for every test molecule — the full "who voted" audit trail.
- `<dataset>/validation_scores*.csv`, `invalid_smiles.csv`, `duplicates.csv`,
  and `config.json` / `selftest.json` at the seed root.

This is **our** KNN implementation, not an exact reproduction of MolRAG's
unspecified KNN configuration.
