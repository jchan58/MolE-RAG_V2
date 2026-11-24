# GIN (trained from scratch) baseline on MoleculeNet (9 tasks)

Supervised **Graph Isomorphism Network** baseline, trained from random initialization on our
scaffold-split training data — a learned-parameter counterpart to the retrieval baselines. Reuses the
molecular GIN architecture + featurization from
[snap-stanford/pretrain-gnns](https://github.com/snap-stanford/pretrain-gnns) `chem/`
(Hu et al., ICLR 2020), **no external-corpus pretraining**.

**Label-access distinction (report explicitly):** GIN and MolE-RAG both see the *same* training
labels. GIN internalizes them through **parameter updates** (gradient descent on the training set);
MolE-RAG accesses them at inference through **retrieved analogues**. This baseline isolates how far a
standard supervised GNN gets with those labels and no language model.

## Method
- **Architecture:** GIN with edge-feature concatenation, 5 layers, 300-dim embeddings, atom-type +
  chirality input embeddings, BatchNorm + ReLU + dropout (no ReLU on the last layer), JK=`last`,
  mean graph pooling, linear head to `num_tasks`. Ported verbatim from pretrain-gnns `model.py`
  (`gin_model.py`), featurization verbatim from `loader.py` (`featurize.py`); only PyG-1.x→2.x API
  shims changed (see `gin_model.py` header). Random init (no `from_pretrained`).
- **One model per dataset.** Multitask datasets (clintox=2, tox21=12, sider=27 endpoints) use
  **multiple output heads**; missing labels are **masked** in the loss and in evaluation (long-format
  splits are pivoted to a per-molecule label vector + observation mask).
- **Loss:** masked BCE-with-logits (classification) / masked MSE on standardized targets (regression;
  train mean/std, inverted at evaluation).
- **Optimizer:** Adam, weight decay 0, dropout 0.5. **HP grid:** lr ∈ {1e-3, 1e-4}.
- **Model selection:** best **checkpoint** chosen by validation metric each epoch (classification →
  macro val ROC-AUC; regression → val RMSE); **HP** chosen by mean validation metric across seeds.
- **Repetition:** ≥3 initialization seeds (0,1,2); report **test mean ± std**.
- **Test:** exactly the MolE-RAG seed-0 **test** molecules. Classification → macro ROC-AUC
  (endpoints with both classes); regression → RMSE (+ MAE).

## Results (seed-0 split, 3 init seeds, mean ± std)
_Pending the GPU run (`jobs/run_gin.sbatch`). `seed_0/summary.csv` + per-task `seed_0/<task>/results.json` on completion._

## Contents
- `gin_model.py` — GIN (GINConv/GNN/GNN_graphpred), pretrain-gnns port to PyG 2.x.
- `featurize.py` — `mol_to_graph_data_obj_simple` + `allowable_features` (verbatim).
- `train_gin.py` — training/eval harness (data pivot + masking, HP×seed search, best-val checkpoint, test mean±std).
- `jobs/run_gin.sbatch` — Delta GPU job (A100, account `bhrm-delta-gpu`).
- `seed_0/` — `config.json`, `summary.csv`, per-task `results.json` + `invalid_smiles.csv`.

## Reproduce
```bash
# env: /workspace/envs/gin  (torch 2.4.1+cu121, torch_geometric 2.6.1, rdkit)
sbatch /workspace/mole-rag_naacl/gin/jobs/run_gin.sbatch
# or directly (from this dir, GPU node):
cd /workspace/mole-rag_naacl/gin
/workspace/envs/gin/bin/python train_gin.py --datasets bbbp --seeds 0 1 2
```

## Notes
- Reference code cloned to `MolE-RAG/external/pretrain-gnns/` (snap-stanford, master).
- clintox uses both endpoints (CT_TOX + FDA_APPROVED), consistent with Morgan-KNN here.
- Large-dataset train budgets (batch/epochs) are set per dataset in `train_gin.py:DATASET_OVERRIDES`
  (hiv: batch 256 / 60 epochs).
