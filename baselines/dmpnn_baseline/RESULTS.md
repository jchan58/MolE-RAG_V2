# DMPNN (trained from scratch) — MoleculeNet (MolE-RAG seed-0 splits)

Chemprop v2 D-MPNN, random init (no pretraining), trained on the MolE-RAG seed-0 **train** split,
best checkpoint by **validation**, evaluated on the MolE-RAG seed-0 **test** molecules.
3 initialization seeds (0,1,2), mean ± std. Multitask datasets (clintox/tox21/sider) use multiple
output heads with masked loss. Ran on SageMaker (4 parallel GPU jobs, PyTorch 2.3 DLC + chemprop 2.3.1).

| task | type | metric | mean ± std (3 seeds) |
|---|---|---|---|
| bbbp | classification | ROC-AUC | 0.931 ± 0.004 |
| bace | classification | ROC-AUC | 0.781 ± 0.010 |
| hiv | classification | ROC-AUC | 0.793 ± 0.009 |
| clintox | classification | ROC-AUC | 0.833 ± 0.003 |
| sider | classification | ROC-AUC | 0.580 ± 0.010 |
| tox21 | classification | ROC-AUC | 0.795 ± 0.003 |
| esol | regression | RMSE | 0.813 ± 0.012 |
| freesolv | regression | RMSE | 2.228 ± 0.251 |
| lipo | regression | RMSE | 0.630 ± 0.013 |

Per-seed values + config in `analysis/dmpnn_results.json`; raw per-run outputs in the SageMaker
model artifacts (s3://molerag/dmpnn_baseline/output/<job>/output/model.tar.gz). Label-access note: DMPNN
internalizes the training labels via gradient descent (contrast with MolE-RAG's retrieval at inference).
