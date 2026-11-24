# GIN (trained from scratch) — MoleculeNet (MolE-RAG seed-0 splits)

Supervised Graph Isomorphism Network (pretrain-gnns `chem/` architecture, **random init, no pretraining**),
trained on MolE-RAG seed-0 **train**, HP grid lr∈{1e-3,1e-4} selected by **validation**, best checkpoint by
val metric, evaluated on MolE-RAG seed-0 **test**. 3 init seeds (0,1,2), mean ± std. Multitask datasets use
multiple masked output heads. Ran on SageMaker (4 parallel GPU jobs, PyTorch 2.4 DLC + PyG 2.6.1).

| task | type | metric | mean ± std (3 seeds) | n_test |
|---|---|---|---|--:|
| bbbp | classification | ROC-AUC | 0.929 ± 0.002 | 197 |
| bace | classification | ROC-AUC | 0.749 ± 0.023 | 152 |
| hiv | classification | ROC-AUC | 0.790 ± 0.002 | 4111 |
| clintox | classification | ROC-AUC | 0.708 ± 0.031 | 148 |
| sider | classification | ROC-AUC | 0.590 ± 0.005 | 144 |
| tox21 | classification | ROC-AUC | 0.786 ± 0.001 | 783 |
| esol | regression | RMSE | 0.979 ± 0.064 | 111 |
| freesolv | regression | RMSE | 3.323 ± 0.297 | 65 |
| lipo | regression | RMSE | 0.759 ± 0.020 | 420 |

Per-seed values + selected HP in `seed_0/gin_results_aggregate.json` (+ per-task `seed_0/<task>_results.json`).
A few molecules dropped per task from invalid-SMILES/featurization (n_test vs full test count). Label-access
note: GIN internalizes training labels via gradient descent (contrast with MolE-RAG's inference-time retrieval).
