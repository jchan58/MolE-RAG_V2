# BestFP-KNN baseline (MolE-RAG per-task best fingerprint) — seed 0

| Dataset | Fingerprint | Task type | k | Test metric | Value | Coverage | Runtime (s) |
|---|---|---|---|---|---|---|---|
| bbbp | ECFP2 | classification | 3 | macro_roc_auc | 0.8777 | 1.000 | 1.7 |
| bace | Topological Torsion | classification | 5 | macro_roc_auc | 0.8033 | 1.000 | 1.8 |
| clintox | FCFP2 | classification | 5 | macro_roc_auc | 0.6556 | 1.000 | 2.1 |
| hiv | RDKit | classification | 5 | macro_roc_auc | 0.7876 | 1.000 | 176.8 |
| tox21 | Atom Pair | classification | 5 | macro_roc_auc | 0.7337 | 1.000 | 42.2 |
| sider | FCFP2 | classification | 5 | macro_roc_auc | 0.6119 | 1.000 | 16.3 |
| esol | Atom Pair | regression | 5 | rmse | 1.0033 | 1.000 | 0.9 |
| freesolv | MACCS | regression | 3 | rmse | 2.7960 | 1.000 | 0.5 |
| lipo | Atom Pair | regression | 5 | rmse | 0.9362 | 1.000 | 4.5 |
