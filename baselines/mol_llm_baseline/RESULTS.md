# MolLM (Gerstein Lab, 2D+3D) — MoleculeNet (MolE-RAG seed-0 splits)

Released **MolLM** unified model (Tang/Tran/Tan/Gerstein, Bioinformatics 2024): a molecular
Transformer encoder over OGB 2D graph features **and** RDKit ETKDG 3D conformers, pretrained
by contrastive learning against biomedical text. The released pretrained graph-encoder
checkpoint (`all_checkpoints/model-epoch=394.ckpt`) is **fine-tuned** end-to-end with a fresh
prediction head on the MolE-RAG seed-0 **train** split, best checkpoint by **validation**,
evaluated on the MolE-RAG seed-0 **test** molecules (same test molecules as the other
baselines). One seed (0). Multitask datasets (clintox/tox21/sider): one model with a
num_targets-wide head and **masked BCE** (missing labels masked from loss AND metric); metric
= mean ROC-AUC over endpoints. Regression uses train-set target standardisation; metric =
RMSE. Ran on SageMaker (1 L40S GPU, PyTorch DLC).

| task | type | metric | MolLM |
|---|---|---|---|
| bbbp | classification | AUC | 0.948 |
| bace | classification | AUC | 0.835 |
| hiv | classification | AUC | 0.478 |
| clintox | classification | AUC | 0.840 |
| sider | classification | AUC | 0.604 |
| tox21 | classification | AUC | 0.795 |
| esol | regression | RMSE | 0.895 |
| freesolv | regression | RMSE | 2.094 |
| lipo | regression | RMSE | 0.696 |

Per-task config (val score, best epoch, epochs run, #test) in `analysis/mol_llm_results.json`;
live per-run JSON in s3://molerag/mol_llm_baseline/output/incremental/.
Total SageMaker BillableTimeInSeconds: 2723.
