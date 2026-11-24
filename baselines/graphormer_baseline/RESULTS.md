# Graphormer (PCQM4Mv2-pretrained, fine-tuned) — MoleculeNet (MolE-RAG seed-0 splits)

Graphormer-base initialized from Microsoft's **PCQM4Mv2** pretrained checkpoint (`checkpoint_best_pcqm4mv2.pt`), then supervised **fine-tuned per task** on the MolE-RAG
seed-0 **train** split (canonical-SMILES match), best checkpoint by **validation**, evaluated
on the MolE-RAG seed-0 **test** molecules. 3 initialization seeds (0,1,2), mean ± std.
Multitask datasets (clintox/tox21/sider) use one output head per endpoint; ROC-AUC is the
mean over endpoints. Ran on SageMaker (legacy PyTorch 1.9.1 DLC + reconstructed PyG 1.7.2 stack;
graphormer spatial encoding compiled in-container via Cython/pyximport).

| task | type | metric | mean ± std (3 seeds) |
|---|---|---|---|
| bbbp | classification | ROC-AUC | 0.886 ± 0.004 |
| bace | classification | ROC-AUC | 0.811 ± 0.044 |
| hiv | classification | ROC-AUC | 0.815 ± 0.007 |
| clintox | classification | ROC-AUC | 0.638 ± 0.045 |
| sider | classification | ROC-AUC | 0.573 ± 0.008 |
| tox21 | classification | ROC-AUC | 0.792 ± 0.004 |
| esol | regression | RMSE | 0.775 ± 0.039 |
| freesolv | regression | RMSE | 3.063 ± 0.069 |
| lipo | regression | RMSE | 0.723 ± 0.002 |

Total SageMaker BillableTimeInSeconds across jobs: **56819** (~15.8 GPU-hours).

Per-seed values + config in `analysis/graphormer_results.json`; raw per-run outputs in the
SageMaker model artifacts (s3://molerag/graphormer_baseline/output/<job>/output/model.tar.gz).
PCQM4Mv2 checkpoint staged at s3://molerag/graphormer_baseline/ckpt/checkpoint_best_pcqm4mv2.pt.
