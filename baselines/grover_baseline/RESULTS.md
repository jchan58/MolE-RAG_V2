# GROVER-Large (GTransformer) baseline on 9 MoleculeNet test sets

**Model.** GROVER-Large (Rong et al., NeurIPS 2020) — the self-supervised **GTransformer** graph encoder, finetuned with rdkit_2d_normalized features. Finetuned from Tencent's public `grover_large.pt` (via GODE's `m_level` GROVER fork) on our seed_0 scaffold splits (separate val/test = exact partition), 20 epochs, batch 32, init_lr 1.5e-4, ffn_hidden 200, ensemble 1 / 1 fold. Ran on SageMaker (PyTorch 1.13 DLC, g5.12xlarge; tasks parallelized across 4 A10G GPUs).

## Classification (ROC-AUC ↑)
| task | n_test | GROVER-L (ours, seed_0) | GROVER paper (reported) |
|---|--:|--:|--:|
| bbbp | 205 | **0.8880** | 0.940 |
| bace | 152 | **0.7164** | 0.894 |
| hiv | 4112 | **0.7414** | 0.800 |
| clintox | 149 | **0.6408** | 0.944 |
| sider | 144 | **0.5566** | 0.658 |
| tox21 | 783 | **0.7918** | 0.831 |

## Regression (RMSE ↓)
| task | n_test | GROVER-L (ours, seed_0) | GROVER paper (reported) |
|---|--:|--:|--:|
| esol | 114 | **1.1756** | 0.831 |
| freesolv | 65 | **2.5288** | 1.544 |
| lipo | 420 | **0.9357** | 0.560 |

## Notes
- GROVER's graph encoder = the GTransformer (dual node/edge message passing + transformer), the 'GTrans' backbone. Finetuned WITH rdkit_2d_normalized molecular features (descriptastorus), per GROVER's standard protocol.
- Multitask AUC = chemprop/GROVER per-endpoint mean (tox21 12, sider 27, clintox 2).
- 'GROVER paper (reported)' = GROVER-large published scaffold-split means (Rong et al. 2020); their splits differ from seed_0, so a reference point not an exact match. Our bbbp (0.888) matched the independent smoke-test run (0.891).
- Pretrained encoder loaded cleanly from the public `grover_large.pt` (all GTransformer params). 20-epoch finetune, single fold.

## Artifacts
`analysis_results_grover.json`, `sm_out_full/{results.json,<task>.log}`, `grover_large.pt` (public ckpt), `sm_entry_grover.py`/`sm_submit_grover.sh`/`requirements.txt`. Reuses the seed_0 wide CSVs from `../kano_baseline/data`.