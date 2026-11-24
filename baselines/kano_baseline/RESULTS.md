# KANO (CMPNN) baseline on 9 MoleculeNet test sets

**Model.** KANO (Fang et al., *Nat. Mach. Intell.* 2023) — knowledge-graph-enhanced molecular **contrastive pretraining** with a **functional prompt**, CMPNN encoder. Finetuned from the released pretrained encoder (`dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl`) on each task.

**Protocol.** Finetune on our MoleculeNet scaffold-split **seed_0** train, early-stop on valid, evaluate on test — the *same* splits as the other baselines in this suite (fed via chemprop's `--separate_val_path/--separate_test_path`, so the test molecules are identical). 100 epochs, batch 256, init_lr 1e-4, `--step functional_prompt`, **1 seed** per task. Multitask sets (tox21/sider/clintox) use chemprop's per-endpoint-mean AUC (macro). Ran on AWS SageMaker (PyTorch 2.1 DLC, 1× A10G).

## Results
| task | type | metric | KANO-CMPNN |
|---|---|---|---|
| bbbp | classification | ROC-AUC ↑ | **0.9606** |
| bace | classification | ROC-AUC ↑ | **0.8318** |
| hiv | classification | ROC-AUC ↑ | **0.8048** |
| clintox | classification | ROC-AUC ↑ | **0.9256** |
| sider | classification | ROC-AUC ↑ | **0.6408** |
| tox21 | classification | ROC-AUC ↑ | **0.8085** |
| esol | regression | RMSE ↓ | **0.7080** |
| freesolv | regression | RMSE ↓ | **1.8782** |
| lipo | regression | RMSE ↓ | **0.6093** |

## Notes
- These are strong, genuine GNN-baseline numbers — far above the LLM/RAG baselines in this suite (e.g. MolRAG/chemrag), as expected for a pretrained graph model, and consistent with KANO's reported scaffold-split performance (KANO is among the strongest MoleculeNet pretrained models).
- **1 seed** per task (per the run plan); KANO's paper averages multiple seeds, so small differences vs the paper are expected. hiv (32,896 train) took ~4 h at 100 epochs on one A10G; it is the long pole.
- Splits are this suite's `seed_0` scaffold splits (identical test molecules across baselines), not KANO's own split, so numbers are directly comparable *within this suite* and approximately comparable to the paper.

## Artifacts
`analysis/results_cmpnn.json` (metrics), `sm_out_cmpnn/{results.json,<task>.log}` (full per-task chemprop finetune logs), `run_kano.sbatch`/`sm_entry.py`/`sm_submit_kano.sh` (Delta + SageMaker runners), `data/<task>/{train,val,test}.csv` (wide CSVs from our seed_0 long-format splits). SageMaker job `kano-cmpnn-20261011-175316`.
