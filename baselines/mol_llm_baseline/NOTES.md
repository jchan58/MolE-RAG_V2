# mol_llm_baseline

Baseline: **MolLM (Gerstein Lab, 2D+3D)** — Tang/Tran/Tan/Gerstein, Bioinformatics 2024
(github.com/gersteinlab/MolLM, commit b2ec6429). NOT the arXiv 2502.02810 "Mol-LLM".

Status: **COMPLETE** — all 9 MoleculeNet tasks finished on MolE-RAG seed-0 TEST molecules
(full test parity; see RESULTS.md + analysis/mol_llm_results.json).

What it is: released MolLM molecular graph encoder (Transformer-M over OGB 2D features + RDKit
ETKDG 3D conformers, contrastively pretrained vs biomedical text), initialised from the released
pretrained checkpoint (HF andrewt28/MolLM, all_checkpoints/model-epoch=394.ckpt, 230 encoder
tensors, 0 missing/unexpected), fine-tuned end-to-end with a fresh MLP head. Best checkpoint by
VALIDATION (the repo's finetune.py selects by TEST; we fixed that), evaluated on seed-0 TEST.
One seed (0). Multitask (clintox/tox21/sider): masked BCE + mean ROC-AUC. Regression: target
standardisation + RMSE.

Compute: SageMaker, 1x L40S (ml.g6e.16xlarge, PyTorch 1.9.1-gpu-py38 DLC). code in sm/.
Checkpoint staged to s3://molerag/mol_llm_baseline/ckpt/ ; per-task results (restart-safe) in
s3://molerag/mol_llm_baseline/output/incremental/.

Key engineering (old fairseq/Transformer-M stack on a modern GPU):
- Instantiate TransformerM directly + copied preprocess_item/collator_3d -> no torch_geometric/scatter.
- torch 1.9.1 runs on L40S (sm_89) EXCEPT the JIT fuser: disable nvFuser+TE+profiling fuser and
  monkeypatch the @torch.jit.script `gaussian` to eager (nvrtc cu111 can't target sm_89).
- Featurisation robust to RDKit EmbedMolecule/MMFF hangs, worker crashes, and FD exhaustion
  (workers return numpy, not torch tensors, to avoid file-descriptor tensor sharing); node-budget
  batching to avoid OOM on large molecules. No TEST molecules dropped.
