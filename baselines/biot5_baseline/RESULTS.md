# BioT5 (biot5-base, fine-tuned) — MoleculeNet (MolE-RAG seed-0 splits)

Pretrained **`QizhiPei/biot5-base`** (BioT5, Pei et al., NeurIPS 2023) **fine-tuned per (sub)task** on the MolE-RAG seed-0 **train** split, best checkpoint by **validation**, evaluated on the MolE-RAG seed-0 **test** molecules. **Single seed (0).**

Faithful to BioT5's `molnet` pipeline: molecule **SELFIES** input `Molecule: <bom>...<eom>` built with BioT5's exact Natural-Instructions template and its own per-dataset task **Definition** strings (from QizhiPei/BioT5_finetune_dataset); only the molecules/labels are swapped in. Model/tokenizer/data-collator/AdamW/cosine-schedule are BioT5's own `biot5/utils`. Classification **ROC-AUC** uses BioT5's exact scoring `logit(Yes)/(logit(Yes)+logit(No))` on the first generated token. Multitask datasets (clintox/tox21/sider) = one independent fine-tune per endpoint, score = **mean ROC-AUC over endpoints** whose test split has both classes.

Regression (esol/freesolv/lipo) is **not part of BioT5's molnet pipeline** (it only emits Yes/No); results marked `*` use a minimal added regression variant (model generates the train-standardized value as text, parsed and de-standardized; metric RMSE).

| task | type | metric | value (seed 0) |
|---|---|---|---|
| bbbp | classification | auroc | 0.910 |
| bace | classification | auroc | 0.819 |
| hiv | classification | auroc | 0.785 |
| clintox | classification | auroc | 0.715 (2/2 endpoints) |
| sider | classification | auroc | 0.598 (23/23 endpoints) |
| tox21 | classification | auroc | 0.754 (12/12 endpoints) |
| esol* | regression | rmse | 1.074 |
| freesolv* | regression | rmse | 3.763 |
| lipo* | regression | rmse | 1.056 |

Total SageMaker BillableTimeInSeconds across jobs: **67886** (~18.9 GPU-hours).

Per-endpoint values, SELFIES-skip counts, best-val step, and per-task config in `analysis/biot5_results.json`; raw per-run outputs in the SageMaker model artifacts (s3://molerag/biot5_baseline/output/<job>/output/model.tar.gz) and incremental records at s3://molerag/biot5_baseline/output/incremental/.
