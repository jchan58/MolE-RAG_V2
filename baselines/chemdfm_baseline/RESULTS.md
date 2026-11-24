# ChemDFM (LLaMA-13B, zero-shot inference) — MoleculeNet (MolE-RAG seed-0 splits)

Released merged checkpoint **`OpenDFM/ChemDFM-v1.0-13B`** (ChemDFM, Zhao et al., *Cell Rep.
Phys. Sci.* 2025; built on LLaMA-1-13B) run **zero-shot** (fp16, single greedy pass, no
fine-tuning, no seeds) on the MolE-RAG seed-0 **test** molecules. Prompts for
**bbbp / bace / hiv / clintox / tox21** are the EXACT ChemLLMBench property-prediction prompts
released in the ChemDFM GitHub repo (verbatim), wrapped in ChemDFM's `[Round 0]\nHuman: ...
\nAssistant:` dialogue format. **sider** (27 ep) and the regression tasks **esol / freesolv /
lipo** have NO ChemDFM-released prompt, so they use ChemDFM-STYLE wording (same instruction
shell) and are flagged accordingly. SMILES canonicalized with rdkit per the repo README.
Classification ROC-AUC = probability of the positive answer token (softmax of the first
generated-step {Yes,No} logits), mean over endpoints (missing labels masked). Regression =
parse the generated number, impute train-mean if unparseable, RMSE. Ran on SageMaker
(L40S 48GB, PyTorch 2.3 DLC + transformers 4.44.2).

| task | type | metric | value | prompt |
|---|---|---|---|---|
| bbbp | classification | AUC | 0.834 | ChemDFM released |
| bace | classification | AUC | 0.791 | ChemDFM released |
| hiv | classification | AUC | 0.788 | ChemDFM released |
| clintox | classification | AUC | 0.607 | ChemDFM released |
| sider | classification | AUC | 0.479 | ours (ChemDFM-style) |
| tox21 | classification | AUC | 0.815 | ChemDFM released |
| esol | regression | RMSE | 5.921 | ours (ChemDFM-style) |
| freesolv | regression | RMSE | 75.343 | ours (ChemDFM-style) |
| lipo | regression | RMSE | 1.994 | ours (ChemDFM-style) |

Total SageMaker BillableTimeInSeconds across jobs: **1884** (~0.52 GPU-hours).

Per-task detail (per-endpoint AUC, accuracy, unparseable counts, exact prompt provenance) in
`analysis/chemdfm_results.json`; live per-task JSON in
s3://molerag/chemdfm_baseline/output/incremental/ and raw run outputs in each job's
model.tar.gz (s3://molerag/chemdfm_baseline/output/<job>/output/model.tar.gz).
