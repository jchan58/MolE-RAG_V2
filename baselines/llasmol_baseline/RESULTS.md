# LlaSMol (Mistral-7B) — MoleculeNet (MolE-RAG seed-0 splits)

`osunlp/LlaSMol-Mistral-7B` is a **LoRA adapter** (`modules_to_save=[]`) on the openly available
base `mistralai/Mistral-7B-v0.1`, instruction-tuned on **SMolInstruct**. This is **released-checkpoint
inference** — the adapter is merged onto the base and run with LlaSMol's **own** property-prediction
generation config (`num_beams=4`, 1 return sequence, `max_new_tokens=20`, deterministic). **No
fine-tuning, no seeds.** Evaluated on the MolE-RAG seed-0 **test** molecules (same as the other baselines).

Prompts are SMolInstruct's **exact verbatim templates** (bundled `smol_templates/`): each test molecule
gets a template by cycling the task's full template set (`index mod n_templates`), the SMILES wrapped in
the dataset `<SMILES> … </SMILES>` core tag and canonicalized by LlaSMol's own
`canonicalize_molecule_smiles`, chat-wrapped as `<s>[INST] … [/INST]`, answers extracted with the repo's
`TASK_TAGS` (`<NUMBER>`/`<BOOLEAN>`).

**Coverage:** SMolInstruct's property-prediction subset has only **6 of the 9** MoleculeNet tasks.
FreeSolv, BACE, Tox21 are **not in SMolInstruct** → not runnable faithfully (no fabricated numbers).
`clintox` is scored against the **CT_TOX** endpoint (LlaSMol asks "Is X toxic?"; verified 16/16 on
overlapping molecules). `sider` is the **mean AUC over the 20 of 27** SIDER endpoints SMolInstruct has
templates for (the other 7 organ systems have no LlaSMol template and are masked).

**Classification ROC-AUC (headline)** = probability of the positive answer `P(Yes)` via the sequence
log-likelihood of the two canonical `<BOOLEAN>` responses, softmax, AUC per endpoint then averaged
(suite-comparable). LlaSMol's **native hard-label** AUC/accuracy (`calculate_boolean_metrics`, copied
verbatim) is in `analysis/llasmol_results.json`. **Regression RMSE (headline)** parses `<NUMBER>` and
imputes the train-target mean for unparseable outputs over all test molecules; native RMSE (drops
unparseable) + counts also recorded.

> **Leakage caveat:** LlaSMol was instruction-tuned on these MoleculeNet PP datasets, and **74–92% of our
> scaffold-split test molecules appear in SMolInstruct's train/dev**. These numbers are an as-released
> model applied to our test molecules, **not** held-out generalization.

| task | type | metric | value |
|---|---|---|---|
| bbbp | classification | ROC-AUC | 0.863 |
| bace | classification | — | not supported (not in SMolInstruct property-prediction subset) |
| hiv | classification | ROC-AUC | 0.669 |
| clintox | classification | ROC-AUC | 0.713 |
| sider | classification | ROC-AUC | 0.557 |
| tox21 | classification | — | not supported (not in SMolInstruct property-prediction subset) |
| esol | regression | RMSE | 1.287 |
| freesolv | regression | — | not supported (not in SMolInstruct (its PP regression subset = ESOL + Lipophilicity only)) |
| lipo | regression | RMSE | 1.028 |

Total SageMaker BillableTimeInSeconds: 7005 (~1.9 GPU-hours).
Per-task config, per-endpoint AUCs, native hard-label metrics, no-answer/unparseable counts, and the
exact prompt provenance are in `analysis/llasmol_results.json`; raw per-unit model outputs are in the
SageMaker model artifact (`s3://molerag/llasmol_baseline/output/<job>/output/model.tar.gz`, `runs/*.jsonl`)
and live per-task JSON in `s3://molerag/llasmol_baseline/output/incremental/`.
