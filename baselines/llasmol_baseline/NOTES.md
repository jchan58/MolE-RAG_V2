# llasmol_baseline

Baseline: **LlaSMol (Mistral-7B)** — released-checkpoint INFERENCE (no fine-tuning, no seeds).
Evaluated on: MolE-RAG seed-0 scaffold-split TEST sets (same convention as the other baselines).

## What it is
`osunlp/LlaSMol-Mistral-7B` = LoRA adapter (`modules_to_save=[]`) on open base `mistralai/Mistral-7B-v0.1`
(neither gated), instruction-tuned on SMolInstruct. Merged (`merge_and_unload`) and run with LlaSMol's own
property-prediction generation config (`num_beams=4`, 1 return seq, `max_new_tokens=20`, deterministic).

## Coverage (SMolInstruct property-prediction subset = 6 of 9 tasks)
- Supported: esol, lipo (regression); bbbp, clintox, hiv, sider (classification).
- clintox → CT_TOX endpoint ("Is X toxic?", verified 16/16 vs our labels; it is the inverse of FDA_APPROVED).
- sider → per-organ question; SMolInstruct has templates for **20 of 27** endpoints; mean AUC over those 20.
- NOT supported (no SMolInstruct task; reported honestly, no numbers): **freesolv, bace, tox21**.

## Fidelity
- Prompts = SMolInstruct's EXACT verbatim templates (extracted from data.zip into code/LLM4Chem/smol_templates/
  {template,core_tag}/). Each molecule cycles the template set (index mod n). SMILES canonicalized with the
  repo's own canonicalize_molecule_smiles; chat `<s>[INST] .. [/INST]`; answers via config.TASK_TAGS.
- Scoring: classification headline AUC = P(Yes) via sequence-LL of the two <BOOLEAN> responses (per-endpoint,
  averaged); native hard-label metrics via verbatim calculate_boolean_metrics. Regression headline RMSE imputes
  TRAIN mean for unparseable (over all test mols); native RMSE drops unparseable.

## Leakage caveat
LlaSMol was tuned on these MoleculeNet PP datasets; 74-92% of our scaffold TEST molecules are in SMolInstruct
train/dev → NOT held-out generalization.

## Compute / run state
- Pool: ml.g5.12xlarge (quota 1) and ml.g5.24xlarge (quota **0**, unusable). So single-job serialization.
- At build time g5.12xlarge was held by `kano-cmpnn-20261011-175316` → ResourceLimitExceeded.
- Autonomous orchestrator `sm/orchestrate.py` (detached via setsid, logs sm/orchestrate.log): waits for slot →
  SMOKE job (8 units/task, validates stack) → on success FULL job → analysis/aggregate.py (writes RESULTS.md +
  analysis/llasmol_results.json). Polls SageMaker states only (no ps|grep self-match). Survives session drops.
- Manual fallback: `python sm/llasmol_launch.py "esol,lipo,bbbp,clintox,hiv,sider" ml.g5.12xlarge 0`
  then `python sm/watch.py` (or `python analysis/aggregate.py`).
- S3: s3://molerag/llasmol_baseline/ (output/, data/, code/, output/incremental[_smoke]/).

## Files
- code/LLM4Chem/ : cloned OSU-NLP-Group/LLM4Chem repo + naacl_entry.py + requirements.txt + smol_templates/.
- sm/ : llasmol_launch.py, orchestrate.py, watch.py, llasmol_jobs.json (job registry).
- analysis/ : aggregate.py → llasmol_results.json. RESULTS.md at baseline root.
