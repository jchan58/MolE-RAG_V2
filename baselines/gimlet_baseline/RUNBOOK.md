# GIMLET (pretrained, zero-shot) baseline - RUNBOOK

Reproduces zero-shot GIMLET predictions on OUR MoleculeNet scaffold-split **test** sets,
scored as ROC-AUC (classification) / RMSE (regression). Weights FROZEN (no fine-tuning).

## Environment (NCSA Delta)
Conda env at `/workspace/envs/gimlet` (Python 3.7.16). Built to match GIMLET's
README (CUDA 11.1 / torch 1.9). Exact pinned versions:

| package | version |
|---|---|
| python | 3.7.16 |
| torch | 1.9.0+cu111 |
| torch-scatter | 2.0.9 |
| torch-sparse | 0.6.12 |
| torch-cluster | 1.5.9 |
| torch-spline-conv | 1.2.1 |
| torch-geometric | 1.7.2 |
| transformers | 4.28.1 |
| huggingface_hub | 0.14.1 |
| tokenizers | 0.13.3 |
| ogb | 1.3.3 |
| rdkit-pypi | 2021.9.3 |
| numpy | 1.21.6 |
| scipy | 1.7.3 |
| Cython | 0.29.36 |
| scikit-learn | 1.0.2 |
| pandas | 1.3.5 |
| datasets | 2.11.0 |

Build script: `code/../logs/install.log` was produced by
`scratchpad/install_gimlet.sh` (pyg wheels from
`https://data.pyg.org/whl/torch-1.9.0+cu111.html`). pip cache:
`PIP_CACHE_DIR=/workspace/.pip-cache`.

GPUs: A100 (sm_80) via SLURM partition `gpuA100x4`, account `bhrm-delta-gpu`.
cu111 supports sm_80. The login node has no GPU and is heavily shared/slow, so all
GIMLET work (incl. the smoke test) runs on the GPU compute node inside the SLURM job.

## Checkpoint + tokenizer (pre-cached; compute nodes are offline)
- Model: HF `haitengzhao/gimlet` (frozen pretrained; `pytorch_model.bin` ~253 MB).
- Tokenizer: `t5-small`.
- Both pre-downloaded on the login node to `HF_HOME=/workspace/models`.
- The job sets `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (no network on compute nodes).

## Instructions / prompts
GIMLET's own per-task instruction text was extracted **verbatim** from the HF dataset
`haitengzhao/molecule_property_instruction` (test split) into
`code/gimlet_prompts.json` - this is the prompt set the paper's zero-shot
`--use_huggingface_pipeline` eval uses (it differs slightly from the repo's
`prompts/selected_prompt_downstream_task.json` for some tasks; the HF text is authoritative).
- Native in GIMLET: bbbp, bace, hiv, tox21, esol, freesolv, lipo.
- tox21: the 12 endpoints are matched to GIMLET's 12 receptor/pathway prompts by the assay
  each prompt describes. The two ER prompts (task_index 0,1) are identical; the two AR prompts
  differ - task_index 4 ("On the MDA cell...") -> NR-AR, task_index 5 -> NR-AR-LBD.
  Mapping is in `code/gimlet_predict.py::TOX21_ENDPOINT2TI`.
- **sider, clintox: NOT defined in GIMLET.** `gimlet_predict.py` uses adapted GIMLET-style
  instructions (property description + Yes/No question). Flagged in RESULTS.md; out-of-distribution.

## Method (faithful zero-shot)
- Reuses GIMLET's graph construction (`ogb.smiles2graph` + graphormer transform,
  `rich_features` off), the GIMLET collator, and `model.generate` (no training).
- Classification: `p_pos = softmax over {Yes,No}` first-token logits = `sigmoid(logit_Yes - logit_No)`
  (= GIMLET's own decoder score; AUC ranking identical to the repo's `logit_Yes - logit_No`).
- Regression: numeric value parsed from the generation with GIMLET's own regex;
  generation length = (label token length + 1), as in the repo's `eval_result`.

## Run
```bash
sbatch /workspace/mole-rag_naacl/gimlet_baseline/code/run_gimlet.sbatch
```
One SLURM job loops over all 9 datasets internally (do NOT submit 9 jobs - tight QOS limit).
It (1) smoke-tests env+checkpoint on a few bace/lipo molecules, (2) runs all 9 tasks
(resume-safe: existing rows in `outputs/<task>_preds.jsonl` are skipped), (3) scores to
`analysis/results.json`, (4) renders `RESULTS.md`.

Manual re-run of pieces (from any node with the env):
```bash
PY=/workspace/envs/gimlet/bin/python
C=/workspace/mole-rag_naacl/gimlet_baseline/code
HF_HOME=/workspace/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  $PY $C/gimlet_predict.py --tasks bace bbbp freesolv esol clintox lipo hiv sider tox21
$PY $C/gimlet_score.py && $PY $C/make_results_md.py
```

## Outputs
- `outputs/<task>_preds.jsonl` - per-row predictions (chemrag schema).
- `analysis/results.json` - per-task metrics.
- `RESULTS.md` - setup + AUC/RMSE tables, labeled "GIMLET (pretrained, zero-shot)".
- `code/` - adapter (`gimlet_predict.py`), scorer (`gimlet_score.py`),
  smoke test, prompt cache (`gimlet_prompts.json`), sbatch, this runbook.
