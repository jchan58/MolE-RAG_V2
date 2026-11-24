# InstructMol-GS (Vicuna-7B + GIN) — MoleculeNet (MolE-RAG seed-0 splits)

**STATUS: BLOCKED — no released checkpoint. No numbers reported (not fabricated).**

The task was *released-checkpoint inference*. After thorough recon, **no trained InstructMol
checkpoint is publicly available** — not InstructMol-GS, not InstructMol-G, not even the
stage-1 alignment projector. Only *component* weights (base Vicuna, the MoleculeSTM GIN
encoder initializer, SciBERT) and the *training* code are released. There is therefore
nothing to run for any task, so every row below is blocked.

| task | type | metric | value | status |
|---|---|---|---|---|
| bbbp | classification | ROC-AUC | — | blocked: checkpoint unreleased |
| bace | classification | ROC-AUC | — | blocked: checkpoint unreleased |
| hiv | classification | ROC-AUC | — | blocked: checkpoint unreleased |
| clintox | classification | ROC-AUC | — | blocked: unreleased + not in protocol |
| sider | classification | ROC-AUC | — | blocked: unreleased + not in protocol |
| tox21 | classification | ROC-AUC | — | blocked: unreleased + not in protocol |
| esol | regression | RMSE | — | blocked: unreleased + not in protocol |
| freesolv | regression | RMSE | — | blocked: unreleased + not in protocol |
| lipo | regression | RMSE | — | blocked: unreleased + not in protocol |

**Coverage: 0/9 runnable.** SageMaker jobs launched: **0**. Billable time: **0 s**.

## Why blocked

InstructMol (arXiv:2311.16208, COLING 2025; repo `github.com/IDEA-XL/InstructMol`, HEAD
`5b2aa484` dated 2024-12-02) is a LLaVA-style multimodal model: a 5-layer GIN graph encoder
initialized from **MoleculeSTM** + an mm-projector + **Vicuna-7B-v1.3**, trained in two stages
(330K-pair alignment pretraining, then MoleculeNet LoRA instruction tuning). The runnable
model = base Vicuna **+ trained projector + trained task LoRA**.

The repo's **`### InstructMol Weights`** section reads verbatim: **`* TODO: coming soon`**.
The learned parts (projector + LoRA) are not published anywhere:

- HF org `IDEA-XL`: only `progen2-large` — no InstructMol.
- HF model search `instructmol`: **0 models**. Author accounts `hcaoaf` / `hcao`: 0 models.
- Project page `idea-xl.github.io/InstructMol`: links only the code and a *dataset*
  (`hcaoaf/PubChemSFT`) — no checkpoint.
- Repo grep for `huggingface.co` / `drive.google` / `*.pth` download URLs: only the
  **component-weight** URLs below, no trained-model URL.

What *is* released (component weights, not a usable model on their own):
`lmsys/vicuna-7b-v1.3` (open, not gated), the MoleculeSTM GIN encoder
(`chao1224/MoleculeSTM/.../molecule_model.pth`, **init only**), `allenai/scibert_scivocab_uncased`
(only for the molecule-captioning Text2Mol metric), and optional GraphMVP.

The eval script `llava/eval/molecule_metrics/MoleculeNet_classification.py` requires
`--model-path .../MoleculeNet-llava-moleculestm-vicuna-v1-3-7b-finetune_lora` and
`load_pretrained_model` loads a LoRA adapter + `non_lora_trainables.bin` (the trained projector).
With no such checkpoint, loading fails; forcing an untrained random projector + no LoRA would
not be InstructMol and would produce meaningless text — i.e. a fabricated baseline — so it was
**not** done.

## Coverage ceiling even if a checkpoint were released

InstructMol's own MoleculeNet protocol covers only **3 of our 9** tasks:
`SUPPORT_DATASETS = ["bace", "bbbp", "hiv"]`. clintox / sider / tox21 are not in it. Its
"regression" eval predicts **Mol-Instructions quantum properties (HOMO / LUMO / HOMO-LUMO gap)**,
not MoleculeNet ESOL / FreeSolv / Lipo, so our 3 regression tasks are out of scope too.

## Prompt / scoring provenance (from the repo, for the record)

- **Prompt (InstructMol-GS):** LLaVA `llava_v1` conversation with a graph token prepended
  (`<image>\n` + instruction), then the `-GS` suffix appended:
  `" The compound SELFIES sequence is: {selfies}"` (SELFIES encoded from SMILES).
- **Per-molecule instruction text** (e.g. the BBBP/BACE/HIV question) lives in `raw['instruction']`
  inside preprocessed `.pkl` files under a **private** path (`/cto_labs/AIDD/DATA/MoleculeNet`);
  those files and their preprocessing are **not** in the repo, so the exact instruction wording is
  itself not reproducible from released material.
- **Scoring:** hard-label only. The model **generates free text**; `active/yes/true → 1`,
  `inactive/no/false → 0`; `roc_auc_score` is then computed on the **0/1 hard labels**
  (not positive-token probability). Generation is stochastic (`do_sample=True`, `temperature=0.2`).

Full details, per-task status, and the paper's reference numbers (BBBP/BACE/HIV, *not* on our
splits, *not* entered above) are in `analysis/instructmol_gs_results.json`. Upstream README,
Evaluation.md, and the two eval/dataset scripts are preserved under `code/UPSTREAM_*`.

## To unblock

Either (a) wait for IDEA-XL to publish the InstructMol-GS checkpoint, or (b) reproduce the
two-stage training (alignment pretraining + MoleculeNet LoRA instruction tuning) — a multi-GPU
training effort outside the "released-checkpoint inference" scope of this baseline — and then run
the inference, which would still cover only bbbp/bace/hiv.
