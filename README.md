<div align="center">

# MolE-RAG

**Molecular Structure-Enhanced Retrieval-Augmented Generation for chemistry**

A training-free method that lets a general-purpose LLM predict molecular
properties by retrieving labeled reference molecules and literature evidence —
no model weights are trained.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![arXiv](https://img.shields.io/badge/arXiv-2606.05693-b31b1b.svg)](https://arxiv.org/abs/2606.05693)
[![Benchmarks](https://img.shields.io/badge/MoleculeNet-9%20tasks-success.svg)](#benchmarks)

</div>

---

## Contents

- [Overview](#overview)
- [Results](#results)
- [Installation](#installation)
- [Configure the models](#configure-the-models)
- [Quickstart](#quickstart)
- [Exact reproduction](#exact-reproduction)
- [Benchmarks](#benchmarks)
- [Baselines](#baselines)
- [Repository layout](#repository-layout)
- [Reproducibility notes](#reproducibility-notes)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

For a query molecule, MolE-RAG:

1. builds a molecular context from cached identifiers and computed RDKit descriptors;
2. retrieves the most similar **labeled training molecules** by fingerprint similarity;
3. lets the LLM **acquire literature evidence** from a chemistry *card corpus*; and
4. predicts the property.

Prediction proceeds in three arms, all saved per molecule:

| Arm | Name | What it is |
|---|---|---|
| **A** | Initial | first answer from structure alone |
| **B** | Self-review | the model critiques and may revise A |
| **C** | Evidence-assisted | the reported **+MolE-RAG** result — asks questions, retrieves evidence, re-assesses |

This repository is a **reproduction package**: the code, data splits, prompts,
card corpus, baselines, and saved scores needed to re-run the method and obtain
the paper's numbers on nine MoleculeNet benchmarks.

---

## Results

SMILES-only baseline vs. the full method (**arm C, +MolE-RAG**). Classification
is ROC-AUC (higher is better); regression is RMSE (lower is better). Retrieving
reference molecules and evidence improves most cells; a few are flat or slightly
worse (see [reproducibility notes](#reproducibility-notes) and the paper).

**Qwen3-32B**

| Dataset | Metric | SMILES-only | +MolE-RAG |
|---|---|--:|--:|
| BBBP | AUC ↑ | 0.550 | **0.902** |
| BACE | AUC ↑ | 0.501 | **0.778** |
| ClinTox | AUC ↑ | 0.427 | 0.426 |
| HIV | AUC ↑ | 0.474 | **0.769** |
| Tox21 | AUC ↑ | 0.528 | **0.757** |
| SIDER | AUC ↑ | 0.511 | **0.704** |
| ESOL | RMSE ↓ | 1.959 | **1.012** |
| FreeSolv | RMSE ↓ | 4.680 | **3.426** |
| Lipophilicity | RMSE ↓ | 1.286 | **0.963** |

<details>
<summary><b>Ministral-3-14B</b></summary>

| Dataset | Metric | SMILES-only | +MolE-RAG |
|---|---|--:|--:|
| BBBP | AUC ↑ | 0.369 | **0.773** |
| BACE | AUC ↑ | 0.493 | **0.718** |
| ClinTox | AUC ↑ | 0.559 | 0.518 |
| HIV | AUC ↑ | 0.585 | **0.758** |
| Tox21 | AUC ↑ | 0.503 | **0.747** |
| SIDER | AUC ↑ | 0.504 | **0.601** |
| ESOL | RMSE ↓ | 3.806 | **1.053** |
| FreeSolv | RMSE ↓ | 4.449 | **2.909** |
| Lipophilicity | RMSE ↓ | 1.127 | **0.946** |

</details>

<details>
<summary><b>Claude Opus 4.6</b></summary>

| Dataset | Metric | SMILES-only | +MolE-RAG |
|---|---|--:|--:|
| BBBP | AUC ↑ | 0.872 | **0.912** |
| BACE | AUC ↑ | 0.729 | **0.819** |
| ClinTox | AUC ↑ | 0.521 | 0.472 |
| HIV | AUC ↑ | 0.730 | **0.826** |
| Tox21 | AUC ↑ | 0.767 | **0.812** |
| SIDER | AUC ↑ | 0.698 | 0.678 |
| ESOL | RMSE ↓ | 0.901 | **0.652** |
| FreeSolv | RMSE ↓ | 1.299 | 1.585 |
| Lipophilicity | RMSE ↓ | 0.956 | **0.718** |

</details>

<details>
<summary><b>GPT-5-mini</b></summary>

| Dataset | Metric | SMILES-only | +MolE-RAG |
|---|---|--:|--:|
| BBBP | AUC ↑ | 0.819 | **0.880** |
| BACE | AUC ↑ | 0.606 | **0.806** |
| ClinTox | AUC ↑ | 0.381 | **0.538** |
| HIV | AUC ↑ | 0.565 | **0.796** |
| Tox21 | AUC ↑ | 0.704 | **0.802** |
| SIDER | AUC ↑ | 0.575 | **0.685** |
| ESOL | RMSE ↓ | 1.122 | **0.977** |
| FreeSolv | RMSE ↓ | 7.245 | **2.802** |
| Lipophilicity | RMSE ↓ | 1.432 | **0.869** |

</details>

All reference numbers (every arm, with paired bootstrap CIs) live in
[`scores/`](scores/). For bit-exact reproduction see [below](#exact-reproduction).

---

## Installation

```bash
git clone https://github.com/jchan58/MolE-RAG_V2.git
cd MolE-RAG_V2
git lfs pull                      # fetch the card corpus (tracked via git-LFS)
pip install -r requirements.txt
```

---

## Configure the models

The framework talks to models through an **OpenAI-compatible API**, so you can
point it at any provider — OpenAI, a local vLLM/TGI server, or a cloud gateway.
Choose models in [`models.json`](models.json): per model a `base_url`, the
provider's `model` id, and the env var holding its key.

```bash
cp .env.example .env     # add the API key(s) named in models.json
$EDITOR models.json      # set base_url for each model you'll run
```

To reproduce the paper, keep the shipped model ids and only set their endpoints:

| Name in repo | Model | Role |
|---|---|---|
| `qwen3_32b` | Qwen3-32B | predictor |
| `ministral_3_14b` | Ministral-3-14B | predictor |
| `opus_4_6` | Claude Opus 4.6 | predictor |
| `gpt5mini` | GPT-5-mini | predictor |
| `embedder` | Cohere embed-multilingual-v3 | evidence retrieval |

---

## Quickstart

Everything runs per `(dataset, model)` and is resumable (one JSON per molecule).

```bash
# 1. predict -> runs/<dataset>__<model>/test_XXXX.json  (arms A / B / C)
python framework/run_fw_multi.py bbbp qwen3_32b

# 2. score -> scores/SCORES_fw_multi.json
#    ROC-AUC (classification) / RMSE (regression), arms A, B, C, paired bootstrap
python framework/score_fw_multi.py bbbp qwen3_32b
```

The reported **+MolE-RAG** number is arm **C**. Other entry points:

| Script | What it runs |
|---|---|
| `framework/run_fw_multi.py <ds> <model>` | the framework on bbbp/esol/freesolv/lipo/sider/clintox |
| `framework/run_hiv_tox21.py` | sharded, self-healing runner for hiv + tox21 |
| `framework/bace/run_final.py <split> <W> <model>` | the framework on BACE (uses `data/bace_pool/`) |
| `framework/smiles_only/run_so_tasks.py <model> <W> <tasks>` | the SMILES-only baseline (Table 2 left column) |
| `framework/score_fw_multi.py [ds] [model]` | scorer; robust/parallel variants in `framework/scoring/` |

Paths are configurable via env (`MOLERAG_DATA`, `MOLERAG_CARDS`, `MOLERAG_RUNS`); defaults are repo-relative.

---

## Exact reproduction

Because the method calls LLMs, a fresh run matches the published cells closely
but not bit-for-bit. For **exact** reproduction — and to inspect what the model
did for every molecule (each question asked, every passage retrieved with its
source PMCID and similarity, and how its probability moved round by round) — we
release the **saved per-molecule prediction files** from the runs that produced
the paper's numbers:

```bash
scripts/fetch_predictions.sh                        # downloads + extracts to predictions/
export MOLERAG_RUNS=$PWD/predictions/runs
python framework/score_fw_multi.py bbbp qwen3_32b    # re-scores -> exact paper cell
```

Re-scoring is deterministic (0 LLM noise). See
[`docs/PREDICTIONS.md`](docs/PREDICTIONS.md) for the full per-molecule record
schema and the per-table scoring recipe.

---

## Benchmarks

Single fixed scaffold split (seed 0), committed under
[`data/moleculenet/seed_0`](data/moleculenet/seed_0).

| Dataset | Task | Metric | #Test |
|---|---|---|--:|
| BBBP | classification | ROC-AUC | 205 |
| BACE | classification | ROC-AUC | 152 |
| ClinTox | classification (2 endpoints) | ROC-AUC | 149 |
| HIV | classification | ROC-AUC | 4112 |
| Tox21 | classification (12 endpoints) | ROC-AUC | 783 |
| SIDER | classification (27 endpoints) | ROC-AUC | 144 |
| ESOL | regression | RMSE | 114 |
| FreeSolv | regression | RMSE | 65 |
| Lipophilicity | regression | RMSE | 420 |

The literature **card corpus** (the evidence arm C retrieves over) ships in
[`data/cards/`](data/cards) via git-LFS, one bundle per dataset
(`passages.json`, `findings.jsonl`, `entities.json`, `mol_map.json`,
`meta.json`), plus [`data/bace_pool/`](data/bace_pool) for BACE.

---

## Baselines

[`baselines/`](baselines) holds the Table-1 comparison methods, each with its own
README and `RESULTS.md`:

- **Retrieval on a shared backbone** — ChemRAG, MolRAG, and a no-LLM Morgan-fingerprint KNN.
- **Graph models** — GIN, DMPNN, Graphormer-p, GROVER, MolCLR, KANO-CMPNN.
- **Molecular LM / multimodal** — GIMLET, MolLM, nach0, BioT5, MolXPT, ChemDFM, LlaSMol, MolecularGPT.

Model weights are not vendored — each baseline's README says how to obtain or train them.

---

## Repository layout

```
framework/     the method: run_fw_multi (+ hiv/tox21, bace, smiles_only) + scorers + model_client
models.json    per-model endpoint + id + api-key env (edit this)
prompts/       released prompt templates + condition definitions
data/
  moleculenet/seed_0/   scaffold splits (committed)
  cards/                card corpus (git-LFS)
  bace_pool/            BACE retrieval pool (git-LFS)
baselines/     one directory per Table-1 method
corpus/        Depth-B pipeline to rebuild the card corpus from PMC
scores/        computed reference numbers to match
scripts/       helpers (fetch_predictions.sh)
docs/          methodology, card schema, predictions schema, reproducibility notes
```

---

## Reproducibility notes

- Single fixed scaffold split (seed 0), no seed averaging — as in the paper.
- Model calls are nondeterministic; a fresh run differs slightly from the
  published cells. Known cell-level caveats are in
  [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
- The card corpus can be rebuilt from scratch from PMC via [`corpus/`](corpus)
  (Depth B); see [`corpus/README.md`](corpus/README.md).

---

## Citation

```bibtex
@article{chan2026mole,
  title={MolE-RAG: Molecular Structure-Enhanced Retrieval-Augmented Generation for Chemistry},
  author={Chan, Joey and Kweon, Wonbin and Shin, Ashley and Bhattacharjee, Niharika and Jiang, Pengcheng and Guo, Yue and Han, Jiawei},
  journal={arXiv preprint arXiv:2606.05693},
  year={2026}
}
```

## Acknowledgments

This work was supported by the Molecule Maker Lab Institute (NSF No. 2019897).
Computation used Delta GPUs at NCSA through allocation CIS240504 from the ACCESS
program (NSF grants #2138259, #2138286, #2138307, #2137603, #2138296).

## License

MIT — see [`LICENSE`](LICENSE).
