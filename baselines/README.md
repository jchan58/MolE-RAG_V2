# Table-1 baselines

Each directory is a self-contained harness that produced one row of Table 1,
with a `RESULTS.md` reporting its per-dataset ROC-AUC (classification) / RMSE
(regression) on the seed-0 scaffold test sets, and the analysis JSON/CSV behind
those numbers. The three methods that share MolE-RAG's Qwen3-32B comparison are
here too (ChemRAG, MolRAG, and the no-LLM Morgan-fingerprint KNN).

## What's included

| Group | Baselines |
|---|---|
| Retrieval (shared backbone) | `chemrag_baseline`*, `molrag_baseline`, `morgan_knn` (KNN; `morgan_knn/k5/` is the paper's k=5 variant) |
| Graph models | `gin`, `dmpnn_baseline`, `graphormer_baseline`, `grover_baseline`, `molclr_baseline`, `kano_baseline` |
| Molecular language / multimodal | `gimlet_baseline`, `mol_llm_baseline`, `nach0_baseline`, `biot5_baseline`, `molxpt_baseline`, `chemdfm_baseline`, `llasmol_baseline`, `moleculargpt_baseline` |
| Extra (not in the paper table) | `unimol`, and the blocked attempts `instructmol_gs_baseline`, `mr_mol_baseline` (kept with their notes) |

\* ChemRAG was run on **Qwen3-4B-Instruct** (see `../docs/REPRODUCIBILITY.md`),
not Qwen3-32B.

Not vendored: `gode` (dropped) and the `kano` model source clone (its results
live in `kano_baseline`).

## Important: weights and paths

- **Model weights/checkpoints are not included** (GROVER and MolXPT alone are
  9 GB / 3.4 GB). Each baseline's `RESULTS.md` / scripts say how to obtain or
  train them — typically cloning the upstream model repo and downloading its
  released checkpoint.
- **The run scripts contain the original absolute paths** used on the authors'
  cluster (e.g. `/work/...`, SageMaker launch configs). They are reference
  implementations: point them at your own data/output locations before running.
  The MoleculeNet scaffold splits are in `../data/moleculenet/seed_0/`.

The point numbers reported in the paper are in each `RESULTS.md` and reproduce
deterministically from the committed analysis files; re-running a model from its
weights reproduces them up to that method's own nondeterminism.
