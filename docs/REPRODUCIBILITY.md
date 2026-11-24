# Reproducibility notes

## Splits and seeds
All results use a single fixed scaffold split (seed 0), 8:1:1 train/valid/test,
with no averaging over seeds — matching the paper. The split indices are
committed under `data/moleculenet/seed_0/`.

## Determinism
- **Scoring is deterministic.** Predictions are saved one JSON per molecule
  under `runs/`, and the scorers read those files. Re-scoring the same
  predictions reproduces the numbers exactly.
- **Model generation is not.** A fresh prediction run will differ slightly from
  the published cells (and from another fresh run). Treat small differences as
  run-to-run noise.
- A small number of saved prediction files contain a trailing second JSON object
  (a re-write). The robust scorers in `framework/scoring/` parse the first object
  and tolerate this; the original `score_full_table.py` can raise
  `JSONDecodeError: Extra data` on such a file.

## Known cell-level differences from the published tables
These were found while re-scoring the saved predictions. None change any
conclusion in the paper; they are recorded for honesty.

- **ChemRAG backbone.** The ChemRAG baseline was run on **Qwen3-4B-Instruct**,
  not Qwen3-32B. The manuscript sentence stating ChemRAG/MolRAG/MolE-RAG share a
  Qwen3-32B backbone should read: ChemRAG = Qwen3-4B; MolRAG and MolE-RAG =
  Qwen3-32B. ChemRAG is at chance on classification and abstains on 73–96% of
  regression molecules.
- **KNN.** The paper's k=5 KNN numbers come from `baselines/morgan_knn/k5/`; a
  validation-tuned-k variant also exists in `baselines/morgan_knn/` and differs
  (e.g. ClinTox 0.750 at k=5 vs 0.625 tuned).
- **SMILES-only (Table 2 left column).** 35 of 36 cells reproduce the published
  values exactly from the saved predictions. The exception is **opus / tox21**:
  the on-disk predictions score 76.7 vs 78.4 published (the predictions appear to
  have been regenerated after the paper's scoring; all 12 endpoints score
  cleanly, so this is not a scoring artifact).
- **+MolE-RAG (arm C).** Most datasets reproduce exactly or within noise. Two to
  note: **hiv / qwen3_32b = 76.9** over the full 4,112 test molecules (0 missing)
  vs 73.5 published — the paper scored over an incomplete shard set, so the
  complete-data number is higher; and **tox21 / opus = 81.2** vs 80.7 published.

## Credentials and models
Model endpoints are configured in `models.json`; keys go in `.env`. The shipped
model ids are the ones evaluated in the paper. Any OpenAI-compatible endpoint
serving those ids reproduces the setup; swapping ids runs a different model.
