# MolE-RAG

MolE-RAG is a training-free retrieval-augmented method for molecular property
prediction. For a query molecule it builds a molecular context from cached
identifiers and computed RDKit descriptors, retrieves the most similar labeled
training molecules by fingerprint similarity, and lets a general-purpose LLM
acquire literature evidence from a chemistry card corpus before predicting. No
model weights are trained.

This repository is a reproduction package: the code, data splits, prompts, and
scoring needed to re-run the framework and the baselines and obtain the paper's
numbers on the nine MoleculeNet benchmarks (BBBP, Tox21, SIDER, ClinTox, BACE,
HIV for classification; ESOL, FreeSolv, Lipophilicity for regression).

## How model calls work

The framework talks to models through an **OpenAI-compatible API**, so you can
point it at any provider — OpenAI, a local vLLM/TGI server, an inference
gateway, a cloud OpenAI-compatible endpoint. You choose the models in
[`models.json`](models.json): per model, a `base_url`, the provider's `model`
id, and the env var holding the API key.

**To reproduce the paper**, keep the model ids shipped in `models.json`
(Qwen3-32B, Ministral-3-14B, GPT-5-mini, Claude Opus 4.6, and the embedding
model) and only set their `base_url`/key for your provider. To run any other
model, add an entry and pass its name on the command line.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # add the API key(s) named in models.json
$EDITOR models.json           # set base_url for each model you'll run
```

You also need the chemistry **card corpus** (the literature evidence the C-stage
retrieves over). Point `MOLERAG_CARDS` at a directory holding one bundle per
dataset — `<dataset>/{passages.json,findings.jsonl,entities.json,mol_map.json,meta.json}`
— plus `data/bace_pool/` for BACE. (A hosted download of the corpus is planned;
until then set `MOLERAG_CARDS` to a local copy.)

## Reproduce a result

Everything runs per `(dataset, model)` and is resumable (one JSON per molecule).

```bash
# predict: writes runs/<dataset>__<model>/test_XXXX.json  (arms A / B / C)
python framework/run_fw_multi.py bbbp qwen3_32b

# score: ROC-AUC (classification) / RMSE (regression), arms A, B (self-review),
# C (evidence-assisted) with paired bootstrap -> scores/SCORES_fw_multi.json
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

Paths are configurable via env (`MOLERAG_DATA`, `MOLERAG_CARDS`, `MOLERAG_RUNS`);
defaults are repo-relative.

## Baselines

`baselines/` holds the Table-1 comparison methods, each with its own README and
`RESULTS.md`: the retrieval methods on a shared backbone (ChemRAG, MolRAG) and a
no-LLM Morgan-fingerprint KNN; the graph models (GIN, DMPNN, Graphormer-p,
GROVER, MolCLR, KANO-CMPNN); and the molecular language / multimodal models
(GIMLET, MolLM, nach0, BioT5, MolXPT, ChemDFM, LlaSMol, MolecularGPT). Model
weights and checkpoints are not vendored — each baseline's README says how to
obtain or train them.

## Layout

```
framework/     the method: run_fw_multi (+ hiv/tox21, bace, smiles_only) + scorers + model_client
models.json    per-model endpoint + id + api-key env (edit this)
prompts/       released prompt templates + condition definitions
data/
  moleculenet/seed_0/   scaffold splits (committed)
  cards/                card corpus (pointed at / fetched; git-ignored)
  bace_pool/            BACE retrieval pool
baselines/     one directory per Table-1 method
scores/        computed scores (reference numbers to match)
docs/          methodology, card schema, reproducibility notes
```

## Reproducibility notes

- Single fixed scaffold split (seed 0), no seed averaging — as in the paper.
- Model calls are nondeterministic; a fresh run will differ slightly from the
  published cells. Known cell-level caveats are listed in `docs/REPRODUCIBILITY.md`.

## Citation

If you use this code or the chemistry card corpus, please cite:

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
