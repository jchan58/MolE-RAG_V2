# Corpus construction (Depth B)

This is the pipeline that builds the chemistry **card corpus** from PMC
full text — the literature evidence MolE-RAG's C-stage retrieves over. Most
users don't run it: the pre-built cards are the input to the framework (Depth
A). Run this only to rebuild the corpus from scratch.

The output is, per dataset, the card bundle the framework reads
(`passages.json`, `findings.jsonl`, `entities.json`, `mol_map.json`,
`meta.json`) plus the identity/index artifacts.

## What it needs

- **AWS Bedrock** (`AWS_BEARER_TOKEN_BEDROCK`): query generation (Claude Opus),
  the multi-model finding-extraction fleet, and Cohere passage embedding/
  reranking all run through Bedrock.
- **NCBI E-utilities** (`NCBI_API_KEY`, `NCBI_EMAIL`): PMC search + metadata.
- **OpenAI** (`OPENAI_API_KEY`): only for the optional OpenAI relevance judge.
- PMC full text is read **anonymously** from the public `pmc-oa-opendata` S3
  bucket — no AWS keys needed for that part.
- Parts of the original run used **SageMaker Processing** (BM25 prescreen) and
  short-lived **EC2** workers; those are optional accelerators, not required.

Credentials come from the repo `.env` (or exported env vars); the
`~/.config/molrag/*.env` files are supported as a fallback.

## Pipeline stages

1. **Query generation** (`querygen.py`) — per-task and per-compound search
   queries (Claude Opus).
2. **Collection** (`run_collect.py`, `search.py`, `download.py`, `s3util.py`) —
   NCBI search → fetch OA full text from the public PMC bucket.
3. **Chunking** (`chunker.py`, `chunk_download.py`, `chunk_run.py`).
4. **BM25 prescreen** (`bm25_prescreen.py`; `sm_launch_chunk_bm25.py` to run it
   on SageMaker).
5. **Rerank** (`cohere_rerank.py`, `cohere_embed.py`; `medcpt_rerank.py` variant).
6. **Extraction** (`extract.py` + `bedrock_*.py`) — multi-model finding
   extraction into quote-verified records.
7. **Identity + indexes + cards** (`identity.py`, `build_indexes.py`,
   `build_cards.py`, `card_schema.py`).

`driver.py` ties the stages together; `env_common.sh` sets the per-dataset
paths and rate limits.

## Running it

```bash
source corpus/env_common.sh <dataset>     # sets EXP70_RUN_DIR, rate limits, etc.
python corpus/driver.py <dataset>         # or run individual stages above
```

Paths default to repo-relative locations (`corpus_runs/`, `/tmp/...` for bulk
scratch) and are overridable via env (`EXP70_BASE`, `EXP70_RUN_DIR`,
`EXP70_BULK_DIR`, `MOLERAG_DATA_ROOT`, `MOLERAG_SYN_CACHE`, `MOLERAG_LLM_CACHE`).

## Notes

- `identity.py` uses PubChem synonym caches (`data/caches/*.json`, ~63 MB). They
  are regenerable (the pipeline re-queries PubChem if absent) but are hosted
  alongside the cards to avoid re-fetching.
- The `*.sh` / `*.sbatch` files are the original orchestration wrappers (SLURM /
  EC2 / SageMaker / login-node watchdogs). Their paths have been parameterized
  (they resolve `CODE`/`PY`/`EXP70_BASE` from `env_common.sh` and env vars, so no
  author paths are hard-coded), but they still assume that class of cluster
  infrastructure. They are kept as reference — **the portable, from-scratch entry
  point is `env_common.sh` + the Python stages above**, which run on a single
  machine given the API credentials in `.env`. Depth B is optional: the shipped
  `data/cards/` already contains the corpus, so reproducing the paper's numbers
  (Depth A) needs none of this.
