# Saved predictions (exact reproduction + full selection traces)

The framework is nondeterministic (it calls LLMs), so a fresh run lands *close*
to the published cells but not bit-identical. To reproduce the table numbers
**exactly**, and to inspect what the model actually did for every molecule, we
release the saved per-molecule prediction files from the runs that produced the
paper's numbers.

These files are large (tens of thousands of molecule-level JSON records), so
they are distributed as a release archive rather than committed into the tree.

## Get them

```bash
scripts/fetch_predictions.sh            # downloads + extracts into predictions/
```

or manually:

```bash
curl -L -o molerag_predictions.tar.gz \
  https://github.com/jchan58/MolE-RAG_V2/releases/download/predictions-v1/molerag_predictions.tar.gz
mkdir -p predictions && tar -xzf molerag_predictions.tar.gz -C predictions
```

## What's inside

```
predictions/
  runs/                      # the main framework (Table 1 methods col + Table 2 +MolE-RAG)
    <dataset>__<model>/
      test_0000.json         # one file per test molecule, all arms A/B/C
      ...
    # datasets: bbbp clintox esol freesolv hiv lipo sider tox21
    # models:   qwen3_32b ministral_3_14b opus_4_6 gpt5mini  (+ bbbp__chemdfm_v2_14b)
  framework_final/           # BACE +MolE-RAG
    <model>_test/            # one file per BACE test molecule
    SCORES_C_final.json      # the exact BACE arm-C scores
  test_eval/                 # Table-2 SMILES-only baseline (left column)
    smilesonly_<model>/      # bace + per-task SMILES-only per-molecule files
    smilesonly_multitask/<model>__<task>/
```

Every file is a single JSON object. Nothing is summarized or truncated — the raw
model text for each stage is kept alongside the parsed fields.

## Per-molecule record schema (`runs/<dataset>__<model>/test_XXXX.json`)

| field | meaning |
|---|---|
| `idx` | test-set index (matches `data/moleculenet/seed_0`) |
| `ds` | dataset |
| `model` | model id |
| `split` | `test` |
| `task_type` | `classification` or `regression` |
| `canonical` | RDKit-canonical SMILES of the query molecule |
| `target_SEALED` | ground-truth label, **never shown to the model** (sealed for scoring) |
| `A` | **arm A — initial answer** (SMILES only, no evidence) |
| `B` | **arm B — self-review** (model critiques/revises A) |
| `C` | **arm C — evidence-assisted** (the reported +MolE-RAG result) |
| `B_final` | B's final record (convenience copy) |
| `calls_so_far` | cumulative LLM call count at completion |

### `A` (initial)
- `raw` — raw model output
- `parsed.prediction`, `parsed.probability_positive` (classification) / value (regression)
- `parsed.justification` — the model's reasoning
- `parsed.unresolved_questions` — questions A itself flags as open

### `B` (self-review) — list, one entry per review pass
- `raw`
- `parsed.prediction`, `parsed.probability_positive`
- `parsed.changed` — did the review change the answer?
- `parsed.stop` — stop reviewing?
- `parsed.justification`

### `C` (evidence-assisted) — `C.rounds` is a list; `C.final` is the committed answer
Each entry in `C.rounds` is one acquire→assess cycle:

- **`ask`** — what the model decided to look up
  - `parsed.stop` — stop asking?
  - `parsed.question` — the natural-language question
  - `parsed.entities` — entity ids the model chose to query
  - `parsed.attributes` — attribute ids the model chose to query
  - `parsed.route` — `literature` (retrieve) or a tool/descriptor route
  - `parsed.why_matters` — why the model thinks this is relevant to the target
- **`evidence`** — what came back / what the model selected
  - `selected_entities`, `selected_attributes` — the model's selections
  - `passages` — retrieved literature, each with `passage_id`, `pmcid`, `cos`
    (cosine similarity to the query), and full `text`
  - `retrieval_status` — `ok`, `no_findings`, `embed_failed`, …
- **`evidence_type`** — `retrieval` (literature) vs tool/descriptor
- **`assess`** — the model's judgement after seeing the evidence
  - `parsed.prediction`, `parsed.probability_positive`
  - `parsed.question_answer_status` — `answered` / `partially_answered` / `unanswered`
  - `parsed.evidence_subject` — e.g. `general_class`, `exact_compound`
  - `parsed.new_information_added` — did this round add anything?
  - `parsed.next_action` — `continue` / `stop`
  - `parsed.justification`

`C.final` carries the committed `prediction`, `probability_positive`,
`question_answer_status`, `evidence_subject`, `new_information_added`,
`next_action`, and `justification`.

So for any molecule you can read, in order: the first guess (A), the self-review
(B), then every question the model asked, every passage it pulled (with source
PMCID and similarity), and how its probability moved round by round to the final
C answer — including the cases where evidence did **not** change the answer.

## Reproduce the exact table numbers from these files

**Framework (Table 1 methods column + Table 2 +MolE-RAG), 8 datasets:**

```bash
export MOLERAG_RUNS=$PWD/predictions/runs
python framework/score_fw_multi.py <dataset> <model>     # e.g. bbbp qwen3_32b
# arm C = +MolE-RAG; arm A = SMILES-only-style initial; arm B = self-review
```

`framework/run_hiv_tox21.py`'s companion scorer handles the sharded hiv/tox21
the same way (point it at `predictions/runs`).

**BACE +MolE-RAG:** the committed exact scores are in
`predictions/framework_final/SCORES_C_final.json` (mirrored at
`scores/molerag_C_bace.json`); the per-molecule files are in
`predictions/framework_final/<model>_test/`, re-scorable with
`framework/bace/run_final.py`'s scoring path.

**Table-2 SMILES-only baseline:** per-molecule files in
`predictions/test_eval/smilesonly_*`; score with
`framework/smiles_only/` (see `scores/smiles_only_scores.json` for the targets).

Because scoring is deterministic, re-scoring these files reproduces the published
cells exactly (0 LLM noise). The only cells that differed from the manuscript at
release time are noted in `REPRODUCIBILITY.md`.
