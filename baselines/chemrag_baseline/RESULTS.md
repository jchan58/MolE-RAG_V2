# ChemRAG framework on 9 MoleculeNet test sets — Qwen3-4B-Instruct

**One-line finding.** Applying the ChemRAG framework (RRF(BM25+Contriever) top-5 passages from the ChemRAG corpus, ChemRAG's own RAG prompt templates) to MoleculeNet property prediction with Qwen3-4B-Instruct-2507 yields **no usable signal**: classification AUC sits at chance on all 6 tasks, and on regression the model **abstains** ("the answer is therefore unknown") on 73–96% of molecules because the retrieved documents do not contain the compound's numeric endpoint. Retrieval was verified real and on-topic (every one of 16,648 queries got 5 non-empty passages; PubChem + semantic-scholar dominate), so this is a faithful result, not a plumbing artifact. Consistent with the programme-wide finding that retrieved text does not help molecular property prediction.

## Setup
- **Tasks / scope:** all 9 MoleculeNet scaffold-split **test** sets (16,648 molecules): classification = bbbp, bace, hiv, clintox, sider, tox21; regression = esol, freesolv, lipo.
- **Retriever:** ChemRAG combined **RRF(BM25, Contriever)**, RRF_K=60, **top-5** passages/molecule, over the ChemRAG corpus (partitions present here: pubchem, semantic_scholar, uspto, pubmed, wikipedia; textbook absent). SPECTER/e5 not built/available, so 2-retriever RRF (BM25 = 130 GB Lucene; Contriever = 101 M-passage fp16 dense, GPU scan).
- **Query:** ChemRAG-style "property question + SMILES" per molecule.
- **Generator:** Qwen3-4B-Instruct-2507 (local, bf16).
- **Prompts:** ChemRAG Table 17 (multi-choice RAG) for classification, Table 18 (numerical RAG) for regression — verbatim structure, "only give me the answer."
- **Scoring:** classification AUC from P(positive) = forced-choice answer-token logprobs (A=Yes vs B=No); regression RMSE from the parsed numeric answer.

## Classification (AUC)
| task | n | AUC | acc@0.5 | pos_rate | non-trivial probs |
|---|--:|--:|--:|--:|--:|
| bbbp | 205 | 0.557 | 0.439 | 0.776 | 18% |
| bace | 152 | **0.394** | 0.421 | 0.559 | 10% |
| hiv | 4112 | 0.524 | 0.748 | 0.027 | 7% |
| clintox | 298 | 0.500 | 0.493 | 0.507 | 7% |
| sider | 3888 | 0.547 | 0.487 | 0.574 | 12% |
| tox21 | 7394 | 0.488 | 0.676 | 0.108 | 11% |

Mean AUC ≈ 0.50. P(positive) is strongly bimodal (0 or 1) — the model gives confident Yes/No answers that do not track the property label (only 7–18% of probabilities are non-extreme). acc@0.5 just tracks the base rate on the imbalanced tasks (hiv 2.7% pos → 0.748 by predicting mostly one way; not informative).

## Regression — abstention, RMSE not meaningful
| task | n | abstained ("unknown") | numeric parsed |
|---|--:|--:|--:|
| esol | 114 | 96% | 5 |
| freesolv | 65 | 91% | 5 |
| lipo | 420 | 73% | 104 |

Under ChemRAG's numerical RAG prompt, the model overwhelmingly answers "The answer is therefore unknown" — the retrieved snippets (even PubChem molecule records) do not state the compound's logS / hydration free energy / logD. RMSE over the handful of molecules that did get a number is not a meaningful estimate (and is huge, e.g. freesolv 53.6 over 5 points), so **RMSE is reported as undefined/abstention-dominated** rather than as a score.

## Interpretation
ChemRAG retrieval works and surfaces relevant chemistry text, but that text does not carry the quantitative/binary endpoints MoleculeNet asks for. The model either abstains (regression) or guesses confidently at chance (classification). This matches the broader programme: measured structural references are the lever for property prediction; retrieved literature/database text is not.

## Caveats / next options
- This is the **RAG** condition only; a **no-retrieval baseline** was not run here (the ChemRAG framing is RAG-vs-baseline — a baseline run would show whether RAG helps/hurts vs the model alone). Offerable.
- Regression abstention is faithful to ChemRAG's prompt; a variant prompt that **forces a numeric estimate** would produce a real RMSE (deviates from ChemRAG's template).
- Only BM25+Contriever RRF (SPECTER/e5 unavailable); Qwen3-4B (small). A larger model / full 4-retriever RRF could differ but the abstention/chance pattern is unlikely to reverse given the corpus lacks the endpoints.

## Artifacts
`queries/` (16,648 across 9 tasks), `retrieval/{bm25_hits,dense_hits,fused_top5,fused_docs,id_contents}.jsonl`, `outputs/{task}_preds.jsonl`, `analysis/results.json`, `generate_score.py`, `jobs/`. Pipeline ran as chained 1-hr interactive A100 jobs (no CPU allocation; batch-GPU backlogged weeks); BM25 ≈0.49 s/q at 64 threads.

---

# Follow-up: Ministral-14B (Bedrock) with FORCED-NUMERIC prompts

Same retrieval (reused top-5 RRF docs, `retrieval/fused_docs.jsonl`), same 9 tasks / 16,648 molecules, generator = **mistral.ministral-3-14b-instruct** (Bedrock Converse, temp 0). Prompts adapted to FORCE a number (no abstention): classification → P(positive) in [0,1] (for AUC, since Bedrock gives no logprobs); regression → terse numeric estimate. Ran on login node, 16 parallel workers, 0 errors, ~minutes (no throttling).

## Classification (AUC) — Ministral-14B vs Qwen3-4B
| task | N | Ministral-14B | Qwen3-4B |
|---|--:|--:|--:|
| bbbp | 205 | 0.536 | 0.557 |
| bace | 152 | 0.446 | 0.394 |
| hiv | 4112 | 0.567 | 0.524 |
| clintox | 298 | 0.491 | 0.500 |
| sider | 3888 | 0.514 | 0.547 |
| tox21 | 7394 | 0.496 | 0.488 |
| mean | | ≈0.51 | ≈0.50 |

## Regression (RMSE) — Ministral-14B (0 abstentions) vs mean-predictor baseline
| task | N | Ministral-14B RMSE | MAE | mean-baseline RMSE |
|---|--:|--:|--:|--:|
| esol | 114 | 1.853 | 1.402 | 1.912 |
| freesolv | 65 | 13.833 | 6.909 | 4.723 |
| lipo | 420 | 1.939 | 1.431 | 1.184 |

**Finding (both models): same null.** Classification at chance (mean AUC ≈0.51). Forcing numbers on regression produces real RMSEs that are **no better than, and mostly worse than, a trivial mean-predictor** (esol ~ties; freesolv and lipo worse). ChemRAG-retrieved text does not enable MoleculeNet property prediction with either Qwen3-4B or Ministral-14B. Artifacts: `outputs_ministral/{task}_preds.jsonl`, `analysis/results_ministral.json`.
