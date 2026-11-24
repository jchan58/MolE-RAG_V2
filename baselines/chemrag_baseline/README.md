# ChemRAG baseline on MoleculeNet (9 tasks)

ChemRAG framework applied to MoleculeNet property prediction. Retriever = **RRF(BM25 + Contriever)** over the ChemRAG corpus, **top-5** passages/molecule; generator uses ChemRAG's RAG prompt templates. Full scaffold-split **test** sets, all 9 tasks (16,648 molecules). Two generators evaluated.

## Headline results

**Classification — ROC-AUC** (P(positive): Qwen via answer-token logprobs; Ministral asked for a 0–1 probability)
| task | N | Qwen3-4B-Instruct | Ministral-14B (Bedrock) |
|---|--:|--:|--:|
| bbbp | 205 | 0.557 | 0.536 |
| bace | 152 | 0.394 | 0.446 |
| hiv | 4112 | 0.524 | 0.567 |
| clintox | 298 | 0.500 | 0.491 |
| sider | 3888 | 0.547 | 0.514 |
| tox21 | 7394 | 0.488 | 0.496 |
| **mean** | | **≈0.50** | **≈0.51** |

**Regression — RMSE** (Qwen abstains "unknown" under ChemRAG's strict prompt; Ministral forced to a number)
| task | N | Qwen3-4B | Ministral-14B | mean-predictor baseline |
|---|--:|--:|--:|--:|
| esol | 114 | — (96% abstain) | 1.853 | 1.912 |
| freesolv | 65 | — (91% abstain) | 13.833 | 4.723 |
| lipo | 420 | — (73% abstain) | 1.939 | 1.184 |

**Conclusion:** ChemRAG retrieval delivers real, on-topic chemistry passages (PubChem + semantic-scholar dominant; verified 5 non-empty docs for all 16,648 queries), but it provides **no usable signal for MoleculeNet property prediction** with either model — classification at chance, and forced regression no better than (mostly worse than) predicting the training mean. The retrieved text carries molecule descriptions, not the quantitative/binary endpoints.

## Contents
- `RESULTS.md` — full write-up (both models, caveats).
- `analysis/results.json` (Qwen3-4B), `analysis/results_ministral.json` (Ministral-14B).
- `outputs/` (Qwen preds), `outputs_ministral/` (Ministral preds) — per-task `{topic_id,label,p_pos|pred,text}`.
- `queries/` — the 16,648 ChemRAG-style queries per task.
- `retrieval/` — scripts + hits: `bm25_hits.jsonl`, `dense_hits.jsonl`, `fused_top5.jsonl`, `fused_docs.jsonl` (top-5 docs w/ contents), `id_contents.jsonl`.
- `generate_score.py` (Qwen, local HF), `generate_ministral.py` (Bedrock), `jobs/` (sbatch + orchestrator).

## Reproduce
Retrieval (chained interactive-GPU pipeline): `jobs/orchestrate.sh` → `jobs/pipeline.sbatch` (BM25 via main py3.13 + `PYTHONPATH=/work/hdd/bhrm/jchan5/site-packages` + Java 21; dense scan GPU fp16; fuse+fetch). Generation+score: `generate_score.py --regress --classify --score` (Qwen) or `generate_ministral.py --generate --score` (Bedrock `mistral.ministral-3-14b-instruct`). Indices reused from exp44: BM25 `MolE-RAG/external/ChemRAG/index/bm25/bm25`, Contriever `contriever_index/`, corpus `MolE-RAG/external/ChemRAG/corpus/chunks/`.

## Caveats / not yet run
- 2-retriever RRF (SPECTER/e5 not built/available); corpus lacks the textbook partition.
- No **no-retrieval baseline** yet (same models/prompts without documents) to isolate RAG-vs-model-alone.
