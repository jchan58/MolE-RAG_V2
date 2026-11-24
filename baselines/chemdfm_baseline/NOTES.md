# chemdfm_baseline

Baseline: **ChemDFM (LLaMA-13B)** — released-checkpoint ZERO-SHOT inference (no fine-tuning, no seeds).
Evaluated on the MolE-RAG seed-0 scaffold-split **test** sets (9 MoleculeNet tasks), same convention
as the other baselines (OUR dmpnn_baseline/data/<task>/test.csv molecules).

## Model
- HF repo `OpenDFM/ChemDFM-v1.0-13B` (verified: public / NOT gated; 6 safetensors shards =
  full MERGED fp16 weights, 13.09B params ~26GB; `LlamaForCausalLM`, model_type llama). Built on
  LLaMA-1-13B. No gated-LLaMA delta needed — faithfully runnable.
- Loaded in-container via `LlamaTokenizer` + `LlamaForCausalLM(torch_dtype=fp16, device_map="auto")`,
  exactly as the repo's `cli_dialogue_inference_demo.py`. HF_HOME=/tmp/hf (downloaded inside the job).
- ~13B fp16 fits one L40S 48GB → SageMaker pool ml.g6e.{2,8,12}xlarge.

## Prompts (code/ChemDFM cloned from github.com/OpenDFM/ChemDFM)
ChemDFM ships its own ChemLLMBench property-prediction eval data (exact prompts) for **5** of our
tasks. Templates extracted verbatim into `sm/chemdfm_templates.json` (prefix+SMILES+suffix
round-trips to the original question for 11,915/11,915 released rows — lossless):
- **bbbp, bace, hiv, clintox (FDA+tox), tox21 (12 endpoints)** → ChemDFM RELEASED prompt, verbatim.
  Wrapped in ChemDFM dialogue format `[Round 0]\nHuman: {question}\nAssistant:` per the repo eval
  README. Zero-shot (the released files have no few-shot examples).
- **sider (27 ep)** and **esol / freesolv / lipo (regression)** → ChemDFM released NO prompt; we use
  ChemDFM-STYLE wording (same "You are an expert chemist…" instruction shell). Flagged in provenance.

SMILES canonicalized with rdkit `Chem.CanonSmiles(useChiral=True)` per the repo README (raw fallback,
counted).

## Scoring
- Classification → ROC-AUC from P(positive answer token): softmax of the FIRST greedy-step logits
  over {Yes, No} answer-token ids (every template phrased so Yes == label 1). Mean AUC over endpoints,
  missing labels masked, single-class endpoints skipped. Also report hard-label accuracy (ChemDFM's
  own string-match scoring) from the generated text.
- Regression → greedy generate, parse first float, impute TRAIN task-mean if unparseable (counted),
  RMSE in original units.

## Compute (SageMaker only; never login node)
- 2 balanced single-GPU shards (~9.3k prompts each): `chemdfm-tox21` (tox21) on ml.g6e.2xlarge,
  `chemdfm-mix` (hiv,sider,bbbp,bace,clintox,esol,freesolv,lipo) on ml.g6e.8xlarge.
- Restart-safe: entry.py streams per-task results to
  s3://molerag/chemdfm_baseline/output/incremental/<task>.json and skips done tasks on resume.
- Launch: `sm/chemdfm_launch.py`; detached watcher `sm/watch.py` re-aggregates on completion;
  `analysis/aggregate.py` → `analysis/chemdfm_results.json` + `RESULTS.md` (+ billable seconds).
- Role arn:aws:iam::833984135295:role/service-role/AmazonSageMaker-ExecutionRole-20260922T220415,
  bucket molerag, prefix s3://molerag/chemdfm_baseline/.
