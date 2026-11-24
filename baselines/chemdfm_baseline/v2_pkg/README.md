# ChemDFM-v2.0-14B zero-shot MoleculeNet baseline (run on your 4x A10G box)

Reproduces the ChemDFM baseline with **OpenDFM/ChemDFM-v2.0-14B** (Qwen2-based) on the 9
MoleculeNet MolE-RAG seed-0 scaffold TEST sets. Same ChemLLMBench prompts + scoring as the
v1 baseline; v2 uses its own Qwen2 chat template.

## Run
    pip install -r requirements.txt
    python run_chemdfm_v2.py                 # all 9 tasks, device_map=auto across all GPUs
    # or a subset:  python run_chemdfm_v2.py tox21 hiv

The 14B model (~28GB fp16) downloads from the HF hub on first run and shards across your
4x A10G (device_map=auto). ~16.6k prompts total; resume-safe (a finished task in results/ is skipped).

## Output
- results/<task>.json  (per-task: AUC for classification / RMSE for regression + provenance)
- results/_summary.json

## Scoring (identical to the v1 ChemDFM baseline)
- classification: P(positive) = softmax of first greedy-step logits over {Yes,No}; macro-AUC over endpoints.
- regression: greedy generate, parse first float, impute train-mean if unparseable; RMSE.
