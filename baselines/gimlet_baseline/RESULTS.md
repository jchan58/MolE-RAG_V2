# GIMLET (zero-shot, instruction-based) — MoleculeNet (MolE-RAG seed-0 test)

GIMLET (Zhao et al., NeurIPS 2023) graph-text T5 model, **zero-shot** (pretrained `haitengzhao/gimlet`,
no task training), its own instruction prompts, evaluated on the **MolE-RAG seed-0 test molecules**
(test set overridden by canonical-SMILES match; coverage full/near-full per task). Ran on SageMaker
(legacy PyTorch 1.9.1 DLC + reconstructed PyG 1.7.2 stack).

| task | metric | GIMLET zero-shot |
|---|---|---|
| bbbp | ROC-AUC | 0.776 |
| bace | ROC-AUC | 0.601 |
| hiv | ROC-AUC | 0.527 |
| tox21 | ROC-AUC | 0.631 (mean of 12 subtasks) |
| clintox | ROC-AUC | — (unsupported by GIMLET's released artifacts) |
| sider | ROC-AUC | — (unsupported by GIMLET's released artifacts) |
| esol | RMSE | 1.057 |
| freesolv | RMSE | 6.221 |
| lipo | RMSE | 1.350 |

**clintox/sider — not supported by GIMLET's released zero-shot artifacts (verified).** We attempted to
finish 9/9 by sourcing authentic instruction templates, but clintox and sider are absent from *every*
released source: both prompt JSONs (`prompts/selected_prompt_downstream_task.json`,
`prompts/prompt_downstream_task.json`) **and** the HF `haitengzhao/molecule_property_instruction` dataset
that supplies GIMLET's instructions. The instruction dataset's split keys are exactly
`cyp450, lipo, bbbp, freesolv, bace, chembl_zero_shot, pcba, chembl_pretraining, tox21, esol, toxcast, hiv, muv`
— no clintox, no sider. GIMLET therefore has no authentic zero-shot prompt for these two tasks, so a
faithful run is not possible without hand-authored (non-GIMLET) prompts. We report GIMLET on the 7 tasks
its released artifacts actually cover. Raw per-task CSV: `results/gimlet_results.csv`; aggregate
`analysis/gimlet_results.json`.
