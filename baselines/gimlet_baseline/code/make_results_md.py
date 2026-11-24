#!/usr/bin/env python
"""Render RESULTS.md from analysis/results.json for the GIMLET zero-shot baseline."""
import json, os

BASE = '/work/hdd/bhrm/jchan5/mole-rag_naacl/gimlet_baseline'
res = json.load(open(os.path.join(BASE, 'analysis', 'results.json')))

# MolRAG-paper GIMLET numbers (for reference; paper uses different scaffold splits).
PAPER_CLA = {'bace': 0.6957, 'hiv': 0.6624, 'tox21': 0.6119, 'bbbp': 0.5939}
PAPER_REG = {'esol': 1.132, 'lipo': 1.345, 'freesolv': 5.103}

CLA = ['bbbp', 'bace', 'hiv', 'clintox', 'sider', 'tox21']
REG = ['esol', 'freesolv', 'lipo']
NATIVE = {'bbbp', 'bace', 'hiv', 'tox21', 'esol', 'freesolv', 'lipo'}

L = []
L.append("# GIMLET (pretrained, zero-shot) on 9 MoleculeNet test sets\n")
L.append("**Label:** GIMLET (pretrained, zero-shot). Frozen pretrained checkpoint "
         "`haitengzhao/gimlet` (NeurIPS 2023), no fine-tuning.\n")
L.append("## Setup")
L.append("- **Model:** GIMLET unified graph-text T5 (`haitengzhao/gimlet`), tokenizer `t5-small`. "
         "Weights FROZEN; inference is `model.generate` only.")
L.append("- **Graph + prompt pipeline:** GIMLET's own `ogb.smiles2graph` + graphormer transform "
         "(rich_features off, as in the repo's zero-shot recipe), GIMLET collator, and GIMLET's own "
         "per-task instruction prompts taken verbatim from the HF dataset "
         "`haitengzhao/molecule_property_instruction` (what the paper's "
         "`--use_huggingface_pipeline` zero-shot eval uses).")
L.append("- **Data:** OUR MoleculeNet scaffold-split **test** CSVs "
         "(`.../moleculenet_property_scaffold/seed_0/<task>/<task>_test.csv`), exact SMILES + endpoint. "
         "tox21/sider/clintox are long-format (one row per molecule x endpoint); rows with empty/NaN "
         "labels skipped.")
L.append("- **Classification score:** continuous `p_pos` = softmax over the {Yes,No} answer-token "
         "logits of the first generated token (= GIMLET's own decoder score; ranking identical to "
         "GIMLET's `logit_Yes - logit_No`). ROC-AUC computed on `p_pos`.")
L.append("- **Regression:** numeric value parsed from the generation (GIMLET's own regex), "
         "RMSE/MAE; mean-predictor RMSE = std of test labels.")
L.append("- **Instruction coverage:** GIMLET natively defines bbbp, bace, hiv, tox21, esol, freesolv, "
         "lipo. tox21's 12 endpoints are matched to GIMLET's 12 receptor/pathway prompts by the assay "
         "each prompt describes (NR-AR->MDA-cell AR prompt, NR-AR-LBD->other AR prompt, the two ER "
         "prompts are identical). **GIMLET does NOT natively define sider or clintox** -> these use "
         "**adapted GIMLET-style instructions** (a property description + Yes/No question) and are "
         "flagged below; they are strongly out-of-distribution for the pretrained model and are NOT "
         "part of the paper's GIMLET comparison.\n")

L.append("## Classification - ROC-AUC (pooled over all rows)\n")
L.append("| task | n | scored | AUC (pooled) | macro-AUC | acc@0.5 | pos_rate | paper GIMLET | native? |")
L.append("|---|--:|--:|--:|--:|--:|--:|--:|:--:|")
for t in CLA:
    r = res.get(t, {})
    auc = r.get('AUC', 'n/a')
    macro = r.get('macro_AUC', '-')
    acc = r.get('acc@0.5', '-')
    pr = r.get('pos_rate', '-')
    paper = PAPER_CLA.get(t, '-')
    nat = 'yes' if t in NATIVE else '**no (adapted)**'
    L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" %
             (t, r.get('n', '-'), r.get('scored', '-'), auc, macro, acc, pr, paper, nat))
L.append("")
L.append("macro-AUC = mean of per-endpoint AUCs (endpoints with both classes present) for the "
         "multitask sets tox21/sider/clintox; '-' for single-endpoint tasks.\n")

L.append("## Regression - RMSE / MAE\n")
L.append("| task | n | numeric | RMSE | MAE | mean-predictor RMSE | paper GIMLET RMSE |")
L.append("|---|--:|--:|--:|--:|--:|--:|")
for t in REG:
    r = res.get(t, {})
    L.append("| %s | %s | %s | %s | %s | %s | %s |" %
             (t, r.get('n', '-'), r.get('numeric', '-'), r.get('RMSE', 'n/a'),
              r.get('MAE', '-'), r.get('mean_predictor_RMSE', '-'), PAPER_REG.get(t, '-')))
L.append("")
L.append("## Notes")
L.append("- Numbers use OUR scaffold splits (seed_0), so they do not match the MolRAG-paper GIMLET "
         "numbers exactly; the paper columns are for reference only.")
L.append("- Frozen zero-shot: no fine-tuning, no few-shot, no prompt tuning. GIMLET's graph encoder, "
         "t5-small tokenizer, and native instruction text are all reused unchanged.")
L.append("- sider / clintox: GIMLET has no pretrained instruction for these; adapted GIMLET-style "
         "instructions were used. Treat those rows as an out-of-distribution probe, not a faithful "
         "GIMLET task.")

open(os.path.join(BASE, 'RESULTS.md'), 'w').write("\n".join(L) + "\n")
print("WROTE", os.path.join(BASE, 'RESULTS.md'))
