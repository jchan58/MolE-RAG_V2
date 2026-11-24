#!/usr/bin/env python
"""
GIMLET (pretrained, zero-shot) baseline on OUR MoleculeNet scaffold-split test sets.

Reuses GIMLET's own graph construction (ogb.smiles2graph + graphormer transform),
t5-small tokenizer, collator, frozen pretrained checkpoint (haitengzhao/gimlet),
and GIMLET's own task instruction prompts (extracted verbatim from the HF dataset
haitengzhao/molecule_property_instruction, which is what the paper's zero-shot
--use_huggingface_pipeline runs).

For classification we extract a CONTINUOUS score p_pos = softmax over the {Yes,No}
answer-token logits of the first generated token (GIMLET's own decoder score), so
ROC-AUC is a proper ranking metric. For regression we parse the numeric generation.

Weights are FROZEN (no fine-tuning): model.eval(), torch.no_grad(), generate only.
"""
import os, sys, json, csv, re, argparse, time, math, types

GIMLET_DIR = '/workspace/baselines/GIMLET'
sys.path.insert(0, GIMLET_DIR)
os.chdir(GIMLET_DIR)   # so pyximport can build dataloaders.algos as a package submodule
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Offline + hard-pin HF caches to the writable bhrm path (a stray TRANSFORMERS_CACHE in
# the shell profile points at an unwritable /work/hdd/bdgw/... and must be overridden).
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/workspace/models'
os.environ['HF_HUB_CACHE'] = '/workspace/models/hub'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/models/hub'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/models/hub'

# Bypass model/__init__.py (it eagerly imports the KVPLM/MoMu baselines, which need
# boto3/fairseq/pytorch_lightning/etc. that we don't install). Inject a lightweight
# 'model' package with the correct __path__ so only GIMLET's own submodules load.
_model_pkg = types.ModuleType('model')
_model_pkg.__path__ = [os.path.join(GIMLET_DIR, 'model')]
sys.modules['model'] = _model_pkg
# Likewise bypass dataloaders/__init__.py (it imports wrap_dataset -> lmdb and other
# unused sibling collators). We only need dataloaders.gimlet_collator, whose relative
# imports (graphormer_transform/graphormer_collator/basic_collate) are all satisfied.
_dl_pkg = types.ModuleType('dataloaders')
_dl_pkg.__path__ = [os.path.join(GIMLET_DIR, 'dataloaders')]
sys.modules['dataloaders'] = _dl_pkg

import numpy as np
import torch
from transformers import AutoTokenizer
from ogb.utils import smiles2graph
from model.GIMLET.GIMLETTransformerForConditionalGeneration import GraphT5TransformerForConditionalGeneration
from dataloaders.gimlet_collator import CollatorForGIMLETLanguageModeling

DATA_ROOT = '/workspace/MolE-RAG/data/moleculenet_property_scaffold/seed_0'
PROMPTS_PATH = '/workspace/mole-rag_naacl/gimlet_baseline/code/gimlet_prompts.json'

REG_TASKS = {'esol', 'freesolv', 'lipo'}
CLA_TASKS = {'bbbp', 'bace', 'hiv', 'clintox', 'sider', 'tox21'}

# tox21 endpoint-name -> HF task_index (matched by the receptor the GIMLET prompt describes).
# ER prompts idx0==idx1 (identical); AR idx4='On the MDA cell...'(full receptor)->NR-AR, idx5->NR-AR-LBD.
TOX21_ENDPOINT2TI = {
    'NR-AR': '4', 'NR-AR-LBD': '5', 'NR-AhR': '3', 'NR-Aromatase': '2',
    'NR-ER': '0', 'NR-ER-LBD': '1', 'NR-PPAR-gamma': '6',
    'SR-ARE': '7', 'SR-ATAD5': '9', 'SR-HSE': '8', 'SR-MMP': '10', 'SR-p53': '11',
}

NUM_RE = re.compile(r"-?\d+\.?\d*e??\d*?")  # same as GIMLET basic_pipeline.eval_result


def build_prompt_fn(task, gimlet_prompts):
    """Return a function endpoint_name -> instruction string for this task."""
    if task in ('bace', 'bbbp', 'hiv', 'esol', 'freesolv', 'lipo'):
        p = gimlet_prompts[task]['0']
        return lambda ep: p
    if task == 'tox21':
        def f(ep):
            ti = TOX21_ENDPOINT2TI[ep]
            return gimlet_prompts['tox21'][ti]
        return f
    if task == 'clintox':
        # GIMLET has NO native clintox instruction -> adapted GIMLET-style instruction.
        def f(ep):
            if ep == 'CT_TOX':
                return ("The ClinTox dataset compares drugs approved by the FDA with drugs that "
                        "have failed clinical trials for toxicity reasons. This task concerns clinical "
                        "trial toxicity. Did this molecule fail clinical trials for toxicity reasons?")
            else:  # FDA_APPROVED
                return ("The ClinTox dataset records the FDA approval status of drug compounds. "
                        "This task concerns FDA approval. Is this molecule an FDA-approved drug?")
        return f
    if task == 'sider':
        # GIMLET has NO native sider instruction -> adapted GIMLET-style instruction.
        def f(ep):
            human = ep.replace('_', ' ')
            return ("The Side Effect Resource (SIDER) is a database of marketed drugs and their "
                    "recorded adverse drug reactions, grouped by MedDRA system organ class. "
                    "This task concerns the '%s' category. Does this molecule cause side effects "
                    "in the %s category?" % (human, human))
        return f
    raise ValueError(task)


def read_test_rows(task):
    """Yield (rowidx, smiles, label_float, endpoint) for rows with a valid label."""
    fn = os.path.join(DATA_ROOT, task, '%s_test.csv' % task)
    with open(fn) as f:
        for i, row in enumerate(csv.DictReader(f)):
            lab = row['label'].strip()
            if lab == '' or lab.lower() in ('nan', 'none'):
                continue
            try:
                y = float(lab)
            except ValueError:
                continue
            if task in CLA_TASKS and math.isnan(y):
                continue
            yield i, row['smiles'], y, row['task']


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)['topic_id'])
                except Exception:
                    pass
    return done


def run_task(task, model, tokenizer, collator, device, out_dir, batch_size, log):
    out_path = os.path.join(out_dir, '%s_preds.jsonl' % task)
    is_reg = task in REG_TASKS
    prompt_fn = build_prompt_fn(task, GIMLET_PROMPTS)

    # Yes/No token ids (GIMLET convention)
    id_y = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('Yes'))[0]
    id_n = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('No'))[0]

    done = load_done(out_path)
    log("[%s] already done: %d rows -> %s" % (task, len(done), out_path))

    # Gather valid records grouped by endpoint; pre-build graphs, drop invalid/oversized.
    by_ep = {}
    n_valid = 0
    n_badsmiles = 0
    for rowidx, smiles, y, ep in read_test_rows(task):
        n_valid += 1
        tid = "%s_%d" % (task, rowidx)
        if tid in done:
            continue
        try:
            g = smiles2graph(smiles)
        except Exception:
            n_badsmiles += 1
            continue
        if g['node_feat'].shape[0] == 0 or g['node_feat'].shape[0] > 512:
            n_badsmiles += 1
            continue
        by_ep.setdefault(ep, []).append((rowidx, smiles, y, g))

    log("[%s] valid-label rows=%d  to-predict=%d  endpoints=%d  bad/oversized_smiles=%d"
        % (task, n_valid, sum(len(v) for v in by_ep.values()), len(by_ep), n_badsmiles))

    fout = open(out_path, 'a', buffering=1)
    t0 = time.time()
    n_written = 0
    for ep in sorted(by_ep.keys()):
        recs = by_ep[ep]
        prompt = prompt_fn(ep)
        tok = tokenizer(prompt, return_special_tokens_mask=True, truncation=True, max_length=512)
        base_ids = tok['input_ids']
        base_mask = tok['attention_mask']
        for s in range(0, len(recs), batch_size):
            chunk = recs[s:s + batch_size]
            examples = []
            for (rowidx, smiles, y, g) in chunk:
                ex = {'graph': g, 'input_ids': list(base_ids), 'attention_mask': list(base_mask)}
                if is_reg:
                    lab = tokenizer(str(round(float(y), 2)))
                    ex['labels'] = lab['input_ids']
                    ex['decoder_attention_mask'] = lab['attention_mask']
                else:
                    ex['labels'] = [id_y] if int(round(y)) == 1 else [id_n]
                examples.append(ex)

            batch = collator(examples)
            for k in list(batch.keys()):
                # move tensors AND the torch_geometric graph Data object to device
                batch[k] = batch[k].to(device) if hasattr(batch[k], 'to') else batch[k]

            labels = batch.pop('labels')
            if 'decoder_attention_mask' in batch:
                dec_len = batch.pop('decoder_attention_mask').shape[1]

            with torch.no_grad():
                if is_reg:
                    max_len = labels.shape[1] + 1
                    ids = model.generate(**batch, max_length=max_len)
                    for i, (rowidx, smiles, y, g) in enumerate(chunk):
                        text = tokenizer.decode(ids[i, :], skip_special_tokens=True)
                        nums = NUM_RE.findall(text)
                        try:
                            pred = float(eval(nums[0]))
                        except Exception:
                            pred = None
                        rec = {"topic_id": "%s_%d" % (task, rowidx), "task": task,
                               "label": float(y),
                               "pred": (None if pred is None or (isinstance(pred, float) and math.isnan(pred)) else pred),
                               "text": text}
                        fout.write(json.dumps(rec) + "\n")
                        n_written += 1
                else:
                    batch['max_length'] = 3
                    out = model.generate(**batch, output_scores=True, return_dict_in_generate=True)
                    logits = out.scores[0]  # [B, vocab]; logits of first decoded (answer) token
                    ly = logits[:, id_y]
                    ln = logits[:, id_n]
                    # p_pos = softmax over {Yes,No} = sigmoid(ly - ln)
                    p_pos = torch.sigmoid(ly - ln).float().cpu().numpy()
                    seqs = out.sequences
                    for i, (rowidx, smiles, y, g) in enumerate(chunk):
                        text = tokenizer.decode(seqs[i, :], skip_special_tokens=True).strip()
                        rec = {"topic_id": "%s_%d" % (task, rowidx), "task": task,
                               "label": float(y), "p_pos": float(p_pos[i]), "text": text}
                        fout.write(json.dumps(rec) + "\n")
                        n_written += 1
        log("[%s]   endpoint %s done (%d recs, %.1fs elapsed)" % (task, ep, len(recs), time.time() - t0))
    fout.close()
    log("[%s] DONE wrote %d new preds in %.1fs" % (task, n_written, time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', required=True)
    ap.add_argument('--out_dir', default='/workspace/mole-rag_naacl/gimlet_baseline/outputs')
    ap.add_argument('--batch_size', type=int, default=40)
    args = ap.parse_args()

    def log(m):
        print("%s  %s" % (time.strftime('%H:%M:%S'), m), flush=True)

    global GIMLET_PROMPTS
    GIMLET_PROMPTS = json.load(open(PROMPTS_PATH))

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log("device=%s torch=%s cuda=%s" % (device, torch.__version__, torch.version.cuda))

    torch.manual_seed(5); np.random.seed(5)
    tokenizer = AutoTokenizer.from_pretrained('t5-small', use_fast=True)
    log("loading frozen pretrained checkpoint haitengzhao/gimlet ...")
    model = GraphT5TransformerForConditionalGeneration.from_pretrained('haitengzhao/gimlet')
    model.to(device).eval()
    collator = CollatorForGIMLETLanguageModeling(
        tokenizer=tokenizer, transform_in_collator=True, rich_features=False)

    for task in args.tasks:
        run_task(task, model, tokenizer, collator, device, args.out_dir, args.batch_size, log)
    log("ALL TASKS COMPLETE")


if __name__ == '__main__':
    main()
