#!/usr/bin/env python
"""Smoke test: env + frozen checkpoint + GIMLET graph/tokenizer/generate on a few molecules (CPU).
Also triggers the one-time pyximport build of dataloaders.algos (cached in ~/.pyxbld, shared home)."""
import os, sys, json, csv, re, types
GIMLET_DIR = '/workspace/baselines/GIMLET'
sys.path.insert(0, GIMLET_DIR); os.chdir(GIMLET_DIR)
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Hard-pin HF caches to the writable bhrm path (a stray TRANSFORMERS_CACHE in the
# shell profile points at an unwritable /work/hdd/bdgw/... and must be overridden).
os.environ['HF_HOME'] = '/workspace/models'
os.environ['HF_HUB_CACHE'] = '/workspace/models/hub'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/models/hub'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/models/hub'
# bypass heavy model/__init__ (KVPLM/MoMu baselines need boto3/fairseq/etc.)
_m = types.ModuleType('model'); _m.__path__ = [os.path.join(GIMLET_DIR, 'model')]
sys.modules['model'] = _m
_dl = types.ModuleType('dataloaders'); _dl.__path__ = [os.path.join(GIMLET_DIR, 'dataloaders')]
sys.modules['dataloaders'] = _dl
import torch
from transformers import AutoTokenizer
from ogb.utils import smiles2graph
from model.GIMLET.GIMLETTransformerForConditionalGeneration import GraphT5TransformerForConditionalGeneration
from dataloaders.gimlet_collator import CollatorForGIMLETLanguageModeling

PR = json.load(open('/workspace/mole-rag_naacl/gimlet_baseline/code/gimlet_prompts.json'))
DATA = '/workspace/MolE-RAG/data/moleculenet_property_scaffold/seed_0'
NUM_RE = re.compile(r"-?\d+\.?\d*e??\d*?")

def sample(task, k):
    out = []
    with open(os.path.join(DATA, task, '%s_test.csv' % task)) as f:
        for row in csv.DictReader(f):
            out.append((row['smiles'], float(row['label'])))
            if len(out) >= k:
                break
    return out

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'device', DEVICE)
tok = AutoTokenizer.from_pretrained('t5-small', use_fast=True)
id_y = tok.convert_tokens_to_ids(tok.tokenize('Yes'))[0]
id_n = tok.convert_tokens_to_ids(tok.tokenize('No'))[0]
print('Yes id', id_y, 'No id', id_n)
model = GraphT5TransformerForConditionalGeneration.from_pretrained('haitengzhao/gimlet').to(DEVICE).eval()
coll = CollatorForGIMLETLanguageModeling(tokenizer=tok, transform_in_collator=True, rich_features=False)
print('model loaded; params', sum(p.numel() for p in model.parameters()))

def to_dev(b):
    for k in list(b.keys()):
        b[k] = b[k].to(DEVICE) if hasattr(b[k], 'to') else b[k]
    return b

# classification: bace
prompt = PR['bace']['0']
mols = sample('bace', 3)
exs = []
for smi, y in mols:
    g = smiles2graph(smi)
    t = tok(prompt, truncation=True, max_length=512)
    exs.append({'graph': g, 'input_ids': t['input_ids'], 'attention_mask': t['attention_mask'],
                'labels': [id_y] if int(round(y)) == 1 else [id_n]})
b = coll(exs)
b.pop('labels')
b['max_length'] = 3
with torch.no_grad():
    o = model.generate(**b, output_scores=True, return_dict_in_generate=True)
lg = o.scores[0]
pp = torch.sigmoid(lg[:, id_y] - lg[:, id_n])
print('\nBACE classification:')
for i, (smi, y) in enumerate(mols):
    txt = tok.decode(o.sequences[i], skip_special_tokens=True).strip()
    print('  label=%s  p_pos=%.4f  gen=%r' % (y, float(pp[i]), txt))

# regression: lipo
prompt = PR['lipo']['0']
mols = sample('lipo', 3)
exs = []
for smi, y in mols:
    g = smiles2graph(smi)
    t = tok(prompt, truncation=True, max_length=512)
    lab = tok(str(round(y, 2)))
    exs.append({'graph': g, 'input_ids': t['input_ids'], 'attention_mask': t['attention_mask'],
                'labels': lab['input_ids'], 'decoder_attention_mask': lab['attention_mask']})
b = coll(exs)
labels = b.pop('labels'); b.pop('decoder_attention_mask')
with torch.no_grad():
    ids = model.generate(**b, max_length=labels.shape[1] + 1)
print('\nLIPO regression:')
for i, (smi, y) in enumerate(mols):
    txt = tok.decode(ids[i], skip_special_tokens=True)
    nums = NUM_RE.findall(txt)
    print('  label=%.2f  gen=%r  parsed=%s' % (y, txt, nums[0] if nums else None))
print('\nSMOKE TEST OK')
