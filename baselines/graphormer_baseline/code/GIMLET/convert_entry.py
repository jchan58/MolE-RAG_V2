"""Stage the official Microsoft PCQM4Mv2 Graphormer-base weights for GIMLET.
Source: clefourrier/graphormer-base-pcqm4mv2 (HuggingFace serialization of the same MS PCQM4Mv2 base
checkpoint; Azure blob is unreachable from both the login node and the SageMaker VPC). Its state_dict keys
are ALREADY in GIMLET's fairseq naming (encoder.graph_encoder.*), so no remap is needed: we wrap it as
{'model': state_dict}, guarantee the two output-head keys that restore_from_file pops are present, dry-run a
load into a freshly built GIMLET graphormer_base to confirm the backbone loads, then upload to S3."""
import os, sys, subprocess, json
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "apex"], check=False)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
from huggingface_hub import hf_hub_download

hf_path = hf_hub_download("clefourrier/graphormer-base-pcqm4mv2", "pytorch_model.bin")
hf = torch.load(hf_path, map_location="cpu")
if isinstance(hf, dict) and "model" in hf and all(not k.startswith("encoder") for k in hf.keys()):
    hf = hf["model"]
print("HF state_dict #keys", len(hf), flush=True)
sample = [k for k in hf.keys()][:6] + [k for k in hf.keys() if not k.startswith("encoder.graph_encoder.layers")][-8:]
for k in sample:
    print("  HF", k, tuple(hf[k].shape), flush=True)

# restore_from_file() unconditionally pop()s these two when --not_load_pretrained_model_output_layer,
# so they MUST exist in the saved dict (they are discarded + reinitialized before load anyway).
if "encoder.embed_out.weight" not in hf:
    hf["encoder.embed_out.weight"] = torch.zeros(1, 768)
    print("added dummy encoder.embed_out.weight", flush=True)
if "encoder.lm_output_learned_bias" not in hf:
    hf["encoder.lm_output_learned_bias"] = torch.zeros(1)
    print("added dummy encoder.lm_output_learned_bias", flush=True)

# Dry-run: build GIMLET graphormer_base and load the backbone to verify key compatibility.
from transformers import HfArgumentParser
from model import GraphormerModel, GraphormerConfig, Graphormer_version_dict
parser = HfArgumentParser(GraphormerConfig); parser = GraphormerModel.add_args(parser)
args = parser.parse_args([
    "--arch", "graphormer_base", "--encoder-layers", "12", "--encoder-embed-dim", "768",
    "--encoder-ffn-embed-dim", "768", "--encoder-attention-heads", "32",
    "--encoder-normalize-before",  # emb_layer_norm present (matches ckpt); NO --pre-layernorm (post-LN layers)
    "--attention-dropout", "0.1", "--act-dropout", "0.1", "--dropout", "0.0",
    "--num_classes", "1", "--not_load_pretrained_model_output_layer"])
args = Graphormer_version_dict[args.arch](args); args.num_classes = 1
model = GraphormerModel.build_model(args)
sd = dict(hf)
sd.pop("encoder.embed_out.weight", None); sd.pop("encoder.lm_output_learned_bias", None)
missing, unexpected = model.load_state_dict(sd, strict=False)
be_missing = [k for k in missing if k.startswith("encoder.graph_encoder.")]
print("DRYRUN total missing %d (graph_encoder-missing %d); total unexpected %d"
      % (len(missing), len(be_missing), len(unexpected)), flush=True)
for k in be_missing[:25]:
    print("  MISS-backbone", k, flush=True)
for k in list(missing)[:25]:
    print("  miss", k, flush=True)
for k in list(unexpected)[:25]:
    print("  unexpected", k, flush=True)

out_pt = "/tmp/checkpoint_best_pcqm4mv2.pt"
torch.save({"model": hf}, out_pt)
print("saved", out_pt, os.path.getsize(out_pt), "bytes", flush=True)
import boto3
boto3.client("s3", region_name="us-east-1").upload_file(
    out_pt, "molerag", "graphormer_baseline/ckpt/checkpoint_best_pcqm4mv2.pt")
print("UPLOADED s3://molerag/graphormer_baseline/ckpt/checkpoint_best_pcqm4mv2.pt", flush=True)
OUT = os.environ.get("SM_MODEL_DIR", "/opt/ml/model"); os.makedirs(OUT, exist_ok=True)
json.dump({"source": "clefourrier/graphormer-base-pcqm4mv2 (HF serialization of MS PCQM4Mv2 base)",
           "hf_keys": len(hf), "dryrun_missing": len(missing),
           "dryrun_graph_encoder_missing": len(be_missing), "dryrun_unexpected": len(unexpected),
           "graph_encoder_missing_keys": be_missing, "missing_keys": list(missing),
           "unexpected_keys": list(unexpected)},
          open(os.path.join(OUT, "convert_manifest.json"), "w"), indent=2)
if len(be_missing) > 0:
    print("WARNING: backbone graph_encoder keys missing -> pretraining would be incomplete", flush=True)
    sys.exit(4)
print("CONVERT_OK", flush=True)
