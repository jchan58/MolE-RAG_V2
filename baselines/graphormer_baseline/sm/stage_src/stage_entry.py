"""Stage the Microsoft Graphormer PCQM4Mv2 base checkpoint to S3.
Runs inside a SageMaker job (AWS us-east-1 has internet + Azure reachability).
Downloads the fairseq-style {'model': state_dict} .pt from Azure blob, verifies it
loads with torch.load, prints key names (to confirm compatibility with GIMLET's
graphormer restore_from_file), and uploads the raw .pt to S3 for reuse."""
import os, sys, time, urllib.request

URL = "https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_best_pcqm4mv2.pt"
BUCKET = "molerag"
KEY = "graphormer_baseline/ckpt/checkpoint_best_pcqm4mv2.pt"
DST = "/tmp/checkpoint_best_pcqm4mv2.pt"

ok = False
for attempt in range(6):
    try:
        print("download attempt", attempt, URL, flush=True)
        urllib.request.urlretrieve(URL, DST)
        ok = True
        break
    except Exception as e:
        print("download failed:", repr(e), flush=True)
        time.sleep(15)
if not ok:
    print("FATAL: could not download checkpoint from Azure blob", flush=True)
    sys.exit(2)

sz = os.path.getsize(DST)
print("downloaded bytes", sz, flush=True)

import torch
state = torch.load(DST, map_location="cpu")
print("top-level type", type(state), flush=True)
if isinstance(state, dict):
    print("top-level keys", list(state.keys())[:12], flush=True)
sd = state["model"] if (isinstance(state, dict) and "model" in state) else state
keys = list(sd.keys())
print("state_dict #keys", len(keys), flush=True)
print("first 40 keys:", flush=True)
for k in keys[:40]:
    shp = tuple(sd[k].shape) if hasattr(sd[k], "shape") else ""
    print("   ", k, shp, flush=True)
for probe in ["encoder.embed_out.weight", "encoder.lm_output_learned_bias"]:
    print("has", probe, ":", probe in sd, flush=True)
n_graph_encoder = sum(1 for k in keys if k.startswith("encoder.graph_encoder."))
print("num encoder.graph_encoder.* keys:", n_graph_encoder, flush=True)
nparam = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
print("total params in state_dict", nparam, flush=True)

import boto3
boto3.client("s3", region_name="us-east-1").upload_file(DST, BUCKET, KEY)
print("UPLOADED s3://%s/%s" % (BUCKET, KEY), flush=True)

out = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "stage_manifest.txt"), "w") as f:
    f.write("url %s\nbytes %d\nstate_dict_keys %d\ngraph_encoder_keys %d\nparams %d\n"
            % (URL, sz, len(keys), n_graph_encoder, nparam))
print("STAGE_OK", flush=True)
