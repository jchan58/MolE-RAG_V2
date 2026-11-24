"""SageMaker entry: Graphormer (PCQM4Mv2-pretrained) supervised fine-tune on MolE-RAG seed-0 splits.
For each task in TASKS and each seed in SEEDS, runs finetune_property_prediction_graph_only.py with the
graphormer_base config, restoring the MS PCQM4Mv2 checkpoint (from the 'ckpt' S3 input channel), and with the
train/val/test splits overridden to MolE-RAG seed-0 (canonical-SMILES match). Per-run metrics (ROC-AUC for
classification, RMSE for regression; test metric at best-validation epoch) are appended to SM_MODEL_DIR/results.jsonl."""
import os, subprocess, sys, json, shutil

# apex in the DLC is ABI-built for torch 1.9.1; we pinned 1.9.0 -> uninstall so transformers uses pure LayerNorm
try:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "apex"], check=False)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SM_MODEL_DIR", "/opt/ml/model"); os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)

# checkpoint from the 'ckpt' input channel
ckpt_dir = os.environ.get("SM_CHANNEL_CKPT", "/opt/ml/input/data/ckpt")
CKPT = os.path.join(ckpt_dir, "checkpoint_best_pcqm4mv2.pt")
print("CKPT exists:", os.path.exists(CKPT), CKPT, flush=True)
if not os.path.exists(CKPT):
    print("FATAL: PCQM4Mv2 checkpoint not found in ckpt channel", flush=True); sys.exit(3)

SPLITS = os.path.join(HERE, "molerag_splits.json")
RESULTS = os.path.join(OUT, "results.jsonl")

ALL = ["bbbp", "bace", "clintox", "sider", "tox21", "hiv", "esol", "freesolv", "lipo"]
sel = os.environ.get("TASKS", "").strip()
tasks = [t for t in (sel.split(",") if sel else ALL)]
seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]

EPOCHS_MAP = {"bbbp": 40, "bace": 40, "clintox": 40, "sider": 40,
              "esol": 50, "freesolv": 50, "lipo": 40, "tox21": 30, "hiv": 16}
# batch_size is the EFFECTIVE batch; the code asserts batch_size % grad_accum_step == 0 and uses
# micro-batch = batch_size/grad_accum_step per forward pass (that is what drives GPU memory).
# sider has large molecules + a 27-wide head and OOM'd at micro-batch 20 on a 24GB GPU, so raise its
# grad_accum so the micro-batch is small (40/8 = 5).  Keep effective batch 40 everywhere.
BATCH_MAP = {}  # keep effective batch_size=40 for all tasks
GA_MAP = {"hiv": 2, "tox21": 2, "sider": 8, "clintox": 2}  # sider micro-batch 5; 40%8==0
if os.environ.get("SMOKE") == "1":
    EPOCHS_MAP = {k: 2 for k in EPOCHS_MAP}

env_epochs = os.environ.get("EPOCHS")

print("TASKS:", tasks, "SEEDS:", seeds, flush=True)
runs = []
for ds in tasks:
    epochs = int(env_epochs) if env_epochs else EPOCHS_MAP.get(ds, 40)
    ga = GA_MAP.get(ds, 1)
    bs = BATCH_MAP.get(ds, 40)
    for seed in seeds:
        mdl = f"/tmp/mdl_{ds}_{seed}"; os.makedirs(mdl, exist_ok=True)
        cmd = [sys.executable, "finetune_property_prediction_graph_only.py",
               "--dataset", ds, "--runseed", str(seed),
               "--batch_size", str(bs), "--grad_accum_step", str(ga),
               "--output_model_dir", mdl, "--lr", "1e-5", "--epochs", str(epochs),
               "--backbone", "graphormer", "--arch", "graphormer_base",
               "--attention-dropout", "0.1", "--act-dropout", "0.1", "--dropout", "0.0",
               "--encoder-layers", "12", "--encoder-embed-dim", "768",
               "--encoder-ffn-embed-dim", "768", "--encoder-attention-heads", "32",
               "--encoder-normalize-before",  # emb_layer_norm present; NO --pre-layernorm -> matches MS PCQM4Mv2 ckpt
               "--num_classes", "1",
               "--restore_file_graphormer", CKPT,
               "--transform_in_collator", "--num_workers", "0",
               "--not_load_pretrained_model_output_layer"]
        runenv = dict(os.environ)
        runenv["MOLERAG_SPLITS_JSON"] = SPLITS
        runenv["MOLERAG_RESULT_JSON"] = RESULTS
        print("=== RUN", ds, "seed", seed, "epochs", epochs, "ga", ga, "===", flush=True)
        rc = subprocess.run(cmd, cwd=HERE, env=runenv).returncode
        print("   ", ds, "seed", seed, "rc=", rc, flush=True)
        runs.append((ds, seed, rc))
        shutil.rmtree(mdl, ignore_errors=True)

bad = [(d, s) for d, s, rc in runs if rc != 0]
print("ALL DONE; failed runs:", bad, flush=True)
# echo collected results
if os.path.exists(RESULTS):
    print("=== results.jsonl ===", flush=True)
    print(open(RESULTS).read(), flush=True)
# propagate failure: nonzero if ANY run failed (so 'Completed' means all seeds produced results)
sys.exit(1 if bad else 0)
