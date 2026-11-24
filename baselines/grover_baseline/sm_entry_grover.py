#!/usr/bin/env python3
"""SageMaker entry for GROVER-Large (GTransformer) finetuning (GODE's m_level GROVER fork).

cwd = /opt/ml/code (unpacked sourcedir: grover/, task/, scripts/, main.py, grover_large.pt,
sm_data/<task>/{train,val,test}.csv). For each of the 9 MoleculeNet tasks:
  (1) generate rdkit_2d_normalized features (.npz) for train/val/test via scripts/save_features.py,
  (2) run `python main.py finetune` from grover_large.pt on our seed_0 split (separate val/test paths),
  (3) capture the test AUC/RMSE line.
Results + per-task logs -> /opt/ml/model. Hyperparams: --tasks (space list or 'all'), --epochs.
"""
import os, json, subprocess, argparse, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "sm_data")
FEAT = os.path.join(HERE, "feat")
OUT = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
CKPT = os.path.join(HERE, "grover_large.pt")
CLS = ["bbbp", "bace", "hiv", "clintox", "sider", "tox21"]
REG = ["esol", "freesolv", "lipo"]
DTYPE = {**{t: "classification" for t in CLS}, **{t: "regression" for t in REG}}
ORDER = ["bbbp", "bace", "clintox", "esol", "freesolv", "lipo", "sider", "tox21", "hiv"]


def gen_feat(csv, npz):
    if os.path.exists(npz):
        return 0
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    return subprocess.run(["python", "scripts/save_features.py", "--data_path", csv,
                           "--save_path", npz, "--features_generator", "rdkit_2d_normalized",
                           "--restart"], cwd=HERE).returncode


def run_task(t, epochs, gpu):
    import threading
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    rc_f = 0
    for sp in ["train", "val", "test"]:
        npz = f"{FEAT}/{t}/{sp}.npz"
        if not os.path.exists(npz):
            os.makedirs(os.path.dirname(npz), exist_ok=True)
            rc_f |= subprocess.run(["python", "scripts/save_features.py", "--data_path", f"{DATA}/{t}/{sp}.csv",
                                    "--save_path", npz, "--features_generator", "rdkit_2d_normalized", "--restart"],
                                   cwd=HERE, env=env).returncode
    save_dir = os.path.join(OUT, f"out_{t}")
    cmd = ["python", "main.py", "finetune",
           "--data_path", f"{DATA}/{t}/train.csv",
           "--separate_val_path", f"{DATA}/{t}/val.csv",
           "--separate_test_path", f"{DATA}/{t}/test.csv",
           "--features_path", f"{FEAT}/{t}/train.npz",
           "--separate_val_features_path", f"{FEAT}/{t}/val.npz",
           "--separate_test_features_path", f"{FEAT}/{t}/test.npz",
           "--checkpoint_path", CKPT, "--save_dir", save_dir,
           "--dataset_type", DTYPE[t], "--split_type", "scaffold_balanced",
           "--ensemble_size", "1", "--num_folds", "1", "--no_features_scaling",
           "--ffn_hidden_size", "200", "--batch_size", "32", "--epochs", epochs,
           "--init_lr", "0.00015", "--show_individual_scores", "--gpu", "0"]
    logp = os.path.join(OUT, f"{t}.log")
    with open(logp, "w") as lf:
        r = subprocess.run(cmd, cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)
    res = None
    try:
        for line in open(logp):
            if "overall_" in line and "_test_" in line:
                res = line.strip()
            elif res is None and ("Ensemble test" in line or "Model 0 test" in line):
                res = line.strip()
    except Exception:
        pass
    return t, {"rc": r.returncode, "feat_rc": rc_f, "result_line": res, "sec": round(time.time() - t0), "gpu": gpu}


def main():
    from concurrent.futures import ThreadPoolExecutor
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--epochs", default="20")
    a, _ = ap.parse_known_args()
    tasks = ORDER if a.tasks.strip() == "all" else a.tasks.split()
    os.makedirs(OUT, exist_ok=True)
    ngpu = max(1, torch.cuda.device_count())
    print(f"nvidia: {ngpu} GPUs; tasks={tasks}", flush=True)
    # big tasks first so they grab GPUs early; round-robin GPU assignment across <=ngpu concurrent
    order = [t for t in ["hiv", "tox21", "sider", "lipo", "bbbp", "bace", "clintox", "esol", "freesolv"] if t in tasks]
    results = {}
    with ThreadPoolExecutor(max_workers=ngpu) as ex:
        futs = [ex.submit(run_task, t, a.epochs, i % ngpu) for i, t in enumerate(order)]
        for f in futs:
            t, info = f.result()
            results[t] = info
            print(f"[{t}] rc={info['rc']} feat_rc={info['feat_rc']} gpu={info['gpu']} :: {info['result_line']} ({info['sec']}s)", flush=True)
            json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print("SM_GROVER_DONE", flush=True)


if __name__ == "__main__":
    main()
