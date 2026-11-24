#!/usr/bin/env python3
"""SageMaker entry point for KANO finetuning (runs inside the PyTorch 2.1 DLC).

cwd is /opt/ml/code (the unpacked sourcedir). Our prepared seed_0 wide CSVs are at
./sm_data/<task>/{train,val,test}.csv. Loops the 9 MoleculeNet tasks, runs KANO
train.py for each (same args as the Delta sbatch), captures the final test metric,
and writes results.json + per-task logs to /opt/ml/model (-> output S3).

Hyperparameters (passed by SageMaker as --key value): --encoder, --tag, --ckpt,
--epochs, --num_runs. Unknown sagemaker_* params are ignored.
"""
import os, json, subprocess, argparse, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "sm_data")
OUT = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
CLS = ["bbbp", "bace", "hiv", "clintox", "sider", "tox21"]
REG = ["esol", "freesolv", "lipo"]
DTYPE = {**{t: "classification" for t in CLS}, **{t: "regression" for t in REG}}
METRIC = {**{t: "auc" for t in CLS}, **{t: "rmse" for t in REG}}
ORDER = ["bbbp", "bace", "clintox", "esol", "freesolv", "lipo", "sider", "tox21", "hiv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="CMPNN")
    ap.add_argument("--tag", default="cmpnn")
    ap.add_argument("--ckpt", default="NONE")
    ap.add_argument("--epochs", default="100")
    ap.add_argument("--num_runs", default="1")
    a, _ = ap.parse_known_args()
    os.makedirs(OUT, exist_ok=True)
    print(f"KANO SM: encoder={a.encoder} tag={a.tag} ckpt={a.ckpt} epochs={a.epochs} runs={a.num_runs}", flush=True)
    print("nvidia-smi:", flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    results = {}
    for t in ORDER:
        cmd = ["python", "train.py",
               "--data_path", f"{DATA}/{t}/train.csv",
               "--separate_val_path", f"{DATA}/{t}/val.csv",
               "--separate_test_path", f"{DATA}/{t}/test.csv",
               "--dataset_type", DTYPE[t], "--metric", METRIC[t],
               "--epochs", a.epochs, "--num_runs", a.num_runs,
               "--gpu", "0", "--batch_size", "256", "--seed", "51", "--init_lr", "1e-4",
               "--split_type", "scaffold_balanced", "--step", "functional_prompt",
               "--encoder_name", a.encoder,
               "--exp_name", f"kano_{a.tag}", "--exp_id", t]
        if a.ckpt and a.ckpt != "NONE":
            cmd += ["--checkpoint_path", a.ckpt]
        logp = os.path.join(OUT, f"{t}.log")
        print(f"\n===== KANO-{a.tag} {t} =====", flush=True)
        t0 = time.time()
        with open(logp, "w") as lf:
            r = subprocess.run(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
        res = None
        try:
            for line in open(logp):
                if line.startswith("Results:") or "Overall test" in line:
                    res = line.strip()
        except Exception:
            pass
        results[t] = {"rc": r.returncode, "result_line": res, "sec": round(time.time() - t0)}
        print(f"[{t}] rc={r.returncode} :: {res} ({results[t]['sec']}s)", flush=True)
        json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print("SM_ENTRY_DONE", flush=True)


if __name__ == "__main__":
    main()
