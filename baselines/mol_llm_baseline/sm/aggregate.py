"""Aggregate MolLM per-task results -> analysis/mol_llm_results.json + RESULTS.md.

Pulls live per-task results from s3://molerag/mol_llm_baseline/output/incremental/<task>_seed0.json
(written by entry.py, survive a failed job), with the final job's model.tar.gz summary as fallback.
One seed (0). Label 'MolLM (Gerstein Lab, 2D+3D)'."""
import boto3, json, os, io, tarfile

BUCKET = "molerag"
INCR = "mol_llm_baseline/output/incremental"
OUT_PREFIX = "mol_llm_baseline/output"
BASE = "/work/hdd/bhrm/jchan5/mole-rag_naacl/mol_llm_baseline"
META = json.load(open(f"{BASE}/../dmpnn_baseline/data/meta.json"))
SEED = 0
TASK_ORDER = ["bbbp", "bace", "hiv", "clintox", "sider", "tox21", "esol", "freesolv", "lipo"]

s3 = boto3.client("s3", region_name="us-east-1")
sm = boto3.client("sagemaker", region_name="us-east-1")


def get_incr(task):
    try:
        o = s3.get_object(Bucket=BUCKET, Key=f"{INCR}/{task}_seed{SEED}.json")
        return json.loads(o["Body"].read())
    except Exception:
        return None


def get_from_tarball():
    found = {}
    try:
        job = json.load(open("job_full.json"))["job"]
    except Exception:
        return found
    try:
        o = s3.get_object(Bucket=BUCKET, Key=f"{OUT_PREFIX}/{job}/output/model.tar.gz")
        tf = tarfile.open(fileobj=io.BytesIO(o["Body"].read()))
        for m in tf.getmembers():
            if m.name.startswith("runs/") and m.name.endswith(".json"):
                r = json.loads(tf.extractfile(m).read())
                found[r["task"]] = r
    except Exception:
        pass
    return found


def main():
    tarball = get_from_tarball()
    results, missing = {}, []
    for task in TASK_ORDER:
        r = get_incr(task) or tarball.get(task)
        mtype = META[task]["task_type"]
        metric = "RMSE" if mtype == "regression" else "AUC"
        if r is None or r.get("status") != "ok" or r.get("value") is None:
            missing.append(f"{task} ({(r or {}).get('status')})")
            results[task] = {"metric": metric, "type": mtype, "value": None}
            continue
        results[task] = {
            "metric": r.get("metric", metric), "type": mtype,
            "value": round(r["value"], 4), "val_value": round(r.get("val_value", 0), 4),
            "best_epoch": r.get("best_epoch"), "epochs_done": r.get("epochs_done"),
            "n_test": r.get("n_test"), "n_targets_scored": r.get("n_targets_scored"),
            "minutes": r.get("minutes"), "use_3d": r.get("use_3d"),
        }

    billing = {}
    for jf in ["job_full.json", "stage_job.json"]:
        try:
            j = json.load(open(jf))["job"]
            d = sm.describe_training_job(TrainingJobName=j)
            billing[j] = {"status": d["TrainingJobStatus"], "billable_s": d.get("BillableTimeInSeconds")}
        except Exception:
            pass
    total_billable = sum(v["billable_s"] or 0 for v in billing.values())

    os.makedirs(f"{BASE}/analysis", exist_ok=True)
    out = {"label": "MolLM (Gerstein Lab, 2D+3D)", "seed": SEED, "results": results,
           "billing": billing, "total_billable_seconds": total_billable, "missing": missing,
           "notes": ("Released MolLM molecular graph encoder (Transformer-M over OGB 2D features + "
                     "RDKit ETKDG 3D conformers), initialised from the released pretrained checkpoint "
                     "(HF andrewt28/MolLM, all_checkpoints/model-epoch=394.ckpt), fine-tuned end-to-end "
                     "on MolE-RAG seed-0 train, best checkpoint by validation, evaluated on MolE-RAG "
                     "seed-0 test. One seed (0). Multitask (clintox/tox21/sider): one model, "
                     "num_targets-wide head, masked BCE + mean ROC-AUC over endpoints. Regression: "
                     "train-set target standardisation, RMSE.")}
    json.dump(out, open(f"{BASE}/analysis/mol_llm_results.json", "w"), indent=2)
    print("wrote analysis/mol_llm_results.json")

    lines = [
        "# MolLM (Gerstein Lab, 2D+3D) — MoleculeNet (MolE-RAG seed-0 splits)",
        "",
        "Released **MolLM** unified model (Tang/Tran/Tan/Gerstein, Bioinformatics 2024): a molecular",
        "Transformer encoder over OGB 2D graph features **and** RDKit ETKDG 3D conformers, pretrained",
        "by contrastive learning against biomedical text. The released pretrained graph-encoder",
        "checkpoint (`all_checkpoints/model-epoch=394.ckpt`) is **fine-tuned** end-to-end with a fresh",
        "prediction head on the MolE-RAG seed-0 **train** split, best checkpoint by **validation**,",
        "evaluated on the MolE-RAG seed-0 **test** molecules (same test molecules as the other",
        "baselines). One seed (0). Multitask datasets (clintox/tox21/sider): one model with a",
        "num_targets-wide head and **masked BCE** (missing labels masked from loss AND metric); metric",
        "= mean ROC-AUC over endpoints. Regression uses train-set target standardisation; metric =",
        "RMSE. Ran on SageMaker (1 L40S GPU, PyTorch DLC).",
        "",
        "| task | type | metric | MolLM |",
        "|---|---|---|---|",
    ]
    for task in TASK_ORDER:
        r = results[task]
        cell = f"{r['value']:.3f}" if r["value"] is not None else "—"
        lines.append(f"| {task} | {r['type']} | {r['metric']} | {cell} |")
    lines += [
        "",
        "Per-task config (val score, best epoch, epochs run, #test) in `analysis/mol_llm_results.json`;",
        "live per-run JSON in s3://molerag/mol_llm_baseline/output/incremental/.",
        f"Total SageMaker BillableTimeInSeconds: {total_billable}.",
    ]
    if missing:
        lines.append(f"Missing/failed: {', '.join(missing)}.")
    open(f"{BASE}/RESULTS.md", "w").write("\n".join(lines) + "\n")
    print("wrote RESULTS.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
