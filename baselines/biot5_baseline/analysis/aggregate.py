"""Aggregate BioT5 fine-tune results from SageMaker job artifacts.
Reads results.jsonl from each job's model.tar.gz (falls back to the incremental/ S3 records),
takes the single-seed (seed 0) dataset-level value per task, writes biot5_results.json + RESULTS.md,
and sums BillableTimeInSeconds. Usage: python aggregate.py [job1 job2 ...] (defaults to biot5_jobs.json)."""
import sys, os, json, io, tarfile
import boto3

BUCKET = "molerag"
REGION = "us-east-1"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
s3 = boto3.client("s3", region_name=REGION)
sm = boto3.client("sagemaker", region_name=REGION)

TASK_TYPE = {"bbbp": "classification", "bace": "classification", "hiv": "classification",
             "clintox": "classification", "sider": "classification", "tox21": "classification",
             "esol": "regression", "freesolv": "regression", "lipo": "regression"}
ORDER = ["bbbp", "bace", "hiv", "clintox", "sider", "tox21", "esol", "freesolv", "lipo"]
LABEL = "BioT5 (biot5-base, fine-tuned)"


def job_names():
    if len(sys.argv) > 1:
        return sys.argv[1:]
    reg = json.load(open(os.path.join(ROOT, "sm", "biot5_jobs.json")))
    return [r["job"] for r in reg if r.get("smoke") != "1"]


def fetch_from_tar(job):
    key = f"biot5_baseline/output/{job}/output/model.tar.gz"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
    except Exception as e:
        print(f"  no artifact for {job}: {e}", flush=True)
        return []
    buf = io.BytesIO(obj["Body"].read())
    recs = []
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith("results.jsonl"):
                for line in tf.extractfile(m).read().decode().strip().splitlines():
                    if line.strip():
                        recs.append(json.loads(line))
    return recs


def fetch_incremental():
    recs = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix="biot5_baseline/output/incremental/"):
        for o in page.get("Contents", []):
            try:
                recs.append(json.loads(s3.get_object(Bucket=BUCKET, Key=o["Key"])["Body"].read()))
            except Exception:
                pass
    return recs


def main():
    jobs = job_names()
    all_recs, billable = [], 0
    for j in jobs:
        try:
            d = sm.describe_training_job(TrainingJobName=j)
            b = d.get("BillableTimeInSeconds") or 0
            billable += b
            print(f"job {j}: {d['TrainingJobStatus']} billable={b}", flush=True)
        except Exception as e:
            print(f"job {j}: describe failed {e}", flush=True)
        all_recs += fetch_from_tar(j)
    # supplement with incremental records (covers jobs still running / artifacts missing)
    all_recs += fetch_incremental()

    # Recompute dataset values from ENDPOINT records (seed 0) directly, so datasets split across
    # multiple jobs (e.g. sider) aggregate correctly regardless of per-job dataset records.
    import numpy as np
    ep_recs = {}
    for r in all_recs:
        if r.get("kind") == "endpoint" and int(r.get("seed", 0)) == 0 and r.get("test"):
            ep_recs.setdefault(r["dataset"], {})[r["target"]] = r

    results = {}
    for t in ORDER:
        eps = ep_recs.get(t, {})
        if not eps:
            continue
        mk = next(iter(eps.values()))["metric"]
        if TASK_TYPE[t] == "classification":
            scored = {tg: e["test"]["auroc"] for tg, e in eps.items()
                      if e["test"].get("auroc") is not None}
            excluded = [tg for tg, e in eps.items() if e["test"].get("auroc") is None]
        else:
            scored = {tg: e["test"]["rmse"] for tg, e in eps.items()}
            excluded = []
        if not scored:
            continue
        any_ep = next(iter(eps.values()))
        results[t] = {
            "type": TASK_TYPE[t], "metric": mk, "value": float(np.mean(list(scored.values()))),
            "seed": 0,
            "n_endpoints": len(eps), "n_scored": len(scored),
            "excluded_endpoints": excluded,
            "per_endpoint_test": scored,
            "selfies_version": any_ep.get("selfies_version"),
            "base_model": any_ep.get("base_model"),
            "budget": any_ep.get("budget"),
            "base_lr": any_ep.get("base_lr"),
            "eff_batch": any_ep.get("eff_batch"),
            "endpoint_counts": {tg: e.get("counts") for tg, e in eps.items()},
            "endpoint_best_step": {tg: e.get("best_step") for tg, e in eps.items()},
        }
    out = {"model": LABEL, "seeds": [0], "billable_time_seconds_total": billable,
           "jobs": jobs, "tasks": results}
    json.dump(out, open(os.path.join(HERE, "biot5_results.json"), "w"), indent=2)
    print("WROTE biot5_results.json", flush=True)

    lines = []
    lines.append(f"# {LABEL} — MoleculeNet (MolE-RAG seed-0 splits)\n")
    lines.append("Pretrained **`QizhiPei/biot5-base`** (BioT5, Pei et al., NeurIPS 2023) "
                 "**fine-tuned per (sub)task** on the MolE-RAG seed-0 **train** split, best checkpoint by "
                 "**validation**, evaluated on the MolE-RAG seed-0 **test** molecules. **Single seed (0).**\n")
    lines.append("Faithful to BioT5's `molnet` pipeline: molecule **SELFIES** input "
                 "`Molecule: <bom>...<eom>` built with BioT5's exact Natural-Instructions template and its "
                 "own per-dataset task **Definition** strings (from QizhiPei/BioT5_finetune_dataset); only "
                 "the molecules/labels are swapped in. Model/tokenizer/data-collator/AdamW/cosine-schedule "
                 "are BioT5's own `biot5/utils`. Classification **ROC-AUC** uses BioT5's exact scoring "
                 "`logit(Yes)/(logit(Yes)+logit(No))` on the first generated token. Multitask "
                 "datasets (clintox/tox21/sider) = one independent fine-tune per endpoint, score = **mean "
                 "ROC-AUC over endpoints** whose test split has both classes.\n")
    lines.append("Regression (esol/freesolv/lipo) is **not part of BioT5's molnet pipeline** (it only "
                 "emits Yes/No); results marked `*` use a minimal added regression variant (model generates "
                 "the train-standardized value as text, parsed and de-standardized; metric RMSE).\n")
    lines.append("| task | type | metric | value (seed 0) |")
    lines.append("|---|---|---|---|")
    for t in ORDER:
        if t not in results:
            continue
        r = results[t]
        star = "*" if r["type"] == "regression" else ""
        extra = ""
        if r.get("n_endpoints") and r["n_endpoints"] > 1:
            extra = f" ({r['n_scored']}/{r['n_endpoints']} endpoints)"
        lines.append(f"| {t}{star} | {r['type']} | {r['metric']} | {r['value']:.3f}{extra} |")
    lines.append("")
    lines.append(f"Total SageMaker BillableTimeInSeconds across jobs: **{billable}** "
                 f"(~{billable/3600:.1f} GPU-hours).\n")
    lines.append("Per-endpoint values, SELFIES-skip counts, best-val step, and per-task config in "
                 "`analysis/biot5_results.json`; raw per-run outputs in the SageMaker model artifacts "
                 "(s3://molerag/biot5_baseline/output/<job>/output/model.tar.gz) and incremental records at "
                 "s3://molerag/biot5_baseline/output/incremental/.")
    open(os.path.join(ROOT, "RESULTS.md"), "w").write("\n".join(lines) + "\n")
    print("WROTE RESULTS.md", flush=True)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
