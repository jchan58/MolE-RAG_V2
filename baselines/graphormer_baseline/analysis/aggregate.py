"""Aggregate Graphormer fine-tune results from SageMaker job artifacts.
Reads results.jsonl from each job's model.tar.gz, aggregates per task (mean/std over seeds),
and writes graphormer_results.json + RESULTS.md. Also sums BillableTimeInSeconds.
Usage: python aggregate.py [job1 job2 ...]   (defaults to jobs in sm/graphormer_jobs.json)"""
import sys, os, json, io, tarfile, statistics
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

def job_names():
    if len(sys.argv) > 1:
        return sys.argv[1:]
    reg = json.load(open(os.path.join(ROOT, "sm", "graphormer_jobs.json")))
    return [r["job"] for r in reg]

def fetch_results(job):
    key = f"graphormer_baseline/output/{job}/output/model.tar.gz"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
    except Exception as e:
        print(f"  WARN: no artifact for {job}: {e}", flush=True)
        return []
    buf = io.BytesIO(obj["Body"].read())
    recs = []
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith("results.jsonl"):
                data = tf.extractfile(m).read().decode()
                for line in data.strip().splitlines():
                    if line.strip():
                        recs.append(json.loads(line))
    return recs

def main():
    jobs = job_names()
    all_recs = []
    billable = 0
    for j in jobs:
        try:
            d = sm.describe_training_job(TrainingJobName=j)
            b = d.get("BillableTimeInSeconds") or 0
            billable += b
            print(f"job {j}: {d['TrainingJobStatus']} billable={b}", flush=True)
        except Exception as e:
            print(f"job {j}: describe failed {e}", flush=True)
        all_recs += fetch_results(j)

    # group by dataset -> {seed: test_best}
    by_task = {}
    for r in all_recs:
        t = r["dataset"]
        by_task.setdefault(t, {})[int(r["runseed"])] = r

    results = {}
    for t in ORDER:
        if t not in by_task:
            continue
        runs = by_task[t]
        seeds = sorted(runs.keys())
        vals = [runs[s]["test_best"] for s in seeds]
        metric = runs[seeds[0]]["metric"]
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0  # population std over the 3 seeds
        sstd = statistics.stdev(vals) if len(vals) > 1 else 0.0  # sample std
        results[t] = {"type": TASK_TYPE[t], "metric": metric,
                      "mean": mean, "std": sstd, "pstd": std,
                      "per_seed": {str(s): runs[s]["test_best"] for s in seeds},
                      "val_per_seed": {str(s): runs[s]["val_best"] for s in seeds},
                      "n_seeds": len(seeds),
                      "n_test": runs[seeds[0]].get("n_test"),
                      "n_train": runs[seeds[0]].get("n_train"),
                      "epochs": runs[seeds[0]].get("epochs"),
                      "lr": runs[seeds[0]].get("lr")}
    out = {"model": "Graphormer (PCQM4Mv2-pretrained, fine-tuned)",
           "billable_time_seconds_total": billable,
           "jobs": jobs, "tasks": results}
    json.dump(out, open(os.path.join(HERE, "graphormer_results.json"), "w"), indent=2)
    print("WROTE graphormer_results.json", flush=True)

    # RESULTS.md
    lines = []
    lines.append("# Graphormer (PCQM4Mv2-pretrained, fine-tuned) — MoleculeNet (MolE-RAG seed-0 splits)\n")
    lines.append("Graphormer-base initialized from Microsoft's **PCQM4Mv2** pretrained checkpoint "
                 "(`checkpoint_best_pcqm4mv2.pt`), then supervised **fine-tuned per task** on the MolE-RAG\n"
                 "seed-0 **train** split (canonical-SMILES match), best checkpoint by **validation**, evaluated\n"
                 "on the MolE-RAG seed-0 **test** molecules. 3 initialization seeds (0,1,2), mean ± std.\n"
                 "Multitask datasets (clintox/tox21/sider) use one output head per endpoint; ROC-AUC is the\n"
                 "mean over endpoints. Ran on SageMaker (legacy PyTorch 1.9.1 DLC + reconstructed PyG 1.7.2 stack;\n"
                 "graphormer spatial encoding compiled in-container via Cython/pyximport).\n")
    lines.append("| task | type | metric | mean ± std (3 seeds) |")
    lines.append("|---|---|---|---|")
    for t in ORDER:
        if t not in results:
            continue
        r = results[t]
        lines.append(f"| {t} | {r['type']} | {r['metric']} | {r['mean']:.3f} ± {r['std']:.3f} |")
    lines.append("")
    lines.append(f"Total SageMaker BillableTimeInSeconds across jobs: **{billable}** "
                 f"(~{billable/3600:.1f} GPU-hours).\n")
    lines.append("Per-seed values + config in `analysis/graphormer_results.json`; raw per-run outputs in the\n"
                 "SageMaker model artifacts (s3://molerag/graphormer_baseline/output/<job>/output/model.tar.gz).\n"
                 "PCQM4Mv2 checkpoint staged at s3://molerag/graphormer_baseline/ckpt/checkpoint_best_pcqm4mv2.pt.")
    open(os.path.join(ROOT, "RESULTS.md"), "w").write("\n".join(lines) + "\n")
    print("WROTE RESULTS.md", flush=True)
    print("\n".join(lines))

if __name__ == "__main__":
    main()
