"""Aggregate ChemDFM-13B zero-shot inference results -> analysis/chemdfm_results.json + RESULTS.md.

Pulls per-task results from s3://molerag/chemdfm_baseline/output/incremental/<task>.json (written live
by entry.py, so present even if a job later failed), with each job's model.tar.gz runs/<task>.json as
a cross-check/fallback. Sums BillableTimeInSeconds across jobs.
Usage: python aggregate.py   (reads sm/chemdfm_jobs.json)"""
import boto3, json, os, io, tarfile, sys

BUCKET = "molerag"
INCR = "chemdfm_baseline/output/incremental"
OUT_PREFIX = "chemdfm_baseline/output"
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
META = json.load(open(f"{BASE}/../dmpnn_baseline/data/meta.json"))
TASK_ORDER = ["bbbp", "bace", "hiv", "clintox", "sider", "tox21", "esol", "freesolv", "lipo"]

s3 = boto3.client("s3", region_name="us-east-1")
sm = boto3.client("sagemaker", region_name="us-east-1")


def get_incr(task):
    try:
        o = s3.get_object(Bucket=BUCKET, Key=f"{INCR}/{task}.json")
        return json.loads(o["Body"].read())
    except Exception:
        return None


def get_from_tarballs(jobs):
    found = {}
    for j in jobs:
        key = f"{OUT_PREFIX}/{j}/output/model.tar.gz"
        try:
            o = s3.get_object(Bucket=BUCKET, Key=key)
            tf = tarfile.open(fileobj=io.BytesIO(o["Body"].read()))
            for m in tf.getmembers():
                if m.name.startswith("runs/") and m.name.endswith(".json"):
                    r = json.loads(tf.extractfile(m).read())
                    found[r["task"]] = r
        except Exception:
            pass
    return found


def main():
    try:
        reg = json.load(open(f"{BASE}/sm/chemdfm_jobs.json"))
    except Exception:
        reg = []
    jobs = [r["job"] for r in reg]
    tarballs = get_from_tarballs(jobs)

    results = {}
    missing = []
    for task in TASK_ORDER:
        r = get_incr(task) or tarballs.get(task)
        if r is None or r.get("status") != "ok" or r.get("value") is None:
            results[task] = {"type": META[task]["task_type"],
                             "metric": "RMSE" if META[task]["task_type"] == "regression" else "AUC",
                             "value": None, "status": (r or {}).get("status", "absent")}
            missing.append(f"{task} ({(r or {}).get('status', 'absent')})")
            continue
        results[task] = r

    # billable seconds
    billing = {}
    for rec in reg:
        try:
            d = sm.describe_training_job(TrainingJobName=rec["job"])
            billing[rec["job"]] = {"tag": rec.get("tag"), "instance": rec.get("instance"),
                                   "status": d["TrainingJobStatus"],
                                   "billable_s": d.get("BillableTimeInSeconds")}
        except Exception as e:
            billing[rec["job"]] = {"error": str(e)[:120]}
    total_billable = sum((v.get("billable_s") or 0) for v in billing.values())

    os.makedirs(HERE, exist_ok=True)
    out = {
        "label": "ChemDFM (LLaMA-13B, zero-shot inference)",
        "model": "OpenDFM/ChemDFM-v1.0-13B",
        "inference": "released merged checkpoint, fp16, single greedy pass, no fine-tuning, no seeds",
        "results": results, "billing": billing,
        "total_billable_seconds": total_billable, "missing": missing,
        "notes": ("Zero-shot. Prompts for bbbp/bace/hiv/clintox/tox21 are the EXACT ChemLLMBench "
                  "property-prediction prompts released in the ChemDFM repo (verbatim), wrapped in "
                  "ChemDFM's dialogue format. sider + esol/freesolv/lipo use ChemDFM-STYLE wording "
                  "(ChemDFM released no prompt for those). Classification: ROC-AUC from P(positive "
                  "answer token) = softmax of first greedy-step {Yes,No} logits, mean over endpoints "
                  "(missing labels masked). Regression: parse generated float, impute train-mean if "
                  "unparseable, RMSE."),
    }
    json.dump(out, open(f"{HERE}/chemdfm_results.json", "w"), indent=2)
    print("wrote analysis/chemdfm_results.json")

    lines = [
        "# ChemDFM (LLaMA-13B, zero-shot inference) — MoleculeNet (MolE-RAG seed-0 splits)",
        "",
        "Released merged checkpoint **`OpenDFM/ChemDFM-v1.0-13B`** (ChemDFM, Zhao et al., *Cell Rep.",
        "Phys. Sci.* 2025; built on LLaMA-1-13B) run **zero-shot** (fp16, single greedy pass, no",
        "fine-tuning, no seeds) on the MolE-RAG seed-0 **test** molecules. Prompts for",
        "**bbbp / bace / hiv / clintox / tox21** are the EXACT ChemLLMBench property-prediction prompts",
        "released in the ChemDFM GitHub repo (verbatim), wrapped in ChemDFM's `[Round 0]\\nHuman: ...",
        "\\nAssistant:` dialogue format. **sider** (27 ep) and the regression tasks **esol / freesolv /",
        "lipo** have NO ChemDFM-released prompt, so they use ChemDFM-STYLE wording (same instruction",
        "shell) and are flagged accordingly. SMILES canonicalized with rdkit per the repo README.",
        "Classification ROC-AUC = probability of the positive answer token (softmax of the first",
        "generated-step {Yes,No} logits), mean over endpoints (missing labels masked). Regression =",
        "parse the generated number, impute train-mean if unparseable, RMSE. Ran on SageMaker",
        "(L40S 48GB, PyTorch 2.3 DLC + transformers 4.44.2).",
        "",
        "| task | type | metric | value | prompt |",
        "|---|---|---|---|---|",
    ]
    prov_short = {True: "ChemDFM released", False: "ours (ChemDFM-style)"}
    for task in TASK_ORDER:
        r = results[task]
        v = r.get("value")
        cell = f"{v:.3f}" if isinstance(v, (int, float)) else "—"
        released = task in ("bbbp", "bace", "hiv", "clintox", "tox21")
        lines.append(f"| {task} | {META[task]['task_type']} | "
                     f"{r.get('metric','AUC' if META[task]['task_type']=='classification' else 'RMSE')} "
                     f"| {cell} | {prov_short[released]} |")
    lines += [
        "",
        f"Total SageMaker BillableTimeInSeconds across jobs: **{total_billable}** "
        f"(~{total_billable/3600:.2f} GPU-hours).",
        "",
        "Per-task detail (per-endpoint AUC, accuracy, unparseable counts, exact prompt provenance) in",
        "`analysis/chemdfm_results.json`; live per-task JSON in",
        "s3://molerag/chemdfm_baseline/output/incremental/ and raw run outputs in each job's",
        "model.tar.gz (s3://molerag/chemdfm_baseline/output/<job>/output/model.tar.gz).",
    ]
    if missing:
        lines.append(f"\nMissing/failed tasks: {', '.join(missing)}.")
    open(f"{BASE}/RESULTS.md", "w").write("\n".join(lines) + "\n")
    print("wrote RESULTS.md")
    print("\n".join(lines))
    if missing:
        print("\nWARNING missing:", missing)


if __name__ == "__main__":
    main()
