"""Aggregate LlaSMol (Mistral-7B) inference results -> analysis/llasmol_results.json + RESULTS.md.

Pulls each supported task's per-task JSON from s3://molerag/llasmol_baseline/output/incremental/<task>.json
(streamed live by naacl_entry.py), with each job's model.tar.gz results.jsonl as a cross-check/fallback.
Single deterministic inference pass (no seeds). Also sums BillableTimeInSeconds."""
import boto3, json, os, io, tarfile

BUCKET = "molerag"
INCR = "llasmol_baseline/output/incremental"
OUT_PREFIX = "llasmol_baseline/output"
BASE = "/work/hdd/bhrm/jchan5/mole-rag_naacl/llasmol_baseline"

SUPPORTED = {"esol": "regression", "lipo": "regression", "freesolv": "regression",
             "bbbp": "classification", "bace": "classification", "hiv": "classification",
             "clintox": "classification", "sider": "classification", "tox21": "classification"}
LLASMOL_SUPPORTED = ["esol", "lipo", "bbbp", "clintox", "hiv", "sider"]
UNSUPPORTED = {
    "freesolv": "not in SMolInstruct (its PP regression subset = ESOL + Lipophilicity only)",
    "bace": "not in SMolInstruct property-prediction subset",
    "tox21": "not in SMolInstruct property-prediction subset",
}
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
                if m.name.endswith("results.jsonl"):
                    for line in tf.extractfile(m).read().decode().strip().splitlines():
                        if line.strip():
                            r = json.loads(line)
                            found[r["task"]] = r
        except Exception:
            pass
    return found


def main():
    try:
        jobrecs = json.load(open(f"{BASE}/sm/llasmol_jobs.json"))
    except Exception:
        jobrecs = []
    jobs = [j["job"] for j in jobrecs if j.get("smoke") != "1"]
    tarballs = get_from_tarballs(jobs)

    results = {}
    for task in LLASMOL_SUPPORTED:
        r = get_incr(task) or tarballs.get(task)
        results[task] = r

    billing = {}
    for jr in jobrecs:
        try:
            d = sm.describe_training_job(TrainingJobName=jr["job"])
            billing[jr["job"]] = {"instance": jr.get("instance"), "tasks": jr.get("tasks"),
                                  "smoke": jr.get("smoke"), "status": d["TrainingJobStatus"],
                                  "billable_s": d.get("BillableTimeInSeconds")}
        except Exception as e:
            billing[jr["job"]] = {"instance": jr.get("instance"), "tasks": jr.get("tasks"),
                                  "status": f"describe_failed:{str(e)[:80]}", "billable_s": None}
    total_billable = sum((v["billable_s"] or 0) for v in billing.values() if v.get("smoke") != "1")

    out = {"label": "LlaSMol (Mistral-7B)", "mode": "released-checkpoint inference (no fine-tuning, no seeds)",
           "supported_tasks": LLASMOL_SUPPORTED, "unsupported_tasks": UNSUPPORTED,
           "results": results, "billing": billing, "total_billable_seconds": total_billable,
           "notes": ("osunlp/LlaSMol-Mistral-7B = LoRA adapter (modules_to_save=[]) on openly-available "
                     "mistralai/Mistral-7B-v0.1, instruction-tuned on SMolInstruct; merged and run with "
                     "LlaSMol's own PP generation config (num_beams=4, 1 return seq, max_new_tokens=20, "
                     "deterministic). Prompts are SMolInstruct's EXACT verbatim templates with <SMILES> "
                     "core tags, SMILES canonicalized by LlaSMol's own canonicalize_molecule_smiles, chat "
                     "wrapped as <s>[INST] .. [/INST], answers extracted with the repo's TASK_TAGS. "
                     "Classification ROC-AUC headline = P(Yes) via sequence-LL of the two canonical "
                     "<BOOLEAN> responses (suite-comparable); LlaSMol-native hard-label AUC/accuracy "
                     "(calculate_boolean_metrics, verbatim) also reported. clintox evaluated against the "
                     "CT_TOX endpoint ('Is X toxic?', verified 16/16). sider = mean AUC over the 20/27 "
                     "endpoints SMolInstruct has templates for (7 masked). Regression RMSE headline imputes "
                     "the train-target mean for unparseable outputs (over all test molecules); native RMSE "
                     "(drops unparseable) + count also reported. WARNING: LlaSMol was instruction-tuned on "
                     "these MoleculeNet PP datasets, and 74-92% of our scaffold-split TEST molecules appear "
                     "in SMolInstruct's train/dev -- these numbers are NOT held-out generalization.")}
    os.makedirs(f"{BASE}/analysis", exist_ok=True)
    json.dump(out, open(f"{BASE}/analysis/llasmol_results.json", "w"), indent=2, default=str)
    print("wrote analysis/llasmol_results.json")

    def cell(task):
        r = results.get(task)
        if not r or r.get("status") != "ok" or r.get("value") is None:
            return "—"
        if r["task_type"] == "classification":
            return f"{r['value']:.3f}"
        return f"{r['value']:.3f}"

    lines = [
        "# LlaSMol (Mistral-7B) — MoleculeNet (MolE-RAG seed-0 splits)",
        "",
        "`osunlp/LlaSMol-Mistral-7B` is a **LoRA adapter** (`modules_to_save=[]`) on the openly available",
        "base `mistralai/Mistral-7B-v0.1`, instruction-tuned on **SMolInstruct**. This is **released-checkpoint",
        "inference** — the adapter is merged onto the base and run with LlaSMol's **own** property-prediction",
        "generation config (`num_beams=4`, 1 return sequence, `max_new_tokens=20`, deterministic). **No",
        "fine-tuning, no seeds.** Evaluated on the MolE-RAG seed-0 **test** molecules (same as the other baselines).",
        "",
        "Prompts are SMolInstruct's **exact verbatim templates** (bundled `smol_templates/`): each test molecule",
        "gets a template by cycling the task's full template set (`index mod n_templates`), the SMILES wrapped in",
        "the dataset `<SMILES> … </SMILES>` core tag and canonicalized by LlaSMol's own",
        "`canonicalize_molecule_smiles`, chat-wrapped as `<s>[INST] … [/INST]`, answers extracted with the repo's",
        "`TASK_TAGS` (`<NUMBER>`/`<BOOLEAN>`).",
        "",
        "**Coverage:** SMolInstruct's property-prediction subset has only **6 of the 9** MoleculeNet tasks.",
        "FreeSolv, BACE, Tox21 are **not in SMolInstruct** → not runnable faithfully (no fabricated numbers).",
        "`clintox` is scored against the **CT_TOX** endpoint (LlaSMol asks \"Is X toxic?\"; verified 16/16 on",
        "overlapping molecules). `sider` is the **mean AUC over the 20 of 27** SIDER endpoints SMolInstruct has",
        "templates for (the other 7 organ systems have no LlaSMol template and are masked).",
        "",
        "**Classification ROC-AUC (headline)** = probability of the positive answer `P(Yes)` via the sequence",
        "log-likelihood of the two canonical `<BOOLEAN>` responses, softmax, AUC per endpoint then averaged",
        "(suite-comparable). LlaSMol's **native hard-label** AUC/accuracy (`calculate_boolean_metrics`, copied",
        "verbatim) is in `analysis/llasmol_results.json`. **Regression RMSE (headline)** parses `<NUMBER>` and",
        "imputes the train-target mean for unparseable outputs over all test molecules; native RMSE (drops",
        "unparseable) + counts also recorded.",
        "",
        "> **Leakage caveat:** LlaSMol was instruction-tuned on these MoleculeNet PP datasets, and **74–92% of our",
        "> scaffold-split test molecules appear in SMolInstruct's train/dev**. These numbers are an as-released",
        "> model applied to our test molecules, **not** held-out generalization.",
        "",
        "| task | type | metric | value |",
        "|---|---|---|---|",
    ]
    metric_name = {"classification": "ROC-AUC", "regression": "RMSE"}
    for task in TASK_ORDER:
        if task in UNSUPPORTED:
            lines.append(f"| {task} | {SUPPORTED[task]} | — | not supported ({UNSUPPORTED[task]}) |")
        else:
            lines.append(f"| {task} | {SUPPORTED[task]} | {metric_name[SUPPORTED[task]]} | {cell(task)} |")
    lines += [
        "",
        f"Total SageMaker BillableTimeInSeconds: {total_billable} (~{total_billable/3600:.1f} GPU-hours).",
        "Per-task config, per-endpoint AUCs, native hard-label metrics, no-answer/unparseable counts, and the",
        "exact prompt provenance are in `analysis/llasmol_results.json`; raw per-unit model outputs are in the",
        "SageMaker model artifact (`s3://molerag/llasmol_baseline/output/<job>/output/model.tar.gz`, `runs/*.jsonl`)",
        "and live per-task JSON in `s3://molerag/llasmol_baseline/output/incremental/`.",
    ]
    open(f"{BASE}/RESULTS.md", "w").write("\n".join(lines) + "\n")
    print("wrote RESULTS.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
