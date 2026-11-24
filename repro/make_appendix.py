#!/usr/bin/env python3
"""Regenerate the appendix tables from the committed score files.

Every number in the appendix comes from a file under ``scores/appendix/``;
nothing here is transcribed by hand. Running this prints the LaTeX for each
table and writes the whole set to ``docs/appendix_tables.tex`` so the paper and
the repository cannot drift apart.

    python repro/make_appendix.py

Tables covered: corpus construction / index / coverage (App. A), selected
fingerprints (App. C), context ablation (App. E), retrieval vs. self-review
(App. F), and retrieval budget / inference cost (App. G), plus the evidence-use
summary (App. H). Two tables are handled elsewhere and noted at the bottom.
"""

import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "scores" / "appendix"

# Canonical display names and the dataset order used in the paper's tables.
NAME = {
    "bbbp": "BBBP", "bace": "BACE", "clintox": "ClinTox", "hiv": "HIV",
    "tox21": "Tox21", "sider": "SIDER", "esol": "ESOL",
    "freesolv": "FreeSolv", "lipo": "Lipophilicity",
}
MODEL = {
    "qwen3_32b": "Qwen3-32B", "ministral_3_14b": "Ministral-3-14B",
    "opus_4_6": "Claude Opus 4.6", "gpt5mini": "GPT-5-mini",
}
CORPUS_ORDER = ["bbbp", "bace", "clintox", "hiv", "tox21", "sider",
                "esol", "freesolv", "lipo"]

# Runs incomplete at the scoring snapshot, flagged "(partial)" in the paper:
# tox21/opus is recorded as partial in prediction_completeness.csv; the two HIV
# runs are from the review-ablation scope note (incomplete shards).
PARTIAL = {("hiv", "ministral_3_14b"), ("hiv", "opus_4_6"), ("tox21", "opus_4_6")}


def rows(name):
    with open(SRC / name, newline="") as fh:
        return list(csv.DictReader(fh))


def thousands(x):
    return f"{int(round(float(x))):,}"


def calls(x):
    """Round a per-molecule call count to two places, half-up like the paper."""
    return Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# --------------------------------------------------------------------------- A
def corpus_construction():
    data = {r["dataset"]: r for r in rows("corpus_construction_counts.csv")}
    body = []
    for ds in CORPUS_ORDER:
        r = data[ds]
        body.append(
            f"{NAME[ds]:<13} & {thousands(r['processed_articles'])} "
            f"& {thousands(r['paragraph_passages'])} "
            f"& {thousands(r['table_passages'])} "
            f"& {thousands(r['total_constructed_passages'])} \\\\"
        )
    return _table(
        cols="lrrrr",
        header=("\\textbf{Dataset} & \\textbf{Processed articles} "
                "& \\textbf{Paragraph passages} & \\textbf{Table passages} "
                "& \\textbf{Total passages} \\\\"),
        body=body,
        caption="Article and passage counts for the nine collections. "
                "Articles and passages are counted once within each dataset.",
        label="tab:corpus-construction",
    )


def corpus_index():
    data = {r["dataset"]: r for r in rows("evaluated_index_statistics.csv")}
    body = []
    for ds in CORPUS_ORDER:
        r = data[ds]
        body.append(
            f"{NAME[ds]:<13} & {thousands(r['task_entities'])} "
            f"& {thousands(r['raw_findings_total(=meta.n_findings)'])} "
            f"& {thousands(r['distinct_findings_by_id'])} "
            f"& {thousands(r['all_supporting_passages_union'])} "
            f"& {thousands(r['supporting_articles'])} \\\\"
        )
    return _table(
        cols="lrrrrr",
        header=("\\textbf{Dataset} & \\textbf{Task entries} "
                "& \\textbf{Finding records} & \\textbf{Unique finding IDs} "
                "& \\textbf{Supporting passages} & \\textbf{Source articles} \\\\"),
        body=body,
        caption="Findings and supporting sources in the indexes used for "
                "prediction. Passages and articles shared by task- and "
                "compound-level findings are counted once.",
        label="tab:corpus-index",
    )


def _pct(x):
    x = float(x)
    return f"{x:.1f}" if x >= 1.0 else f"{x:.2f}"


def compound_coverage():
    data = {r["dataset"]: r for r in rows("molecule_coverage.csv")}
    body = []
    for ds in CORPUS_ORDER:
        r = data[ds]
        tw, tr = int(r["test_with_findings"]), int(r["test_records"])
        body.append(
            f"{NAME[ds]:<13} & {thousands(r['cards'])} "
            f"& {thousands(r['molecules_with_findings'])} "
            f"& {_pct(r['card_coverage_pct(findings/cards)'])}\\% "
            f"& {tw:,} / {tr:,} "
            f"& {float(r['benchmark_coverage_test_pct(test findings/test records)']):.1f}\\% \\\\"
        )
    return _table(
        cols="lrrrlr",
        header=("\\textbf{Dataset} & \\textbf{Compound cards} "
                "& \\textbf{With findings} & \\textbf{Coverage} "
                "& \\textbf{Covered test records / total} & \\textbf{Test coverage} \\\\"),
        body=body,
        caption="Molecules with at least one indexed literature finding. "
                "Compound cards represent distinct standardized structures "
                "across all splits.",
        label="tab:compound-evidence-coverage",
    )


# --------------------------------------------------------------------------- C
def fingerprints():
    order = ["bace", "bbbp", "clintox", "esol", "freesolv", "hiv",
             "lipo", "sider", "tox21"]
    data = {r["dataset"]: r for r in rows("fingerprints_manuscript_vs_actual.csv")}
    body = []
    for ds in order:
        r = data[ds]
        body.append(
            f"{NAME[ds]:<13} & {r['manuscript_fingerprint']:<20} "
            f"& {r['validation_selected_k']} \\\\"
        )
    return _table(
        cols="llr",
        header=("\\textbf{Dataset} & \\textbf{Selected fingerprint} "
                "& \\textbf{Selection $k$} \\\\"),
        body=body,
        caption="Dataset-specific fingerprints selected using GPT-4o-mini "
                "validation performance. The task-specific reference pools used "
                "subsequently contain five references per query.",
        label="tab:best_fingerprints",
    )


# --------------------------------------------------------------------------- E
def context_ablation():
    vals = {}  # (ds, cond) -> value
    for r in rows("context_ablation_scores.csv"):
        if r["status"] == "scored":
            vals[(r["ds"], r["condition"])] = float(r["value"])

    def cell(ds, cond):
        v = vals[(ds, cond)]
        return f"{v * 100:.2f}" if ds in ("bbbp", "clintox") else f"{v:.3f}"

    # best point estimate per column (higher AUC, lower RMSE)
    best = {
        "bbbp": max(vals[("bbbp", c)] for c in ("S", "R", "M", "MR", "MRD")),
        "freesolv": min(vals[("freesolv", c)] for c in ("S", "R", "M", "MR", "MRD")),
        "clintox": max(vals[("clintox", c)] for c in ("S", "R", "M", "MR", "MRD")),
    }

    def row(label, cond):
        out = [label]
        for ds in ("bbbp", "freesolv", "clintox"):
            txt = cell(ds, cond)
            if abs(vals[(ds, cond)] - best[ds]) < 1e-9:
                txt = f"\\textbf{{{txt}}}"
            out.append(txt)
        return " & ".join(out) + " \\\\"

    body = [row("$S$", "S"), row("$R$", "R"), row("$M$", "M"),
            row("$MR$", "MR"), row("$MRD$", "MRD"), "\\midrule",
            row("$MR$ w/o identifiers", "MR_no_id"),
            row("$MR$ w/o descriptors", "MR_no_desc"),
            row("$MR$ w/o groups", "MR_no_fg")]
    return _table(
        cols="lrrr",
        header="Condition & BBBP $\\uparrow$ & FreeSolv $\\downarrow$ & ClinTox $\\uparrow$ \\\\",
        body=body,
        caption="Initial-context ablations with Qwen3-32B, before literature "
                "retrieval. Classification scores are ROC-AUC percentages; "
                "FreeSolv reports RMSE. Bold marks the best point estimate.",
        label="tab:context-ablation",
    )


# --------------------------------------------------------------------------- F
def review_comparison():
    cls = ["bbbp", "hiv", "tox21", "sider", "clintox"]
    reg = ["esol", "freesolv", "lipo"]
    models = ["qwen3_32b", "ministral_3_14b", "opus_4_6", "gpt5mini"]
    by = {}
    for r in rows("review_paired_contrasts.csv"):
        if r["comparison"] == "operational" and r["contrast"] == "C_minus_B":
            by[(r["dataset"], r["model"])] = r

    def block(dslist):
        out = []
        for ds in dslist:
            for i, m in enumerate(models):
                r = by.get((ds, m))
                if not r:
                    continue
                lead = NAME[ds] if i == 0 else ""
                d = float(r["delta"])
                ci = f"[{float(r['ci_lo']):.3f}, {float(r['ci_hi']):.3f}]"
                mlabel = MODEL[m] + (" (partial)" if (ds, m) in PARTIAL else "")
                out.append(f"{lead:<5} & {mlabel:<26} & {int(r['n_paired']):,} "
                           f"& {d:+.3f} & {ci} \\\\")
            out.append("\\addlinespace")
        return out

    body = ["\\multicolumn{5}{l}{\\textit{Classification: "
            "$\\mathrm{AUC}_{C}-\\mathrm{AUC}_{B}$}} \\\\", "\\addlinespace"]
    body += block(cls)
    body += ["\\midrule", "\\multicolumn{5}{l}{\\textit{Regression: "
             "$\\mathrm{RMSE}_{B}-\\mathrm{RMSE}_{C}$}} \\\\", "\\addlinespace"]
    body += block(reg)
    return _table(
        cols="llrrl",
        header=("\\textbf{Dataset} & \\textbf{Model} & \\textbf{Paired molecules} "
                "& \\textbf{Difference} & \\textbf{95\\% CI} \\\\"),
        body=body,
        caption="Evidence-assisted review ($C$) compared with self-review "
                "($B$). Positive differences favor evidence-assisted review.",
        label="tab:review-comparison",
        star=True,
    )


# --------------------------------------------------------------------------- G
def inference_cost():
    order = ["bbbp", "hiv", "tox21", "sider", "clintox", "esol", "freesolv", "lipo"]
    usage = {r["dataset"]: r for r in rows("budget_usage_summary.csv")
             if r["model"] == "qwen3_32b"}
    cost = {}  # ds -> stage -> row
    for r in rows("budget_cost_summary.csv"):
        if r["model"] == "qwen3_32b":
            cost.setdefault(r["dataset"], {})[r["stage"]] = r
    body = []
    for ds in order:
        n = int(usage[ds]["n_molecules"])
        c = cost[ds]
        pmc = lambda s: Decimal(c[s]["per_molecule_calls"])
        b_calls = pmc("INITIAL_ASSESSMENT") + pmc("SELF_REVIEW")
        c_calls = pmc("INITIAL_ASSESSMENT") + pmc("QUESTION_GENERATION") + pmc("EVIDENCE_ASSESSMENT")
        embed = pmc("EMBEDDING")
        c_in = sum(float(c[s]["input_tokens_est"]) for s in
                   ("INITIAL_ASSESSMENT", "QUESTION_GENERATION", "EVIDENCE_ASSESSMENT")) / n
        c_out = sum(float(c[s]["output_tokens_recounted"]) for s in
                    ("INITIAL_ASSESSMENT", "QUESTION_GENERATION", "EVIDENCE_ASSESSMENT")) / n
        body.append(
            f"{NAME[ds]:<13} & {float(usage[ds]['mean_rounds_executed']):.3f} "
            f"& {calls(b_calls)} & {calls(c_calls)} & {calls(embed)} "
            f"& {thousands(c_in)} & {thousands(c_out)} \\\\"
        )
    return _table(
        cols="lrrrrrr",
        header=("\\textbf{Dataset} & \\textbf{Mean rounds} & \\textbf{$B$ calls} "
                "& \\textbf{$C$ calls} & \\textbf{Embed calls} "
                "& \\textbf{$C$ input tokens} & \\textbf{$C$ output tokens} \\\\"),
        body=body,
        caption="Mean online resource use per molecule for Qwen3-32B. "
                "Generation-call counts for self-review ($B$) and "
                "evidence-assisted review ($C$) each include the initial "
                "assessment once.",
        label="tab:inference-budget-cost",
        star=True,
    )


# --------------------------------------------------------------------------- H
def evidence_use_summary():
    g = json.load(open(SRC / "evidence_use_aggregate.json"))["global"]
    it = json.load(open(SRC / "evidence_use_interpretation.json"))
    lines = ["% Appendix H -- evidence use and errors (prose statistics)."]

    def g1(*keys):
        for k in keys:
            if k in g:
                return g[k]
        return None
    pairs = [
        ("traces examined", g1("n_units", "n_traces")),
        ("retrieval attempts", g1("retrieval_attempts")),
        ("empty retrievals", g1("empty_retrieval_count", "empty_retrievals")),
        ("delivered passage events", g1("delivered_passages")),
        ("repeated passages", g1("repeated_passages")),
        ("assessed cases", it.get("n_prevalence_assessed")),
        ("revisions", it.get("n_revisions")),
        ("direction supported", it.get("direction_supported")),
        ("magnitude quant-anchored", it.get("magnitude_quant_anchored")),
        ("retentions", it.get("n_retentions")),
        ("supported retentions", it.get("supported_retention")),
    ]
    for label, val in pairs:
        if val is not None:
            lines.append(f"%   {label}: {val}")
    for dist in ("applicability_dist", "sufficiency_dist"):
        if dist in it:
            lines.append(f"%   {dist}: {json.dumps(it[dist])}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- util
def _table(cols, header, body, caption, label, star=False):
    env = "table*" if star else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", "\\small",
             f"\\begin{{tabular}}{{{cols}}}", "\\toprule", header, "\\midrule"]
    lines += body
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\caption{{{caption}}}", f"\\label{{{label}}}",
              f"\\end{{{env}}}"]
    return "\n".join(lines) + "\n"


TABLES = [
    ("A. Corpus construction", corpus_construction),
    ("A. Corpus index", corpus_index),
    ("A. Compound-evidence coverage", compound_coverage),
    ("C. Selected fingerprints", fingerprints),
    ("E. Context ablation", context_ablation),
    ("F. Retrieval vs. self-review", review_comparison),
    ("G. Retrieval budget and cost", inference_cost),
    ("H. Evidence-use summary", evidence_use_summary),
]


def main():
    out = []
    for title, fn in TABLES:
        block = fn()
        out.append(f"% ===== {title} =====\n{block}")
    text = "\n".join(out)
    text += (
        "\n% ----- handled separately -----\n"
        "% B (FreeSolv screening, tab:freesolv-screening-results): the 16 RMSE\n"
        "%   cells live only as prose in the screening notes; a dedicated scorer\n"
        "%   over runs_frozen/{c1,c2,c3} is needed before they can be emitted here.\n"
        "% D (RDKit descriptors, tab:rdkit_*): descriptor_rankings.csv holds the\n"
        "%   per-dataset |corr| ranking, but the correlation convention for the\n"
        "%   BACE rank-1 ordering must be pinned to match the manuscript.\n"
    )
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "appendix_tables.tex").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
