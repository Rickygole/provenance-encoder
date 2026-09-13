"""Over-calling decomposed into TPR, FPR and prevalence.

WHY THIS EXISTS
---------------
"Over-calling ratio" (flag rate over base rate) is a single number that
conflates two different behaviours: a labeller with a high true positive rate
at a low base rate can carry the same ratio as one that simply flags almost
everything. Splitting the ratio into TPR and FPR, and checking the algebraic
identity that ties them back together,

    over_call = TPR + FPR * (1 - p) / p          (p = true prevalence)

lets a reader tell which behaviour produced a given ratio, and gives a
consistency check for every reported cell: if a cell's TPR, FPR and stated
over_call don't satisfy the identity within floating-point tolerance, either
the wrong prevalence was used to compute the ratio or the confusion counts
don't match the ratio that was reported next to them.

This script computes it for two populations:

  PUBLIC  -- the SNLI prevalence-sweep artifact (accord/prevalence_replication),
             three local 8B models x two prompts x four prevalence levels,
             24 cells with full TP/FP/FN/TN counts.
  CLINICAL -- the two local 8B labellers that were run under the frontier
             template with full confusion counts (granite3.3:8b, llama3:8b;
             qwen3:8b has no frontier-template confusion counts registered),
             plus the three baseline-prompt rows that carry only a positive
             rate, not a confusion breakdown -- for those, FPR (and TPR) are
             stated as not recoverable rather than invented.

A NOTE ON THE ABSTENTION PROMPT AND THE IDENTITY
-------------------------------------------------
Under prompt A_bare there is no abstention: the decided subset is the whole
100-item draw, and its realised prevalence equals the nominal target exactly
by construction of the pool. Under prompt B_baserate_abstain, items can be
withheld, and abstention is not label-neutral (Table in sweep_results.json
shows different abstain rates for positives and negatives), so the decided
subset's realised prevalence can drift away from the nominal target that the
publicly reported over_call_ratio divides by. The identity is checked BOTH
ways: once using the nominal target prevalence (which is what the source
artifact's own over_call_ratio divides by, and which can therefore fail under
abstention) and once using the realised decided-subset prevalence (which is a
tautology and should never fail; if it does, something else is wrong).

SNLI three-way label (entailment vs. neutral): the per-item CSVs under
accord/prevalence_replication/results/raw_predictions* and the pool files
under accord/prevalence_replication/data/pools carry only a binary
is_contradiction column, not the original SNLI 3-way label. A spot check
tried to recover it by joining premise+hypothesis text back to the public
snli_test.parquet / snli_validation.parquet: only 11/20 and 9/20 of a 20-item
sample matched respectively, because the pools were drawn from SNLI splits
that were not all saved locally (some rows are from the train split, which
this repository does not carry). That join is unreliable here, so the
entailment-vs-neutral split of the public-side negatives is NOT computed;
this is stated rather than silently skipped.

Reads (public, non-clinical, safe to print):
    results/accord/prevalence_replication/sweep_results.json
    accord/prevalence_replication/results/raw_predictions/*.csv (existence check only)
Reads (register, read-only):
    results/accord/VERIFIED_FACTS.json
Writes:
    results/accord/stats/overcall_decomposition.json

Run:    python3 accord/stats/overcall_decomposition.py
"""

import collections
import glob
import json
import pathlib

from _overcall_common import ROOT, load_facts, reg, wilson_ci

SWEEP = ROOT / "results" / "accord" / "prevalence_replication" / "sweep_results.json"
RAW_DIR = ROOT / "accord" / "prevalence_replication" / "results" / "raw_predictions"
OUT = ROOT / "results" / "accord" / "stats" / "overcall_decomposition.json"

TOL = 1e-6


def identity_rhs(tpr, fpr, p):
    return tpr + fpr * (1.0 - p) / p


def public_side():
    sweep = json.loads(SWEEP.read_text())
    cells = collections.OrderedDict()
    for model, by_prompt in sweep["results"].items():
        for prompt, by_prev in by_prompt.items():
            for prev_str, cell in by_prev.items():
                conf = cell["confusion_decided_only"]
                tp, fp, fn, tn = conf["TP"], conf["FP"], conf["FN"], conf["TN"]
                n = tp + fp + fn + tn
                n_pos, n_neg = tp + fn, fp + tn
                tpr = (tp / n_pos) if n_pos > 0 else None
                fpr = (fp / n_neg) if n_neg > 0 else None
                tpr_lo, tpr_hi = wilson_ci(tp, n_pos) if n_pos > 0 else (None, None)
                fpr_lo, fpr_hi = wilson_ci(fp, n_neg) if n_neg > 0 else (None, None)

                p_nom = cell["true_prevalence"]
                over_call_src = cell["over_call_ratio"]
                pos_rate_decided = cell["positive_rate_of_decided"]
                over_call_recomp_nominal = (
                    (pos_rate_decided / p_nom) if p_nom > 0 else None
                )

                p_real = (n_pos / n) if n > 0 else None
                over_call_recomp_real = (
                    (pos_rate_decided / p_real) if p_real else None
                )

                def check(p, target):
                    if tpr is None or fpr is None or not p or target is None:
                        return (None, None, None)
                    rhs = identity_rhs(tpr, fpr, p)
                    diff = abs(rhs - target)
                    return (rhs, diff, diff < TOL)

                # Nominal-p identity is checked against the ratio the source
                # artifact actually reports (over_call_ratio_source), because
                # that ratio divides by the nominal target prevalence. The
                # realised-p identity is checked against the ratio recomputed
                # with the realised decided-subset prevalence
                # (over_call_recomp_real): dividing by that same p is what
                # makes it a tautology, not a data-dependent check.
                rhs_nom, diff_nom, holds_nom = check(p_nom, over_call_src)
                rhs_real, diff_real, holds_real = check(p_real, over_call_recomp_real)

                key = f"{model}::{prompt}::p{prev_str}"
                cells[key] = collections.OrderedDict([
                    ("model", model), ("prompt", prompt),
                    ("prevalence_nominal", p_nom),
                    ("n_decided", n), ("n_abstain", cell["n_abstain"]),
                    ("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn),
                    ("n_pos_decided", n_pos), ("n_neg_decided", n_neg),
                    ("tpr", tpr), ("tpr_wilson_lo", tpr_lo), ("tpr_wilson_hi", tpr_hi),
                    ("fpr", fpr), ("fpr_wilson_lo", fpr_lo), ("fpr_wilson_hi", fpr_hi),
                    ("positive_rate_of_decided", pos_rate_decided),
                    ("over_call_ratio_source", over_call_src),
                    ("over_call_ratio_recomputed_nominal_p", over_call_recomp_nominal),
                    ("over_call_ratio_recomputed_realised_p", over_call_recomp_real),
                    ("prevalence_realised_decided", p_real),
                    ("identity_rhs_nominal_p", rhs_nom),
                    ("identity_diff_nominal_p", diff_nom),
                    ("identity_holds_nominal_p", holds_nom),
                    ("identity_rhs_realised_p", rhs_real),
                    ("identity_diff_realised_p", diff_real),
                    ("identity_holds_realised_p", holds_real),
                    ("ratio_gt1", over_call_src > 1.0 + TOL),
                    ("ratio_lt1", over_call_src < 1.0 - TOL),
                    ("ratio_eq1", abs(over_call_src - 1.0) <= TOL),
                ])

    n_gt1 = sum(1 for c in cells.values() if c["ratio_gt1"])
    n_lt1 = sum(1 for c in cells.values() if c["ratio_lt1"])
    n_eq1 = sum(1 for c in cells.values() if c["ratio_eq1"])
    finite_ratios = {k: c["over_call_ratio_source"] for k, c in cells.items()}
    max_key = max(finite_ratios, key=lambda k: finite_ratios[k])
    identity_fail_nominal = [k for k, c in cells.items()
                             if c["identity_holds_nominal_p"] is False]
    identity_fail_real = [k for k, c in cells.items()
                          if c["identity_holds_realised_p"] is False]

    raw_files = sorted(p.name for p in RAW_DIR.glob("*.csv")) if RAW_DIR.exists() else []
    three_way_available = False
    for f in raw_files:
        # Header-only check: no row content read, just column names.
        import csv
        with open(RAW_DIR / f) as fh:
            header = next(csv.reader(fh))
        if any("label" in h.lower() and "is_contradiction" not in h.lower()
               for h in header) or "gold_label" in header or "snli_label" in header:
            three_way_available = True
            break

    return collections.OrderedDict([
        ("source_artifact", "results/accord/prevalence_replication/sweep_results.json"),
        ("per_item_files_checked", raw_files),
        ("three_way_snli_label_available_in_per_item_files", three_way_available),
        ("three_way_snli_label_note",
         "Per-item CSVs and pool files carry only a binary is_contradiction "
         "column. A join back to snli_test.parquet/snli_validation.parquet "
         "by premise+hypothesis text recovered only 11/20 and 9/20 of a "
         "20-item spot check respectively (pools include rows from splits "
         "not saved locally), so the entailment-vs-neutral split of the "
         "public-side negatives is not computed and is skipped, as directed."),
        ("n_cells", len(cells)),
        ("n_ratio_gt1", n_gt1), ("n_ratio_lt1", n_lt1), ("n_ratio_eq1", n_eq1),
        ("max_over_call_ratio", finite_ratios[max_key]),
        ("max_over_call_ratio_cell", max_key),
        ("identity_tolerance", TOL),
        ("cells_failing_identity_nominal_p", identity_fail_nominal),
        ("n_cells_failing_identity_nominal_p", len(identity_fail_nominal)),
        ("cells_failing_identity_realised_p", identity_fail_real),
        ("n_cells_failing_identity_realised_p", len(identity_fail_real)),
        ("cells", cells),
    ])


# Frontier-template rows: full confusion counts registered under
# frontier_ollama_<label>.<strategy>.<basis>.{tp,fp,fn,tn,n,n_positives,
# over_calling_ratio}. Only granite3.3:8b and llama3:8b were run under this
# template with per-pair confusion counts; qwen3:8b was not.
FRONTIER_ROWS = [
    ("granite3.3:8b", "frontier_ollama_granite33", "zero_shot", "G_253"),
    ("granite3.3:8b", "frontier_ollama_granite33", "zero_shot", "G_299"),
    ("granite3.3:8b", "frontier_ollama_granite33", "few_shot", "G_253"),
    ("granite3.3:8b", "frontier_ollama_granite33", "few_shot", "G_299"),
    ("llama3:8b", "frontier_ollama_llama3", "zero_shot", "G_253"),
    ("llama3:8b", "frontier_ollama_llama3", "zero_shot", "G_299"),
    ("llama3:8b", "frontier_ollama_llama3", "few_shot", "G_253"),
    ("llama3:8b", "frontier_ollama_llama3", "few_shot", "G_299"),
]

# Baseline-prompt rows: only a positive rate and a ratio are registered, on
# the G_299 ("gold") basis; no confusion breakdown. All three local 8B models
# ran this row (qwen3:8b included, unlike the frontier template).
BASELINE_RATE_ONLY_ROWS = [
    ("llama3:8b", "overcall.llama3_8b"),
    ("granite3.3:8b", "overcall.granite3_3_8b"),
    ("qwen3:8b", "overcall.qwen3_8b"),
]


def clinical_side(facts):
    frontier = collections.OrderedDict()
    for model, prefix, strategy, basis in FRONTIER_ROWS:
        base = f"{prefix}.{strategy}.{basis}"
        tp = reg(facts, f"{base}.tp"); fp = reg(facts, f"{base}.fp")
        fn = reg(facts, f"{base}.fn"); tn = reg(facts, f"{base}.tn")
        n = reg(facts, f"{base}.n"); n_pos = reg(facts, f"{base}.n_positives")
        n_neg = n - n_pos
        tpr = tp / n_pos; fpr = fp / n_neg
        tpr_lo, tpr_hi = wilson_ci(tp, n_pos)
        fpr_lo, fpr_hi = wilson_ci(fp, n_neg)
        p = n_pos / n
        over_call_registered = reg(facts, f"{base}.over_calling_ratio")
        rhs = identity_rhs(tpr, fpr, p)
        diff = abs(rhs - over_call_registered)
        key = f"{model}::{strategy}::{basis}"
        frontier[key] = collections.OrderedDict([
            ("model", model), ("strategy", strategy), ("basis", basis),
            ("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn),
            ("n", n), ("n_positives", n_pos), ("n_negatives", n_neg),
            ("prevalence", p),
            ("tpr", tpr), ("tpr_wilson_lo", tpr_lo), ("tpr_wilson_hi", tpr_hi),
            ("fpr", fpr), ("fpr_wilson_lo", fpr_lo), ("fpr_wilson_hi", fpr_hi),
            ("over_calling_ratio_registered", over_call_registered),
            ("identity_rhs", rhs),
            ("identity_diff", diff),
            ("identity_holds", diff < TOL),
            ("register_source", f"results/accord/VERIFIED_FACTS.json::{base}.*"),
        ])

    baseline = collections.OrderedDict()
    for model, prefix in BASELINE_RATE_ONLY_ROWS:
        rate = reg(facts, f"{prefix}.gold.positive_rate")
        ratio = reg(facts, f"{prefix}.gold.ratio")
        key = f"{model}::baseline_prompt::gold"
        baseline[key] = collections.OrderedDict([
            ("model", model), ("basis", "gold (G_299)"),
            ("positive_rate", rate),
            ("over_calling_ratio_registered", ratio),
            ("tpr_recoverable", False),
            ("fpr_recoverable", False),
            ("note", "Only an overall positive (flag) rate is registered for "
                     "this row; there is no per-pair confusion breakdown, so "
                     "TPR and FPR cannot be recovered from it. Reporting a "
                     "value for either here would be invented, not measured."),
            ("register_source", f"results/accord/VERIFIED_FACTS.json::{prefix}.gold.*"),
        ])

    all_ratios = ([r["over_calling_ratio_registered"] for r in frontier.values()]
                  + [r["over_calling_ratio_registered"] for r in baseline.values()])
    return collections.OrderedDict([
        ("frontier_template", frontier),
        ("baseline_prompt_rate_only", baseline),
        ("n_frontier_rows", len(frontier)),
        ("n_baseline_rate_only_rows", len(baseline)),
        ("n_rows_with_recoverable_fpr", len(frontier)),
        ("n_rows_with_fpr_not_recoverable", len(baseline)),
        ("min_over_calling_ratio_all_rows", min(all_ratios)),
        ("max_over_calling_ratio_all_rows", max(all_ratios)),
        ("all_ratios_gt_1", all(r > 1.0 for r in all_ratios)),
    ])


def main():
    facts = load_facts()
    public = public_side()
    clinical = clinical_side(facts)

    out = collections.OrderedDict([
        ("provenance", {
            "generated_by": "accord/stats/overcall_decomposition.py",
            "method": "Public side from the SNLI prevalence sweep's own "
                      "confusion_decided_only counts; clinical side from "
                      "registered confusion counts (frontier template) or "
                      "registered rates only (baseline prompt). No number "
                      "here is computed from anything other than a "
                      "committed artifact or the register.",
        }),
        ("public", public),
        ("clinical", clinical),
    ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"public: {public['n_cells']} cells, "
          f"gt1={public['n_ratio_gt1']} lt1={public['n_ratio_lt1']} eq1={public['n_ratio_eq1']}, "
          f"max_ratio={public['max_over_call_ratio']:.4f} at {public['max_over_call_ratio_cell']}")
    print(f"identity failures (nominal p): {public['n_cells_failing_identity_nominal_p']} "
          f"of {public['n_cells']}; (realised p): {public['n_cells_failing_identity_realised_p']}")
    print(f"clinical: {clinical['n_frontier_rows']} frontier rows with recoverable FPR, "
          f"{clinical['n_baseline_rate_only_rows']} baseline rows with FPR not recoverable")
    print(f"clinical ratio range: [{clinical['min_over_calling_ratio_all_rows']:.4f}, "
          f"{clinical['max_over_calling_ratio_all_rows']:.4f}]")


if __name__ == "__main__":
    main()
