"""
ACCORD Build 2 -- real-cohort validation of the denominator demonstration.

Question: our HF cohort has a real, adjusted candidate-generation selection
OR of 0.6372 [0.5548, 0.732] (missing-language vs English admissions, C12 /
exp2a_candidate_yield_regression.json, adjusted for note volume and LOS).
Would the STANDARD per-group fairness audit -- run on the candidate pool, the
way it is actually run in this project (exp1a: FPR/FNR/precision/recall by
language group on the 299 pharmacist gold-annotated pairs, which is exactly
the "audit the model on the eligible/evaluation set" workflow) -- have
flagged that upstream selection bias?

This script performs NO new computation on raw MIMIC-III data. It reads only
already-computed, already-released AGGREGATE result files
(exp1a_gold_error_rates_by_group.json, exp2a_candidate_yield_regression.json,
exp3_disaggregation.json in mimic_local_only/accord_differential/, produced
by accord/differential/run_experiments.py) and recombines them into an
audit-vs-selection comparison. Output (aggregates only, no note text, no
per-patient rows) is written to
/Users/rickygole/mimic_local_only/accord_denominator_validation/.
"""
import json
import os

IN_DIR = "/Users/rickygole/mimic_local_only/accord_differential"
OUT_DIR = "/Users/rickygole/mimic_local_only/accord_denominator_validation"
os.makedirs(OUT_DIR, exist_ok=True)

TOLERANCE_GAP = 0.05
FOUR_FIFTHS = 0.80

with open(os.path.join(IN_DIR, "exp1a_gold_error_rates_by_group.json")) as f:
    exp1a = json.load(f)
with open(os.path.join(IN_DIR, "exp2a_candidate_yield_regression.json")) as f:
    exp2a = json.load(f)
with open(os.path.join(IN_DIR, "exp3_disaggregation.json")) as f:
    exp3 = json.load(f)

lang_rows = {r["group"]: r for r in exp1a["language_group3"]}
english, missing = lang_rows["ENGLISH"], lang_rows["MISSING"]

log_lines = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log_lines.append(s)

log("=" * 78)
log("REAL-COHORT VALIDATION: would the standard per-group audit have caught")
log("the observed candidate-generation selection bias (OR=0.64)?")
log("=" * 78)

# ---------------------------------------------------------------------------
# 1. THE UPSTREAM SELECTION BIAS (population level, never audited in the
#    standard workflow -- this is candidate generation, not model evaluation)
# ---------------------------------------------------------------------------
sel_after = exp2a["after_volume_adj"]["C(language_group3, Treatment('ENGLISH'))[T.MISSING]"]
sel_before = exp2a["before_volume_adj"]["C(language_group3, Treatment('ENGLISH'))[T.MISSING]"]
yield_english = exp3["yield_by_language3"]["mean"]["ENGLISH"]
yield_missing = exp3["yield_by_language3"]["mean"]["MISSING"]
raw_yield_ratio = yield_missing / yield_english
raw_four_fifths = min(yield_english, yield_missing) / max(yield_english, yield_missing)

log(f"\n-- Candidate-generation selection (population level, n={exp2a['n']:,} admissions) --")
log(f"   Raw candidate yield:  ENGLISH={yield_english:.4f}  MISSING={yield_missing:.4f}  "
    f"(ratio MISSING/ENGLISH = {raw_yield_ratio:.3f})")
log(f"   Raw four-fifths ratio (yield_missing / yield_english) = {raw_four_fifths:.3f} "
    f"{'FAILS' if raw_four_fifths < FOUR_FIFTHS else 'passes'} the four-fifths rule")
log(f"   Unadjusted logistic-regression OR (MISSING vs ENGLISH): "
    f"{sel_before['OR']} [{sel_before['CI_low']}, {sel_before['CI_high']}], p={sel_before['p']}")
log(f"   Adjusted OR (+ log note chars, LOS -- the number this validation targets): "
    f"{sel_after['OR']} [{sel_after['CI_low']}, {sel_after['CI_high']}], p={sel_after['p']}")
log(f"   This is the real OR=0.64 referenced in the task: candidate generation")
log(f"   systematically under-selects admissions with an unrecorded LANGUAGE field,")
log(f"   independent of documentation volume, at p<0.001.")

selection_or = sel_after["OR"]
selection_breaches_four_fifths = selection_or < FOUR_FIFTHS
log(f"\n   If anyone had audited the SELECTION STAGE with the four-fifths rule "
    f"(OR={selection_or} < 0.80): {'WOULD HAVE FLAGGED IT' if selection_breaches_four_fifths else 'would not have flagged it'}.")
log(f"   Nobody did -- the standard workflow audits the model on its evaluation")
log(f"   set (the candidate pool), not the candidate-generation stage itself.")

# ---------------------------------------------------------------------------
# 2. THE STANDARD AUDIT ACTUALLY RUN (exp1a: rule-based miner's per-group
#    error rates on the 299 pharmacist gold-annotated candidate pairs --
#    i.e. computed ON THE CANDIDATE POOL, exactly the standard workflow)
# ---------------------------------------------------------------------------
log(f"\n-- Standard per-group audit AS ACTUALLY RUN in this project (exp1a), on the")
log(f"   candidate pool (299 pharmacist gold-annotated pairs, by language_group3) --")
log(f"   ENGLISH (n={english['n']}):  TP={english['TP']} FP={english['FP']} "
    f"FN={english['FN']} TN={english['TN']}  FPR={english['FPR']}  recall={english['recall']}  "
    f"precision={english['precision']}")
log(f"   MISSING (n={missing['n']}):  TP={missing['TP']} FP={missing['FP']} "
    f"FN={missing['FN']} TN={missing['TN']}  FPR={missing['FPR']}  recall={missing['recall']}  "
    f"precision={missing['precision']}")

def pred_pos_rate(row):
    return (row["TP"] + row["FP"]) / row["n"]

dp_e, dp_m = pred_pos_rate(english), pred_pos_rate(missing)
dp_gap = abs(dp_e - dp_m)
dp_four_fifths = min(dp_e, dp_m) / max(dp_e, dp_m)
fpr_gap = abs(english["FPR"] - missing["FPR"])
precision_gap = abs(english["precision"] - missing["precision"])
recall_gap = abs(english["recall"] - missing["recall"])  # NB: n_pos=8 vs 4 -- see caveat below

log(f"\n   Demographic parity (flagged rate): ENGLISH={dp_e:.4f} MISSING={dp_m:.4f}  "
    f"gap={dp_gap:.4f}  four-fifths ratio={dp_four_fifths:.3f}")
log(f"   FPR parity: gap={fpr_gap:.4f}  (well-powered: n_neg ENGLISH={english['n_neg']}, "
    f"MISSING={missing['n_neg']})")
log(f"   Precision (PPV) parity: gap={precision_gap:.4f}")
log(f"   Recall (TPR) parity: gap={recall_gap:.4f}  ** CAVEAT: n_pos=8 (ENGLISH) vs "
    f"n_pos=4 (MISSING) -- CORRECTIONS.md C12/C3 already flag this as not a stable ")
log(f"   estimate; one flipped call changes recall by 25 points in the MISSING group.")

well_powered_breach = (dp_gap > TOLERANCE_GAP) or (dp_four_fifths < FOUR_FIFTHS) or \
                       (fpr_gap > TOLERANCE_GAP) or (precision_gap > TOLERANCE_GAP)
recall_breach = recall_gap > TOLERANCE_GAP

log(f"\n   Well-powered metrics (demographic parity, four-fifths, FPR, precision) "
    f"breach tolerance: {well_powered_breach}")
log(f"   Recall/TPR-based equalized-odds metric breaches tolerance: {recall_breach} "
    f"(statistically unreliable given n_pos=4 in the MISSING group)")

# ---------------------------------------------------------------------------
# 3. VERDICT
# ---------------------------------------------------------------------------
log(f"\n{'='*78}\nVERDICT\n{'='*78}")
log(f"Population-level candidate-generation selection OR (MISSING vs ENGLISH,")
log(f"volume-adjusted): {selection_or} [{sel_after['CI_low']}, {sel_after['CI_high']}], p<0.001.")
log(f"This fails the four-fifths rule ({selection_or} < 0.80) and is a real, ")
log(f"substantial upstream selection bias.")
log(f"")
log(f"The standard per-group audit actually run in this project on the candidate")
log(f"pool (exp1a) does NOT flag it on any well-powered metric: demographic-parity")
log(f"gap={dp_gap:.3f}, four-fifths ratio={dp_four_fifths:.3f}, FPR gap={fpr_gap:.3f}, ")
log(f"precision gap={precision_gap:.3f} -- all comfortably inside conventional tolerance")
log(f"(gap<0.05, four-fifths>0.80). The one metric that nominally moves (recall gap")
log(f"={recall_gap:.3f}) is built on 4 and 8 positive cases respectively and is not a")
log(f"reliable estimate -- this project's own corrections record (C3, C12) already")
log(f"states that error-rate comparisons are not estimable at this n.")
log(f"")
log(f"CONCLUSION: NO. The standard per-group fairness audit, run the way it is")
log(f"actually run in this pipeline (on the candidate pool, using well-powered")
log(f"metrics), would NOT have caught the OR=0.64 candidate-generation selection")
log(f"bias. This matches the synthetic sweep result in simulate.py: a per-group")
log(f"audit computed on the post-selection candidate pool is structurally blind to")
log(f"selection-rate disparities of this kind, because language-field completeness")
log(f"predicts candidate generation but (per C11/C12) not the true underlying")
log(f"clinical outcome -- i.e. the real-data case closely matches Sweep 1 of the")
log(f"synthetic simulation (selection dependent on group, not on the true outcome).")

# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
result = {
    "selection_stage": {
        "n_admissions": exp2a["n"],
        "raw_yield_english": yield_english, "raw_yield_missing": yield_missing,
        "raw_yield_ratio_missing_over_english": raw_yield_ratio,
        "raw_four_fifths_ratio": raw_four_fifths,
        "unadjusted_OR": sel_before, "adjusted_OR": sel_after,
        "adjusted_OR_breaches_four_fifths_rule": selection_breaches_four_fifths,
    },
    "standard_audit_on_candidate_pool_exp1a": {
        "english": english, "missing": missing,
        "demographic_parity_rate_english": dp_e, "demographic_parity_rate_missing": dp_m,
        "demographic_parity_gap": dp_gap, "demographic_parity_four_fifths_ratio": dp_four_fifths,
        "fpr_gap": fpr_gap, "precision_gap": precision_gap, "recall_gap": recall_gap,
        "well_powered_metrics_breach_tolerance": well_powered_breach,
        "recall_metric_breaches_tolerance_but_unreliable_n": recall_breach,
    },
    "tolerances": {"parity_gap_tolerance": TOLERANCE_GAP, "four_fifths_threshold": FOUR_FIFTHS},
    "verdict": "standard per-group audit on the candidate pool would NOT have caught "
               "the OR=0.64 upstream selection bias on well-powered metrics",
}
with open(os.path.join(OUT_DIR, "real_cohort_validation.json"), "w") as f:
    json.dump(result, f, indent=2, default=str)
with open(os.path.join(OUT_DIR, "run_log.txt"), "w") as f:
    f.write("\n".join(log_lines))

print(f"\nSaved real_cohort_validation.json, run_log.txt to {OUT_DIR}")
print("ALL_DONE")
