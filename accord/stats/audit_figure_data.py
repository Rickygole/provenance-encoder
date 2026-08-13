#!/usr/bin/env python3
"""
Independent provenance audit of the thirteen thesis figures.

For every number that reaches the canvas in
  accord/thesis/figures/make_figures.py
  accord/thesis/figures/fig_batch2.py
  accord/thesis/figures/fig_threshold_surface.py
  accord/thesis/figures/make_figures_extra.py
  accord/thesis/figures/Thesis_Figures_PerFigure.ipynb   (the per-figure notebook)
this script re-opens the primary result file, re-reads the key, and re-derives
the value from scratch. It does not trust the figure code, the manifests, or
results/accord/VERIFIED_FACTS.json.

Classification:
  LOADED   read from a result file
  DERIVED  arithmetic on LOADED values only
  SETTING  axis limit / tolerance / figure size / tick position
  TYPED    a data value written as a literal in the drawing code

Writes results/accord/figure_data_audit.json. Changes nothing else.
"""

import json
import math
import os
import statistics as st

import numpy as np
from scipy.stats import norm

REPO = "/Users/rickygole/provenance-encoder"
RES = os.path.join(REPO, "results")
RA = os.path.join(RES, "accord")
OUT = os.path.join(RA, "figure_data_audit.json")


def J(*parts):
    p = os.path.join(*parts)
    with open(p) as f:
        return json.load(f), os.path.relpath(p, REPO)


def close(a, b, tol=5e-4):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


FIGS = []
V = []  # flat list of every value record


def rec(fig, name, cls, value, verdict, file=None, key=None, expr=None, note=""):
    r = {
        "figure": fig,
        "value_name": name,
        "classification": cls,
        "plotted_value": value,
        "source_file": file,
        "key_path": key,
        "expression": expr,
        "verdict": verdict,
        "note": note,
    }
    V.append(r)
    return r


# ===========================================================================
# Load every primary source once, independently.
# ===========================================================================
ex_all, P_EX = J(RA, "exemplar", "exemplar_correctness_recomputed.json")
s01, P_S01 = J(RA, "diag", "step0_1_join_and_calibration.json")
s2, P_S2 = J(RA, "diag", "step2_signal_check.json")
s3, P_S3 = J(RA, "diag", "step3_baselines.json")
dec, P_DEC = J(RA, "decontam", "corrected_gold_metrics.json")
decagg, P_DECAGG = J(RA, "decontam", "accord_v2_decontam_gold_aggregate.json")
iv, P_IV = J(RA, "stats", "intervals.json")
strata, P_STR = J(RA, "defect_free_strata.json")
exp1a, P_E1A = J(RA, "differential", "exp1a_gold_error_rates_by_group.json")
c28, P_C28 = J(RA, "reconciliation_c28.json")
sweep, P_SW = J(RA, "prevalence_replication", "sweep_results.json")
pw, P_PW = J(RA, "section6_auroc_power.json")
tr, P_TR = J(RES, "task_runs.json")
sc, P_SC = J(RA, "scaling", "scaling_results.json")
thr, P_THR = J(RA, "selection", "threshold_robustness_results.json")
trans, P_TRANS = J(RA, "generality", "transfer_results_correct.json")
npa, P_NPA = J(RA, "number_provenance_audit.json")


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - a) / den), min(1.0, (c + a) / den))


def mcc_counts(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else num / den


def hanley_mde(m, n, alpha=0.05, power=0.80):
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    lo, hi = 0.5, 0.9999
    for _ in range(200):
        a = (lo + hi) / 2
        q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
        v = (a * (1 - a) + (m - 1) * (q1 - a * a) + (n - 1) * (q2 - a * a)) / (m * n)
        if (a - 0.5) / math.sqrt(max(v, 1e-12)) < z:
            lo = a
        else:
            hi = a
    return (lo + hi) / 2


# ===========================================================================
# 1. fig-pipeline-status
# ===========================================================================
F = "fig-pipeline-status"
p = ex_all["rule_1_2_pooled"]
n_wrong, n_res = p["n_wrong"], p["n_resolved"]
defect = n_wrong / n_res
rec(F, "exemplar n_wrong", "LOADED", n_wrong, "PASS", P_EX, "rule_1_2_pooled.n_wrong")
rec(F, "exemplar n_resolved", "LOADED", n_res, "PASS", P_EX, "rule_1_2_pooled.n_resolved")
rec(F, "badge: defect rate %", "DERIVED", round(defect * 100, 4), "PASS", P_EX,
    None, "n_wrong/n_resolved = 65/137", "badge prints 47.4%; uses the defensible ratio, NOT the struck 0.586/0.414")

xt = s01["weak_confidence_is_degenerate"]["crosstab_label_x_confidence"]
wp, wn = xt["0.95"]["contradiction"], xt["1.0"]["not_contradiction"]
off_diag = xt["0.95"]["not_contradiction"] + xt["1.0"]["contradiction"]
wrate = wp / (wp + wn)
rec(F, "weak_pos", "LOADED", wp, "PASS", P_S01,
    "weak_confidence_is_degenerate.crosstab_label_x_confidence.0.95.contradiction")
rec(F, "weak_neg", "LOADED", wn, "PASS", P_S01,
    "weak_confidence_is_degenerate.crosstab_label_x_confidence.1.0.not_contradiction")
rec(F, "weak_pos_rate", "DERIVED", round(wrate, 6),
    "PASS" if off_diag == 0 else "FAIL", P_S01, None, "1425/(1425+254)",
    f"off-diagonal crosstab cells sum to {off_diag}; denominator 1679 is the full pool only because they are zero")
pharm = s2["contra_prob_vs_pharm_label__ambiguous_as_negative_n299"]["base_rate"]
ratio = wrate / pharm
rec(F, "pharm base rate n=299", "LOADED", pharm, "PASS", P_S2,
    "contra_prob_vs_pharm_label__ambiguous_as_negative_n299.base_rate")
rec(F, "badge: prevalence mismatch ratio", "DERIVED", round(ratio, 4), "PASS", P_S2,
    None, "0.848720/0.0401 = 21.16", "badge prints 21x")
auroc = dec["aggregate"]["ensemble_auroc"]
ci = dec["aggregate"]["ensemble_ci95"]
rec(F, "badge: AUROC", "LOADED", auroc, "PASS", P_DEC, "aggregate.ensemble_auroc")
rec(F, "badge: AUROC CI95", "LOADED", ci, "PASS", P_DEC, "aggregate.ensemble_ci95")
for nm, val in [("xlim", [0, 6]), ("ylim", [0.55, 2]), ("figsize", [6.5, 2.55]),
                ("box_w", 0.82), ("box x positions", "linspace(0.5,5.5,6)")]:
    rec(F, nm, "SETTING", val, "PASS")
rec(F, "6 stage names + 3 categorical badges (WORKS / UNVALIDATED / THREE CLAIMS FALSIFIED)",
    "SETTING", "categorical text", "PASS", None, None, None,
    "narrative labels, no numeric claim on canvas")

# ===========================================================================
# 2. fig-intro-headline
# ===========================================================================
F = "fig-intro-headline"
lang = exp1a["language_group3"]
pred_pos = sum(g["TP"] + g["FP"] for g in lang)
pred_n = sum(g["n"] for g in lang)
c28_278 = c28["a_what_each_number_counts"]["278"]
c28_261 = c28["a_what_each_number_counts"]["261"]
rec(F, "bar1 weak labeller train pool", "DERIVED", f"{wp}/{wp+wn}", "PASS", P_S01,
    "crosstab 0.95.contradiction / (…+1.0.not_contradiction)", "1425/1679 = 0.8487")
rec(F, "bar2 numerator (labelled 'ACCORD pred.-positive rate (gold set)')", "LOADED",
    pred_pos, "FAIL", P_E1A, "language_group3[*].TP + language_group3[*].FP",
    "sum(TP+FP) over 3 language groups = 261",
    "MISLABELLED QUANTITY. reconciliation_c28.json states exp1a 'is titled Llama weak-label error "
    "rates by group and never touches accord_pred'. 261 is the Llama 3 8B WEAK LABELLER's "
    "predicted-positive count. ACCORD's own is 278/299 (gold_standard_accord_results.csv, "
    "accord_pred=='contradiction'). Correct value for the axis label as written: 278/299 = 93.0%.")
rec(F, "bar2 denominator", "LOADED", pred_n, "PASS", P_E1A, "sum(language_group3[*].n) = 299")
rec(F, "c28 cross-check: ACCORD predicted-positive", "LOADED", c28_278["value"], "PASS",
    P_C28, "a_what_each_number_counts.278.value",
    None, "the value the axis label describes; NOT plotted")
rec(F, "c28 cross-check: weak-labeller marginal", "LOADED", c28_261["value"], "PASS",
    P_C28, "a_what_each_number_counts.261.value", None, "equals the bar that is plotted")
d299 = s2["contra_prob_vs_pharm_label__ambiguous_as_negative_n299"]
rec(F, "bar3 pharmacist n_pos", "LOADED", d299["n_pos"], "PASS", P_S2,
    "contra_prob_vs_pharm_label__ambiguous_as_negative_n299.n_pos")
rec(F, "bar3 pharmacist n", "LOADED", d299["n"], "PASS", P_S2,
    "contra_prob_vs_pharm_label__ambiguous_as_negative_n299.n")
for k, n_, lab in [(wp, wp + wn, "bar1"), (pred_pos, pred_n, "bar2"), (d299["n_pos"], d299["n"], "bar3")]:
    lo, hi = wilson(k, n_)
    rec(F, f"{lab} Wilson 95% CI", "DERIVED", [round(lo, 5), round(hi, 5)], "PASS", None,
        None, f"wilson_ci({k},{n_})", "closed form, deterministic")
rec(F, "annotation '~21x'", "DERIVED", round(wrate / (d299["n_pos"] / d299["n"]), 3), "PASS",
    None, None, "0.848720/0.040134 = 21.15")
mcc = iv["E2_mcc_intervals"]
for k in ["gpt4o_zeroshot", "gpt4o_fewshot", "accord_binary_at_0.5_253core", "llama_baseline"]:
    ok = close(mcc[k]["mcc"], s3[k]["mcc"], 1e-9)
    rec(F, f"MCC {k}", "LOADED", mcc[k]["mcc"], "PASS" if ok else "FAIL", P_IV,
        f"E2_mcc_intervals.{k}.mcc", None,
        f"cross-checks against {P_S3}:{k}.mcc = {s3[k]['mcc']}")
    tpx, tnx, fpx, fnx = s3[k]["tp"], s3[k]["tn"], s3[k]["fp"], s3[k]["fn"]
    re_mcc = mcc_counts(tpx, tnx, fpx, fnx)
    rec(F, f"MCC {k} (recomputed from confusion matrix)", "DERIVED", round(re_mcc, 6),
        "PASS" if close(re_mcc, mcc[k]["mcc"], 5e-5) else "FAIL", P_S3,
        f"{k}.tp/tn/fp/fn", "mcc_from_counts", "independent re-derivation")
    rec(F, f"MCC {k} CI95", "LOADED", mcc[k]["ci95"], "PASS", P_IV,
        f"E2_mcc_intervals.{k}.ci95", None, "bootstrap B=10000, seed=42, read not recomputed")
for nm, val in [("axL xlim", [0, 1.75]), ("axL ylim", [-0.6, 2.6]), ("bar height", 0.55),
                ("figsize", [7.0, 4.3]), ("annotation x=1.48/1.56", [1.48, 1.56])]:
    rec(F, nm, "SETTING", val, "PASS")

# ===========================================================================
# 3. fig-rule-precision
# ===========================================================================
F = "fig-rule-precision"
per_rule = sorted([s for s in strata["strata"] if s["name"].startswith("rule ")],
                  key=lambda s: s["name"])
for s in per_rule:
    rec(F, f"{s['name']} AUROC (bar height)", "LOADED", s["auroc"], "PASS", P_STR,
        f"strata[name=={s['name']}].auroc")
    rec(F, f"{s['name']} n", "LOADED", s["n"], "PASS", P_STR, f"strata[name=={s['name']}].n")
    rec(F, f"{s['name']} pos", "LOADED", s["pos"], "PASS", P_STR, f"strata[name=={s['name']}].pos")
n_sum = sum(s["n"] for s in per_rule)
pos_sum = sum(s["pos"] for s in per_rule)
rec(F, "per-rule n sums to the gold set", "DERIVED", n_sum,
    "PASS" if n_sum == 299 else "FAIL", P_STR, None, "134+31+79+55")
rec(F, "per-rule pos sums to gold positives", "DERIVED", pos_sum,
    "PASS" if pos_sum == 12 else "FAIL", P_STR, None, "3+3+4+2")
e3 = iv["E3_defect_free_stratum"]
rec(F, "pooled rules 3+4 CI band", "LOADED", e3["unstratified_ci95"], "PASS", P_IV,
    "E3_defect_free_stratum.unstratified_ci95")
rec(F, "pooled rules 3+4 n", "LOADED", e3["n"], "PASS", P_IV, "E3_defect_free_stratum.n")
rec(F, "pooled rules 3+4 n_pos", "LOADED", e3["n_pos"], "PASS", P_IV, "E3_defect_free_stratum.n_pos")
r34 = next(s for s in strata["strata"] if s["name"] == "rules 3+4 (DEFECT-FREE)")
rec(F, "E3 n/pos cross-check vs defect_free_strata", "DERIVED", [r34["n"], r34["pos"]],
    "PASS" if (r34["n"] == e3["n"] and r34["pos"] == e3["n_pos"]) else "FAIL", P_STR,
    "strata[rules 3+4 (DEFECT-FREE)].n/pos")
rec(F, "which ACCORD run the per-rule AUROCs come from", "LOADED", "contaminated",
    "PASS(caveat)", P_STR, "strata[ALL].auroc = 0.4602", None,
    "All four bars and the reference band come from the SAME run, so the figure is internally "
    "consistent. But strata[ALL].auroc = 0.4602 identifies it as the contaminated released score "
    "(control/final_report.json accord_contra_prob_same_299_pairs.gold_all.auroc = 0.46022), the "
    "run CORRECTIONS.md C16 retired. The y-axis says only 'ACCORD AUROC (per rule)'. No "
    "decontaminated per-rule breakdown exists in results/, so there is nothing to substitute; the "
    "run should be named on the axis or in the caption.")
rec(F, "chance line 0.5", "SETTING", 0.5, "PASS")
rec(F, "ylim", "SETTING", [0, 1.0], "PASS")
rec(F, "band xmin=(2-0.4)/4", "SETTING", 0.4, "PASS", None, None, None,
    "axes-fraction start of the band so it spans only the rule-3/rule-4 bars")

# ===========================================================================
# 4. fig-prevalence-response
# ===========================================================================
F = "fig-prevalence-response"
n_pts = 0
for m in sweep["models"]:
    for pr in ["A_bare", "B_baserate_abstain"]:
        for lev in ["0.02", "0.05", "0.1", "0.25"]:
            e = sweep["results"][m][pr][lev]
            recomputed = e["n_contradiction"] / e["n_decided"]
            ok = close(recomputed, e["positive_rate_of_decided"], 1e-12)
            rec(F, f"{m}/{pr}/{lev} y", "LOADED", e["positive_rate_of_decided"],
                "PASS" if ok else "FAIL", P_SW,
                f"results.{m}.{pr}.{lev}.positive_rate_of_decided", None,
                f"re-derived n_contradiction/n_decided = {e['n_contradiction']}/{e['n_decided']}")
            rec(F, f"{m}/{pr}/{lev} x", "LOADED", float(lev), "PASS", P_SW,
                f"results.{m}.{pr}.{lev}.true_prevalence")
            n_pts += 1
rec(F, "clinical marker x (notebook)", "DERIVED", pharm, "PASS", P_S2,
    "…ambiguous_as_negative_n299.base_rate")
rec(F, "clinical marker y (notebook)", "DERIVED", round(wrate, 6), "PASS", P_S01,
    "crosstab 1425/1679")
rec(F, "clinical marker (fig_batch2.py literal)", "TYPED", [0.04, 0.849], "FAIL", None, None,
    "ax.plot([0.04],[0.849],'s',...)",
    "fig_batch2.py:35 hard-codes the marker. True values are 0.040134 and 0.848719 "
    "(step2_signal_check.json / step0_1_join_and_calibration.json). The notebook loads them; "
    "the .py rounds them into literals.")
rec(F, "clinical annotation text (fig_batch2.py literal)", "TYPED", "clinical (4% -> 85%)", "FAIL",
    None, None, "ax.annotate('clinical\\n(4% -> 85%)')",
    "fig_batch2.py:37. Notebook builds the same string from the loaded values (4% -> 85%), so the "
    "displayed text is correct, but in the .py it is a literal.")
rec(F, "identity/perfect-calibration line", "SETTING", [[0, 0.28], [0, 0.28]], "PASS")
rec(F, "xlim/ylim", "SETTING", [[0, 0.285], [0, 0.92]], "PASS")

# ===========================================================================
# 5. fig-falsification
# ===========================================================================
F = "fig-falsification"
ag = decagg["aggregate"]
aN, aD = ag["alpha_n"]["mean"], ag["alpha_d"]["mean"]
sN, sD = ag["alpha_n"]["std"], ag["alpha_d"]["std"]
rec(F, "alpha_N mean", "LOADED", aN, "PASS", P_DECAGG, "aggregate.alpha_n.mean")
rec(F, "alpha_D mean", "LOADED", aD, "PASS", P_DECAGG, "aggregate.alpha_d.mean")
rec(F, "alpha_N std (error bar)", "LOADED", sN, "PASS", P_DECAGG, "aggregate.alpha_n.std",
    None, "std across 3 seeds, NOT sem; n=3")
rec(F, "alpha_D std (error bar)", "LOADED", sD, "PASS", P_DECAGG, "aggregate.alpha_d.std")
rec(F, "alpha_N initialisation bar = 0.5", "TYPED", 0.5, "FAIL", None, None, "a.bar(...,[0.5,1.0],...)",
    "Initialisation values are literals in BOTH make_figures_extra.py:61 and notebook cell 16. "
    "No committed result file records alpha_n_init/alpha_d_init; CORRECTIONS.md C1 is prose. "
    "Value is plausible but UNVERIFIABLE from any stored file.")
rec(F, "alpha_D initialisation bar = 1.0", "TYPED", 1.0, "FAIL", None, None, "a.bar(...,[0.5,1.0],...)",
    "same as above")
pctN = 100 * (aN - 0.5) / 0.5
pctD = 100 * (aD - 1.0) / 1.0
rec(F, "percent-change label alpha_N", "DERIVED", round(pctN, 4), "FAIL(dependency)", None, None,
    "100*(0.5076-0.5)/0.5 = +1.5%", "arithmetic is correct but one operand (0.5) is TYPED")
rec(F, "percent-change label alpha_D", "DERIVED", round(pctD, 4), "FAIL(dependency)", None, None,
    "100*(1.00377-1.0)/1.0 = +0.4%", "arithmetic is correct but one operand (1.0) is TYPED")
for cohort, key in [("HF (in domain)", "hf_reference"), ("sepsis", "sepsis"), ("AKI", "aki")]:
    v = trans[key]["rule_type_f1"]
    rec(F, f"rule-type F1 {cohort}", "LOADED", v, "PASS", P_TRANS, f"{key}.rule_type_f1",
        None, "notebook cell 16 loads this; make_figures_extra.py:71 types it as a literal")
rec(F, "rule-type F1 list (make_figures_extra.py literal)", "TYPED", [0.9603, 0.0654, 0.0595],
    "FAIL(cosmetic)", P_TRANS, "hf_reference/sepsis/aki .rule_type_f1",
    "f1 = [0.9603, 0.0654, 0.0595]",
    "make_figures_extra.py:71. The literals round-trip to the stored values "
    "(0.9603 / 0.065439 / 0.059545) at the 4dp the figure prints, so the canvas is not wrong, but "
    "the values are typed rather than read. Note hf_reference.rule_type_f1 is itself stored only to "
    "4dp (0.9603) while sepsis/aki are full precision.")
claim = next(r for r in npa["records"] if "45.2" in r["claim"])
rec(F, "held-out rule accuracy reference line 0.4515", "TYPED", 0.4515, "FAIL", None, None,
    "b.axhline(0.4515) / UNSOURCED_held_out_rule_acc = 0.4515",
    "STILL DRAWN, in both make_figures_extra.py:74 and notebook cell 16. "
    f"number_provenance_audit.json rates it {claim['verdict']}: \"{claim['note']}\". "
    "No file under results/ contains 0.4515 or 0.452 as a held-out rule accuracy. "
    "There is no correct value to substitute; the line has no support and should be removed.")
rec(F, "'45.2% held-out rule acc.' text label", "TYPED", "45.2%", "FAIL", None, None, None,
    "same unsourced quantity, printed as text")
rec(F, "left ylim", "SETTING", [0, 1.35], "PASS")
rec(F, "right ylim", "SETTING", [0, 1.12], "PASS")
rec(F, "bar offsets +/-0.18, width 0.35/0.55", "SETTING", [0.18, 0.35, 0.55], "PASS")
rec(F, "text x=2.35 for the held-out label", "SETTING", 2.35, "PASS")

# ===========================================================================
# 6. fig-power
# ===========================================================================
F = "fig-power"
mda = pw["minimum_detectable_auc"]
npos, nneg = pw["n_positives"], pw["n_negatives"]
alpha, power = pw["alpha"], pw["power"]
rec(F, "this-study n_positives (x of red marker)", "LOADED", npos, "PASS", P_PW, "n_positives")
rec(F, "this-study n_negatives", "LOADED", nneg, "PASS", P_PW, "n_negatives")
rec(F, "alpha", "LOADED", alpha, "PASS", P_PW, "alpha")
rec(F, "power", "LOADED", power, "PASS", P_PW, "power")
mde_recalc = hanley_mde(npos, nneg, alpha, power)
z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
se_null = math.sqrt((0.25 + (npos - 1) * (1 / 3 - 0.25) + (nneg - 1) * (1 / 3 - 0.25)) / (npos * nneg))
mde_nullse = 0.5 + z * se_null
rec(F, "this-study MDE (y of red marker)", "LOADED", mda, "FAIL", P_PW, "minimum_detectable_auc",
    None,
    f"The marker is drawn at the FILE value {mda:.4f}, but the CURVE on the same axes is drawn by "
    f"hanley_mde(), which at n_pos=12,n_neg=241 returns {mde_recalc:.4f}. The marker therefore does "
    f"not lie on the plotted curve (gap {mda-mde_recalc:+.4f}). The file value is also not "
    f"reproducible by the null-variance route either: 0.5 + z*se_auc_under_null = {mde_nullse:.4f} "
    f"(se_auc_under_null={pw['se_auc_under_null']}, z={z:.6f}). Neither method yields 0.7382.")
rec(F, "se_auc_under_null cross-check", "DERIVED", round(se_null, 6),
    "PASS" if close(se_null, pw["se_auc_under_null"], 5e-5) else "FAIL", P_PW,
    "se_auc_under_null", "Hanley-McNeil null variance", "file stores 0.08555")
rec(F, "gap_above_chance cross-check", "DERIVED", round(mda - 0.5, 6),
    "PASS" if close(mda - 0.5, pw["gap_above_chance"], 1e-9) else "FAIL", P_PW,
    "gap_above_chance", "minimum_detectable_auc - 0.5")
rec(F, "MDE curve (60 points)", "DERIVED", "hanley_mde over logspace(5,3000)", "PASS", P_PW,
    None, "closed-form root search on the Hanley-McNeil variance",
    "deterministic, no RNG; uses alpha/power read from the file in the notebook version")
ml = tr["concat_s1"]["data_lineage"]
mort_pos = ml["test_n_pos"]
mort_neg = ml["test_n"] - ml["test_n_pos"]
rec(F, "mortality-task n_pos (notebook)", "LOADED", mort_pos, "PASS", P_TR,
    "concat_s1.data_lineage.test_n_pos")
rec(F, "mortality-task n_neg (notebook)", "DERIVED", mort_neg, "PASS", P_TR,
    "concat_s1.data_lineage.test_n - test_n_pos", "6776-570")
rec(F, "mortality-task MDE", "DERIVED", round(hanley_mde(mort_pos, mort_neg, alpha, power), 6),
    "PASS", None, None, "hanley_mde(570,6206)")
rec(F, "mortality point (fig_batch2.py literals)", "TYPED", [570, 6206], "FAIL(cosmetic)", P_TR,
    "concat_s1.data_lineage", "hanley_mde([570],[6206])",
    "fig_batch2.py:162-165 types both class counts. They match task_runs.json exactly, "
    "so the canvas is right, but the numbers are typed.")
rec(F, "neg:pos ratio for the curve (fig_batch2.py literal)", "TYPED", "241/12", "FAIL(cosmetic)",
    P_PW, "n_negatives/n_positives", "n_neg = n_pos*(241/12)",
    "fig_batch2.py:151. Matches the file; notebook computes n_neg_study/n_pos_study instead.")
rec(F, "MDE fallback default (fig_batch2.py)", "TYPED", 0.7382, "FAIL(latent)", P_PW,
    "minimum_detectable_auc", "pw.get('minimum_detectable_auc', 0.7382)",
    "fig_batch2.py:131. A literal fallback that silently reproduces the file value if the key "
    "ever disappears; it would draw a number with no source and no warning.")
rec(F, "xlim/ylim/log x", "SETTING", [[5, 3000], [0.5, 1.0]], "PASS")
rec(F, "text anchors (700,0.93) and (9,0.56)", "SETTING", [[700, 0.93], [9, 0.56]], "PASS")

# ===========================================================================
# 7. fig-forest
# ===========================================================================
F = "fig-forest"
rows = []
for k, v in iv["E2_mcc_intervals"].items():
    rows.append((f"MCC {k}", v["mcc"], v["ci95"][0], v["ci95"][1]))
    rec(F, f"row MCC {k}", "LOADED", [v["mcc"], v["ci95"]], "PASS", P_IV,
        f"E2_mcc_intervals.{k}.mcc / .ci95")
af, cf = dec["aggregate"]["ensemble_auroc"], dec["aggregate"]["ensemble_ci95"]
rec(F, "row AUROC-0.5 (decontaminated)", "DERIVED",
    [round(af - 0.5, 6), round(cf[0] - 0.5, 6), round(cf[1] - 0.5, 6)], "PASS", P_DEC,
    "aggregate.ensemble_auroc / ensemble_ci95", "value - 0.5 (notebook)")
rec(F, "row AUROC-0.5 (decontaminated) — fig_batch2.py literals", "TYPED",
    [0.5145, 0.321, 0.703], "FAIL(cosmetic)", P_DEC, "aggregate.ensemble_auroc / ensemble_ci95",
    "rows.append(('AUROC - 0.5 (decontaminated)', 0.5145-0.5, 0.321-0.5, 0.703-0.5))",
    f"fig_batch2.py:96. Stored values are {af!r} and {cf!r}; the literals are 4/3dp truncations. "
    "Effect on the plotted estimate: 0.0145 vs 0.01452282 (invisible). Notebook loads them.")
rec(F, "row AUROC-0.5 (defect-free stratum)", "DERIVED",
    [round(e3["auroc"] - 0.5, 6), round(e3["unstratified_ci95"][0] - 0.5, 6),
     round(e3["unstratified_ci95"][1] - 0.5, 6)], "PASS", P_IV,
    "E3_defect_free_stratum.auroc / unstratified_ci95", "value - 0.5")
rec(F, "row AUROC-0.5 (defect-free) — fig_batch2.py literals", "TYPED",
    [0.5078, 0.2217, 0.7615], "FAIL(cosmetic)", P_IV, "E3_defect_free_stratum.*",
    "rows.append(... 0.5078-0.5, 0.2217-0.5, 0.7615-0.5)",
    "fig_batch2.py:97. Literals equal the stored values exactly (the file is stored to 4dp).")
pb = [tr[f"concat_s{s}"]["auroc"] for s in (1, 2, 3)]
ou = [tr[f"ours_current_s{s}"]["auroc"] for s in (0, 1, 2)]
g = st.mean(pb) - st.mean(ou)
se = math.sqrt(st.stdev(pb) ** 2 / 3 + st.stdev(ou) ** 2 / 3)
for s in (1, 2, 3):
    rec(F, f"concat_s{s} auroc", "LOADED", tr[f"concat_s{s}"]["auroc"], "PASS", P_TR,
        f"concat_s{s}.auroc")
for s in (0, 1, 2):
    rec(F, f"ours_current_s{s} auroc", "LOADED", tr[f"ours_current_s{s}"]["auroc"], "PASS", P_TR,
        f"ours_current_s{s}.auroc")
rec(F, "row C37 gap estimate", "DERIVED", round(g, 6), "PASS", P_TR, None,
    "mean(concat_s1..3) - mean(ours_current_s0..2)",
    "arms use MISMATCHED seed sets (1,2,3 vs 0,1,2); treated as unpaired")
rec(F, "row C37 gap CI", "DERIVED", [round(g - 1.96 * se, 6), round(g + 1.96 * se, 6)], "PASS",
    P_TR, None, "g +/- 1.96*sqrt(sd_a^2/3 + sd_b^2/3)",
    "normal-approx CI on n=3 vs n=3; a t critical value (t_4=2.776) would widen it ~42%")
p100 = [x["accuracy"] for x in sc if x["tag"] == "scratch_100M"]
pbert = [x["accuracy"] for x in sc if x["tag"] == "pretrained_bert"]
g44 = st.mean(pbert) - st.mean(p100)
se44 = math.sqrt(st.stdev(pbert) ** 2 / 3 + st.stdev(p100) ** 2 / 3)
rec(F, "row C44 gap estimate", "DERIVED", round(g44, 6), "PASS", P_SC,
    "[tag==pretrained_bert].accuracy / [tag==scratch_100M].accuracy",
    "mean(pretrained_bert) - mean(scratch_100M)", "uses the test 'accuracy' key, not a val key")
rec(F, "row C44 gap CI", "DERIVED", [round(g44 - 1.96 * se44, 6), round(g44 + 1.96 * se44, 6)],
    "PASS", P_SC, None, "g44 +/- 1.96*sqrt(sd^2/3+sd^2/3)",
    "the ONLY row that excludes zero; with t_4=2.776 the CI is "
    f"[{g44-2.776*se44:.4f}, {g44+2.776*se44:.4f}] and still excludes zero")
fr2, P_FR2 = J(RA, "control", "final_report.json")
_e34 = fr2["ensemble_mean_of_seed_scores"]["gold_defect_free_rule34"]
rec(F, "row 'AUROC-0.5 (defect-free stratum)' model provenance", "LOADED", e3["auroc"], "FAIL",
    P_IV, "E3_defect_free_stratum.auroc", None,
    "The row directly above it is labelled '(decontaminated)'; this one is labelled only "
    "'(defect-free stratum)' and a reader will assume the same model. It is not: 0.5078 is the "
    "CONTAMINATED released score (identical to control/final_report.json "
    "accord_contra_prob_same_299_pairs.gold_defect_free_rule34.auroc). The matched decontaminated "
    f"value is {_e34['auroc']} with CI {_e34['bootstrap']['ci95']}, i.e. the row should sit at "
    f"{_e34['auroc']-0.5:+.4f} CI [{_e34['bootstrap']['ci95'][0]-0.5:+.4f}, "
    f"{_e34['bootstrap']['ci95'][1]-0.5:+.4f}]. It still crosses zero, so the headline "
    "'7 of 8 cross zero' survives, but the plotted point and interval are from the wrong run.")
rec(F, "zero reference line", "SETTING", 0, "PASS")
rec(F, "row ordering (sorted by estimate)", "SETTING", "rows.sort(key=estimate)", "PASS")
allrows = rows + [(0, g, g - 1.96 * se, g + 1.96 * se), (0, g44, g44 - 1.96 * se44, g44 + 1.96 * se44),
                  (0, af - 0.5, cf[0] - 0.5, cf[1] - 0.5),
                  (0, e3["auroc"] - 0.5, e3["unstratified_ci95"][0] - 0.5, e3["unstratified_ci95"][1] - 0.5)]
n_cross = sum(1 for r in allrows if r[2] <= 0 <= r[3])
rec(F, "grey/coloured encoding (interval contains zero)", "DERIVED", f"{n_cross} of {len(allrows)}",
    "PASS", None, None, "lo <= 0 <= hi", "7 of 8 cross zero; recomputed independently")

# ===========================================================================
# 8. fig-disc-strata
# ===========================================================================
F = "fig-disc-strata"
agg = dec["aggregate"]
rec(F, "row1 AUROC", "LOADED", agg["ensemble_auroc"], "PASS", P_DEC, "aggregate.ensemble_auroc")
rec(F, "row1 CI95", "LOADED", agg["ensemble_ci95"], "PASS", P_DEC, "aggregate.ensemble_ci95")
rec(F, "row1 n", "LOADED", dec["n"], "FAIL(label)", P_DEC, "n", None,
    "The row is labelled 'All annotated pairs' but n=253, which is the ambiguous-EXCLUDED core. "
    "The annotated gold set is 299 (defect_free_strata.json strata[ALL].n = 299, "
    "reconciliation_c28 pharm_label_value_counts sums to 299). 46 pharmacist-ambiguous pairs are "
    "dropped. The number printed on the canvas (n=253) is right; the words next to it are not.")
rec(F, "row1 n_positives", "LOADED", dec["n_positives"], "PASS", P_DEC, "n_positives")
rec(F, "row1 MWU p", "LOADED", agg["ensemble_mannwhitney_p"], "PASS", P_DEC,
    "aggregate.ensemble_mannwhitney_p")
per_seed_mean = st.mean([dec["per_seed"][f"seed{i}"]["auroc"] for i in range(3)])
rec(F, "auroc_mean cross-check (not plotted)", "DERIVED", round(per_seed_mean, 6),
    "PASS" if close(per_seed_mean, agg["auroc_mean"], 1e-9) else "FAIL", P_DEC,
    "aggregate.auroc_mean", "mean(per_seed.seed0..2.auroc)",
    "confirms the ensemble AUROC (0.5145) is a distinct quantity from the seed mean (0.5211); "
    "the figure plots the ensemble, matching the CI it is paired with")
rec(F, "row2 AUROC", "LOADED", e3["auroc"], "PASS", P_IV, "E3_defect_free_stratum.auroc")
rec(F, "row2 unstratified CI95", "LOADED", e3["unstratified_ci95"], "PASS", P_IV,
    "E3_defect_free_stratum.unstratified_ci95")
rec(F, "row2 stratified CI95 (diamond)", "LOADED", e3["stratified_ci95"], "PASS", P_IV,
    "E3_defect_free_stratum.stratified_ci95")
rec(F, "row2 n / n_pos", "LOADED", [e3["n"], e3["n_pos"]], "PASS", P_IV,
    "E3_defect_free_stratum.n / n_pos")
rec(F, "row2 MWU p (notebook)", "LOADED", r34["mwu_p"], "PASS", P_STR,
    "strata[rules 3+4 (DEFECT-FREE)].mwu_p")
rec(F, "row2 MWU p (make_figures.py literal 0.953)", "TYPED", 0.953, "FAIL(cosmetic)", P_STR,
    "strata[rules 3+4 (DEFECT-FREE)].mwu_p", "rows = [... 0.953, ...]",
    "make_figures.py:730 types it; it equals the stored value exactly. Notebook loads it.")
rec(F, "footer: draws dropped", "LOADED", e3["draws_dropped_zero_positives"], "PASS", P_IV,
    "E3_defect_free_stratum.draws_dropped_zero_positives")
rec(F, "footer: bootstrap B", "LOADED", iv["bootstrap_B"], "PASS", P_IV, "bootstrap_B")
rec(F, "drop_rate cross-check", "DERIVED",
    round(e3["draws_dropped_zero_positives"] / iv["bootstrap_B"], 6),
    "PASS" if close(e3["draws_dropped_zero_positives"] / iv["bootstrap_B"], e3["drop_rate"], 1e-9) else "FAIL",
    P_IV, "E3_defect_free_stratum.drop_rate", "20/10000 = 0.002")
fr, P_FR = J(RA, "control", "final_report.json")
contam34 = fr["accord_contra_prob_same_299_pairs"]["gold_defect_free_rule34"]
ens34 = fr["ensemble_mean_of_seed_scores"]["gold_defect_free_rule34"]
rec(F, "row1 vs row2 come from the same model", "LOADED", False, "FAIL", P_IV,
    "E3_defect_free_stratum.auroc vs decontam/corrected_gold_metrics.json aggregate.ensemble_auroc",
    None,
    "BRANCH MISMATCH. Row 1 (0.5145, n=253) is the ACCORD v2 DECONTAMINATED 3-seed ensemble "
    "(seeds 0/1/2, ambiguous pairs excluded). Row 2 (0.5078, n=134) is byte-identical to "
    f"control/final_report.json accord_contra_prob_same_299_pairs.gold_defect_free_rule34.auroc = "
    f"{contam34['auroc']} -- the CONTAMINATED released per-pair score, on a 134-of-299 subset in "
    "which pharmacist-ambiguous pairs are counted as negatives. The two rows differ in BOTH the "
    "model and the denominator convention, and are drawn on one axis as one comparison with no "
    "marking. A matched-branch value exists in the same repository: "
    f"control/final_report.json ensemble_mean_of_seed_scores.gold_defect_free_rule34.auroc = "
    f"{ens34['auroc']}, CI {ens34['bootstrap']['ci95']}. "
    "number_provenance_audit.json statistical_concerns S6 flags exactly this.")
rec(F, "chance line 0.5", "SETTING", 0.5, "PASS")
rec(F, "xlim/ylim", "SETTING", [[0.1, 0.9], [-0.95, 1.55]], "PASS")

# ===========================================================================
# 9. fig-scaling
# ===========================================================================
F = "fig-scaling"
by, pbt = {}, {}
for x in sc:
    by.setdefault(x["tag"], []).append(x["accuracy"])
    pbt[x["tag"]] = x["non_emb_params"]
for t in ["scratch_10M", "scratch_30M", "scratch_100M", "pretrained_bert"]:
    rec(F, f"{t} accuracies (3 seeds)", "LOADED", by[t], "PASS", P_SC,
        f"[tag=={t}].accuracy", None,
        "scaling_results.json carries ONLY 'accuracy' (no val_accuracy), so the "
        "val/test key confusion in scaling/convergence_results.json cannot bite here")
    rec(F, f"{t} mean (marker y)", "DERIVED", round(st.mean(by[t]), 6), "PASS", P_SC, None,
        f"mean of {len(by[t])} seeds")
    rec(F, f"{t} stdev (error bar)", "DERIVED", round(st.stdev(by[t]), 6), "PASS", P_SC, None,
        "sample stdev, n=3", "std NOT sem; only 3 seeds")
    rec(F, f"{t} non_emb_params (marker x)", "LOADED", pbt[t], "PASS", P_SC,
        f"[tag=={t}].non_emb_params")
pm = st.mean(by["pretrained_bert"])
for t in ["scratch_10M", "scratch_30M", "scratch_100M"]:
    rec(F, f"gap annotation {t}", "DERIVED", round(pm - st.mean(by[t]), 6), "PASS", P_SC, None,
        "mean(pretrained_bert) - mean(tag)")
rec(F, "inset PubMedBERT band mean/std", "DERIVED", [round(st.mean(pb), 6), round(st.stdev(pb), 6)],
    "PASS", P_TR, "concat_s1..3.auroc", "mean/stdev of 3 seeds")
rec(F, "inset from-scratch point mean/std", "DERIVED",
    [round(st.mean(ou), 6), round(st.stdev(ou), 6)], "PASS", P_TR,
    "ours_current_s0..2.auroc", "mean/stdev of 3 seeds")
rec(F, "inset title gap", "DERIVED", round(g, 6), "PASS", P_TR, None,
    "mean(concat) - mean(ours_current)")
rec(F, "inset x position of the from-scratch point", "TYPED", 1.06e7, "FAIL(cosmetic)", P_TR,
    "ours_current_s0.non_emb_params", "ins.errorbar([1.06e7], ...)",
    "Both fig_batch2.py:85 and notebook cell 24 place the point at a typed 1.06e7. The notebook "
    "even LOADS clinical_params = 10638720, prints it, and then does not use it. The inset x axis "
    "is unlabelled and untickmarked, so nothing visible is wrong; still a typed coordinate where a "
    "loaded one was available.")
rec(F, "inset yticks [0.75,0.77,0.79]", "SETTING", [0.75, 0.77, 0.79], "PASS")
rec(F, "main ylim", "SETTING", [0.865, 0.952], "PASS")
rec(F, "label anchor x=1.05e7", "SETTING", 1.05e7, "PASS")
rec(F, "inset axes rect / xlim", "SETTING", [[0.60, 0.24, 0.28, 0.24], [5e6, 2e8]], "PASS")

# ===========================================================================
# 10. fig-threshold-surface
# ===========================================================================
F = "fig-threshold-surface"
cells = thr["cells"]
rates = sorted({c["achieved_rate"] for c in cells})
betas = sorted({c["beta"] for c in cells})
Z = [c["demographic_parity_gap"] for c in cells]
M = [c["mde_demographic_parity_gap"] for c in cells]
rec(F, "grid: n operating points x n betas", "LOADED", [len(rates), len(betas)],
    "PASS" if (len(rates) == thr["n_operating_points"] and len(betas) == len(thr["beta_grid"])) else "FAIL",
    P_THR, "cells[].achieved_rate / cells[].beta",
    None, f"file also asserts n_operating_points={thr['n_operating_points']}, total_cells={thr['total_cells']}")
rec(F, "270 demographic_parity_gap values (colour)", "LOADED",
    {"min": round(min(Z), 6), "max": round(max(Z), 6), "n": len(Z)}, "PASS", P_THR,
    "cells[].demographic_parity_gap")
rec(F, "270 mde_demographic_parity_gap values (veil)", "LOADED",
    {"min": round(min(M), 6), "max": round(max(M), 6), "n": len(M)}, "PASS", P_THR,
    "cells[].mde_demographic_parity_gap")
n_under = sum(1 for c in cells if c["mde_demographic_parity_gap"] > 0.05)
n_notpow = sum(1 for c in cells if not c["adequately_powered"])
rec(F, "veiled (underpowered) cell count", "DERIVED", n_under,
    "PASS" if n_under == n_notpow else "FAIL", P_THR,
    "cells[].mde_demographic_parity_gap > 0.05 vs cells[].adequately_powered",
    "count(mde > 0.05)",
    f"the figure re-derives the veil from the tolerance; the file's own 'adequately_powered' flag "
    f"gives {n_notpow}. {'They agree.' if n_under == n_notpow else 'THEY DISAGREE.'}")
n_breach = sum(1 for c in cells if c["demographic_parity_gap"] > 0.05)
n_cellbreach = sum(1 for c in cells if c["cell_breach"])
rec(F, "cells above the 0.05 contour", "DERIVED", n_breach, "PASS(caveat)", P_THR,
    "cells[].demographic_parity_gap", "count(demographic_parity_gap > 0.05)",
    f"The black contour is drawn on the POINT ESTIMATE, so {n_breach} of 270 cells lie above it. "
    f"The file's own formal breach count is a different quantity: n_breach_cells="
    f"{thr['n_breach_cells']} (= cells[].cell_breach, which also requires adequate power). "
    "The figure never prints 36, so nothing on the canvas is wrong, but a reader who reads the "
    "contour as 'the audit's breach set' will be off by 173 cells.")
e9 = thr["original_e9_population_pred_pos_rate"]
rec(F, "E9 operating-point line y (notebook)", "LOADED", e9, "PASS", P_THR,
    "original_e9_population_pred_pos_rate")
rec(F, "E9 operating-point line y (fig_threshold_surface.py literal)", "TYPED", 0.1994,
    "FAIL(cosmetic)", P_THR, "original_e9_population_pred_pos_rate",
    "ax.plot(B, [0.1994]*len(B), ...)",
    f"fig_threshold_surface.py:41-43. Stored value is {e9!r}; 0.1994 is a 4dp rounding. "
    "The notebook loads it.")
rec(F, "0.05 tolerance (contour + veil threshold)", "SETTING", 0.05, "PASS", None, None, None,
    "the audit's own tolerance; named TOLERANCE in the notebook, a bare 0.05 in the .py. "
    "It is a decision rule, not a measurement, and is not stored in the results file.")
rec(F, "colour scale vmin/vmax", "SETTING", [0, 0.20], "PASS")
rec(F, "veil alpha 0.62", "SETTING", 0.62, "PASS")
rec(F, "annotation ytext 0.132", "SETTING", 0.132, "PASS")
rec(F, "data source path", "LOADED", "cells", "PASS(divergent)", P_THR, "cells",
    None,
    "The notebook reads results/accord/selection/threshold_robustness_results.json (committed). "
    "fig_threshold_surface.py:20 instead reads "
    "/Users/rickygole/mimic_local_only/accord_selection/threshold_robustness_cells.csv, which is "
    "OUTSIDE the repository and not committed. Both exist on this machine; only one is auditable "
    "by anyone else.")

# ===========================================================================
# 11. fig-disc-actionability
# ===========================================================================
F = "fig-disc-actionability"
for nm, val in [("timeline x positions", [1.0, 4.0, 5.4, 8.2]),
                ("label ytext", [3.35, 3.0, 3.55, 3.05]),
                ("baseline y", 2.3), ("xlim", [0, 10.5]), ("ylim", [0, 4.0]),
                ("shaded span", [5.4, 8.2]), ("figsize", [6.6, 3.4])]:
    rec(F, nm, "SETTING", val, "PASS", None, None, None,
        "schematic layout coordinate; the x axis has no scale and no tick labels, so these "
        "encode ordering only, not durations")
rec(F, "4 event labels + 3 annotations", "SETTING", "categorical text", "PASS", None, None, None,
    "asserts a MIMIC-III schema fact (CHARTTIME null on discharge summaries) sourced to "
    "accord/CORRECTIONS.md C6, which is prose, not a result file")
rec(F, "numeric estimates on canvas", "DERIVED", 0, "PASS", None, None, None,
    "there are none; nothing to trace")

# ===========================================================================
# 12. fig-met-exchange
# ===========================================================================
F = "fig-met-exchange"
sysmap = {"ACCORD": "accord_binary_at_0.5_253core", "GPT-4o zero-shot": "gpt4o_zeroshot",
          "GPT-4o few-shot": "gpt4o_fewshot", "Llama 3 8B": "llama_baseline"}
ref = s3["accord_binary_at_0.5_253core"]
TP0, TN0, FP0, FN0 = ref["tp"], ref["tn"], ref["fp"], ref["fn"]
P, N = TP0 + FN0, TN0 + FP0
for nm, k in sysmap.items():
    s = s3[k]
    tot = s["tp"] + s["tn"] + s["fp"] + s["fn"]
    ok = tot == s["n"] == 253
    rec(F, f"{nm} confusion matrix", "LOADED", [s["tp"], s["tn"], s["fp"], s["fn"]],
        "PASS" if ok else "FAIL", P_S3, f"{k}.tp/tn/fp/fn", None,
        f"tp+tn+fp+fn = {tot}, file n = {s['n']}")
    rec(F, f"{nm} marker (delta_TP, delta_TN)", "DERIVED", [TP0 - s["tp"], s["tn"] - TN0],
        "PASS", P_S3, None, f"({TP0}-{s['tp']}, {s['tn']}-{TN0})")
    rec(F, f"{nm} recall cross-check", "DERIVED", round(s["tp"] / (s["tp"] + s["fn"]), 4),
        "PASS" if close(s["tp"] / (s["tp"] + s["fn"]), s["recall"], 1e-4) else "FAIL", P_S3,
        f"{k}.recall", "tp/(tp+fn)")
    rec(F, f"{nm} precision cross-check", "DERIVED", round(s["tp"] / (s["tp"] + s["fp"]), 4),
        "PASS" if close(s["tp"] / (s["tp"] + s["fp"]), s["precision"], 1e-4) else "FAIL", P_S3,
        f"{k}.precision", "tp/(tp+fp)")
mcc0 = mcc_counts(TP0, TN0, FP0, FN0)
rec(F, "P (positives)", "DERIVED", P, "PASS", P_S3, None, "tp+fn = 11+1")
rec(F, "N (negatives)", "DERIVED", N, "PASS", P_S3, None, "tn+fp = 18+223")
rec(F, "balanced-accuracy slope N/P", "DERIVED", round(N / P, 6), "PASS", P_S3, None, "241/12")
rec(F, "PABAK slope", "SETTING", 1, "PASS", None, None, None,
    "closed-form property of the statistic, not data")
rec(F, "MCC indifference level (legend)", "DERIVED", round(mcc0, 6),
    "PASS" if close(mcc0, s3["accord_binary_at_0.5_253core"]["mcc"], 5e-5) else "FAIL", P_S3,
    "accord_binary_at_0.5_253core.mcc", "mcc_from_counts(11,18,223,1)",
    "re-derived from the confusion matrix; matches the file's stored mcc")
rec(F, "MCC indifference curve (60 brentq roots)", "DERIVED", "brentq over tn in [0,N]", "PASS",
    P_S3, None, "solve mcc_from_counts(TP0-d, tn, N-tn, P-TP0+d) = mcc0",
    "deterministic root-finding on loaded counts; NaN where no root exists")
rec(F, "xlim/ylim", "SETTING", [[-1, P + 1], [-32, N + 10]], "PASS")
rec(F, "marker size 45, 4 marker glyphs, 4 label offsets", "SETTING", "cosmetic", "PASS")

# ===========================================================================
# 13. fig-exemplar
# ===========================================================================
F = "fig-exemplar"
rec(F, "caption defect rate", "DERIVED", round(defect * 100, 4), "PASS", P_EX,
    "rule_1_2_pooled.n_wrong / rule_1_2_pooled.n_resolved", "65/137*100 = 47.445",
    "prints 47.4%. Uses the defensible ratio.")
rec(F, "n_wrong", "LOADED", n_wrong, "PASS", P_EX, "rule_1_2_pooled.n_wrong")
rec(F, "n_resolved", "LOADED", n_res, "PASS", P_EX, "rule_1_2_pooled.n_resolved")
corr = ex_all["rule_1_2_pooled"]["correct_rate_among_resolved"]
rec(F, "complement cross-check", "DERIVED", round(1 - defect, 6),
    "PASS" if close(1 - defect, corr, 5e-5) else "FAIL", P_EX,
    "rule_1_2_pooled.correct_rate_among_resolved", "1 - 65/137 = 0.5255")
rec(F, "struck supplied rate present in the same file but NOT used", "LOADED", 0.586,
    "PASS", P_EX, "rule_1_2_pooled.reference_full_2296_candidate_pool_rate_from_task_description",
    None, "neither 0.586 nor its complement 0.414 reaches the canvas in either the .py or the "
    "notebook; the caption text in make_figures.py's manifest entry still says '41.4%' but that "
    "string is in manifest.json, not on the figure")
rec(F, "5 slot labels M1..M5", "SETTING", "abstract", "PASS")
rec(F, "5 action tags CONT/CONT/CONT/CONT/CESS", "SETTING", "abstract", "PASS", None, None, None,
    "illustrative schematic, no count claimed")
rec(F, "column x, row y, box sizes, xlim/ylim", "SETTING",
    {"col_x": [2.2, 7.2], "row_dy": 0.78, "xlim": [0, 10], "ylim": [0, 6]}, "PASS")

# ===========================================================================
# Summary
# ===========================================================================
by_fig = {}
for r in V:
    by_fig.setdefault(r["figure"], []).append(r)

summary = {"by_classification": {}, "by_verdict": {}, "by_figure": {}}
for r in V:
    summary["by_classification"][r["classification"]] = \
        summary["by_classification"].get(r["classification"], 0) + 1
    summary["by_verdict"][r["verdict"]] = summary["by_verdict"].get(r["verdict"], 0) + 1
for f, rs in by_fig.items():
    summary["by_figure"][f] = {
        "n_values": len(rs),
        "LOADED": sum(1 for r in rs if r["classification"] == "LOADED"),
        "DERIVED": sum(1 for r in rs if r["classification"] == "DERIVED"),
        "SETTING": sum(1 for r in rs if r["classification"] == "SETTING"),
        "TYPED": sum(1 for r in rs if r["classification"] == "TYPED"),
        "n_fail": sum(1 for r in rs if r["verdict"].startswith("FAIL")),
        "clean": all(not r["verdict"].startswith("FAIL") for r in rs)
        and not any(r["classification"] == "TYPED" for r in rs),
    }

doc = {
    "generated_by": "accord/stats/audit_figure_data.py",
    "scope": "the thirteen thesis figures",
    "method": "Every value that reaches the canvas was classified by walking the drawing code, "
              "then every LOADED value was re-read from the primary result file and every DERIVED "
              "value recomputed from scratch. VERIFIED_FACTS.json and the figure manifests were "
              "NOT used as a source of truth.",
    "code_audited": [
        "accord/thesis/figures/make_figures.py",
        "accord/thesis/figures/fig_batch2.py",
        "accord/thesis/figures/fig_threshold_surface.py",
        "accord/thesis/figures/make_figures_extra.py",
        "accord/thesis/figures/Thesis_Figures_PerFigure.ipynb",
    ],
    "summary": summary,
    "figures": {f: rs for f, rs in by_fig.items()},
}
with open(OUT, "w") as fh:
    json.dump(doc, fh, indent=2)

print(f"wrote {OUT}")
print("by classification:", summary["by_classification"])
print("by verdict:", summary["by_verdict"])
print()
for f, s in summary["by_figure"].items():
    print(f"  {f:26s} n={s['n_values']:3d} L={s['LOADED']:3d} D={s['DERIVED']:3d} "
          f"S={s['SETTING']:2d} T={s['TYPED']:2d} fail={s['n_fail']:2d} "
          f"{'CLEAN' if s['clean'] else ''}")
print()
print("TYPED values:")
for r in V:
    if r["classification"] == "TYPED":
        print(f"  [{r['figure']}] {r['value_name']} = {r['plotted_value']!r}")
print()
print("Non-TYPED failures:")
for r in V:
    if r["verdict"].startswith("FAIL") and r["classification"] != "TYPED":
        print(f"  [{r['figure']}] {r['value_name']} -> {r['verdict']}")
