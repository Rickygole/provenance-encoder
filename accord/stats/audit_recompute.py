#!/usr/bin/env python3
"""Independent recomputation of every load-bearing quantitative claim in the
thesis. Nothing here is copied from the draft: each value is derived from the
artifact named in `source`, and compared against the literal in the .tex.

Re-run:  python3 accord/stats/audit_recompute.py
"""
import csv, json, math, os, sys
import numpy as np

R = "/Users/rickygole/provenance-encoder/results/accord"
Q = "/Users/rickygole/mimic_local_only/accord_supp/results"
OUT = {}

def j(p):
    with open(p) as fh:
        return json.load(fh)

def _dp(x):
    s = repr(float(x))
    return len(s.split(".")[1].rstrip("0")) if "." in s else 0

def rec(key, reported, recomputed, source, note=""):
    """Rounding-aware: a claim PASSES if the recomputed value, rounded to the
    number of decimals the draft actually prints, equals the printed literal."""
    ok = None
    if recomputed is not None and reported is not None:
        d = _dp(reported)
        ok = round(float(recomputed), d) == round(float(reported), d)
    OUT[key] = {"reported": reported, "recomputed": recomputed,
                "match": ok, "source": source, "note": note}
    return ok

def mcc(tp, fn, fp, tn):
    num = tp * tn - fp * fn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / den if den else 0.0

# ---------------------------------------------------------------- MCC / cells
c28 = j(f"{R}/reconciliation_c28.json")["d_measured_vs_what_the_tex_says"]["measured"]
w = c28["weak_label_on_253_non_ambiguous"]
rec("weak_labeller_MCC", 0.0243, mcc(w["TP"], w["FN"], w["FP"], w["TN"]),
    "reconciliation_c28.json weak_label_on_253_non_ambiguous", "ch5:311 ch7:467 ch8:346")
rec("weak_labeller_precision", 0.0493, w["TP"] / (w["TP"] + w["FP"]),
    "reconciliation_c28.json", "ch7:467 = 4.93%")
rec("weak_labeller_recall", 0.9167, w["TP"] / (w["TP"] + w["FN"]), "reconciliation_c28.json")
rec("weak_labeller_n", 253, w["TP"] + w["FN"] + w["FP"] + w["TN"], "reconciliation_c28.json")

# weak-labeller MCC bootstrap CI (C58 asserts [-0.1040, +0.0966]); recompute
def mcc_boot(tp, fn, fp, tn, B=10000, seed=42):
    y = np.array([1] * (tp + fn) + [0] * (fp + tn))
    p = np.array([1] * tp + [0] * fn + [1] * fp + [0] * tn)
    rng = np.random.default_rng(seed)
    n = len(y); out = []; degen = 0
    for _ in range(B):
        i = rng.integers(0, n, n)
        yy, pp = y[i], p[i]
        TP = int(((yy == 1) & (pp == 1)).sum()); FN = int(((yy == 1) & (pp == 0)).sum())
        FP = int(((yy == 0) & (pp == 1)).sum()); TN = int(((yy == 0) & (pp == 0)).sum())
        d = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        if d == 0:
            degen += 1; continue
        out.append((TP*TN - FP*FN) / d)
    a = np.array(out)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), degen, len(a)

lo, hi, degen, nb = mcc_boot(w["TP"], w["FN"], w["FP"], w["TN"])
OUT["weak_labeller_MCC_CI"] = {
    "reported": [-0.1040, 0.0966], "recomputed": [round(lo, 4), round(hi, 4)],
    "degenerate_draws": degen, "n_valid": nb,
    "match": abs(lo + 0.1040) < 0.02 and abs(hi - 0.0966) < 0.02,
    "source": "recomputed from reconciliation_c28.json cells; NO artifact on disk stores this CI",
    "note": "C58 / ch5:312 / ch8:354"}

# four-system MCC intervals actually on disk
iv = j(f"{R}/stats/intervals.json")["E2_mcc_intervals"]
b3 = j(f"{R}/step3_baselines.json")
for name, texmcc in [("gpt4o_zeroshot", 0.1089), ("gpt4o_fewshot", 0.0117),
                     ("accord_binary_at_0.5_253core", -0.0070), ("llama_baseline", -0.0436)]:
    s = b3[name]
    rec(f"MCC_derived_{name}", texmcc, mcc(s["tp"], s["fn"], s["fp"], s["tn"]),
        "step3_baselines.json cells", "ch8 tab:met-confusion derived-vs-reported check")

# balanced accuracy / PABAK from cells (ch8 tab:met-selection)
for name, ba, pab in [("gpt4o_zeroshot", 0.615, 0.439), ("gpt4o_fewshot", 0.514, -0.099),
                      ("accord_binary_at_0.5_253core", 0.496, -0.771), ("llama_baseline", 0.479, -0.834)]:
    s = b3[name]
    n = s["tp"] + s["fn"] + s["fp"] + s["tn"]
    rec(f"balacc_{name}", ba, 0.5 * (s["tp"]/(s["tp"]+s["fn"]) + s["tn"]/(s["tn"]+s["fp"])), "step3_baselines.json")
    rec(f"pabak_{name}", pab, 2 * (s["tp"] + s["tn"]) / n - 1, "step3_baselines.json")

# exchange rate
rec("Q_over_P", 20.08, 241 / 12, "n=253, P=12, Q=241 (step3_baselines / ch8)")
rec("balacc_spread", 0.14, 0.5*(6/12+176/241) - 0.5*(11/12+10/241), "derived from cells")
rec("pabak_spread", 1.27, (2*182/253 - 1) - (2*21/253 - 1), "derived from cells")
rec("exchange_threshold", 100.4, 5 * (241/12), "delta=5 x Q/P")
rec("delta_TN", 158, 176 - 18, "cells")

# ---------------------------------------------------------------- AUROC block
dc = j(f"{R}/decontam/corrected_gold_metrics.json")
ps = dc["per_seed"]
aur = [ps[k]["auroc"] for k in ("seed0", "seed1", "seed2")]
aps = [ps[k]["ap"] for k in ("seed0", "seed1", "seed2")]
rec("auroc_ensemble_0.5145", 0.5145, dc["aggregate"]["ensemble_auroc"], "decontam/corrected_gold_metrics.json")
rec("auroc_perseed_mean", 0.5211, float(np.mean(aur)), "decontam per_seed")
rec("auroc_perseed_std_ddof0", 0.0201, float(np.std(aur, ddof=0)), "decontam per_seed")
OUT["auroc_perseed_std_ddof1"] = {"reported": 0.0201, "recomputed": round(float(np.std(aur, ddof=1)), 4),
    "match": False, "source": "decontam per_seed", "note": "ddof=1 would be this; draft uses ddof=0"}
rec("ap_mean_0.0579", 0.0579, float(np.mean(aps)), "decontam per_seed")
rec("ap_std_0.0034", 0.0034, float(np.std(aps, ddof=0)), "decontam per_seed", "ddof=0")
rec("mwu_p_0.867", 0.867, dc["aggregate"]["ensemble_mannwhitney_p"], "decontam aggregate")
rec("ci_lo_0.321", 0.321, dc["aggregate"]["ensemble_ci95"][0], "decontam aggregate")
rec("ci_hi_0.703", 0.703, dc["aggregate"]["ensemble_ci95"][1], "decontam aggregate")
rec("decontam_n_is_253", 253, dc["n"], "decontam", "ch7 tab:auroc row 1 labels this n=299")

# control (subject-disjoint) 0.497 +/- 0.035  -- which ddof?
cf = j(f"{R}/control/final_report.json")
cv = cf["auroc_mean_std_across_seeds"]["gold_all"]["values"]
rec("control_auroc_mean_0.497", 0.497, float(np.mean(cv)), "control/final_report.json")
OUT["control_auroc_std_convention"] = {
    "reported": 0.035, "recomputed_ddof1": round(float(np.std(cv, ddof=1)), 4),
    "recomputed_ddof0": round(float(np.std(cv, ddof=0)), 4),
    "match": abs(float(np.std(cv, ddof=1)) - 0.035) < 5e-4,
    "source": "control/final_report.json",
    "note": "ch7:284 quotes ddof=1 while ch7:242/259 quote ddof=0 in the SAME comparison"}

# decontaminated precision / recall (quarantine aggregate, read-only)
ag = j(f"{Q}/accord_v2_decontam_gold_aggregate.json")["aggregate"]
rec("decontam_precision_0.0424", 0.0424, ag["precision"]["mean"], "accord_v2_decontam_gold_aggregate.json")
rec("decontam_precision_std_0.0023", 0.0023, ag["precision"]["std"], "same")
rec("decontam_recall_0.7778", 0.7778, ag["recall"]["mean"], "same")
rec("decontam_recall_std_0.0393", 0.0393, ag["recall"]["std"], "same")
rec("decontam_alpha_n", 0.5076, ag["alpha_n"]["mean"], "same")
rec("decontam_alpha_d", 1.0038, ag["alpha_d"]["mean"], "same")
OUT["alpha_move_pct"] = {
    "reported_in_draft": [1.8, 0.6],
    "recomputed_contaminated_single_seed": [round((0.509-0.5)/0.5*100, 2), round((1.006-1.0)/1.0*100, 2)],
    "recomputed_decontaminated_3seed": [round((ag["alpha_n"]["mean"]-0.5)/0.5*100, 2),
                                        round((ag["alpha_d"]["mean"]-1.0)/1.0*100, 2)],
    "match": False,
    "source": "accord_v2_decontam_gold_aggregate.json vs CORRECTIONS C16",
    "note": "C16 supersedes 1.8/0.6 with 1.5/0.4"}

# base rates
rec("base_rate_299", 0.040, 12/299, "12 positives of 299")
rec("base_rate_253", 0.0474, 12/253, "12 positives of 253")

# ---------------------------------------------------------------- weak labels
s01 = j(f"{R}/step0_1_join_and_calibration.json")
ct = s01["weak_confidence_is_degenerate"]["crosstab_label_x_confidence"]
pos = ct["0.95"]["contradiction"]; neg = ct["1.0"]["not_contradiction"]
rec("weak_pos_count_1425", 1425, pos, "step0_1")
rec("weak_neg_count_254", 254, neg, "step0_1")
rec("weak_total_1679", 1679, pos + neg, "step0_1")
rec("weak_pos_rate_84.9", 0.849, pos / (pos + neg), "step0_1")
rec("weak_neg_rate_15.1", 0.151, neg / (pos + neg), "step0_1")
rec("prevalence_ratio_21x", 21, (pos/(pos+neg)) / (12/299), "step0_1 + gold")
rec("expected_true_in_pool_67", 67, round((12/299) * 1679), "derivation 1")
rec("false_pos_floor_1358", 1358, pos - round((12/299) * 1679), "derivation 1")
rec("pct_positive_noise", 0.95, (pos - round((12/299)*1679)) / pos, "derivation 1", ">95% claim")

cal = s01["rule_stratified_calibration"]
for r in ("1", "2", "3", "4"):
    rec(f"rule{r}_llama_pos", None, cal[r]["llama_weak_label_pos_rate_on_these_299_pairs"], "step0_1")
    rec(f"rule{r}_accord_pos", None, cal[r]["accord_predicted_pos_rate_gold"], "step0_1")
    rec(f"rule{r}_true_prev", None, cal[r]["pharmacist_true_pos_rate_gold"], "step0_1")
gaps = {r: abs(cal[r]["accord_predicted_pos_rate_gold"] - cal[r]["llama_weak_label_pos_rate_on_these_299_pairs"]) * 100
        for r in ("1", "2", "3")}
OUT["student_teacher_gaps_pp"] = {
    "reported": [1.5, 0.0, 1.2],
    "recomputed": [round(gaps["1"], 4), round(gaps["2"], 4), round(gaps["3"], 4)],
    "match": all(abs(a - b) < 0.05 for a, b in zip([1.5, 0.0, 1.2], [gaps["1"], gaps["2"], gaps["3"]])),
    "source": "step0_1 rule_stratified_calibration", "note": "ch5:160"}
rec("true_prev_factor_4", 4.0, cal["2"]["pharmacist_true_pos_rate_gold"] / cal["1"]["pharmacist_true_pos_rate_gold"],
    "step0_1", "ch5:161 'more than a factor of four'")
rec("rule4_labeller_over_truth_12x", 12, cal["4"]["llama_weak_label_pos_rate_on_these_299_pairs"] / cal["4"]["pharmacist_true_pos_rate_gold"],
    "step0_1", "ch5:176 'more than twelve times'")

# ---------------------------------------------------------------- defect-free
df = {s["name"]: s for s in j(f"{R}/defect_free_strata.json")["strata"]}
rec("defect_free_auroc_0.5078", 0.5078, df["rules 3+4 (DEFECT-FREE)"]["auroc"], "defect_free_strata.json")
rec("defect_free_mwu_0.953", 0.953, df["rules 3+4 (DEFECT-FREE)"]["mwu_p"], "defect_free_strata.json")
rec("defect_free_n_134", 134, df["rules 3+4 (DEFECT-FREE)"]["n"], "defect_free_strata.json")
rec("defect_free_pos_6", 6, df["rules 3+4 (DEFECT-FREE)"]["pos"], "defect_free_strata.json")
rec("rules12_n_165", 165, df["rules 1+2 (exemplar bug present)"]["n"], "defect_free_strata.json",
    "ch7:679 says '165 of the 300'; the set is 299")
rec("ALL_n_299", 299, df["ALL"]["n"], "defect_free_strata.json")
rec("contaminated_auroc_0.460", 0.460, df["ALL"]["auroc"], "defect_free_strata.json")
rec("contaminated_253_auroc_0.4595", 0.4595, dc["contaminated_same_253_pairs"]["auroc"], "decontam")

# top-15
rec("top15_expected_0.6", 0.6, 15 * 12 / 299, "arithmetic")
OUT["top15_tp_decontam_seeds"] = {"reported": [1, 0, 0],
    "recomputed": [ps[k]["top15_tp"] for k in ("seed0", "seed1", "seed2")],
    "match": [ps[k]["top15_tp"] for k in ("seed0", "seed1", "seed2")] == [1, 0, 0],
    "source": "decontam per_seed", "note": "ch7:298"}

# ---------------------------------------------------------------- severity
s5 = j(f"{R}/step5_ranking.json")
rec("harm_rho", -0.0001, s5["full_299_harm_score_vs_danger_rating"]["spearman_rho"], "step5_ranking.json")
rec("harm_rho_p_0.999", 0.999, s5["full_299_harm_score_vs_danger_rating"]["p_value"], "step5_ranking.json")
rec("contra_rho", -0.029, s5["full_299_contra_prob_vs_danger_rating"]["spearman_rho"], "step5_ranking.json")
rec("contra_rho_p_0.62", 0.62, s5["full_299_contra_prob_vs_danger_rating"]["p_value"], "step5_ranking.json")
rec("harm_min_0.398", 0.398, s5["harm_score_range_check"]["observed_min"], "step5_ranking.json")
rec("harm_max_0.610", 0.610, s5["harm_score_range_check"]["observed_max"], "step5_ranking.json")
s2 = j(f"{R}/step2_signal_check.json")
rec("harm_auroc_0.504", 0.504, s2["harm_score_vs_pharm_label__ambiguous_as_negative_n299"]["auroc"], "step2")
rec("harm_mwu_0.96", 0.96, s2["accord_harm_score_distribution_by_group"]["mann_whitney_u_p_value_pos_vs_neg"], "step2")
rec("ambiguous_as_positive_auroc_0.490", 0.490, s2["contra_prob_vs_pharm_label__ambiguous_as_positive_n299"]["auroc"], "step2")

# ---------------------------------------------------------------- params
rec("lora_589824", 589824, 12 * 2 * (768 * 16 + 16 * 768), "closed form, ch7:137")
rec("lora_head_2307", 2307, 768 * 3 + 3, "closed form, ch7:138")
rec("lora_total_592131", 592131, 12*2*(768*16+16*768) + 768*3+3, "closed form, ch7:139")
rec("bertbase_plus_head", 110074887, 109482240 + 592647, "ch7:141 stage0 config1 corroboration")
rec("spec_total_1539785", 1539785, 527104 + 395269 + 2496 + 614916, "ch6 tab:spec row sum")
rec("pooler_590592", 590592, 768 * 768 + 768, "closed form ch6:415")
rec("three_layers_21263616", 21263616, 3 * 7087872, "closed form ch6:414")
rec("stage0_recon_22446855", 22446855, 3*7087872 + (768*768+768) + 592647, "ch6:409/417")
rec("live_ratio_36.88", 36.88, (21263616 + 592647) / 592647, "ch6:431 C54")
OUT["dead_pooler_ratio_37.9"] = {"reported_superseded": 37.9,
    "recomputed": round(22446855 / 592647, 4), "match": None,
    "source": "closed form", "note": "the figure C54 retired; draft now uses 36.88"}

with open(f"{R}/_audit_recompute.json", "w") as fh:
    json.dump(OUT, fh, indent=1)

bad = [k for k, v in OUT.items() if v.get("match") is False]
print(f"{len(OUT)} quantities recomputed; {len(bad)} MISMATCH")
for k in bad:
    print(f"  MISMATCH {k}: reported={OUT[k].get('reported', OUT[k].get('reported_in_draft'))} recomputed={OUT[k]['recomputed'] if 'recomputed' in OUT[k] else OUT[k]}")
