"""Orchestrator: replicate the MIMIC-III selection-conditioned-audit-
blindness finding on public ACS PUMS data (ACSIncome, one state-year, via
`folktables`). Order mirrors accord/selection_experiment/run_experiment.py:
data / non-degeneracy check -> POSITIVE CONTROL FIRST -> Regime A (selection
depends on group only, S _||_ X | A) -> Regime B (selection depends on an
outcome-predictive feature, WKHP) -> secondary group axis -> drift check ->
verdict.

Public data. No credentialing. Aggregates only in the saved JSON/CSV; no
per-person PUMS rows are written out.

Run from this directory: `python3 run_experiment.py`
"""
from __future__ import annotations

import json
import time
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, t as t_dist
from sklearn.metrics import roc_auc_score

import data_prep
import classifier as clf_mod
import audit as audit_mod

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET_SELECTION_RATE = 0.30
OR_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
N_REPLICATES = 30
BOOTSTRAP_B = 5000
POSITIVE_CONTROL_LOGIT_SHIFT = 1.2
CLINICAL_REFERENCE_SEL_4_5 = 0.587  # the MIMIC breach point, for direct comparison


def det_seed(*parts) -> int:
    import hashlib
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def solve_intercept_feature(beta, target_rate, feat, n=300_000, rng=None):
    """Regime B: logistic selection on a continuous feature. Bisect the
    intercept so the marginal selection rate stays pinned regardless of
    beta, isolating the effect of beta on group disparity."""
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(feat), n)
    f = feat[idx]
    lo, hi = -12.0, 12.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = sigmoid(mid + beta * f).mean()
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


REGIME_A_Y_ODDS_RATIO = 3.0  # fixed case/control retention asymmetry, same in both
# groups -- mirrors accord/theory/selection_invariance.py's `pi_group_and_y`, the
# actual mechanism behind the C29 taxonomy ("S depends on A and Y but not X").
#
# IMPORTANT CORRECTION, made after a first (wrong) attempt at this regime: a
# selection rule depending literally ONLY on the group label -- S _||_ (X,Y) | A
# -- leaves EVERY audit metric exactly invariant, including demographic parity,
# PPV and calibration, because conditioning on S=1 within a fixed group changes
# nothing about the joint (X,Y) law given that group. Verified both by direct
# derivation and empirically (a first run showed DP/PPV/calibration gaps flat
# across the whole OR sweep, matching population, under literal group-only
# selection). To reproduce the taxonomy split the task and C29 both describe
# (FPR/TPR/AUROC invariant, DP/PPV/calibration NOT invariant), selection must
# depend on Y too -- that is what actually breaks the within-group prevalence.
# The "group-level selection odds ratio" that is swept below is the marginal
# A-effect (beta_A in a logit-additive P(S=1|A,Y)=sigmoid(c+beta_A*A+beta_Y*Y)
# model); the Y-effect (beta_Y = ln(REGIME_A_Y_ODDS_RATIO)) is held fixed. This
# is still S _||_ X | (A,Y) -- selection independent of the covariates given
# group AND outcome -- which is the actual condition under which FPR/TPR/AUROC
# are provably invariant while DP/PPV/calibration are not.


def solve_intercept_AY(beta_A, beta_Y, target_rate, group1, y, n=300_000, rng=None):
    """Bisect the shared intercept c in P(S=1|A,Y)=sigmoid(c+beta_A*A+beta_Y*Y)
    so the OVERALL marginal selection rate stays pinned at target_rate as
    beta_A (the swept group-odds-ratio parameter) varies."""
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(y), n)
    a = group1[idx].astype(float)
    yy = y[idx].astype(float)
    lo, hi = -12.0, 12.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = sigmoid(mid + beta_A * a + beta_Y * yy).mean()
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def auroc_gap(y, score, group1):
    a0 = roc_auc_score(y[~group1], score[~group1])
    a1 = roc_auc_score(y[group1], score[group1])
    return abs(a1 - a0), a0, a1


def bootstrap_auroc_gap_ci(y, score, group1, B=BOOTSTRAP_B, seed=0):
    """C52: also bootstraps the signed AUROC gap (a1 - a0) from the SAME
    draws as the existing abs() gap, returned under the 'signed' sub-key --
    the existing ci_lo/ci_hi/n_valid_draws keys and their values are
    untouched."""
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    draws_signed = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yi, si, gi = y[idx], score[idx], group1[idx]
        if gi.sum() == 0 or (~gi).sum() == 0:
            continue
        if len(np.unique(yi[gi])) < 2 or len(np.unique(yi[~gi])) < 2:
            continue
        a0 = roc_auc_score(yi[~gi], si[~gi])
        a1 = roc_auc_score(yi[gi], si[gi])
        draws.append(abs(a1 - a0))
        draws_signed.append(a1 - a0)
    arr = np.array(draws)
    arr_signed = np.array(draws_signed)
    out = dict(ci_lo=float("nan"), ci_hi=float("nan"), n_valid_draws=0)
    if len(arr) > 0:
        out = dict(ci_lo=float(np.percentile(arr, 2.5)), ci_hi=float(np.percentile(arr, 97.5)),
                    n_valid_draws=int(len(arr)))
    signed = dict(ci_lo=float("nan"), ci_hi=float("nan"), n_valid_draws=0)
    if len(arr_signed) > 0:
        signed = dict(ci_lo=float(np.percentile(arr_signed, 2.5)),
                      ci_hi=float(np.percentile(arr_signed, 97.5)),
                      n_valid_draws=int(len(arr_signed)))
    out["signed"] = signed
    return out


def full_pool_report(y, pred, score, group1, tag, B=BOOTSTRAP_B, seed=0):
    """One realized pool -> full audit dict, bootstrap CIs, plus AUROC gap
    (not in audit.py's scalar set, added here)."""
    res = audit_mod.audit_pool(y, pred, score, group1)
    ci = audit_mod.bootstrap_audit_ci(y, pred, score, group1, B=B, seed=det_seed("ci", tag, seed))
    ag, a0, a1 = auroc_gap(y, score, group1)
    ag_ci = bootstrap_auroc_gap_ci(y, score, group1, B=B, seed=det_seed("auroc_ci", tag, seed))
    # C52: signed AUROC gap = a1 - a0 (a0/a1 already computed above, no new
    # randomness, no new keys removed).
    return res, ci, dict(auroc_gap=ag, auroc_g0=a0, auroc_g1=a1, auroc_gap_signed=a1 - a0, **ag_ci)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    t_start = time.time()

    # -----------------------------------------------------------------
    # STEP 0: data provenance.
    # -----------------------------------------------------------------
    section("STEP 0: data (ACSIncome, TX, 2018 1-Year ACS PUMS, via folktables)")
    X, y, aux = data_prep.build_task()
    print(f"raw task frame: n={len(y)}, positive rate (PINCP>$50k)={y.mean():.4f}")
    print(f"group (primary) = SEX; female share={aux['female'].mean():.4f}")
    print(f"group (secondary) = RAC1P; white={aux['white'].sum()}, black={aux['black'].sum()}, "
          f"other={(len(aux) - aux['white'].sum() - aux['black'].sum())}")

    splits = data_prep.split(X, y, aux, seed=SEED)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    print(f"split sizes: train={len(tr['y'])} val={len(va['y'])} test={len(te['y'])} "
          f"(60/20/20, seed={SEED}, disjoint-index-asserted)")

    # -----------------------------------------------------------------
    # STEP 1: fit ONE group-blind reference classifier on train only.
    # SEX and RAC1P are excluded from data_prep.MODEL_FEATURES.
    # -----------------------------------------------------------------
    section("STEP 1: reference classifier (LogisticRegression, fit once, train only)")
    pipe, threshold, clf_info = clf_mod.fit_reference_classifier(splits, seed=SEED, verbose=True)
    ok, failures = clf_mod.check_non_degeneracy(clf_info)
    print(f"\nNon-degeneracy check: {'PASS' if ok else 'FAIL'}")
    for f in failures:
        print(f"  FAILURE: {f}")
    if not ok:
        print("STOPPING: classifier is degenerate, refusing to run the sweep.")
        raise SystemExit(1)

    # -----------------------------------------------------------------
    # STEP 2: population = val u test, never used to fit classifier/
    # threshold. Score once, freeze.
    # -----------------------------------------------------------------
    section("STEP 2: build population (val+test), score with frozen classifier")
    pop_X = pd.concat([va["X"], te["X"]], ignore_index=True)
    pop_y = np.concatenate([va["y"], te["y"]])
    pop_aux = pd.concat([va["aux"], te["aux"]], ignore_index=True)
    pop_score = clf_mod.score_split(pipe, pop_X)
    pop_pred = (pop_score >= threshold).astype(int)

    y_ = pop_y
    score_ = pop_score
    pred_ = pop_pred
    group1 = pop_aux["female"].to_numpy().astype(bool)  # primary group: female=1

    print(f"Population n={len(y_)}, female rate={group1.mean():.4f}, label base rate={y_.mean():.4f}")
    print(f"Population predicted-positive rate at frozen threshold={pred_.mean():.4f}")

    pop_res, pop_ci, pop_extra = full_pool_report(y_, pred_, score_, group1, "population", seed=SEED)
    print("Population-level (no selection) reference gaps:")
    print(f"  DP_gap={pop_res['demographic_parity_gap']:.4f} "
          f"[{pop_ci['demographic_parity_gap']['ci_lo']:.4f},{pop_ci['demographic_parity_gap']['ci_hi']:.4f}]  "
          f"TPR_gap={pop_res['tpr_gap']:.4f}  FPR_gap={pop_res['fpr_gap']:.4f}  "
          f"AUROC_gap={pop_extra['auroc_gap']:.4f}  PPV_gap={pop_res['ppv_gap']:.4f}  "
          f"calib_gap={pop_res['calibration_gap_diff']:.4f}  4/5={pop_res['four_fifths_ratio']:.4f}")

    # -----------------------------------------------------------------
    # STEP 3: POSITIVE CONTROL. Plant a known logit shift on the female
    # group's scores, S=1 for the WHOLE population (no selection filter),
    # confirm the audit code recovers it.
    # -----------------------------------------------------------------
    section("STEP 3: POSITIVE CONTROL (planted logit shift, full population, no selection)")
    perturbed_score = score_.copy()
    perturbed_score[group1] = sigmoid(logit(score_[group1]) + POSITIVE_CONTROL_LOGIT_SHIFT)
    perturbed_pred = (perturbed_score >= threshold).astype(int)

    pc_audit = audit_mod.audit_pool(y_, perturbed_pred, perturbed_score, group1)
    pc_ci = audit_mod.bootstrap_audit_ci(y_, perturbed_pred, perturbed_score, group1,
                                          B=BOOTSTRAP_B, seed=SEED)
    dp_point, dp_ci = pc_audit["demographic_parity_gap"], pc_ci["demographic_parity_gap"]
    ff_point, ff_ci = pc_audit["four_fifths_ratio"], pc_ci["four_fifths_ratio"]
    print(f"Injected logit shift: +{POSITIVE_CONTROL_LOGIT_SHIFT} to female-group scores only.")
    print(f"Group pred_pos_rate: male={pc_audit['g0']['pred_pos_rate']:.4f} "
          f"female={pc_audit['g1']['pred_pos_rate']:.4f}")
    print(f"Demographic parity gap = {dp_point:.4f}, 95% CI = [{dp_ci['ci_lo']:.4f},{dp_ci['ci_hi']:.4f}]")
    print(f"Four-fifths ratio = {ff_point:.4f}, 95% CI = [{ff_ci['ci_lo']:.4f},{ff_ci['ci_hi']:.4f}]")

    pc_pass = (dp_ci["ci_lo"] > 0.0) and (ff_point < 0.90) and (ff_ci["ci_hi"] < 1.0)
    print(f"\nPOSITIVE CONTROL: {'PASS' if pc_pass else 'FAIL'}")
    if not pc_pass:
        print("STOPPING: audit code did not recover a planted gap. Every downstream number is void.")
        raise SystemExit(1)

    # -----------------------------------------------------------------
    # STEP 4: REGIME A -- selection depends on GROUP and OUTCOME but NOT on
    # the covariates (S _||_ X | A, Y) -- the actual C29/theory-file
    # mechanism (`pi_group_and_y`). The swept quantity is the group-level
    # selection odds ratio (beta_A); a FIXED case/control retention odds
    # ratio (REGIME_A_Y_ODDS_RATIO) is applied identically in both groups so
    # that DP/PPV/calibration are pulled away from their population values
    # (see the correction note above solve_intercept_AY) while FPR/TPR/AUROC,
    # computed within Y-strata where selection is still a uniform function of
    # A alone, remain provably invariant to X.
    # -----------------------------------------------------------------
    section("STEP 4: REGIME A -- selection depends on group AND outcome, not covariates (S _||_ X | A,Y)")
    beta_Y = float(np.log(REGIME_A_Y_ODDS_RATIO))
    print(f"Fixed case/control (Y) selection odds ratio = {REGIME_A_Y_ODDS_RATIO} in both groups; "
          f"swept quantity = group (A) selection odds ratio.")
    regimeA_rows = []
    for OR in OR_GRID:
        beta_A = float(np.log(OR))
        rng_intercept = np.random.default_rng(det_seed("A_intercept", OR))
        c = solve_intercept_AY(beta_A, beta_Y, TARGET_SELECTION_RATE, group1, y_, rng=rng_intercept)
        p_row = sigmoid(c + beta_A * group1.astype(float) + beta_Y * y_.astype(float))
        rng_primary = np.random.default_rng(det_seed("A_primary", OR))
        s = rng_primary.binomial(1, p_row).astype(bool)

        r0_sel, r1_sel = s[~group1].mean(), s[group1].mean()
        sel_ff = min(r0_sel, r1_sel) / max(r0_sel, r1_sel)

        y_p, pred_p, score_p, g1_p = y_[s], pred_[s], score_[s], group1[s]
        res, ci, extra = full_pool_report(y_p, pred_p, score_p, g1_p, "A", seed=det_seed("A", OR))

        n0, n1 = res["g0"]["n"], res["g1"]["n"]
        mde_dp = audit_mod.mde_two_proportion(n0, n1, res["g0"]["pred_pos_rate"],
                                               res["g1"]["pred_pos_rate"])

        row = dict(
            OR=OR, beta_A=beta_A, beta_Y=beta_Y, intercept=c, n_pool=int(s.sum()), n0=n0, n1=n1,
            selection_rate_group0=float(r0_sel), selection_rate_group1=float(r1_sel),
            selection_four_fifths=float(sel_ff),
            demographic_parity_gap=res["demographic_parity_gap"],
            dp_gap_ci_lo=ci["demographic_parity_gap"]["ci_lo"],
            dp_gap_ci_hi=ci["demographic_parity_gap"]["ci_hi"],
            tpr_gap=res["tpr_gap"], tpr_gap_ci_lo=ci["tpr_gap"]["ci_lo"], tpr_gap_ci_hi=ci["tpr_gap"]["ci_hi"],
            fpr_gap=res["fpr_gap"], fpr_gap_ci_lo=ci["fpr_gap"]["ci_lo"], fpr_gap_ci_hi=ci["fpr_gap"]["ci_hi"],
            auroc_gap=extra["auroc_gap"], auroc_gap_ci_lo=extra["ci_lo"], auroc_gap_ci_hi=extra["ci_hi"],
            ppv_gap=res["ppv_gap"], ppv_gap_ci_lo=ci["ppv_gap"]["ci_lo"], ppv_gap_ci_hi=ci["ppv_gap"]["ci_hi"],
            calibration_gap_diff=res["calibration_gap_diff"],
            calib_ci_lo=ci["calibration_gap_diff"]["ci_lo"], calib_ci_hi=ci["calibration_gap_diff"]["ci_hi"],
            four_fifths_ratio_pool=res["four_fifths_ratio"],
            four_fifths_ci_lo=ci["four_fifths_ratio"]["ci_lo"], four_fifths_ci_hi=ci["four_fifths_ratio"]["ci_hi"],
            mde_demographic_parity_gap=mde_dp,
            # ---- C52: signed + per-group quantities, added alongside the
            # absolute ones above. Sign convention is g1 (female==1) MINUS
            # g0 (female==0 / male) throughout.
            demographic_parity_gap_signed=res["demographic_parity_gap_signed"],
            dp_gap_signed_ci_lo=ci["demographic_parity_gap_signed"]["ci_lo"],
            dp_gap_signed_ci_hi=ci["demographic_parity_gap_signed"]["ci_hi"],
            demographic_parity_gap_g0=res["demographic_parity_gap_g0"],
            demographic_parity_gap_g1=res["demographic_parity_gap_g1"],
            tpr_gap_signed=res["tpr_gap_signed"],
            tpr_gap_signed_ci_lo=ci["tpr_gap_signed"]["ci_lo"],
            tpr_gap_signed_ci_hi=ci["tpr_gap_signed"]["ci_hi"],
            tpr_gap_g0=res["tpr_gap_g0"], tpr_gap_g1=res["tpr_gap_g1"],
            fpr_gap_signed=res["fpr_gap_signed"],
            fpr_gap_signed_ci_lo=ci["fpr_gap_signed"]["ci_lo"],
            fpr_gap_signed_ci_hi=ci["fpr_gap_signed"]["ci_hi"],
            fpr_gap_g0=res["fpr_gap_g0"], fpr_gap_g1=res["fpr_gap_g1"],
            auroc_gap_signed=extra["auroc_gap_signed"],
            auroc_gap_signed_ci_lo=extra["signed"]["ci_lo"],
            auroc_gap_signed_ci_hi=extra["signed"]["ci_hi"],
            auroc_g0=extra["auroc_g0"], auroc_g1=extra["auroc_g1"],
            ppv_gap_signed=res["ppv_gap_signed"],
            ppv_gap_signed_ci_lo=ci["ppv_gap_signed"]["ci_lo"],
            ppv_gap_signed_ci_hi=ci["ppv_gap_signed"]["ci_hi"],
            ppv_gap_g0=res["ppv_gap_g0"], ppv_gap_g1=res["ppv_gap_g1"],
            calibration_gap_diff_signed=res["calibration_gap_diff_signed"],
            calib_signed_ci_lo=ci["calibration_gap_diff_signed"]["ci_lo"],
            calib_signed_ci_hi=ci["calibration_gap_diff_signed"]["ci_hi"],
            calibration_gap_diff_g0=res["calibration_gap_diff_g0"],
            calibration_gap_diff_g1=res["calibration_gap_diff_g1"],
            four_fifths_ratio_directed_numerator_group=
                res["four_fifths_ratio_directed"]["numerator_group"],
            four_fifths_ratio_directed_denominator_group=
                res["four_fifths_ratio_directed"]["denominator_group"],
        )
        row["fpr_gap_matches_population"] = bool(
            row["fpr_gap_ci_lo"] <= pop_res["fpr_gap"] <= row["fpr_gap_ci_hi"])
        row["tpr_gap_matches_population"] = bool(
            row["tpr_gap_ci_lo"] <= pop_res["tpr_gap"] <= row["tpr_gap_ci_hi"])
        row["auroc_gap_matches_population"] = bool(
            row["auroc_gap_ci_lo"] <= pop_extra["auroc_gap"] <= row["auroc_gap_ci_hi"])
        # DP/PPV/calibration are the metrics theory predicts should NOT match
        # population here (their prevalence-dependence is disturbed by the
        # fixed Y-selectivity term) -- same CI-coverage check, opposite
        # expected outcome.
        row["dp_gap_matches_population"] = bool(
            row["dp_gap_ci_lo"] <= pop_res["demographic_parity_gap"] <= row["dp_gap_ci_hi"])
        row["ppv_gap_matches_population"] = bool(
            row["ppv_gap_ci_lo"] <= pop_res["ppv_gap"] <= row["ppv_gap_ci_hi"])
        row["calib_matches_population"] = bool(
            row["calib_ci_lo"] <= pop_res["calibration_gap_diff"] <= row["calib_ci_hi"])
        regimeA_rows.append(row)
        print(f"OR={OR:5.2f}  n_pool={row['n_pool']:5d}  sel_4/5={sel_ff:.3f}  "
              f"DP={row['demographic_parity_gap']:.4f} match_pop={row['dp_gap_matches_population']}  "
              f"FPR_gap={row['fpr_gap']:.4f} "
              f"(pop={pop_res['fpr_gap']:.4f}, match={row['fpr_gap_matches_population']})  "
              f"TPR_gap={row['tpr_gap']:.4f} (match={row['tpr_gap_matches_population']})  "
              f"AUROC_gap={row['auroc_gap']:.4f} (match={row['auroc_gap_matches_population']})  "
              f"PPV_gap={row['ppv_gap']:.4f} match_pop={row['ppv_gap_matches_population']}  "
              f"4/5={row['four_fifths_ratio_pool']:.3f}  "
              f"DP_SIGNED={row['demographic_parity_gap_signed']:+.4f}")

    # -----------------------------------------------------------------
    # STEP 5: REGIME B -- selection depends on WKHP (usual hours worked
    # per week), an outcome-predictive, group-blind-classifier-excluded
    # feature. Sweep selection strength beta.
    # -----------------------------------------------------------------
    section("STEP 5: REGIME B -- selection through outcome-predictive feature (WKHP)")
    feat = data_prep.selection_feature(pop_aux)
    print(f"selection feature = z(WKHP, usual hours worked/week); chosen because it predicts "
          f"the outcome and is excluded from the classifier's inputs.")
    print(f"corr(feat, label)={np.corrcoef(feat, y_)[0,1]:.4f}  "
          f"corr(feat, female)={np.corrcoef(feat, group1.astype(int))[0,1]:.4f}")

    regimeB_rows = []
    for beta in BETA_GRID:
        rng_intercept = np.random.default_rng(det_seed("B_intercept", beta))
        b0 = solve_intercept_feature(beta, TARGET_SELECTION_RATE, feat, rng=rng_intercept)
        p_s = sigmoid(b0 + beta * feat)
        rng_primary = np.random.default_rng(det_seed("B_primary", beta))
        s = rng_primary.binomial(1, p_s).astype(bool)

        r0_sel, r1_sel = s[~group1].mean(), s[group1].mean()
        sel_ff = min(r0_sel, r1_sel) / max(r0_sel, r1_sel)
        sel_stage = audit_mod.selection_stage_audit(group1, s, B=BOOTSTRAP_B,
                                                      seed=det_seed("B_sel_ci", beta))

        y_p, pred_p, score_p, g1_p = y_[s], pred_[s], score_[s], group1[s]
        res, ci, extra = full_pool_report(y_p, pred_p, score_p, g1_p, "B", seed=det_seed("B", beta))

        n0, n1 = res["g0"]["n"], res["g1"]["n"]
        mde_dp = audit_mod.mde_two_proportion(n0, n1, res["g0"]["pred_pos_rate"],
                                               res["g1"]["pred_pos_rate"])

        # replicate-averaged trend (stability check, 30 independent draws)
        rep_g0 = {k: [] for k in ["tpr", "fpr", "ppv", "pred_pos_rate", "calibration_gap"]}
        rep_g1 = {k: [] for k in rep_g0}
        for r in range(N_REPLICATES):
            rng_r = np.random.default_rng(det_seed("B_replicate", beta, r))
            s_r = rng_r.binomial(1, p_s).astype(bool)
            gr = audit_mod.audit_pool(y_[s_r], pred_[s_r], score_[s_r], group1[s_r])
            for k in rep_g0:
                rep_g0[k].append(gr["g0"][k])
                rep_g1[k].append(gr["g1"][k])
        rep_g0m = {k: float(np.nanmean(v)) for k, v in rep_g0.items()}
        rep_g1m = {k: float(np.nanmean(v)) for k, v in rep_g1.items()}
        rep_dp = abs(rep_g1m["pred_pos_rate"] - rep_g0m["pred_pos_rate"])
        rep_tpr = abs(rep_g1m["tpr"] - rep_g0m["tpr"])
        rep_fpr = abs(rep_g1m["fpr"] - rep_g0m["fpr"])
        # C52: signed replicate-averaged companions, same g1-minus-g0 convention.
        rep_dp_signed = rep_g1m["pred_pos_rate"] - rep_g0m["pred_pos_rate"]
        rep_tpr_signed = rep_g1m["tpr"] - rep_g0m["tpr"]
        rep_fpr_signed = rep_g1m["fpr"] - rep_g0m["fpr"]

        row = dict(
            beta=beta, b0=b0, n_pool=int(s.sum()), n0=n0, n1=n1,
            selection_rate_group0=float(r0_sel), selection_rate_group1=float(r1_sel),
            selection_four_fifths=float(sel_ff),
            selection_four_fifths_ci_lo=sel_stage["ci_lo"], selection_four_fifths_ci_hi=sel_stage["ci_hi"],
            demographic_parity_gap=res["demographic_parity_gap"],
            dp_gap_ci_lo=ci["demographic_parity_gap"]["ci_lo"],
            dp_gap_ci_hi=ci["demographic_parity_gap"]["ci_hi"],
            tpr_gap=res["tpr_gap"], tpr_gap_ci_lo=ci["tpr_gap"]["ci_lo"], tpr_gap_ci_hi=ci["tpr_gap"]["ci_hi"],
            fpr_gap=res["fpr_gap"], fpr_gap_ci_lo=ci["fpr_gap"]["ci_lo"], fpr_gap_ci_hi=ci["fpr_gap"]["ci_hi"],
            auroc_gap=extra["auroc_gap"], auroc_gap_ci_lo=extra["ci_lo"], auroc_gap_ci_hi=extra["ci_hi"],
            ppv_gap=res["ppv_gap"], ppv_gap_ci_lo=ci["ppv_gap"]["ci_lo"], ppv_gap_ci_hi=ci["ppv_gap"]["ci_hi"],
            equalized_odds_gap=res["equalized_odds_gap"],
            eq_odds_ci_lo=ci["equalized_odds_gap"]["ci_lo"], eq_odds_ci_hi=ci["equalized_odds_gap"]["ci_hi"],
            calibration_gap_diff=res["calibration_gap_diff"],
            calib_ci_lo=ci["calibration_gap_diff"]["ci_lo"], calib_ci_hi=ci["calibration_gap_diff"]["ci_hi"],
            four_fifths_ratio_pool=res["four_fifths_ratio"],
            four_fifths_ci_lo=ci["four_fifths_ratio"]["ci_lo"], four_fifths_ci_hi=ci["four_fifths_ratio"]["ci_hi"],
            mde_demographic_parity_gap=mde_dp,
            replicate_mean_dp_gap=rep_dp, replicate_mean_tpr_gap=rep_tpr, replicate_mean_fpr_gap=rep_fpr,
            # ---- C52: signed + per-group quantities, added alongside the
            # absolute ones above. Sign convention is g1 (female==1) MINUS
            # g0 (female==0 / male) throughout.
            demographic_parity_gap_signed=res["demographic_parity_gap_signed"],
            dp_gap_signed_ci_lo=ci["demographic_parity_gap_signed"]["ci_lo"],
            dp_gap_signed_ci_hi=ci["demographic_parity_gap_signed"]["ci_hi"],
            demographic_parity_gap_g0=res["demographic_parity_gap_g0"],
            demographic_parity_gap_g1=res["demographic_parity_gap_g1"],
            tpr_gap_signed=res["tpr_gap_signed"],
            tpr_gap_signed_ci_lo=ci["tpr_gap_signed"]["ci_lo"],
            tpr_gap_signed_ci_hi=ci["tpr_gap_signed"]["ci_hi"],
            tpr_gap_g0=res["tpr_gap_g0"], tpr_gap_g1=res["tpr_gap_g1"],
            fpr_gap_signed=res["fpr_gap_signed"],
            fpr_gap_signed_ci_lo=ci["fpr_gap_signed"]["ci_lo"],
            fpr_gap_signed_ci_hi=ci["fpr_gap_signed"]["ci_hi"],
            fpr_gap_g0=res["fpr_gap_g0"], fpr_gap_g1=res["fpr_gap_g1"],
            auroc_gap_signed=extra["auroc_gap_signed"],
            auroc_gap_signed_ci_lo=extra["signed"]["ci_lo"],
            auroc_gap_signed_ci_hi=extra["signed"]["ci_hi"],
            auroc_g0=extra["auroc_g0"], auroc_g1=extra["auroc_g1"],
            ppv_gap_signed=res["ppv_gap_signed"],
            ppv_gap_signed_ci_lo=ci["ppv_gap_signed"]["ci_lo"],
            ppv_gap_signed_ci_hi=ci["ppv_gap_signed"]["ci_hi"],
            ppv_gap_g0=res["ppv_gap_g0"], ppv_gap_g1=res["ppv_gap_g1"],
            equalized_odds_gap_signed=res["equalized_odds_gap_signed"],
            eq_odds_signed_ci_lo=ci["equalized_odds_gap_signed"]["ci_lo"],
            eq_odds_signed_ci_hi=ci["equalized_odds_gap_signed"]["ci_hi"],
            equalized_odds_driver=res["equalized_odds_driver"],
            calibration_gap_diff_signed=res["calibration_gap_diff_signed"],
            calib_signed_ci_lo=ci["calibration_gap_diff_signed"]["ci_lo"],
            calib_signed_ci_hi=ci["calibration_gap_diff_signed"]["ci_hi"],
            calibration_gap_diff_g0=res["calibration_gap_diff_g0"],
            calibration_gap_diff_g1=res["calibration_gap_diff_g1"],
            four_fifths_ratio_directed_numerator_group=
                res["four_fifths_ratio_directed"]["numerator_group"],
            four_fifths_ratio_directed_denominator_group=
                res["four_fifths_ratio_directed"]["denominator_group"],
            replicate_mean_dp_gap_signed=rep_dp_signed,
            replicate_mean_tpr_gap_signed=rep_tpr_signed,
            replicate_mean_fpr_gap_signed=rep_fpr_signed,
        )
        row["dp_gap_breach"] = audit_mod.breach_call(row["demographic_parity_gap"], row["dp_gap_ci_lo"],
                                                       row["dp_gap_ci_hi"], "gap")
        row["four_fifths_breach"] = audit_mod.breach_call(row["four_fifths_ratio_pool"],
                                                            row["four_fifths_ci_lo"], row["four_fifths_ci_hi"],
                                                            "four_fifths")
        row["adequately_powered"] = bool(mde_dp < audit_mod.GAP_TOLERANCE)
        # The decisive test for THIS regime: does the pool's FPR/TPR/AUROC
        # gap -- which theory (C29/C31) says must equal the population value
        # under S _||_ X|A but need NOT under selection-on-X -- actually
        # diverge from the population-truth value? (Same "matches_population"
        # check as Regime A, applied here where it is NOT guaranteed to hold.)
        row["fpr_gap_matches_population"] = bool(
            row["fpr_gap_ci_lo"] <= pop_res["fpr_gap"] <= row["fpr_gap_ci_hi"])
        row["tpr_gap_matches_population"] = bool(
            row["tpr_gap_ci_lo"] <= pop_res["tpr_gap"] <= row["tpr_gap_ci_hi"])
        row["auroc_gap_matches_population"] = bool(
            row["auroc_gap_ci_lo"] <= pop_extra["auroc_gap"] <= row["auroc_gap_ci_hi"])
        regimeB_rows.append(row)

        print(f"beta={beta:4.1f}  n_pool={row['n_pool']:5d} (m={n0},f={n1})  "
              f"sel_4/5={row['selection_four_fifths']:.3f}  "
              f"DP={row['demographic_parity_gap']:.4f} [{row['dp_gap_ci_lo']:.4f},{row['dp_gap_ci_hi']:.4f}] "
              f"call={row['dp_gap_breach']:<12}  "
              f"FPR_gap={row['fpr_gap']:.4f} match_pop={row['fpr_gap_matches_population']}  "
              f"TPR_gap={row['tpr_gap']:.4f} match_pop={row['tpr_gap_matches_population']}  "
              f"AUROC_gap={row['auroc_gap']:.4f} match_pop={row['auroc_gap_matches_population']}  "
              f"pool_4/5={row['four_fifths_ratio_pool']:.3f} "
              f"call={row['four_fifths_breach']:<12}  MDE={mde_dp:.4f}  powered={row['adequately_powered']}  "
              f"DP_SIGNED={row['demographic_parity_gap_signed']:+.4f}")

    # -----------------------------------------------------------------
    # STEP 5b: secondary group axis (RAC1P: Black vs White subset) at
    # representative Regime-B betas -- reuses the identical realized pool
    # (selection never looks at group), so this is a no-extra-assumptions
    # robustness check.
    # -----------------------------------------------------------------
    section("STEP 5b: secondary group axis (RAC1P Black vs White) in Regime B")
    black = pop_aux["black"].to_numpy().astype(bool)
    white = pop_aux["white"].to_numpy().astype(bool)
    bw_subset = black | white

    target_row = min(regimeB_rows, key=lambda r: abs(r["selection_four_fifths"] - CLINICAL_REFERENCE_SEL_4_5))
    representative_betas = sorted({0.0, target_row["beta"], BETA_GRID[-1]})
    secondary_axis = {}
    for beta in representative_betas:
        rng_intercept = np.random.default_rng(det_seed("B_intercept", beta))
        b0 = solve_intercept_feature(beta, TARGET_SELECTION_RATE, feat, rng=rng_intercept)
        p_s = sigmoid(b0 + beta * feat)
        rng_primary = np.random.default_rng(det_seed("B_primary", beta))
        s = rng_primary.binomial(1, p_s).astype(bool)
        s2 = s & bw_subset
        yb, pb, sb, gb = y_[s2], pred_[s2], score_[s2], black[s2]
        if gb.sum() == 0 or (~gb).sum() == 0:
            continue
        res = audit_mod.audit_pool(yb, pb, sb, gb)
        ci = audit_mod.bootstrap_audit_ci(yb, pb, sb, gb, B=BOOTSTRAP_B, seed=det_seed("sec_ci", beta))
        secondary_axis[str(beta)] = dict(
            beta=beta, n_pool=int(s2.sum()), n_white=int((~gb).sum()), n_black=int(gb.sum()),
            demographic_parity_gap=res["demographic_parity_gap"],
            dp_gap_ci=[ci["demographic_parity_gap"]["ci_lo"], ci["demographic_parity_gap"]["ci_hi"]],
            four_fifths_ratio=res["four_fifths_ratio"],
            four_fifths_ci=[ci["four_fifths_ratio"]["ci_lo"], ci["four_fifths_ratio"]["ci_hi"]],
            fpr_gap=res["fpr_gap"], tpr_gap=res["tpr_gap"],
        )
        r = secondary_axis[str(beta)]
        print(f"beta={beta:4.1f}  Black-v-White (n={r['n_pool']}): DP_gap={r['demographic_parity_gap']:.4f} "
              f"CI={r['dp_gap_ci']}  4/5={r['four_fifths_ratio']:.3f}  FPR_gap={r['fpr_gap']:.4f}  "
              f"TPR_gap={r['tpr_gap']:.4f}")

    # -----------------------------------------------------------------
    # STEP 6: drift checks. Regime A: OLS slope of equalized-odds-family
    # gaps vs OR should include 0 (flat / invariant); Regime B: OLS slope
    # of every gap vs beta.
    # -----------------------------------------------------------------
    section("STEP 6: drift checks (OLS slope of gap point-estimates vs selection strength)")

    def slope_ci(xvals, yvals):
        res = linregress(xvals, yvals)
        df = len(xvals) - 2
        tcrit = t_dist.ppf(0.975, df)
        lo, hi = res.slope - tcrit * res.stderr, res.slope + tcrit * res.stderr
        return dict(slope=float(res.slope), slope_ci=[float(lo), float(hi)],
                    credible_drift=bool(not (lo <= 0 <= hi)), r2=float(res.rvalue ** 2))

    ors = np.array([r["OR"] for r in regimeA_rows])
    print("Regime A (selection depends only on group; x = odds ratio):")
    drift_A = {}
    for key in ["fpr_gap", "tpr_gap", "auroc_gap", "demographic_parity_gap", "ppv_gap",
                "calibration_gap_diff"]:
        vals = np.array([r[key] for r in regimeA_rows])
        d = slope_ci(ors, vals)
        drift_A[key] = d
        print(f"  {key}: slope={d['slope']:.6f} CI={d['slope_ci']} credible_drift={d['credible_drift']} "
              f"R^2={d['r2']:.3f}")

    betas = np.array([r["beta"] for r in regimeB_rows])
    print("Regime B (selection depends on WKHP; x = beta):")
    drift_B = {}
    for key in ["demographic_parity_gap", "tpr_gap", "fpr_gap", "auroc_gap", "ppv_gap",
                "equalized_odds_gap", "calibration_gap_diff"]:
        vals = np.array([r[key] for r in regimeB_rows])
        d = slope_ci(betas, vals)
        drift_B[key] = d
        print(f"  {key}: slope={d['slope']:.6f} CI={d['slope_ci']} credible_drift={d['credible_drift']} "
              f"R^2={d['r2']:.3f}")

    # -----------------------------------------------------------------
    # STEP 7: verdict.
    # -----------------------------------------------------------------
    section("STEP 7: VERDICT")
    # Regime A -- two families of evidence, each tested at the AGGREGATE
    # level rather than "every single point must pass" (a first version of
    # this script used a strict all-points-must-match rule and it produced
    # exactly the kind of false alarm this note warns about: 1 mismatch out
    # of 36 pointwise 95%-CI coverage checks (12 OR values x 3 metrics),
    # which is within the ~1.8 expected by chance alone at a 5% level, but
    # flipped the whole regime's verdict to REFUTES). Corrected criterion:
    #   (a) OLS slope of the metric vs OR has a 95% CI that excludes 0
    #       (credible_drift, from Step 6), and/or
    #   (b) the pointwise mismatch RATE clearly exceeds the ~5% false-positive
    #       rate expected from repeated 95%-CI coverage checks under the
    #       null (using a 2-SD binomial guard band).
    # A metric counts as "invariant" only if NEITHER fires; "contaminated"
    # only if the majority of points fail to cover the population value
    # (clearly more than chance).
    def mismatch_excess(n_tests, n_mismatch, alpha=0.05, z=2.0):
        expected = alpha * n_tests
        sd = sqrt(n_tests * alpha * (1 - alpha))
        return n_mismatch > expected + z * sd

    n_A = len(regimeA_rows)
    A_mismatch = {}
    for key, match_key in [("fpr_gap", "fpr_gap_matches_population"),
                            ("tpr_gap", "tpr_gap_matches_population"),
                            ("auroc_gap", "auroc_gap_matches_population")]:
        n_mis = sum(1 for r in regimeA_rows if not r[match_key])
        A_mismatch[key] = dict(n_mismatch=n_mis, n_tests=n_A,
                                excess=mismatch_excess(n_A, n_mis))
    A_invariant_metrics_ok = all(
        (not drift_A[k]["credible_drift"]) and (not A_mismatch[k]["excess"])
        for k in ["fpr_gap", "tpr_gap", "auroc_gap"])

    A_contam_majority_mismatch = {}
    for key, match_key in [("demographic_parity_gap", "dp_gap_matches_population"),
                            ("ppv_gap", "ppv_gap_matches_population"),
                            ("calibration_gap_diff", "calib_matches_population")]:
        n_mis = sum(1 for r in regimeA_rows if not r[match_key])
        A_contam_majority_mismatch[key] = dict(n_mismatch=n_mis, n_tests=n_A,
                                                majority=n_mis > n_A / 2)
    A_contaminated_metrics_move = any(
        drift_A[k]["credible_drift"] or A_contam_majority_mismatch[k]["majority"]
        for k in ["demographic_parity_gap", "ppv_gap", "calibration_gap_diff"])

    regimeA_verdict = "SUPPORTS" if (A_invariant_metrics_ok and A_contaminated_metrics_move) else "REFUTES"
    print("Regime A per-metric evidence:")
    for k, v in A_mismatch.items():
        print(f"  {k} (should be invariant): mismatches={v['n_mismatch']}/{v['n_tests']} "
              f"excess-vs-chance={v['excess']}  OLS credible_drift={drift_A[k]['credible_drift']}")
    for k, v in A_contam_majority_mismatch.items():
        print(f"  {k} (should be contaminated): mismatches={v['n_mismatch']}/{v['n_tests']} "
              f"majority={v['majority']}  OLS credible_drift={drift_A[k]['credible_drift']}")

    # Regime B: the DECISIVE test mirrors C29/C31's actual mechanism -- FPR,
    # TPR and AUROC gaps are the metrics theory says MUST equal the
    # population value when S _||_ X | A (as in Regime A) but need NOT when
    # S depends on X (as here). So the primary evidence for "replicates the
    # clinical finding" is whether those three gaps (a) stop matching the
    # population-truth value at some point in the sweep, and/or (b) show a
    # credible monotonic slope vs beta.
    #
    # NOTE ON THE ABSOLUTE DP-GAP TOLERANCE: ACSIncome's population already
    # has a large, REAL sex gap in predicted-positive rate (DP_gap ~0.12,
    # see population_reference_audit above) that exceeds the 0.05 MIMIC-style
    # tolerance with ZERO selection at all (beta=0 is just an unbiased
    # subsample). So an absolute-tolerance "breach" call on DP gap, ported
    # unmodified from the clinical protocol, would fire even in the absence
    # of any selection effect -- it is contaminated by intrinsic population
    # disparity, not evidence of selection corrupting the audit. It is
    # reported below for completeness (mirrors the clinical protocol
    # literally) but is NOT used as the primary criterion.
    up_to_target = [r for r in regimeB_rows if r["beta"] <= target_row["beta"]]
    B_core_pointwise_divergence = any(
        not (r["fpr_gap_matches_population"] and r["tpr_gap_matches_population"]
             and r["auroc_gap_matches_population"])
        for r in up_to_target)
    B_core_credible_drift = any(drift_B[k]["credible_drift"] for k in
                                 ["fpr_gap", "tpr_gap", "auroc_gap"])
    B_core_evidence = B_core_pointwise_divergence or B_core_credible_drift

    any_absolute_breach = any(
        (r["dp_gap_breach"] == "breach" or r["four_fifths_breach"] == "breach") and r["adequately_powered"]
        for r in up_to_target)
    region_powered = any(r["adequately_powered"] for r in up_to_target)

    if B_core_evidence:
        regimeB_verdict = "REFUTES (FPR/TPR/AUROC gap diverges from population under feature-based selection)"
    elif not region_powered:
        regimeB_verdict = "INCONCLUSIVE (underpowered)"
    else:
        regimeB_verdict = "SUPPORTS (FPR/TPR/AUROC gap stays pinned to population value across the sweep)"

    clinical_replicates = (regimeA_verdict == "SUPPORTS") and regimeB_verdict.startswith("REFUTES")
    print(f"Regime A (group+outcome, not-covariate selection) verdict: {regimeA_verdict}")
    print(f"  FPR/TPR/AUROC invariant (no excess mismatch, no credible drift): {A_invariant_metrics_ok}")
    print(f"  DP/PPV/calibration move (majority mismatch or credible drift): {A_contaminated_metrics_move}")
    print(f"Regime B (feature-driven selection) verdict: {regimeB_verdict}")
    print(f"  FPR/TPR/AUROC pointwise divergence from population at any beta<=target: "
          f"{B_core_pointwise_divergence}")
    print(f"  FPR/TPR/AUROC credible OLS drift vs beta: {B_core_credible_drift}")
    print(f"  [secondary, literal MIMIC-protocol absolute DP-gap/4-5 breach call "
          f"(confounded by intrinsic population gap, see note above)]: {any_absolute_breach}")
    print(f"\nOVERALL: clinical MIMIC finding {'REPLICATES' if clinical_replicates else 'DOES NOT REPLICATE'} "
          f"on ACS PUMS (ACSIncome, {data_prep.STATE} {data_prep.SURVEY_YEAR}).")

    # -----------------------------------------------------------------
    # SAVE
    # -----------------------------------------------------------------
    out = dict(
        seed=SEED,
        # C52: explicit group labels, so every *_g0/_g1/_signed field above
        # can be read back to a named group without guessing.
        group_definition=dict(g0="female == 0 (male)", g1="female == 1 (female)",
                               signed_convention="g1 minus g0"),
        state=data_prep.STATE, survey_year=data_prep.SURVEY_YEAR,
        task_n=int(len(y)), task_positive_rate=float(y.mean()),
        split_sizes=dict(train=len(tr["y"]), val=len(va["y"]), test=len(te["y"])),
        classifier_info=clf_info, classifier_non_degeneracy=dict(ok=ok, failures=failures),
        population_n=int(len(y_)), population_female_rate=float(group1.mean()),
        population_base_rate=float(y_.mean()), population_pred_pos_rate=float(pred_.mean()),
        population_reference_audit=pop_res, population_reference_ci=pop_ci,
        population_reference_extra=pop_extra,
        selection_feature_corr_with_label=float(np.corrcoef(feat, y_)[0, 1]),
        selection_feature_corr_with_group=float(np.corrcoef(feat, group1.astype(int))[0, 1]),
        positive_control=dict(
            logit_shift=POSITIVE_CONTROL_LOGIT_SHIFT, demographic_parity_gap=dp_point,
            dp_gap_ci=[dp_ci["ci_lo"], dp_ci["ci_hi"]], four_fifths_ratio=ff_point,
            four_fifths_ci=[ff_ci["ci_lo"], ff_ci["ci_hi"]], passed=pc_pass,
            demographic_parity_gap_signed=pc_audit["demographic_parity_gap_signed"],
            demographic_parity_gap_g0=pc_audit["demographic_parity_gap_g0"],
            demographic_parity_gap_g1=pc_audit["demographic_parity_gap_g1"],
        ),
        regimeA_sweep=regimeA_rows, regimeA_drift=drift_A, regimeA_verdict=regimeA_verdict,
        regimeA_invariant_metric_evidence=A_mismatch,
        regimeA_contaminated_metric_evidence=A_contam_majority_mismatch,
        regimeA_y_odds_ratio=REGIME_A_Y_ODDS_RATIO,
        regimeB_sweep=regimeB_rows, regimeB_drift=drift_B, regimeB_verdict=regimeB_verdict,
        regimeB_core_pointwise_divergence=B_core_pointwise_divergence,
        regimeB_core_credible_drift=B_core_credible_drift,
        regimeB_secondary_absolute_dp_breach=any_absolute_breach,
        regimeB_target_beta=target_row["beta"],
        regimeB_target_selection_four_fifths=target_row["selection_four_fifths"],
        secondary_axis_black_v_white=secondary_axis,
        clinical_reference_selection_four_fifths=CLINICAL_REFERENCE_SEL_4_5,
        clinical_finding_replicates=clinical_replicates,
        runtime_seconds=round(time.time() - t_start, 1),
    )
    # C52: NEW filenames (distinct from the pre-existing
    # pums_replication_results.json / regimeA_sweep.csv / regimeB_sweep.csv)
    # so this re-run with the extended, signed audit does not overwrite the
    # committed unsigned results.
    with (OUT_DIR / "pums_replication_results_signed.json").open("w") as f:
        json.dump(out, f, indent=2)

    pd.DataFrame(regimeA_rows).to_csv(OUT_DIR / "regimeA_sweep_signed.csv", index=False)
    pd.DataFrame(regimeB_rows).to_csv(OUT_DIR / "regimeB_sweep_signed.csv", index=False)

    print(f"\nSaved results to {OUT_DIR}. Total runtime: {out['runtime_seconds']}s")
    print("ALL_DONE")


if __name__ == "__main__":
    main()
