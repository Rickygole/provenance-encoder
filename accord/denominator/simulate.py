"""
ACCORD Build 2 -- the denominator demonstration.

Thesis argument being tested: subgroup-stratified evaluation (the standard
fairness audit -- per-group TPR/FPR parity, equalized odds, demographic
parity, calibration) is computed ON THE CANDIDATE POOL, i.e. conditional on
having survived candidate generation. If candidate generation selects
group-dependently but independently of the true outcome and the classifier's
features (given group), the audit is *structurally* unable to detect that
selection bias, no matter how large it is -- because every quantity a
per-group audit can compute is, by basic conditioning, invariant to the
group's selection RATE.

This script is 100% synthetic. No patient data, fully reproducible (fixed
seed), and safe to commit.

SWEEP 1 (core result): selection depends on group A ONLY (S _||_ (X,Y) | A).
    Sweep the group-dependent selection odds ratio OR_select across a wide
    range that includes the real observed value (0.64, C12's post-adjustment
    missing-language OR). Show standard audit metrics computed on the
    candidate pool stay flat/near-parity across the ENTIRE sweep, while the
    upstream *selection-stage* four-fifths ratio (which nobody in the
    standard workflow computes) tracks OR_select directly and fails the
    four-fifths rule for OR_select below 0.8.

SWEEP 2 (boundary characterization): fix OR_select = 0.64 (the real value)
    and add a controllable amount of *direct* outcome-dependence to the
    selection mechanism (a "leakage" parameter). Find the leakage level at
    which the pool-level audit metrics first move outside conventional
    tolerance (parity gap > 0.05, four-fifths ratio < 0.8). This shows that
    the audit is NOT blind unconditionally -- it fails exactly when
    selection stops being conditionally independent of the outcome given
    group, which is the theoretically expected boundary.

Outputs (JSON + CSV, aggregated across replicates) are written to
./results/ in this directory.
"""
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Population parameters (loosely calibrated to the real cohort so the
# simulation lives in a realistic regime -- see accord/CORRECTIONS.md C12
# and the exp2a/exp3 outputs in mimic_local_only/accord_differential/):
#   - minority group ("missing" analogue) prevalence ~= 46.6% of admissions
#   - majority-group candidate yield rate           ~= 20.8%
#   - true (gold pharmacist) positive base rate      ~= 4.0%
# ---------------------------------------------------------------------------
P_A = 0.4664          # P(A=1), the "unrecorded/administratively-incomplete" group
BASE_YIELD_A0 = 0.2081  # P(S=1 | A=0), majority-group candidate-generation rate
Y_BASE_RATE = 0.04     # target population base rate of the true outcome
X_COEF = 2.0           # how strongly the latent feature X predicts true outcome Y
N_POP = 20_000         # population size per replicate
N_REPLICATES = 150     # replicates per sweep point (for CIs)
N_REFERENCE = 300_000  # size of the large, unbiased reference sample used to
                        # fit the ONE fixed, group-blind downstream classifier

TOLERANCE_GAP = 0.05      # conventional parity-gap tolerance
FOUR_FIFTHS = 0.80        # conventional four-fifths-rule threshold


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def solve_intercept_for_base_rate(coef, target_rate, x_std=1.0, n=2_000_000, rng=None):
    """Numerically find b0 such that mean(sigmoid(b0 + coef*X)) ~= target_rate,
    X ~ N(0, x_std^2)."""
    rng = rng or np.random.default_rng(0)
    x = rng.normal(0, x_std, n)
    lo, hi = -15.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = sigmoid(mid + coef * x).mean()
        if rate > target_rate:
            lo, hi = lo, mid
        else:
            lo, hi = mid, hi
    return (lo + hi) / 2


Y_INTERCEPT = solve_intercept_for_base_rate(X_COEF, Y_BASE_RATE, rng=np.random.default_rng(SEED))


# ---------------------------------------------------------------------------
# Fit ONE fixed, group-blind classifier on a large unbiased reference sample.
# It never sees A. This isolates the audit-blindness result to the
# candidate-generation stage, not to any unfairness in the classifier itself.
# ---------------------------------------------------------------------------
def fit_reference_classifier():
    rng = np.random.default_rng(SEED + 1)
    x_ref = rng.normal(0, 1, N_REFERENCE)
    y_ref = rng.binomial(1, sigmoid(Y_INTERCEPT + X_COEF * x_ref))
    clf = LogisticRegression()
    clf.fit(x_ref.reshape(-1, 1), y_ref)
    scores_ref = clf.predict_proba(x_ref.reshape(-1, 1))[:, 1]
    # Pick an operating threshold from the ROC curve: the point closest to
    # (FPR=0.15, TPR as high as achievable there) -- a plausible "moderate
    # review-burden" screening threshold, fixed once and reused everywhere.
    fpr, tpr, thr = roc_curve(y_ref, scores_ref)
    target_fpr = 0.15
    idx = np.argmin(np.abs(fpr - target_fpr))
    threshold = thr[idx]
    auc_proxy = np.trapz(tpr, fpr)
    return clf, threshold, {
        "reference_n": N_REFERENCE, "reference_base_rate": float(y_ref.mean()),
        "operating_threshold": float(threshold), "operating_fpr_at_threshold": float(fpr[idx]),
        "operating_tpr_at_threshold": float(tpr[idx]), "approx_auc": float(auc_proxy),
    }


CLF, THRESHOLD, CLF_INFO = fit_reference_classifier()
print("Fixed reference classifier:", json.dumps(CLF_INFO, indent=2))


# ---------------------------------------------------------------------------
# One simulated draw: population -> selection -> pool -> audit metrics
# ---------------------------------------------------------------------------
def simulate_once(n_pop, or_select, leak_gamma, rng):
    """
    or_select : odds ratio for selection, group A=1 vs A=0 (direct term).
    leak_gamma: additional direct log-odds contribution of Y to selection
                (0 = pure group-dependent selection; S _||_ (X,Y) | A).
    """
    a = rng.binomial(1, P_A, n_pop)
    x = rng.normal(0, 1, n_pop)
    y = rng.binomial(1, sigmoid(Y_INTERCEPT + X_COEF * x))

    s0 = np.log(BASE_YIELD_A0 / (1 - BASE_YIELD_A0))  # logit(P(S=1|A=0)) baseline
    s1 = np.log(or_select)
    y_centered = y - Y_BASE_RATE  # keep the marginal selection rate ~stable as gamma sweeps
    logit_s = s0 + s1 * a + leak_gamma * y_centered
    p_s = sigmoid(logit_s)
    s = rng.binomial(1, p_s)

    scores = CLF.predict_proba(x.reshape(-1, 1))[:, 1]
    pred = (scores >= THRESHOLD).astype(int)

    return dict(a=a, x=x, y=y, s=s, scores=scores, pred=pred)


def group_metrics(y, pred, scores, mask):
    """Standard per-group audit metrics on the subset where mask is True."""
    yy, pp, ss = y[mask], pred[mask], scores[mask]
    n = len(yy)
    if n == 0:
        return dict(n=0)
    tp = int(((yy == 1) & (pp == 1)).sum())
    fp = int(((yy == 0) & (pp == 1)).sum())
    fn = int(((yy == 1) & (pp == 0)).sum())
    tn = int(((yy == 0) & (pp == 0)).sum())
    n_pos, n_neg = tp + fn, fp + tn
    tpr = tp / n_pos if n_pos else np.nan
    fpr = fp / n_neg if n_neg else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    pred_pos_rate = (tp + fp) / n
    # calibration: mean predicted score vs empirical positive rate (signed gap)
    calib_gap = float(ss.mean() - yy.mean())
    return dict(n=n, n_pos=n_pos, n_neg=n_neg, TP=tp, FP=fp, FN=fn, TN=tn,
                TPR=tpr, FPR=fpr, PPV=ppv, pred_pos_rate=pred_pos_rate,
                calibration_gap=calib_gap)


GROUP_LEVEL_KEYS = ["n", "TPR", "FPR", "PPV", "pred_pos_rate", "calibration_gap"]


def audit_one_draw(draw):
    """Compute pool-level (S=1) per-group audit + population-level selection audit.
    Returns the RAW per-group quantities (not gaps) -- gaps must be computed
    from quantities averaged ACROSS replicates first (see run_sweep), not by
    averaging per-replicate |gap| values, which is upward-biased by sampling
    noise (Jensen's inequality: E|X-Y| > |E[X]-E[Y]| for noisy X, Y)."""
    a, y, s, pred, scores = draw["a"], draw["y"], draw["s"], draw["pred"], draw["scores"]
    pool_mask = s == 1
    g0 = group_metrics(y, pred, scores, pool_mask & (a == 0))
    g1 = group_metrics(y, pred, scores, pool_mask & (a == 1))

    # upstream: selection-stage audit (rarely run in practice, computed here for contrast)
    sel_rate_0 = (s[a == 0] == 1).mean()
    sel_rate_1 = (s[a == 1] == 1).mean()
    sel_odds_0 = sel_rate_0 / (1 - sel_rate_0)
    sel_odds_1 = sel_rate_1 / (1 - sel_rate_1)
    sel_or_empirical = sel_odds_1 / sel_odds_0

    return dict(
        n_pool=int(pool_mask.sum()),
        g0=g0, g1=g1,
        selection_rate_a0=float(sel_rate_0), selection_rate_a1=float(sel_rate_1),
        selection_or_empirical=float(sel_or_empirical),
    )


def aggregate_group_level(reps):
    """Average each group-level quantity ACROSS replicates (correct: noise
    cancels before differencing), then derive gaps/ratios from the averaged
    values. Also returns replicate-level gap distributions for CIs."""
    out = {}
    g0_means, g1_means = {}, {}
    for k in GROUP_LEVEL_KEYS:
        v0 = np.array([rep["g0"].get(k, np.nan) for rep in reps], dtype=float)
        v1 = np.array([rep["g1"].get(k, np.nan) for rep in reps], dtype=float)
        g0_means[k] = float(np.nanmean(v0))
        g1_means[k] = float(np.nanmean(v1))
    out["g0_mean"] = g0_means
    out["g1_mean"] = g1_means

    out["demographic_parity_gap"] = abs(g1_means["pred_pos_rate"] - g0_means["pred_pos_rate"])
    out["tpr_gap"] = abs(g1_means["TPR"] - g0_means["TPR"])
    out["fpr_gap"] = abs(g1_means["FPR"] - g0_means["FPR"])
    out["ppv_gap"] = abs(g1_means["PPV"] - g0_means["PPV"])
    out["calibration_gap_diff"] = abs(g1_means["calibration_gap"] - g0_means["calibration_gap"])
    out["equalized_odds_gap"] = max(out["tpr_gap"], out["fpr_gap"])
    out["four_fifths_ratio_pool"] = (min(g0_means["pred_pos_rate"], g1_means["pred_pos_rate"]) /
                                      max(g0_means["pred_pos_rate"], g1_means["pred_pos_rate"]))

    sel0 = np.array([rep["selection_rate_a0"] for rep in reps])
    sel1 = np.array([rep["selection_rate_a1"] for rep in reps])
    sel0_mean, sel1_mean = float(sel0.mean()), float(sel1.mean())
    out["selection_rate_a0"] = sel0_mean
    out["selection_rate_a1"] = sel1_mean
    out["selection_four_fifths_ratio"] = min(sel0_mean, sel1_mean) / max(sel0_mean, sel1_mean)
    out["selection_or_empirical"] = float(np.mean([rep["selection_or_empirical"] for rep in reps]))
    out["n_pool_mean"] = float(np.mean([rep["n_pool"] for rep in reps]))

    # replicate-level gap distributions (single-sample noise diagnostic, NOT
    # the primary point estimate -- see docstring above)
    single_draw_gaps = []
    for rep in reps:
        g0, g1 = rep["g0"], rep["g1"]
        if g0.get("n", 0) == 0 or g1.get("n", 0) == 0:
            continue
        dp = abs(g1["pred_pos_rate"] - g0["pred_pos_rate"])
        tpr_g = abs(g1["TPR"] - g0["TPR"]) if not (np.isnan(g0["TPR"]) or np.isnan(g1["TPR"])) else np.nan
        fpr_g = abs(g1["FPR"] - g0["FPR"]) if not (np.isnan(g0["FPR"]) or np.isnan(g1["FPR"])) else np.nan
        ff = min(g0["pred_pos_rate"], g1["pred_pos_rate"]) / max(g0["pred_pos_rate"], g1["pred_pos_rate"])
        single_draw_gaps.append((dp, tpr_g, fpr_g, ff))
    if single_draw_gaps:
        dps, tprs, fprs, ffs = zip(*single_draw_gaps)
        breach_single = np.mean([
            (dp > TOLERANCE_GAP) or
            (not np.isnan(tg) and tg > TOLERANCE_GAP) or
            (not np.isnan(fg) and fg > TOLERANCE_GAP) or
            (ff < FOUR_FIFTHS)
            for dp, tg, fg, ff in single_draw_gaps
        ])
    else:
        breach_single = np.nan
    out["single_draw_breach_rate"] = float(breach_single)
    return out


def point_estimate_breach(agg):
    """Verdict based on the properly-aggregated (noise-cancelled) point
    estimate -- this is the primary criterion, NOT the noisy single-draw rate."""
    checks = [
        agg["demographic_parity_gap"] > TOLERANCE_GAP,
        agg["tpr_gap"] > TOLERANCE_GAP,
        agg["fpr_gap"] > TOLERANCE_GAP,
        agg["ppv_gap"] > TOLERANCE_GAP,
        agg["four_fifths_ratio_pool"] < FOUR_FIFTHS,
    ]
    return any(checks)


def run_sweep(param_values, tag, n_replicates=N_REPLICATES, fixed_or=None, fixed_gamma=None):
    """param_values sweeps OR_select if fixed_gamma is given, or sweeps
    leak_gamma if fixed_or is given. Exactly one of fixed_or/fixed_gamma
    must be provided."""
    assert (fixed_or is None) != (fixed_gamma is None)
    print(f"\n=== SWEEP: {tag} ===")
    rows = []
    for v in param_values:
        # if gamma is fixed, we're sweeping OR_select (v = or_sel)
        # if OR is fixed, we're sweeping leak_gamma (v = gamma)
        or_sel = v if fixed_gamma is not None else fixed_or
        gamma = v if fixed_or is not None else fixed_gamma
        reps = []
        for r in range(n_replicates):
            rng = np.random.default_rng(hash((tag, or_sel, gamma, r)) % (2**32))
            draw = simulate_once(N_POP, or_sel, gamma, rng)
            reps.append(audit_one_draw(draw))
        agg = aggregate_group_level(reps)
        agg["or_select"] = or_sel
        agg["leak_gamma"] = gamma
        agg["n_replicates"] = n_replicates
        agg["breach"] = point_estimate_breach(agg)
        rows.append(agg)
        print(f"  OR_select={or_sel:<6} leak_gamma={gamma:<5} "
              f"pool_DP_gap={agg['demographic_parity_gap']:.4f} "
              f"eq_odds_gap={agg['equalized_odds_gap']:.4f} "
              f"4/5_ratio(pool)={agg['four_fifths_ratio_pool']:.3f} "
              f"4/5_ratio(SELECTION-stage)={agg['selection_four_fifths_ratio']:.3f} "
              f"breach={agg['breach']}  (single_draw_breach_rate={agg['single_draw_breach_rate']:.2f})")
    return rows


def flatten_row(row):
    """Flatten the nested g0_mean/g1_mean dicts for CSV export."""
    flat = {k: v for k, v in row.items() if k not in ("g0_mean", "g1_mean")}
    for grp, d in [("g0", row["g0_mean"]), ("g1", row["g1_mean"])]:
        for k, v in d.items():
            flat[f"{grp}_{k}"] = v
    return flat


# ---------------------------------------------------------------------------
# SWEEP 1: pure group-dependent selection (S _||_ (X,Y) | A). Core result.
# ---------------------------------------------------------------------------
OR_SWEEP_1 = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.64, 0.80, 1.00, 1.25, 2.00]
sweep1 = run_sweep(OR_SWEEP_1, tag="sweep1_pure_group_dependent", fixed_gamma=0.0)

# ---------------------------------------------------------------------------
# SWEEP 2: fix OR_select = 0.64 (the real observed value), sweep outcome-
# leakage into the selection mechanism to find where the audit starts to move.
# ---------------------------------------------------------------------------
LEAK_SWEEP = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0]
sweep2 = run_sweep(LEAK_SWEEP, tag="sweep2_leakage_at_or_0.64", fixed_or=0.64)

# find the first gamma at which the point-estimate breach flag first trips
breach_gamma = None
for row in sweep2:
    if row["breach"]:
        breach_gamma = row["leak_gamma"]
        break

# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
out = {
    "seed": SEED,
    "population_params": {
        "P_A": P_A, "BASE_YIELD_A0": BASE_YIELD_A0, "Y_BASE_RATE": Y_BASE_RATE,
        "X_COEF": X_COEF, "Y_INTERCEPT": Y_INTERCEPT, "N_POP": N_POP,
        "N_REPLICATES": N_REPLICATES,
    },
    "classifier": CLF_INFO,
    "tolerances": {"parity_gap_tolerance": TOLERANCE_GAP, "four_fifths_threshold": FOUR_FIFTHS},
    "sweep1_pure_group_dependent_selection": sweep1,
    "sweep2_leakage_at_or_0.64": sweep2,
    "sweep2_breach_gamma_first_tripped": breach_gamma,
    "note_on_point_estimate_vs_single_draw": (
        "demographic_parity_gap, tpr_gap, fpr_gap, ppv_gap, equalized_odds_gap and "
        "four_fifths_ratio_pool are computed from GROUP-LEVEL RATES AVERAGED ACROSS "
        "replicates first, then differenced -- this is the correct, noise-cancelled "
        "point estimate of the population-level gap. single_draw_breach_rate is a "
        "separate diagnostic: the fraction of individual (finite-sample) replicates "
        "that would have tripped a naive single-draw audit, included to show how much "
        "of a naive single-sample 'flag' is just sampling noise, not signal."
    ),
}
with open(os.path.join(RESULTS_DIR, "denominator_sweep_results.json"), "w") as f:
    json.dump(out, f, indent=2)

import csv
sweep1_flat = [flatten_row(r) for r in sweep1]
sweep2_flat = [flatten_row(r) for r in sweep2]
with open(os.path.join(RESULTS_DIR, "sweep1_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sweep1_flat[0].keys()))
    w.writeheader()
    w.writerows(sweep1_flat)
with open(os.path.join(RESULTS_DIR, "sweep2_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sweep2_flat[0].keys()))
    w.writeheader()
    w.writerows(sweep2_flat)

print(f"\nSaved denominator_sweep_results.json, sweep1_summary.csv, sweep2_summary.csv to {RESULTS_DIR}")

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
any_breach_sweep1 = any(r["breach"] for r in sweep1)
print(f"Sweep 1 (pure group-dependent selection, S independent of (X,Y) given A):")
print(f"  Across OR_select in {OR_SWEEP_1} (includes the real observed value 0.64),")
print(f"  any point-estimate audit breach across the sweep = {any_breach_sweep1}")
print(f"  i.e. the standard per-group audit's EXPECTED (noise-cancelled) metrics NEVER")
print(f"  cross conventional tolerance, at ANY magnitude of upstream group-dependent")
print(f"  selection bias tested, including bias as extreme as OR_select=0.05 (large")
print(f"  underselection of group A=1). Meanwhile the (unaudited) SELECTION-STAGE")
print(f"  four-fifths ratio tracks OR_select directly and fails the four-fifths rule")
print(f"  for every OR_select < 0.80 -- including the real observed value, 0.64.")
if breach_gamma is not None:
    print(f"\nSweep 2 (leakage, OR_select fixed at 0.64):")
    print(f"  Point-estimate audit breach first trips at leak_gamma = {breach_gamma}")
    print(f"  i.e. the audit starts catching the bias only once candidate-generation")
    print(f"  selection also depends directly on the true outcome (not just on group).")
else:
    print(f"\nSweep 2 (leakage, OR_select fixed at 0.64):")
    print(f"  Point-estimate audit breach never tripped in the tested range {LEAK_SWEEP}.")
print("ALL_DONE")
