"""
ACCORD Build 2 -- the denominator demonstration, REBUILT (v2).

--------------------------------------------------------------------------
WHY THIS FILE EXISTS (do not delete simulate.py -- it is kept for contrast)
--------------------------------------------------------------------------
simulate.py has a structural bug, not a numerical one. Its generative model
draws the group attribute A and the feature X *independently*:

    a = rng.binomial(1, P_A, n_pop)
    x = rng.normal(0, 1, n_pop)                       # <- unrelated to a
    ...
    logit_s = s0 + s1 * a + leak_gamma * y_centered    # <- no X term at all

Because A carries no information about X or Y, Cov(f(X), pi(a)) = 0 by
construction, selection can only ever move the pool composition along an
axis (A) the audit is blind to *by definition* (the audit conditions on A),
and the "blindness" result that script reports -- flat pool-level metrics
across an 11-point OR_select sweep from 0.05 to 2.00 -- is not a finding, it
is the null model reproducing itself. That is confirmed empirically: the
demographic-parity gap barely moves between OR_select=1.00 (no bias at all)
and OR_select=0.05 (20x under-selection): 0.00112 vs 0.00129. There is
nothing in the generative model for the audit to fail to notice.

This file rebuilds the generative model so the instrument CAN fail:
  1. A is associated with X (group difference in features) at a sweepable
     strength rho, and with Y directly (a residual outcome gap not mediated
     by X) at a sweepable strength beta_AY. This plants a REAL population-
     level disparity.
  2. Selection depends on X directly (a "documentation volume" analogue --
     feature-like, correlated with the outcome) at a sweepable strength
     sx_coef, in addition to the original A-term (or_select) and Y-leakage
     term (leak_gamma).
  3. Three controls the original lacked: a POSITIVE control (plant a real
     gap, confirm the audit code reports it -- if it doesn't, nothing else
     in this file means anything), a NEGATIVE control (the sampling-noise
     floor, with CIs), and a POWER layer (minimum detectable gap at every
     grid point, from the standard two-proportion power formula).
  4. Monte Carlo statistics done properly: 300 replicates per grid point
     (500 for the two controls), CIs on every reported point estimate. No
     "3 seeds" (see accord/CORRECTIONS.md C7 -- reporting initialization
     noise as if it were sampling variance is exactly the mistake this file
     is built to not repeat).
  5. Grid design: the dense 11-point OR_select-only grid is cut to 5 points.
     The saved budget goes into replicate depth and a second dimension: a
     15x15 joint boundary map over (rho, leak_gamma), plus a dedicated sweep
     over the new X-selection term (sx_coef).
  6. Full metric ablation at every point: demographic parity, TPR gap, FPR
     gap, PPV gap, calibration gap, equalized odds, four-fifths -- computed,
     not assumed, so PPV's expected non-invariance and FPR's conditional
     invariance are TESTED rather than asserted.

Every population draw is decomposed into a POPULATION-level slice (before
selection, mask = everyone) and a POOL-level slice (mask = S==1, what a real
audit actually sees). Reporting both lets us separate "the audit is
reporting a real disparity that happens to survive selection" from "the
audit is reporting an artifact selection introduced" -- a distinction the
original script's single-axis design could not make even in principle.

100% synthetic, fixed seeds, fully reproducible, safe to commit. No MIMIC,
no credentialed data.

REFUTATION CONDITION (checked explicitly, reported plainly either way): if
pool-level gaps drift monotonically with or_select even at rho=0 and
beta_AY=0 (i.e. in the trivial null regime), the invariance claim is false
under conditions where the theory says it must hold, and that is reported
as a failure of the theory, not smoothed over.

Outputs (JSON + CSV) are written to ./results/ in this directory, with a
"v2_" filename prefix so they sit alongside (not over) simulate.py's output.
"""
import csv
import json
import os
import time
import zlib

import numpy as np
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Population parameters. or_select's realistic anchor (0.64) and the base
# demographic/prevalence numbers are calibrated to the real cohort exactly as
# in simulate.py (see accord/CORRECTIONS.md C12 and
# mimic_local_only/accord_differential/ exp2a/exp3 outputs).
#
# rho (A->X strength) and beta_AY (A->Y direct strength) have NO equivalent
# real-cohort-calibrated number in this project -- there is no measured
# "language group's feature-distribution shift in SD units" or "residual
# outcome log-odds gap not mediated by documentation volume" on file. They
# are therefore chosen as illustrative, clearly-labeled synthetic magnitudes
# (rho=0.45 SD, beta_AY=0.40 log-odds, a "medium" effect by conventional
# standards) whose ONLY job is to make the generative model non-degenerate
# so the instrument has something real to detect. This is stated plainly so
# the numbers are not mistaken for measured quantities.
# ---------------------------------------------------------------------------
P_A = 0.4664            # P(A=1), the "unrecorded/administratively-incomplete" group
BASE_YIELD_A0 = 0.2081   # P(S=1 | A=0, no extra X/Y dependence), majority-group yield
Y_BASE_RATE = 0.04       # target population base rate of the true outcome at rho=beta_AY=0
X_COEF = 2.0             # how strongly the latent feature X predicts true outcome Y
N_POP = 20_000           # population size per replicate
N_REPLICATES = 300       # replicates per one-dimensional sweep grid point
N_REPLICATES_CONTROL = 500  # replicates for the two controls (higher precision)
N_REPLICATES_MAP = 300   # replicates per cell of the 2-D boundary map
N_REFERENCE = 300_000    # size of the unbiased reference sample for the fixed classifier

RHO_REALISTIC = 0.45     # A->X mean shift, in SD of X (illustrative, see note above)
BETA_AY_REALISTIC = 0.40  # A->Y direct log-odds term, not mediated by X (illustrative)
SX_REALISTIC = 1.0        # selection log-odds per SD of X, "documentation volume" analogue

TOLERANCE_GAP = 0.05     # conventional parity-gap tolerance
FOUR_FIFTHS = 0.80        # conventional four-fifths-rule threshold
ALPHA = 0.05              # for CIs and power calc
POWER = 0.80               # target power for minimum-detectable-gap calc
Z_ALPHA2 = float(sps.norm.ppf(1 - ALPHA / 2))
Z_POWER = float(sps.norm.ppf(POWER))


def stable_seed(*parts):
    """Deterministic, cross-process-stable seed derivation. Python's builtin
    hash() is salted per-process for str/tuple-of-str (PYTHONHASHSEED
    randomization is on by default) -- confirmed empirically: hash() of the
    identical tuple differs between two separate `python3` invocations. That
    means a naive `hash((tag, params, r)) % 2**32` seeding scheme (as used
    in simulate.py) is NOT reproducible run-to-run, only self-consistent
    within a single run. zlib.crc32 has no such salting and is stable across
    processes and machines, so it is used here to satisfy the "fixed seeds,
    fully reproducible" requirement literally."""
    s = "|".join(str(p) for p in parts)
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def solve_intercept_for_base_rate(coef, target_rate, x_std=1.0, n=2_000_000, rng=None):
    """Numerically find b0 such that mean(sigmoid(b0 + coef*X)) ~= target_rate,
    X ~ N(0, x_std^2). Same routine as simulate.py -- this fixes Y_INTERCEPT
    for the rho=0, beta_AY=0 baseline; when rho/beta_AY are switched on, the
    realized population base rate will drift from Y_BASE_RATE (expected and
    harmless -- our target quantities are GAPS, not the absolute rate)."""
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
# Fit ONE fixed, group-blind classifier on a large unbiased reference sample
# (X ~ N(0,1), i.e. the idealized/group-unskewed population). It never sees
# A. Identical logic to simulate.py -- isolates the result to the
# candidate-generation stage, not to unfairness baked into the classifier.
# ---------------------------------------------------------------------------
def fit_reference_classifier():
    rng = np.random.default_rng(SEED + 1)
    x_ref = rng.normal(0, 1, N_REFERENCE)
    y_ref = rng.binomial(1, sigmoid(Y_INTERCEPT + X_COEF * x_ref))
    clf = LogisticRegression()
    clf.fit(x_ref.reshape(-1, 1), y_ref)
    scores_ref = clf.predict_proba(x_ref.reshape(-1, 1))[:, 1]
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
# One simulated draw: population -> selection -> pool -> audit metrics.
# ---------------------------------------------------------------------------
def simulate_once(n_pop, or_select, leak_gamma, sx_coef, rho, beta_ay, rng):
    """
    or_select : odds ratio for selection, group A=1 vs A=0 (direct A term).
    leak_gamma: direct log-odds contribution of Y to selection (0 = pure
                group-dependent selection given (A, X)).
    sx_coef   : direct log-odds contribution of X to selection, per SD of X
                (0 = selection independent of X given A; the "documentation
                volume" analogue described in the task).
    rho       : A->X mean shift, in SD of X (0 = groups identical in X).
    beta_ay   : A->Y direct log-odds term, not mediated by X (0 = no
                residual group gap in the outcome beyond what X carries).
    """
    a = rng.binomial(1, P_A, n_pop)
    x = rng.normal(0, 1, n_pop) + rho * a
    y = rng.binomial(1, sigmoid(Y_INTERCEPT + X_COEF * x + beta_ay * a))

    s0 = np.log(BASE_YIELD_A0 / (1 - BASE_YIELD_A0))  # logit(P(S=1|A=0)) baseline
    s1 = np.log(or_select)
    # Center X and Y by their EMPIRICAL population mean in this draw (not a
    # fixed population constant): this keeps the marginal selection rate
    # ~stable as sx_coef/leak_gamma/rho/beta_ay sweep, exactly as
    # simulate.py's y_centered trick did for its one axis, generalized to
    # both new continuous predictors so each axis can be interpreted without
    # a confounding shift in overall pool size.
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    logit_s = s0 + s1 * a + sx_coef * x_centered + leak_gamma * y_centered
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
    calib_gap = float(ss.mean() - yy.mean())
    return dict(n=n, n_pos=n_pos, n_neg=n_neg, TP=tp, FP=fp, FN=fn, TN=tn,
                TPR=tpr, FPR=fpr, PPV=ppv, pred_pos_rate=pred_pos_rate,
                calibration_gap=calib_gap)


GROUP_LEVEL_KEYS = ["n", "n_pos", "n_neg", "TPR", "FPR", "PPV", "pred_pos_rate", "calibration_gap"]


def audit_one_draw(draw):
    """Compute BOTH population-level (pre-selection) and pool-level (S=1)
    per-group audit metrics, plus the selection-stage audit. Returning both
    lets us separate a real population disparity that survives selection
    from a disparity selection itself introduced -- the decomposition the
    original single-axis script could not make."""
    a, y, s, pred, scores = draw["a"], draw["y"], draw["s"], draw["pred"], draw["scores"]
    pool_mask = s == 1
    pop_mask = np.ones_like(s, dtype=bool)

    pop_g0 = group_metrics(y, pred, scores, pop_mask & (a == 0))
    pop_g1 = group_metrics(y, pred, scores, pop_mask & (a == 1))
    pool_g0 = group_metrics(y, pred, scores, pool_mask & (a == 0))
    pool_g1 = group_metrics(y, pred, scores, pool_mask & (a == 1))

    sel_rate_0 = (s[a == 0] == 1).mean()
    sel_rate_1 = (s[a == 1] == 1).mean()
    sel_odds_0 = sel_rate_0 / (1 - sel_rate_0)
    sel_odds_1 = sel_rate_1 / (1 - sel_rate_1)
    sel_or_empirical = sel_odds_1 / sel_odds_0

    return dict(
        n_pool=int(pool_mask.sum()),
        pop_g0=pop_g0, pop_g1=pop_g1,
        pool_g0=pool_g0, pool_g1=pool_g1,
        selection_rate_a0=float(sel_rate_0), selection_rate_a1=float(sel_rate_1),
        selection_or_empirical=float(sel_or_empirical),
    )


def _agg_slice(reps, g0_key, g1_key):
    """Average each group-level quantity across replicates for one slice
    (population or pool), then derive gaps/ratios from the averaged values.
    Also returns per-replicate SIGNED gap arrays for CIs/power."""
    g0_means, g1_means = {}, {}
    per_rep_signed = {}
    for k in GROUP_LEVEL_KEYS:
        v0 = np.array([rep[g0_key].get(k, np.nan) for rep in reps], dtype=float)
        v1 = np.array([rep[g1_key].get(k, np.nan) for rep in reps], dtype=float)
        g0_means[k] = float(np.nanmean(v0))
        g1_means[k] = float(np.nanmean(v1))
        per_rep_signed[k] = v1 - v0  # signed, per-replicate gap (for CI on the mean)

    out = {"g0_mean": g0_means, "g1_mean": g1_means,
           "_raw_signed": {k: per_rep_signed[k] for k in ["pred_pos_rate", "TPR", "FPR", "PPV", "calibration_gap"]}}
    out["demographic_parity_gap"] = abs(g1_means["pred_pos_rate"] - g0_means["pred_pos_rate"])
    out["tpr_gap"] = abs(g1_means["TPR"] - g0_means["TPR"])
    out["fpr_gap"] = abs(g1_means["FPR"] - g0_means["FPR"])
    out["ppv_gap"] = abs(g1_means["PPV"] - g0_means["PPV"])
    out["calibration_gap_diff"] = abs(g1_means["calibration_gap"] - g0_means["calibration_gap"])
    out["equalized_odds_gap"] = max(out["tpr_gap"], out["fpr_gap"])
    out["four_fifths_ratio"] = (min(g0_means["pred_pos_rate"], g1_means["pred_pos_rate"]) /
                                 max(g0_means["pred_pos_rate"], g1_means["pred_pos_rate"]))

    # CI on the point estimate: SE of the replicate-averaged mean signed gap
    # = SD(per-replicate signed gap) / sqrt(n_replicates). This is the CI on
    # the noise-cancelled AGGREGATE point estimate reported as the primary
    # number (mirrors simulate.py's aggregation logic, formalized with a CI).
    n_reps = len(reps)
    for metric_name, key in [("demographic_parity_gap", "pred_pos_rate"),
                              ("tpr_gap", "TPR"), ("fpr_gap", "FPR"),
                              ("ppv_gap", "PPV"), ("calibration_gap_diff", "calibration_gap")]:
        signed = per_rep_signed[key]
        signed = signed[~np.isnan(signed)]
        if len(signed) > 1:
            se = float(np.std(signed, ddof=1) / np.sqrt(len(signed)))
        else:
            se = float("nan")
        pe = out[metric_name]
        out[f"{metric_name}_se"] = se
        out[f"{metric_name}_ci95"] = [pe - Z_ALPHA2 * se, pe + Z_ALPHA2 * se]
        # single-draw (one replicate, i.e. what one real audit run would see) noise SD:
        out[f"{metric_name}_single_draw_sd"] = float(np.std(np.abs(signed), ddof=1)) if len(signed) > 1 else float("nan")

    # Minimum detectable gap (two-proportion power formula), using average
    # group sample sizes from this slice. Reported for DP (uses n) and FPR
    # (uses n_neg, the actual denominator FPR is computed on).
    n0, n1 = g0_means["n"], g1_means["n"]
    p_dp = 0.5 * (g0_means["pred_pos_rate"] + g1_means["pred_pos_rate"])
    out["mde_demographic_parity"] = _mde(p_dp, n0, n1)
    n0_neg, n1_neg = g0_means["n_neg"], g1_means["n_neg"]
    p_fpr = 0.5 * (g0_means["FPR"] + g1_means["FPR"]) if not (np.isnan(g0_means["FPR"]) or np.isnan(g1_means["FPR"])) else np.nan
    out["mde_fpr_gap"] = _mde(p_fpr, n0_neg, n1_neg) if not np.isnan(p_fpr) else float("nan")

    out["n_g0_mean"] = n0
    out["n_g1_mean"] = n1
    return out


def _mde(p, n0, n1):
    """Standard two-independent-proportions minimum-detectable-effect
    formula: MDE = (z_{alpha/2} + z_power) * sqrt(p(1-p)*(1/n0+1/n1)),
    evaluated at the pooled rate p and the observed (average) group sample
    sizes. This is the size of the TRUE gap that a study with these sample
    sizes has an 80% chance of detecting at alpha=0.05, two-sided."""
    if n0 <= 0 or n1 <= 0 or np.isnan(p) or p <= 0 or p >= 1:
        return float("nan")
    se = np.sqrt(p * (1 - p) * (1 / n0 + 1 / n1))
    return float((Z_ALPHA2 + Z_POWER) * se)


def aggregate_group_level(reps):
    """Aggregate both the population slice and the pool slice, plus the
    selection-stage audit, plus the selection-induced excess (pool gap minus
    population gap) for each headline metric."""
    pop = _agg_slice(reps, "pop_g0", "pop_g1")
    pool = _agg_slice(reps, "pool_g0", "pool_g1")

    excess = {}
    for m in ["demographic_parity_gap", "tpr_gap", "fpr_gap", "ppv_gap", "calibration_gap_diff"]:
        excess[m] = pool[m] - pop[m]

    sel0 = np.array([rep["selection_rate_a0"] for rep in reps])
    sel1 = np.array([rep["selection_rate_a1"] for rep in reps])
    sel0_mean, sel1_mean = float(sel0.mean()), float(sel1.mean())
    selection = dict(
        selection_rate_a0=sel0_mean, selection_rate_a1=sel1_mean,
        selection_four_fifths_ratio=min(sel0_mean, sel1_mean) / max(sel0_mean, sel1_mean),
        selection_or_empirical=float(np.mean([rep["selection_or_empirical"] for rep in reps])),
    )
    n_pool_mean = float(np.mean([rep["n_pool"] for rep in reps]))

    return dict(population=pop, pool=pool, selection_induced_excess=excess,
                selection=selection, n_pool_mean=n_pool_mean, n_replicates=len(reps))


def trend_test(rows, grid_values, metric_raw_key, slice_key="pool"):
    """Proper test for 'does this metric drift with the swept parameter':
    pool the REPLICATE-LEVEL signed gaps (not the aggregated point
    estimates) across all grid points, with the grid value as covariate,
    and run ordinary least-squares (scipy.stats.linregress) for a slope and
    its p-value. This is the correct significance test because it uses the
    SAME per-replicate noise for every grid point in the sweep -- unlike
    comparing point-estimate magnitudes against a noise floor computed at a
    DIFFERENT replicate count (that conflates two different standard
    errors and produces spurious "drift" flags). Returns (slope, p_value,
    n_total_replicate_points)."""
    xs, ys = [], []
    for v, row in zip(grid_values, rows):
        arr = row[slice_key]["_raw_signed"][metric_raw_key]
        arr = arr[~np.isnan(arr)]
        xs.extend([v] * len(arr))
        ys.extend(arr.tolist())
    if len(set(xs)) < 2:
        return float("nan"), float("nan"), len(xs)
    slope, intercept, r, p, se = sps.linregress(xs, ys)
    return float(slope), float(p), len(xs)


def strip_raw(agg):
    """Remove the (numpy-array-valued, not JSON-serializable) raw
    per-replicate arrays before writing results to disk. Call once the
    trend tests that need them have already been computed."""
    for slice_key in ("population", "pool"):
        agg[slice_key].pop("_raw_signed", None)
    return agg


def point_estimate_breach(pool_agg):
    checks = [
        pool_agg["demographic_parity_gap"] > TOLERANCE_GAP,
        pool_agg["tpr_gap"] > TOLERANCE_GAP,
        pool_agg["fpr_gap"] > TOLERANCE_GAP,
        pool_agg["ppv_gap"] > TOLERANCE_GAP,
        pool_agg["four_fifths_ratio"] < FOUR_FIFTHS,
    ]
    return any(checks)


def run_point(tag, or_select, leak_gamma, sx_coef, rho, beta_ay, n_replicates, seed_key=None):
    reps = []
    key = seed_key if seed_key is not None else (tag, or_select, leak_gamma, sx_coef, rho, beta_ay)
    for r in range(n_replicates):
        rng = np.random.default_rng(stable_seed(key, r))
        draw = simulate_once(N_POP, or_select, leak_gamma, sx_coef, rho, beta_ay, rng)
        reps.append(audit_one_draw(draw))
    agg = aggregate_group_level(reps)
    agg["breach_pool"] = point_estimate_breach(agg["pool"])
    agg["breach_population"] = point_estimate_breach(agg["population"])
    agg["params"] = dict(or_select=or_select, leak_gamma=leak_gamma, sx_coef=sx_coef,
                          rho=rho, beta_ay=beta_ay, n_replicates=n_replicates)
    return agg


def run_sweep(tag, grid_key, grid_values, base_params, n_replicates=N_REPLICATES):
    """grid_key is one of 'or_select','leak_gamma','sx_coef','rho','beta_ay';
    grid_values sweeps that param, all others held at base_params."""
    print(f"\n=== SWEEP: {tag} (varying {grid_key}) ===")
    rows = []
    for v in grid_values:
        params = dict(base_params)
        params[grid_key] = v
        agg = run_point(tag, n_replicates=n_replicates, **params)
        rows.append(agg)
        pool = agg["pool"]
        print(f"  {grid_key}={v:<8} pool_DP_gap={pool['demographic_parity_gap']:.4f} "
              f"pool_FPR_gap={pool['fpr_gap']:.4f} pool_PPV_gap={pool['ppv_gap']:.4f} "
              f"4/5={pool['four_fifths_ratio']:.3f} sel_4/5={agg['selection']['selection_four_fifths_ratio']:.3f} "
              f"breach={agg['breach_pool']} mde_DP={pool['mde_demographic_parity']:.4f} "
              f"mde_FPR={pool['mde_fpr_gap']:.4f}")
    return rows


t_start = time.time()

# ---------------------------------------------------------------------------
# CONTROL 1: POSITIVE CONTROL. Plant a real population-level gap (rho and
# beta_ay both on) with NO selection differential at all (or_select=1,
# sx_coef=0, leak_gamma=0). If the audit reports ~0 here, the audit code is
# broken and nothing downstream means anything -- run this first.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("CONTROL 1 / POSITIVE CONTROL (run first -- validates the audit code)")
print("=" * 78)
positive_control = run_point(
    "positive_control", or_select=1.0, leak_gamma=0.0, sx_coef=0.0,
    rho=RHO_REALISTIC, beta_ay=BETA_AY_REALISTIC, n_replicates=N_REPLICATES_CONTROL)
pc_pool = positive_control["pool"]
print(f"Planted gap: rho={RHO_REALISTIC}, beta_AY={BETA_AY_REALISTIC}, NO selection differential")
print(f"  pool DP gap  = {pc_pool['demographic_parity_gap']:.4f}  "
      f"95% CI [{pc_pool['demographic_parity_gap_ci95'][0]:.4f}, {pc_pool['demographic_parity_gap_ci95'][1]:.4f}]")
print(f"  pool FPR gap = {pc_pool['fpr_gap']:.4f}  "
      f"95% CI [{pc_pool['fpr_gap_ci95'][0]:.4f}, {pc_pool['fpr_gap_ci95'][1]:.4f}]")
print(f"  pool PPV gap = {pc_pool['ppv_gap']:.4f}  "
      f"95% CI [{pc_pool['ppv_gap_ci95'][0]:.4f}, {pc_pool['ppv_gap_ci95'][1]:.4f}]")
print(f"  pool four-fifths ratio = {pc_pool['four_fifths_ratio']:.3f}")
positive_control_detected = pc_pool["demographic_parity_gap_ci95"][0] > 0 or pc_pool["fpr_gap_ci95"][0] > 0
print(f"AUDIT CODE VALID (CI excludes 0 for DP or FPR gap): {positive_control_detected}")
if not positive_control_detected:
    print("!!! POSITIVE CONTROL FAILED: the audit did not detect a planted gap. "
          "Every downstream number in this file should be treated as void until "
          "this is fixed. Continuing to run so the failure is on record.")

# ---------------------------------------------------------------------------
# CONTROL 2: NEGATIVE CONTROL. No planted gap, no selection at all. This is
# the sampling-noise floor: what the audit reports when there is truly
# nothing there. If the sweep effects below are smaller than this floor,
# the sweep resolves nothing.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("CONTROL 2 / NEGATIVE CONTROL (sampling-noise floor)")
print("=" * 78)
negative_control = run_point(
    "negative_control", or_select=1.0, leak_gamma=0.0, sx_coef=0.0,
    rho=0.0, beta_ay=0.0, n_replicates=N_REPLICATES_CONTROL)
nc_pool = negative_control["pool"]
print(f"  pool DP gap  = {nc_pool['demographic_parity_gap']:.5f}  "
      f"95% CI [{nc_pool['demographic_parity_gap_ci95'][0]:.5f}, {nc_pool['demographic_parity_gap_ci95'][1]:.5f}]  "
      f"single-draw SD={nc_pool['demographic_parity_gap_single_draw_sd']:.5f}")
print(f"  pool FPR gap = {nc_pool['fpr_gap']:.5f}  "
      f"95% CI [{nc_pool['fpr_gap_ci95'][0]:.5f}, {nc_pool['fpr_gap_ci95'][1]:.5f}]  "
      f"single-draw SD={nc_pool['fpr_gap_single_draw_sd']:.5f}")
print(f"  pool PPV gap = {nc_pool['ppv_gap']:.5f}  "
      f"95% CI [{nc_pool['ppv_gap_ci95'][0]:.5f}, {nc_pool['ppv_gap_ci95'][1]:.5f}]  "
      f"single-draw SD={nc_pool['ppv_gap_single_draw_sd']:.5f}")
noise_floor_dp = nc_pool["demographic_parity_gap_ci95"][1]
noise_floor_fpr = nc_pool["fpr_gap_ci95"][1]
print(f"NOISE FLOOR (upper 95% CI bound, aggregate point estimate): DP={noise_floor_dp:.5f}, FPR={noise_floor_fpr:.5f}")

# ---------------------------------------------------------------------------
# SWEEP 0 (REFUTATION CHECK): the trivial regime (rho=0, beta_ay=0) swept
# across or_select. This is the direct test of the condition that would
# REFUTE the theory: gaps drifting monotonically with selection magnitude at
# zero X-dependence and zero Y-dependence. Reported plainly either way.
# ---------------------------------------------------------------------------
OR_GRID = [0.05, 0.20, 0.50, 0.64, 1.00]  # 5 points (cut from the original 11); includes real value 0.64
sweep0 = run_sweep(
    "sweep0_refutation_check_trivial_regime", "or_select", OR_GRID,
    base_params=dict(leak_gamma=0.0, sx_coef=0.0, rho=0.0, beta_ay=0.0))

dp_vals_sweep0 = [r["pool"]["demographic_parity_gap"] for r in sweep0]
fpr_vals_sweep0 = [r["pool"]["fpr_gap"] for r in sweep0]
spearman_dp = float(sps.spearmanr(OR_GRID, dp_vals_sweep0).correlation)
spearman_fpr = float(sps.spearmanr(OR_GRID, fpr_vals_sweep0).correlation)
# Correct significance test: regress the REPLICATE-LEVEL signed gap on
# or_select, pooling all 300-replicate points (matched replicate count,
# matched noise level throughout -- NOT a magnitude comparison against a
# noise floor computed at a different replicate count, which produces
# spurious "drift" flags; see trend_test() docstring).
dp_slope0, dp_p0, dp_n0 = trend_test(sweep0, OR_GRID, "pred_pos_rate")
fpr_slope0, fpr_p0, fpr_n0 = trend_test(sweep0, OR_GRID, "FPR")
refutation_triggered = (dp_p0 < ALPHA) or (fpr_p0 < ALPHA)
print(f"\nREFUTATION CHECK: trivial-regime gaps vs or_select -- "
      f"DP values={['%.5f' % v for v in dp_vals_sweep0]}, FPR values={['%.5f' % v for v in fpr_vals_sweep0]}")
print(f"  Spearman rho(or_select, DP gap) = {spearman_dp:.3f} (descriptive only, n=5 points, not the test used), "
      f"Spearman rho(or_select, FPR gap) = {spearman_fpr:.3f}")
print(f"  Replicate-level trend test (linear regression, {dp_n0} DP / {fpr_n0} FPR data points): "
      f"DP slope={dp_slope0:.6f} p={dp_p0:.4f}; FPR slope={fpr_slope0:.6f} p={fpr_p0:.4f}")
print(f"  Any trend significant at alpha={ALPHA}: {refutation_triggered}")
print(f"  REFUTATION CONDITION MET: {refutation_triggered}")

# ---------------------------------------------------------------------------
# SWEEP 1 (core result, redone): realistic regime (rho, beta_ay fixed at
# planted-gap values, confirmed detectable by the positive control) swept
# across the SAME 5-point or_select grid, with sx_coef=leak_gamma=0 (pure
# selection-on-A). Tests the actual invariance claim: does the pool-level
# gap stay ~constant (at its REAL, non-zero magnitude) as or_select sweeps,
# i.e. is the audit blind to SELECTION MAGNITUDE specifically, while still
# correctly reflecting the underlying disparity?
# ---------------------------------------------------------------------------
sweep1 = run_sweep(
    "sweep1_core_realistic_regime", "or_select", OR_GRID,
    base_params=dict(leak_gamma=0.0, sx_coef=0.0, rho=RHO_REALISTIC, beta_ay=BETA_AY_REALISTIC))
dp_vals_sweep1 = [r["pool"]["demographic_parity_gap"] for r in sweep1]
fpr_vals_sweep1 = [r["pool"]["fpr_gap"] for r in sweep1]
dp_range_sweep1 = max(dp_vals_sweep1) - min(dp_vals_sweep1)
fpr_range_sweep1 = max(fpr_vals_sweep1) - min(fpr_vals_sweep1)
dp_slope1, dp_p1, _ = trend_test(sweep1, OR_GRID, "pred_pos_rate")
fpr_slope1, fpr_p1, _ = trend_test(sweep1, OR_GRID, "FPR")
invariant_sweep1 = (dp_p1 >= ALPHA) and (fpr_p1 >= ALPHA)
print(f"\nSWEEP 1 range across or_select: DP gap range = {dp_range_sweep1:.5f} "
      f"(values {['%.4f' % v for v in dp_vals_sweep1]}), FPR gap range = {fpr_range_sweep1:.5f}")
print(f"  Replicate-level trend test: DP slope={dp_slope1:.6f} p={dp_p1:.4f}; "
      f"FPR slope={fpr_slope1:.6f} p={fpr_p1:.4f}")
print(f"  Invariant to or_select at the REAL nonzero gap level (no significant trend, alpha={ALPHA}): {invariant_sweep1}")

# ---------------------------------------------------------------------------
# SWEEP 2: selection depends on X directly (sx_coef sweep). This is the
# direct test of "FPR should be invariant when selection is independent of
# X given A" -- here selection is made to DEPEND on X, so this is the
# regime where invariance is NOT expected to hold, and we check whether it
# in fact breaks.
# ---------------------------------------------------------------------------
SX_GRID = [0.0, 0.25, 0.50, 1.00, 1.50, 2.00, 3.00]
sweep2 = run_sweep(
    "sweep2_selection_depends_on_x", "sx_coef", SX_GRID,
    base_params=dict(or_select=0.64, leak_gamma=0.0, rho=RHO_REALISTIC, beta_ay=BETA_AY_REALISTIC))

# ---------------------------------------------------------------------------
# SWEEP 3: outcome-leakage into selection (leak_gamma sweep), realistic
# regime, or_select fixed at the real observed value 0.64. Reduced grid from
# the original 11 points.
# ---------------------------------------------------------------------------
LEAK_GRID = [0.0, 1.0, 2.0, 4.0, 6.0, 10.0, 14.0, 20.0]
sweep3 = run_sweep(
    "sweep3_leakage_at_or_0.64", "leak_gamma", LEAK_GRID,
    base_params=dict(or_select=0.64, sx_coef=0.0, rho=RHO_REALISTIC, beta_ay=BETA_AY_REALISTIC))
breach_gamma = next((r["params"]["leak_gamma"] for r in sweep3 if r["breach_pool"]), None)

# ---------------------------------------------------------------------------
# BOUNDARY MAP: joint (rho, leak_gamma), 15x15, or_select fixed at 0.64
# (real value), sx_coef=0 and beta_ay=0 so the map isolates exactly these
# two axes (A->X strength and selection's Y-dependence) without a third
# hidden parameter confounding the picture.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BOUNDARY MAP: rho (A->X strength) x leak_gamma (Y-dependence of selection)")
print("=" * 78)
RHO_GRID = np.linspace(0.0, 1.0, 15).tolist()
GAMMA_GRID = np.linspace(0.0, 20.0, 15).tolist()
map_rows = []
t_map0 = time.time()
for i, rho_v in enumerate(RHO_GRID):
    for j, gamma_v in enumerate(GAMMA_GRID):
        agg = run_point("boundary_map", or_select=0.64, leak_gamma=gamma_v, sx_coef=0.0,
                         rho=rho_v, beta_ay=0.0, n_replicates=N_REPLICATES_MAP,
                         seed_key=("boundary_map", i, j))
        map_rows.append(dict(
            rho=rho_v, leak_gamma=gamma_v,
            pool_dp_gap=agg["pool"]["demographic_parity_gap"],
            pool_fpr_gap=agg["pool"]["fpr_gap"],
            pool_four_fifths=agg["pool"]["four_fifths_ratio"],
            population_dp_gap=agg["population"]["demographic_parity_gap"],
            selection_induced_excess_dp=agg["selection_induced_excess"]["demographic_parity_gap"],
            breach=agg["breach_pool"],
        ))
    print(f"  rho={rho_v:.3f} row done ({i+1}/{len(RHO_GRID)}), elapsed={time.time()-t_map0:.1f}s")
print(f"Boundary map: {len(map_rows)} cells, {time.time()-t_map0:.1f}s")

# first breach along gamma at rho=0 (pure trivial-regime leakage boundary)
rho0_rows = [r for r in map_rows if abs(r["rho"] - 0.0) < 1e-9]
breach_gamma_at_rho0 = next((r["leak_gamma"] for r in sorted(rho0_rows, key=lambda r: r["leak_gamma"]) if r["breach"]), None)

# ---------------------------------------------------------------------------
# METRIC ABLATION TABLE: full metric set at every sweep-1 grid point (the
# realistic regime, pure selection-on-A) -- this is where invariance is
# theoretically expected to hold for FPR (S _||_ X | A there) and NOT
# expected to hold for PPV (pool prevalence changes with selection even
# when selection is independent of X and Y given A, because P(Y=1|A) is
# already unequal across groups from beta_ay/rho, and the SIZE of the pool
# for each group changes with or_select, taking the group-level average of
# the SAME per-group PPV -- so this ablation tests whether PPV actually
# moves and FPR actually doesn't, rather than assuming it).
# ---------------------------------------------------------------------------
metric_ablation_rows = []
for row in sweep1:
    p = row["params"]
    pool = row["pool"]
    metric_ablation_rows.append(dict(
        or_select=p["or_select"], leak_gamma=p["leak_gamma"], sx_coef=p["sx_coef"],
        rho=p["rho"], beta_ay=p["beta_ay"],
        demographic_parity_gap=pool["demographic_parity_gap"],
        tpr_gap=pool["tpr_gap"], fpr_gap=pool["fpr_gap"], ppv_gap=pool["ppv_gap"],
        calibration_gap_diff=pool["calibration_gap_diff"],
        equalized_odds_gap=pool["equalized_odds_gap"],
        four_fifths_ratio=pool["four_fifths_ratio"],
    ))

fpr_gaps_ablation = [r["fpr_gap"] for r in metric_ablation_rows]
ppv_gaps_ablation = [r["ppv_gap"] for r in metric_ablation_rows]
fpr_range_ablation = max(fpr_gaps_ablation) - min(fpr_gaps_ablation)
ppv_range_ablation = max(ppv_gaps_ablation) - min(ppv_gaps_ablation)
# Same trend-test methodology as the refutation check / sweep-1 invariance
# check above (replicate-level regression, matched noise level across grid
# points) -- NOT a range-vs-external-noise-floor magnitude comparison.
ppv_slope1, ppv_p1, _ = trend_test(sweep1, OR_GRID, "PPV")
fpr_invariant = bool(fpr_p1 >= ALPHA)
ppv_invariant = bool(ppv_p1 >= ALPHA)
print(f"\nMETRIC ABLATION (sweep 1, realistic regime, pure selection-on-A):")
print(f"  FPR gap range across or_select = {fpr_range_ablation:.5f}, trend p={fpr_p1:.4f} -> "
      f"{'INVARIANT (no significant trend)' if fpr_invariant else 'NOT INVARIANT (significant trend)'}")
print(f"  PPV gap range across or_select = {ppv_range_ablation:.5f}, trend p={ppv_p1:.4f} -> "
      f"{'INVARIANT (no significant trend)' if ppv_invariant else 'NOT INVARIANT (significant trend) -- expected'}")

elapsed_total = time.time() - t_start

# ---------------------------------------------------------------------------
# SAVE (strip the not-JSON-serializable raw per-replicate arrays now that
# every trend test that needed them has already run)
# ---------------------------------------------------------------------------
strip_raw(positive_control)
strip_raw(negative_control)
for row in sweep0 + sweep1 + sweep2 + sweep3:
    strip_raw(row)

out = {
    "seed": SEED,
    "elapsed_seconds": elapsed_total,
    "population_params": {
        "P_A": P_A, "BASE_YIELD_A0": BASE_YIELD_A0, "Y_BASE_RATE": Y_BASE_RATE,
        "X_COEF": X_COEF, "Y_INTERCEPT": Y_INTERCEPT, "N_POP": N_POP,
        "N_REPLICATES": N_REPLICATES, "N_REPLICATES_CONTROL": N_REPLICATES_CONTROL,
        "N_REPLICATES_MAP": N_REPLICATES_MAP,
        "RHO_REALISTIC": RHO_REALISTIC, "BETA_AY_REALISTIC": BETA_AY_REALISTIC,
        "SX_REALISTIC": SX_REALISTIC,
    },
    "classifier": CLF_INFO,
    "tolerances": {"parity_gap_tolerance": TOLERANCE_GAP, "four_fifths_threshold": FOUR_FIFTHS,
                   "alpha": ALPHA, "power": POWER},
    "control_1_positive": positive_control,
    "control_1_positive_detected": positive_control_detected,
    "control_2_negative": negative_control,
    "noise_floor": {"demographic_parity_gap_upper95": noise_floor_dp, "fpr_gap_upper95": noise_floor_fpr},
    "sweep0_refutation_check_trivial_regime": sweep0,
    "refutation_check": {
        "method": "OLS regression of replicate-level signed gap on or_select, "
                  "pooled across all replicates at all 5 grid points (matched "
                  "replicate count / noise level throughout the sweep)",
        "spearman_or_select_vs_dp_gap_descriptive_n5": spearman_dp,
        "spearman_or_select_vs_fpr_gap_descriptive_n5": spearman_fpr,
        "dp_trend_slope": dp_slope0, "dp_trend_p": dp_p0,
        "fpr_trend_slope": fpr_slope0, "fpr_trend_p": fpr_p0,
        "refutation_condition_met": refutation_triggered,
    },
    "sweep1_core_realistic_regime": sweep1,
    "sweep1_dp_gap_range": dp_range_sweep1,
    "sweep1_fpr_gap_range": fpr_range_sweep1,
    "sweep1_invariance_trend_test": {
        "dp_trend_slope": dp_slope1, "dp_trend_p": dp_p1,
        "fpr_trend_slope": fpr_slope1, "fpr_trend_p": fpr_p1,
        "invariant": invariant_sweep1,
    },
    "sweep2_selection_depends_on_x": sweep2,
    "sweep3_leakage_at_or_0.64": sweep3,
    "sweep3_breach_gamma_first_tripped": breach_gamma,
    "boundary_map": {
        "rho_grid": RHO_GRID, "gamma_grid": GAMMA_GRID, "cells": map_rows,
        "breach_gamma_at_rho0": breach_gamma_at_rho0,
    },
    "metric_ablation": {
        "rows": metric_ablation_rows,
        "fpr_gap_range": fpr_range_ablation, "fpr_trend_p": fpr_p1, "fpr_invariant": bool(fpr_invariant),
        "ppv_gap_range": ppv_range_ablation, "ppv_trend_p": ppv_p1, "ppv_invariant": bool(ppv_invariant),
    },
}
with open(os.path.join(RESULTS_DIR, "v2_denominator_results.json"), "w") as f:
    json.dump(out, f, indent=2)


def flatten_sweep_row(row):
    p = row["params"]
    pool, pop, exc, sel = row["pool"], row["population"], row["selection_induced_excess"], row["selection"]
    flat = dict(or_select=p["or_select"], leak_gamma=p["leak_gamma"], sx_coef=p["sx_coef"],
                rho=p["rho"], beta_ay=p["beta_ay"], n_replicates=p["n_replicates"],
                breach_pool=row["breach_pool"], breach_population=row["breach_population"],
                n_pool_mean=row["n_pool_mean"])
    for k in ["demographic_parity_gap", "tpr_gap", "fpr_gap", "ppv_gap", "calibration_gap_diff",
              "equalized_odds_gap", "four_fifths_ratio"]:
        flat[f"pool_{k}"] = pool[k]
        flat[f"population_{k}"] = pop[k]
    for k in ["demographic_parity_gap", "tpr_gap", "fpr_gap", "ppv_gap", "calibration_gap_diff"]:
        flat[f"pool_{k}_ci_low"] = pool[f"{k}_ci95"][0]
        flat[f"pool_{k}_ci_high"] = pool[f"{k}_ci95"][1]
        flat[f"selection_induced_excess_{k}"] = exc[k]
    flat["pool_mde_demographic_parity"] = pool["mde_demographic_parity"]
    flat["pool_mde_fpr_gap"] = pool["mde_fpr_gap"]
    flat["selection_four_fifths_ratio"] = sel["selection_four_fifths_ratio"]
    flat["selection_or_empirical"] = sel["selection_or_empirical"]
    return flat


all_sweep_rows = (
    [("sweep0_refutation_check", r) for r in sweep0]
    + [("sweep1_core_realistic", r) for r in sweep1]
    + [("sweep2_x_selection", r) for r in sweep2]
    + [("sweep3_leakage", r) for r in sweep3]
    + [("control_positive", positive_control)]
    + [("control_negative", negative_control)]
)
sweep_flat = []
for tag, r in all_sweep_rows:
    flat = flatten_sweep_row(r)
    flat["sweep"] = tag
    sweep_flat.append(flat)

fieldnames = ["sweep"] + [k for k in sweep_flat[0].keys() if k != "sweep"]
with open(os.path.join(RESULTS_DIR, "v2_sweeps_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(sweep_flat)

with open(os.path.join(RESULTS_DIR, "v2_boundary_map.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(map_rows[0].keys()))
    w.writeheader()
    w.writerows(map_rows)

with open(os.path.join(RESULTS_DIR, "v2_metric_ablation.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(metric_ablation_rows[0].keys()))
    w.writeheader()
    w.writerows(metric_ablation_rows)

print(f"\nSaved v2_denominator_results.json, v2_sweeps_summary.csv, v2_boundary_map.csv, "
      f"v2_metric_ablation.csv to {RESULTS_DIR}")
print(f"Total elapsed: {elapsed_total:.1f}s")

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
print(f"1. POSITIVE CONTROL (audit code validity): gap detected with CI excluding 0 = "
      f"{positive_control_detected}. {'AUDIT CODE VALIDATED.' if positive_control_detected else 'AUDIT CODE BROKEN -- STOP.'}")
print(f"2. NEGATIVE CONTROL / noise floor: DP upper95={noise_floor_dp:.5f}, FPR upper95={noise_floor_fpr:.5f}")
print(f"3. REFUTATION CHECK (trivial regime, zero X- and Y-dependence): "
      f"significant drift with or_select (replicate-level trend test, alpha={ALPHA}) = {refutation_triggered} "
      f"({'THEORY REFUTED IN THIS REGIME' if refutation_triggered else 'no significant drift -- consistent with theory'})")
print(f"4. SWEEP 1 (realistic regime, pure selection-on-A): DP gap range across or_select = "
      f"{dp_range_sweep1:.5f} (p={dp_p1:.4f}), FPR gap range = {fpr_range_sweep1:.5f} (p={fpr_p1:.4f}) -- "
      f"{'INVARIANT to selection magnitude, at the REAL nonzero gap level' if invariant_sweep1 else 'gap moves significantly with or_select -- audit is NOT blind here'}")
print(f"5. SWEEP 2 (selection depends on X): FPR gap at sx_coef=0 -> {sweep2[0]['pool']['fpr_gap']:.4f}, "
      f"at sx_coef={SX_GRID[-1]} -> {sweep2[-1]['pool']['fpr_gap']:.4f}")
print(f"6. SWEEP 3 (leakage at or_select=0.64): breach first at leak_gamma = {breach_gamma}")
print(f"7. METRIC ABLATION: FPR invariant across or_select = {fpr_invariant}; "
      f"PPV invariant across or_select = {ppv_invariant} (expected: PPV moves, FPR does not)")
print(f"8. BOUNDARY MAP: breach region computed over {len(RHO_GRID)}x{len(GAMMA_GRID)} grid; "
      f"breach_gamma_at_rho=0 = {breach_gamma_at_rho0}")
print("ALL_DONE")
