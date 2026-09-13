"""Pure evaluation primitives for the public-controls work (Items 4 and 10).

WHY THIS FILE EXISTS
---------------------
`accord/stats/intervals.py` reads MIMIC-derived gold data at import time (it
opens a PhysioNet-credentialed local path at module scope, not inside a
function), so it
cannot be imported from any script that must stay MIMIC-free. This module
reimplements, from scratch, the one piece of it every public-controls check
needs: the stratified percentile bootstrap over an AUROC. `test_bootstrap_matches_intervals_source`
in `tests/test_public_controls_pins.py` reads the TEXT of `intervals.py` and
pins the exact lines this function must keep reproducing; if that pin ever
fails it means `intervals.py` changed underneath this reimplementation, not
that this file is wrong.

Everything below is pure: numpy/scipy/sklearn only, no filesystem read of any
gold or clinical table, no network call, no model load. Item 4 (C101) and
Item 10 (C102) both import this module.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy import stats as _stats
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These three modules are on the plan's explicit "reuse without side effects"
# list; each is imported here for the specific function(s) named there, not
# reimplemented. accord/control/delong.py is pure numpy/scipy. injection_
# sensitivity.py reads results/accord/mde_recomputed.json (a public, already-
# registered artifact, not MIMIC) at import time to set its own ANALYTIC_MDE
# constant; harmless to import, and its `interpolate_crossing` is reused
# as-is rather than copied. marginal_feasibility_gate.py only reads anything
# inside `main()`, guarded by `__name__ == "__main__"`, so importing it for
# `best_table`/`attainable_floor_p`/`mcc` triggers no read.
from accord.control.delong import _fast_delong, delong_roc_test  # noqa: E402
from accord.stats.injection_sensitivity import interpolate_crossing  # noqa: E402,F401
from accord.stats.marginal_feasibility_gate import (  # noqa: E402,F401
    attainable_floor_p, best_table, mcc,
)


# ---------------------------------------------------------------------------
# Pinned reimplementation of intervals.py's stratified percentile bootstrap
# (intervals.py lines ~74-76, the E3 stratified loop). Same RNG order: a
# fresh default_rng(seed), then B iterations of
#   i = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
# in exactly that order, one np.random call sequence per replicate.
# ---------------------------------------------------------------------------
def stratified_bootstrap_auroc(y, s, seed, B=10_000):
    """Exact loop reimplementation. Slow (O(B) Python-level roc_auc_score
    calls) by design, so it stays a faithful, auditable copy of the pinned
    lines rather than an optimization of them. Returns an array of B AUROCs.
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    strat = []
    for _ in range(B):
        i = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
        strat.append(roc_auc_score(y[i], s[i]))
    return np.array(strat)


def percentile_ci(boots, lo=2.5, hi=97.5):
    return float(np.percentile(boots, lo)), float(np.percentile(boots, hi))


def stratified_bootstrap_auroc_fast(y, s, seed, B=10_000):
    """Vectorized equivalent of `stratified_bootstrap_auroc`, used only where
    B=10,000 x many outer replicates makes the plain Python loop too slow to
    run at the pre-registered replicate count (the C101 bootstrap-type rule,
    C102 S2b). Draws the SAME two index streams in the SAME order as the
    pinned loop (rng.choice(pos, m, True) then rng.choice(neg, n, True), one
    rng per call, called once per outer replicate here since this function
    handles one (y, s) pair per call, same as the slow version).

    Equivalence with `stratified_bootstrap_auroc` is checked by
    `test_bootstrap_fast_matches_slow` in the pin test: same seed, same B,
    same y/s must give identical bootstrap arrays. Ties are handled with
    average (mid-)ranks, matching sklearn's roc_auc_score on data without
    ties and matching it closely with the average-rank convention when there
    are ties (rare here: NLI log-odds and additive Gaussian noise are
    continuous).
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    pos_idx, neg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    m, n = len(pos_idx), len(neg_idx)
    rng = np.random.default_rng(seed)
    boots = np.empty(B, dtype=float)
    # Draw exactly like the pinned loop, replicate by replicate, but stack
    # the per-replicate resampled score rows into one (B, m+n) matrix so the
    # rank computation vectorizes over the B axis.
    pos_draw = np.empty((B, m), dtype=np.int64)
    neg_draw = np.empty((B, n), dtype=np.int64)
    for b in range(B):
        pos_draw[b] = rng.choice(pos_idx, m, True)
        neg_draw[b] = rng.choice(neg_idx, n, True)
    combined = np.concatenate([s[pos_draw], s[neg_draw]], axis=1)  # (B, m+n)
    order = np.argsort(combined, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    row_idx = np.arange(combined.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(1, m + n + 1)[None, :]
    rank_sum_pos = ranks[:, :m].sum(axis=1)
    boots[:] = (rank_sum_pos - m * (m + 1) / 2.0) / (m * n)
    return boots


# ---------------------------------------------------------------------------
# Decision rules, each a function (y, s) -> bool "fires" (rejects / flags).
# y is the fixed slot-role vector (1 = positive slot, 0 = negative slot); s
# is the replicate's drawn score vector, same length.
# ---------------------------------------------------------------------------

def mw_one_sided_p(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    _, p = _stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="greater")
    return float(p)


def mw_two_sided_p(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    _, p = _stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="two-sided")
    return float(p)


def rule_mw_one_sided(y, s, alpha=0.05):
    return mw_one_sided_p(y, s) < alpha


def rule_mw_two_sided(y, s, alpha=0.05):
    return mw_two_sided_p(y, s) < alpha


def rule_c98(y, s, mde=0.7355, alpha=0.05):
    """C98's rule, reimplemented for measurement only: AUROC >= mde AND
    one-sided Mann-Whitney p < alpha. `accord/differential/C98_PREREGISTRATION.md`
    and `graded_score_probe.py` are never touched; this is a separate copy of
    the same boolean condition, used to measure the power of that rule at
    this evaluation's own size, per the C101 pre-registration.
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    auc = roc_auc_score(y, s)
    return bool(auc >= mde and mw_one_sided_p(y, s) < alpha)


def delong_single_auc_var(y, s):
    """AUROC and its DeLong variance for a single scorer, via the k=1 case of
    `_fast_delong`. Reused, not reimplemented: `accord/control/delong.py`.
    """
    y = np.asarray(y, dtype=float)
    order = (-y).argsort()
    m = int(y.sum())
    preds = np.asarray(s, dtype=float)[order][None, :]
    aucs, cov = _fast_delong(preds, m)
    # np.cov collapses to a 0-d scalar for a single scorer (k=1); `_fast_delong`
    # inherits that from np.cov(v01)/np.cov(v10), so guard both shapes.
    var = float(cov) if np.ndim(cov) == 0 else float(cov[0, 0])
    return float(aucs[0]), var


def rule_delong_single_one_sided(y, s, alpha=0.05):
    """One-sided DeLong z-test of H0: population AUROC = 0.5 against the
    one-sided alternative AUROC > 0.5, using the DeLong variance of the
    single AUROC estimate (no second scorer needed)."""
    auc, var = delong_single_auc_var(y, s)
    if var <= 0:
        return bool(auc > 0.5)
    z = (auc - 0.5) / np.sqrt(var)
    p = float(1.0 - _stats.norm.cdf(z))
    return p < alpha


def rule_delong_paired_vs_null(y, s, seed, alpha=0.05):
    """Paired DeLong test of the replicate's score against a fresh, iid
    N(0,1) null score evaluated on the SAME sample (paired, matching
    `delong_roc_test`'s design intent: two scorers on the identical items).
    The null score is uninformative by construction (independent of y), so
    its population AUROC is exactly 0.5; the test is one-sided in the
    direction that the real score beats the null.
    """
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    null = rng.normal(size=len(s))
    out = delong_roc_test(y, s, null)
    diff = out["auc_diff"]
    p_two = out["p"]
    p_one = float(p_two / 2.0) if diff > 0 else float(1.0 - p_two / 2.0)
    return p_one < alpha


def rule_bootstrap_lower_excludes_chance(y, s, seed, B=10_000, alpha=0.05, fast=True):
    """The one bootstrap-type rule: stratified percentile bootstrap (pinned
    reimplementation above), fires if the lower tail bound excludes 0.5.
    `fast=True` uses the vectorized equivalent (checked against the slow,
    pinned loop by the pin test) because this rule is run at 1,000 outer
    replicates per grid point, each with its own B=10,000 inner bootstrap.
    """
    boots = (stratified_bootstrap_auroc_fast if fast else stratified_bootstrap_auroc)(y, s, seed, B)
    lo, _ = percentile_ci(boots, alpha / 2 * 100, 100 - alpha / 2 * 100)
    return bool(lo > 0.5)


RULES_CHEAP = {
    "mw_one_sided": rule_mw_one_sided,
    "mw_two_sided": rule_mw_two_sided,
    "c98_rule": rule_c98,
    # delong_single_one_sided and delong_paired_vs_null need extra args and
    # are wired directly by the caller (power_curves.py), not through this
    # dict, because delong_paired_vs_null needs its own seed stream.
}


# ---------------------------------------------------------------------------
# Target-AUROC constructions (C101).
# ---------------------------------------------------------------------------

def label_mix_lambda(a0, target):
    """lambda = (A0 - target) / (A0 - 0.5). Each of the n_pos positive slots
    is drawn from the true positive-score pool with probability (1-lambda),
    else from the true negative-score pool (a mislabeled positive slot).
    Negative slots are always drawn from the true negative pool. Population
    AUROC of this construction is exact: P(X>Y) for the mixture X against Y
    is (1-lambda)*A0 + lambda*0.5 by linearity of P(X>Y) in the mixture
    weights over the two component distributions Pos and Neg (the second
    component, drawn from the same distribution as Y, contributes exactly
    0.5 by symmetry, ties broken at 0.5 as usual for continuous scores).
    """
    if a0 <= 0.5:
        raise ValueError("label_mix_lambda requires A0 > 0.5")
    lam = (a0 - target) / (a0 - 0.5)
    return float(lam)


def draw_label_mix_replicate(pos_pool, neg_pool, n_pos, n_neg, lam, rng):
    """One replicate under the label-mix construction. Returns (y, s)."""
    lam = min(max(lam, 0.0), 1.0)
    from_neg = rng.random(n_pos) < lam
    pos_slot_scores = np.where(
        from_neg,
        rng.choice(neg_pool, n_pos, replace=True),
        rng.choice(pos_pool, n_pos, replace=True),
    )
    neg_slot_scores = rng.choice(neg_pool, n_neg, replace=True)
    y = np.r_[np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)]
    s = np.r_[pos_slot_scores, neg_slot_scores]
    return y, s


def draw_noise_mix_replicate(pos_z, neg_z, n_pos, n_neg, sigma, rng):
    """One replicate under the noise-mix construction: resample from the
    z-scored empirical pools with replacement, then add iid N(0, sigma^2).
    """
    pos_slot_scores = rng.choice(pos_z, n_pos, replace=True) + rng.normal(0, sigma, n_pos)
    neg_slot_scores = rng.choice(neg_z, n_neg, replace=True) + rng.normal(0, sigma, n_neg)
    y = np.r_[np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)]
    s = np.r_[pos_slot_scores, neg_slot_scores]
    return y, s


def estimate_noise_mix_auroc(pos_z, neg_z, sigma, rng, n_mc=20_000):
    """Monte Carlo estimate of the population AUROC of the noise-mix
    construction at a given sigma, used only inside the bisection search
    that calibrates sigma to a target AUROC (not used per-replicate)."""
    a = rng.choice(pos_z, n_mc, replace=True) + rng.normal(0, sigma, n_mc)
    b = rng.choice(neg_z, n_mc, replace=True) + rng.normal(0, sigma, n_mc)
    y = np.r_[np.ones(n_mc, dtype=int), np.zeros(n_mc, dtype=int)]
    s = np.r_[a, b]
    return float(roc_auc_score(y, s))


def bisect_sigma_for_target(pos_z, neg_z, target, seed, tol=0.002, max_iter=40):
    """Bisection on sigma so the noise-mix population AUROC (Monte Carlo
    estimate, n_mc=20,000 each side, fixed seed per evaluation) lands within
    `tol` of `target`. AUROC is monotonically non-increasing in sigma
    (more noise -> less separation -> AUROC -> 0.5), so bisection applies."""
    rng = np.random.default_rng(seed)
    lo_sigma, hi_sigma = 0.0, 50.0
    a_lo = estimate_noise_mix_auroc(pos_z, neg_z, lo_sigma, rng)
    a_hi = estimate_noise_mix_auroc(pos_z, neg_z, hi_sigma, rng)
    if target >= a_lo:
        return 0.0, a_lo
    if target <= a_hi:
        return hi_sigma, a_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo_sigma + hi_sigma)
        a_mid = estimate_noise_mix_auroc(pos_z, neg_z, mid, rng)
        if abs(a_mid - target) <= tol:
            return mid, a_mid
        # AUROC decreases as sigma increases
        if a_mid > target:
            lo_sigma = mid
        else:
            hi_sigma = mid
    a_mid = estimate_noise_mix_auroc(pos_z, neg_z, 0.5 * (lo_sigma + hi_sigma), rng)
    return 0.5 * (lo_sigma + hi_sigma), a_mid


