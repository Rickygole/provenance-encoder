"""The standard per-group fairness audit, plus the two things that are
usually missing from it: bootstrap CIs and a power layer (minimum detectable
gap), and the selection-stage four-fifths ratio nobody computes.

Operates entirely on arrays of (y, pred, score, group) -- aggregates only,
no text, no per-patient identifiers retained past the join.
"""
from __future__ import annotations

from math import sqrt

import numpy as np
from scipy.stats import norm

ALPHA = 0.05
POWER = 0.80
GAP_TOLERANCE = 0.05
FOUR_FIFTHS = 0.80
BOOTSTRAP_B = 5000


def group_rates(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict:
    n = len(y)
    if n == 0:
        return dict(n=0)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    n_pos, n_neg = tp + fn, fp + tn
    return dict(
        n=n, n_pos=n_pos, n_neg=n_neg, tp=tp, fp=fp, fn=fn, tn=tn,
        tpr=(tp / n_pos) if n_pos else float("nan"),
        fpr=(fp / n_neg) if n_neg else float("nan"),
        ppv=(tp / (tp + fp)) if (tp + fp) else float("nan"),
        pred_pos_rate=(tp + fp) / n,
        calibration_gap=float(score.mean() - y.mean()),
    )


def audit_pool(y, pred, score, group1_mask) -> dict:
    """Full per-group audit on one realized pool. group1_mask selects the
    "1" group (e.g. lang_missing==1); its complement is "0"."""
    g0 = group_rates(y[~group1_mask], pred[~group1_mask], score[~group1_mask])
    g1 = group_rates(y[group1_mask], pred[group1_mask], score[group1_mask])
    out = dict(g0=g0, g1=g1)
    out["demographic_parity_gap"] = abs(g1["pred_pos_rate"] - g0["pred_pos_rate"])
    out["tpr_gap"] = abs(g1["tpr"] - g0["tpr"])
    out["fpr_gap"] = abs(g1["fpr"] - g0["fpr"])
    out["ppv_gap"] = abs(g1["ppv"] - g0["ppv"])
    out["equalized_odds_gap"] = max(out["tpr_gap"], out["fpr_gap"])
    out["calibration_gap_diff"] = abs(g1["calibration_gap"] - g0["calibration_gap"])
    p0, p1 = g0["pred_pos_rate"], g1["pred_pos_rate"]
    out["four_fifths_ratio"] = (min(p0, p1) / max(p0, p1)) if max(p0, p1) > 0 else float("nan")
    return out


AUDIT_SCALAR_KEYS = [
    "demographic_parity_gap", "tpr_gap", "fpr_gap", "ppv_gap",
    "equalized_odds_gap", "calibration_gap_diff", "four_fifths_ratio",
]


def bootstrap_audit_ci(y, pred, score, group1_mask, B: int = BOOTSTRAP_B, seed: int = 0) -> dict:
    """Percentile bootstrap CI on every audit scalar, resampling rows of the
    realized pool with replacement (standard row-resample bootstrap, same
    convention as accord/stats/intervals.py)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = {k: [] for k in AUDIT_SCALAR_KEYS}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yi, pi, si, gi = y[idx], pred[idx], score[idx], group1_mask[idx]
        if gi.sum() == 0 or (~gi).sum() == 0:
            continue
        res = audit_pool(yi, pi, si, gi)
        for k in AUDIT_SCALAR_KEYS:
            draws[k].append(res[k])
    out = {}
    for k in AUDIT_SCALAR_KEYS:
        arr = np.array([v for v in draws[k] if not np.isnan(v)])
        if len(arr) == 0:
            out[k] = dict(ci_lo=float("nan"), ci_hi=float("nan"), n_valid_draws=0)
            continue
        out[k] = dict(
            ci_lo=float(np.percentile(arr, 2.5)),
            ci_hi=float(np.percentile(arr, 97.5)),
            n_valid_draws=int(len(arr)),
        )
    return out


def selection_stage_audit(group1_mask: np.ndarray, selected_mask: np.ndarray,
                           B: int = BOOTSTRAP_B, seed: int = 0) -> dict:
    """Four-fifths ratio on the SELECTION stage itself (rate of S=1 by group,
    pre-classifier) -- the statistic nobody in the standard workflow
    computes. Bootstrap CI by resampling rows of the full population."""
    r0 = selected_mask[~group1_mask].mean()
    r1 = selected_mask[group1_mask].mean()
    point = min(r0, r1) / max(r0, r1) if max(r0, r1) > 0 else float("nan")

    rng = np.random.default_rng(seed)
    n = len(group1_mask)
    draws = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        gi, si = group1_mask[idx], selected_mask[idx]
        if gi.sum() == 0 or (~gi).sum() == 0:
            continue
        rr0, rr1 = si[~gi].mean(), si[gi].mean()
        if max(rr0, rr1) == 0:
            continue
        draws.append(min(rr0, rr1) / max(rr0, rr1))
    arr = np.array(draws)
    return dict(
        selection_rate_group0=float(r0), selection_rate_group1=float(r1),
        selection_four_fifths_ratio=float(point),
        ci_lo=float(np.percentile(arr, 2.5)) if len(arr) else float("nan"),
        ci_hi=float(np.percentile(arr, 97.5)) if len(arr) else float("nan"),
        n_valid_draws=int(len(arr)),
    )


def mde_two_proportion(n0: int, n1: int, p0: float, p1: float,
                        alpha: float = ALPHA, power: float = POWER) -> float:
    """Minimum detectable absolute difference in two proportions, normal
    approximation, given the REALIZED group sizes and a pooled-rate variance
    estimate. Standard two-sample proportion-test power formula."""
    if n0 == 0 or n1 == 0:
        return float("nan")
    p_pool = (p0 * n0 + p1 * n1) / (n0 + n1)
    p_pool = min(max(p_pool, 1e-6), 1 - 1e-6)
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    se = sqrt(p_pool * (1 - p_pool) * (1 / n0 + 1 / n1))
    return float((z_alpha + z_power) * se)


def breach_call(point_estimate: float, ci_lo: float, ci_hi: float, kind: str) -> str:
    """kind in {'gap','four_fifths'}. Returns 'breach' / 'pass' / 'inconclusive'
    per the pre-registered CI criterion."""
    if kind == "gap":
        crosses = point_estimate > GAP_TOLERANCE
        ci_excludes_below = ci_lo > GAP_TOLERANCE
        if crosses and ci_excludes_below:
            return "breach"
        if (not crosses) and ci_hi <= GAP_TOLERANCE:
            return "pass"
        return "inconclusive"
    elif kind == "four_fifths":
        crosses = point_estimate < FOUR_FIFTHS
        ci_excludes_above = ci_hi < FOUR_FIFTHS
        if crosses and ci_excludes_above:
            return "breach"
        if (not crosses) and ci_lo >= FOUR_FIFTHS:
            return "pass"
        return "inconclusive"
    raise ValueError(kind)
