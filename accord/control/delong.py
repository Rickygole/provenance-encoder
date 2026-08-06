"""
DeLong's test for two possibly-correlated ROC curves evaluated on the SAME
sample (Sun & Xu, 2014, "Fast Implementation of DeLong's Algorithm..."),
needed here because ACCORD's contra_prob and the vanilla control's score are
both evaluated on the identical 299 gold pairs -- an ordinary two-sample
AUROC test would ignore that correlation and be wrong.

Standard fast-DeLong implementation (structural-component / covariance
approach); functionally the well-known public algorithm, not tied to any
particular scale (0.4-0.6 or the sign of the AUROC gap will work
identically).
"""
import numpy as np
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_T: np.ndarray, m: int):
    """preds_sorted_T: (k, n) array of k scorers' predictions, columns
    reordered so the m positive examples come first. Returns (aucs, cov)."""
    n_total = preds_sorted_T.shape[1]
    n = n_total - m
    k = preds_sorted_T.shape[0]
    pos = preds_sorted_T[:, :m]
    neg = preds_sorted_T[:, m:]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, n_total])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_T[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray):
    """Paired DeLong test for AUROC(score_a) vs AUROC(score_b) on the SAME
    y_true. Returns dict(auc_a, auc_b, auc_diff, z, p, ci95_diff)."""
    y_true = np.asarray(y_true, dtype=float)
    order = (-y_true).argsort()
    m = int(y_true.sum())
    preds = np.vstack([np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)])[:, order]

    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = np.sqrt(max(var_diff, 0.0))
    if se == 0:
        z, p = 0.0, 1.0
    else:
        z = diff / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_diff": float(diff),
        "se_diff": float(se),
        "z": float(z),
        "p": float(p),
        "ci95_diff": [float(ci_low), float(ci_high)],
    }
