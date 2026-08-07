"""
Numerical verification of the selection-invariance proposition for fairness metrics.

Self-contained, synthetic, seed-fixed. No external data of any kind.

DESIGN NOTE (this is the whole point of the file):
Every metric is computed by ONE function, `metrics(joint)`, applied to a joint
probability table over (A, X, Y). The "population" metrics are that function on
the population table. The "pool" metrics are that SAME function on the
selection-reweighted table p(a,x,y) * pi(a,x,y), renormalized. Nothing about
the pool is computed from the closed-form formula being tested. So the
closed-form check is a genuine check, not a tautology.

Run:  python3 selection_invariance.py
"""

import itertools
import json
import os

import numpy as np

SEED = 20260807
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# Model: discrete so that everything below is EXACT (no Monte Carlo error).
# --------------------------------------------------------------------------
# X takes K atoms. The group-blind score s(x) takes L distinct values, with
# K/L atoms per level. Multiple atoms per score level is essential: it is what
# makes "selection through the score only" different from "selection on X".

K = 12          # number of X atoms
L = 4           # number of distinct score levels
PER_LEVEL = K // L
THRESH_LEVEL = 2  # f(x) = 1 iff score level index >= 2  (levels 2,3 positive)

SCORE_LEVEL = np.repeat(np.arange(L), PER_LEVEL)          # level index per atom
SCORE_VALUE = np.array([0.15, 0.35, 0.60, 0.85])[SCORE_LEVEL]  # group-blind score
F = (SCORE_LEVEL >= THRESH_LEVEL).astype(float)           # binary prediction


def build_population(rng):
    """Return P(A), P(X|A) [2,K], eta(x,a)=P(Y=1|X=x,A=a) [2,K]."""
    pA = np.array([0.55, 0.45])

    pX_given_A = np.zeros((2, K))
    for a in range(2):
        w = rng.gamma(shape=2.0, scale=1.0, size=K)
        pX_given_A[a] = w / w.sum()

    # eta is loosely increasing in the score but NOT a deterministic function of
    # the score level -- atoms within a level differ in true risk. That is what
    # lets "selection on X within a score level" break calibration while
    # "selection on the score" does not.
    eta = np.zeros((2, K))
    for a in range(2):
        base = 0.10 + 0.55 * (SCORE_LEVEL / (L - 1))
        jitter = rng.uniform(-0.07, 0.07, size=K)
        offset = 0.05 * a  # group 1 slightly higher risk at equal score
        eta[a] = np.clip(base + jitter + offset, 0.01, 0.99)
    return pA, pX_given_A, eta


def joint_table(pA, pX_given_A, eta):
    """Joint p[a, x, y] over the population."""
    p = np.zeros((2, K, 2))
    for a in range(2):
        for x in range(K):
            m = pA[a] * pX_given_A[a, x]
            p[a, x, 0] = m * (1.0 - eta[a, x])
            p[a, x, 1] = m * eta[a, x]
    return p


def select(p, pi):
    """Reweight the joint by selection probs pi[a,x,y], renormalize. This IS
    the law of (A,X,Y) given S=1."""
    q = p * pi
    return q / q.sum()


# --------------------------------------------------------------------------
# Metrics. All group-conditional. Computed from a joint table only.
# --------------------------------------------------------------------------

def _auroc(w1, w0):
    """Tie-corrected AUROC (Mann-Whitney) from unnormalized class weights over
    atoms. w1: weights of Y=1 atoms, w0: weights of Y=0 atoms."""
    n1, n0 = w1.sum(), w0.sum()
    if n1 <= 0 or n0 <= 0:
        return np.nan
    s1 = SCORE_VALUE[:, None]
    s0 = SCORE_VALUE[None, :]
    kernel = (s1 > s0).astype(float) + 0.5 * (s1 == s0).astype(float)
    return float(w1 @ kernel @ w0) / (n1 * n0)


def metrics(p):
    """All group-conditional metrics from joint p[a,x,y]. Returns dict."""
    out = {}
    for a in range(2):
        pa = p[a]                      # [K, 2]
        tot = pa.sum()
        w0, w1 = pa[:, 0], pa[:, 1]    # unnormalized Y=0 / Y=1 weights

        fpr = float(F @ w0 / w0.sum())
        tpr = float(F @ w1 / w1.sum())
        ppr = float(F @ (w0 + w1) / tot)                       # demographic parity
        pred_pos = float(F @ (w0 + w1))
        ppv = float(F @ w1 / pred_pos) if pred_pos > 0 else np.nan
        pred_neg = float((1 - F) @ (w0 + w1))
        npv = float((1 - F) @ w0 / pred_neg) if pred_neg > 0 else np.nan
        prev = float(w1.sum() / tot)
        auc = _auroc(w1, w0)

        calib = []
        for lev in range(L):
            m = SCORE_LEVEL == lev
            d = (w0[m] + w1[m]).sum()
            calib.append(float(w1[m].sum() / d) if d > 0 else np.nan)

        out[a] = dict(FPR=fpr, TPR=tpr, DP=ppr, PPV=ppv, NPV=npv,
                      prevalence=prev, AUROC=auc, calibration=calib)

    gaps = {}
    for k in ["FPR", "TPR", "DP", "PPV", "NPV", "AUROC", "prevalence"]:
        gaps[k] = out[1][k] - out[0][k]
    gaps["calibration"] = [out[1]["calibration"][i] - out[0]["calibration"][i]
                           for i in range(L)]
    out["gap"] = gaps
    return out


# --------------------------------------------------------------------------
# Closed-form bias term:  b_a = Cov(f, pi_y | Y=y, A=a) / E[pi_y | Y=y, A=a]
# where pi_y(x) = pi(a, x, y) is the Y=y SLICE of the selection function.
# --------------------------------------------------------------------------

def cond_X_given_YA(p, a, y):
    w = p[a, :, y]
    return w / w.sum()


def bias_term(p, pi, a, y, g):
    """Closed-form bias for a rate metric with integrand g(x), conditioned on
    (Y=y, A=a). For FPR: y=0, g=F. For TPR: y=1, g=F."""
    px = cond_X_given_YA(p, a, y)
    piy = pi[a, :, y]
    Ep = float(px @ piy)
    Eg = float(px @ g)
    Egp = float(px @ (g * piy))
    cov = Egp - Eg * Ep
    return cov / Ep, cov, Ep, Eg


def bias_term_marginal(p, pi, a, g):
    """Closed-form bias for a metric conditioned only on A=a (demographic
    parity). pi is averaged over Y within each x."""
    w = p[a].sum(axis=1)
    px = w / w.sum()
    # E[1{S=1} | X=x, A=a] = sum_y pi(a,x,y) P(Y=y | x, a)
    pi_bar = (pi[a] * p[a]).sum(axis=1) / p[a].sum(axis=1)
    Ep = float(px @ pi_bar)
    Eg = float(px @ g)
    cov = float(px @ (g * pi_bar)) - Eg * Ep
    return cov / Ep, cov, Ep, Eg


def ppv_bias_term(p, pi, a):
    """Closed-form bias for PPV:
       PPV^pool - PPV^pop = Cov(eta, pi_bar | f=1, A=a) / E[pi_bar | f=1, A=a]
    where eta(x) = P(Y=1|X=x,A=a) and pi_bar(x) = E[1{S=1}|X=x,A=a].
    NOTE: this identity requires pi to not depend on Y. Under Y-dependent
    selection the PPV shift is NOT a covariance in eta alone -- see report."""
    w = p[a].sum(axis=1)
    m = F > 0
    px = w[m] / w[m].sum()
    eta = p[a, :, 1][m] / w[m]
    pi_bar = ((pi[a] * p[a]).sum(axis=1) / w)[m]
    Ep = float(px @ pi_bar)
    cov = float(px @ (eta * pi_bar)) - float(px @ eta) * Ep
    return cov / Ep, cov, Ep


# --------------------------------------------------------------------------
# Selection-probability families
# --------------------------------------------------------------------------

def pi_const(c=0.4):
    return np.full((2, K, 2), float(c))


def pi_group_only(p0=0.9, p1=0.05):
    """S depends on A only. Deliberately EXTREME asymmetry (18x)."""
    pi = np.zeros((2, K, 2))
    pi[0] = p0
    pi[1] = p1
    return pi


def pi_group_and_y(p=((0.9, 0.3), (0.1, 0.7))):
    """S depends on (A, Y) but not X. p[a][y]."""
    pi = np.zeros((2, K, 2))
    for a in range(2):
        for y in range(2):
            pi[a, :, y] = p[a][y]
    return pi


def pi_x_multiplicative(h, pibar, lam):
    """pi(x,a) = pibar_a * (1 + lam * h[a,x]), NO Y-dependence.
    h is centered under P(X|Y=0,A=a) so that E[pi|Y=0,A=a] = pibar_a exactly,
    independent of lam. This makes the FPR bias EXACTLY LINEAR in lam."""
    pi = np.zeros((2, K, 2))
    for a in range(2):
        v = pibar[a] * (1.0 + lam * h[a])
        pi[a, :, 0] = v
        pi[a, :, 1] = v
    return pi


def pi_score_only(g_by_level):
    """S depends on X ONLY through the score level. g_by_level[a][level]."""
    pi = np.zeros((2, K, 2))
    for a in range(2):
        v = np.array([g_by_level[a][SCORE_LEVEL[x]] for x in range(K)])
        pi[a, :, 0] = v
        pi[a, :, 1] = v
    return pi


def pi_within_level(rng):
    """S depends on X in a way that varies WITHIN score levels (so it is not a
    function of the score). Used to break calibration."""
    pi = np.zeros((2, K, 2))
    for a in range(2):
        v = rng.uniform(0.15, 0.85, size=K)
        pi[a, :, 0] = v
        pi[a, :, 1] = v
    return pi


# --------------------------------------------------------------------------
def close(a, b, tol=1e-12):
    return abs(a - b) <= tol
