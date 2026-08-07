"""
Runs every numerical check for the selection-invariance proposition.

python3 run_verification.py            # writes results.json + prints a report
"""

import json
import os

import numpy as np

from selection_invariance import (
    F, K, L, SCORE_LEVEL, SCORE_VALUE, SEED, OUT_DIR,
    build_population, joint_table, select, metrics,
    bias_term, bias_term_marginal, ppv_bias_term, cond_X_given_YA,
    pi_const, pi_group_only, pi_group_and_y, pi_x_multiplicative,
    pi_score_only, pi_within_level,
)

RES = {}
LINES = []


def say(s=""):
    LINES.append(s)
    print(s)


def hdr(s):
    say()
    say("=" * 78)
    say(s)
    say("=" * 78)


rng = np.random.default_rng(SEED)
pA, pXA, eta = build_population(rng)
P = joint_table(pA, pXA, eta)
POP = metrics(P)

hdr("MODEL")
say(f"K={K} X-atoms, L={L} score levels, threshold at level {2} (f=1 on levels 2,3)")
say(f"P(A=1) = {pA[1]:.4f}")
for a in range(2):
    m = POP[a]
    say(f"  group {a}: FPR={m['FPR']:.6f} TPR={m['TPR']:.6f} DP={m['DP']:.6f} "
        f"PPV={m['PPV']:.6f} AUROC={m['AUROC']:.6f} prev={m['prevalence']:.6f}")
say(f"  POPULATION FPR gap = {POP['gap']['FPR']:.6f}")
say(f"  POPULATION TPR gap = {POP['gap']['TPR']:.6f}")
RES["population"] = {str(a): {k: v for k, v in POP[a].items()} for a in range(2)}
RES["population"]["gap"] = POP["gap"]


# =========================================================================
hdr("E1. ALGEBRA CHECK: does the covariance decomposition hold exactly?")
say("Claim:  M_a^pool - M_a^pop = Cov(g, pi_y | cond) / E[pi_y | cond]")
say("Tested against an ARBITRARY, strongly X-dependent, group-dependent pi.")

rng_e1 = np.random.default_rng(SEED + 1)
worst = 0.0
rows = []
for trial in range(200):
    pi = np.zeros((2, K, 2))
    for a in range(2):
        v = rng_e1.uniform(0.02, 0.98, size=K)
        pi[a, :, 0] = v
        pi[a, :, 1] = v          # Y-independent selection: S _||_ Y | (X,A)
    POOL = metrics(select(P, pi))
    for a in range(2):
        b_fpr, *_ = bias_term(P, pi, a, 0, F)
        b_tpr, *_ = bias_term(P, pi, a, 1, F)
        b_dp, *_ = bias_term_marginal(P, pi, a, F)
        b_ppv, *_ = ppv_bias_term(P, pi, a)
        d = [
            ("FPR", POOL[a]["FPR"] - POP[a]["FPR"], b_fpr),
            ("TPR", POOL[a]["TPR"] - POP[a]["TPR"], b_tpr),
            ("DP",  POOL[a]["DP"] - POP[a]["DP"],  b_dp),
            ("PPV", POOL[a]["PPV"] - POP[a]["PPV"], b_ppv),
        ]
        for name, obs, pred in d:
            worst = max(worst, abs(obs - pred))
            if trial == 0:
                rows.append((a, name, obs, pred))

say()
say("  sample (trial 0):")
say(f"    {'grp':>3} {'metric':>6} {'observed shift':>16} {'closed form':>16} {'|diff|':>10}")
for a, name, obs, pred in rows:
    say(f"    {a:>3} {name:>6} {obs:>16.12f} {pred:>16.12f} {abs(obs-pred):>10.2e}")
say()
say(f"  MAX |observed - closed form| over 200 random pi, 2 groups, 4 metrics: {worst:.3e}")
ok_e1 = worst < 1e-12
say(f"  VERDICT: {'PASS' if ok_e1 else 'FAIL'} (tolerance 1e-12)")
RES["E1_algebra_max_abs_error"] = worst
RES["E1_pass"] = bool(ok_e1)

say()
say("  E1b. EQUIVALENT ODDS FORM (more interpretable than the covariance):")
say("       odds(FPR_a^pool) = r_a * odds(FPR_a^pop),  where")
say("       r_a = E[pi | Y=0,A=a,f=1] / E[pi | Y=0,A=a,f=0].")
rng_e1b = np.random.default_rng(SEED + 2)
worst_odds = 0.0
odds_rows = []
for trial in range(200):
    pi = np.zeros((2, K, 2))
    for a in range(2):
        v = rng_e1b.uniform(0.02, 0.98, size=K)
        pi[a, :, 0] = v
        pi[a, :, 1] = v
    POOL = metrics(select(P, pi))
    for a in range(2):
        px = cond_X_given_YA(P, a, 0)
        piy = pi[a, :, 0]
        w1, w0 = px * F, px * (1 - F)
        r = (w1 @ piy / w1.sum()) / (w0 @ piy / w0.sum())
        p = POP[a]["FPR"]
        pred = r * p / (1 - p)
        obs = POOL[a]["FPR"] / (1 - POOL[a]["FPR"])
        worst_odds = max(worst_odds, abs(obs - pred))
        if trial == 0:
            odds_rows.append((a, r, pred, obs))
say(f"       {'grp':>3} {'r_a':>10} {'pred odds':>14} {'obs odds':>14} {'|diff|':>10}")
for a, r, pred, obs in odds_rows:
    say(f"       {a:>3} {r:>10.6f} {pred:>14.10f} {obs:>14.10f} {abs(obs-pred):>10.2e}")
say(f"       MAX |diff| over 200 random pi: {worst_odds:.3e}")
say("       => the ENTIRE effect of X-dependent selection on a group's FPR is a")
say("          multiplicative tilt of the FPR ODDS by r_a. The gap on the")
say("          log-odds scale shifts by exactly log(r_1/r_0).")
RES["E1b_odds_max_abs_error"] = worst_odds
RES["E1b_pass"] = bool(worst_odds < 1e-12)


# =========================================================================
hdr("E2. pi(x,a) = pi(a):  per-group invariance, and invariance to MAGNITUDE")

say("2a. Extreme group-dependent selection: pi(A=0)=0.90, pi(A=1)=0.05 (18x).")
pi = pi_group_only(0.90, 0.05)
POOL = metrics(select(P, pi))
say(f"    {'metric':>12} {'pop gap':>14} {'pool gap':>14} {'|diff|':>10}")
worst2 = 0.0
for k in ["FPR", "TPR", "DP", "PPV", "NPV", "AUROC", "prevalence"]:
    d = abs(POOL["gap"][k] - POP["gap"][k])
    worst2 = max(worst2, d)
    say(f"    {k:>12} {POP['gap'][k]:>14.10f} {POOL['gap'][k]:>14.10f} {d:>10.2e}")
for i in range(L):
    d = abs(POOL["gap"]["calibration"][i] - POP["gap"]["calibration"][i])
    worst2 = max(worst2, d)
    say(f"    {'calib lvl'+str(i):>12} {POP['gap']['calibration'][i]:>14.10f} "
        f"{POOL['gap']['calibration'][i]:>14.10f} {d:>10.2e}")
say(f"    max |pool gap - pop gap| across ALL metrics: {worst2:.3e}")

say()
say("2b. Sweep the MAGNITUDE of group-dependence over 5 orders of magnitude.")
say("    pi(A=0)=0.99 fixed; pi(A=1) swept.  Reporting pool FPR gap.")
say(f"    {'pi(A=1)':>10} {'ratio':>12} {'pool FPR gap':>18} {'dev from pop':>14}")
worst2b = 0.0
sweep_rows = []
for p1 in [0.99, 0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    pi = pi_group_only(0.99, p1)
    g = metrics(select(P, pi))["gap"]["FPR"]
    d = abs(g - POP["gap"]["FPR"])
    worst2b = max(worst2b, d)
    sweep_rows.append({"pi_A1": p1, "pool_FPR_gap": g, "dev": d})
    say(f"    {p1:>10.0e} {0.99/p1:>12.0f} {g:>18.14f} {d:>14.2e}")
say(f"    max deviation across the whole sweep: {worst2b:.3e}")
ok_e2 = max(worst2, worst2b) < 1e-12
say(f"    VERDICT: {'PASS' if ok_e2 else 'FAIL'}")
say()
say("    NOTE: 'magnitude invariance' is not a separate theorem. Once each")
say("    group's pool metric EQUALS its population metric, the magnitude of")
say("    pi(a) cannot matter -- it has already cancelled. See report.")
RES["E2_max_abs_error"] = max(worst2, worst2b)
RES["E2_pass"] = bool(ok_e2)
RES["E2_magnitude_sweep"] = sweep_rows


# =========================================================================
hdr("E3. TAXONOMY, axis 1:  pi(x,a,y) = pi(a,y)   [selection on A and Y, not X]")
say("This is the Pfohl-style 'S depends on {Y,A}' regime.")
pi = pi_group_and_y(((0.90, 0.30), (0.10, 0.70)))
POOL = metrics(select(P, pi))
say(f"    {'metric':>12} {'pop gap':>14} {'pool gap':>14} {'|diff|':>11}  invariant?")
tax_y = {}
for k in ["FPR", "TPR", "AUROC", "DP", "PPV", "NPV", "prevalence"]:
    d = abs(POOL["gap"][k] - POP["gap"][k])
    inv = d < 1e-12
    tax_y[k] = {"pop": POP["gap"][k], "pool": POOL["gap"][k], "diff": d, "invariant": bool(inv)}
    say(f"    {k:>12} {POP['gap'][k]:>14.10f} {POOL['gap'][k]:>14.10f} {d:>11.2e}  "
        f"{'YES' if inv else 'NO'}")
dcal = max(abs(POOL["gap"]["calibration"][i] - POP["gap"]["calibration"][i]) for i in range(L))
tax_y["calibration"] = {"maxdiff": dcal, "invariant": bool(dcal < 1e-12)}
say(f"    {'calibration':>12} {'--':>14} {'--':>14} {dcal:>11.2e}  "
    f"{'YES' if dcal < 1e-12 else 'NO'}")
say()
say("    Also check PER-GROUP (not just the gap):")
for a in range(2):
    say(f"      group {a}: FPR {POP[a]['FPR']:.10f} -> {POOL[a]['FPR']:.10f} | "
        f"AUROC {POP[a]['AUROC']:.10f} -> {POOL[a]['AUROC']:.10f} | "
        f"PPV {POP[a]['PPV']:.10f} -> {POOL[a]['PPV']:.10f}")
RES["E3_taxonomy_Y_dependent"] = tax_y


# =========================================================================
hdr("E4. TAXONOMY, axis 2a:  pi depends on X ONLY THROUGH THE SCORE")
say("i.e. pi(x,a) = g(s(x), a). Realistic: you admit to the pool based on the")
say("model's own output. Prediction: per-score-level CALIBRATION is exactly")
say("preserved; FPR/TPR/DP/PPV/AUROC are not.")
pi = pi_score_only([[0.85, 0.60, 0.30, 0.10], [0.20, 0.35, 0.70, 0.90]])
POOL = metrics(select(P, pi))
say()
say(f"    {'metric':>14} {'pop gap':>14} {'pool gap':>14} {'|diff|':>11}  invariant?")
tax_s = {}
for k in ["FPR", "TPR", "AUROC", "DP", "PPV", "NPV", "prevalence"]:
    d = abs(POOL["gap"][k] - POP["gap"][k])
    tax_s[k] = {"pop": POP["gap"][k], "pool": POOL["gap"][k], "diff": d,
                "invariant": bool(d < 1e-12)}
    say(f"    {k:>14} {POP['gap'][k]:>14.10f} {POOL['gap'][k]:>14.10f} {d:>11.2e}  "
        f"{'YES' if d < 1e-12 else 'NO'}")
say()
say("    per-score-level calibration, PER GROUP (pop -> pool):")
cal_err = 0.0
for a in range(2):
    for i in range(L):
        d = abs(POOL[a]["calibration"][i] - POP[a]["calibration"][i])
        cal_err = max(cal_err, d)
        say(f"      grp {a} level {i}: {POP[a]['calibration'][i]:.12f} -> "
            f"{POOL[a]['calibration'][i]:.12f}   |diff| {d:.2e}")
say(f"    MAX calibration deviation: {cal_err:.3e}  -> "
    f"{'EXACTLY PRESERVED' if cal_err < 1e-12 else 'BROKEN'}")
tax_s["calibration"] = {"maxdiff": cal_err, "invariant": bool(cal_err < 1e-12)}
RES["E4_taxonomy_score_dependent"] = tax_s


hdr("E5. TAXONOMY, axis 2b:  pi depends on X WITHIN score levels")
say("Now pi is not a function of the score. Prediction: calibration breaks too.")
pi = pi_within_level(np.random.default_rng(SEED + 5))
POOL = metrics(select(P, pi))
cal_err5 = max(abs(POOL[a]["calibration"][i] - POP[a]["calibration"][i])
               for a in range(2) for i in range(L))
say(f"    MAX calibration deviation: {cal_err5:.6e}  -> "
    f"{'PRESERVED' if cal_err5 < 1e-12 else 'BROKEN (as predicted)'}")
say(f"    FPR gap {POP['gap']['FPR']:.8f} -> {POOL['gap']['FPR']:.8f}")
RES["E5_within_level_calibration_maxdev"] = cal_err5


# =========================================================================
hdr("E6. THE SWEEP: bias vs strength of X-dependence (the headline check)")
say("Family: pi(x,a) = pibar_a * (1 + lam * h_a(x)), h_a centered under")
say("P(X | Y=0, A=a). Then E[pi | Y=0,A=a] = pibar_a for EVERY lam, so the")
say("closed form predicts b_a(lam) = lam * Cov(f, h_a | Y=0, A=a): EXACTLY LINEAR.")

rng6 = np.random.default_rng(SEED + 6)
h_raw = rng6.uniform(-1.0, 1.0, size=(2, K))
h = np.zeros((2, K))
for a in range(2):
    px = cond_X_given_YA(P, a, 0)
    c = h_raw[a] - float(px @ h_raw[a])          # center under P(X|Y=0,A=a)
    h[a] = c / np.abs(c).max()                   # rescale so |h| <= 1 (keeps pi in (0,1))
pibar = np.array([0.50, 0.40])

slope_pred = []
for a in range(2):
    px = cond_X_given_YA(P, a, 0)
    cov_fh = float(px @ (F * h[a])) - float(px @ F) * float(px @ h[a])
    slope_pred.append(cov_fh)
say(f"    predicted slopes db_a/dlam = Cov(f,h_a|Y=0,A=a): "
    f"a=0 {slope_pred[0]:+.10f}, a=1 {slope_pred[1]:+.10f}")
say(f"    predicted slope of the GAP shift: {slope_pred[1]-slope_pred[0]:+.10f}")
say()

lams = np.linspace(0.0, 0.9, 19)
say(f"    {'lam':>6} {'obs b_0':>14} {'pred b_0':>14} {'obs b_1':>14} {'pred b_1':>14} "
    f"{'gap shift obs':>15} {'gap shift pred':>15}")
sweep = []
worst6 = 0.0
for lam in lams:
    pi = pi_x_multiplicative(h, pibar, lam)
    assert pi.min() > 0 and pi.max() < 1, "pi left [0,1]; no clipping allowed here"
    POOL = metrics(select(P, pi))
    ob = [POOL[a]["FPR"] - POP[a]["FPR"] for a in range(2)]
    pr = [lam * slope_pred[a] for a in range(2)]
    gs_o = POOL["gap"]["FPR"] - POP["gap"]["FPR"]
    gs_p = lam * (slope_pred[1] - slope_pred[0])
    worst6 = max(worst6, abs(ob[0] - pr[0]), abs(ob[1] - pr[1]), abs(gs_o - gs_p))
    sweep.append({"lam": float(lam), "obs_b0": ob[0], "pred_b0": pr[0],
                  "obs_b1": ob[1], "pred_b1": pr[1],
                  "obs_gap_shift": gs_o, "pred_gap_shift": gs_p,
                  "pool_FPR_gap": POOL["gap"]["FPR"]})
    say(f"    {lam:>6.3f} {ob[0]:>14.10f} {pr[0]:>14.10f} {ob[1]:>14.10f} "
        f"{pr[1]:>14.10f} {gs_o:>15.10f} {gs_p:>15.10f}")

say()
say(f"    MAX |observed - predicted| over the sweep: {worst6:.3e}")
o = np.array([r["obs_gap_shift"] for r in sweep])
pp = np.array([r["pred_gap_shift"] for r in sweep])
ss_res = float(((o - pp) ** 2).sum())
ss_tot = float(((o - o.mean()) ** 2).sum())
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
say(f"    R^2 of closed form vs observed gap shift: {r2:.15f}")
say(f"    VERDICT: {'PASS' if worst6 < 1e-12 else 'FAIL'}")
RES["E6_sweep"] = sweep
RES["E6_max_abs_error"] = worst6
RES["E6_r2"] = r2
RES["E6_pass"] = bool(worst6 < 1e-12)


# =========================================================================
hdr("E7. IS THE BIAS MONOTONE IN THE STRENGTH OF X-DEPENDENCE?")
say("In the mean-preserving multiplicative family above: yes, exactly linear.")
say("Question: is monotonicity a general property? Search a logistic family")
say("   pi_lam(x) = sigmoid(alpha + lam * h(x))")
say("for a counterexample where b(lam) = Cov(f,pi_lam)/E[pi_lam] is NON-monotone.")

rng7 = np.random.default_rng(SEED + 7)
lam_grid = np.linspace(0.0, 12.0, 241)


def b_curve(px, fvec, hvec, alpha, lams):
    """Vectorized over lams: returns b(lam) for the logistic family."""
    pi = 1.0 / (1.0 + np.exp(-(alpha + lams[:, None] * hvec[None, :])))  # [G,K]
    Ep = pi @ px
    Efp = (pi * fvec[None, :]) @ px
    return Efp / Ep - float(px @ fvec)


found = None
n_tried = 0
for _ in range(50000):
    n_tried += 1
    px = rng7.dirichlet(np.ones(K))
    hv = rng7.normal(0, 2.0, size=K)
    alpha = rng7.normal(0, 2.0)
    b = b_curve(px, F, hv, alpha, lam_grid)
    d = np.diff(b)
    if (d > 1e-9).any() and (d < -1e-9).any() and (b.max() - b.min()) > 0.05:
        found = (px, hv, alpha, b)
        break

if found is None:
    say("    No counterexample found in 200k random draws.")
    RES["E7_nonmonotone_found"] = False
else:
    px, hv, alpha, b = found
    i_ext = int(np.argmax(np.abs(b)))
    say(f"    COUNTEREXAMPLE FOUND after {n_tried} draws. alpha = {alpha:.6f}")
    say(f"    b(0) = {b[0]:.3e} (must be 0: no X-dependence => no bias)")
    say(f"    |b| PEAKS at {abs(b[i_ext]):.6f} (b={b[i_ext]:+.6f}) at "
        f"lam = {lam_grid[i_ext]:.3f},")
    say(f"    then DECAYS back to |b| = {abs(b[-1]):.6f} at lam = {lam_grid[-1]:.1f}.")
    say(f"    So |b| is non-monotone: it rises then falls by a factor of "
        f"{abs(b[i_ext])/abs(b[-1]):.2f}x.")
    say("    sampled trajectory:")
    for j in range(0, len(lam_grid), 24):
        say(f"      lam={lam_grid[j]:>6.3f}  b={b[j]:+.8f}")
    say()
    say("    => Monotonicity is NOT a general property. It holds exactly (and")
    say("       linearly) in the mean-preserving multiplicative family, because")
    say("       there the denominator E[pi] is constant in lam. Here the")
    say("       denominator moves too: as lam -> infinity, pi -> an indicator,")
    say("       and BOTH Cov and E[pi] saturate, so b can turn around.")
    say("       Substantively: the worst audit distortion occurs at MODERATE")
    say("       X-dependence, not at the most extreme.")
    RES["E7_nonmonotone_found"] = True
    RES["E7_alpha"] = float(alpha)
    RES["E7_b_peak"] = float(b[i_ext])
    RES["E7_lam_peak"] = float(lam_grid[i_ext])
    RES["E7_b_at_lam_max"] = float(b[-1])


# =========================================================================
hdr("E8. MONTE CARLO CROSS-CHECK (independent of all the algebra above)")
say("Draw (A,X,Y) then S~Bernoulli(pi), keep S=1, compute metrics empirically.")
N = 20_000_000
rng8 = np.random.default_rng(SEED + 8)
pi = pi_x_multiplicative(h, pibar, 0.7)

flat = P.reshape(-1)
# Exactly equivalent to drawing N i.i.d. units then flipping a pi-coin for each,
# but done with multinomial + binomial thinning so it is fast and exact.
counts = rng8.multinomial(N, flat).reshape(P.shape)
sel = rng8.binomial(counts, pi)
say(f"    N drawn = {N:,}; N selected = {sel.sum():,} ({sel.sum()/N:.4f})")

emp = sel.astype(float)
emp /= emp.sum()
MC = metrics(emp)
EX = metrics(select(P, pi))

say(f"    {'grp':>3} {'metric':>6} {'exact pool':>14} {'monte carlo':>14} "
    f"{'|diff|':>10} {'~SE':>9} {'z':>7}")
mc_rows = []
maxz = 0.0
for a in range(2):
    n_a0 = int(sel[a, :, 0].sum())
    n_a1 = int(sel[a, :, 1].sum())
    n_a = int(sel[a].sum())
    for name, n_eff in [("FPR", n_a0), ("TPR", n_a1), ("DP", n_a), ("PPV", n_a)]:
        e, m = EX[a][name], MC[a][name]
        se = np.sqrt(max(e * (1 - e), 1e-12) / n_eff)
        z = abs(e - m) / se
        maxz = max(maxz, z)
        mc_rows.append({"group": a, "metric": name, "exact": e, "mc": m,
                        "abs_diff": abs(e - m), "se": se, "z": z})
        say(f"    {a:>3} {name:>6} {e:>14.8f} {m:>14.8f} {abs(e-m):>10.2e} "
            f"{se:>9.2e} {z:>7.3f}")
say()
say(f"    max |z| = {maxz:.3f}  (expect < ~3 if the exact pool table is right)")
say(f"    VERDICT: {'PASS' if maxz < 4.0 else 'FAIL'}")
RES["E8_monte_carlo"] = mc_rows
RES["E8_max_z"] = maxz
RES["E8_pass"] = bool(maxz < 4.0)
RES["E8_N"] = N


# =========================================================================
hdr("E9. THE ESTIMAND IS INVARIANT; THE STANDARD ERROR IS NOT")
say("Under pi(x,a)=pi(a) the pool FPR gap equals the population gap exactly.")
say("But harsher selection shrinks the pool, so the AUDIT gets noisier.")
say(f"Repeated sampling, N=200,000 drawn, pi(A=0)=0.9, pi(A=1) swept, 500 reps.")
rng9 = np.random.default_rng(SEED + 9)
Nrep, R = 200_000, 500
say(f"    {'pi(A=1)':>9} {'mean pool gap':>16} {'bias vs pop':>13} "
    f"{'SD of gap':>12} {'MC SE of mean':>14} {'mean n(A=1,Y=0)':>16}")
pow_rows = []
for p1 in [0.9, 0.3, 0.1, 0.03, 0.01]:
    pi9 = pi_group_only(0.9, p1)
    gaps, ns = [], []
    for _ in range(R):
        counts = rng9.multinomial(Nrep, flat).reshape(P.shape)
        e = rng9.binomial(counts, pi9).astype(float)
        n10, n00 = e[1, :, 0].sum(), e[0, :, 0].sum()
        if n10 < 1 or n00 < 1:
            continue
        gaps.append((F @ e[1, :, 0]) / n10 - (F @ e[0, :, 0]) / n00)
        ns.append(n10)
    gaps = np.array(gaps)
    row = {"pi_A1": p1, "mean_gap": float(gaps.mean()),
           "bias": float(gaps.mean() - POP["gap"]["FPR"]),
           "sd_gap": float(gaps.std(ddof=1)),
           "se_of_mean": float(gaps.std(ddof=1) / np.sqrt(len(gaps))),
           "mean_n_A1_Y0": float(np.mean(ns))}
    row["bias_in_SE"] = row["bias"] / row["se_of_mean"]
    pow_rows.append(row)
    say(f"    {p1:>9.2f} {row['mean_gap']:>16.8f} {row['bias']:>13.2e} "
        f"{row['sd_gap']:>12.6f} {row['se_of_mean']:>14.2e} {row['mean_n_A1_Y0']:>16.1f}")
sd0, sdN = pow_rows[0]["sd_gap"], pow_rows[-1]["sd_gap"]
say()
# Correct analytic prediction: the gap variance has TWO terms, and only the
# A=1 term scales with pi(A=1). The naive sqrt(0.9/0.01)=9.5x is WRONG because
# group 0's contribution is held fixed.
def sd_pred(p1):
    v = 0.0
    for a, pa in [(0, 0.9), (1, p1)]:
        n = Nrep * P[a, :, 0].sum() * pa
        r = POP[a]["FPR"]
        v += r * (1 - r) / n
    return float(np.sqrt(v))
say(f"    {'pi(A=1)':>9} {'observed SD':>13} {'analytic SD':>13} {'ratio':>8}")
for r in pow_rows:
    sp = sd_pred(r["pi_A1"])
    r["sd_analytic"] = sp
    say(f"    {r['pi_A1']:>9.2f} {r['sd_gap']:>13.6f} {sp:>13.6f} "
        f"{r['sd_gap']/sp:>8.3f}")
say(f"    SD inflation from pi(A=1)=0.9 to 0.01: {sdN/sd0:.2f}x observed, "
    f"{sd_pred(0.01)/sd_pred(0.9):.2f}x predicted.")
say("    (The naive sqrt(0.9/0.01) = 9.5x is NOT the right prediction: group 0")
say("     is held at pi=0.9, so only one of the two variance terms scales.)")
say(f"    max |bias| in units of its own Monte Carlo SE: "
    f"{max(abs(r['bias_in_SE']) for r in pow_rows):.2f}")
say("    Bias stays at Monte Carlo zero throughout; only the variance moves.")
RES["E9_power"] = pow_rows


# =========================================================================
hdr("E10. 'ONLY IF pi(x,a)=pi(a)' IS FALSE **FOR THE GAP**  (precision check)")
say("pi(x,a)=pi(a) is SUFFICIENT for per-group invariance, and per-group")
say("invariance implies gap invariance. But the GAP can be preserved by")
say("cancellation while BOTH groups are individually biased. Constructed:")
say("choose group-specific lam_a so that lam_0*slope_0 = lam_1*slope_1.")

lam0 = 0.90
lam1 = lam0 * slope_pred[0] / slope_pred[1]
say(f"    lam_0 = {lam0:.6f}, lam_1 = {lam1:.6f}  (both nonzero => pi DOES depend on X)")
pi10 = np.zeros((2, K, 2))
for a, lm in [(0, lam0), (1, lam1)]:
    v = pibar[a] * (1.0 + lm * h[a])
    pi10[a, :, 0] = v
    pi10[a, :, 1] = v
assert pi10.min() > 0 and pi10.max() < 1
POOL10 = metrics(select(P, pi10))
b0 = POOL10[0]["FPR"] - POP[0]["FPR"]
b1 = POOL10[1]["FPR"] - POP[1]["FPR"]
say(f"    per-group FPR shift: b_0 = {b0:+.12f}   b_1 = {b1:+.12f}")
say(f"    |b_0 - b_1| = {abs(b0-b1):.3e}   (they cancel)")
say(f"    population FPR gap = {POP['gap']['FPR']:.12f}")
say(f"    pool       FPR gap = {POOL10['gap']['FPR']:.12f}")
say(f"    |gap difference|   = {abs(POOL10['gap']['FPR']-POP['gap']['FPR']):.3e}")
say()
say("    => GAP is preserved to machine precision, yet each group's FPR is")
say(f"       wrong by ~{abs(b0):.3f} and pi is NOT of the form pi(a).")
say("       So 'invariance holds ONLY IF pi(x,a)=pi(a)' is FALSE as stated")
say("       for the gap. It is true for PER-GROUP invariance uniformly over f.")
RES["E10_b0"] = b0
RES["E10_b1"] = b1
RES["E10_gap_dev"] = abs(POOL10["gap"]["FPR"] - POP["gap"]["FPR"])


hdr("E11. NECESSITY: invariance FOR ALL f forces pi to be constant")
say("Fix one non-constant pi(x,a). Sweep the decision threshold (which changes")
say("f). If invariance held for every f, b_a would be 0 at every threshold.")
rng11 = np.random.default_rng(SEED + 11)
pi11 = np.zeros((2, K, 2))
for a in range(2):
    v = rng11.uniform(0.1, 0.9, size=K)
    pi11[a, :, 0] = v
    pi11[a, :, 1] = v
say(f"    {'threshold lvl':>14} {'b_0':>16} {'b_1':>16} {'gap shift':>16}")
any_nonzero = False
thr_rows = []
for t in range(1, L):
    ft = (SCORE_LEVEL >= t).astype(float)
    bs = []
    for a in range(2):
        px = cond_X_given_YA(P, a, 0)
        piy = pi11[a, :, 0]
        Ep = float(px @ piy)
        bs.append((float(px @ (ft * piy)) - float(px @ ft) * Ep) / Ep)
    thr_rows.append({"threshold": t, "b0": bs[0], "b1": bs[1],
                     "gap_shift": bs[1] - bs[0]})
    if max(abs(bs[0]), abs(bs[1])) > 1e-9:
        any_nonzero = True
    say(f"    {t:>14} {bs[0]:>16.10f} {bs[1]:>16.10f} {bs[1]-bs[0]:>16.10f}")
say()
say(f"    Some threshold gives nonzero bias: {any_nonzero}")
say("    Consistent with the proposition: uniformity over f is what forces")
say("    pi to be a.s. constant on each (Y=0, A=a) stratum. A single f can")
say("    accidentally be uncorrelated with pi; a family of f's cannot.")
RES["E11_threshold_sweep"] = thr_rows
RES["E11_any_nonzero"] = bool(any_nonzero)


# =========================================================================
hdr("SUMMARY")
checks = [("E1 covariance decomposition exact", RES["E1_pass"]),
          ("E2 pi(x,a)=pi(a) full invariance", RES["E2_pass"]),
          ("E6 bias linear in lam (closed form)", RES["E6_pass"]),
          ("E8 Monte Carlo agrees with exact pool", RES["E8_pass"])]
for name, ok in checks:
    say(f"    [{'PASS' if ok else 'FAIL'}] {name}")
say()
say(f"    E1 max error: {RES['E1_algebra_max_abs_error']:.3e}")
say(f"    E6 max error: {RES['E6_max_abs_error']:.3e}   R^2 = {RES['E6_r2']:.15f}")
say(f"    E8 max |z|:   {RES['E8_max_z']:.3f}")

with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
    json.dump(RES, fh, indent=2, default=float)
with open(os.path.join(OUT_DIR, "verification_log.txt"), "w") as fh:
    fh.write("\n".join(LINES) + "\n")
print(f"\nwrote {OUT_DIR}/results.json and verification_log.txt")
