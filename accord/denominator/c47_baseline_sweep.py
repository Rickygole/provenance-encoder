"""
C47 test: is the audit's sign response a mechanism, or a coincidence of two datasets?

C47 claims the sign of a pool fairness audit's response to upstream selection is
determined by the population's PRE-EXISTING disparity: on a population with no
disparity the audit invents one, on a population with a real one it erases one.

The entire evidence for that is TWO POINTS -- E9 on MIMIC (baseline DP gap 0.0084,
rises to 0.0895) and E10 on PUMS (baseline 0.1181, falls to 0.0988). MIMIC and PUMS
differ in far more than baseline disparity, so the mechanism is currently *consistent
with* the data rather than identified by it. This sweeps baseline disparity as a free
parameter in the committed simulator, which turns n=2 into a curve and makes the
claim falsifiable.

------------------------------------------------------------------------------
PRE-REGISTRATION -- written and committed BEFORE the run. Do not edit after.
------------------------------------------------------------------------------
The baseline knob for DEMOGRAPHIC PARITY is `rho` (the A->X mean shift), not
`beta_ay`. Predictions are a function of X alone, so a residual A->Y term moves
outcome-conditioned metrics (TPR/FPR/PPV) but cannot move the predicted-positive
rate gap. `rho` is the analogue of what separates PUMS (real feature-level group
difference) from MIMIC (almost none).

H1 -- C47's mechanism. Selection drives the pool DP gap toward a selection-induced
     value that does not depend on where the population started. Therefore the
     drift slope d(DP gap)/d(selection severity) is a DECREASING function of the
     baseline DP gap and CHANGES SIGN at some baseline rho*: positive below it
     (audit invents a gap), negative above it (audit erases one).

H0 -- No mechanism. The slope does not change sign anywhere over the swept range,
     or shows no monotone relationship to the baseline.

KILL CONDITION, binding: if the slope does not change sign anywhere over
rho in [0.0, 0.9], C47's mechanism is NOT supported by this test. The E9/E10 sign
difference then has to be attributed to some other difference between the two
datasets, and C47 must be rewritten as an observation about two datasets rather
than a mechanism -- exactly the demotion C41 should have received.

SECONDARY, and the honest framing if H1 holds: a sign change at a crossover
baseline IS "regression toward a selection-induced fixed point." That is a more
boring mechanism than the "invents/erases" language, and it is also more
defensible and more predictive, because it names the crossover as a quantity that
can be estimated and checked. If H1 holds, C47 should be restated in fixed-point
terms and the rhetorical framing dropped.
------------------------------------------------------------------------------

Simulation code is not duplicated here. This module execs the definition prefix of
simulate_v2.py verbatim, so the two can never drift apart -- the failure mode
correction C48 records for the figure code.
"""

import json
import pathlib
import statistics as st

import numpy as np

HERE = pathlib.Path(__file__).parent
SRC = HERE / "simulate_v2.py"

# Everything above `t_start = time.time()` is constants + function definitions;
# everything below it is simulate_v2's own top-level sweeps, which we must not run.
_lines = SRC.read_text().splitlines(keepends=True)
_cut = next(i for i, l in enumerate(_lines) if l.startswith("t_start"))
exec(compile("".join(_lines[:_cut]), str(SRC), "exec"), globals())
print(f"loaded {_cut} lines of definitions from simulate_v2.py (no sweeps executed)")

# ---------------------------------------------------------------------------
# The two axes.
#   rho        -- baseline population disparity, the thing C47 says sets the sign
#   leak_gamma -- selection severity, the analogue of beta in the E9/E10 sweeps
#                 (selection acting through an outcome-predictive channel, which
#                  is the regime where E9/E10 both showed movement)
# ---------------------------------------------------------------------------
RHO_GRID = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
GAMMA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
N_REP = 40

BASE = dict(or_select=1.0, sx_coef=0.0, beta_ay=0.0)


def population_dp_gap(rho, n=400_000, seed=20260809):
    """DP gap in the population BEFORE any selection -- the baseline C47 is about."""
    rng = np.random.default_rng(seed)
    a = rng.binomial(1, P_A, n)
    x = rng.normal(0, 1, n) + rho * a
    pred = (CLF.predict_proba(x.reshape(-1, 1))[:, 1] >= THRESHOLD).astype(int)
    return float(pred[a == 1].mean() - pred[a == 0].mean())


def slope(xs, ys):
    """OLS slope of ys on xs."""
    mx, my = st.mean(xs), st.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


rows, summary = [], []
for rho in RHO_GRID:
    base_gap = population_dp_gap(rho)
    gaps = []
    for g in GAMMA_GRID:
        agg = run_point(f"c47_rho{rho}_g{g}", leak_gamma=g, rho=rho,
                        n_replicates=N_REP, **BASE)
        pool = agg["pool"]
        gaps.append(pool["demographic_parity_gap"])
        rows.append({"rho": rho, "leak_gamma": g,
                     "population_dp_gap": base_gap,
                     "pool_dp_gap": pool["demographic_parity_gap"],
                     "four_fifths_ratio": pool["four_fifths_ratio"],
                     "breach": agg["breach_pool"],
                     "mde_dp": pool["mde_demographic_parity"]})
    s = slope(GAMMA_GRID, gaps)
    summary.append({"rho": rho, "population_dp_gap": base_gap,
                    "pool_dp_gap_at_min_selection": gaps[0],
                    "pool_dp_gap_at_max_selection": gaps[-1],
                    "drift_slope": s, "direction": "rises" if s > 0 else "falls"})
    print(f"  rho={rho:<5} baseline_DP={base_gap:.4f}  "
          f"{gaps[0]:.4f} -> {gaps[-1]:.4f}  slope={s:+.5f}  "
          f"{'RISES' if s > 0 else 'FALLS'}", flush=True)

# ---------------------------------------------------------------------------
# Adjudicate against the pre-registered kill condition.
# ---------------------------------------------------------------------------
signs = [1 if r["drift_slope"] > 0 else -1 for r in summary]
sign_changes = sum(1 for i in range(len(signs) - 1) if signs[i] != signs[i + 1])
h1_supported = sign_changes >= 1

crossover = None
if h1_supported:
    for i in range(len(summary) - 1):
        if signs[i] != signs[i + 1]:
            a, b = summary[i], summary[i + 1]
            w = abs(a["drift_slope"]) / (abs(a["drift_slope"]) + abs(b["drift_slope"]))
            crossover = a["population_dp_gap"] + w * (b["population_dp_gap"] - a["population_dp_gap"])
            break

# Monotone decreasing slope-vs-baseline is the other half of H1.
slope_vs_base = slope([r["population_dp_gap"] for r in summary],
                      [r["drift_slope"] for r in summary])

verdict = {
    "hypothesis": "H1 supported -- sign of audit drift is set by baseline disparity"
                  if h1_supported else
                  "KILL CONDITION MET -- no sign change; C47's mechanism is not supported",
    "h1_supported": h1_supported,
    "sign_changes": sign_changes,
    "crossover_population_dp_gap": crossover,
    "slope_of_drift_vs_baseline": slope_vs_base,
    "monotone_decreasing": slope_vs_base < 0,
    "n_replicates": N_REP,
    "rho_grid": RHO_GRID, "gamma_grid": GAMMA_GRID,
    "note": "If H1 holds this must be written as regression toward a selection-induced "
            "fixed point, with the crossover reported as an estimated quantity -- not as "
            "'the audit invents/erases' rhetoric. See the pre-registration above.",
}

out = pathlib.Path("/Users/rickygole/provenance-encoder/results/accord/c47_baseline_sweep.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"verdict": verdict, "summary": summary, "rows": rows}, indent=2))

print("\n" + "=" * 74)
print(verdict["hypothesis"])
print(f"  sign changes across the baseline grid : {sign_changes}")
print(f"  slope of drift vs baseline            : {slope_vs_base:+.5f} "
      f"({'decreasing, as predicted' if slope_vs_base < 0 else 'NOT decreasing'})")
if crossover is not None:
    print(f"  crossover baseline DP gap             : {crossover:.4f}")
    print(f"    E9  MIMIC baseline 0.0084 -> predicted {'RISE' if 0.0084 < crossover else 'FALL'} (observed RISE)")
    print(f"    E10 PUMS  baseline 0.1181 -> predicted {'RISE' if 0.1181 < crossover else 'FALL'} (observed FALL)")
print("=" * 74)
print("wrote", out)
