"""Instrument-level sensitivity check for the 253-pair contradiction evaluation.

The analytic minimum detectable AUROC (Hanley and McNeil normal approximation,
Section "Minimum detectable effect" of the methodology chapter) says the
253-pair instrument (12 positives, 241 negatives) cannot resolve a true AUROC
below 0.7382 at 80% power and alpha = 0.05. That is a formula, not a check run
on the instrument itself. This script runs the check: it injects a known true
AUROC into synthetic score sets of exactly this instrument's shape and asks
whether the instrument (a two-sided Mann-Whitney U test at alpha = 0.05, the
same test used throughout Chapter 4) actually detects it.

Pure simulation. No MIMIC data, no real scores, nothing from the trained
detector. The only inputs are n_positives = 12, n_negatives = 241, alpha =
0.05 and the true-AUROC grid; all four are configuration, not results.

Model. Equal-variance binormal: negatives ~ N(0, 1), positives ~ N(mu, 1),
with mu chosen so that the population AUROC equals the target value under
that model, mu = sqrt(2) * Phi^{-1}(AUROC). This is the standard binormal
construction for a target ROC area under equal variances.

For each true AUROC on the grid, draw 5,000 independent synthetic samples (12
positives, 241 negatives), run scipy.stats.mannwhitneyu (two-sided) on each,
and record the fraction of samples with p < alpha: the empirical power of
this exact instrument at that true AUROC. The interpolated AUROC at which
that empirical power curve crosses 0.80 is the instrument's own measured
detection threshold, compared directly against the analytic value.

Run:  python3 accord/stats/injection_sensitivity.py
Writes: results/accord/injection_sensitivity.json
"""

import json
import pathlib

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "accord" / "injection_sensitivity.json"
MDE_FILE = ROOT / "results" / "accord" / "mde_recomputed.json"

SEED = 42
N_POS = 12
N_NEG = 241
ALPHA = 0.05
N_REPS = 5000
GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
# Read rather than type: the analytic comparator is the full Hanley and
# McNeil (1982) value from accord/stats/recompute_mde.py's output, not the
# stored 0.7382 that script found unreproducible.
ANALYTIC_MDE = json.loads(MDE_FILE.read_text())["strata"]["gold_ambiguous_excluded"]["minimum_detectable_auc"]


def mu_for_auroc(auroc: float) -> float:
    """Equal-variance binormal mean shift giving population AUROC = auroc."""
    return np.sqrt(2.0) * stats.norm.ppf(auroc)


def empirical_power(auroc: float, rng: np.random.Generator) -> float:
    mu = mu_for_auroc(auroc)
    rejections = 0
    for _ in range(N_REPS):
        pos = rng.normal(loc=mu, scale=1.0, size=N_POS)
        neg = rng.normal(loc=0.0, scale=1.0, size=N_NEG)
        _, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        if p < ALPHA:
            rejections += 1
    return rejections / N_REPS


def interpolate_crossing(grid, power, target=0.80):
    """Linear interpolation of the AUROC at which power first reaches target."""
    for i in range(1, len(grid)):
        if power[i - 1] < target <= power[i]:
            x0, x1 = grid[i - 1], grid[i]
            y0, y1 = power[i - 1], power[i]
            frac = (target - y0) / (y1 - y0)
            return round(x0 + frac * (x1 - x0), 4)
    if power[-1] < target:
        return None
    return grid[0]


def main():
    rng = np.random.default_rng(SEED)
    power_curve = {}
    for auroc in GRID:
        power_curve[f"{auroc:.2f}"] = round(empirical_power(auroc, rng), 4)

    crossing = interpolate_crossing(GRID, [power_curve[f"{a:.2f}"] for a in GRID])
    agrees = crossing is not None and abs(crossing - ANALYTIC_MDE) <= 0.02

    out = {
        "seed": SEED,
        "n_positives": N_POS,
        "n_negatives": N_NEG,
        "alpha": ALPHA,
        "n_reps_per_grid_point": N_REPS,
        "grid": GRID,
        "empirical_power": power_curve,
        "empirical_mde_interpolated": crossing,
        "analytic_mde_hanley_mcneil": ANALYTIC_MDE,
        "agrees_with_analytic_mde_within_0.02": agrees,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
