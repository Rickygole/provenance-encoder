"""
Sign-provenance sweep support script (see results/accord/sign_provenance_audit.json,
findings F1/F2).

Recomputes OLS drift slopes (metric vs selection-strength parameter) using the
SIGNED per-point gap values already stored in existing, on-disk artifacts
(selection_experiment_results_signed.json, pums_replication_results_signed.json),
and compares them against the ABS-valued drift slopes ALSO stored in those same
artifacts under `drift_check` / `regimeA_drift` / `regimeB_drift`.

Those two fields (accord/selection_experiment/run_experiment.py STEP 5,
accord/pums_replication/run_experiment.py STEP 6) fit linregress() against the
unsigned demographic_parity_gap / tpr_gap / fpr_gap / ppv_gap /
equalized_odds_gap / calibration_gap_diff columns -- the SAME sign-destroying
pattern C52/C56 already retracted for other quantities in this project -- even
though the C52/C56 retrofit (commit def441c) added *_signed companion columns
to the same sweep rows in the same commit. This script fits the identical OLS
regression on the *_signed columns instead and reports where the sign (or the
CI-credibility call) differs from what is currently stored.

No new experiments are run. No raw MIMIC/PUMS rows are read -- only the
already-released, already-aggregated sweep tables in results/accord/.

Run: python3 sign_provenance_drift_recheck.py
"""
import json
from pathlib import Path

from scipy.stats import linregress, t as t_dist

ROOT = Path("/Users/rickygole/provenance-encoder")
SEL = json.loads((ROOT / "results/accord/selection/selection_experiment_results_signed.json").read_text())
PUMS = json.loads((ROOT / "results/accord/pums/pums_replication_results_signed.json").read_text())


def slope_ci(xs, ys):
    res = linregress(xs, ys)
    df = len(xs) - 2
    tcrit = t_dist.ppf(0.975, df)
    lo, hi = res.slope - tcrit * res.stderr, res.slope + tcrit * res.stderr
    return dict(slope=float(res.slope), ci=[float(lo), float(hi)],
                credible=bool(not (lo <= 0 <= hi)), r2=float(res.rvalue ** 2))


def compare(stored_abs_slope, xs, abs_vals, signed_vals):
    signed_fit = slope_ci(xs, signed_vals)
    abs_fit = slope_ci(xs, abs_vals)
    return dict(
        stored_abs_slope=stored_abs_slope,
        recomputed_abs_slope=abs_fit["slope"],
        recomputed_signed_slope=signed_fit["slope"],
        recomputed_signed_ci=signed_fit["ci"],
        recomputed_signed_credible=signed_fit["credible"],
        recomputed_abs_credible=abs_fit["credible"],
        sign_flip_between_abs_and_signed=(abs_fit["slope"] * signed_fit["slope"] < 0),
        credibility_call_differs=(abs_fit["credible"] != signed_fit["credible"]),
    )


out = {}

# ---------------------------------------------------------------------
# E9 (selection_experiment, MIMIC): drift_check vs beta
# ---------------------------------------------------------------------
rows = SEL["sweep"]
betas = [r["beta"] for r in rows]
e9 = {}
for key in ["demographic_parity_gap", "tpr_gap", "fpr_gap", "ppv_gap",
            "equalized_odds_gap", "calibration_gap_diff"]:
    skey = key + "_signed"
    if skey not in rows[0]:
        continue
    abs_stored = SEL["drift_check"].get(key)
    e9[key] = compare(
        abs_stored["slope"] if abs_stored else None,
        betas, [r[key] for r in rows], [r[skey] for r in rows],
    )
out["E9_MIMIC_drift_vs_beta"] = e9

# ---------------------------------------------------------------------
# PUMS Regime A: drift vs OR (selection depends on group label only)
# ---------------------------------------------------------------------
rowsA = PUMS["regimeA_sweep"]
ors = [r["OR"] for r in rowsA]
regA = {}
for key in ["demographic_parity_gap", "ppv_gap", "calibration_gap_diff"]:
    skey = key + "_signed"
    if skey not in rowsA[0]:
        continue
    abs_stored = PUMS["regimeA_drift"].get(key)
    regA[key] = compare(
        abs_stored["slope"] if abs_stored else None,
        ors, [r[key] for r in rowsA], [r[skey] for r in rowsA],
    )
out["PUMS_RegimeA_drift_vs_OR"] = regA

# ---------------------------------------------------------------------
# PUMS Regime B: drift vs beta (selection depends on WKHP feature)
# ---------------------------------------------------------------------
rowsB = PUMS["regimeB_sweep"]
betasB = [r["beta"] for r in rowsB]
regB = {}
for key in ["demographic_parity_gap", "tpr_gap", "fpr_gap", "ppv_gap",
            "equalized_odds_gap", "calibration_gap_diff"]:
    skey = key + "_signed"
    if skey not in rowsB[0]:
        continue
    abs_stored = PUMS["regimeB_drift"].get(key)
    regB[key] = compare(
        abs_stored["slope"] if abs_stored else None,
        betasB, [r[key] for r in rowsB], [r[skey] for r in rowsB],
    )
out["PUMS_RegimeB_drift_vs_beta"] = regB

print(json.dumps(out, indent=2))

out_path = ROOT / "results/accord/sign_provenance_drift_recomputation.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nwrote {out_path}")
