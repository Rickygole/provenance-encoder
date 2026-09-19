"""Exact power of the two-sided Fisher test behind the Rule 1/2 vs Rule 3/4 comparison.

Replaces exemplar.power_analysis.achieved_power_at_observed_effect_rule12_vs_rule34
(0.0576), which was exported from a notebook with no committed generator and no stated
method. Group sizes and positives are the gold counts already registered (165 with 6,
134 with 6). Power is computed exactly: the sum over every (x1, x2) outcome of the two
binomial probabilities at the true rates, over outcomes where scipy's two-sided Fisher
exact p is below alpha. Also reports the smallest absolute rate difference, above the
pooled base rate 12/299 in the Rule 1/2 arm, that reaches 80 percent power.

Public counts only; no credentialed data is read.

    python3 accord/stats/exemplar_power_exact.py
"""
import collections
import json
import pathlib

import numpy as np
from scipy.stats import binom, fisher_exact

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "accord" / "stats" / "exemplar_power_exact.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "exemplarpower"
N1, K1, N2, K2, ALPHA = 165, 6, 134, 6, 0.05


def reject_matrix(n1, n2, kmax1, kmax2):
    rej = np.zeros((kmax1 + 1, kmax2 + 1), dtype=bool)
    for x1 in range(kmax1 + 1):
        for x2 in range(kmax2 + 1):
            rej[x1, x2] = fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])[1] < ALPHA
    return rej


def power(p1, p2, rej):
    k1, k2 = rej.shape[0] - 1, rej.shape[1] - 1
    b1 = binom.pmf(np.arange(k1 + 1), N1, p1)
    b2 = binom.pmf(np.arange(k2 + 1), N2, p2)
    # Outcomes beyond the enumerated range carry negligible mass at these rates; the
    # captured mass is reported so that is checked rather than assumed.
    return float((np.outer(b1, b2) * rej).sum()), float(b1.sum() * b2.sum())


def main():
    kmax1, kmax2 = 60, 50
    rej = reject_matrix(N1, N2, kmax1, kmax2)
    p1, p2 = K1 / N1, K2 / N2
    pw, mass = power(p1, p2, rej)
    base = (K1 + K2) / (N1 + N2)
    mdd, grid = None, np.round(np.arange(0.001, 0.30, 0.001), 3)
    for d in grid:
        pw_d, m_d = power(base + d, base, rej)
        if pw_d >= 0.80:
            mdd = float(d)
            break
    out = collections.OrderedDict([
        ("method", "exact enumeration, two-sided Fisher exact test, alpha 0.05"),
        ("n1_rule12", N1), ("positives_rule12", K1), ("n2_rule34", N2), ("positives_rule34", K2),
        ("fisher_p_observed", float(fisher_exact([[K1, N1 - K1], [K2, N2 - K2]])[1])),
        ("power_at_observed_rates", pw), ("probability_mass_enumerated", mass),
        ("base_rate", base),
        ("min_detectable_abs_diff_80pct_rule12_above_base", mdd),
    ])
    # Stored at 12 significant digits. The 16th and 17th digits of these floats
    # move between SciPy builds, which made every rerun a diff in the artifact
    # and the register with no change in anything the thesis prints.
    for k, v in out.items():
        if isinstance(v, float):
            out[k] = float(f"{v:.12g}")
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    # Keys this run no longer produces are dropped; the rest are assigned in
    # place, so a rerun leaves the register's key order as it found it.
    for k in [k for k in facts if k.split(".")[0] == PREFIX and k.split(".", 1)[1] not in out]:
        del facts[k]
    for k, v in out.items():
        facts[f"{PREFIX}.{k}"] = {"artifact": f"results/accord/stats/exemplar_power_exact.json::{k}", "value": v}
    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
