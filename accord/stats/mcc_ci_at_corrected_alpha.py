"""Matthews correlation intervals for the two frontier zero-shot rows, read at the
Bonferroni-corrected level instead of at 0.05.

WHY THIS EXISTS
---------------
Chapter 4 applies its multiplicity correction to p values and not to the interval
estimates, and says so. To show what the asymmetry costs it reports Haiku's G_253
interval widened to the corrected level, where it admits zero. That number reached
the register as a typed literal with no generator behind it: the register recorded
it as derived at alpha 0.05/26 with B = 10,000 at seed 42, and no committed script
produced it. It could not be reproduced from the registered cells by the method the
register described, so it is withdrawn and regenerated here. See C88.

METHOD
------
The interval is a percentile bootstrap over the 253 evaluated pairs. Resampling
pairs with replacement is the same thing as drawing the four cell counts from a
multinomial at the observed cell proportions, which is how it is done here, so the
only inputs needed are the registered confusion cells. Alpha is read from
test_register.json rather than written here, so a test entering the family moves
this interval with it and cannot leave a stale bound behind in the prose.

Run:  python3 accord/stats/mcc_ci_at_corrected_alpha.py
Writes: results/accord/stats/mcc_ci_at_corrected_alpha.json
"""

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "accord" / "stats" / "mcc_ci_at_corrected_alpha.json"
REGISTER = ROOT / "results" / "accord" / "stats" / "test_register.json"

B = 10000
SEED = 42

ROWS = {
    "haiku_zero_shot_G253": "results/accord/differential/frontier_anthropic_metrics_claudehaiku4520251001.json",
    "sonnet_zero_shot_G253": "results/accord/differential/frontier_anthropic_metrics.json",
}


def mcc_vec(a, b, c, d):
    """a=tp b=fp c=fn d=tn, elementwise over bootstrap draws."""
    den = np.sqrt((a + b) * (a + c) * (d + b) * (d + c), dtype=float)
    return np.where(den == 0, 0.0, (a * d - b * c) / np.where(den == 0, 1.0, den))


def interval(cells, alpha, seed=SEED, B=B):
    tp, fp, fn, tn = cells
    n = tp + fp + fn + tn
    p = np.array([tp, fp, fn, tn], dtype=float) / n
    draws = np.random.default_rng(seed).multinomial(n, p, size=B)
    m = mcc_vec(draws[:, 0], draws[:, 1], draws[:, 2], draws[:, 3])
    lo, hi = np.percentile(m, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main():
    reg = json.loads(REGISTER.read_text())
    k = reg["n_tests"]
    alpha_corrected = reg["bonferroni_threshold"]

    out = {"provenance": {
        "generated_by": "accord/stats/mcc_ci_at_corrected_alpha.py",
        "method": ("percentile bootstrap over the evaluated pairs, implemented as a "
                   "multinomial draw on the observed confusion cells, which is the "
                   "same resampling distribution"),
        "B": B, "seed": SEED,
        "alpha_uncorrected": 0.05,
        "alpha_corrected": alpha_corrected,
        "n_tests_in_family": k,
        "alpha_source": "results/accord/stats/test_register.json::bonferroni_threshold"}}

    for name, path in ROWS.items():
        node = json.loads((ROOT / path).read_text())["by_strategy"]["zero_shot"]["G_253"]
        cells = (node["tp"], node["fp"], node["fn"], node["tn"])
        lo05, hi05 = interval(cells, 0.05)
        loc, hic = interval(cells, alpha_corrected)
        out[name] = {
            "cells": {"tp": cells[0], "fp": cells[1], "fn": cells[2], "tn": cells[3]},
            "mcc_point_as_reported": round(node["mcc"], 4),
            "reported_ci_at_0.05": [round(node["mcc_ci_lo"], 4), round(node["mcc_ci_hi"], 4)],
            "reproduced_ci_at_0.05": [round(lo05, 4), round(hi05, 4)],
            "ci_at_corrected_alpha": [round(loc, 4), round(hic, 4)],
            "admits_zero_at_corrected_alpha": bool(loc <= 0.0 <= hic),
            "admits_zero_at_0.05": bool(lo05 <= 0.0 <= hi05)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
