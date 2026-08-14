"""Cohen's d for the anisotropy control, alongside the Welch t already
reported. A t statistic conflates effect size with sample size (n = 1,956 on
the clinical side), so the same comparison is also reported as a standardised
effect size that does not grow with n.

Pooled standard deviation (Cohen, 1988):
    s_pooled = sqrt(((n1-1) s1^2 + (n2-1) s2^2) / (n1+n2-2))
    d = (mean2 - mean1) / s_pooled

Inputs are the same registered means, standard deviations and counts the
Welch t-test in Chapter 4 already uses.

Run:  python3 accord/stats/recompute_anisotropy_effect_size.py
Writes: results/accord/stats/anisotropy_effect_size.json
"""

import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FACTS = json.loads((ROOT / "results" / "accord" / "VERIFIED_FACTS.json").read_text())
OUT = ROOT / "results" / "accord" / "stats" / "anisotropy_effect_size.json"


def v(key):
    return FACTS[key]["value"]


def main():
    n1 = v("aniso.anisotropy_control.n_pairs")
    s1 = v("aniso.anisotropy_control.std")
    n2 = v("aniso.homogeneity.results.head.matched.n")
    s2 = v("aniso.homogeneity.results.head.matched.std")
    diff = v("stats.anisotropy_welch.diff")

    pooled_sd = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    d = diff / pooled_sd

    out = {"n1_control": n1, "n2_clinical": n2, "mean_diff": diff,
           "pooled_sd": round(pooled_sd, 6), "cohens_d": round(d, 4)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
