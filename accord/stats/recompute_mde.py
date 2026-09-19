"""Recompute the minimum detectable AUROC (Hanley and McNeil, 1982) for the
three ambiguity treatments in the methodology chapter's sensitivity table.

The thesis's stored value, 0.7382 for the 12-positive, 241-negative primary
evaluation set, does not reproduce under either route actually documented in
this project: not the simple normal-approximation formula written out in the
methodology chapter (SE0 = sqrt((n+ + n- + 1) / (12 n+ n-)), which evaluates
the standard error only at the null, a = 0.5), and not the full Hanley and
McNeil (1982) sampling-variance formula, which evaluates the standard error
at the candidate AUROC itself and has to be solved for by search because the
variance depends on the value being solved for. This script implements the
second, full procedure, the one the fig-power figure script
(accord/thesis/figures/fig_batch2.py, function hanley_mde) already uses, so
that the recomputed text values and the plotted marker agree.

Hanley and McNeil (1982) variance of an AUROC estimate a, with m positives
and n negatives:
    Q1 = a / (2 - a)
    Q2 = 2 a^2 / (1 + a)
    Var(a) = [a(1-a) + (m-1)(Q1 - a^2) + (n-1)(Q2 - a^2)] / (m n)

The minimum detectable AUROC at power (1 - beta) and two-sided alpha is the
smallest a > 0.5 such that (a - 0.5) / sqrt(Var(a)) >= z_(1-alpha/2) +
z_(1-beta), found here by bisection on a in [0.5, 0.9999].

Run:  python3 accord/stats/recompute_mde.py
Writes: results/accord/mde_recomputed.json
"""

import json
import math
import pathlib

from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "accord" / "mde_recomputed.json"

ALPHA = 0.05
POWER = 0.80

STRATA = [
    {"name": "gold_ambiguous_excluded", "n_pos": 12, "n_neg": 241},
    {"name": "gold_ambiguous_retained", "n_pos": 12, "n_neg": 287},
    {"name": "defect_free_subset", "n_pos": 6, "n_neg": 128},
    # C110/C112's Rules 3+4 stratum with ambiguous rows dropped (108 pairs). The
    # 134-pair floor above was being applied to it; committee round after C113.
    {"name": "rules34_ambiguous_dropped", "n_pos": 6, "n_neg": 102},
    # The Rules 1 and 2 stratum with ambiguous rows dropped (145 pairs), the
    # complement of the 108 above within the 253. The thesis prints its class
    # counts, and a printed count the register does not hold is not traceable.
    {"name": "rules12_ambiguous_dropped", "n_pos": 6, "n_neg": 139},
]

FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "mde_recomputed"
# n_pos and n_neg are registered too: the strata tables print the class counts,
# and a printed number the register does not hold fails the numeric gate.
FIELDS = ("gap_above_chance", "minimum_detectable_auc",
          "minimum_detectable_auc_null_variance", "se0_null_approx",
          "n_pos", "n_neg")


def se0(n_pos, n_neg):
    """Simple normal-approximation standard error at the null (a = 0.5), as
    written out in the methodology chapter. Kept for the table's SE0 column;
    not used to derive the minimum detectable AUROC below."""
    return math.sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg))


def hanley_mcneil_mde(n_pos, n_neg, alpha=ALPHA, power=POWER, iters=100):
    """Full Hanley and McNeil (1982) procedure: bisect on the candidate AUROC
    a, evaluating the sampling variance at a itself rather than at the null."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    lo, hi = 0.5, 0.9999
    for _ in range(iters):
        a = (lo + hi) / 2
        q1 = a / (2 - a)
        q2 = 2 * a * a / (1 + a)
        var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a)
               + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
        stat = (a - 0.5) / math.sqrt(max(var, 1e-12))
        if stat < z:
            lo = a
        else:
            hi = a
    return (lo + hi) / 2


def null_variance_mde(n_pos, n_neg, alpha=ALPHA, power=POWER):
    """The same threshold with the sampling variance held at its null value
    for both terms: 0.5 + (z_alpha + z_power) * SE0. It is reported beside the
    Hanley and McNeil value because the two differ in the third decimal, and a
    score that clears one by less than that difference has not cleared the
    other. The thesis's registered decision rule uses the Hanley and McNeil
    value; this one is a sensitivity reading, not a second rule."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return 0.5 + z * se0(n_pos, n_neg)


def main():
    out = {"alpha": ALPHA, "power": POWER, "strata": {}}
    for s in STRATA:
        n_pos, n_neg = s["n_pos"], s["n_neg"]
        mde = round(hanley_mcneil_mde(n_pos, n_neg), 4)
        out["strata"][s["name"]] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "se0_null_approx": round(se0(n_pos, n_neg), 4),
            "minimum_detectable_auc": mde,
            "minimum_detectable_auc_null_variance": round(null_variance_mde(n_pos, n_neg), 4),
            "gap_above_chance": round(mde - 0.5, 4),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    # Register from the artifact just written; the prefix was previously kept by hand.
    import collections
    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]
    for name, st in out["strata"].items():
        for f in FIELDS:
            facts[f"{PREFIX}.{name}.{f}"] = {"artifact": "results/accord/mde_recomputed.json", "value": st[f]}
    FACTS.write_text(json.dumps(facts, indent=1) + "\n")


if __name__ == "__main__":
    main()
