"""The minimum detectable AUROC under the STANDARD Hanley and McNeil condition,
and its register prefix.

WHY THIS FILE EXISTS, AND WHY IT DOES NOT SIMPLY EDIT recompute_mde.py.
accord/stats/recompute_mde.py solves a condition that is not a power
calculation. It divides the WHOLE effect by the standard error at the
ALTERNATIVE:

    (a - 0.5) / SE(a)  >=  z_{1-alpha/2} + z_{1-beta}

The standard Hanley and McNeil / Obuchowski condition splits the two terms. The
critical value of the test statistic is set under the null, so the significance
term takes the NULL standard error, and only the power term takes the
alternative's:

    a - 0.5  >=  z_{1-alpha/2} * SE(0.5)  +  z_{1-beta} * SE(a)

SE(a) < SE(0.5) for every a > 0.5, so the implemented route asks the effect to
clear a smaller quantity and every floor it returns is TOO LOW. This file
solves the standard condition and registers its output under the prefix
mdestandard, which is what the thesis now prints wherever it makes a claim
about the SENSITIVITY of the evaluation.

The mde_recomputed.* prefix is deliberately left alone. Its value on G_253,
0.7355, is a historical fact rather than a sensitivity claim: it is what the
procedure as implemented returned at the time the decision rule was registered,
and it is the number the pre-registered checks were judged against. Withdrawing
it would falsify the record of what was fixed in advance. The two prefixes
therefore answer two different questions, and the thesis cites whichever one the
sentence is actually about.

ONE ARITHMETIC FACT, recorded here because it is why SE(0.5) is not a second
approximation smuggled in beside the Hanley and McNeil variance. At a = 0.5 that
variance has Q1 = Q2 = 1/3, and it collapses EXACTLY to

    (n_+ + n_- + 1) / (12 n_+ n_-),

which is the simple normal approximation the methodology chapter already writes
out for SE_0. The two are the same number, not two conventions, so the corrected
condition mixes no conventions: se_null below is the Hanley and McNeil variance
evaluated at the null and nothing else. The identity is asserted on every
stratum in this file before anything is written; the residual is floating-point
rounding, about 1e-18 in absolute terms on these class counts.

    python3 accord/stats/mde_standard_condition.py

Writes:  results/accord/mde_standard_condition.json
Registers prefix: mdestandard
"""

import collections
import json
import math
import pathlib
import sys

from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accord.cloud_paths import assert_local_only  # noqa: E402

SRC = "results/accord/mde_standard_condition.json"
OUT = ROOT / SRC
FACTS = assert_local_only(ROOT / "results/accord/VERIFIED_FACTS.json")
PREFIX = "mdestandard"

ALPHA = 0.05
POWER = 0.80

# Aggregate class counts only. Nothing here reads a row, a note or an identifier.
STRATA = [
    # The four strata of Table tab:m-mde in the methodology chapter.
    {"name": "g253", "n_pos": 12, "n_neg": 241},
    {"name": "g299", "n_pos": 12, "n_neg": 287},
    {"name": "g134", "n_pos": 6, "n_neg": 128},
    {"name": "g108", "n_pos": 6, "n_neg": 102},
    # The counts the greedy tipping-point flips leave behind on G_253: each flip
    # relabels a confirmed positive negative, so the 253 pairs are conserved.
    {"name": "flip7_pos", "n_pos": 7, "n_neg": 246},
    {"name": "flip6_pos", "n_pos": 6, "n_neg": 247},
    {"name": "flip5_pos", "n_pos": 5, "n_neg": 248},
    # The opposite direction of error: one negative relabelled positive on G_253.
    {"name": "relabel13_pos", "n_pos": 13, "n_neg": 240},
]

# The Monte Carlo cross-check released with the thesis draws 5,000 synthetic
# score sets at 12 positives and 241 negatives and crosses 80 percent power at
# this value (register key injection_sensitivity.mde_empirical). The gap to the
# analytic floor is registered below rather than subtracted in prose.
MC_EMPIRICAL_G253 = 0.7324

FIELDS = ("n_pos", "n_neg", "se0_null_approx", "minimum_detectable_auc",
          "minimum_detectable_auc_as_implemented", "gap_above_chance",
          "correction_above_as_implemented")


def var_hanley(a, n_pos, n_neg):
    """Hanley and McNeil (1982) sampling variance of an AUROC estimate a."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return (a * (1 - a) + (n_pos - 1) * (q1 - a * a)
            + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)


def se0(n_pos, n_neg):
    """Simple normal approximation of the standard error under the null, as the
    methodology chapter writes it out. Equal to sqrt(var_hanley(0.5, ...))."""
    return math.sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg))


def bisect(holds, iters=200):
    """Smallest a in (0.5, 1) for which the condition holds. Monotone in a."""
    lo, hi = 0.5, 0.9999
    for _ in range(iters):
        a = (lo + hi) / 2
        if holds(a):
            hi = a
        else:
            lo = a
    return (lo + hi) / 2


def mde_standard(n_pos, n_neg, alpha=ALPHA, power=POWER):
    """a - 0.5 >= z_{1-alpha/2} * SE(0.5) + z_{1-beta} * SE(a)."""
    z_sig, z_pow = norm.ppf(1 - alpha / 2), norm.ppf(power)
    se_null = se0(n_pos, n_neg)
    return bisect(lambda a: (a - 0.5) >= z_sig * se_null
                  + z_pow * math.sqrt(max(var_hanley(a, n_pos, n_neg), 1e-12)))


def mde_as_implemented(n_pos, n_neg, alpha=ALPHA, power=POWER):
    """The defective route recompute_mde.py solves, kept here only so the
    artifact can show that the corrected floor is the higher of the two and so
    that the assertion below can never be reverted past silently."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return bisect(lambda a: (a - 0.5)
                  / math.sqrt(max(var_hanley(a, n_pos, n_neg), 1e-12)) >= z)


# The four rows of Table tab:m-mde in the methodology chapter, in the order the
# table prints them, with the label each row carries.
TABLE_ROWS = [("$G_{253}$", "g253"), ("$G_{299}$", "g299"),
              ("$G_{134}$", "g134"), ("$G_{108}$", "g108")]
CH3 = ROOT / "accord" / "thesis" / "msu_ch3_methodology.tex"


def table_rows(out):
    """The AUROC_min column of Table tab:m-mde, emitted from the artifact rather
    than typed. SE_0 comes from the same artifact and is unchanged by the
    correction, since it was always the null standard error."""
    rows = []
    for label, name in TABLE_ROWS:
        st = out["strata"][name]
        rows.append(f"{label} & {st['n_pos']} & {st['n_neg']} & "
                    f"{st['se0_null_approx']:.4f} & {st['minimum_detectable_auc']:.4f} \\\\")
    return rows


def check_table(out):
    """Fail if the committed chapter does not print exactly the emitted rows.
    Run after any edit to the table: a hand-typed digit in that column is the
    failure mode this whole correction came from."""
    text = CH3.read_text()
    missing = [r for r in table_rows(out) if r not in text]
    if missing:
        raise AssertionError(
            "msu_ch3_methodology.tex does not contain these generated rows of "
            "tab:m-mde:\n  " + "\n  ".join(missing))
    print(f"tab:m-mde AUROC_min column matches {SRC} on all "
          f"{len(TABLE_ROWS)} rows")


def main():
    out = {"alpha": ALPHA, "power": POWER,
           "condition": "a - 0.5 >= z_{1-alpha/2} * SE(0.5) + z_{1-beta} * SE(a)",
           "strata": collections.OrderedDict()}
    for s in STRATA:
        m, n = s["n_pos"], s["n_neg"]
        # The identity that makes the mixed condition legitimate, checked before
        # anything is written rather than asserted in a comment.
        # The two expressions are the same number in exact arithmetic, so the
        # only admissible residual is floating-point rounding. It lands at about
        # 1e-18 in absolute terms on these counts, which is why the comparison is
        # made on the variances (where SE_0 is squared back) at a relative
        # tolerance rather than on an absolute difference of standard errors.
        assert math.isclose(var_hanley(0.5, m, n), se0(m, n) ** 2,
                            rel_tol=1e-12, abs_tol=0.0), \
            f"Hanley and McNeil variance at the null does not equal SE_0^2 on {s['name']}"
        std = mde_standard(m, n)
        impl = mde_as_implemented(m, n)
        assert std > impl, \
            f"the corrected floor is not above the as-implemented one on {s['name']}"
        out["strata"][s["name"]] = collections.OrderedDict([
            ("n_pos", m),
            ("n_neg", n),
            ("se0_null_approx", round(se0(m, n), 4)),
            ("minimum_detectable_auc", round(std, 4)),
            ("minimum_detectable_auc_as_implemented", round(impl, 4)),
            ("gap_above_chance", round(std - 0.5, 4)),
            # How far the correction moves this stratum's floor. On G_253 this is
            # also the distance between the corrected floor and the registered
            # AUROC threshold, which was set from the defective route.
            # Differenced on the ROUNDED values, because those are the digits
            # the thesis prints and a reader subtracting them must land here.
            ("correction_above_as_implemented",
             round(round(std, 4) - round(impl, 4), 4)),
        ])
    g253 = out["strata"]["g253"]["minimum_detectable_auc"]
    out["monte_carlo_empirical_g253"] = MC_EMPIRICAL_G253
    out["monte_carlo_gap_g253"] = round(g253 - MC_EMPIRICAL_G253, 4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]
    for name, st in out["strata"].items():
        for f in FIELDS:
            facts[f"{PREFIX}.{name}.{f}"] = collections.OrderedDict(
                [("artifact", f"{SRC}::strata.{name}.{f}"), ("value", st[f])])
    for top in ("monte_carlo_empirical_g253", "monte_carlo_gap_g253"):
        facts[f"{PREFIX}.{top}"] = collections.OrderedDict(
            [("artifact", f"{SRC}::{top}"), ("value", out[top])])
    FACTS.write_text(json.dumps(facts, indent=1) + "\n")

    print(json.dumps(out, indent=2))
    print(f"registered {PREFIX}: {len(out['strata'])} strata")
    print("\ntab:m-mde rows (paste these, do not retype them):")
    for r in table_rows(out):
        print("  " + r)
    if "--check-table" in sys.argv:
        check_table(out)


if __name__ == "__main__":
    main()
