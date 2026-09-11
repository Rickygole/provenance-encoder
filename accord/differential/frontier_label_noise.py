"""How much of the frontier MCC interval is label noise, and does it matter.

WHY THIS EXISTS
---------------
The frontier labellers' per-pair labels are draws, not fixed values: Sonnet 5
does not accept a temperature parameter at all, and a hosted endpoint is not a
reproducibility guarantee even when it does. The committed bootstrap resamples
PAIRS with the labels held fixed, so it measures pair-sampling uncertainty and
nothing else. This script asks how unstable the labels would have to be before
the conclusion changes.

WHAT IT DOES
------------
It simulates the stated mechanism directly: resample the 253 pairs, then flip
each predicted label independently with probability p, and recompute MCC. No
linearisation, no analytic variance propagation.

An earlier version of this script did propagate variance analytically, holding
the point estimate fixed and adding only the binomial variance of the flip count
carried through a per-flip central difference. That construction is wrong, and
wrong in the direction that flatters the result: flipping labels does not only
widen the interval, it destroys the association, and the mean shift is the
dominant term. At p = 0.5 the old code reported [+0.0829, +0.5353] for a
configuration in which every label is a fair coin and MCC is necessarily
centred on zero. The correct answer at that point is [-0.1241, +0.1209] with a
mean of -0.0004. The old conclusion, that the zero-shot result "survives even
the fair-coin worst case", was an artifact of the missing term and is withdrawn.

Reads:  results/accord/differential/frontier_anthropic_metrics{TAG}.json
Writes: results/accord/differential/frontier_label_noise{TAG}.json
        and the labelnoise.* block of results/accord/VERIFIED_FACTS.json

The register block is rewritten here rather than maintained by hand. It was
maintained by hand, and when this script was corrected every labelnoise.* key in
the register stayed behind on the withdrawn linearised artifact, including
labelnoise.zero_shot.survives_any_label_noise = True, which is the claim the
correction withdrew. Keys from the old construction that this one does not
produce, the per-flip sensitivities and the arm-wise standard deviations, are
deleted rather than left to resolve; see CORRECTIONS.md C68.

    python3 accord/differential/frontier_label_noise.py
"""

import collections
import json
import math
import os

import numpy as np

FACTS = "results/accord/VERIFIED_FACTS.json"

TAG = os.environ.get("ACCORD_FRONTIER_MODEL_TAG", "")
IN = f"results/accord/differential/frontier_anthropic_metrics{TAG}.json"
OUT = f"results/accord/differential/frontier_label_noise{TAG}.json"
BASIS = "G_253"
B = 10000
SEED = 42
GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def mcc_cells(tp, fp, fn, tn):
    d = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


def simulate(y, yh, p, rng, B=B):
    """Pair bootstrap and independent label flips at rate p, jointly."""
    n = len(y)
    out = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        ys, yhs = y[i], yh[i].copy()
        flip = rng.random(n) < p
        yhs[flip] = 1 - yhs[flip]
        tp = int(((ys == 1) & (yhs == 1)).sum()); fp = int(((ys == 0) & (yhs == 1)).sum())
        fn = int(((ys == 1) & (yhs == 0)).sum()); tn = int(((ys == 0) & (yhs == 0)).sum())
        out[b] = mcc_cells(tp, fp, fn, tn)
    return out


# The untagged run is the one the chapters cite, so it owns the bare prefix. A
# tagged run writes under its own prefix instead of overwriting it.
PREFIX = "labelnoise" + (TAG or "")


def update_register(out):
    """Rewrite this script's block of the fact register from what it just wrote.

    Every key under PREFIX is dropped first, so a key the old construction
    produced and this one does not cannot survive as a resolvable number with no
    artifact behind it.
    """
    facts = json.load(open(FACTS), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    def put(key, value):
        facts[f"{PREFIX}.{key}"] = collections.OrderedDict(
            [("artifact", OUT), ("value", value)])

    n = 0
    for strat, r in out["by_strategy"].items():
        put(f"{strat}.mcc_point", r["mcc_point"])
        for arm in ("pair_sampling_sd_lower_arm", "pair_sampling_sd_upper_arm"):
            put(f"{strat}.{arm}", r[arm])
            n += 1
        put(f"{strat}.flip_probability_at_which_zero_enters",
            r["flip_probability_at_which_zero_enters"])
        put(f"{strat}.survives_any_label_noise", r["survives_any_label_noise"])
        n += 3
        for row in r["sweep"]:
            tag = f"p{int(round(row['p'] * 100)):02d}"
            for field in ("ci_lo", "ci_hi", "mean", "p_mcc_le_zero",
                          "excludes_zero"):
                put(f"{strat}.sweep.{tag}.{field}", row[field])
                n += 1
    json.dump(facts, open(FACTS, "w"), indent=1)
    return n


def main():
    src = json.load(open(IN))
    out = {"provenance": {
        "generated_by": "accord/differential/frontier_label_noise.py",
        "input": IN, "basis": BASIS, "B": B, "seed": SEED,
        "method": ("Direct simulation of the stated mechanism: resample pairs, "
                   "flip each predicted label independently with probability p, "
                   "recompute MCC. Both the mean shift and the added variance "
                   "are therefore carried. A previous version propagated "
                   "variance only and held the point estimate fixed, which "
                   "reported an interval far from zero at p = 0.5 where the "
                   "statistic is necessarily centred on zero."),
    }, "by_strategy": {}}

    for strat, sd in src["by_strategy"].items():
        c = sd[BASIS]
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        y = np.array([1] * (tp + fn) + [0] * (fp + tn))
        yh = np.array([1] * tp + [0] * fn + [1] * fp + [0] * tn)
        rng = np.random.default_rng(SEED)
        rows = []
        for p in GRID:
            s = simulate(y, yh, p, rng)
            lo, hi = (float(v) for v in np.percentile(s, [2.5, 97.5]))
            rows.append({"p": p, "ci_lo": lo, "ci_hi": hi,
                         "mean": float(s.mean()),
                         "p_mcc_le_zero": float((s <= 0).mean()),
                         "excludes_zero": bool(lo > 0)})
        # p = 0 is the pair bootstrap alone: no label is flipped. The chapter
        # quotes the spread of each arm about the point estimate there, to say
        # that the interval is asymmetric and must be carried arm by arm, so
        # those two numbers are emitted here rather than left to a reader to
        # subtract. They replace the arm-wise standard deviations the withdrawn
        # construction reported, which were a different quantity.
        base = simulate(y, yh, 0.0, np.random.default_rng(SEED))
        pt = c["mcc"]
        lower = base[base <= pt]
        upper = base[base >= pt]
        arms = {
            "pair_sampling_sd_lower_arm": float(lower.std(ddof=1)) if len(lower) > 1 else None,
            "pair_sampling_sd_upper_arm": float(upper.std(ddof=1)) if len(upper) > 1 else None,
        }

        crossing = None
        for a, b in zip(rows, rows[1:]):
            if a["excludes_zero"] and not b["excludes_zero"]:
                lo_p, hi_p = a["p"], b["p"]
                for _ in range(30):
                    mid = (lo_p + hi_p) / 2
                    s = simulate(y, yh, mid, np.random.default_rng(SEED), B=2500)
                    if float(np.percentile(s, 2.5)) > 0: lo_p = mid
                    else: hi_p = mid
                crossing = (lo_p + hi_p) / 2
                break
        out["by_strategy"][strat] = {
            "cells": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "mcc_point": c["mcc"], "sweep": rows, **arms,
            "flip_probability_at_which_zero_enters": crossing,
            "survives_any_label_noise": bool(rows[-1]["excludes_zero"]),
        }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)
    print(f"wrote {OUT}")
    print(f"register keys under {PREFIX}.*: {update_register(out)} written\n")
    for strat, r in out["by_strategy"].items():
        cr = r["flip_probability_at_which_zero_enters"]
        print(f"{strat:<10} MCC {r['mcc_point']:+.4f}  "
              f"crosses zero at p = {cr:.4f}" if cr else
              f"{strat:<10} MCC {r['mcc_point']:+.4f}  survives to p = 0.5")
        for row in r["sweep"]:
            if row["p"] in (0.0, 0.15, 0.25, 0.50):
                print(f"     p={row['p']:.2f}  [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  "
                      f"mean {row['mean']:+.4f}  P(<=0)={row['p_mcc_le_zero']:.3f}")


if __name__ == "__main__":
    main()
