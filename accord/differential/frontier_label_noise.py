"""How much of the frontier MCC interval is label noise, and does it matter.

WHY THIS EXISTS
---------------
The frontier labeller does not accept a temperature parameter, so each per-pair
label is one draw from a distribution rather than a fixed value. That much is
recorded in the run artifacts, which carry temperature_supported: false.

The instability was also observed directly: one pair returned opposite labels on
two invocations of the diagnose path at the same token budget. That path prints
to the console and writes nothing, by design, so no committed artifact records
it and it is stated here as an observation rather than as evidence. Nothing in
the thesis rests on it; p below is swept as a hypothetical over the whole range,
which is the conservative treatment and does not need the observation to hold.

The committed bootstrap resamples PAIRS with the labels held fixed. It therefore
measures pair-sampling uncertainty and nothing else, and understates the total.
The obvious remedy is to re-run every pair k times and measure the instability.
That costs thousands of calls and, at 12 positives, buys a few points off an
interval more than thirty points wide, because the binding constraint is the
number of positives and not the number of draws.

This script asks the cheaper question instead: how unstable would the labels
have to be before the conclusion changes? One interpretable parameter, p, the
per-draw probability that any given label flips. Sweeping it to its worst case
of 0.5, where every label is a fair coin, bounds the damage without a single
API call.

WHAT IT DOES AND DOES NOT CLAIM
-------------------------------
It bounds the INTERVAL with the point estimate held fixed. A fully coherent
hierarchical model would also shrink the point estimate toward chance, so this
is a sensitivity bound and not a posterior. It is stated that way because it is
the calculation an examiner asks for, and overstating it would be the same class
of error this thesis exists to catalogue.

Reads:  results/accord/differential/frontier_anthropic_metrics.json
Writes: results/accord/differential/frontier_label_noise.json

    python3 accord/differential/frontier_label_noise.py
"""

import json
import math
import os

IN = "results/accord/differential/frontier_anthropic_metrics.json"
OUT = "results/accord/differential/frontier_label_noise.json"
BASIS = "G_253"
Z = 1.959963985                    # two-sided 95 percent normal quantile
GRID = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


def mcc(tp, fp, fn, tn):
    d = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


def sensitivities(tp, fp, fn, tn):
    """Change in MCC from flipping one label, by central difference.

    A positive pair flipping moves TP and FN in opposite directions; a negative
    pair flipping moves FP and TN. Averaging the two one-sided differences keeps
    the estimate honest near the cell boundaries, where the two directions are
    not symmetric.
    """
    base = mcc(tp, fp, fn, tn)
    d_pos = []
    if fn >= 1:
        d_pos.append(abs(mcc(tp + 1, fp, fn - 1, tn) - base))
    if tp >= 1:
        d_pos.append(abs(mcc(tp - 1, fp, fn + 1, tn) - base))
    d_neg = []
    if tn >= 1:
        d_neg.append(abs(mcc(tp, fp + 1, fn, tn - 1) - base))
    if fp >= 1:
        d_neg.append(abs(mcc(tp, fp - 1, fn, tn + 1) - base))
    return (sum(d_pos) / len(d_pos) if d_pos else 0.0,
            sum(d_neg) / len(d_neg) if d_neg else 0.0)


def main():
    src = json.load(open(IN))
    out = {
        "provenance": {
            "generated_by": "accord/differential/frontier_label_noise.py",
            "input": IN,
            "basis": BASIS,
            "method": ("Total variance is pair-sampling plus label noise. The "
                       "pair-sampling standard deviation is recovered "
                       "separately for each arm of the committed percentile "
                       "bootstrap interval, because that interval is not "
                       "symmetric about the point estimate and treating it as "
                       "symmetric moves the lower bound away from zero. Label "
                       "noise "
                       "is modelled as an independent flip with probability p "
                       "per label, so the flip count on each class is binomial "
                       "and its variance is n*p*(1-p); that count is carried "
                       "into MCC by the per-flip sensitivity of the observed "
                       "confusion matrix. The point estimate is held fixed, so "
                       "this bounds the interval and is not a posterior."),
            "worst_case": ("p = 0.5 makes every label an independent fair coin, "
                           "which is the largest label noise any model can "
                           "assert."),
        },
        "by_strategy": {},
    }

    for strat, sd in src["by_strategy"].items():
        c = sd[BASIS]
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        point = c["mcc"]
        # The committed interval is a PERCENTILE bootstrap and is not symmetric
        # about the point estimate. An earlier version of this script recovered
        # a single standard deviation from the full width, which silently
        # re-symmetrised it: at p = 0 the sweep then failed to reproduce the
        # very interval it starts from, and it did so by moving the lower bound
        # AWAY from zero, which is the anti-conservative direction. Each arm is
        # now carried separately, so the p = 0 row reproduces the committed
        # interval exactly and the widening is applied to the real bounds.
        pair_sd_lo = (point - c["mcc_ci_lo"]) / Z
        pair_sd_hi = (c["mcc_ci_hi"] - point) / Z
        s_pos, s_neg = sensitivities(tp, fp, fn, tn)
        n_pos, n_neg = tp + fn, fp + tn

        def lower_bound(p):
            var = (s_pos ** 2) * n_pos * p * (1 - p) + \
                  (s_neg ** 2) * n_neg * p * (1 - p)
            return point - Z * math.sqrt(pair_sd_lo ** 2 + var)

        rows = []
        for p in GRID:
            var = (s_pos ** 2) * n_pos * p * (1 - p) + \
                  (s_neg ** 2) * n_neg * p * (1 - p)
            label_sd = math.sqrt(var)
            total_lo = math.sqrt(pair_sd_lo ** 2 + label_sd ** 2)
            total_hi = math.sqrt(pair_sd_hi ** 2 + label_sd ** 2)
            lo, hi = point - Z * total_lo, point + Z * total_hi
            rows.append({"p": p, "label_sd": label_sd,
                         "total_sd_lower_arm": total_lo,
                         "total_sd_upper_arm": total_hi,
                         "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0)})

        # Solve for the crossing rather than reporting the first grid point at
        # which zero has already entered. On a 0.05/0.10 grid that reported
        # 0.20 for a value nearer 0.11, which overstates how much instability
        # the few-shot result tolerates.
        crossing = None
        if lower_bound(0.5) <= 0.0:
            a, b = 0.0, 0.5
            for _ in range(200):
                mid = (a + b) / 2.0
                if lower_bound(mid) > 0.0:
                    a = mid
                else:
                    b = mid
            crossing = (a + b) / 2.0

        out["by_strategy"][strat] = {
            "cells": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "mcc_point": point,
            "pair_sampling_sd_lower_arm": pair_sd_lo,
            "pair_sampling_sd_upper_arm": pair_sd_hi,
            "mcc_change_per_positive_flip": s_pos,
            "mcc_change_per_negative_flip": s_neg,
            "sweep": rows,
            "worst_case_ci_lo": rows[-1]["ci_lo"],
            "survives_any_label_noise": bool(rows[-1]["ci_lo"] > 0),
            "flip_probability_at_which_zero_enters": crossing,
        }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"wrote {OUT}\n")
    for strat, r in out["by_strategy"].items():
        print(f"{strat}  MCC {r['mcc_point']:+.4f}  "
              f"pair SD {r['pair_sampling_sd_lower_arm']:.4f}/"
              f"{r['pair_sampling_sd_upper_arm']:.4f}  "
              f"per-flip {r['mcc_change_per_positive_flip']:.4f} pos / "
              f"{r['mcc_change_per_negative_flip']:.4f} neg")
        for row in r["sweep"]:
            mark = "excludes 0" if row["excludes_zero"] else "CONTAINS 0"
            print(f"    p={row['p']:.2f}  "
                  f"SD {row['total_sd_lower_arm']:.4f}  95% CI [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  {mark}")
        if r["survives_any_label_noise"]:
            print("    -> survives even the fair-coin worst case\n")
        else:
            print(f"    -> zero enters at p = "
                  f"{r['flip_probability_at_which_zero_enters']:.2f}\n")


if __name__ == "__main__":
    main()
