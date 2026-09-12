"""Three computations a committee asked for that cost nothing but had not been run.

Each closes a question the thesis itself raises and leaves open. None needs a new
experiment; all three read artifacts already on the machine.

1. The weak labeller's own MCC. The thesis measures the detector against the gold
   standard and the weak labeller only by its marginal positive rate, which is why
   over-calling is reported as a ratio rather than as a correlation. The labeller's
   own agreement with the pharmacist has never been stated. It is computable, but
   only through a reconstruction: the merged file joins on a nursing-side key rather
   than on pair identity, so 21 of 299 pairs receive two conflicting weak labels.
   The determinate subset is reported as the point value and the two extreme
   resolutions of the conflicted pairs as bounds. This is the same limitation the
   join-feasibility diagnostic records; it is not evaded here, it is bounded.

2. The detector's threshold-optimal MCC. Chapter 4 reports the operating point at
   the configured threshold of 0.5, where agreement is worse than chance. The
   obvious objection is that a different cutoff would rescue it. The answer is the
   best MCC attainable by any cutoff, chosen with full knowledge of the labels,
   which is an optimistic bound no honest procedure could reach. Computed on the
   released score file, the same file behind the AUROC finding, not on the
   three-seed run behind the operating table.

3. Joinability against correctness. Chapter 4 keeps two irreconcilable wrong rates
   for the exemplar defect, 0.4745 from a reconstruction over 137 rows and 0 from an
   exact key join over 23, and states that keeping both rests on an untested
   assumption: that a row is joinable when the pipeline's bookkeeping about it was
   intact, which is the same bookkeeping the defect corrupts. The test is a Fisher
   exact on joinability against correctness. No artifact carries both attributes per
   row, so the test is run under both readings of how the 23 joinable rows overlap
   the 137 resolved ones, and reported only if it holds under both.

Run:  python3 accord/stats/reviewer_free_computations.py
Writes: results/accord/stats/reviewer_free_computations.json
"""

import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "accord" / "stats" / "reviewer_free_computations.json"

# Both inputs carry note sentences and stay outside the repository.
MERGED = pathlib.Path.home() / "mimic_local_only" / "from_drive_2026-08-08" / "outputs" / "pharmacist_llama_merged.csv"
SCORES = pathlib.Path.home() / "Downloads" / "gold_standard_accord_results.csv"


def mcc(tp, tn, fp, fn):
    den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else (tp * tn - fp * fn) / den


def cells(y, p):
    return (int((y & p).sum()), int((~y & ~p).sum()),
            int((~y & p).sum()), int((y & ~p).sum()))


def weak_labeller_mcc():
    d = pd.read_csv(MERGED, low_memory=False)
    g = d.groupby("pair_id").agg(
        gold=("label_clean", "first"),
        n_distinct=("weak_label", "nunique"),
        any_pos=("weak_label", lambda s: (s == "contradiction").any()),
        all_pos=("weak_label", lambda s: (s == "contradiction").all()))
    out = {}
    for name, keep in [("G_299", ["contradiction", "not_contradiction", "ambiguous"]),
                       ("G_253", ["contradiction", "not_contradiction"])]:
        sub = g[g.gold.isin(keep)]
        y = (sub.gold == "contradiction").values
        det = (sub.n_distinct == 1).values
        s = sub[det]
        tp, tn, fp, fn = cells((s.gold == "contradiction").values, s.any_pos.values)
        row = {"n": int(len(sub)), "positives": int(y.sum()),
               "n_join_determinate": int(det.sum()),
               "n_join_conflicted": int((~det).sum()),
               "determinate": {"n": int(len(s)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                               "mcc": round(mcc(tp, tn, fp, fn), 4)}}
        for lab, col in [("conflicted_as_positive", "any_pos"),
                         ("conflicted_as_negative", "all_pos")]:
            tp, tn, fp, fn = cells(y, sub[col].values)
            row[lab] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
                        "mcc": round(mcc(tp, tn, fp, fn), 4)}
        row["mcc_bounds"] = [min(row["conflicted_as_positive"]["mcc"], row["conflicted_as_negative"]["mcc"]),
                             max(row["conflicted_as_positive"]["mcc"], row["conflicted_as_negative"]["mcc"])]
        out[name] = row
    return out


def detector_threshold_optimal_mcc():
    d = pd.read_csv(SCORES)
    out = {}
    for name, sub in [("G_299", d), ("G_253", d[d.pharm_label != "ambiguous"])]:
        y = (sub.pharm_label == "contradiction").values
        col_out = {}
        for col in ["accord_contra_prob", "accord_harm_score"]:
            s = sub[col].values
            best = (-2.0, None)
            for t in sorted(set(s)):
                tp, tn, fp, fn = cells(y, s >= t)
                m = mcc(tp, tn, fp, fn)
                if m > best[0]:
                    best = (m, (float(t), tp, tn, fp, fn))
            t, tp, tn, fp, fn = best[1]
            ctp, ctn, cfp, cfn = cells(y, s >= 0.5)
            col_out[col] = {
                "threshold_optimal": {"mcc": round(best[0], 4), "threshold": round(t, 6),
                                      "tp": tp, "tn": tn, "fp": fp, "fn": fn},
                "at_configured_0.5": {"mcc": round(mcc(ctp, ctn, cfp, cfn), 4),
                                      "tp": ctp, "tn": ctn, "fp": cfp, "fn": cfn}}
        col_out["n"] = int(len(sub))
        col_out["positives"] = int(y.sum())
        out[name] = col_out
    return out


def joinability_vs_correctness():
    # Counts as registered: 165 pooled Rule 1 and 2 candidates; the reconstruction
    # resolves 137 of them, 72 correct and 65 wrong; the exact key join resolves 23,
    # all correct. Which of the 137 the 23 are is not recorded by any artifact, so
    # both readings are run.
    n_recon_correct, n_recon_wrong, n_join = 72, 65, 23
    out = {"n_pooled_rule12": 165, "reconstruction": {"resolved": 137,
           "correct": n_recon_correct, "wrong": n_recon_wrong},
           "key_join": {"resolved": n_join, "correct": n_join, "wrong": 0}}
    for label, table in [
            ("joinable_subset_of_resolved", [[n_join, 0], [n_recon_correct - n_join, n_recon_wrong]]),
            ("joinable_disjoint_from_resolved", [[n_join, 0], [n_recon_correct, n_recon_wrong]])]:
        orr, p = fisher_exact(table)
        out[label] = {"table": table, "odds_ratio": ("inf" if math.isinf(orr) else round(orr, 4)),
                      "p_value": float("%.6g" % p)}
    ps = [out[k]["p_value"] for k in ("joinable_subset_of_resolved", "joinable_disjoint_from_resolved")]
    out["max_p_across_readings"] = max(ps)
    out["conclusion"] = (
        "Joinability is associated with exemplar correctness under both readings of the "
        "overlap, at a largest p of %.6g. The assumption Chapter 4 states and does not test, "
        "that a row is joinable when the bookkeeping the defect corrupts was intact, is "
        "supported. It does not identify which wrong rate is correct; it establishes that the "
        "two measurements are drawn from differently behaved subsets and so cannot be read as "
        "the ends of an interval." % max(ps))
    return out


def join_permutation_null(cells, B=20000, seed=0):
    """Is the weak-labeller MCC distinguishable from a scrambled join?

    The weak-label file and the gold file share no identifying column, so the
    join is a reconstruction. Under a random alignment the labeller's marginal is
    unchanged and only which rows it lands on varies, which is a hypergeometric
    draw on the true positives. If the observed MCC sits inside that distribution
    then no pair-level agreement is measurable from the retained artifacts, and
    the number must not be reported as a measurement of the labeller.
    """
    tp, tn, fp, fn = cells
    n = tp + tn + fp + fn
    npos, nflag = tp + fn, tp + fp
    rng = np.random.default_rng(seed)
    draws = rng.hypergeometric(nflag, n - nflag, npos, size=B)
    m = np.array([mcc(k, (n - nflag) - (npos - k), nflag - k, npos - k) for k in draws])
    obs = mcc(tp, tn, fp, fn)
    # Ceiling: the labeller emits hard labels at a fixed marginal, so perfect
    # agreement is arithmetically unreachable. Best case is every actual positive
    # inside the flagged set, with the flagged count held where it is.
    ceiling = mcc(npos, (n - nflag), nflag - npos, 0)
    return {"observed_mcc": round(obs, 4),
            "permutation_B": B, "permutation_seed": seed,
            "null_mean": round(float(m.mean()), 4),
            "null_p95_hi": round(float(np.percentile(m, 97.5)), 4),
            "null_p95_lo": round(float(np.percentile(m, 2.5)), 4),
            "p_null_ge_observed": float("%.4g" % float((m >= obs).mean())),
            "distinguishable_from_random_join": bool((m >= obs).mean() < 0.05),
            "predicted_positive_rate": round(nflag / n, 4),
            "expected_tp_under_random_join": round(npos * nflag / n, 2),
            "observed_tp": tp,
            "marginal_constrained_max_mcc": round(ceiling, 4),
            "observed_as_fraction_of_ceiling": round(obs / ceiling, 4) if ceiling else None}


def oracle_threshold_null(n=253, npos=12, B=4000, seed=1):
    """Null distribution of the MAX-over-threshold MCC.

    A statistic maximised over a tuning parameter on the same sample it is
    evaluated on is not centred at zero under the null, so comparing an oracle
    threshold MCC against 0 overstates it. This is the reference the comparison
    needs instead.
    """
    rng = np.random.default_rng(seed)
    y0 = np.zeros(n, bool); y0[:npos] = True
    best = []
    for _ in range(B):
        s_ = rng.random(n)
        y = y0[rng.permutation(n)]
        o = np.argsort(-s_); yo = y[o]
        ctp = np.cumsum(yo); k = np.arange(1, n + 1)
        cfp = k - ctp; cfn = npos - ctp; ctn = (n - npos) - cfp
        den = np.sqrt((ctp + cfp) * (ctp + cfn) * (ctn + cfp) * (ctn + cfn), dtype=float)
        best.append(float(np.where(den == 0, 0, (ctp * ctn - cfp * cfn) / np.where(den == 0, 1, den)).max()))
    b = np.array(best)
    return {"n": n, "n_positives": npos, "B": B, "seed": seed,
            "null_median": round(float(np.median(b)), 4),
            "null_p05": round(float(np.percentile(b, 5)), 4),
            "null_p95": round(float(np.percentile(b, 95)), 4),
            "p_null_ge": {}}


def main():
    wl = weak_labeller_mcc()
    for base in ("G_299", "G_253"):
        d0 = wl[base]["determinate"]
        wl[base]["random_join_control"] = join_permutation_null(
            (d0["tp"], d0["tn"], d0["fp"], d0["fn"]))

    onull = oracle_threshold_null()
    det = detector_threshold_optimal_mcc()
    rng = np.random.default_rng(1)
    # score each reported optimum against that null
    y0 = np.zeros(253, bool); y0[:12] = True
    best = []
    for _ in range(4000):
        s_ = rng.random(253); y = y0[rng.permutation(253)]
        o = np.argsort(-s_); yo = y[o]
        ctp = np.cumsum(yo); k = np.arange(1, 254)
        cfp = k - ctp; cfn = 12 - ctp; ctn = 241 - cfp
        den = np.sqrt((ctp + cfp) * (ctp + cfn) * (ctn + cfp) * (ctn + cfn), dtype=float)
        best.append(float(np.where(den == 0, 0, (ctp * ctn - cfp * cfn) / np.where(den == 0, 1, den)).max()))
    barr = np.array(best)
    for col in ("accord_contra_prob", "accord_harm_score"):
        v = det["G_253"][col]["threshold_optimal"]["mcc"]
        onull["p_null_ge"][col] = float("%.4g" % float((barr >= v).mean()))

    out = {"weak_labeller_mcc": wl,
           "oracle_threshold_null": onull,
           "detector_threshold_optimal_mcc": det,
           "joinability_vs_correctness": joinability_vs_correctness()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
