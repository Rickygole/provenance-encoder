"""Gold-label tipping points: how many gold labels would have to be wrong.

WHY THIS EXISTS
---------------
Two of this thesis's numbers are both computed against a 299-pair, mostly
hand-labelled gold standard. Neither number says how sensitive it is to that
gold standard being wrong. This script computes two tipping points, both
against the SAME 253-pair (ambiguous-excluded) gold set:

  (a) How many gold labels would a worst-case adversary have to flip to
      raise the detector's own G_253 AUROC (accord_contra_prob vs
      pharm_label, currently far below chance) up to the evaluation's own
      minimum detectable AUROC (mde_recomputed.gold_ambiguous_excluded
      .minimum_detectable_auc)? Computed by a GREEDY heuristic: at each
      step, flip whichever remaining label increases AUROC the most, and
      stop once the target is reached. This is a bound on adversarial
      flips, not a probability that the gold standard is that wrong, and
      not a proof of the true minimum (greedy is not guaranteed optimal).

  (b) For the headline over-calling finding, the smallest of the registered
      local-labeller over-calling ratios (results/accord/stats
      /overcall_decomposition.json::clinical) is 14.67x
      (granite3.3:8b, few-shot, G_253). How many of that row's 176 flagged
      pairs would have to be reclassified before its ratio fell to the
      public-text maximum (3.5x, from the same artifact)? Two adversarial
      directions are considered, because they move the ratio in OPPOSITE
      directions: discounting confirmed positives as gold errors (over_call
      = n_flagged / n_positives, so REMOVING positives only increases the
      ratio; this direction cannot reach a lower target at all) versus
      relabelling flagged negatives as missed true contradictions (which
      DOES lower the ratio, because it moves items from the flagged
      negatives bucket into the positives bucket without changing
      n_flagged).

  (c) A precondition, done FIRST: recompute the detector's own G_253 AUROC
      directly from the joined CSV and assert it equals the registered
      signal.contra_prob_vs_pharm_label__ambiguous_excluded_n253.auroc to 4
      decimals. If it does not match, everything downstream in (a) and this
      analysis's premise is unreliable, so the script aborts rather than
      reporting a number computed on the wrong join.

Nothing beyond aggregate counts, an AUROC and a small integer flip count is
ever printed or written; no pair_id appears in the artifact.

Reads (local, aggregate only): ~/mimic_local_only/derived/
                                gold_standard_accord_results.csv
Reads (register, read-only): results/accord/VERIFIED_FACTS.json
Reads (public artifact):     results/accord/stats/overcall_decomposition.json
Writes: results/accord/stats/gold_tipping_points.json

Run (MUST redirect, this script touches MIMIC-derived data):
    python3 accord/stats/gold_tipping_points.py \
        > ~/mimic_local_only/logs/gold_tipping_points.log 2>&1
"""

import collections
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _overcall_common import ROOT, load_facts, reg

GOLD_CSV = pathlib.Path.home() / "mimic_local_only" / "derived" / "gold_standard_accord_results.csv"
PUBLIC_ARTIFACT = ROOT / "results" / "accord" / "stats" / "overcall_decomposition.json"
OUT = ROOT / "results" / "accord" / "stats" / "gold_tipping_points.json"
MAX_FLIPS = 80  # greedy search cap; documented, not tuned to a result


def greedy_flip_to_target(y, s, target_auc, max_flips=MAX_FLIPS):
    """Greedily flip the single label per step that raises AUROC the most.

    Returns (n_flips, final_auc, direction_counts, reached: bool, trace).
    trace is a list of (auc_after_this_flip,) with no item-level identity, so
    nothing here can be traced back to a specific pair.
    """
    y = y.copy()
    n = len(y)
    flipped = np.zeros(n, dtype=bool)
    auc = float(roc_auc_score(y, s))
    trace = [auc]
    n_neg_to_pos, n_pos_to_neg = 0, 0
    if auc >= target_auc:
        return 0, auc, {"neg_to_pos": 0, "pos_to_neg": 0}, True, trace

    for _ in range(max_flips):
        best_auc, best_i = -1.0, None
        remaining = np.where(~flipped)[0]
        for i in remaining:
            y[i] = 1 - y[i]
            a = roc_auc_score(y, s) if len(np.unique(y)) > 1 else 0.5
            if a > best_auc:
                best_auc, best_i = a, i
            y[i] = 1 - y[i]  # revert probe
        if best_i is None:
            break
        if y[best_i] == 0:
            n_neg_to_pos += 1
        else:
            n_pos_to_neg += 1
        y[best_i] = 1 - y[best_i]
        flipped[best_i] = True
        auc = best_auc
        trace.append(auc)
        if auc >= target_auc:
            return (len(trace) - 1, auc,
                    {"neg_to_pos": n_neg_to_pos, "pos_to_neg": n_pos_to_neg},
                    True, trace)
    return (len(trace) - 1, auc,
            {"neg_to_pos": n_neg_to_pos, "pos_to_neg": n_pos_to_neg},
            False, trace)


def part_c_precondition(facts):
    gold = pd.read_csv(GOLD_CSV)
    sub = gold[gold.pharm_label != "ambiguous"].copy()
    y = (sub.pharm_label == "contradiction").astype(int).to_numpy()
    s = sub.accord_contra_prob.to_numpy()
    n = len(sub)
    n_pos = int(y.sum())
    auc_recomputed = float(roc_auc_score(y, s))
    auc_registered = reg(facts, "signal.contra_prob_vs_pharm_label__ambiguous_excluded_n253.auroc")
    match = round(auc_recomputed, 4) == round(auc_registered, 4)
    return {
        "n": n, "n_pos": n_pos,
        "auc_recomputed": auc_recomputed,
        "auc_recomputed_rounded4": round(auc_recomputed, 4),
        "auc_registered": auc_registered,
        "matches_to_4dp": match,
    }, y, s, n, n_pos


def part_a(facts, y, s, n, n_pos):
    mde = reg(facts, "mde_recomputed.gold_ambiguous_excluded.minimum_detectable_auc")
    n_flips, final_auc, directions, reached, trace = greedy_flip_to_target(y, s, mde)
    return {
        "target_auc": mde,
        "target_source": "results/accord/VERIFIED_FACTS.json"
                         "::mde_recomputed.gold_ambiguous_excluded.minimum_detectable_auc",
        "baseline_auc": trace[0],
        "n_flips_greedy": n_flips,
        "n_flips_as_fraction_of_n": round(n_flips / n, 4),
        "n_flips_as_fraction_of_n_pos": round(n_flips / n_pos, 4) if n_pos else None,
        "final_auc_after_flips": final_auc,
        "flip_directions": directions,
        "target_reached_within_cap": reached,
        "search_cap": MAX_FLIPS,
        "auc_trace_by_flip_count": [round(a, 4) for a in trace],
        "method": "Greedy: at each step, flip whichever remaining label "
                  "increases AUROC the most; stop once the target is "
                  "reached. This is an upper-bound heuristic on the "
                  "adversarial flip count, not a proof of the true minimum "
                  "(greedy search is not guaranteed globally optimal), and "
                  "it is a bound on how many labels an adversary would have "
                  "to get wrong, not a probability that they are wrong.",
    }


def part_b(public):
    clinical = public["clinical"]
    frontier = clinical["frontier_template"]
    baseline = clinical["baseline_prompt_rate_only"]
    all_rows = {**{k: v for k, v in frontier.items()},
               **{k: v for k, v in baseline.items()}}
    # Only frontier rows carry the n_flagged/n_positives breakdown this
    # direction needs; identify the minimum ratio among ALL rows, but note
    # if it lands on a baseline (rate-only) row this computation cannot be
    # carried out for it.
    min_key = min(all_rows, key=lambda k: all_rows[k]["over_calling_ratio_registered"])
    min_row = all_rows[min_key]
    is_frontier = min_key in frontier
    public_max = public["public"]["max_over_call_ratio"]
    public_max_cell = public["public"]["max_over_call_ratio_cell"]

    out = {
        "min_ratio_row": min_key,
        "min_ratio_row_is_frontier_template": is_frontier,
        "min_ratio_value": min_row["over_calling_ratio_registered"],
        "public_max_over_call_ratio": public_max,
        "public_max_over_call_ratio_cell": public_max_cell,
    }
    if not is_frontier:
        out["computable"] = False
        out["why_not"] = ("The minimum-ratio row is a baseline-prompt row "
                          "with only a positive rate registered; it has no "
                          "n_flagged/n_positives breakdown, so this "
                          "reclassification calculation cannot be carried "
                          "out for it.")
        return out

    n_flagged = min_row["tp"] + min_row["fp"]
    n_positives = min_row["n_positives"]
    n_flagged_negatives = min_row["fp"]  # only these can become "missed contradictions"

    # Direction 1: discount m of the n_positives confirmed positives as gold
    # errors. n_flagged is unaffected (the labeller's own flags don't change);
    # n_positives shrinks, so ratio = n_flagged / n_positives strictly
    # increases in m. It cannot reach a LOWER target at all.
    def ratio_direction1(m):
        return n_flagged / (n_positives - m)

    direction1_is_monotonically_increasing = all(
        ratio_direction1(m) > ratio_direction1(0) for m in range(1, n_positives))

    # Direction 2: relabel m of the flagged negatives as missed true
    # contradictions. n_flagged unaffected; n_positives grows by m.
    # ratio(m) = n_flagged / (n_positives + m); solve for the smallest m with
    # ratio(m) <= public_max.
    m_exact = n_flagged / public_max - n_positives
    m_reach = max(0, int(np.ceil(m_exact - 1e-12)))
    m_feasible = m_reach <= n_flagged_negatives
    ratio_at_m_reach = n_flagged / (n_positives + m_reach)
    ratio_at_m_reach_minus_1 = (n_flagged / (n_positives + m_reach - 1)
                               if m_reach > 0 else None)

    out.update({
        "computable": True,
        "n_flagged": n_flagged, "n_positives": n_positives,
        "n_flagged_negatives_available_to_reclassify": n_flagged_negatives,
        "direction_1_discount_confirmed_positives": {
            "formula": "ratio(m) = n_flagged / (n_positives - m)",
            "monotonically_increasing_in_m": direction1_is_monotonically_increasing,
            "conclusion": "This direction moves the ratio AWAY from the "
                          "public-text maximum for every m > 0; no finite "
                          "number of direction-1-only flips reaches it. The "
                          "largest number of direction-1 flips that help is "
                          "therefore 0.",
        },
        "direction_2_relabel_flagged_negatives_as_missed_contradictions": {
            "formula": "ratio(m) = n_flagged / (n_positives + m)",
            "m_exact_real_valued": m_exact,
            "n_flips_to_reach_public_max": m_reach,
            "feasible_within_available_flagged_negatives": m_feasible,
            "ratio_at_m_flips": ratio_at_m_reach,
            "ratio_at_m_minus_1_flips": ratio_at_m_reach_minus_1,
            "largest_n_flips_that_keep_ratio_above_public_max": max(0, m_reach - 1),
            "conclusion": (f"Reclassifying {m_reach} of the {n_flagged_negatives} "
                           f"pairs this row flagged as negative (out of the "
                           f"12 confirmed positives being fixed) as missed "
                           f"true contradictions brings this row's ratio to "
                           f"or below the public-text maximum of "
                           f"{public_max:.4f}x."),
        },
    })
    return out


def main():
    facts = load_facts()
    public = json.loads(PUBLIC_ARTIFACT.read_text())

    # (c) FIRST, as a precondition.
    c, y, s, n, n_pos = part_c_precondition(facts)
    if not c["matches_to_4dp"]:
        raise SystemExit(
            f"ABORT: recomputed G_253 AUROC {c['auc_recomputed_rounded4']} does not "
            f"match registered signal.contra_prob_vs_pharm_label__"
            f"ambiguous_excluded_n253.auroc {c['auc_registered']} to 4 decimals; "
            f"the join or the gold file has drifted from what the register describes.")

    a = part_a(facts, y, s, n, n_pos)
    b = part_b(public)

    out = collections.OrderedDict([
        ("provenance", {
            "generated_by": "accord/stats/gold_tipping_points.py",
            "precondition_checked_first": True,
        }),
        ("c_auroc_precondition", c),
        ("a_greedy_flip_to_mde", a),
        ("b_flip_to_public_max_over_call", b),
    ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")

    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"(c) n={c['n']} n_pos={c['n_pos']} auc_recomputed={c['auc_recomputed_rounded4']} "
          f"auc_registered={c['auc_registered']} matches={c['matches_to_4dp']}")
    print(f"(a) baseline_auc={a['baseline_auc']:.4f} target_mde={a['target_auc']:.4f} "
          f"n_flips_greedy={a['n_flips_greedy']} reached={a['target_reached_within_cap']} "
          f"final_auc={a['final_auc_after_flips']:.4f} directions={a['flip_directions']}")
    print(f"(b) min_ratio_row={b['min_ratio_row']} min_ratio={b['min_ratio_value']:.4f} "
          f"public_max={b['public_max_over_call_ratio']:.4f}")
    if b.get("computable"):
        d2 = b["direction_2_relabel_flagged_negatives_as_missed_contradictions"]
        print(f"    direction 2: n_flips_to_reach_public_max={d2['n_flips_to_reach_public_max']} "
              f"feasible={d2['feasible_within_available_flagged_negatives']}")


if __name__ == "__main__":
    main()
