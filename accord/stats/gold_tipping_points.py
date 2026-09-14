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


def part_b_frontier_template_only(public):
    """ORIGINAL computation, unchanged: minimum over the 8 frontier-template
    rows and the 3 baseline-prompt rate-only rows ONLY (11 rows total). This
    is kept, clearly labelled, alongside the comprehensive computation in
    part_b_all_hard_label_rows below, which the thesis now needs because it
    prints more hard-label over-calling rows than these 11 (the 9 C103
    prompt-sensitivity cells and the 8 C109 replication-prompt cells)."""
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
        "scope": "frontier_template_and_baseline_rate_only_rows_ONLY "
                 "(11 rows: 8 frontier-template + 3 baseline-prompt rate-only). "
                 "Superseded as the headline row-selection by "
                 "part_b_all_hard_label_rows below; kept for comparison.",
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


# ---------------------------------------------------------------------------
# Comprehensive version: every hard-label over-calling row the thesis prints,
# under reading A (12/299, the 46 ambiguous pairs read as NOT contradiction;
# see overcall_ambiguity_readings.py for reading A vs reading B). This adds
# two families of rows the original part_b above never saw:
#   - the 9 C103 prompt-sensitivity cells (3 models x P_asserting/P_neutral/
#     P_reversed), read directly from results/accord/prompt_sensitivity/
#     cells/*.json (n_flagged, n=299; no per-pair confusion breakdown, since
#     these cells were never joined against gold at the 299 basis, only at
#     the 253 basis for AUROC -- see accord/prompt_sensitivity/analyze_c103.py)
#   - the 8 C109 replication-prompt cells (granite3.3:8b and llama3:8b x
#     P_rev_agree/P_rev_compatible/P_rev_follow/P_reversed_rerun), read
#     directly from results/accord/prompt_sensitivity/c109/cells/*.json,
#     same shape, same caveat.
# For every row, ratio = n_flagged / n_positives, n_positives = 12 (the
# confirmed-positive count is fixed at 12 regardless of whether the row's
# own basis is 253 or 299, because reading A treats the 46 ambiguous pairs
# as not-confirmed everywhere). This is checked against the frontier rows'
# own registered ratios below (identity check), then applied uniformly.
#
# n_flagged_negatives (the pool available for direction 2) is EXACT for the
# frontier rows (fp is a registered per-pair count). For every other row
# (baseline rate-only, C103, C109) only the margins n_flagged and
# n_positives=12 are known, not the per-pair overlap between them. In that
# case this script uses the provable bound
#     fp >= n_flagged - min(n_positives, n_flagged)
# which holds because at most n_positives=12 of the n_flagged flagged pairs
# can be among the 12 confirmed positives, however they are distributed; the
# other n_flagged - 12 (when n_flagged > 12) are PROVABLY not confirmed
# positives, i.e. provably available to reclassify under direction 2. This
# is a lower bound on the true fp, not an estimate: using it for feasibility
# is conservative (if this bound already reaches the target, the true row
# does too).
C103_CELLS_DIR = ROOT / "results" / "accord" / "prompt_sensitivity" / "cells"
C109_CELLS_DIR = ROOT / "results" / "accord" / "prompt_sensitivity" / "c109" / "cells"

C103_SLUGS = {"llama3:8b": "llama3_8b", "granite3.3:8b": "granite33_8b", "qwen3:8b": "qwen3_8b"}
C103_PROMPTS = ["P_asserting", "P_neutral", "P_reversed"]
C109_SLUGS = {"llama3:8b": "llama3_8b", "granite3.3:8b": "granite33_8b"}  # no qwen3:8b C109 cells exist
C109_PROMPTS = ["P_rev_agree", "P_rev_compatible", "P_rev_follow", "P_reversed_rerun"]


def _cell_row(path, model, prompt, family):
    cell = json.loads(path.read_text())
    if cell["n"] != 299:
        raise SystemExit(f"unexpected n={cell['n']} in {path}, expected 299")
    return {
        "model": model, "prompt": prompt, "family": family,
        "n": cell["n"], "n_flagged": cell["n_flagged"],
        "source": str(path.relative_to(ROOT)),
    }


def load_c103_rows():
    rows = {}
    for model, slug in C103_SLUGS.items():
        for prompt in C103_PROMPTS:
            path = C103_CELLS_DIR / f"_{slug}_nothink__{prompt}.json"
            key = f"{model}::{prompt}::c103_prompt_sensitivity"
            rows[key] = _cell_row(path, model, prompt, "c103_prompt_sensitivity")
    return rows


def load_c109_rows():
    rows = {}
    for model, slug in C109_SLUGS.items():
        for prompt in C109_PROMPTS:
            path = C109_CELLS_DIR / f"_{slug}_nothink__{prompt}.json"
            key = f"{model}::{prompt}::c109_replication_prompt"
            rows[key] = _cell_row(path, model, prompt, "c109_replication_prompt")
    return rows


def part_b_all_hard_label_rows(facts, public):
    n_positives = 12  # reading A: 46 ambiguous pairs read as NOT contradiction
    public_max = public["public"]["max_over_call_ratio"]
    public_max_cell = public["public"]["max_over_call_ratio_cell"]

    all_rows = collections.OrderedDict()

    # Frontier-template rows: exact fp from the register (via overcall_decomposition.json).
    for key, row in public["clinical"]["frontier_template"].items():
        n_flagged = row["tp"] + row["fp"]
        ratio = n_flagged / n_positives
        # Identity check against the row's own registered ratio (should match
        # exactly: both are n_flagged / 12, the registered one just computed
        # a different but algebraically identical way upstream).
        registered_ratio = row["over_calling_ratio_registered"]
        if abs(ratio - registered_ratio) > 1e-6:
            raise SystemExit(f"{key}: n_flagged/12={ratio} != registered {registered_ratio}")
        all_rows[key] = {
            "family": "frontier_template", "n_flagged": n_flagged,
            "n_positives": n_positives, "ratio": ratio,
            "n_flagged_negatives": row["fp"], "n_flagged_negatives_is_exact": True,
            "source": row["register_source"],
        }

    # Baseline-prompt rate-only rows: n_flagged recovered from the registered
    # rate (must land on an integer out of n=299; asserted below).
    for key, row in public["clinical"]["baseline_prompt_rate_only"].items():
        n = 299
        n_flagged_f = row["positive_rate"] * n
        n_flagged = round(n_flagged_f)
        if abs(n_flagged_f - n_flagged) > 1e-6:
            raise SystemExit(f"{key}: positive_rate * {n} = {n_flagged_f} is not integral")
        ratio = n_flagged / n_positives
        registered_ratio = row["over_calling_ratio_registered"]
        if abs(ratio - registered_ratio) > 1e-6:
            raise SystemExit(f"{key}: n_flagged/12={ratio} != registered {registered_ratio}")
        fp_lower_bound = max(0, n_flagged - min(n_positives, n_flagged))
        all_rows[key] = {
            "family": "baseline_prompt_rate_only", "n_flagged": n_flagged,
            "n_positives": n_positives, "ratio": ratio,
            "n_flagged_negatives": fp_lower_bound, "n_flagged_negatives_is_exact": False,
            "source": row["register_source"],
        }

    # C103 (9 rows) and C109 (8 rows): margins only.
    for key, row in {**load_c103_rows(), **load_c109_rows()}.items():
        n_flagged = row["n_flagged"]
        ratio = n_flagged / n_positives
        fp_lower_bound = max(0, n_flagged - min(n_positives, n_flagged))
        all_rows[key] = {
            "family": row["family"], "n_flagged": n_flagged,
            "n_positives": n_positives, "ratio": ratio,
            "n_flagged_negatives": fp_lower_bound, "n_flagged_negatives_is_exact": False,
            "source": row["source"],
        }

    min_key = min(all_rows, key=lambda k: all_rows[k]["ratio"])
    min_row = all_rows[min_key]
    n_flagged = min_row["n_flagged"]
    n_flagged_negatives = min_row["n_flagged_negatives"]

    m_exact = n_flagged / public_max - n_positives
    m_reach = max(0, int(np.ceil(m_exact - 1e-12)))
    m_feasible = m_reach <= n_flagged_negatives
    ratio_at_m_reach = n_flagged / (n_positives + m_reach)
    ratio_at_m_reach_minus_1 = (n_flagged / (n_positives + m_reach - 1)
                               if m_reach > 0 else None)

    return {
        "scope": "EVERY hard-label over-calling row the thesis prints: 8 "
                 "frontier-template + 3 baseline-prompt rate-only + 9 C103 "
                 "prompt-sensitivity + 8 C109 replication-prompt = 28 rows, "
                 "all under reading A (12/299).",
        "n_rows": len(all_rows),
        "n_flagged_negatives_is_exact_by_family": {
            fam: all(r["n_flagged_negatives_is_exact"] for r in all_rows.values()
                     if r["family"] == fam)
            for fam in sorted({r["family"] for r in all_rows.values()})
        },
        "min_ratio_row": min_key,
        "min_ratio_row_family": min_row["family"],
        "min_ratio_row_source": min_row["source"],
        "min_ratio_value": min_row["ratio"],
        "public_max_over_call_ratio": public_max,
        "public_max_over_call_ratio_cell": public_max_cell,
        "computable": True,
        "n_flagged": n_flagged, "n_positives": n_positives,
        "n_flagged_negatives_available_to_reclassify": n_flagged_negatives,
        "n_flagged_negatives_bound_is_exact": min_row["n_flagged_negatives_is_exact"],
        "direction_2_relabel_flagged_negatives_as_missed_contradictions": {
            "formula": "ratio(m) = n_flagged / (n_positives + m)",
            "m_exact_real_valued": m_exact,
            "n_flips_to_reach_public_max": m_reach,
            "feasible_within_available_flagged_negatives": m_feasible,
            "ratio_at_m_flips": ratio_at_m_reach,
            "ratio_at_m_minus_1_flips": ratio_at_m_reach_minus_1,
            "conclusion": (f"Reclassifying {m_reach} of the {min_key} row's "
                           f"{n_flagged} flagged pairs (at least "
                           f"{n_flagged_negatives} of which are provably not "
                           f"among the 12 confirmed positives) as missed true "
                           f"contradictions brings this row's ratio to or "
                           f"below the public-text maximum of "
                           f"{public_max:.4f}x."),
        },
        "all_rows_by_ratio_ascending": collections.OrderedDict(
            sorted(((k, v["ratio"]) for k, v in all_rows.items()), key=lambda kv: kv[1])),
    }


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
    b_frontier = part_b_frontier_template_only(public)
    b_all = part_b_all_hard_label_rows(facts, public)

    out = collections.OrderedDict([
        ("provenance", {
            "generated_by": "accord/stats/gold_tipping_points.py",
            "precondition_checked_first": True,
        }),
        ("c_auroc_precondition", c),
        ("a_greedy_flip_to_mde", a),
        ("b_flip_to_public_max_over_call_frontier_template_only", b_frontier),
        ("b_flip_to_public_max_over_call_all_hard_label_rows", b_all),
    ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")

    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"(c) n={c['n']} n_pos={c['n_pos']} auc_recomputed={c['auc_recomputed_rounded4']} "
          f"auc_registered={c['auc_registered']} matches={c['matches_to_4dp']}")
    print(f"(a) baseline_auc={a['baseline_auc']:.4f} target_mde={a['target_auc']:.4f} "
          f"n_flips_greedy={a['n_flips_greedy']} reached={a['target_reached_within_cap']} "
          f"final_auc={a['final_auc_after_flips']:.4f} directions={a['flip_directions']}")
    print(f"(b, frontier-template only, {b_frontier.get('min_ratio_row')}) "
          f"min_ratio={b_frontier['min_ratio_value']:.4f} "
          f"public_max={b_frontier['public_max_over_call_ratio']:.4f}")
    if b_frontier.get("computable"):
        d2f = b_frontier["direction_2_relabel_flagged_negatives_as_missed_contradictions"]
        print(f"    direction 2: n_flips_to_reach_public_max={d2f['n_flips_to_reach_public_max']} "
              f"feasible={d2f['feasible_within_available_flagged_negatives']}")
    print(f"(b, ALL {b_all['n_rows']} hard-label rows) min_ratio_row={b_all['min_ratio_row']} "
          f"({b_all['min_ratio_row_family']}) min_ratio={b_all['min_ratio_value']:.4f} "
          f"public_max={b_all['public_max_over_call_ratio']:.4f}")
    d2a = b_all["direction_2_relabel_flagged_negatives_as_missed_contradictions"]
    print(f"    direction 2: n_flips_to_reach_public_max={d2a['n_flips_to_reach_public_max']} "
          f"feasible={d2a['feasible_within_available_flagged_negatives']} "
          f"n_flagged_negatives_available>={b_all['n_flagged_negatives_available_to_reclassify']} "
          f"(exact={b_all['n_flagged_negatives_bound_is_exact']})")


if __name__ == "__main__":
    main()
