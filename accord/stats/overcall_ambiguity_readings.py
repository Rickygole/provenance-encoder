"""Over-calling under both readings of the 46 ambiguous pairs.

WHY THIS EXISTS
---------------
The headline over-calling numbers (14.7x to 23.75x, see
results/accord/stats/overcall_decomposition.json::clinical) all divide a local
labeller's flag rate by a base rate that treats the 46 ambiguous pairs as
NOT contradictions: 12/299 = 0.0401. That is a reading choice, not a fact
about the pairs; if the ambiguous pairs are read the other way, as
contradictions, the base rate is 58/299 = 0.1940. This script recomputes
every registered local-labeller over-calling ratio under both readings and
asks the only question that matters for the headline's robustness: does the
ratio still exceed the largest over-calling ratio seen on the public,
non-clinical SNLI sweep (results/accord/stats/overcall_decomposition.json
::public.max_over_call_ratio) under EITHER reading, BOTH readings, or
NEITHER.

Wilson intervals on the two base rates (12/299, 58/299) are carried into the
ratio as a bound: since ratio = rate / p is monotonically decreasing in p,
the base rate's Wilson upper bound gives the ratio's conservative (lowest)
value and the Wilson lower bound gives the ratio's least-conservative
(highest) value.

Both counts (12 confirmed, 46 ambiguous, out of 299) are read from the
register, not typed. As a local independent cross-check, they are also
recomputed by grouping ~/mimic_local_only/from_drive_2026-08-08/outputs/
pharmacist_labeled_clean.csv on pair_id and value_counting label_clean's
per-pair label. That check reads clinical annotation text (nursing_sentence,
discharge_sentence, justification columns exist in that file) so this script
must be run with stdout redirected to ~/mimic_local_only/logs/, and prints
only the three aggregate counts and whether they match the register -- never
a pair_id, a sentence or a justification string.

Reads (register, read-only): results/accord/VERIFIED_FACTS.json
Reads (public artifact):     results/accord/stats/overcall_decomposition.json
Reads (local, aggregate only): ~/mimic_local_only/from_drive_2026-08-08/
                                outputs/pharmacist_labeled_clean.csv
Writes: results/accord/stats/overcall_ambiguity_readings.json

Run (MUST redirect, this script touches MIMIC-derived data):
    python3 accord/stats/overcall_ambiguity_readings.py \
        > ~/mimic_local_only/logs/overcall_ambiguity_readings.log 2>&1
"""

import collections
import json
import pathlib

import pandas as pd

from _overcall_common import ROOT, load_facts, reg, wilson_ci

PUBLIC_ARTIFACT = ROOT / "results" / "accord" / "stats" / "overcall_decomposition.json"
OUT = ROOT / "results" / "accord" / "stats" / "overcall_ambiguity_readings.json"
PHARM_CSV = (pathlib.Path.home() / "mimic_local_only" / "from_drive_2026-08-08"
             / "outputs" / "pharmacist_labeled_clean.csv")

# Every local labeller positive rate registered on G_299, frontier template
# and baseline-prompt rows alike. Same row list as
# overcall_decomposition.py's FRONTIER_ROWS / BASELINE_RATE_ONLY_ROWS,
# restricted to the G_299 basis (this analysis is about the 46 ambiguous
# pairs, which only exist inside the 299-pair set; G_253 already excludes
# them by construction and so cannot be re-read under either ambiguity
# convention).
FRONTIER_G299_ROWS = [
    ("granite3.3:8b", "frontier_ollama_granite33", "zero_shot"),
    ("granite3.3:8b", "frontier_ollama_granite33", "few_shot"),
    ("llama3:8b", "frontier_ollama_llama3", "zero_shot"),
    ("llama3:8b", "frontier_ollama_llama3", "few_shot"),
]
BASELINE_G299_ROWS = [
    ("llama3:8b", "overcall.llama3_8b"),
    ("granite3.3:8b", "overcall.granite3_3_8b"),
    ("qwen3:8b", "overcall.qwen3_8b"),
]


def local_cross_check():
    """Aggregate-only cross-check against the register's 241/46/12 split.

    Reads clinical annotation text but never prints or stores it: only the
    three category counts (an integer per label, not a row) leave this
    function.
    """
    df = pd.read_csv(PHARM_CSV)
    # One row per (pair_id, rule) in this file; take one label_clean per
    # pair_id after checking it is in fact constant within a pair (a pair
    # should not be both "ambiguous" and "contradiction" under different
    # rules in the same annotation pass).
    per_pair = df.groupby("pair_id")["label_clean"].agg(lambda s: s.mode().iloc[0])
    inconsistent = df.groupby("pair_id")["label_clean"].nunique()
    n_inconsistent_pairs = int((inconsistent > 1).sum())
    counts = per_pair.value_counts().to_dict()
    return {
        "n_unique_pairs": int(per_pair.shape[0]),
        "n_pairs_with_inconsistent_label_clean_across_rows": n_inconsistent_pairs,
        "label_clean_counts_per_pair": {str(k): int(v) for k, v in counts.items()},
    }


def main():
    facts = load_facts()
    public = json.loads(PUBLIC_ARTIFACT.read_text())
    public_max = public["public"]["max_over_call_ratio"]
    public_max_cell = public["public"]["max_over_call_ratio_cell"]

    n_total = reg(facts, "gold.n_total")
    n_confirmed = reg(facts, "exemplar.power_analysis.n_positives_total")
    n_ambiguous_candidates = {
        "frontier_ollama_granite33.ambiguity_gradient.zero_shot.n_ambiguous",
        "frontier_ollama_granite33.ambiguity_gradient.few_shot.n_ambiguous",
        "frontier_ollama_llama3.ambiguity_gradient.zero_shot.n_ambiguous",
        "frontier_ollama_llama3.ambiguity_gradient.few_shot.n_ambiguous",
    }
    n_ambiguous_values = {k: reg(facts, k) for k in n_ambiguous_candidates}
    if len(set(n_ambiguous_values.values())) != 1:
        raise SystemExit(f"n_ambiguous disagrees across registered keys: {n_ambiguous_values}")
    n_ambiguous = next(iter(n_ambiguous_values.values()))

    n_clear_negative_candidates = {
        "frontier_ollama_granite33.ambiguity_gradient.zero_shot.n_clear_negative",
        "frontier_ollama_llama3.ambiguity_gradient.zero_shot.n_clear_negative",
    }
    n_clear_negative_values = {k: reg(facts, k) for k in n_clear_negative_candidates}
    if len(set(n_clear_negative_values.values())) != 1:
        raise SystemExit(f"n_clear_negative disagrees: {n_clear_negative_values}")
    n_clear_negative = next(iter(n_clear_negative_values.values()))
    if n_confirmed + n_ambiguous + n_clear_negative != n_total:
        raise SystemExit(
            f"counts do not add to gold.n_total: {n_confirmed} + {n_ambiguous} + "
            f"{n_clear_negative} != {n_total}")

    n_pos_A = n_confirmed              # ambiguous read as NOT contradiction (current reading)
    n_pos_B = n_confirmed + n_ambiguous  # ambiguous read as contradiction
    p_A, p_B = n_pos_A / n_total, n_pos_B / n_total
    ciA_lo, ciA_hi = wilson_ci(n_pos_A, n_total)
    ciB_lo, ciB_hi = wilson_ci(n_pos_B, n_total)

    cross_check = local_cross_check()
    lc = cross_check["label_clean_counts_per_pair"]
    cross_check_matches_register = (
        lc.get("not_contradiction") == n_clear_negative
        and lc.get("ambiguous") == n_ambiguous
        and lc.get("contradiction") == n_confirmed
    )
    cross_check["matches_register_241_46_12"] = cross_check_matches_register

    def ratio_and_bounds(rate):
        ratio_A = rate / p_A
        ratio_B = rate / p_B
        # ratio = rate / p is decreasing in p, so the base rate's Wilson
        # upper bound gives the ratio's lower (conservative) bound and the
        # lower bound gives the ratio's upper bound.
        ratio_A_lo, ratio_A_hi = rate / ciA_hi, rate / ciA_lo
        ratio_B_lo, ratio_B_hi = rate / ciB_hi, rate / ciB_lo
        return ratio_A, (ratio_A_lo, ratio_A_hi), ratio_B, (ratio_B_lo, ratio_B_hi)

    rows = collections.OrderedDict()
    for model, prefix, strategy in FRONTIER_G299_ROWS:
        rate = reg(facts, f"{prefix}.{strategy}.G_299.flag_rate")
        ratio_A, (a_lo, a_hi), ratio_B, (b_lo, b_hi) = ratio_and_bounds(rate)
        above_A, above_B = ratio_A > public_max, ratio_B > public_max
        above_A_conservative, above_B_conservative = a_lo > public_max, b_lo > public_max
        rows[f"{model}::{strategy}::frontier_template"] = collections.OrderedDict([
            ("model", model), ("strategy", strategy), ("prompt_family", "frontier_template"),
            ("positive_rate_g299", rate),
            ("ratio_reading_A_ambiguous_as_negative", ratio_A),
            ("ratio_reading_A_wilson_bound", [a_lo, a_hi]),
            ("ratio_reading_B_ambiguous_as_positive", ratio_B),
            ("ratio_reading_B_wilson_bound", [b_lo, b_hi]),
            ("above_public_max_reading_A", above_A),
            ("above_public_max_reading_B", above_B),
            ("above_public_max_reading_A_conservative_bound", above_A_conservative),
            ("above_public_max_reading_B_conservative_bound", above_B_conservative),
            ("verdict", "both" if above_A and above_B else
                        ("neither" if not above_A and not above_B else "one")),
            ("verdict_conservative_bound",
             "both" if above_A_conservative and above_B_conservative else
             ("neither" if not above_A_conservative and not above_B_conservative else "one")),
            ("register_source", f"results/accord/VERIFIED_FACTS.json::{prefix}.{strategy}.G_299.flag_rate"),
        ])

    for model, prefix in BASELINE_G299_ROWS:
        rate = reg(facts, f"{prefix}.gold.positive_rate")
        ratio_A, (a_lo, a_hi), ratio_B, (b_lo, b_hi) = ratio_and_bounds(rate)
        above_A, above_B = ratio_A > public_max, ratio_B > public_max
        above_A_conservative, above_B_conservative = a_lo > public_max, b_lo > public_max
        rows[f"{model}::baseline_prompt"] = collections.OrderedDict([
            ("model", model), ("strategy", "baseline_prompt"), ("prompt_family", "baseline_prompt"),
            ("positive_rate_g299", rate),
            ("ratio_reading_A_ambiguous_as_negative", ratio_A),
            ("ratio_reading_A_wilson_bound", [a_lo, a_hi]),
            ("ratio_reading_B_ambiguous_as_positive", ratio_B),
            ("ratio_reading_B_wilson_bound", [b_lo, b_hi]),
            ("above_public_max_reading_A", above_A),
            ("above_public_max_reading_B", above_B),
            ("above_public_max_reading_A_conservative_bound", above_A_conservative),
            ("above_public_max_reading_B_conservative_bound", above_B_conservative),
            ("verdict", "both" if above_A and above_B else
                        ("neither" if not above_A and not above_B else "one")),
            ("verdict_conservative_bound",
             "both" if above_A_conservative and above_B_conservative else
             ("neither" if not above_A_conservative and not above_B_conservative else "one")),
            ("register_source", f"results/accord/VERIFIED_FACTS.json::{prefix}.gold.positive_rate"),
        ])

    n_verdict_both = sum(1 for r in rows.values() if r["verdict"] == "both")
    n_verdict_one = sum(1 for r in rows.values() if r["verdict"] == "one")
    n_verdict_neither = sum(1 for r in rows.values() if r["verdict"] == "neither")

    out = collections.OrderedDict([
        ("provenance", {
            "generated_by": "accord/stats/overcall_ambiguity_readings.py",
            "public_max_source": "results/accord/stats/overcall_decomposition.json"
                                 "::public.max_over_call_ratio",
        }),
        ("n_total", n_total), ("n_confirmed_positive", n_confirmed),
        ("n_ambiguous", n_ambiguous), ("n_clear_negative", n_clear_negative),
        ("reading_A_ambiguous_as_negative", {
            "n_pos": n_pos_A, "base_rate": p_A, "wilson_ci95": [ciA_lo, ciA_hi]}),
        ("reading_B_ambiguous_as_positive", {
            "n_pos": n_pos_B, "base_rate": p_B, "wilson_ci95": [ciB_lo, ciB_hi]}),
        ("public_max_over_call_ratio", public_max),
        ("public_max_over_call_ratio_cell", public_max_cell),
        ("rows", rows),
        ("n_rows", len(rows)),
        ("n_verdict_both", n_verdict_both),
        ("n_verdict_one", n_verdict_one),
        ("n_verdict_neither", n_verdict_neither),
        ("local_register_cross_check", cross_check),
    ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")

    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"n_total={n_total} confirmed={n_confirmed} ambiguous={n_ambiguous} "
          f"clear_negative={n_clear_negative}")
    print(f"reading A (12/299) base_rate={p_A:.4f} ci=[{ciA_lo:.4f},{ciA_hi:.4f}]")
    print(f"reading B (58/299) base_rate={p_B:.4f} ci=[{ciB_lo:.4f},{ciB_hi:.4f}]")
    print(f"public max over-call ratio={public_max:.4f} ({public_max_cell})")
    print(f"verdicts: both={n_verdict_both} one={n_verdict_one} neither={n_verdict_neither} "
          f"of {len(rows)} rows")
    print(f"local cross-check matches register 241/46/12: {cross_check_matches_register} "
          f"(counts: {cross_check['label_clean_counts_per_pair']}, "
          f"n_inconsistent_pairs={cross_check['n_pairs_with_inconsistent_label_clean_across_rows']})")


if __name__ == "__main__":
    main()
