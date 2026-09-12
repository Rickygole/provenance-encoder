"""The family of hypothesis tests the Results chapter reports, enumerated.

WHY THIS EXISTS
---------------
The multiplicity subsection of Chapter 4 stated a count of fourteen tests and
then enumerated a family that, read against the chapter itself, contains
sixteen: the enumeration named one Fisher exact test, on the mechanism split,
and omitted the two Fisher tests on the ambiguity gradient in
Section~\\ref{sec:r-ambiguity}. A hand-typed count in prose cannot survive a
chapter that gains tests, and this one gained two and then did not move.

The count matters. The Bonferroni threshold is 0.05 divided by it, and the
zero-shot ambiguity gradient sits at p = 0.0046: it clears 0.05/14 = 0.0036 and
does not clear 0.05/16 = 0.0031. The stated count was therefore the count under
which the headline gradient cleared correction, which is exactly the direction
of error a reader is entitled to be suspicious of.

So the family is enumerated here instead, one entry per test, each bound to the
artifact and key its p value is read from. The count, the Bonferroni threshold
and the familywise upper bound are computed from the list. Adding a test means
adding an entry, and every derived number moves with it.

WHAT THIS IS NOT
----------------
It is not a correction. Chapter 4 applies no multiple-comparison correction and
says so; this script computes what the threshold would be so that the sentence
quoting it cannot drift from the family it describes. The familywise bound is
the independence bound, 1 - 0.95^k, and the tests are positively dependent, so
it is an upper bound and is labelled one.

Writes: results/accord/stats/test_register.json

    python3 accord/stats/test_register.py
"""

import json
import math
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "accord" / "stats" / "test_register.json"
ALPHA = 0.05

# Each entry: the test, where its p value lives, and which tests it is
# dependent with. `group` names the dependence block; tests sharing a group are
# not independent, which is why the familywise figure is an upper bound only.
#
# `source` is "path::dotted.key" into a committed artifact, or "derived:..."
# for the two values Chapter 4 derives from registered summaries rather than
# reading from a run. A derived entry carries its value explicitly and says so.
FAMILY = [
    # Five Mann-Whitney tests, Table tab:decontam. Three seeds of one model,
    # their ensemble, and the contaminated run, all on the same 253 rows.
    dict(id="mw_seed0", test="Mann-Whitney U", where="tab:decontam",
         what="control model seed 0, scores on G_253", group="decontam_253",
         source="results/accord/decontam/corrected_gold_metrics.json::per_seed.seed0.mannwhitney_p"),
    dict(id="mw_seed1", test="Mann-Whitney U", where="tab:decontam",
         what="control model seed 1, scores on G_253", group="decontam_253",
         source="results/accord/decontam/corrected_gold_metrics.json::per_seed.seed1.mannwhitney_p"),
    dict(id="mw_seed2", test="Mann-Whitney U", where="tab:decontam",
         what="control model seed 2, scores on G_253", group="decontam_253",
         source="results/accord/decontam/corrected_gold_metrics.json::per_seed.seed2.mannwhitney_p"),
    dict(id="mw_ensemble", test="Mann-Whitney U", where="tab:decontam",
         what="three-seed ensemble, scores on G_253", group="decontam_253",
         source="results/accord/decontam/corrected_gold_metrics.json::aggregate.ensemble_mannwhitney_p"),
    dict(id="mw_contaminated", test="Mann-Whitney U", where="tab:decontam",
         what="contaminated released run, same 253 pairs", group="decontam_253",
         source="results/accord/step2_signal_check.json::accord_contra_prob_distribution_by_group.mann_whitney_u_p_value_pos_vs_neg"),

    # Five DeLong tests, Table tab:delong. All five are comparisons against a
    # single reference run, so all five share one dependence block.
    dict(id="delong_ensemble_299", test="DeLong", where="tab:delong",
         what="control ensemble vs ACCORD, G_299", group="delong_vs_one_ref",
         source="results/accord/control/final_report.json::delong_ensemble_vs_accord.gold_all.p"),
    dict(id="delong_seed42_299", test="DeLong", where="tab:delong",
         what="control seed 42 vs ACCORD, G_299", group="delong_vs_one_ref",
         source="results/accord/control/final_report.json::delong_per_seed_vs_accord_gold_all.42.p"),
    dict(id="delong_seed123_299", test="DeLong", where="tab:delong",
         what="control seed 123 vs ACCORD, G_299", group="delong_vs_one_ref",
         source="results/accord/control/final_report.json::delong_per_seed_vs_accord_gold_all.123.p"),
    dict(id="delong_seed2024_299", test="DeLong", where="tab:delong",
         what="control seed 2024 vs ACCORD, G_299", group="delong_vs_one_ref",
         source="results/accord/control/final_report.json::delong_per_seed_vs_accord_gold_all.2024.p"),
    dict(id="delong_ensemble_134", test="DeLong", where="tab:delong",
         what="control ensemble vs ACCORD, G_134", group="delong_vs_one_ref",
         source="results/accord/control/final_report.json::delong_ensemble_vs_accord.gold_defect_free_rule34.p"),

    # One Fisher exact test on the mechanism split.
    dict(id="fisher_mechanism", test="Fisher exact, two-sided",
         where="sec:r-rule-mechanism", what="rules 1 and 2 against rules 3 and 4",
         group="mechanism",
         source="results/accord/exemplar/step2plus_results.json::confounded_proxy_rule12_vs_rule34.fisher_exact.p_value"),

    # One Welch test on the anisotropy control. Chapter 4 derives this from the
    # two registered summaries rather than reading it from a run, so it is
    # recomputed here from those summaries and not read from a p field.
    dict(id="welch_anisotropy", test="Welch two-sample t",
         where="sec:res-rq3", what="control condition vs matched clinical pairs",
         group="anisotropy",
         source="derived: Welch two-sample t on the registered anisotropy "
                "summaries, control mean/std/n against the matched clinical "
                "mean/std/n; see stats.anisotropy_welch.* in VERIFIED_FACTS",
         value=8.48e-05),

    # Two Spearman correlations against the pharmacist's danger rating. Four
    # were computed; the two that reach the page are counted here and the
    # subsection names the other two, which is the disclosure, not the family.
    dict(id="spearman_full299_harm", test="Spearman rank correlation",
         where="sec:r-ranking", what="harm score vs danger rating, all 299",
         group="danger_rating",
         source="results/accord/step5_ranking.json::full_299_harm_score_vs_danger_rating.p_value"),
    dict(id="spearman_pos12_harm", test="Spearman rank correlation",
         where="sec:r-multiplicity", what="harm score vs danger rating, 12 positives",
         group="danger_rating",
         source="results/accord/step5_ranking.json::positives_only_n12_harm_score_vs_danger_rating.p_value"),

    # Two Fisher exact tests on the ambiguity gradient. These are the two the
    # prose count omitted. They are the same model on the same items under two
    # prompts, so they share one dependence block and the second carries much
    # less than its face value.
    dict(id="fisher_ambiguity_zs", test="Fisher exact, one-sided",
         where="sec:r-ambiguity", what="ambiguous vs clear negatives, zero-shot",
         group="ambiguity_gradient",
         source="results/accord/differential/frontier_anthropic_metrics.json::ambiguity_gradient.zero_shot.fisher_p_one_sided"),
    dict(id="fisher_ambiguity_fs", test="Fisher exact, one-sided",
         where="sec:r-ambiguity", what="ambiguous vs clear negatives, few-shot",
         group="ambiguity_gradient",
         source="results/accord/differential/frontier_anthropic_metrics.json::ambiguity_gradient.few_shot.fisher_p_one_sided"),

    # Four replication attempts of the gradient, on a second Anthropic model and
    # on a Llama 3 70B through Bedrock. They are in the family because the
    # chapter reports them. Adding them takes the count from sixteen to twenty
    # and the threshold from 0.0031 to 0.0025, at which Sonnet's few-shot
    # gradient stops clearing; that is the register doing the job a prose count
    # could not, and is recorded as C70.
    dict(id="fisher_ambiguity_haiku_zs", test="Fisher exact, one-sided",
         where="sec:r-ambiguity", what="gradient replication, Haiku 4.5, zero-shot",
         group="ambiguity_gradient_haiku",
         source="results/accord/differential/frontier_anthropic_metrics_claudehaiku4520251001.json::ambiguity_gradient.zero_shot.fisher_p_one_sided"),
    dict(id="fisher_ambiguity_haiku_fs", test="Fisher exact, one-sided",
         where="sec:r-ambiguity", what="gradient replication, Haiku 4.5, few-shot",
         group="ambiguity_gradient_haiku",
         source="results/accord/differential/frontier_anthropic_metrics_claudehaiku4520251001.json::ambiguity_gradient.few_shot.fisher_p_one_sided"),
    dict(id="fisher_ambiguity_llama70b_zs", test="Fisher exact, one-sided",
         where="sec:r-ambiguity", what="gradient replication, Llama 3 70B, zero-shot",
         group="ambiguity_gradient_llama70b",
         source="results/accord/differential/frontier_bedrock_metrics_metallama370binstructv1.json::ambiguity_gradient.zero_shot.fisher_p_one_sided"),
    dict(id="fisher_ambiguity_llama70b_fs", test="Fisher exact, one-sided",
         where="sec:r-ambiguity", what="gradient replication, Llama 3 70B, few-shot",
         group="ambiguity_gradient_llama70b",
         source="results/accord/differential/frontier_bedrock_metrics_metallama370binstructv1.json::ambiguity_gradient.few_shot.fisher_p_one_sided"),

    # The fourth labeller. These two were computed by the C72 run, registered with
    # its other 110 keys, and left out of this list, while the chapter said "four
    # replication attempts on two further labellers" and "six tests". Four
    # labellers were asked and eight tests exist. The omission is the exact defect
    # C68 records one level up: a family understated by the tests that were run
    # most recently. Adding them takes k from 20 to 22 and the threshold from
    # 0.0025 to 0.0023; see C76.
    dict(id="fisher_ambiguity_llama8b_zs", test="Fisher exact, one-sided",
         where="sec:r-ambiguity", what="gradient replication, Llama 3 8B hosted, zero-shot",
         group="ambiguity_gradient_llama8b",
         source="results/accord/differential/frontier_bedrock_metrics_metallama38binstructv1.json::ambiguity_gradient.zero_shot.fisher_p_one_sided"),
    dict(id="fisher_ambiguity_llama8b_fs", test="Fisher exact, one-sided",
         where="sec:r-ambiguity", what="gradient replication, Llama 3 8B hosted, few-shot",
         group="ambiguity_gradient_llama8b",
         source="results/accord/differential/frontier_bedrock_metrics_metallama38binstructv1.json::ambiguity_gradient.few_shot.fisher_p_one_sided"),

    # The four frontier discrimination tests, on the same 253 rows. An earlier version
    # of this list used Mann-Whitney U on the binary label vector. That is the wrong
    # instrument: the comparison is a 2x2 table with fixed margins, the vector is
    # binary so roughly 78 percent of pairs are ties, and the rank test's normal
    # approximation is anti-conservative there. It was anti-conservative in all four
    # cases, by factors of 4 to 54, and every error favoured this thesis's sharpest
    # claim. The exact conditional test was already stored in the same artifacts. One
    # of the four changes verdict: Sonnet few-shot cleared under the rank test and does
    # not clear under the exact one. See C85. These are the same tables, on the same
    # 253 rows, that Table tab:decontam runs five tests on, and
    # they were computable from committed cells throughout. They were outside the
    # family while the family was presented as exhaustive, and they are the only
    # rejections on the clinical task, so the omission ran in the direction that
    # kept the strongest results out of this thesis's own correction. Adding them
    # takes k from 22 to 26 and the threshold from 0.0023 to 0.0019; both still
    # clear it, which is why the omission is a reporting defect rather than a
    # changed conclusion. See C79.
    dict(id="fisher_frontier_sonnet_zs", test="Fisher exact, two-sided",
         where="sec:r-frontier", what="Sonnet 5 zero-shot labels against the pharmacist on G_253",
         group="frontier_discrimination",
         source="results/accord/differential/frontier_anthropic_metrics.json::by_strategy.zero_shot.G_253.fisher_exact_p_two_sided"),
    dict(id="fisher_frontier_sonnet_fs", test="Fisher exact, two-sided",
         where="sec:r-frontier", what="Sonnet 5 few-shot labels on G_253",
         group="frontier_discrimination",
         source="results/accord/differential/frontier_anthropic_metrics.json::by_strategy.few_shot.G_253.fisher_exact_p_two_sided"),
    dict(id="fisher_frontier_haiku_zs", test="Fisher exact, two-sided",
         where="sec:r-frontier", what="Haiku 4.5 zero-shot labels on G_253",
         group="frontier_discrimination",
         source="results/accord/differential/frontier_anthropic_metrics_claudehaiku4520251001.json::by_strategy.zero_shot.G_253.fisher_exact_p_two_sided"),
    dict(id="fisher_frontier_haiku_fs", test="Fisher exact, two-sided",
         where="sec:r-frontier", what="Haiku 4.5 few-shot labels on G_253",
         group="frontier_discrimination",
         source="results/accord/differential/frontier_anthropic_metrics_claudehaiku4520251001.json::by_strategy.few_shot.G_253.fisher_exact_p_two_sided"),
    # C88 added the exemplar joinability test here and C90 removes it. Two grounds,
    # both independent of what it did to the threshold.
    #
    # First, a multiplicity family is defined by inferential target, and that test's
    # target is this project's own bookkeeping rather than any claim about labellers
    # or medication contradictions. This subsection said exactly that in print while
    # counting it in the same denominator, which is an inconsistency inside one page.
    #
    # Second, and worse, it is not a test of association. Correctness for the 23
    # key-joined rows was assessed BY the key join and for the 137 others by the
    # reconstruction, so no row is measured by both instruments and the null it
    # rejects contains the clause "the two instruments have equal error rates",
    # which Section r-exemplar already declares false where it calibrates the
    # reconstruction at 0.8934 and 0.9508 fidelity. Rejecting a null one clause of
    # which the author has already denied in print is arithmetic, not evidence.
    #
    # Removing it restores k to 26, the threshold to 0.001923, and Haiku 4.5's
    # few-shot ambiguity gradient at 0.0018909 to the clearing set. That the removal
    # returns a rejection is why the reasoning is recorded here in full rather than
    # asserted: the direction is favourable, so the ground has to be checkable.
]


def resolve(source):
    """Read a p value out of a committed artifact at path::dotted.key."""
    path, _, key = source.partition("::")
    node = json.load(open(ROOT / path))
    for part in key.split("."):
        node = node[part]
    if not isinstance(node, (int, float)) or isinstance(node, bool):
        raise TypeError(f"{source} is not a number: {node!r}")
    return float(node)


def main():
    tests = []
    for entry in FAMILY:
        e = dict(entry)
        if "value" in e:
            # Derived in the chapter, not read from a run. Recorded as such so
            # that a reader can tell which of the sixteen are read and which
            # are recomputed from registered summaries.
            e["p"] = float(e.pop("value"))
            e["p_is_derived"] = True
        else:
            e["p"] = resolve(e["source"])
            e["p_is_derived"] = False
        tests.append(e)

    k = len(tests)
    threshold = ALPHA / k
    clears = sorted((t for t in tests if t["p"] < threshold),
                    key=lambda t: t["p"])
    nominal = sorted((t for t in tests if t["p"] < ALPHA), key=lambda t: t["p"])

    # Holm, computed for reference only. Chapter 4 applies no correction; this
    # is here so the subsection can say what a correction would do without a
    # second hand-typed number appearing somewhere else.
    ordered = sorted(tests, key=lambda t: t["p"])
    holm = []
    for i, t in enumerate(ordered):
        thr = ALPHA / (k - i)
        if t["p"] >= thr:
            break
        holm.append(t["id"])

    groups = {}
    for t in tests:
        groups.setdefault(t["group"], []).append(t["id"])

    out = {
        "provenance": {
            "generated_by": "accord/stats/test_register.py",
            "method": ("The family is enumerated in the script, one entry per "
                       "test, each bound to the artifact and key its p value is "
                       "read from. The count and every threshold derived from "
                       "it are computed from that list, so a test added to the "
                       "chapter moves them. A prose count cannot."),
            "alpha": ALPHA,
            "correction_applied_in_thesis": ("none; Chapter 4 states this "
                                             "explicitly and no reported p "
                                             "value is to be read as corrected"),
        },
        "n_tests": k,
        "bonferroni_threshold": threshold,
        "familywise_upper_bound": 1.0 - (1.0 - ALPHA) ** k,
        "familywise_bound_is_upper_only_because": (
            "the tests are positively dependent: five Mann-Whitney tests on one "
            "set of 253 rows, five DeLong tests against a single reference run, "
            "two Fisher tests on one model under two prompts, and two Spearman "
            "correlations against one rating. The true familywise rate is lower "
            "and this thesis does not compute it."),
        "dependence_groups": groups,
        "n_dependence_groups": len(groups),
        "clears_bonferroni": [t["id"] for t in clears],
        "nominally_significant_uncorrected": [t["id"] for t in nominal],
        "holm_rejections_for_reference": holm,
        "tests": tests,
    }

    os.makedirs(OUT.parent, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)
    print(f"wrote {OUT.relative_to(ROOT)}\n")
    print(f"{k} tests   Bonferroni {ALPHA}/{k} = {threshold:.6f}   "
          f"familywise upper bound {out['familywise_upper_bound']:.4f}")
    print(f"{len(groups)} dependence groups: "
          + ", ".join(f"{g} ({len(v)})" for g, v in groups.items()))
    print("\nclears Bonferroni: " + (", ".join(out["clears_bonferroni"]) or "none"))
    print("nominal at 0.05 but not corrected: "
          + (", ".join(t["id"] for t in nominal if t not in clears) or "none"))
    print()
    for t in sorted(tests, key=lambda t: t["p"]):
        mark = "*" if t["p"] < threshold else (
            "." if t["p"] < ALPHA else " ")
        print(f" {mark} {t['p']:>10.6f}  {t['id']:<24} {t['test']}")


if __name__ == "__main__":
    main()
