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
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accord.differential.model_tags import LEGACY_TAGS, ollama_model_tag  # noqa: E402

OUT = ROOT / "results" / "accord" / "stats" / "test_register.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
ALPHA = 0.05
# New prefix for this rebuild. Chosen and checked against
# accord/hosted_withdrawal.py's HOSTED_ARTIFACT_TOKENS: "testfam" is not a
# substring of any of them, and every source this family reads from (see
# FAMILY below) is already live and non-withdrawn in the register under its
# own prefix (decontam.*, control.*, exemplar/step2plus, step5_ranking,
# scalestep.*, nlibaseline.*, promptsens.*, tfidfperm.*, gradedscore.*,
# frontier_ollama_*), so nothing this script now points at can make
# withdraw_hosted.py flag testfam.* either directly or by derivation.
PREFIX = "testfam"

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

    # The Welch test on the anisotropy control was listed here. The thesis no
    # longer reports that control, so the test is in no family it prints.

    # Four Spearman correlations against the pharmacist's danger rating. The
    # thesis prints all four (one in the architecture section, three in the
    # multiplicity subsection) and counts all four in its uncorrected family.
    dict(id="spearman_full299_harm", test="Spearman rank correlation",
         where="sec:r-ranking", what="harm score vs danger rating, all 299",
         group="danger_rating",
         source="results/accord/step5_ranking.json::full_299_harm_score_vs_danger_rating.p_value"),
    dict(id="spearman_pos12_harm", test="Spearman rank correlation",
         where="sec:r-multiplicity", what="harm score vs danger rating, 12 positives",
         group="danger_rating",
         source="results/accord/step5_ranking.json::positives_only_n12_harm_score_vs_danger_rating.p_value"),
    dict(id="spearman_full299_contra_prob", test="Spearman rank correlation",
         where="sec:r-multiplicity", what="contradiction probability vs danger rating, all 299",
         group="danger_rating",
         source="results/accord/step5_ranking.json::full_299_contra_prob_vs_danger_rating.p_value"),
    dict(id="spearman_pos12_contra_prob", test="Spearman rank correlation",
         where="sec:r-multiplicity", what="contradiction probability vs danger rating, 12 positives",
         group="danger_rating",
         source="results/accord/step5_ranking.json::positives_only_n12_contra_prob_vs_danger_rating.p_value"),

    # REMOVED (this rebuild): sixteen tests that lived here across C70, C72,
    # C76, C79, C85, C88 to C92 -- two hosted Fisher tests on the ambiguity
    # gradient (Sonnet), four hosted replications of it (Haiku 4.5, Llama 3
    # 70B, Llama 3 8B, all through Bedrock or the Anthropic API), and ten
    # hosted frontier-discrimination Fisher tests (Sonnet, Haiku 4.5, Llama 3
    # 70B, Llama 3 8B, zero-shot and few-shot each). PhysioNet's "Use of MIMIC
    # Data with Large Language Models and Online Services" (24 September 2025)
    # withdrew every run this thesis made through a hosted channel; see
    # accord/hosted_withdrawal.py. Their source artifacts
    # (frontier_anthropic_metrics*.json, frontier_bedrock_metrics*.json) are no
    # longer on disk, and every one of their p values is now unreachable except
    # under withdrawn.hosted.* in the register. A test whose only source is a
    # withdrawn artifact cannot be re-verified, so it cannot stay in a family
    # this script re-derives from artifacts on every run: FAMILY entries that
    # `resolve()` cannot resolve are a hard failure, not a silent gap, which is
    # what caught this rebuild's predecessor (see the class of defect CLAUDE.md
    # calls "a register key maintained by hand"). Removing them is not a
    # verdict on C70 to C92's reasoning about the family's membership rule; it
    # is that the tests those corrections argued about no longer exist to be a
    # member of anything. The two prior grounds for exclusion recorded at C88
    # and C90 (wrong inferential target; not a test of association) remain true
    # of the exemplar joinability test and are why it is not resurrected here
    # either.
    #
    # ADDED (this rebuild): every hard-label or graded-score hypothesis test
    # this project has run and committed since, none of them hosted. C99's
    # scale step and C100's NLI baseline replace nothing removed above; they
    # are new local runs. C103's prompt-sensitivity family and C107's TF-IDF
    # permutation family are new local runs too. The eight local
    # frontier_ollama_* ambiguity-gradient Fisher tests were already computed
    # and registered (frontier_ollama_*.ambiguity_gradient.*.fisher_p_one_sided)
    # but were never added to this family; leaving a computed, registered,
    # local test out of the family it belongs to is the same defect C68 and
    # C72 record for the hosted replications, so they are added now rather than
    # left for a future correction to catch. C98's graded-score Mann-Whitney is
    # the same story.

    # C99: one McNemar test, a within-family scale step (qwen3:8b vs
    # qwen3:14b, both local Ollama, Q4_K_M) on whether flagging changes with
    # model scale. See accord/differential/C99_PREREGISTRATION.md.
    dict(id="mcnemar_scalestep_c99", test="McNemar exact, two-sided",
         where="C99", what="qwen3:8b vs qwen3:14b flag agreement, local scale step",
         group="scalestep_c99",
         source="results/accord/differential/scale_step_c99.json::mcnemar_exact_two_sided_p"),

    # C100: the NLI baseline (roberta-large-mnli, local) scored against gold on
    # its own primary basis (Mann-Whitney) and DeLong'd against the ACCORD
    # detector on both G_253 and G_299. One model, one scored run, three
    # statistics off it, so one dependence block.
    dict(id="mw_nlibaseline_g253", test="Mann-Whitney U", where="C100",
         what="roberta-large-mnli entailment score vs gold, G_253", group="nlibaseline",
         source="results/accord/differential/nli_roberta_large_mnli.json::primary_g253.mannwhitney_p_one_sided"),
    dict(id="delong_nlibaseline_vs_detector_g253", test="DeLong", where="C100",
         what="roberta-large-mnli vs ACCORD detector AUROC, G_253", group="nlibaseline",
         source="results/accord/differential/nli_roberta_large_mnli.json::delong_vs_detector.g253.p"),
    dict(id="delong_nlibaseline_vs_detector_g299", test="DeLong", where="C100",
         what="roberta-large-mnli vs ACCORD detector AUROC, G_299", group="nlibaseline",
         source="results/accord/differential/nli_roberta_large_mnli.json::delong_vs_detector.g299.p"),

    # C103: three local models (llama3:8b, granite3.3:8b, qwen3:8b), each
    # under three prompts (P_asserting, P_neutral, P_reversed) on G_299/G_253.
    # Each model contributes one McNemar test (its three prompts' paired flag
    # agreement) and three Mann-Whitney tests (each prompt's score vs gold on
    # G_253), all on the same 299/253 items, so each model is its own
    # dependence block -- the same "one model, same items, multiple prompts"
    # rule the removed ambiguity-gradient entries used.
    dict(id="mcnemar_promptsens_llama3_8b", test="McNemar exact, two-sided",
         where="C103", what="llama3:8b, flag agreement across P_asserting/P_neutral/P_reversed",
         group="promptsens_llama3_8b",
         source="results/accord/prompt_sensitivity/analysis.json::model_verdicts.llama3:8b.mcnemar_p_two_sided_exact"),
    dict(id="mw_promptsens_llama3_8b_asserting", test="Mann-Whitney U", where="C103",
         what="llama3:8b, P_asserting score vs gold, G_253", group="promptsens_llama3_8b",
         source="results/accord/prompt_sensitivity/analysis.json::per_model_prompt.llama3:8b.P_asserting.mannwhitney_p_one_sided_253"),
    dict(id="mw_promptsens_llama3_8b_neutral", test="Mann-Whitney U", where="C103",
         what="llama3:8b, P_neutral score vs gold, G_253", group="promptsens_llama3_8b",
         source="results/accord/prompt_sensitivity/analysis.json::per_model_prompt.llama3:8b.P_neutral.mannwhitney_p_one_sided_253"),
    dict(id="mw_promptsens_llama3_8b_reversed", test="Mann-Whitney U", where="C103",
         what="llama3:8b, P_reversed score vs gold, G_253", group="promptsens_llama3_8b",
         source="results/accord/prompt_sensitivity/analysis.json::per_model_prompt.llama3:8b.P_reversed.mannwhitney_p_one_sided_253"),
    dict(id="mcnemar_promptsens_granite33_8b", test="McNemar exact, two-sided",
         where="C103", what="granite3.3:8b, flag agreement across P_asserting/P_neutral/P_reversed",
         group="promptsens_granite33_8b",
         source="results/accord/prompt_sensitivity/analysis.json::model_verdicts.granite3.3:8b.mcnemar_p_two_sided_exact"),
    dict(id="mw_promptsens_granite33_8b_asserting", test="Mann-Whitney U", where="C103",
         what="granite3.3:8b, P_asserting score vs gold, G_253", group="promptsens_granite33_8b",
         source="results/accord/prompt_sensitivity/analysis.json::per_model_prompt.granite3.3:8b.P_asserting.mannwhitney_p_one_sided_253"),
    dict(id="mw_promptsens_granite33_8b_neutral", test="Mann-Whitney U", where="C103",
         what="granite3.3:8b, P_neutral score vs gold, G_253", group="promptsens_granite33_8b",
         source="results/accord/prompt_sensitivity/analysis.json::per_model_prompt.granite3.3:8b.P_neutral.mannwhitney_p_one_sided_253"),
    dict(id="mw_promptsens_granite33_8b_reversed", test="Mann-Whitney U", where="C103",
         what="granite3.3:8b, P_reversed score vs gold, G_253", group="promptsens_granite33_8b",
         source="results/accord/prompt_sensitivity/analysis.json::per_model_prompt.granite3.3:8b.P_reversed.mannwhitney_p_one_sided_253"),
    dict(id="mcnemar_promptsens_qwen3_8b", test="McNemar exact, two-sided",
         where="C103", what="qwen3:8b, flag agreement across P_asserting/P_neutral/P_reversed",
         group="promptsens_qwen3_8b",
         source="results/accord/prompt_sensitivity/analysis.json::model_verdicts.qwen3:8b.mcnemar_p_two_sided_exact"),
    dict(id="mw_promptsens_qwen3_8b_asserting", test="Mann-Whitney U", where="C103",
         what="qwen3:8b, P_asserting score vs gold, G_253", group="promptsens_qwen3_8b",
         source="results/accord/prompt_sensitivity/analysis.json::per_model_prompt.qwen3:8b.P_asserting.mannwhitney_p_one_sided_253"),
    dict(id="mw_promptsens_qwen3_8b_neutral", test="Mann-Whitney U", where="C103",
         what="qwen3:8b, P_neutral score vs gold, G_253", group="promptsens_qwen3_8b",
         source="results/accord/prompt_sensitivity/analysis.json::per_model_prompt.qwen3:8b.P_neutral.mannwhitney_p_one_sided_253"),
    dict(id="mw_promptsens_qwen3_8b_reversed", test="Mann-Whitney U", where="C103",
         what="qwen3:8b, P_reversed score vs gold, G_253", group="promptsens_qwen3_8b",
         source="results/accord/prompt_sensitivity/analysis.json::per_model_prompt.qwen3:8b.P_reversed.mannwhitney_p_one_sided_253"),

    # C107: four permutation tests, task A (the TF-IDF linear model fit on the
    # released training pairs, scored on G_253 and G_299 without a refit, by
    # AUROC and by average precision). Same fitted classifier, same scores,
    # nested bases and two correlated rank statistics: one dependence block.
    dict(id="perm_tfidf_taska_g253_auroc", test="Permutation test (label shuffle)",
         where="C107", what="TF-IDF linear model, AUROC on G_253, no refit", group="tfidfperm_taska",
         source="results/accord/tfidf_permutation/tfidf_permutation.json::task_a.g253.p_permutation_auroc_one_sided"),
    dict(id="perm_tfidf_taska_g253_ap", test="Permutation test (label shuffle)",
         where="C107", what="TF-IDF linear model, average precision on G_253, no refit", group="tfidfperm_taska",
         source="results/accord/tfidf_permutation/tfidf_permutation.json::task_a.g253.p_permutation_ap_one_sided"),
    dict(id="perm_tfidf_taska_g299_auroc", test="Permutation test (label shuffle)",
         where="C107", what="TF-IDF linear model, AUROC on G_299, no refit", group="tfidfperm_taska",
         source="results/accord/tfidf_permutation/tfidf_permutation.json::task_a.g299.p_permutation_auroc_one_sided"),
    dict(id="perm_tfidf_taska_g299_ap", test="Permutation test (label shuffle)",
         where="C107", what="TF-IDF linear model, average precision on G_299, no refit", group="tfidfperm_taska",
         source="results/accord/tfidf_permutation/tfidf_permutation.json::task_a.g299.p_permutation_ap_one_sided"),

    # C98: the graded-score probe's Mann-Whitney, granite3.3:8b log-odds score
    # vs gold on G_253.
    dict(id="mw_gradedscore_granite33_8b_g253", test="Mann-Whitney U", where="C98",
         what="granite3.3:8b graded log-odds score vs gold, G_253", group="gradedscore_granite33_8b",
         source="results/accord/differential/graded_score_granite33_g253.json::mannwhitney_p_one_sided"),

    # The eight local ambiguity-gradient Fisher tests: llama3:8b, granite3.3:8b,
    # qwen3:8b (nothink) and qwen3:14b (nothink), each zero-shot and few-shot,
    # all served locally by Ollama. Already computed and registered under
    # frontier_ollama_*.ambiguity_gradient.*.fisher_p_one_sided (see
    # accord/stats/overcall_ambiguity_readings.py, which reads the same
    # underlying n_ambiguous/n_clear_negative counts); added to this family for
    # the first time here. One model under two prompts per dependence block,
    # the same rule the removed hosted entries used.
    dict(id="fisher_ambiguity_llama3_8b_zs_local", test="Fisher exact, one-sided",
         where="sec:r-ambiguity", what="ambiguous vs clear negatives, llama3:8b (local), zero-shot",
         group="ambiguity_gradient_llama3_8b_ollama",
         source="results/accord/differential/frontier_ollama_metrics_llama3.json::ambiguity_gradient.zero_shot.fisher_p_one_sided"),
    dict(id="fisher_ambiguity_llama3_8b_fs_local", test="Fisher exact, one-sided",
         where="sec:r-ambiguity", what="ambiguous vs clear negatives, llama3:8b (local), few-shot",
         group="ambiguity_gradient_llama3_8b_ollama",
         source="results/accord/differential/frontier_ollama_metrics_llama3.json::ambiguity_gradient.few_shot.fisher_p_one_sided"),
    dict(id="fisher_ambiguity_granite33_8b_zs_local", test="Fisher exact, one-sided",
         where="sec:r-ambiguity", what="ambiguous vs clear negatives, granite3.3:8b (local), zero-shot",
         group="ambiguity_gradient_granite33_8b_ollama",
         source="results/accord/differential/frontier_ollama_metrics_granite33.json::ambiguity_gradient.zero_shot.fisher_p_one_sided"),
    dict(id="fisher_ambiguity_granite33_8b_fs_local", test="Fisher exact, one-sided",
         where="sec:r-ambiguity", what="ambiguous vs clear negatives, granite3.3:8b (local), few-shot",
         group="ambiguity_gradient_granite33_8b_ollama",
         source="results/accord/differential/frontier_ollama_metrics_granite33.json::ambiguity_gradient.few_shot.fisher_p_one_sided"),
    dict(id="fisher_ambiguity_qwen3_8b_zs_local", test="Fisher exact, one-sided",
         where="sec:r-ambiguity", what="ambiguous vs clear negatives, qwen3:8b nothink (local), zero-shot",
         group="ambiguity_gradient_qwen3_8b_ollama",
         source="results/accord/differential/frontier_ollama_metrics_qwen3_8b_nothink.json::ambiguity_gradient.zero_shot.fisher_p_one_sided"),
    dict(id="fisher_ambiguity_qwen3_8b_fs_local", test="Fisher exact, one-sided",
         where="sec:r-ambiguity", what="ambiguous vs clear negatives, qwen3:8b nothink (local), few-shot",
         group="ambiguity_gradient_qwen3_8b_ollama",
         source="results/accord/differential/frontier_ollama_metrics_qwen3_8b_nothink.json::ambiguity_gradient.few_shot.fisher_p_one_sided"),
    dict(id="fisher_ambiguity_qwen3_14b_zs_local", test="Fisher exact, one-sided",
         where="sec:r-ambiguity", what="ambiguous vs clear negatives, qwen3:14b nothink (local), zero-shot",
         group="ambiguity_gradient_qwen3_14b_ollama",
         source="results/accord/differential/frontier_ollama_metrics_qwen3_14b_nothink.json::ambiguity_gradient.zero_shot.fisher_p_one_sided"),
    dict(id="fisher_ambiguity_qwen3_14b_fs_local", test="Fisher exact, one-sided",
         where="sec:r-ambiguity", what="ambiguous vs clear negatives, qwen3:14b nothink (local), few-shot",
         group="ambiguity_gradient_qwen3_14b_ollama",
         source="results/accord/differential/frontier_ollama_metrics_qwen3_14b_nothink.json::ambiguity_gradient.few_shot.fisher_p_one_sided"),

    # C109: three permutation tests, llama3:8b's three NEW reversed-prompt
    # paraphrases (P_rev_agree, P_rev_follow, P_rev_compatible) on G_253,
    # exactly the family C109_PREREGISTRATION.md's primary decision rule is
    # built from ("at least 2 of the 3 NEW paraphrases" clearing both an
    # AUROC floor and a permutation p; the AUROC floor is a threshold
    # comparison with no p value of its own, so only the permutation test
    # enters a p-value family). One model, one basis (G_253), three
    # paraphrases: one dependence block, the same "one model under multiple
    # prompts on the same items" rule every other prompt-sensitivity block in
    # this family uses. The determinism-check cell (P_reversed_rerun, a
    # byte-for-byte rerun of C103's own P_reversed, not a new paraphrase) is
    # excluded, the same way this family excludes a rerun/replication cell
    # that reproduces a test already counted elsewhere rather than posing a
    # new one. granite3.3:8b's parallel C109 cells (per_model_prompt in the
    # same artifact) are scored but are NOT part of C109's primary decision
    # rule (see "primary_paraphrase_verdicts_llama3_8b" in the artifact,
    # llama3:8b only) and are not quoted as a decision in the chapter text;
    # disclosed, not counted, the same convention this file already applies
    # to the two Spearman correlations that do not reach the page.
    #
    # C109 ran its own internal Bonferroni correction over these same three
    # tests (alpha/3 = 0.016667, bonferroni_alpha_over_3 in the artifact) to
    # reach its own outcome (B: 0 of 3 cleared). That correction is orthogonal
    # to this family's Bonferroni threshold, which corrects across every test
    # this chapter reports, not within one pre-registered sub-decision; a test
    # can be Bonferroni-corrected within its own pre-registration and still be
    # a single, uncorrected member of this larger family, exactly as every
    # other pre-registered block here (C99, C100, C103, C107) already is.
    dict(id="perm_c109_llama3_8b_rev_agree", test="Permutation test (label shuffle)",
         where="C109", what="llama3:8b, P_rev_agree AUROC vs shuffled labels, G_253",
         group="c109_llama3_8b_reversed_paraphrases",
         source="results/accord/prompt_sensitivity/c109/analysis.json::per_model_prompt.llama3:8b.P_rev_agree.permutation_p_253"),
    dict(id="perm_c109_llama3_8b_rev_follow", test="Permutation test (label shuffle)",
         where="C109", what="llama3:8b, P_rev_follow AUROC vs shuffled labels, G_253",
         group="c109_llama3_8b_reversed_paraphrases",
         source="results/accord/prompt_sensitivity/c109/analysis.json::per_model_prompt.llama3:8b.P_rev_follow.permutation_p_253"),
    dict(id="perm_c109_llama3_8b_rev_compatible", test="Permutation test (label shuffle)",
         where="C109", what="llama3:8b, P_rev_compatible AUROC vs shuffled labels, G_253",
         group="c109_llama3_8b_reversed_paraphrases",
         source="results/accord/prompt_sensitivity/c109/analysis.json::per_model_prompt.llama3:8b.P_rev_compatible.permutation_p_253"),
]


# The frozen legacy map, exactly as it shipped in C94/C98-era runs. Frozen
# here as a literal rather than re-derived from LEGACY_TAGS, so a change to
# model_tags.py that alters the map is caught by a diff against this literal,
# not just against itself.
_FROZEN_LEGACY_TAGS = {
    ("llama3:8b", None): "_llama3",
    ("granite3.3:8b", None): "_granite33",
    ("qwen3:8b", None): "_qwen3",
}

# Every (label, think) combination this project has actually produced or is
# about to produce, used to check model_tags.py's invariants. Not every
# combination that could ever exist: the point is to catch a regression in the
# labels this project uses, not to enumerate the whole input space.
_KNOWN_MODEL_THINK_PAIRS = [
    ("llama3:8b", None),
    ("granite3.3:8b", None),
    ("qwen3:8b", None),
    ("qwen3:8b", False),
    ("qwen3:8b", True),
    ("qwen3:14b", False),
    ("qwen3:14b", True),
]


def check_model_tags():
    """Invariants accord/differential/model_tags.py must hold.

    1. The frozen legacy map is unchanged.
    2. No two of the known (label, think) pairs produce the same tag: a
       collision here is exactly the qwen3:14b/qwen3:8b defect this file
       exists to close.
    3. No tag equals another tag with "_zero" or "_few" appended. The runner
       builds filenames as f"...{model_tag}_{strategy}..." with strategy
       "zero_shot" or "few_shot"; a tag that is itself another tag plus a
       "_zero"/"_few" fragment would make two distinct (model, strategy)
       filenames differ only in where the model/strategy boundary falls,
       which is the "model tag in the wrong filename position" defect
       CLAUDE.md records twice.
    """
    if LEGACY_TAGS != _FROZEN_LEGACY_TAGS:
        raise AssertionError(
            f"model_tags.LEGACY_TAGS changed from the frozen map: "
            f"{dict(LEGACY_TAGS)} != {_FROZEN_LEGACY_TAGS}")

    tags = {}
    for label, think in _KNOWN_MODEL_THINK_PAIRS:
        tag = ollama_model_tag(label, think)
        if tag in tags:
            raise AssertionError(
                f"tag collision: {label!r} think={think!r} and "
                f"{tags[tag]!r} both produce {tag!r}")
        tags[tag] = (label, think)

    all_tags = list(tags)
    for t1 in all_tags:
        for t2 in all_tags:
            if t1 == t2:
                continue
            for frag in ("_zero", "_few"):
                if t1 == t2 + frag:
                    raise AssertionError(
                        f"tag {t1!r} ({tags[t1]!r}) equals tag {t2!r} "
                        f"({tags[t2]!r}) plus {frag!r}")
    print(f"model_tags: PASS  ({len(all_tags)} known tags, no collisions, "
          f"legacy map unchanged)")


def resolve(source):
    """Read a p value out of a committed artifact at path::dotted.key.

    A naive split on "." breaks for the C103 model_verdicts entries: the model
    name "granite3.3:8b" itself contains a dot, so "model_verdicts.granite3.3:
    8b.mcnemar_p_two_sided_exact".split(".") does not put the model name back
    together as one segment. Walked greedily instead: at each step, the
    LONGEST remaining run of dot-joined parts that is an actual key of the
    current node is consumed as one segment, so "granite3" and "3:8b" recombine
    into "granite3.3:8b" before "llama3:8b" (which has no internal dot and is
    never split) would need any different handling.
    """
    path, _, key = source.partition("::")
    node = json.load(open(ROOT / path))
    parts = key.split(".")
    i = 0
    while i < len(parts):
        for j in range(len(parts), i, -1):
            candidate = ".".join(parts[i:j])
            if isinstance(node, dict) and candidate in node:
                node = node[candidate]
                i = j
                break
        else:
            raise KeyError(f"{source}: no key at or after {parts[i]!r} "
                           f"(remaining path {'.'.join(parts[i:])!r})")
    if not isinstance(node, (int, float)) or isinstance(node, bool):
        raise TypeError(f"{source} is not a number: {node!r}")
    return float(node)


def main():
    check_model_tags()
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

    # The family Chapter 4 prints: the tests no pre-registration covers. Every
    # other group belongs to a pre-registered check, which the chapter reads
    # against its own rule as a separate decision family.
    def uncorrected(t):
        return (t["group"] in ("decontam_253", "delong_vs_one_ref", "mechanism", "danger_rating")
                or t["group"].startswith("ambiguity_gradient_"))
    unc = [t for t in tests if uncorrected(t)]
    unc_by_kind = {}
    for t in unc:
        kind = "ambiguity_gradient" if t["group"].startswith("ambiguity_gradient_") else t["group"]
        unc_by_kind[kind] = unc_by_kind.get(kind, 0) + 1

    out = {
        "uncorrected_family": {"n_tests": len(unc), "by_kind": unc_by_kind,
                               "ids": [t["id"] for t in unc]},
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
            "the tests are positively dependent, in "
            + str(sum(1 for v in groups.values() if len(v) > 1))
            + " dependence groups of more than one test: five Mann-Whitney tests "
            "on one set of 253 rows (decontam_253), five DeLong tests against a "
            "single reference run (delong_vs_one_ref), two Spearman correlations "
            "against one rating (danger_rating), one Mann-Whitney and two DeLong "
            "tests off a single NLI baseline run (nlibaseline), one McNemar and "
            "three Mann-Whitney tests per C103 model under three prompts on the "
            "same items (promptsens_llama3_8b, promptsens_granite33_8b, "
            "promptsens_qwen3_8b), four permutation tests off one fitted TF-IDF "
            "model on nested bases (tfidfperm_taska), four pairs of one model "
            "under two prompts each for the local ambiguity gradient "
            "(ambiguity_gradient_*_ollama), and three permutation tests off "
            "llama3:8b's three C109 reversed-prompt paraphrases on one basis "
            "(c109_llama3_8b_reversed_paraphrases). The true familywise rate is "
            "lower and this thesis does not compute it."),
        "dependence_groups": groups,
        "n_dependence_groups": len(groups),
        "clears_bonferroni": [t["id"] for t in clears],
        "nominally_significant_uncorrected": [t["id"] for t in nominal],
        "holm_rejections_for_reference": holm,
        "tests": tests,
    }

    os.makedirs(OUT.parent, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)

    # Register the family size and dependence-group outputs under a NEW
    # prefix (see PREFIX above), never by hand: this deletes and rewrites its
    # own namespace wholesale on every run, the same convention every other
    # generator in this project follows (CLAUDE.md: "a register key
    # maintained by hand" is the defect this closes).
    facts = json.load(open(FACTS), object_pairs_hook=__import__("collections").OrderedDict)
    for fk in [fk for fk in facts if fk.split(".")[0] == PREFIX]:
        del facts[fk]

    def put(key, value, node):
        facts[f"{PREFIX}.{key}"] = __import__("collections").OrderedDict(
            [("artifact", f"results/accord/stats/test_register.json::{node}"),
             ("value", value)])

    put("n_tests", out["n_tests"], "n_tests")
    put("uncorrected_family.n_tests", out["uncorrected_family"]["n_tests"],
        "uncorrected_family.n_tests")
    for kind, n in unc_by_kind.items():
        put(f"uncorrected_family.by_kind.{kind}", n, f"uncorrected_family.by_kind.{kind}")
    put("bonferroni_threshold", out["bonferroni_threshold"], "bonferroni_threshold")
    put("familywise_upper_bound", out["familywise_upper_bound"], "familywise_upper_bound")
    put("n_dependence_groups", out["n_dependence_groups"], "n_dependence_groups")
    put("n_clears_bonferroni", len(out["clears_bonferroni"]), "clears_bonferroni")
    put("n_nominally_significant_uncorrected",
        len(out["nominally_significant_uncorrected"]), "nominally_significant_uncorrected")
    put("n_holm_rejections_for_reference",
        len(out["holm_rejections_for_reference"]), "holm_rejections_for_reference")
    for g, members in groups.items():
        put(f"groups.{g}.n", len(members), f"dependence_groups.{g}.{len(members) - 1}")
    for t in tests:
        put(f"tests.{t['id']}.p", t["p"], f"tests.{tests.index(t)}.p")
        put(f"tests.{t['id']}.p_is_derived", t["p_is_derived"],
            f"tests.{tests.index(t)}.p_is_derived")
    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {sum(1 for fk in facts if fk.startswith(PREFIX + '.'))} "
          f"keys under {PREFIX}.*")

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
