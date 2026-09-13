# C98 pre-registration: a graded local score on the 253 non-ambiguous gold pairs

Committed before the first datum. The outcome is appended below after the run; the
rule above the outcome line is never edited.

## Why

Every local labeller in the thesis emits a hard label, and the marginal feasibility
check bounds only thresholded decisions. The thesis states that a score-emitting
labeller is not bounded that way, because its threshold can be swept, but it never
measures one. This run measures one.

## What runs

- Model: `granite3.3:8b`, served locally by Ollama on the analysis machine. No
  transmission off the machine. Temperature 0, one output token.
- Score per pair: log probability of the first token of "Contradiction" minus log
  probability of the first token of "Not", read from the top-8 first-token log
  probabilities. If either is absent from the top 8 it is floored at -20 and the pair
  is counted as degenerate.
- Pairs: the 253 non-ambiguous gold pairs (12 contradiction, 241 not), pharmacist
  labels from `label_clean`. Ambiguous pairs are excluded, matching the thesis's
  G_253 basis.
- Prompt: fixed in `graded_score_probe.py` at this commit.

## Decision rule

The minimum detectable AUROC for this basis is registered at
`mde_recomputed.strata.gold_ambiguous_excluded.minimum_detectable_auc` (0.7355, at
80% power and two-sided alpha 0.05).

- **Outcome A, discriminates:** AUROC at or above 0.7355 AND one-sided Mann-Whitney
  p below 0.05. Reported as: a graded local score separates the confirmed cases at
  a size this evaluation could resolve.
- **Outcome B, not resolved:** anything else. Reported as: the graded score does not
  reach the effect size this evaluation can resolve; nothing is concluded about a
  smaller effect.
- **Outcome C, invalid:** more than 25% of pairs degenerate, or the run does not
  complete all 253 pairs. Reported as not run; no AUROC is quoted.

Whatever the outcome, it is reported. It is not a test of RQ4 and is not added to
the RQ answers; it is reported once, as a secondary analysis, beside the feasibility
check's scope statement.

## Kill condition

If the run is interrupted, it is restarted from scratch, never spliced.

## Outcome

Recorded 2026-09-13 after the run; the rule above is unchanged.

**Outcome B, not resolved.** All 253 pairs scored, 0 degenerate, 251 distinct score
values, so the score was genuinely graded and ties were not the constraint. AUROC
0.5467, one-sided Mann-Whitney p 0.2934, against a registered minimum detectable AUROC
of 0.7355. A graded local score does not reach the effect size this evaluation can
resolve; nothing is concluded about a smaller effect. Artifact:
`results/accord/differential/graded_score_granite33_g253.json`; register prefix
`gradedscore.*`.
