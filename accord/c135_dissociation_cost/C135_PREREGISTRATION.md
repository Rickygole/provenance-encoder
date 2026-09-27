# C135 pre-registration: what trigger-exemplar dissociation costs a measured number

Committed before the first datum. The outcome is appended below the outcome line; nothing above that
line is ever edited.

## Why

Chapter 3 (`sec:m-exemplar`) names trigger-exemplar dissociation and Chapter 5 records that its rate is
unresolved (three readings: 0, 0.4745, 0.9315; the rebuilt-pool design rate from C129 is 0.414, Wilson
0.388 to 0.441). The thesis can say the defect exists and that the instruments disagree. It cannot say
what the defect DOES to a measured number, so the 0.0401 base rate carries no bounded interpretation in
the presence of the defect. This check supplies the missing transfer function: planted defect rate to
bias in measured precision, plus the knock-on effect on a weak labeller and on a student model.

## Data. Public only. No MIMIC.

SNLI (Bowman et al. 2015), `stanfordnlp/snli`, already in the local Hugging Face cache at
`~/.cache/huggingface/hub/datasets--stanfordnlp--snli/snapshots/cdb5c3d5eed6ead6e5a341c8e56e669bb666725b/plain_text/train-00000-of-00001.parquet`.
Read-only, via pandas/pyarrow. No network call, no hosted API, no MIMIC file is opened by any script in
this check. Only premise groups with exactly three hypotheses carrying exactly the label multiset
(0, 1, 2) are used (138,927 such groups exist in train), so every group has exactly one contradiction
hypothesis and exactly two non-contradiction hypotheses.

## The analogue, stated exactly

| ACCORD | C135 on SNLI |
| --- | --- |
| admission-drug candidate | one premise group |
| mentions of the drug | the hypotheses of that premise |
| a mention's action | that hypothesis's SNLI gold label |
| set-level conflict fires the rule | the group contains a gold contradiction hypothesis |
| trigger set | the hypotheses whose own gold label is contradiction |
| rebuild selector | display a trigger-set member |
| original positional selector | display the first non-trigger hypothesis by position |
| displayed exemplar | the pair (premise, displayed hypothesis) |
| pharmacist confirming the displayed pair | the gold label of the displayed pair is contradiction |

**True positive candidate**: the group holds a contradiction hypothesis, so the rule genuinely fired.
`true = 1`.
**False positive candidate**: modelled as an upstream extraction error. The group is restricted to its
two non-contradiction hypotheses, and one of them is treated as if the action classifier had read it as
conflicting, which is what admits the candidate to the pool. `true = 0`. Its displayed pair is a
non-contradiction whatever is displayed. TP groups and FP groups are drawn from disjoint premise sets.

**Pool base rate.** Two pools, each N = 8000 candidates, built independently:
- `low` with true precision 0.0401, matched to the thesis's gold base rate 12/299;
- `mid` with true precision 0.50, so intervals on the LLM and student arms are estimable.

## Planting

Planted rates: 0, 0.10, 0.25, 0.414, 0.60. The 0.414 cell is the C129 rebuilt-pool design rate.
At rate r, the eligible set is the true-positive candidates (every one is eligible: each has two
non-trigger hypotheses). Candidates are ordered by one seeded permutation fixed once per pool, and the
first `round(r * n_true_positive)` of them are planted. Nesting is therefore monotone: the rate 0.10
planted set is a subset of the 0.25 set, and so on.
- Planted true positive: displayed hypothesis is the first non-trigger hypothesis by position in the
  group.
- Unplanted true positive: displayed hypothesis is the trigger-set member.
- False positive: displayed hypothesis is the mis-read non-contradiction hypothesis, first by position.
  Planting does not apply to false positives and never changes their displayed evidence.

Seed 20260926, recorded in every artifact.

## What is measured, per pool and per planted rate

**(1) Bias in measured precision.** `measured_precision` is the share of pool candidates whose displayed
pair a judge reading only the displayed evidence would confirm, the judge being the SNLI gold label of
the displayed pair. `true_precision` is the share of pool candidates for which the rule genuinely fired.
Report both with Wilson 95 percent intervals, and `gap = measured_precision - true_precision` with a
paired bootstrap 95 percent interval over candidates, 10,000 resamples, seed 20260926. The headline cell
is `low`, r = 0.414.

**(2) Labeller agreement.** Two local Ollama models, `llama3:8b` and `granite3.3:8b`, temperature 0,
seed 7, over the displayed evidence of 400 candidates per rate drawn from the `mid` pool (the same 400
candidate slots at every rate, so the only thing that changes across rates is the displayed evidence).
Report flag rate, accuracy against the true label, recall on true = 1, specificity on true = 0. If a
model is too slow or unreachable, the arm is reported as skipped and no number from it is used.

**(3) Student AUROC.** TF-IDF (word 1-2 grams, min_df 2) plus logistic regression on
`premise [SEP] displayed hypothesis`. Trained on the `mid` pool. Evaluated as AUROC against TRUE labels
on a clean held-out set of 2000 candidates built from disjoint premises at rate 0, that is with
trigger-based exemplars. Primary, student A: training labels are derived from the displayed evidence,
the same judge as in (1). Secondary, student B: training labels are the candidate-level firing indicator
while the text stays the displayed evidence, which is genuine label noise rather than lost signal. Both
reported; A is primary because it is what a weak labeller reading only the exemplar produces.

**(4) Assertion check.** The one-line trigger-membership assertion is
`displayed_index in trigger_index_set`. Evaluated on true-positive candidates, which is where the defect
is defined. Requirement, at every rate and both pools: the set of candidates the assertion flags equals
the planted set exactly, so zero flags at rate 0, and no trigger-based pick is ever flagged. On false
positives the assertion flags all of them by construction, since their trigger set is empty; that is
reported separately and is a different failure mode, not this defect.

## Decision rule

Let `g` be the point estimate of `gap` in the `low` pool at r = 0.414, and `[lo, hi]` its paired
bootstrap interval.

- **Outcome A, cost established.** `g < 0` and `hi < 0`. The thesis may then write that at the 41.4
  percent design rate, measured precision understates true precision by `|g|` (interval `|hi|` to
  `|lo|`), on this public analogue under this planting model.
- **Outcome B, no effect established.** `[lo, hi]` contains 0. The check reports that the cost on
  measured precision is not established, and the thesis claims nothing about magnitude.
- **Outcome C, opposite sign.** `g > 0` and `lo > 0`. Reported as such: the defect would inflate
  measured precision, and the direction the thesis assumed is wrong.

## Kill conditions. Any one voids the run.

1. At r = 0 the gap is not exactly 0 in either pool. Construction demands equality; a nonzero value is a
   harness defect, not a finding.
2. The assertion check's flagged set differs from the planted set for any rate or pool, in either
   direction.
3. Realized `true_precision` differs from the pool's target base rate by more than 0.0005.
4. Any script in this check opens a path under `local_only/`, `mimic_local_only/`, or reaches a network
   host other than the local Ollama socket.

On a void run nothing is registered and no number reaches prose.

## Scope, stated before the numbers exist

This is a public-data analogue with a planting model chosen by the author. It bounds what the defect does
to a measured precision under that model. It is not a measurement of ACCORD's gold set, it does not
resolve the disagreement among the three defect-rate readings, and it does not convert 0.0401 into a
corrected number. What it can do is attach a signed, bounded interpretation to 0.0401 conditional on a
defect rate.

## OUTCOME (appended after the run; the rule above is unedited)
