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

Run 2026-09-26, seed 20260926. Artifacts:
`results/accord/c135_dissociation_cost/c135_bias_curve.json` and
`results/accord/c135_dissociation_cost/c135_labeller.json`. Registered under prefix `c135` in
`results/accord/VERIFIED_FACTS.json` (188 keys).

**Outcome A, cost established.** No kill condition fired: the gap is exactly 0 at rate 0 in both pools,
the assertion check's flagged set equals the planted set at every rate in both pools, realized base rates
are 0.040125 and 0.500000 against targets 0.0401 and 0.50, and the provenance audit over this check's own
sources returned no hits.

Headline, `low` pool (true precision 0.040125, matched to the thesis's 12/299), planted rate 0.414:
measured precision 0.023500, gap -0.016625, paired bootstrap 95 percent interval -0.019375 to -0.013875,
relative bias -0.4143. Measured precision understates true precision by 0.016625 (0.013875 to 0.019375).

Bias curve, gap = measured minus true:

| rate | low pool gap | low 95 pct | mid pool gap | mid 95 pct |
| --- | --- | --- | --- | --- |
| 0.000 | 0.000000 | 0.000000 to 0.000000 | 0.000000 | 0.000000 to 0.000000 |
| 0.100 | -0.004000 | -0.005375 to -0.002625 | -0.050000 | -0.054875 to -0.045247 |
| 0.250 | -0.010000 | -0.012250 to -0.007875 | -0.125000 | -0.132250 to -0.117875 |
| 0.414 | -0.016625 | -0.019375 to -0.013875 | -0.207000 | -0.215875 to -0.198000 |
| 0.600 | -0.024125 | -0.027500 to -0.020875 | -0.300000 | -0.310125 to -0.289872 |

The gap equals minus the planted count over the pool size in every cell, residual at most 2.8e-17, so
under this planting model the bias is the closed form minus (defect rate times true precision) and the
curve is a straight line through the origin with slope minus true precision. The Wilson interval on
measured precision stops covering true precision at rate 0.25 and above in the `low` pool, and at 0.10
and above in the `mid` pool.

Student, `mid` pool, AUROC against true labels on the clean held-out set. Primary A, labels from the
displayed evidence: 0.656402, 0.657165, 0.656549, 0.656176, 0.654509 across the five rates, flat, while
the training positive rate falls 0.500 to 0.200. Secondary B, candidate-level labels on displayed text:
0.656402, 0.648008, 0.632636, 0.610888, 0.581553, a fall of 0.074849. The defect costs the student
positive training signal rather than ranking ability when the labeller reads only the exemplar; it costs
ranking ability when the label is attached at candidate level.

Labeller arm: **both models are skipped under the rule above.** The local Ollama server was held for
most of an hour by unrelated local work, and `llama3:8b` completed no rate. `granite3.3:8b` completed
four of five rates before the server was taken again and is recorded with `skipped: true`. By the rule
above no number from this arm is used; its four rows are retained in the artifact as description only.
For the record, those rows fell monotonically: flag rate 0.4200, 0.3750, 0.3200, 0.2625 and accuracy
against the true label 0.8500, 0.8050, 0.7500, 0.6925 at rates 0, 0.10, 0.25, 0.414, with specificity
constant at 0.9257 and recall on true positives falling 0.7727 to 0.4545. The arm needs a rerun on an
idle server before any of it can be claimed.

## OUTCOME, labeller arm rerun (appended 2026-09-27; the rule above is unedited)

Rerun on an idle server, seed 20260926, LLM seed 7, temperature 0, local Ollama socket only. Artifacts:
`results/accord/c135_dissociation_cost/c135_labeller.json` and the derived comparison
`results/accord/c135_dissociation_cost/c135_labeller_vs_gold.json`. Both models completed every planted
rate, so **neither model is skipped** and the arm now stands. 400 candidate slots from the `mid` pool,
the same slots at every rate, realized true precision on those slots 0.495.

Per model and planted rate:

| model | rate | flag rate | accuracy vs true | specificity | recall on true positives |
| --- | --- | --- | --- | --- | --- |
| granite3.3:8b | 0.000 | 0.4200 | 0.8500 | 0.9257 | 0.7727 |
| granite3.3:8b | 0.100 | 0.3750 | 0.8050 | 0.9257 | 0.6818 |
| granite3.3:8b | 0.250 | 0.3200 | 0.7500 | 0.9257 | 0.5707 |
| granite3.3:8b | 0.414 | 0.2625 | 0.6925 | 0.9257 | 0.4545 |
| granite3.3:8b | 0.600 | 0.2050 | 0.6350 | 0.9257 | 0.3384 |
| llama3:8b | 0.000 | 0.0625 | 0.4475 | 0.8812 | 0.0051 |
| llama3:8b | 0.100 | 0.0750 | 0.4600 | 0.8812 | 0.0303 |
| llama3:8b | 0.250 | 0.0850 | 0.4700 | 0.8812 | 0.0505 |
| llama3:8b | 0.414 | 0.0925 | 0.4775 | 0.8812 | 0.0657 |
| llama3:8b | 0.600 | 0.1025 | 0.4875 | 0.8812 | 0.0859 |

Specificity is constant within each model because planting never changes a false positive's displayed
evidence, which is the construction working as registered.

The four `granite3.3:8b` rows the aborted attempt had reached reproduced bit for bit (0.4200, 0.3750,
0.3200, 0.2625 flag rate and 0.8500, 0.8050, 0.7500, 0.6925 accuracy), so the arm is deterministic at
temperature 0 and this rerun supersedes those rows rather than disagreeing with them.

**Does the measured-precision bias survive a fallible judge?** Measured precision under a judge is that
judge's flag rate. The gold judge on these same 400 slots gives 0.4950, 0.4425, 0.3700, 0.2900, 0.2125.

- `granite3.3:8b`: **yes.** Measured precision falls monotonically 0.4200 to 0.2050. From rate 0 to the
  0.414 design rate it drops 0.1575, unpaired Newcombe 95 percent interval 0.0649 to 0.2471, which
  excludes 0. The gold judge drops 0.2050 over the same contrast, so the fallible judge recovers 0.7683
  of the error-free effect and the bias is attenuated by 0.0475. The interval is conservative: the rates
  share their slots, so the paired interval is narrower than the one quoted.
- Because this judge is also pessimistic at rate 0 (gap against true precision -0.0750 with no defect
  present), its total understatement at rate 0.414 is -0.2325 against the gold judge's -0.2050. Annotator
  error and the dissociation defect compound here; they do not cancel.
- `llama3:8b`: **no.** Its measured precision rises 0.0625 to 0.1025, a change of +0.0300 with interval
  -0.0228 to +0.0822 that contains 0. Its recall on true positives is 0.0051 at rate 0, so it almost
  never confirms a genuine contradiction and cannot register the defect at all. Its flag rate does move
  systematically with the displayed evidence and its specificity stays fixed, so it is responding to its
  input rather than emitting constant unparseable text; the script does not retain raw responses, so the
  share of replies that fell through `parse` to 0 is not recoverable from the artifact. Recorded as a
  completed arm with a degenerate judge, not as a skip.

**What this licenses.** The direction and rough magnitude of the bias survive a fallible language-model
annotator that is competent on the task, and the two error sources compound. It does not survive a judge
whose recall on the trigger condition is near zero, which is a floor effect in the judge and not evidence
against the defect. One of two models reproducing the effect is weak support for generality across
annotators, and no claim about human annotators follows from either row.
