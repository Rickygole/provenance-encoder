# C139. Does the graded labeller score beat a pure rule-identity prior?

Committed before the first datum. The rule below is never edited; the outcome is
appended at the end of this file after the run.

## Why this check exists

Chapter 4 now closes the 17 graded labeller configurations as a family and
reports that 16 of 17 rank above chance on G253, 12 of 17 have a bootstrap lower
bound above 0.5, 6 survive a Bonferroni threshold across the family, and the
strongest cell reaches AUROC 0.7936. The same paragraph records that 11 cells
carry a rule cue in one direction or the other.

Pharmacist precision varies roughly fourfold across the four mining rules. A
score that merely tracked which rule generated a pair would therefore rank above
chance on G253 without reading any contradiction. Until that is excluded, the
graded result cannot be promoted, because the thesis already argues that rule
identity leaks into this pipeline (the rule-type head's target sits in its
input).

## The comparison score

For each pair in G253, the rule-prior score is the observed pharmacist precision
of that pair's generating rule, computed on G253 itself:

    s_rule(pair) = (positives in that pair's rule stratum) / (pairs in that stratum)

This is deliberately optimistic for the rule prior: it is fitted in sample on the
same labels it is scored against, so it is an upper reading of what rule identity
alone can do. That makes the comparison conservative in favour of the graded
scores, which is the direction this check should err in.

## Primary test

For each of the 17 cells, a paired DeLong test of the graded score against the
rule-prior score on the same 253 pairs, using accord/control/delong.py. The
family is the same 17 cells, so the corrected threshold is

    alpha / 17 = 0.05 / 17 = 0.0029411764705882353

## Decision rule, fixed now

- **Outcome A, the graded score beats the rule prior.** At least one cell has a
  DeLong two-sided p below alpha/17 with its AUROC above the rule prior's. The
  graded finding is then promoted: named in RQ2, in the contributions, in
  Chapter 5, and in the abstract, with the beaten rule-prior AUROC quoted beside
  it.
- **Outcome B, not separable.** No cell clears alpha/17 in either direction. The
  thesis then states plainly that the graded ranking is not shown to exceed a
  rule-identity prior on these pairs, and the result stays where it is as an
  illustration of the rule-identity leakage the thesis already documents. It is
  not promoted.
- **Outcome C, the rule prior wins.** A cell clears alpha/17 with the rule prior
  above the graded score. Reported as such, in the same places Outcome A would
  have been.

Outcome B and Outcome C are both reported in full. No outcome of this check
licenses dropping it.

## Secondary, registered as descriptive only

The graded score's AUROC computed within each rule stratum separately, for the
strongest cell. Rule identity is constant inside a stratum, so signal that
survives there is not rule identity. Strata with fewer than 2 positives or fewer
than 2 negatives are reported as not estimable rather than pooled away. This is
descriptive because the per-stratum positive counts are too small to carry a test.

## Kill conditions

The arm reports nothing if any of these holds:

1. The rule column does not cover all 253 non-ambiguous pairs.
2. The rule-prior score takes fewer than 2 distinct values on G253.
3. The G253 positive count is not 12, or the pair count is not 253.
4. Any graded cell's per-pair score cannot be joined to the gold pairs one to one.

## Inputs, no new inference

- Graded scores: the committed per-pair caches already used by
  accord/labeller_interval_closure/intervals_17.py (flag_logodds only).
- Gold labels and the rule column: the same two files that generator reads.
- No model is run. This is reanalysis of stored per-pair values.

## Outcome

To be appended by accord/c139_rule_prior/c139.py after the run.

**Outcome B, not separable.** Run on 17 configurations.

The rule-identity prior alone reaches AUROC 0.6478 on G253, bootstrap
95 percent interval 0.4778 to 0.8024,
label-permutation p 0.0367. A score carrying no information
except which of the four rules generated a pair therefore ranks above chance on
this set at the uncorrected level.

Against that prior, **no cell clears the corrected threshold of
0.0029411765**, and none clears 0.05 uncorrected either.
15 of 17 sit
above the prior pointwise. The strongest cell, promptconfirm::granite3.3:8b::P_rev_agree, reaches
0.7936 against the prior's 0.6478,
a difference of +0.1457 with a paired DeLong two-sided p
of 0.1396.

Under the rule fixed above this is Outcome B. The graded finding is **not
promoted**. The thesis states that the graded ranking is not shown to exceed a
rule-identity prior on these pairs, and the result stands as a further instance
of the rule-identity leakage the thesis already documents elsewhere.

The registered descriptive secondary is suggestive in the other direction and is
reported as descriptive only: within the strongest cell, AUROC by rule stratum is
1: 0.7339, 2: 0.5600, 3: 0.8771, 4: 0.8953,
on 3, 3, 4, 2
positives respectively. Rule identity is constant inside a stratum, so signal
there is not rule identity, but two to four positives per stratum cannot carry a
test and no claim is made from them.
