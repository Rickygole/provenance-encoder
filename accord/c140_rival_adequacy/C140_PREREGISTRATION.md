# C140. Is the rival check runnable at this gold-set size, and does the rival rank above chance when its fitting is modelled?

Written and committed BEFORE any datum of this check was read. The rule below is
fixed. It is not edited after the run; the outcome is appended in a separate
section at the end.

## Why this check exists

C139 compared 17 graded labeller configurations against a rule-identity prior and
reported Outcome B, not separable. Three objections to that reading were raised in
review and none can be answered from the C139 artifact:

1. The prior is fitted in sample. Its four stratum rates are estimated from the
   same 12 positives it is then scored against.
2. The permutation null that gives the prior its p of 0.0367 holds the fitted
   score vector fixed while permuting the labels the score was fitted to. Under
   the null an analyst would have refitted.
3. C139 never computed what difference its paired DeLong tests could detect. The
   registered detectability floor of 0.7355 bounds the against-chance test, not
   the rival test.

## Data

The same 253 non-ambiguous pairs, 12 positive, and the same committed per-pair
score caches used by C139. No new model inference is run. No MIMIC note text is
read, printed or written. Aggregates only.

## The five quantities, each with its rule fixed here

**Q1. Rule-label association.** Pearson chi-square on the 2x4 table of rule by
label, and an exact permutation test of the same statistic with B = 20000 draws
at seed 140. Reported with the expected counts.

**Q2. The correct permutation null for the fitted prior.** B = 20000 draws at
seed 140. On each draw the labels are permuted AND the four stratum rates are
refitted to the permuted labels before the AUROC is recomputed. Reported
alongside C139's fixed-score p so the two are comparable.

**Q3. Cross-fitted rival.** Leave-one-pair-out: for each pair, the four stratum
rates are fitted on the other 252 pairs and that pair is scored from the fit that
did not see it. AUROC, a bootstrap interval at B = 10000 seed 140, and a
permutation p computed by the Q2 refit rule. Paired DeLong of every one of the 17
cells against this out-of-sample rival.

**Q4. Detectability of the rival comparison.** For each of the 17 cells, from its
committed se_diff: the minimum detectable difference at 80% power at alpha 0.05
and at alpha 0.05/k, the implied AUROC, the power at the observed difference, and
whether the required difference exceeds the arithmetic maximum 1 minus the rival
AUROC. Reported per cell and as a family summary.

**Q5. Deduplicated counts and harmonised sidedness.** Two of the 17
configurations are determinism re-runs of two others. Every family count is
recomputed over distinct cells only, and the Bonferroni divisor is set to the
number of distinct cells. Every reported p is restated one-sided and two-sided so
the against-chance and against-rival families are compared on one footing.

## Outcomes, fixed now

- **Outcome A, the rival survives cross-fitting and the correct null.** Q2 gives
  p < 0.05 and the Q3 cross-fitted AUROC bootstrap lower bound exceeds 0.5. The
  C139 reading stands and the rival is a real comparator.
- **Outcome B, the rival is an in-sample artefact.** Q2 gives p >= 0.05, or the
  Q3 lower bound does not exceed 0.5. The claim that the prior ranks above chance
  is withdrawn, and with it the mechanism claim of rule-correlated confounding.
- **Outcome C, the comparison was never runnable.** Q4 shows the corrected
  minimum detectable difference exceeds 1 minus the rival AUROC for a majority of
  cells. The non-separation is then a property of the design, and the publishable
  claim is the unrunnability itself, not the null.

Outcomes B and C are not exclusive. If both fire, both are reported.

## Kill conditions

- If the 253 pair count, the 12 positive count, or the four stratum sizes
  117/28/63/45 do not reproduce from the committed CSVs, the check is KILLED and
  writes no result.
- If the recomputed in-sample rival AUROC does not reproduce
  0.6478215767634855 exactly, the check is KILLED: the data access path differs
  from C139's and nothing downstream is comparable.
- If any cell's se_diff is absent from the committed C139 artifact, that cell is
  excluded and the exclusion is reported. The check is not re-specified to
  accommodate it.

## What this check cannot do

It cannot add power. It cannot make 12 positives into more. It cannot establish
that the graded labellers carry signal; a negative result here remains a failure
to demonstrate and not a demonstration of absence.
