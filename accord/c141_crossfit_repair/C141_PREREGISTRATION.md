# C141. Repairing C140's Q3 and Q4, which were specification errors

Written and committed BEFORE any datum of this check was read. C140's rule is not
edited; it stands with its outcome. This is a separate, later check that replaces
two of its five quantities and says so.

## What was wrong

**Q3 was not a cross-fit.** C140 scored each pair from stratum rates refitted on
the other 252 pairs. For pair $i$ in stratum $r$ that score is
$(k_r - y_i)/(n_r - 1)$, a function of the pair's own label. Verified: within
every stratum every positive receives a strictly lower score than every negative,
deterministically, for any label vector. The reported AUROC of 0.4412 is the
signature of naive leave-one-out target encoding and carries no information about
out-of-sample performance. The reported in-sample optimism of 0.2066 is the same
artefact.

**Q4 held the standard error fixed in a counterfactual where it does not hold.**
C140 asked whether a cell could reach significance "at AUROC 1.0" while holding
that cell's observed `se_diff`. The DeLong variance of an AUC difference shrinks
as the AUC rises, so the standard error at the alternative is not the standard
error at the observed value. The reported count of cells unreachable at AUROC 1.0
is therefore computed with a variance that does not apply.

## Data

The same 253 pairs, 12 positive, the same committed per-pair caches. No new model
inference. Aggregates only; no note text, no identifiers.

## Q3R. A genuine cross-fit

Repeated stratified K-fold. For each repeat and each fold, the four stratum rates
are fitted on the training folds only and every held-out pair in a stratum
receives the same score, so held-out positives and negatives in one stratum tie
rather than being forced apart. $K = 5$ and $K = 3$ are both reported because at
12 positives the fold count matters. 200 repeats at seed 141. A stratum absent
from a training fold scores its held-out pairs at the training-fold overall
positive rate, which is the only estimate available and is recorded as such.

Reported: mean and standard deviation of the cross-fitted AUROC across repeats,
the 2.5th and 97.5th percentiles across repeats, the in-sample value, and the
difference. A permutation null is computed by the C140 Q2 refit rule, applied to
the cross-fitted statistic, with 2000 draws at seed 141.

## Q4R. Detectability with the variance recomputed at the alternative

For each cell, the minimum AUROC that would reach significance against the rival
is found by solving for $A$ where the required difference equals $z \cdot
\mathrm{SE}(A)$, with $\mathrm{SE}$ recomputed by the Hanley and McNeil variance
at $A$ and at the rival's value, with zero covariance. Zero covariance is
optimistic for the stage and is stated as such. Reported at $\alpha = 0.05$ and
at $\alpha/k$ with $k$ the number of DISTINCT cells, 15, and the denominator
reported as 15 throughout so that divisor and denominator agree.

## Outcomes, fixed now

- **Outcome A, the rival survives cross-fitting.** The Q3R mean cross-fitted
  AUROC exceeds 0.5 and its 2.5th percentile exceeds 0.5. C139's comparison was
  against a real comparator and only its null was wrong.
- **Outcome B, the rival is uninformative but not anti-informative.** The Q3R
  mean is at or above 0.5 but its lower percentile is not. The comparator cannot
  be shown to carry information.
- **Outcome C, the rival is genuinely below chance out of sample.** The Q3R mean
  is below 0.5 and its 97.5th percentile is below 0.5.

Whatever Q3R returns, C140's 0.4412, its 0.2066 optimism, its 0.2086 to 0.6680
interval and its 0.2381 permutation p are withdrawn and replaced, and the
withdrawal of C139's reading stands on C140's Q2 alone, which is unaffected.

- **Q4R has no outcome.** It reports numbers that replace C140's Q4. If the
  corrected count of unreachable cells is zero, the manuscript's claim that eight
  cells could not have returned an answer is withdrawn.

## Kill conditions

- If the 253 pair count, the 12 positive count or the stratum sizes 117/28/63/45
  do not reproduce, the check is KILLED.
- If the recomputed in-sample AUROC is not 0.6478215767634855 exactly, KILLED.
- If any fold assignment places all 12 positives in training or all in test for a
  given repeat, that repeat is discarded and the count of discards is reported.

## What this cannot do

It cannot add power, and it cannot rescue C140's Q3 or Q4. Those are withdrawn
regardless of what this returns.
