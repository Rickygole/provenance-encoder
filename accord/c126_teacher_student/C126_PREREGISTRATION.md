# C126 pre-registration: does the detector reproduce its weak labeller's ranking of the gold pairs?

Committed before the first datum. Outcome appended below; the rule is never edited. Requested by the final red-team
pass after C125, so post hoc with respect to the original plan.

## Why
The deployed weak labeller's labels, used as a binary score, have an AUROC against the pharmacist that follows from
C120's confusion counts: on G_253, (recall + specificity) / 2 = (11/12 + 29/241) / 2, about 0.518. The decontaminated
three-seed detector ensemble has AUROC 0.5145 on the same pairs. The thesis calls the labeller check a precondition
rather than an explanation of the detector's null, but never compares the teacher's ranking with the student's.

## What is computed
Inputs: the exact five-key join (reconcile_c28.build_join, 1:1 on 299 rows, as C120) for the weak label and pharmacist
label; the decontaminated per-seed scores (mean of three seeds) for the 253 non-ambiguous pairs; rule membership.
- Teacher AUROC (weak label as a 0/1 score) and student AUROC (ensemble) on G_253; teacher AUROC on G_299; both on the
  Rules 1 and 2 and the Rules 3 and 4 strata of G_253 (ambiguous pairs dropped).
- Percentile bootstrap 95 percent intervals, B = 10000, seed 42, resampling pairs with replacement and skipping
  single-class resamples (the method of accord/control/report.py::bootstrap_auroc_ci).
- A paired bootstrap of the difference, student AUROC minus teacher AUROC, on G_253 (same resampled pairs for both),
  B = 10000, seed 43.

## Decision (G_253)
- **Outcome A (student reproduces teacher):** the paired 95 percent interval for the difference contains 0 and the
  absolute point difference is below 0.05. The thesis may then say the detector's ranking of the gold pairs is not
  distinguishable from its weak labeller's.
- **Outcome B:** otherwise; the thesis reports the difference and its interval without that sentence.
- Strata are descriptive only (6 positives each).
- Outcome C (nothing reported): the join is not 1:1 on 299 rows, a label disagrees, or the ensemble AUROC does not
  reproduce decontam.aggregate.ensemble_auroc to 4 dp.

## Compliance
Local only; stdout/stderr to ~/mimic_local_only/logs; output JSON holds aggregates only.

## Outcome
(appended after the run)
