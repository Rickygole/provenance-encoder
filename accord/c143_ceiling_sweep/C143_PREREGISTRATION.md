# C143 pre-registration: does the ceiling-normalised coefficient carry decision value?

Written and committed before any datum of this analysis is read. The decision
rules and the kill condition below are fixed and will not be edited afterwards;
the outcome is recorded in this file under a separate heading once the run has
completed.

## Why

The manuscript recommends computing MCC_max from (N, K, n) and reporting the
observed coefficient as a fraction of it. That recommendation has never been
tested against any outcome. A referee observed that the only corpus in the
project carrying a downstream outcome, Demonstration 1, may show the
ceiling-normalised statistic ranking labellers WORSE than the raw coefficient,
which if true makes the recommendation as written unsupported by the paper's
own evidence.

The manuscript also demonstrates the ceiling only on a clinical stage its own
figure caption calls visibly degenerate from the flag rate alone, so it never
shows the check binding where it would be non-obvious.

## Data

`results/accord/c136_gate_validation/c136.json`, 31 labellers, each carrying
pilot marginals (`flag_count`, `pilot_true_positives_flagged`), the gate frame
N = 300 with K = 12, and a downstream outcome `student_auroc_minilm`. No new
inference is run. The clinical corpus is not touched.

## Quantities

For each labeller, from the pilot confusion matrix implied by its marginals:
observed MCC; MCC_max by Equation 1; the ratio observed/MCC_max where MCC_max
is non-zero.

## Pre-registered questions and decision rules

**Q1. Does the ratio rank labellers by downstream usefulness better than the
raw coefficient?** Spearman rho of each against `student_auroc_minilm`, over
the labellers with a defined ratio. Decision rule, fixed now:

- Outcome A, the ratio helps: rho(ratio) exceeds rho(raw) by more than 0.05.
- Outcome B, indistinguishable: the two rho differ by 0.05 or less.
- Outcome C, the ratio hurts: rho(raw) exceeds rho(ratio) by more than 0.05.

Under Outcome B or C the manuscript's recommendation is narrowed in the same
revision to "publish MCC_max as context for interpreting a reported
coefficient", and the instruction to report the observed value as a fraction of
the ceiling is withdrawn. We commit to that narrowing now, before seeing which
outcome obtains.

**Q2. Does a labeller exist in the non-degenerate regime where the ceiling
changes the reading of its coefficient?** A labeller qualifies if its flag rate
lies inside the band Figure 2 computes, 0.1225 to 0.5534, and either its ratio
exceeds 0.90 while its raw coefficient is below 0.50, or its ratio is below
0.40 while its raw coefficient is above 0.20. Qualifying labellers are reported
by name with their marginals.

**Q3.** Report the full 31-labeller table so a reader can check Q1 and Q2.

## Kill condition

No threshold, no outcome definition, no choice of outcome variable and no
labeller inclusion rule in this file may be changed after the first number is
read. `student_auroc_minilm` is the outcome variable; the TF-IDF arm is not an
alternative outcome and will not be substituted. If the analysis cannot be run
as specified, it is reported as not run, not as run differently.

Sample size is fixed at whatever the artifact holds and no power claim is made:
with 31 units and a grid we chose, no rho here estimates a population quantity,
and both rho are reported as descriptions of this grid.

## Outcome

To be completed after the run.
