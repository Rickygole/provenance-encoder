# C120 pre-registration: the deployed weak labeller's agreement on the exact join

Committed before the first datum. Outcome appended below; the rule is never edited. Requested by the
Stage 1 red-team pass after C119, so post hoc with respect to the original plan.

## Why
Chapter 4 states that the weak-label file and the gold file share no identifying column, so the
deployed labeller's agreement is not recoverable (C88 built a nursing-side reconstruction with 19
conflicted rows; C90 withdrew its correlation as indistinguishable from a random join). A second,
stricter join already exists in the repository: `accord/stats/reconcile_c28.py::build_join`, which
bridges gold -> annotation_sample_300.csv (pair_id -> admission, drug, rule, both sentences) ->
accord_decontamination_audit.csv (weak_label) on all five keys and asserts a 1:1 result on 299 rows.
The thesis already takes the deployed labeller's 261 of 299 from that join, and the C115 comparison
(deployed_vs_frontier_zeroshot.py) used it. The thesis cannot both use it and say no join exists.

## Rule
The exact five-key join is authoritative if and only if, re-run now: (i) the join returns 299 rows,
(ii) every row carries exactly one weak label (1:1), and (iii) the joined pharmacist label agrees with
the gold file's label on all 299 rows. If all hold (Outcome A), Chapter 4's non-identifiability
paragraph and the 0.0768 reconstructed-join ceiling are replaced by the measured values below, and
C88's reconstruction is named as superseded. If any fails (Outcome C), the text keeps the
non-identifiability statement and removes every pair-level count derived from this join (including
C115's 24 disagreements and the "11 true positives").

## What is computed (aggregates only), on G_299 and G_253
Confusion cells TP FP FN TN; flag rate; Matthews correlation; the marginal-constrained maximum MCC
(the same formula as marginal_feasibility_gate.py); observed MCC as a fraction of that ceiling;
one-sided hypergeometric p for TP given the marginals; and the random-join percentile: the fraction of
10000 random permutations of the weak-label vector (seed 42) whose MCC is at least the observed MCC.
No threshold, no model, no new labels.

## Compliance
Local only; stdout/stderr to ~/mimic_local_only/logs; output JSON holds counts and floats only.

## Outcome
(appended after the run)
