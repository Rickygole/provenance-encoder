# C116 pre-registration: paired AP tests on a common scale

Committed before the first datum. Outcome appended below; the rule is never edited.
Committee-requested (final Reviewer 2 pass after C115), so post hoc with respect to the
thesis's original plan.

## Why
C111 task b (TF-IDF minus detector ensemble, p 0.00005) and C114 task A (NLI minus detector
ensemble, p 0.10035) use `c111_run.paired_swap_perm_ap`, which swaps RAW scores between the two
systems pair by pair. The three systems score on different scales, so a swapped slot mixes scales
and its ranking depends on which system each pair was drawn from; the null is not the intended
exchangeability null. Average precision depends only on the within-system ranking, so the scale-free
version replaces each system's scores by their within-system rank before swapping.

## What runs
No model is trained except the TF-IDF refit, reproduced exactly as in C111 (same vectoriser,
classifier, seed; gated on reproducing the registered TF-IDF AP on G_299 and G_253 to 4 dp).
Inputs, gates and G_253 frame identical to C111. Each system's G_253 scores are transformed to
average ranks divided by n (scipy.stats.rankdata, method "average"); observed AP is unchanged by
this transform (gate: equal to the raw-score AP to 1e-12). Then `paired_swap_perm_ap` unchanged,
B = 20000, seed 42, two-sided, for:
- T1: TF-IDF minus detector ensemble;
- T2: NLI minus detector ensemble;
- T3: TF-IDF minus NLI (descriptive consistency check).

## Decision
- **T1 p < 0.05 and difference > 0**: the thesis may keep "TF-IDF's average precision exceeds the
  detector's on G_253", citing C116 instead of C111 task b.
- **Otherwise**: every claim that TF-IDF (or any simple model) finds a signal the detector lacks is
  removed; the text keeps only that TF-IDF's and NLI's AP exceed their own label-permutation nulls
  (C111 task a) and the detector's does not.
- T2 below 0.05 with a positive difference would allow the same statement for NLI.
- Outcome C (nothing reported): any C111 gate fails or the rank transform changes an observed AP.

## Compliance
Local only; stdout/stderr redirected to ~/mimic_local_only/logs; output JSON holds aggregates only.

## Outcome
(appended after the run)
