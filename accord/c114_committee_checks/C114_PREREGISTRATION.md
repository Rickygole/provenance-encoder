# C114 pre-registration: three checks requested by the committee after C113

Committed before the first datum. The outcome is appended below after the run; the rule
above the outcome line is never edited. These checks are requested after the committee
read results already in the thesis, so every result here is reported as post hoc with
respect to the thesis's original analysis plan, even though each rule is fixed before
its own run.

## Why

The C113 committee round found three claims the existing artifacts do not carry:

1. The thesis says two simple models (TF-IDF and an off-the-shelf NLI model) find a
   precision signal the trained detector lacks. Only TF-IDF was compared against the
   detector (C111 task b). NLI was compared only against TF-IDF, and a non-significant
   difference does not show equivalence.
2. The detector gold-label tipping point (7 flips to reach 0.7355) was computed from the
   CONTAMINATED released detector column (AUROC 0.4595), not from the decontaminated
   three-seed ensemble the thesis reports (AUROC 0.5145).
3. The defect-free stratum result (C112, Rules 3+4 ambiguous-dropped, n=108, 6 positives)
   is reported with no interval.

## What runs

No model is trained or scored. Inputs, all local, all existing:
- decontaminated per-seed scores `prob_seed0..2` and `y_true`,
  `~/mimic_local_only/from_drive_2026-09-13/outputs/gold_eval_decontam_predictions.csv` (253 rows);
- NLI scores `contra_prob_mean` and `label_clean`,
  `~/mimic_local_only/derived/nli_scores_roberta_large_mnli.csv`;
- rule membership `rule`, `~/mimic_local_only/derived/gold_standard_accord_results.csv`.
Ensemble score = mean of the three seeds, as in C111 and C112.

### Task A: paired average-precision test, NLI against the detector ensemble, G_253
The paired swap-permutation test of `c111_run.paired_swap_perm_ap`, unchanged: B = 20000,
seed 42, statistic AP(NLI) minus AP(ensemble), two-sided.
- **A1**: p < 0.05 and observed difference > 0. The thesis may say the NLI model's average
  precision exceeds the detector's on G_253.
- **A2**: otherwise. The claim "two simple models find a signal the detector lacks" is
  reduced to TF-IDF only; NLI is described only as having average precision above its
  own label-permutation null (C111 task a), with no comparison to the detector.

### Task B: detector tipping point from the decontaminated ensemble, G_253
`gold_tipping_points.greedy_flip_to_target`, unchanged (search cap 80), on the ensemble
scores, for targets 0.7355 (analytic MDE), 0.7466 (C108 two-sided Mann-Whitney 80 percent
power) and 0.8084 (C108 C98-rule 80 percent power). Descriptive; no decision. Whatever it
returns replaces the contaminated 7-flip figure wherever the thesis states the detector's
tipping point; the contaminated figure may remain only labelled as the contaminated run.

### Task C: bootstrap interval, ensemble AUROC on the Rules 3+4 ambiguous-dropped stratum
`accord/control/report.py::bootstrap_auroc_ci`, unchanged: percentile bootstrap, 10000
valid resamples, seed 42, single-class resamples skipped. Descriptive; no decision. The
interval is stated next to C112's stratum AUROC.

## Validity gates (Outcome C, nothing reported beyond the reason)
- the decontaminated file has 253 rows and its `y_true` agrees with the NLI file's
  `label_clean` on every joined pair (0 mismatches, 253 joined);
- the recomputed ensemble AUROC on G_253 equals `decontam.aggregate.ensemble_auroc` to 4 dp;
- the Rules 3+4 stratum has n = 108 and 6 positives, and its ensemble AUROC equals
  `c112.rules34_ambiguous_dropped.ensemble.auroc` to 4 dp.

## Compliance
Local only. All stdout/stderr while credentialed data is in scope goes to a log under
`~/mimic_local_only/logs`, which is not read back. The output JSON holds counts and
aggregate floats only: no pair_id, identifier, row, or sentence text.

## Outcome
(appended after the run)

Run 2026-09-14, artifact `results/accord/c114_committee_checks/c114.json`, all validity
gates passed (253 joined, 0 label mismatches, ensemble AUROC and stratum AUROC reproduced).

- **Task A: A2.** AP(NLI) minus AP(ensemble) = 0.1088, two-sided paired p = 0.10035. The
  claim that two simple models find a signal the detector lacks is reduced to TF-IDF only.
- **Task B.** From the decontaminated ensemble (0.5145), greedy relabelling reaches 0.7355 after
  5 flips, 0.7466 after 6, and 0.8084 after 7. These replace the contaminated-run figure.
- **Task C.** Rules 3+4 ambiguous-dropped stratum (n 108, 6 positives): percentile bootstrap
  95 percent interval 0.3029 to 0.9010 around AUROC 0.6095.
