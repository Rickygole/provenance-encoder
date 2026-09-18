# C132 pre-registration: does the bounded null survive a cluster bootstrap?

Committed before the first datum. The outcome is appended below after the run; the rule above the outcome line is
never edited. This check was requested by an examiner after the main results were in the thesis, so it is post hoc
with respect to the thesis's original analysis plan, even though its rule is fixed here before its own run. It adds
no test to the thesis's uncorrected family of hypothesis tests (`accord/stats/test_register.py`): it computes no p
value and reports intervals only.

## Why

The thesis's bounded null rests on two percentile-bootstrap upper ends lying below the simulated Mann-Whitney power
floor, 0.7466 (`sizectl108.power_curves.primary_crossing_power_0_80`), on the 253 non-ambiguous gold pairs (G_253,
12 positives, 241 negatives):

- the subject-disjoint decontaminated three-seed ensemble, AUROC 0.5145, interval 0.3210 to 0.7034
  (`decontam.aggregate.ensemble_ci95_lo/hi`, reproduced from local per-row scores by
  `accord/stats/recompute_decontam_aggregate.py`), margin 0.043;
- the reproducible-label retrain ensemble of C130, AUROC 0.5360, interval 0.3454 to 0.7196
  (`c130.g253.ensemble.ci95_lo/hi`), margin 0.027.

Both intervals resample the 253 pairs as if they were independent. The thesis itself discloses that they are not:
several candidate pairs can come from one admission, and one subject can have several admissions. Positive
within-cluster dependence makes a row-level interval too narrow, and the margins above are small. The honest
resolution is to resample the unit that is actually independent and see whether the upper ends still clear the
floor. No number from this check has been computed at the time of writing: the dry run (below) computes no AUROC,
no interval and no cluster count.

## What runs

No model is trained or scored. No hosted API, no Ollama call. Inputs, all local, all existing:

- **Ensemble (i), subject-disjoint decontaminated ensemble:** `prob_seed0..2` and `y_true` from
  `~/mimic_local_only/from_drive_2026-09-13/outputs/gold_eval_decontam_predictions.csv` (253 rows), ensemble score
  the mean of the three seeds, loaded exactly as `recompute_decontam_aggregate.py` loads it (the path object is
  imported from that script, not retyped). Row order: that file's.
- **Ensemble (ii), reproducible-label retrain:** `score` from
  `~/mimic_local_only/accord_c130_reproducible_student/scores_seed{42,123,2024}.csv` (299 rows each), ensemble score
  the mean of the three seeds, restricted to `pharm_label != "ambiguous"`, loaded exactly as
  `c130_run.py::run_full` loads it (directory and seeds imported from that script; same pair_id order checks across
  seeds and against the gold frame). Row order: the gold frame's.
- **Cluster identifiers:** `HADM_ID` (admission) and `SUBJECT_ID` (subject) from
  `accord.control.data.load_gold()`, the project's one pair_id to admission to subject join, joined to each
  ensemble's rows on `pair_id`. They are used only to group row positions inside the process. They are never
  written, printed or logged, and neither is any pair_id or per-row value.

### Task A: cluster structure of G_253 (counts only)
The number of distinct admissions and of distinct subjects among the 253 pairs, and for each level the largest
number of pairs in one cluster and the number of clusters holding more than one pair. Nothing else about the
distribution is reported.

### Task B: cluster bootstrap of the AUROC on G_253
For each of the two ensembles, a percentile cluster bootstrap 95 percent interval (2.5th and 97.5th percentiles) of
the AUROC, resampling whole **admissions** with replacement, and separately whole **subjects** with replacement.
B = 10,000 valid resamples, `numpy.random.default_rng(20260806)` (the project's bootstrap seed,
`accord/control/report.py::BOOT_SEED`), a fresh generator with that seed for each of the four intervals. One resample
draws G cluster indices uniformly with replacement, G being the number of distinct clusters at that level, and takes
every row of every drawn cluster, so the resample's row count varies; clusters are numbered by first appearance in
the ensemble's row order. A resample with no positive or no negative is redrawn and does not count toward B; the
number redrawn and the redraw rate (redrawn divided by all draws attempted) are reported for every interval. If
20 x B draws do not yield B valid resamples the run is invalid (Outcome C).

**Row-level check, same code.** The same function, called with one cluster per row, is draw for draw the row-level
bootstrap of `recompute_decontam_aggregate.py::boot_default_rng` and `accord/control/report.py::bootstrap_auroc_ci`.
It is run first, for both ensembles, and must reproduce the registered intervals 0.3210 to 0.7034 and 0.3454 to
0.7196 to four decimals. **Seed of this check: 42, not 20260806.** Both registered intervals were recorded at seed
42 (`recompute_decontam_aggregate.py`'s declared method; C130's pre-registration), and a percentile interval
reproduces to four decimals only under the seed it was drawn with. Nothing clustered is computed unless both
reproduce. So that the row-level and cluster intervals can also be compared at one seed, the row-level interval at
seed 20260806 is reported beside them; it is descriptive and enters neither the check nor the decision. Each
ensemble's point AUROC is reported with a flag for whether it reproduces its registered value to four decimals.

### Task C: analytic floors at the tipping-point check's post-flip class counts (arithmetic only, no data)
The C114 tipping-point check (`c114.task_b.*`) reaches its three targets after 5, 6 and 7 positive-to-negative
flips, leaving 7, 6 and 5 positives. The analytic minimum detectable AUROC, by the same Hanley and McNeil search
`accord/stats/recompute_mde.py::hanley_mcneil_mde` uses (the function is imported, not re-implemented), two-sided
alpha 0.05, power 0.80, rounded to four decimals as that script rounds, at 7 positives and 246 negatives, 6 and
247, and 5 and 248, so the tipping-point result can be read against floors computed for the class counts it
actually leaves. It touches no data, is not part of the decision below, and is written in every outcome, C included.

### Task D: ambiguous pairs read as POSITIVE (descriptive)
The ensemble AUROC on all 299 pairs with the 46 ambiguous pairs counted as positive (58 positives, 241 negatives),
with a row-level and a subject-level cluster bootstrap interval (B = 10,000, seed 20260806, the Task B function).
Descriptive only: it has no decision rule and the floor 0.7466 is not calibrated to these class counts. It is
computed for ensemble (ii), whose score files cover all 299 pairs. For ensemble (i) it cannot be computed: its
predictions file scores only the 253 non-ambiguous pairs, so no score exists for the 46 pairs this reading relabels.
The artifact records that as unavailable with the reason. No other model's scores are substituted (the
`accord_control` three-seed scores cover 299 pairs but are a different detector).

## Decision rule (fixed now)

The four upper ends are the 97.5th percentiles of Task B: two ensembles by two cluster levels, compared unrounded
with the registered floor 0.7466.

- **Outcome A (the bounded null holds under clustering):** BOTH ensembles' subject-level AND admission-level 95
  percent upper ends are below 0.7466.
- **Outcome B (the bounded null does not hold under clustering):** any of the four upper ends is at or above
  0.7466. The thesis must then state that the evaluation cannot exclude a detectable effect once clustering is
  respected, wherever it now states the bounded null.
- **Outcome C (invalid, no decision recorded):** the row-level check fails to reproduce either registered interval
  to four decimals, or cluster identifiers cannot be joined for every one of the 253 pairs. The join gate is: both
  score sources cover the same 253 unique pair_ids; every one has an admission and a subject identifier; the labels
  agree with the pharmacist label; none is ambiguous; 12 positives and 241 negatives in each. Also invalid: an input
  file missing, a registered value above missing from the register or no longer rounding to the figure quoted here,
  Task D's class counts not 58 and 241 on 299 pairs with a subject identifier for each, or the redraw cap of Task B.

The lower ends, the seed-20260806 row-level intervals, and Tasks A, C and D are reported and decide nothing.

## Kill condition

No level of clustering, seed, B, percentile, ensemble or threshold is changed after any number from this run is
seen. If the outcome is B it is reported as B; a second bootstrap variant (a different seed, a studentised or BCa
interval, a two-stage resample) is not run in search of A, nor, if the outcome is A, in search of B.

## Compliance

Local only. Every code path that reads `~/mimic_local_only` runs with stdout and stderr redirected to a file under
`~/mimic_local_only/logs/`, which is never read back; an unexpected exception sends its traceback there and only its
type name to the terminal. Every output path passes through `accord.cloud_paths.assert_local_only`. The one
artifact, `results/accord/c132_cluster_bootstrap/c132.json`, holds aggregates only: counts, AUROCs, interval ends,
redraw counts and rates, seeds, booleans. No note text, `HADM_ID`, `SUBJECT_ID`, pair_id or per-row value is
written, printed, logged or committed, and this run writes no row-level file at all. `c132_run.py --dry-run` checks
that every input exists, that the registered values are present, and that the join covers all 253 pairs; it computes
no AUROC, interval or cluster count, writes nothing to `results/`, and prints counts and booleans only.
`register_c132.py` (prefix `c132`) deletes its prefix and re-adds it from the artifact; it is written now and run
only after the full run.

## Outcome

(appended after the run)
