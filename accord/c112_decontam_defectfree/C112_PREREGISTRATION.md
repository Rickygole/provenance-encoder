# C112 pre-registration: re-running C110's decision rule on the decontaminated detector

Committed before the first datum. The outcome is appended below after the run; the
rule above the outcome line is never edited.

## Why

`C110_PREREGISTRATION.md` (`accord/defectfree_rescore/C110_PREREGISTRATION.md`) tested
whether the detector's chance-level ranking on the 253 non-ambiguous gold pairs (G_253)
is concentrated in the Rule 1 and Rule 2 rows that carry the exemplar-selection defect,
by re-scoring on the Rules 3 and 4 stratum (dosage / omission pairs, which never had
that defect). C110 used the column `accord_contra_prob` from
`~/mimic_local_only/derived/gold_standard_accord_results.csv`. C111
(`accord/c111_robustness/C111_PREREGISTRATION.md`, `results/accord/c111_robustness/`)
established that this column is the superseded CONTAMINATED configuration (see
`CORRECTIONS.md`, C16), while the thesis's headline detector result is the
decontaminated three-seed ensemble in
`results/accord/decontam/corrected_gold_metrics.json`. C110's answer is therefore about
an instrument the thesis does not report.

This run re-executes C110's decision rule, unchanged in form, on the decontaminated
per-seed scores in
`~/mimic_local_only/from_drive_2026-09-13/outputs/gold_eval_decontam_predictions.csv`
(columns `prob_seed0`, `prob_seed1`, `prob_seed2`, covering the 253 non-ambiguous
pairs). The Rules 3+4, ambiguous-dropped stratum C110 used (n=108, 6 positives) is a
subset of those 253 pairs, so the test can be run on the detector configuration the
thesis actually reports.

**C110 is not corrected or withdrawn by this run.** It remains reported as run, on the
contaminated column, with its own Outcome A recorded in its own file. This is a
separate, later pre-registration answering the same question on a different,
non-superseded instrument. Nothing in C110's text is edited.

## What runs

No new model is trained or scored. Everything below reads scores that already exist.

- **Detector (decontaminated, three-seed ensemble)**: `prob_seed0`, `prob_seed1`,
  `prob_seed2` from
  `~/mimic_local_only/from_drive_2026-09-13/outputs/gold_eval_decontam_predictions.csv`
  (253 rows, one row per non-ambiguous gold pair; `y_true` column carried for the
  reproduction check below). The ensemble score is the unweighted mean of the three
  `prob_seed*` columns per pair -- confirmed as the definition
  `results/accord/decontam/corrected_gold_metrics.json` itself uses for its
  `aggregate.ensemble_auroc` field (its `note` field: "Decontaminated configuration
  recomputed from gold_eval_decontam_predictions.csv"; the same mean-of-three-seeds
  construction is used by `accord/c111_robustness/c111_run.py`, which reproduces the
  identical `aggregate.ensemble_auroc` value from the identical file before using it).
- **Rule labels**: the `rule` column (1/2/3/4) from
  `~/mimic_local_only/derived/gold_standard_accord_results.csv` (299 rows), joined on
  `pair_id` -- the same file and column C110 used for stratification.
- **Gold labels**: `label_clean` from
  `~/mimic_local_only/from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv`,
  joined on `pair_id` -- the same source and column C110 used, not the detector file's
  own `pharm_label` column and not `gold_eval_decontam_predictions.csv`'s own `y_true`
  column.

### Reproduction check (mandatory, before any stratum number is used)

Recompute, on all 253 pairs in `gold_eval_decontam_predictions.csv`:
- per-seed AUROC (`roc_auc_score(y_true, prob_seedN)` for N in 0,1,2), and
- ensemble AUROC (`roc_auc_score(y_true, mean(prob_seed0, prob_seed1, prob_seed2))`).

Each recomputed value, rounded to 4 decimals, must equal the corresponding value in
`results/accord/decontam/corrected_gold_metrics.json` (`per_seed.seedN.auroc` for N in
0,1,2; `aggregate.ensemble_auroc`), rounded to 4 decimals. If any of the four does not
match, the run aborts with no stratum datum written (Outcome C).

### Label-agreement check (mandatory, before any stratum number is used)

Two joins are checked, both required to show 0 mismatches:
1. `gold_standard_accord_results.csv`'s own `pharm_label` column against
   `pharmacist_labeled_clean.csv`'s `label_clean` column, joined on `pair_id` over all
   299 rows (the check C110 already performed on this pair of files).
2. `gold_eval_decontam_predictions.csv`'s own `y_true` column against a label derived
   from `pharmacist_labeled_clean.csv`'s `label_clean` (1 if `label_clean ==
   "contradiction"` else 0), joined on `pair_id` over the 253 rows common to both
   files.

Any mismatch in either check is reported as-is; if the join in either check does not
cover the expected row count (299 for the first, 253 for the second) or any mismatch
count is nonzero, the run aborts (Outcome C).

Any script touching `~/mimic_local_only` redirects stdout/stderr to a file under
`~/mimic_local_only/logs/`; only the aggregate JSON this script writes under
`results/accord/c112_decontam_defectfree/` and a short numeric summary reach real
stdout. That log file is never read back as part of this task.

## Strata (fixed now)

`gold_eval_decontam_predictions.csv` covers only the 253 non-ambiguous gold pairs, so
"ambiguous dropped" is already the state of this file; there is no "ambiguous as
negative" version to build from it (unlike C110, which worked from the full 299-row
contaminated file and had to construct both versions).

1. **Decision stratum -- Rules 3+4, ambiguous dropped.** Rule in {3, 4}, restricted to
   the 253-pair join. Expected n=108, positives=6, matching C110's cell 3
   (`rules34_ambiguous_dropped` in `results/accord/defectfree_rescore/
   dfrescore_strata.json`) exactly, since it is the same rows minus the exemplar-defect
   instrument swapped for the decontaminated one. This expected count is checked
   against C110's artifact before use; a mismatch is reported, not silently
   reconciled.
2. **Descriptive stratum -- Rules 1+2, ambiguous dropped.** Rule in {1, 2}, restricted
   to the 253-pair join. Expected n=145, positives=6, matching C110's cell 1
   (`rules12_ambiguous_dropped`).

## Metrics (fixed now)

For the decontaminated ensemble and for each of the three individual seeds, on each of
the two strata above: AUROC, average precision, n, positives, and a 20,000-draw
one-sided label-permutation p for AUROC and for AP each, holding the system's scores
fixed and permuting labels, `numpy.random.default_rng`, one shared
`numpy.random.SeedSequence(42)` spawned into per-stratum-per-system child streams
(never two independent roots for the same nominal seed). This mirrors
`accord/defectfree_rescore/dfrescore_run.py`'s `perm_p`, reused or reimplemented
identically.

Only the ensemble's numbers on the Rules 3+4, ambiguous-dropped stratum are
decision-relevant. Everything else in this section (per-seed numbers, and everything
on Rules 1+2) is descriptive only.

## Decision rule (fixed now, same form as C110)

The reference value is the decontaminated ensemble's own AUROC on all 253 pairs, i.e.
the reproduction check's recomputed `aggregate.ensemble_auroc` value (registered at
`results/accord/decontam/corrected_gold_metrics.json`'s `aggregate.ensemble_auroc`,
currently 0.5145228215767634) -- this run's own G_253 basis for the decontaminated
ensemble, in place of C110's G_253 reference for the contaminated column (0.4595, from
`results/accord/VERIFIED_FACTS.json`'s
`signal.contra_prob_vs_pharm_label__ambiguous_excluded_n253.auroc`).

- **Outcome A, the decontaminated detector's null is not concentrated in the defect
  rows**: on the Rules 3+4, ambiguous-dropped stratum, the ensemble's AUROC one-sided
  label-permutation p (20,000 draws, seed 42) is >= 0.05 AND the ensemble's AUROC
  there is within 0.10 of its own G_253 ensemble AUROC (i.e. within
  [0.4145228215767634, 0.6145228215767634] if the reproduction check's recomputed
  value matches the registered 0.5145228215767634 exactly, as expected).
- **Outcome B, the stratum differs**: the stratum exists (>=5 positives), the
  reproduction check and both label-agreement checks passed, but the Outcome-A
  condition above is not met.
- **Outcome C, uninformative**: the reproduction check fails, OR either
  label-agreement check fails (nonzero mismatch or wrong row count), OR the Rules 3+4,
  ambiguous-dropped stratum has fewer than 5 positives.

**Registered interpretation, fixed now and identical in substance to C110's**: with 6
positives on this stratum and a registered minimum detectable AUROC of 0.8058
(`results/accord/mde_recomputed.json`'s `strata.defect_free_subset
.minimum_detectable_auc`, 80% power, two-sided alpha 0.05, n_pos=6, n_neg=128 -- the
same power calculation C110 cited, unchanged by which detector column is scored),
Outcome A fails to show that the null is a defect artifact; it cannot establish that it
is not. This ceiling holds regardless of which outcome letter is recorded below, and
regardless of C110's own recorded outcome.

**Descriptive, not part of the decision rule**: per-seed AUROC and AP with permutation
p, on both the Rules 1+2 and the Rules 3+4 (ambiguous-dropped) strata, for all three
seeds individually alongside the ensemble.

## Kill condition

No stratum definition, metric, or system is changed after any number from this run is
seen. If the reproduction check or either label-agreement check fails, that is reported
as Outcome C; the check is not loosened, retried with different code, or dropped to
make a number appear.

## Compliance

No pair_id, sentence, or row is printed or written outside
`results/accord/c112_decontam_defectfree/` aggregate JSON (counts, AUROC, AP, p-values
only) and this file. Any script touching `~/mimic_local_only` redirects its
stdout/stderr to `~/mimic_local_only/logs/`; that log is never read back as part of
this task. The register (`results/accord/VERIFIED_FACTS.json`) is only read here (for
the pre-existing 0.4595 contaminated reference quoted above, unchanged), never
written; a `c112`-prefixed register script is written for a separate agent to run, and
is not executed as part of this task.

## Outcome

Recorded 2026-09-14 after the run; the rule above is unchanged. Artifact:
`results/accord/c112_decontam_defectfree/c112_defectfree.json`. Script:
`accord/c112_decontam_defectfree/c112_run.py`.

**Label-agreement checks.** `pharm_label` (in `gold_standard_accord_results.csv`) vs
`label_clean` (in `pharmacist_labeled_clean.csv`): 0 mismatches over the 299-row join.
`y_true` (in `gold_eval_decontam_predictions.csv`) vs a `label_clean`-derived label: 0
mismatches over the 253-row join. Both checks passed; Outcome C does not apply on this
ground.

**Reproduction check.** Recomputed on all 253 pairs from
`gold_eval_decontam_predictions.csv`: seed0 AUROC 0.5494, seed1 AUROC 0.5090, seed2
AUROC 0.5048, ensemble (mean of the three `prob_seed*` columns) AUROC 0.5145
(0.5145228215767634 at full precision). All four match
`results/accord/decontam/corrected_gold_metrics.json`'s `per_seed.seed0.auroc`
(0.5494467496542186), `per_seed.seed1.auroc` (0.5089903181189488),
`per_seed.seed2.auroc` (0.5048409405255878), and `aggregate.ensemble_auroc`
(0.5145228215767634) to 4 decimals. Reproduction check passed; Outcome C does not
apply on this ground either.

**Strata.** Rules 3+4, ambiguous-dropped (decision cell): n=108, positives=6 -- exact
match to C110's `rules34_ambiguous_dropped` cell
(`results/accord/defectfree_rescore/dfrescore_strata.json`). Rules 1+2,
ambiguous-dropped (descriptive): n=145, positives=6 -- exact match to C110's
`rules12_ambiguous_dropped` cell. Both stratum row counts matched their C110
counterparts exactly, as expected since the row set is identical and only the
detector instrument changed. The decision cell clears the >=5-positives floor, so
Outcome C does not apply on that ground either.

**Numbers (seed 42, B=20,000 permutation draws per AUROC/AP per system per
stratum):**

| stratum | system | n | pos | AUROC | AP | p_perm(AUROC) | p_perm(AP) |
|---|---|---|---|---|---|---|---|
| Rules 3+4, ambig. dropped | ensemble | 108 | 6 | 0.6095 | 0.1264 | 0.1911 | 0.1763 |
| Rules 3+4, ambig. dropped | seed0 | 108 | 6 | 0.5899 | 0.0997 | 0.2339 | 0.2753 |
| Rules 3+4, ambig. dropped | seed1 | 108 | 6 | 0.5866 | 0.1013 | 0.2464 | 0.2776 |
| Rules 3+4, ambig. dropped | seed2 | 108 | 6 | 0.5882 | 0.1086 | 0.2412 | 0.2343 |
| Rules 1+2, ambig. dropped | ensemble | 145 | 6 | 0.4221 | 0.0421 | 0.7353 | 0.7652 |
| Rules 1+2, ambig. dropped | seed0 | 145 | 6 | 0.5192 | 0.0509 | 0.4342 | 0.5720 |
| Rules 1+2, ambig. dropped | seed1 | 145 | 6 | 0.3681 | 0.0398 | 0.8601 | 0.8193 |
| Rules 1+2, ambig. dropped | seed2 | 145 | 6 | 0.4209 | 0.0404 | 0.7397 | 0.8056 |

G_253 reference (this run's own recomputation, used as the decision rule's reference
value, not the pre-existing contaminated 0.4595): ensemble AUROC 0.5145228215767634,
matching `results/accord/decontam/corrected_gold_metrics.json`'s
`aggregate.ensemble_auroc` exactly (the reproduction check above). The pre-existing
contaminated-column reference C110 used, `signal.contra_prob_vs_pharm_label__
ambiguous_excluded_n253.auroc` = 0.4595, is carried in the artifact for context only
and plays no role in this run's decision.

**Decision.** Decision cell: Rules 3+4, ambiguous dropped (n=108, 6 positives).
Ensemble AUROC there is 0.6095, within 0.10 of its own G_253 ensemble AUROC of 0.5145
(gap 0.0950). Ensemble's AUROC permutation p there is 0.1911, which is >= 0.05. Both
legs of the Outcome-A condition hold.

**Outcome A: the decontaminated detector's null is not concentrated in the defect
rows**, subject to the registered interpretive ceiling below.

**Registered interpretation, restated.** Outcome A holds under the rule fixed above,
but the Rules 3+4 stratum carries only 6 positives against a registered minimum
detectable AUROC of 0.8058. Per the interpretation fixed before this run: Outcome A
cannot establish that the null is not an instrument artifact of the exemplar-selection
defect; it only fails to show that it is, at an effect size below about 0.30
AUROC-points above chance that this evaluation cannot resolve with 6 positives.
Nothing stronger is claimed. This is the same registered ceiling C110 stated, applied
now to the decontaminated instrument; both runs land on Outcome A, but neither run
resolves the question the ceiling describes.

**C110 vs C112, side by side (descriptive, not decided on).** On the same 108-row,
6-positive decision cell: C110's contaminated `accord_contra_prob` scored AUROC
0.5114 (p_perm 0.4686); this run's decontaminated ensemble scores AUROC 0.6095
(p_perm 0.1911) -- higher point estimate, still not permutation-significant, still
Outcome A by the same rule form applied to each instrument's own G_253 reference. The
two runs answer the same question about two different, non-interchangeable
instruments and are not pooled or averaged here.

**What could not be done / was not done.** No new model runs were made. All four
systems' (ensemble, seed0, seed1, seed2) stratum numbers come from the existing
`gold_eval_decontam_predictions.csv` score file; no NLI or TF-IDF baseline was rerun
in this pre-registration (out of scope for this specific re-run, unlike C110, which
covered three systems). The `c112`-prefixed register script
(`accord/c112_decontam_defectfree/register_c112.py`) was written but not run, per
instruction; the numbers above are quoted from
`results/accord/c112_decontam_defectfree/c112_defectfree.json` directly, not from the
central register.
