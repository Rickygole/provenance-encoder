# C133 pre-registration: how many wrong labels until the bounded null's own interval tips?

Committed before the first datum. The outcome is appended below after the run; the rule above the outcome line is
never edited. This check was requested in the project's internal review after the main results were in the thesis,
so it is post hoc with respect to the thesis's original analysis plan, even though its rule is fixed here before its
own run. It adds no test to any family, including the thesis's uncorrected family of hypothesis tests
(`accord/stats/test_register.py`): it computes no p value, decides no hypothesis, and reports intervals and counts
only.

## Why

The thesis defines its bounded null by the UPPER END of a row-level percentile bootstrap 95 percent interval lying
below the simulated Mann-Whitney power floor, 0.7466 (`sizectl108.power_curves.primary_crossing_power_0_80`), on the
253 non-ambiguous gold pairs (G_253, 12 positives, 241 negatives). Two registered intervals carry it, both recorded
at seed 42 with B = 10,000:

- the subject-disjoint decontaminated three-seed ensemble, interval 0.3210 to 0.7034
  (`decontam.aggregate.ensemble_ci95_lo/hi`), margin 0.043;
- the plain control detector retrained on reproducible labels, the C130 ensemble, interval 0.3454 to 0.7196
  (`c130.g253.ensemble.ci95_lo/hi`), margin 0.027.

The thesis's existing tipping-point check (C114 task B, `c114.task_b.*`) measures something else: it counts greedy
label flips until the POINT ESTIMATE reaches a floor, and reports 5, 6 and 7 flips for its three targets. Its search
was free to flip either way and chose positive to negative at every step; it never tested the negative-to-positive
direction on its own. An examiner pointed out that under the thesis's own criterion the null may tip far sooner,
possibly after a single relabelled positive, because the upper ends sit only 0.043 and 0.027 below the floor. A
claim's fragility has to be measured by the criterion the claim is made with. This check replaces the count. No
number from it has been computed at the time of writing: the dry run (below) computes no AUROC, no flip and no
interval.

## What runs

No model is trained or scored. No hosted API, no Ollama call. Inputs, all local, all existing, and the same ones
C132 used, loaded by C132's own code (`c132_run.py::load_inputs`, `load_registered` and `input_paths` are imported,
not re-implemented), so the score files, labels, row orders and join gate cannot differ from C132's:

- **Ensemble (i), subject-disjoint decontaminated ensemble:** `prob_seed0..2` and `y_true` from
  `~/mimic_local_only/from_drive_2026-09-13/outputs/gold_eval_decontam_predictions.csv` (253 rows), ensemble score
  the mean of the three seeds. Row order: that file's.
- **Ensemble (ii), reproducible-label retrain:** `score` from
  `~/mimic_local_only/accord_c130_reproducible_student/scores_seed{42,123,2024}.csv` (299 rows each), ensemble score
  the mean of the three seeds, restricted to `pharm_label != "ambiguous"`. Row order: the gold frame's.
- **Gold labels:** the pharmacist label as C132 checks it (each source's label column must agree with
  `accord.control.data.load_gold()` on every pair; 12 positives and 241 negatives in each ensemble).

**The bootstrap, everywhere in this check.** The registered row-level bootstrap and nothing else: a percentile 95
percent interval (2.5th and 97.5th percentiles) of the AUROC over B = 10,000 valid resamples of the 253 rows with
replacement, `numpy.random.default_rng(42)`, seed 42 being the seed both registered intervals were recorded with,
and a fresh generator with that seed for every interval. A resample with no positive or no negative is redrawn and
does not count toward B; the number redrawn and the redraw rate (redrawn divided by all draws attempted) are
reported for every interval. If 20 x B draws do not yield B valid resamples the run is invalid (Outcome C). The
implementation is vectorised: because the index stream of a generator seeded 42 does not depend on labels or scores,
the attempts are drawn once and shared, and each labelling takes the first B attempts that hold both classes; the
AUROC of a resample is computed from row multiplicities and the pairwise kernel (2 if the first score is higher, 1
if tied), which is the Mann-Whitney form with ties counted one half. This is draw for draw
`recompute_decontam_aggregate.py::boot_default_rng` and C132's row-level check, and the validity check below is the
test that it is.

**The greedy step, both tasks.** At each step, among the rows that currently hold the source label and have not
been relabelled, relabel the single row whose relabelling gives the highest AUROC of the resulting labelling (the
largest rise; if no relabelling raises the AUROC, the smallest fall). Every candidate at one step leaves the same
class counts, so candidates are compared through the exact integer statistic 2U, not through rounded floats. Ties
are broken by pair index order: the lowest row position in the ensemble's own row order, the rule C114's search
also gives. The number of candidates tied for the best is reported at every step. A relabelled row is never
relabelled back. The search is run separately for each ensemble, so the two ensembles may relabel different rows.

### Task A: positives relabelled negative (the direction C114's search took)
For each ensemble, k = 1, 2, ... up to 8 greedy relabellings of a positive as negative, leaving 12 - k positives
and 241 + k negatives. After each flip: the point AUROC, the bootstrap interval above with its redraw count and
rate, and whether the upper end is at or above 0.7466. **k_A** is the smallest k at which the upper end is at or
above 0.7466, or "not reached within 8". All 8 steps are computed and reported whether or not the floor is reached
earlier.

### Task B: negatives relabelled positive (the direction C114 did not test)
The same, greedily relabelling a negative as positive, k = 1 to 8, leaving 12 + k positives and 241 - k negatives,
and **k_B** defined the same way.

### Validity check (k = 0)
Before any relabelling, the same function on the unaltered labels must reproduce both registered intervals, 0.3210
to 0.7034 and 0.3454 to 0.7196, to four decimals. Nothing is relabelled unless both reproduce. Each ensemble's point
AUROC is reported with a flag for whether it reproduces its registered value to four decimals.

### Floors at changed class counts (information only, arithmetic, no data)
Each flip changes the class counts, and a power floor depends on them. **The floor is nevertheless held at its
registered value, 0.7466, at every step of both tasks**, because that is the number the thesis's claim is stated
against, and a moving threshold would make the count unreadable. For the positive-to-negative direction the floors
at changed counts are the ones C132 task C reports (`c132.task_c.*`). For the negative-to-positive direction they
are recomputed here analytically, at 12 + k positives and 241 - k negatives for k = 1 to 8, with
`accord/stats/recompute_mde.py::hanley_mcneil_mde` (imported, not re-implemented; two-sided alpha 0.05, power 0.80,
rounded to four decimals as that script rounds). They touch no data, are not part of the reading rule, and are
written in every outcome, C included. They are analytic Hanley and McNeil floors and the registered 0.7466 is a
simulated Mann-Whitney floor, so they inform the reading and are not substitutes for it.

## Reading rule (fixed now)

This check decides no hypothesis; it replaces a count. The four counts are k_A and k_B for each of the two
ensembles. An upper end is compared unrounded with the registered floor 0.7466. "Not reached within 8" counts as 3
or more.

- **Outcome A:** for BOTH ensembles, k_A and k_B are all 3 or more. The bounded null, as the thesis defines it,
  survives two wrong labels of either kind, even when they are the two most damaging ones.
- **Outcome B:** any of the four counts is 1 or 2. The thesis must then state that the bounded null, as it defines
  it, does not survive that many wrong labels, and must give the count wherever it discusses fragility.
- **Outcome C (invalid, no reading recorded):** the k = 0 check fails to reproduce either registered interval to
  four decimals. Also invalid: an input file missing; a registered value above missing from the register or no
  longer rounding to the figure quoted here; C132's join gate failing (both score sources cover the same 253 unique
  pair_ids, labels agree with the pharmacist label, none ambiguous, 12 positives and 241 negatives in each); the
  redraw cap; a relabelling step that would leave a class empty.

In either valid outcome the four counts are reported as they are, and the thesis quotes them in place of, or beside,
the C114 point-estimate counts wherever it discusses fragility. The lower ends, the point AUROCs, the redraw rates,
the tie counts and the floors at changed counts are reported and decide nothing.

## What the counts do and do not mean

- **A greedy search picks the most damaging labels first.** The counts therefore bound what targeted, adversarial
  labelling error could do. They say little about random error: a label chosen at random is far less damaging than
  the one the search picks, and nothing here estimates how likely any labelling error is. A small count means the
  claim rests on few labels, which is already visible in the 12 positives; it does not mean the labels are wrong.
- **Greedy is not exhaustive.** The search maximises the point AUROC one step at a time, the rule C114 used, and
  reads the interval afterwards. It does not search for the set of k labels that maximises the upper end, so a
  count from it is an upper bound on the smallest number of relabellings that could tip the interval, not the
  minimum itself.
- **The floor is held fixed** although each flip changes the class counts, as stated above.
- **Rows are treated as independent**, as in the two registered intervals. C132 already reports that those
  intervals hold under admission-level and subject-level clustering; this check does not repeat that.
- **No pair is re-annotated by this check.** It is a sensitivity analysis of the claim, not an audit of the
  annotation.

## Kill condition

No direction, step rule, tie rule, cap of 8, seed, B, percentile, ensemble, floor or outcome threshold is changed
after any number from this run is seen. If the outcome is B it is reported as B; a second variant (a different
seed, a BCa or studentised interval, a random-flip simulation, a floor moved to the changed class counts, a
different search) is not run in search of A, nor, if the outcome is A, in search of B. A later check that asks a
different question, random labelling error for one, would need its own pre-registration and would not replace this
outcome.

## Compliance

Local only. Every code path that reads `~/mimic_local_only` runs with stdout and stderr redirected to a file under
`~/mimic_local_only/logs/`, which is never read back; an unexpected exception sends its traceback there and only its
type name to the terminal. Credentialed inputs are reached only through `MIMIC_LOCAL_ONLY_DIR`. Every output path
passes through `accord.cloud_paths.assert_local_only`. The one artifact,
`results/accord/c133_interval_tipping/c133.json`, holds aggregates only: counts, AUROCs, interval ends, redraw counts
and rates, tie counts, seeds, booleans. **A flip is recorded by its step number only**: no pair_id, row position,
`HADM_ID`, `SUBJECT_ID`, score, note text or per-row value is written, printed, logged or committed, and this run
writes no row-level file at all. `c133_run.py --dry-run` checks that every input exists, that the registered values
are present, and that C132's join gate passes; it computes no AUROC, flip or interval, writes nothing to
`results/`, and prints counts and booleans only. `register_c133.py` (prefix `c133`) deletes its prefix and re-adds
it from the artifact; it is written now and run only after the full run.

## Outcome

(appended after the run)
