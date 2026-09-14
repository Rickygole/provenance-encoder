# C110 pre-registration: is the ranking null an artifact of the exemplar-selection defect?

Committed before the first datum. The outcome is appended below after the run; the
rule above the outcome line is never edited.

## Why

The exemplar-selection bug lives only in Rule 1 and Rule 2 pairs (the selected
mention can be the wrong one). Rules 3 (dosage) and 4 (omission) never had that
bug: their evidence is the dose difference or the absence itself, not a selected
mention. `results/accord/defect_free_strata.json` already shows the ACCORD
detector at chance (AUROC 0.5078, Mann-Whitney p=0.953) on the combined
Rules-3+4 subset (n=134, 6 positives, ambiguous counted as negative), which is
consistent with the null being independent of the defect, but that file does not
report a permutation p, does not cover the ambiguous-dropped version, and covers
only one system (the ACCORD detector itself). This run adds the NLI baseline and
the TF-IDF logistic-regression baseline on the same strata, adds a 20,000-draw
label-permutation p for every AUROC on every stratum, and fixes the decision rule
and its interpretation before any of those numbers are computed.

`results/accord/mde_recomputed.json` registers the minimum detectable AUROC for
the defect-free subset (n=134, 6 positives) at 0.8058 (80% power, two-sided
alpha 0.05). With 6 positives, no stratum built from this gold set can resolve an
effect smaller than that gap. That is stated here, before the run, as the
interpretive ceiling on whatever outcome follows.

## What runs

No new model runs. Everything below reads scores that already exist.

- **Detector**: `accord_contra_prob` from
  `~/mimic_local_only/derived/gold_standard_accord_results.csv` (299 rows; this
  file already carries a `rule` column, 1/2/3/4, verified against
  `defect_free_strata.json`'s per-rule counts: 134/31/79/55). Gold labels are
  taken by joining on `pair_id` against
  `~/mimic_local_only/from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv`,
  column `label_clean` (not the `pharm_label` column already sitting in the
  detector file itself, per instruction). The two label columns are checked for
  agreement before use; a mismatch is reported, not silently resolved.
- **NLI baseline**: `contra_prob_mean` from
  `~/mimic_local_only/derived/nli_scores_roberta_large_mnli.csv` (299 rows, no
  `rule` column of its own). Rule is joined in from the detector file above on
  `pair_id`. Gold labels: the same `label_clean` join as above.
- **TF-IDF logistic regression**: recomputed deterministically, following
  `accord/tfidf_permutation/tfidfperm_run.py` exactly: `accord.control.data.
  build_splits()` (imported, not modified, not re-run as a model -- it is a
  fixed deterministic join + subject-level split, no randomness that varies
  across calls) for the training pool and the fixed 299-pair gold set;
  `TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)` fit on
  `train["nursing_sentence"] + " [SEP] " + train["discharge_sentence"]`,
  `LogisticRegression(max_iter=2000, random_state=42)` fit on the same rows,
  scored on the gold set with the identical `[SEP]`-joined text, no refit on
  gold. **Reproduction check, mandatory before any stratum number is used**:
  the recomputed AUROC and AP on all 299 gold pairs must match
  `results/accord/control/baselines.json`'s `tfidf_logreg.gold_all` (AUROC
  0.5760743321718931, AP 0.1570977148911732) to 4 decimals. If it does not
  match, the run aborts for this system with no datum written, and that
  failure is reported as-is (this is one of the two things that can produce
  Outcome C below). Any script that calls `build_splits()` runs with stdout
  and stderr redirected to a file under `~/mimic_local_only/logs/`; only the
  aggregate JSON this script writes under `results/accord/defectfree_rescore/`
  and a short numeric summary reach real stdout, and that log file is never
  read back by this task.

## Strata (fixed now, four cells)

Built from the `rule` column, ambiguous handled two ways per the task:

1. **Rules 1+2, ambiguous dropped.** n = 134 + 31 - (ambiguous count within
   rules 1/2), positives = contradiction count within rules 1/2.
2. **Rules 1+2, ambiguous as negative.** n = 165, positives = contradiction
   count within rules 1/2 (matches `defect_free_strata.json`'s "rules 1+2"
   row: n=165, pos=6).
3. **Rules 3+4, ambiguous dropped.** n = 108 (134 - 26 ambiguous, per the task
   statement), positives = contradiction count within rules 3/4.
4. **Rules 3+4, ambiguous as negative.** n = 134, positives = 6 (matches
   `defect_free_strata.json`'s "rules 3+4" row and
   `mde_recomputed.json`'s `defect_free_subset`: n_pos=6, n_neg=128).

These counts are computed from the join above and checked against the two
registered files quoted in this document before being used in any metric; a
mismatch is reported, not silently reconciled.

## Metrics (fixed now)

Per system (detector, NLI, TF-IDF) x per stratum (the four cells above):
AUROC, average precision, n, positives, and a 20,000-draw one-sided
label-permutation p for AUROC and for AP each, holding the system's scores
fixed and permuting labels, `numpy.random.default_rng(42)`, one shared
`SeedSequence(42)` spawned into per-stratum-per-system child streams (never
two independent roots for the same nominal seed). This mirrors
`accord/tfidf_permutation/tfidfperm_run.py`'s `perm_null`, reused or
reimplemented identically.

## Decision rule (fixed now)

Read once, before the run: `signal.contra_prob_vs_pharm_label__
ambiguous_excluded_n253.auroc` from `results/accord/VERIFIED_FACTS.json`
(currently 0.4595; this is the detector's G_253 AUROC, ambiguous excluded).

- **Outcome A, the null is not concentrated in the defect rows**: on the
  Rules-3+4, ambiguous-dropped stratum (cell 3 above), the detector's AUROC
  permutation p is >= 0.05 AND the detector's AUROC there is within 0.10 of
  0.4595 (i.e. in [0.3595, 0.5595]).
- **Outcome B, the stratum differs**: cell 3 exists (>=5 positives) and the
  TF-IDF reproduction check passed, but the Outcome-A condition above is not
  met.
- **Outcome C, uninformative**: the Rules-3+4 stratum (either ambiguous
  handling) has fewer than 5 positives, OR the TF-IDF reproduction check
  fails, OR the detector/NLI files fail to join to gold labels for all 299
  pairs.

**Registered interpretation, stated before results**: with 6 positives and a
registered minimum detectable AUROC of about 0.81 on the ambiguous-as-negative
version of this stratum, Outcome A cannot establish that the null is not an
instrument artifact -- it can only fail to show that it is. This ceiling holds
regardless of which outcome letter is recorded below.

**Descriptive, not part of the decision rule**: whether the ordering of the
three systems (detector, NLI, TF-IDF) by AUROC is the same on each of the four
strata as on G_253. G_253 AUROCs for this comparison are read from the same
existing artifacts, not recomputed: detector 0.4595 (register, as above); NLI
and TF-IDF G_253 AUROC are computed by this run's own code on the existing
G_253 mask, reported alongside, not decided on.

## Kill condition

No stratum definition, metric, or system is changed after any number from this
run is seen. If the TF-IDF reproduction check fails, that system's numbers are
reported as not computed (Outcome C applies); the check is not loosened, retried
with different code, or dropped to make a number appear.

## Compliance

No pair_id, sentence, or row is printed or written outside
`results/accord/defectfree_rescore/` aggregate JSON (counts, AUROC, AP, p-values
only) and this file. Any script touching `~/mimic_local_only` redirects its
stdout/stderr to `~/mimic_local_only/logs/`; that log is never read back as
part of this task. The register (`results/accord/VERIFIED_FACTS.json`) is only
read here, never written; a `dfrescore`-prefixed register script is written for
a separate agent to run, and is not executed as part of this task.

## Outcome

(Recorded after the run; the rule above is unchanged.)
