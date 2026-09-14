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

Recorded 2026-09-14 after the run; the rule above is unchanged. Artifact:
`results/accord/defectfree_rescore/dfrescore_strata.json`. Script:
`accord/defectfree_rescore/dfrescore_run.py`.

**Preliminary checks.** Label join: 299/299 rows joined between
`gold_standard_accord_results.csv` and `pharmacist_labeled_clean.csv` on
`pair_id`; 0 mismatches between the detector file's own `pharm_label` and
`pharmacist_labeled_clean.csv`'s `label_clean`. TF-IDF reproduction check:
recomputed AUROC 0.5760743321718931 and AP 0.1570977148911732 on all 299 gold
pairs, exact match to `results/accord/control/baselines.json`'s
`tfidf_logreg.gold_all` (same values to full precision) -- passed, no abort.
Rules-3+4, ambiguous-as-negative stratum n=134, positives=6, matching
`defect_free_strata.json` and `mde_recomputed.json`'s `defect_free_subset`
exactly; the TF-IDF AUROC recomputed there (0.5729166666666666) matches
`baselines.json`'s separately-registered `tfidf_logreg.gold_defect_free_rule34`
exactly, a second independent cross-check that this recomputation reproduces
the existing pipeline rather than a new one. Rules-3+4, ambiguous-dropped
stratum n=108, positives=6 (46 total ambiguous pairs, 26 of them in rules 3/4,
134-26=108). Both decision-relevant strata clear the >=5-positives floor and
the TF-IDF check passed, so Outcome C does not apply.

**Numbers (seed 42, B=20,000 permutation draws per AUROC/AP per system per
stratum):**

| stratum | system | n | pos | AUROC | AP | p_perm(AUROC) | p_perm(AP) |
|---|---|---|---|---|---|---|---|
| Rules 1+2, ambig. dropped | detector | 145 | 6 | 0.4269 | 0.0427 | 0.7285 | 0.7560 |
| Rules 1+2, ambig. dropped | NLI | 145 | 6 | 0.6127 | 0.1165 | 0.1804 | 0.1159 |
| Rules 1+2, ambig. dropped | TF-IDF | 145 | 6 | 0.5635 | 0.0546 | 0.3043 | 0.5143 |
| Rules 1+2, ambig. as neg. | detector | 165 | 6 | 0.4256 | 0.0379 | 0.7332 | 0.7433 |
| Rules 1+2, ambig. as neg. | NLI | 165 | 6 | 0.6080 | 0.0937 | 0.1870 | 0.1409 |
| Rules 1+2, ambig. as neg. | TF-IDF | 165 | 6 | 0.5818 | 0.0508 | 0.2565 | 0.4723 |
| Rules 3+4, ambig. dropped | detector | 108 | 6 | 0.5114 | 0.0688 | 0.4686 | 0.5663 |
| Rules 3+4, ambig. dropped | NLI | 108 | 6 | 0.5114 | 0.2191 | 0.4628 | 0.0542 |
| Rules 3+4, ambig. dropped | TF-IDF | 108 | 6 | 0.5801 | 0.2898 | 0.2698 | 0.0126 |
| Rules 3+4, ambig. as neg. | detector | 134 | 6 | 0.5078 | 0.0555 | 0.4729 | 0.5645 |
| Rules 3+4, ambig. as neg. | NLI | 134 | 6 | 0.5013 | 0.2082 | 0.4928 | 0.0447 |
| Rules 3+4, ambig. as neg. | TF-IDF | 134 | 6 | 0.5729 | 0.2768 | 0.2852 | 0.0082 |

G_253 reference (descriptive, not decided on): detector AUROC 0.4595 (register,
`signal.contra_prob_vs_pharm_label__ambiguous_excluded_n253.auroc`); this run's
own recomputation on the same G_253 mask: detector 0.4595 (matches to 4
decimals), NLI 0.5861, TF-IDF 0.5730.

**Decision.** Decision cell: Rules 3+4, ambiguous dropped (n=108, 6 positives).
Detector AUROC there is 0.5114, within 0.10 of its G_253 AUROC of 0.4595 (gap
0.0519). Detector's AUROC permutation p there is 0.4686, which is >= 0.05.
Both legs of the Outcome-A condition hold.

**Outcome A: the null is not concentrated in the defect rows**, subject to the
registered interpretive ceiling below.

**Ordering comparison (descriptive, not part of the decision rule).** On
G_253, AUROC order is NLI (0.5861) > TF-IDF (0.5730) > detector (0.4595). On
both Rules-1+2 strata (ambiguous dropped and ambiguous-as-negative), the order
is the same: NLI > TF-IDF > detector. On both Rules-3+4 strata, the order is
different: TF-IDF > detector > NLI in both. The reordering is driven mainly by
NLI, whose AUROC falls from about 0.59 on G_253 to about 0.50-0.51 on Rules
3+4, while TF-IDF and the detector both move less. With 6 positives per
stratum this difference in ordering is not itself tested against a null and
is reported only as a descriptive observation, not a finding.

**Registered interpretation, restated.** Outcome A holds under the rule fixed
above, but the Rules-3+4 strata carry only 6 positives against a registered
minimum detectable AUROC of 0.8058 (ambiguous-as-negative version) /
comparable order for the ambiguous-dropped version. Per the interpretation
fixed before this run: Outcome A cannot establish that the null is not an
instrument artifact of the exemplar-selection defect; it only fails to show
that it is, at an effect size below about 0.30 AUROC-points above chance that
this evaluation cannot resolve with 6 positives. Nothing stronger is claimed.

**What could not be done / was not done.** No new model runs were made,
per the task's constraint; all three systems' stratum numbers come from
existing score files or a script-level reproduction of the existing TF-IDF
pipeline. The `dfrescore`-prefixed register script
(`accord/defectfree_rescore/register_dfrescore.py`) was written but not run,
per instruction; the numbers above are quoted from
`results/accord/defectfree_rescore/dfrescore_strata.json` directly, not from
the central register.
