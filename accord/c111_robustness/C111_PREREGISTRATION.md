# C111 pre-registration: detector and NLI permutation tests, a paired
# comparison against TF-IDF, leave-one-positive-out fragility, and
# training-set disjointness for the C107 bag-of-words finding

Committed before the first datum. The outcome is appended below after the run; the
rule above the outcome line is never edited.

## Why

C107 (`accord/tfidf_permutation/C107_PREREGISTRATION.md`) found that a TF-IDF
plus logistic-regression baseline trained on the weak-label pool clears a
one-sided label-permutation null on average precision: p = 0.0109 on $G_{299}$,
p = 0.0159 on $G_{253}$ (10,000 draws, seed 42), while the thesis's own prose
dismisses that ranking as noise without ever testing it. This is the thesis's
most-quoted positive finding attributed to the weak labels. A committee review
raised four objections none of which C107 (or anything since) closed:

1. No detector or NLI average-precision permutation test was ever run the same
   way, so there is no evidence the detector or the off-the-shelf NLI model
   *lack* the same signal, only that TF-IDF has it.
2. No paired comparison of TF-IDF against the detector, or TF-IDF against NLI,
   has been run; C107 and C100 each report their own system against a no-skill
   null, never against each other.
3. C100 (`accord/differential/C100_PREREGISTRATION.md`) already shows an
   off-the-shelf NLI model, never trained on the weak labels, at AP 0.1537
   against TF-IDF's 0.1571 on $G_{299}$ -- nearly identical -- which undercuts
   attributing the C107 signal to the weak labels specifically, since a
   label-free model finds nearly the same thing.
4. No leave-one-positive-out check has been run on the C107 finding. Leave-one-
   positive-out is exactly the fragility test that demoted a different result
   in this project (C107 task (b) itself, and the convention it set), and
   $G_{253}$ has only 12 positives, so a single pair could be carrying the
   ranking.
5. Separately: whether TF-IDF's 1,183-row training pool excludes the 299 gold
   pairs and their subjects was asserted in `accord/control/data.py`'s
   docstring and in-code assertions, but never independently re-derived by a
   script outside that module for this specific claim.

This pre-registration commits, before any number is computed, to closing all
five gaps on existing scores only -- no new model is trained or run.

## What runs

No new model runs. TF-IDF is refit exactly as `tfidfperm_run.py` already does
(the pipeline is identical; only the evaluation wrapped around it is new).
Everything else reads scores that already exist on disk.

### Score sources, decided now

- **TF-IDF**: recomputed exactly as `accord/tfidf_permutation/tfidfperm_run.py`
  task (a) step 1 does -- `accord.control.data.build_splits` (imported,
  unmodified) gives `splits.train` (1,183 rows) and `splits.gold` (299 rows);
  `TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)` fit on
  `splits.train`'s `nursing_sentence + " [SEP] " + discharge_sentence`,
  `LogisticRegression(max_iter=2000, random_state=42)` fit on the same rows'
  labels, scored on `splits.gold`. **Reproduction gate**: the recomputed AP
  must match `results/accord/VERIFIED_FACTS.json`'s
  `tfidfperm.task_a.g299.ap` (0.1570977148911732) and
  `tfidfperm.task_a.g253.ap` (0.1650667451480374) to 4 decimal places. Abort
  (Outcome C, write nothing) if either fails.

- **Detector, per seed and ensemble**: checked before this pre-registration
  and re-checked as a hard gate inside the run script (not reported as a
  registered number itself) --
  `~/mimic_local_only/derived/gold_standard_accord_results.csv` carries no
  seed columns; its only score column, `accord_contra_prob`, is a single run,
  and `accord/CORRECTIONS.md`'s resolution of the "C16" finding establishes
  that column as the **contaminated** configuration (0.4595 AUROC on
  $G_{253}$), superseded by a decontaminated 3-seed retrain whose per-pair
  scores live at
  `~/mimic_local_only/from_drive_2026-09-13/outputs/gold_eval_decontam_predictions.csv`
  (columns `pair_id`, `y_true`, `prob_seed0`, `pred_seed0`, `prob_seed1`,
  `pred_seed1`, `prob_seed2`, `pred_seed2`; 253 rows). That file is **the
  per-seed score source the thesis's decontam tables use**: recomputing AUROC
  and AP from its three `prob_seed*` columns and their row-wise mean
  reproduces `results/accord/decontam/corrected_gold_metrics.json`'s
  `per_seed.seed{0,1,2}` and `aggregate.ensemble_auroc` to full float
  precision, not merely 4 decimals. The run script re-derives this match and
  aborts (Outcome C) if it does not hold to 4 decimals. Ensemble score =
  row-wise mean of `prob_seed0`, `prob_seed1`, `prob_seed2`, matching
  `accord/control/report.py`'s own definition of an ensemble.
  **This file covers exactly the 253 non-ambiguous pairs and no others**
  (checked: its 253 `pair_id`s are exactly $G_{253}$'s, and its `y_true`
  agrees with `pharm_label`-derived positivity on all 253 rows, 0 mismatches).
  `accord/thesis/msu_ch4_results.tex` (the paragraph around line 220-221)
  states directly that no per-pair score from the decontaminated retrain was
  retained for the ambiguous rows, so **$G_{299}$ is not computable for the
  detector from any per-pair decontaminated score, and this is registered now
  as a gap that will be reported "not done", not filled by a different
  detector configuration.** A single supplementary, explicitly
  descriptive-only, quantity is registered alongside it: the same
  label-permutation AP test on $G_{299}$ using the contaminated single-run
  `accord_contra_prob` column, clearly labeled as the superseded
  configuration and excluded from the decision rule below.

- **NLI**: `contra_prob_mean` from
  `~/mimic_local_only/derived/nli_scores_roberta_large_mnli.csv` (299 rows,
  joined by `pair_id`). $G_{253}$ = `label_clean != "ambiguous"` (253 rows);
  $G_{299}$ = all 299 rows. Checked: `label_clean` agrees with `pharm_label`
  on all 299 joined rows (0 mismatches, including 0 disagreements on which
  rows are `"ambiguous"`), so the two label columns may be used
  interchangeably for masking without a reconciliation step.

- **Labels**: `splits.gold["pharm_label"]` throughout (`pharm_pos` = label ==
  `"contradiction"`), the same column `accord/control/data.py` already
  populates and C107 already used. Checked: this agrees with the NLI file's
  `label_clean` and the decontam file's `y_true` everywhere they overlap (0
  mismatches each), so one label vector serves all three systems without a
  reconciliation step.

- **Validity gate on alignment**: the three per-pair score sources (TF-IDF
  via `splits.gold`, NLI via its CSV, detector via the decontam CSV) must
  share exactly 253 common `pair_id`s for the $G_{253}$ basis used in (a),
  (b) and (c). Checked already (253/253 for all three); re-checked as a hard
  gate in the run script. A mismatch is Outcome C.

### (a) One-sided label-permutation p for average precision, 20,000 draws, seed 42

For each of TF-IDF, NLI, detector-seed0, detector-seed1, detector-seed2,
detector-ensemble, on $G_{253}$ and (where the score source covers it)
$G_{299}$: fix the score vector, draw 20,000 permutations of the label
vector, recompute AP each draw, report one-sided p =
`(null_ap >= observed_ap).mean()`. Observed AUROC is also recorded per cell,
descriptively, without its own permutation test (only AP is decision-relevant
here, per the objections above).

One shared `numpy.random.SeedSequence(42)`, spawned into 35 independent
children in one call, in this fixed order, so no two tests share a
permutation stream:

0. detector seed0, $G_{253}$
1. detector seed1, $G_{253}$
2. detector seed2, $G_{253}$
3. detector ensemble, $G_{253}$
4. detector `accord_contra_prob` (contaminated, descriptive only), $G_{299}$
5. NLI, $G_{253}$
6. NLI, $G_{299}$
7. TF-IDF, $G_{253}$
8. TF-IDF, $G_{299}$
9. paired permutation, TF-IDF minus detector ensemble, $G_{253}$ (task b)
10. paired permutation, TF-IDF minus NLI, $G_{253}$ (task b)
11-22. TF-IDF leave-one-positive-out, $G_{253}$, dropped-positive index 0..11 (task c)
23-34. TF-IDF leave-one-positive-out, $G_{299}$, dropped-positive index 0..11 (task c)

Detector on $G_{299}$ (children 4) is descriptive only, per the score-source
note above, and is not part of the decision rule.

### (b) Paired comparison of AP, $G_{253}$ only, two-sided, 20,000 draws, seed 42

Two comparisons: TF-IDF minus detector-ensemble, and TF-IDF minus NLI. For
each of the 253 pairs, independently and with probability 0.5, swap which of
the two systems' scores is attributed to "system 1" vs "system 2" at that
pair (the label vector never moves). Recompute AP for each side under the
swapped assignment; the null statistic is
`AP_system1_swapped - AP_system2_swapped`. The observed statistic is
`AP_tfidf - AP_other` with no swap. Two-sided p =
`(abs(null_diff) >= abs(observed_diff)).mean()` over 20,000 draws. Uses
children 9 and 10 above.

### (c) Leave-one-positive-out, TF-IDF, $G_{253}$ and $G_{299}$

Both strata have exactly 12 positives (checked: ambiguous rows are never
labeled `"contradiction"`, so $G_{299}$'s positive count equals $G_{253}$'s).
For dropped-positive index $i$ in 0..11, remove that one positive row from
the stratum (no refit -- the same fitted TF-IDF/logistic scores from (a) are
reused, only the evaluated row set shrinks by one), recompute AP over the
remaining rows (252 for $G_{253}$, 298 for $G_{299}$), and recompute the same
one-sided 20,000-draw label-permutation p as in (a), using children 11-22
($G_{253}$) and 23-34 ($G_{299}$) respectively. Report, per stratum: the
range (min, max) of AP across the 12 drops, the range of p across the 12
drops, and the count of the 12 drops with p < 0.05. "Survives
leave-one-positive-out in at least 10 of 12 drops" (decision rule below)
means this count, on $G_{253}$.

### (d) Training-set disjointness

Using the same `splits = build_splits()` call already made above (no
rebuild, no refit):

- **Pair-level overlap**: number of rows in `splits.train` whose
  `(HADM_ID, drug, rule, nursing_sentence, discharge_sentence)` tuple exactly
  matches a row of `splits.gold` on the same key -- the identical key
  `accord/control/data.py`'s own docstring says the training pool was
  already row-decontaminated against.
- **Subject-level overlap**: number of distinct `SUBJECT_ID`s in
  `splits.train` that also appear in `splits.gold`'s `SUBJECT_ID` column.

Both are reported as counts only. No pair_id, HADM_ID, or SUBJECT_ID is ever
printed or written to any artifact; only the two integer counts.

## Decision rule, fixed now

- **Outcome A** (the bag-of-words finding stands as a weak-label signal the
  detector lacks): TF-IDF's AP permutation p < 0.05 on $G_{253}$ **and**
  survives leave-one-positive-out in at least 10 of 12 drops (on $G_{253}$),
  **and** the detector ensemble's AP permutation p >= 0.05 (on $G_{253}$,
  the only stratum a per-pair decontaminated detector score covers), **and**
  the TF-IDF training set has 0 pair and 0 subject overlap with gold.
- **Outcome B** (the finding does not stand as stated): any of those fails.
  Report which.
- **Registered interpretation, fixed now**: if NLI's AP permutation p < 0.05
  on $G_{253}$ (the stratum shared with every other leg of this rule) as
  well, the signal cannot be attributed to the weak labels specifically,
  whatever the outcome letter, and must be described as a surface lexical
  signal that label-free models also find. $G_{299}$'s NLI result is reported
  alongside for completeness but is not what this interpretation reads; if it
  disagrees with $G_{253}$'s, both numbers are stated and the disagreement is
  noted, not resolved by picking the more convenient one.
- **Outcome C** (invalid): the TF-IDF reproduction check fails; the detector
  per-seed/ensemble recomputation fails to match
  `results/accord/decontam/corrected_gold_metrics.json` to 4 decimals; the
  three $G_{253}$ score sources do not share exactly 253 common `pair_id`s;
  or a score source cannot otherwise be identified unambiguously.

## Kill condition

No change of system, metric, draws, seed, stratum, or the spawn order above,
after any number is seen. If a run is interrupted, it restarts from scratch,
never spliced.

## Scope note on multiplicity

This adds AP permutation tests (and the paired and leave-one-positive-out
variants of them) for three more systems to the family
`accord/stats/test_register.py` enumerates, on top of the four C107 already
added and never registered there. That register is out of edit scope for
this task; it is not edited here either, and the family's stated count in
Chapter 4 remains short by however many of these numbers the thesis goes on
to quote, exactly the gap C107 already flagged and left open.

## Outcome

Recorded 2026-09-14 after the run; the rule above is unchanged.

**All gates passed; no Outcome C.** TF-IDF reproduction matched
`tfidfperm.task_a.g299.ap` (0.1570977148911732) and `tfidfperm.task_a.g253.ap`
(0.1650667451480374) to full float precision. Detector per-seed AUROC
(0.5494 / 0.5090 / 0.5048) and ensemble AUROC (0.5145) matched
`results/accord/decontam/corrected_gold_metrics.json` to full float
precision. All three $G_{253}$ score sources shared exactly 253 common
`pair_id`s; label agreement was exact (0 mismatches) between `pharm_label`,
NLI's `label_clean`, and the decontam file's `y_true`.

**(a) One-sided AP permutation, 20,000 draws, seed 42.**

| system | stratum | n | pos | AP | AUROC (descriptive) | p (AP) |
|---|---|---|---|---|---|---|
| detector seed0 | $G_{253}$ | 253 | 12 | 0.0613 | 0.5494 | 0.4336 |
| detector seed1 | $G_{253}$ | 253 | 12 | 0.0592 | 0.5090 | 0.4672 |
| detector seed2 | $G_{253}$ | 253 | 12 | 0.0533 | 0.5048 | 0.5979 |
| detector ensemble | $G_{253}$ | 253 | 12 | 0.0598 | 0.5145 | 0.4629 |
| detector (contaminated `accord_contra_prob`, descriptive only) | $G_{299}$ | 299 | 12 | 0.0394 | 0.4602 | 0.7731 |
| NLI (`contra_prob_mean`) | $G_{253}$ | 253 | 12 | 0.1685 | 0.5861 | **0.0134** |
| NLI (`contra_prob_mean`) | $G_{299}$ | 299 | 12 | 0.1537 | 0.5758 | **0.0120** |
| TF-IDF | $G_{253}$ | 253 | 12 | 0.1651 | 0.5730 | **0.0132** |
| TF-IDF | $G_{299}$ | 299 | 12 | 0.1571 | 0.5761 | **0.0101** |

The detector (every individual seed and the ensemble) does not clear the
permutation null on $G_{253}$; TF-IDF and NLI both do, on both strata. The
detector's $G_{299}$ figure is the contaminated single-run configuration,
descriptive only per the pre-registered score-source note, and is not part
of the decision rule; it also does not clear the null.

**(b) Paired comparison of AP, $G_{253}$, two-sided, 20,000 draws, seed 42.**
TF-IDF minus detector ensemble: AP 0.1651 vs 0.0598, observed difference
+0.1053, p = 0.0001. TF-IDF beats the detector ensemble by a margin the
paired-swap null essentially never reproduces. TF-IDF minus NLI: AP 0.1651
vs 0.1685, observed difference -0.0035, p = 0.9912. TF-IDF and the
off-the-shelf NLI model are statistically indistinguishable on this basis,
consistent with objection 3.

**(c) Leave-one-positive-out, TF-IDF, 20,000-draw permutation p per drop.**
$G_{253}$: AP ranges 0.0673 to 0.1758 across the 12 drops, p ranges 0.0107
to 0.2870, **11 of 12 drops keep p < 0.05** (the twelfth drop's p is 0.2870,
so one single positive pair is responsible for pushing the test over the
0.05 line when it alone is removed; the finding is not carried by that one
pair, since removing any of the other 11 leaves it significant, but it is
not perfectly uniform either). $G_{299}$: AP ranges 0.0600 to 0.1677, p
ranges 0.0073 to 0.2563, also 11 of 12. The pre-registered threshold (>= 10
of 12) is cleared on $G_{253}$.

**(d) Training-set disjointness.** 1,183 training rows, 299 gold rows: 0
rows share the `(HADM_ID, drug, rule, nursing_sentence, discharge_sentence)`
key with any gold pair, and 0 `SUBJECT_ID`s are shared between the training
pool and the gold set. Independently re-derived here, not merely trusted
from `accord/control/data.py`'s own assertions.

**Decision: Outcome A.** TF-IDF's $G_{253}$ AP permutation p is 0.0132 (<
0.05); it survives leave-one-positive-out in 11 of 12 drops (>= 10); the
detector ensemble's $G_{253}$ AP permutation p is 0.4629 (>= 0.05); the
training set has 0 pair and 0 subject overlap with gold. All four legs of
the Outcome-A condition hold, so, narrowly read, the bag-of-words finding
stands as a signal the trained detector's own scores do not carry.

**Registered interpretation triggers, and changes what Outcome A means.**
NLI's AP permutation p is significant on both $G_{253}$ (0.0134) and
$G_{299}$ (0.0120) -- consistent in direction and magnitude with TF-IDF's
own p-values, and paired testing (task b) cannot distinguish TF-IDF from NLI
at all (p = 0.9912). Per the interpretation fixed before this run: **the
signal cannot be attributed to the weak labels specifically.** An
off-the-shelf NLI model that never saw a weak label finds the same
ranking, at the same significance, to within sampling noise of the same
permutation test. What C107 demonstrated and this pre-registration confirms
under a matched test is real and survives its fragility check: something in
these 253-299 pairs separates the 12 confirmed contradictions from the rest
well enough for a permutation test to notice, using either bag-of-words
features or a pretrained NLI model's sentence embeddings -- and the trained
detector's own weak-label-supervised scores do not carry it (p = 0.4629,
and the detector loses the paired comparison to TF-IDF outright, p =
0.0001). But because a label-free model finds essentially the same thing,
the correct description is a surface lexical/semantic-similarity signal
present in the raw sentence pairs, not a signal specifically attributable to
the weak labels TF-IDF and the detector were both meant to be trained
against. C107's phrase "a weak-label signal the detector lacks" is not
supported by this run; "a signal in the raw text neither the weak-label
training pool nor the trained detector was able to convert into a working
score, and which is not specific to being trained on the weak labels" is.

**What could not be done.** The detector's $G_{299}$ permutation test could
not be run against the decontaminated 3-seed ensemble, because no per-pair
score from the decontaminated retrain was retained for the 46 ambiguous
pairs (`msu_ch4_results.tex`, paragraph around line 220-221). The
descriptive-only substitute reported above uses the contaminated single-run
`accord_contra_prob` column and is explicitly not part of the decision
rule.

Artifact: `results/accord/c111_robustness/c111_robustness.json`. Register
prefix `c111.*` (register script written, not run, per task scope; see
`accord/c111_robustness/register_c111.py`). This run adds AP permutation
tests, a paired comparison, and a leave-one-positive-out check for two more
systems (detector, NLI) to the hypothesis family
`accord/stats/test_register.py` enumerates; that register is out of edit
scope for this task and was not touched, so the family's stated count in
Chapter 4 remains short by however many of these the thesis goes on to
quote -- the same gap C107 already flagged, now larger.
