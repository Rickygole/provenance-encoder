# Pre-registration: selection-conditioned audit blindness, real data

Written before any audit gap, TPR/FPR gap, four-fifths ratio, or power number
has been looked at. The only numbers examined before writing this were (a)
the pinned task-split hash/subject counts (a lineage check, not a result),
(b) the ADMISSIONS join rate and group prevalence (a feasibility check), and
(c) the classifier's AUROC/PPV-lift and the selection-rule beta needed to hit
a target selection four-fifths ratio (a power/design calibration, not an
audit outcome). No demographic-parity gap, TPR/FPR gap, PPV gap, calibration
gap, or pool-level four-fifths ratio has been computed yet.

## Data / lineage

- `/Users/rickygole/mimic_local_only/task/{train,val,test}.jsonl`.
- Verified via `src/data/lineage.py:task_dir_lineage` against
  `results/task_lineage_pin.json`: **task_split_hash=9a9fe073f4ea9544 MATCHES**
  the pinned reference (subject counts train=19977, val=2854, test=5708 — all
  match exactly). Split integrity confirmed before touching text.
- **Correction to the task brief's recalled numbers**: the brief states
  "34,018 examples, 3,730 positives, 734 positives in test." The data on disk
  under the pinned hash has 34,018 examples total (train 23,846 + val 3,396 +
  test 6,776) but **2,975 positives** (train 2,091 + val 314 + test 570), not
  3,730/734. This is reported as measured, not reconciled to the brief.

## Design

- Group-blind classifier: TF-IDF (unigrams, max_features=50000, min_df=5,
  max_df=0.9, sublinear tf, English stopwords) + `LogisticRegression`, fit
  ONCE on `train` only (the unfiltered reference split), never sees group
  labels or ADMISSIONS fields.
- Population for the selection/audit demonstration: `val` ∪ `test`
  (n=10,172, not used to fit the classifier or its threshold — threshold is
  chosen on `val`'s ROC curve at target FPR=0.15, then frozen and applied to
  the full val+test population, matching the operating-point convention
  already used in `accord/denominator/simulate.py`).
- Primary group axis: `LANGUAGE` missing in ADMISSIONS (joined on HADM_ID,
  100% join rate, 41.7% missing rate in the val+test population). Secondary
  axes computed for completeness: ETHNICITY-unknown-ish (UNKNOWN/NOT
  SPECIFIED, UNABLE TO OBTAIN, PATIENT DECLINED TO ANSWER, or null; 11.1%),
  INSURANCE (5-category).
- Selection feature: `doc_feature = 0.7 * z(total_chars) + 0.3 * z(n_segments)`
  computed on the val+test population — documentation volume (primary) and
  note-mention count (secondary), NOT group membership. Measured
  `corr(total_chars, lang_missing) = -0.204` (missing-language admissions are
  genuinely lower-volume in this data — consistent with C25's CareVue/
  MetaVision era finding) and `corr(total_chars, label) = +0.074` (volume is
  real signal, not noise), so a selection rule that reacts to volume alone
  will induce a language-missing selection disparity *through* a real,
  uncontrolled correlation, not through Cov(f(X),A)=0-by-construction as in
  the failed C24 simulation.
- Selection rule: `logit P(S=1) = b0(beta) + beta * doc_feature`, with
  `b0(beta)` solved by bisection so the marginal selection rate stays pinned
  at 30% for every beta (isolates the group-disparity effect of beta from a
  changing overall admission rate, mirroring the y-centering trick in
  `simulate.py`). Swept grid:
  `beta in {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0}`.
  Calibration run (design-only, not an audit outcome) found beta≈5.0 gives a
  **selection-stage four-fifths ratio ≈0.47**, the value named in the brief;
  the grid spans well below and above it (0.95 at beta=0 down to ≈0.40 at
  beta=8).
- Every sweep point reports: demographic parity gap, TPR gap, FPR gap,
  equalized-odds gap (max of TPR/FPR gap), PPV gap, calibration gap,
  four-fifths ratio (pool-level, on the induced pool), AND the
  selection-stage four-fifths ratio (pre-classifier, on the full population,
  selected vs. not) — each with a 95% bootstrap CI (percentile method,
  B=5,000, resampling rows of the realized pool/population with
  replacement, fixed seed). A replicate-averaged trend (30 independent
  selection draws per beta, group rates averaged before differencing, same
  convention as `simulate.py`) is reported alongside as a stability check on
  the point estimate, not as the primary CI.

## Decision rule (fixed now, not adjustable after seeing results)

**Tolerance boundary** (conventional, matches `simulate.py` and this
project's prior corrections): parity/TPR/FPR/PPV/equalized-odds gap
tolerance = **0.05**; four-fifths threshold = **0.80**.

**CI criterion.** A sweep point counts as a genuine breach only if BOTH:
(a) the point estimate crosses the tolerance boundary (gap > 0.05 or
four-fifths < 0.80), AND (b) the 95% bootstrap CI excludes the boundary in
the breaching direction (CI lower bound > 0.05 for a gap breach, or CI upper
bound < 0.80 for a four-fifths breach). A point estimate that crosses the
boundary but whose CI still straddles it is called "inconclusive," not a
breach — this is the guard against the C24-style false read of noise as
signal, and against the opposite error of dismissing a real breach as noise.

**"Flat" is defined as:** across the ENTIRE beta grid (including the
beta≈5.0 point where selection four-fifths ≈0.47), every pool-level audit
gap's point estimate stays ≤0.05 with its 95% CI upper bound ≤0.05, AND the
pool-level four-fifths ratio stays ≥0.80 with its 95% CI lower bound ≥0.80,
AND an OLS regression of each gap on beta has a 95% CI on its slope that
includes 0 (no credible monotonic drift). If any one of these fails at any
beta ≤ the beta that produces selection four-fifths ≈0.47, that is a
REFUTATION-direction reading regardless of what happens at more extreme
beta.

**Power layer.** At every sweep point, report the minimum detectable
demographic-parity gap (two-proportion normal-approximation MDE, alpha=0.05
two-sided, power=0.80) given the REALIZED per-group pool sizes at that beta.
A "flat" reading is only informative where MDE < 0.05 (adequately powered to
see a 0.05 gap if one were there); if MDE ≥0.05 at any point in the primary
range, that point is reported as underpowered, not as evidence of blindness,
per the brief's explicit warning about conflating "no gap" with "no power."

**Positive control (run first, before the classifier non-degeneracy check is
trusted and before the sweep).** Inject a known log-odds shift onto the
frozen classifier's real scores for the lang_missing=1 group only (no
selection filtering: S=1 for the entire val+test population), holding y and
group fixed. Shift magnitude: +1.2 logit-points added to lang_missing=1
scores only (chosen to produce roughly a 15-20 point predicted-positive-rate
gap given the ~30-40% baseline predicted-positive-rate mass near the
operating threshold). PASS condition: the audit's own code reports a
demographic-parity-gap point estimate whose 95% CI excludes 0, and a
four-fifths ratio point estimate <0.90 with CI upper bound <1.0. If this does
not hold, STOP — the audit code is broken and no downstream number is
reported as a finding.

**Classifier non-degeneracy bar (checked before the sweep is trusted).**
AUROC on held-out val ≥0.65 (well above chance for this task), predicted-
positive rate at the operating threshold strictly between 5% and 50% (not
near-constant in either direction — this is the exact failure mode of C28's
87.3%), and precision lift over the empirical base rate ≥1.5x. If violated,
STOP and report the classifier as degenerate; no sweep is run.

## Verdict mapping

- **SUPPORTS the invariance claim** if: positive control passes, classifier
  is non-degenerate, and the sweep is "flat" (as defined above) at every
  beta up to and including the one nearest selection four-fifths≈0.47, in
  regions where MDE < 0.05.
- **REFUTES it** if any pool-level audit gap breaches (point estimate over
  boundary AND CI excludes boundary) at or before that beta, in a region
  where MDE < 0.05 (so the breach is not itself a power artifact). If that
  happens, the claim narrows to the theoretical/pure-group-dependent regime
  (`simulate.py`'s Sweep 1) and does not extend to selection mechanisms that
  operate through real, outcome-correlated text features — the exact
  mechanism this experiment was built to test because the prior simulation
  couldn't.
- **Inconclusive** if MDE ≥0.05 across the region up to beta≈5.0 (would mean
  the real data, even combining val+test, is underpowered for this
  question) — reported honestly rather than forced into either verdict.
