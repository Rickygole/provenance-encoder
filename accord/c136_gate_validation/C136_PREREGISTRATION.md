# C136 pre-registration: the marginal feasibility gate as a screen for student failure, on a design with student headroom

Committed before the first C136 datum. The outcome is appended below; the rule above the outcome line is never
edited. Convention set by C44.

## Superseded first attempt

(Commit hashes updated after the duplicate-identifier resolution rewrote history; the originals are unreachable from HEAD.), this check's own history

This check was attempted once before and abandoned. That attempt is **not a separately numbered check**: it is this
check's first attempt, and it holds no identifier of its own. Its rule text and its script are preserved unedited
under `accord/c136_gate_validation/superseded_first_attempt/`, as `PREREGISTRATION_SUPERSEDED.md` and
`run_superseded.py`, with its artifacts under
`results/accord/c136_gate_validation/superseded_first_attempt/`. The convention is that a rule is never edited, so
the abandoned rule is kept as superseded history rather than corrected or deleted.

It was abandoned because its pre-registered student, TF-IDF plus logistic regression on 5000 SNLI items at 4 percent
prevalence, had no headroom: its clean-label reference student reached AUROC 0.6037, four thousandths above its own
0.60 failure line, so every labeller in the grid produced a failing student, the outcome variable had no variance,
and specificity was undefined. No screening instrument could have been scored on that design.

Commits that hold it: `cdec8d0` committed its rule before its first datum, and `8adec39` appended its outcome and
recorded it as uninformative. Both commits describe it as "C135". That identifier was a collision: the
trigger-exemplar dissociation-cost check, `accord/c135_dissociation_cost/`, is the live holder of C135, it has a
completed run and 188 keys under the `c135` register prefix, and this abandoned attempt wrote nothing to the
register. The directories asserting the duplicate identity were therefore renamed into the superseded history above.
The commit messages are left as they stand, since history is not rewritten.

## What changed from C135 and why, stated before any C136 number exists

C135 fixed everything below and used a TF-IDF plus logistic regression student. Its clean-label reference student
reached AUROC 0.6037, four thousandths above the 0.60 failure line, so every labeller in the grid produced a failing
student, the outcome variable had no variance, and specificity was undefined. **The only change here is the
student's feature extractor.** It was chosen from that clean-label reference alone. No gate verdict, screening
statistic or 2x2 cell from C135 entered the choice, and C135's gate reading is not carried forward.

Unchanged from C135: seed, data source, all three frames and their sizes and prevalences, the gate definition, the
labeller grid, the real arm, the failure thresholds, the screening definitions, the alternative predictor, and the
reading rule.

## The gate, as tested

A labeller flags `n` of `N` pilot items; the pilot contains `K` true positives from gold.

- **Ceiling.** Maximum Matthews correlation attainable with `(N, K, n)` fixed, every true positive that fits placed
  inside the flagged set: `TP = min(n, K)`, `FP = n - TP`, `FN = K - TP`, `TN = N - n - FN`, then MCC, defined as 0
  when any factor of its denominator is 0.
- **Floor.** `p_floor = P(X >= min(n, K))` for `X ~ Hypergeometric(N, K, n)`, as
  `scipy.stats.hypergeom.sf(min(n, K) - 1, N, K, n)`.
- **Verdict.** FAIL when `p_floor >= 0.05`. The verdict is computed from the 300 item pilot ONLY and never sees the
  pool, the test set or the student.

## Data

SNLI (`stanfordnlp/snli`), binary: `y = 1` iff contradiction, `y = 0` iff entailment or neutral, unlabelled rows
dropped. Seed 42, `numpy.random.default_rng(42)`. Three disjoint frames at 4 percent prevalence:
pilot 300 with 12 positives from the validation split; pool 5000 with 200 positives and test 10000 with 400
positives, both from the train split and disjoint by row index. Identical to C135, same draw order, same rows.

## Labellers

### (i) Synthetic, the primary arm, 24 labellers, grid unchanged from C135
A labeller is a pair `(TPR, FPR)`; it flags each true positive with probability TPR and each true negative with
probability FPR, drawn once for the pilot and once for the pool from the seed 42 stream.

- `TPR = 0.90`, `FPR` in {0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.87, 0.93, 0.97}.
- `TPR = 0.50`, `FPR` in {0.01, 0.05, 0.20, 0.50, 0.87}.
- `TPR = 0.10`, `FPR` in {0.01, 0.05, 0.20, 0.50}.
- `TPR = 0.04`, `FPR` in {0.03, 0.30, 0.90}, near chance labellers, included because a near chance labeller with a
  low flag rate is the case in which the gate is expected to pass a useless labeller.

The top of the FPR range covers the clinical regime in which 261 of 299 items were flagged, a flag rate of 0.87.

### (ii) Real, secondary, reported separately
Local Ollama models, one plain contradiction prompt, temperature 0, seed 42, thinking disabled, `num_predict` 8,
YES or NO requested. A flag iff the stripped upper cased reply begins with YES; replies matching neither YES nor NO
are counted and reported. Attempted in this order: `qwen3:0.6b`, `qwen3:1.7b`, `qwen3:4b`, `llama3:8b`,
`granite3.3:8b`, `qwen3:8b`, `qwen3:14b`, each over the same 300 pilot and same 5000 pool items. A model that does
not finish in the run budget is reported as not run; no model is dropped after its numbers are seen.

## Students

Two students per labeller. Both are trained on the 5000 pool texts with that labeller's pool labels and scored by
AUROC against TRUE labels on the 10000 item test set. Features never see any label, so one feature matrix serves
every labeller.

- **Primary, `minilm`.** `sentence-transformers/all-MiniLM-L6-v2` run locally, mean pooled over the attention mask,
  L2 normalised, max length 96, premise and hypothesis encoded separately to `u` and `v`, item features
  `[u, v, |u - v|, u * v]`, then `LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced")`. Chosen because
  its clean-label reference AUROC on these exact frames is 0.823, which leaves headroom above both failure lines.
- **Secondary, `tfidf`.** The C135 student verbatim: `TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=2,
  ngram_range=(1,2))` on `premise + " [SEP] " + hypothesis`, then `LogisticRegression(C=1.0, solver="liblinear",
  class_weight="balanced", max_iter=2000)`. Reported in full for every cell. Its clean-label reference is 0.6037, so
  it is expected to be degenerate; it is reported, not read.

A labeller whose pool labels are all one class yields no trainable student; AUROC 0.5, counted as failed. The
clean-label reference AUROC of each student is recorded in the artifact and is not a labeller.

## Reading rule (fixed now)

**Student failure** is test AUROC below 0.60, primary, and below 0.70, sensitivity analysis. The gate is scored as a
screening test: "gate FAILS the labeller" is the positive test result, "student failed" is the condition. Reported
for the synthetic arm, the real arm and pooled, for both students: 2x2 cell counts, sensitivity, specificity,
positive predictive value, negative predictive value.

**The primary reading is the synthetic arm, the `minilm` student, the 0.60 threshold.**

- **Outcome A, validated.** Sensitivity >= 0.90 AND PPV >= 0.90, AND the gate's verdict is not reproducible by a
  single threshold on the raw flag rate ratio below.
- **Outcome B, partially validated.** Sensitivity >= 0.90 AND PPV >= 0.90, but either the verdict is exactly
  reproducible by a threshold on the raw ratio, or NPV is below 0.90, or both. The specific limitation is named.
- **Outcome C, not validated.** Sensitivity < 0.90 or PPV < 0.90.
- **Outcome D, invalid, no reading recorded.** SNLI unobtainable; a frame not buildable at the specified size and
  prevalence; the pilot not holding exactly 12 positives; frames not disjoint; **or the primary student's
  clean-label reference AUROC below 0.75, which would mean this design again lacks headroom.** That last condition
  is added because it is the defect that voided C135, and it is checked before the grid is read.

## The honest limit, reported whatever the outcome

The thesis states that a passing gate certifies nothing. Tested directly: among gate PASSING labellers, the fraction
whose student failed, `1 - NPV`, with its count.

## The simple alternative predictor

`ratio = (n / N) / (K / N) = n / K`, pilot flag rate over pilot base rate. Reported:

1. The gate's FAIL region as an explicit set of flag counts, obtained by evaluating `p_floor` at every `n` in
   `0..N`, so that "the gate is a threshold rule on the flag count" is checked, not asserted.
2. The screening statistics of the ratio predictor at the threshold reproducing that FAIL region, and whether the
   two verdict vectors are identical labeller by labeller.
3. The screening statistics of the ratio predictor swept over every threshold, with the best Youden J, so that
   "could a better threshold on the raw ratio beat the gate" is answered rather than assumed.

If the gate's verdicts are identical to a single threshold on the raw ratio, the report says plainly that the gate
adds nothing over the raw ratio, as a negative finding.

Also reported, deciding nothing: Spearman correlation of ceiling MCC with student AUROC, and of `p_floor` with
student AUROC.

## Kill condition

No threshold (0.05 on `p_floor`, 0.60 and 0.70 on AUROC, 0.90 on sensitivity and PPV, 0.75 on the reference), no
seed, no frame size, no prevalence, no grid entry, no student hyperparameter, no prompt and no outcome definition is
changed after any C136 number is seen. If the outcome is C it is reported as C. No third student, second seed,
second prevalence, second prompt or second grid is run in search of a different outcome. This is the second
pre-registration of this question; there will not be a third that keeps the question and moves the design. Any
further change of design is reported as a failure to design the check, not as a result.

## Compliance

Public data only, no credentialed input, no hosted API, no en dashes or em dashes, nothing under `accord/thesis/`
read or written. Every number in the report comes from `c136_run.py` into
`results/accord/c136_gate_validation/c136.json`.

## Outcome (appended after the run; the text above is unchanged)

Run on 2026-09-26 and 2026-09-27, seed 42, data SNLI (stanfordnlp/snli), binary contradiction vs not. Frames: pilot 300 with 12 positives, pool 5000 with 200, test 10000 with 400, pool and test disjoint: True. Artifact `results/accord/c136_gate_validation/c136.json`. Every number below is read from that artifact by `c136_outcome.py`; none is typed.

**Headroom precondition, checked before the grid was read: PASSED.** The primary `minilm` student's clean-label reference AUROC is 0.8167, above the pre-registered floor of 0.75. The secondary `tfidf` student's reference is 0.6037, which is why C135 was uninformative; it is reported below and not read.

## Outcome C, NOT VALIDATED

Primary reading, synthetic arm, `minilm` student, failure at AUROC below 0.60, 24 labellers. 2x2 as gate fail and student failed / gate fail and student useful / gate pass and student failed / gate pass and student useful: **5 / 0 / 12 / 7**. Sensitivity 0.2941, specificity 1.0000, ppv 1.0000, npv 0.3684. Youden J 0.2941. Sensitivity 0.2941 is far below the pre-registered 0.90, so the outcome is C.

Sensitivity analysis, failure at AUROC below 0.70, same arm and student: 2x2 **5 / 0 / 16 / 3**, sensitivity 0.2381, specificity 1.0000, PPV 1.0000, NPV 0.1579, Youden J 0.2381. The conclusion does not move.

Secondary `tfidf` student, failure below 0.60: 2x2 **5 / 0 / 19 / 0**, sensitivity 0.2083, specificity undefined, PPV 1.0000, NPV 0.0000. Every student failed including the clean-label reference, so specificity is undefined; this reproduces C135 and is reported only to show the defect is the student, not the gate.

### The honest limit: a passing gate certifies nothing, quantified

Among the 19 gate PASSING labellers in the primary reading, 12 produced a failing student, a fraction of **0.6316**, which is 1 minus the NPV of 0.3684. At the 0.70 threshold it is 0.8421 (16 of 19). The thesis's own caveat is confirmed numerically: most labellers the gate passes still yield a useless student.

The other direction is the one place the gate holds up. In the primary reading no gate FAILING labeller produced a useful student (0 such cells), so specificity is 1.0000 and PPV is 1.0000. The gate never wrongly condemns a labeller. It just almost never condemns one at all.

### Why: the gate is arithmetically a flag count threshold

With N = 300 and K = 12, evaluating `p_floor` at every flag count from 0 to 300 gives a FAIL region that is exactly the single point 0 together with the upper tail 235 to 300 (`fail_region_is_zero_plus_upper_tail` = True). The gate therefore fails a labeller only if it flags at least 235 of 300 pilot items, a flag rate of 0.7833 or more, or flags nothing at all. It carries no information beyond the flag count.

### The gate against the raw ratio, flag rate over base rate

**The gate adds nothing over the raw ratio.** Predicting failure when the ratio is at or above 19.5833 reproduces the gate's verdict on every labeller (`identical_to_gate_verdicts` = True, 0 disagreements), giving the same 2x2 5 / 0 / 12 / 7. No synthetic labeller flagged zero items (their flag counts run from 5 to 291), so in the primary arm the gate's zero point never separates it from that single threshold. The real arm does contain zero flag labellers (2 of 7 models flagged none of the 300 pilot items), so there the gate is a two sided rule on the flag count, low or high, rather than a one sided one. Either way the hypergeometric floor buys no information beyond the flag count itself.

**And a better threshold on the same raw ratio strictly dominates the gate.** Sweeping every threshold, the best Youden J is 0.6807 at ratio 3.5833, that is a flag count of about 43 of 300, with sensitivity 0.8235 and specificity 0.8571, against the gate's J of 0.2941 (`ratio_beats_gate` = True). That sweep chooses its threshold on the same 24 labellers it is scored on, so it is an in sample upper bound and flatters the ratio; the gate's operating point is fixed a priori and does not get that advantage. The comparison that does not depend on the sweep is the previous paragraph: the gate IS a ratio threshold, and it is a badly placed one.

Rank correlations, deciding nothing: ceiling MCC against student AUROC, Spearman rho 0.5310 (p 0.00759); `p_floor` against student AUROC, rho -0.5197 (p 0.00925). Both are moderate and in the expected direction, which is the strongest thing that can be said for the gate's quantities: they are correlated with student quality, they are not a decision rule for it.

### Real labellers, secondary

7 local Ollama models ran under the plain contradiction prompt:

| model | pilot flags of 300 | true positives flagged of 12 | p_floor | gate | student AUROC minilm | student AUROC tfidf |
|---|---|---|---|---|---|---|
| qwen3:0.6b | 272 | 6 | 0.301 | FAIL | 0.2409 | 0.4110 |
| qwen3:1.7b | 176 | 0 | 0.00142 | PASS | 0.2028 | 0.4598 |
| qwen3:4b | 0 | 0 | 1 | FAIL | 0.5000 | 0.5000 |
| llama3:8b | 17 | 2 | 6.97e-18 | PASS | 0.4228 | 0.4668 |
| granite3.3:8b | 18 | 10 | 2.09e-17 | PASS | 0.7808 | 0.5852 |
| qwen3:8b | 0 | 0 | 1 | FAIL | 0.4120 | 0.4915 |
| qwen3:14b | 8 | 5 | 3.34e-13 | PASS | 0.7580 | 0.5949 |

Real arm screening at 0.60: 2x2 3 / 0 / 2 / 2, sensitivity 0.6000, specificity 1.0000, PPV 1.0000, NPV 0.5000.

Pooled synthetic and real at 0.60: 2x2 8 / 0 / 14 / 9, sensitivity 0.3636, specificity 1.0000, PPV 1.0000, NPV 0.3913, fraction of gate passers still failing 0.6087.

### What this means for the thesis

The marginal feasibility gate is **not validated as a screen for student failure**. It is validated as one thing only, and a narrow one: a sufficient condition for hopelessness that fires almost never. Nothing it condemned was salvageable, and it condemned 5 of 24 labellers while 17 of 24 students actually failed. The thesis may claim that the gate, when it fires, is a correct veto, and that it formalises why the observed clinical labelling was hopeless before any student was trained. It may not claim the gate screens labellers, predicts student utility, or is a feasibility criterion: it passes most labellers whose students are useless, and its verdict is a restatement of the pilot flag count with no added information. Any claim beyond the veto reading is contradicted by this run.
