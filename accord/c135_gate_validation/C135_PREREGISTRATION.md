# C135 pre-registration: does the marginal feasibility gate, computed on a 300 item pilot, predict that a student trained on that labeller's labels is useless?

Committed before the first datum. The outcome is appended below after the run; the rule above the outcome line is
never edited. Convention set by C44.

## Why

The thesis proposes the MARGINAL FEASIBILITY GATE as its central method, but the gate has only ever been applied
post hoc to one labeller on one clinical pilot. It has never been shown to predict anything. A gate is a screening
instrument, and a screening instrument that has not been scored against an outcome on data where the outcome is
observable is not a validated contribution. This check scores it.

No hosted API. No MIMIC data. Nothing under `accord/thesis/` is touched. No number below has been computed at the
time of writing.

## The gate, as tested

A labeller flags `n` of `N` pilot items. The pilot contains `K` true positives, known from the pilot's gold labels.

- **Ceiling.** The maximum Matthews correlation attainable with the marginals `(N, K, n)` fixed, obtained by placing
  every true positive that can fit inside the flagged set: `TP = min(n, K)`, `FP = n - TP`, `FN = K - TP`,
  `TN = N - n - FN`, then the usual MCC, defined as 0 when any factor of its denominator is 0.
- **Floor.** The smallest one sided hypergeometric p value those marginals permit,
  `p_floor = P(X >= min(n, K))` for `X ~ Hypergeometric(population N, successes K, draws n)`, computed as
  `scipy.stats.hypergeom.sf(min(n, K) - 1, N, K, n)`.
- **Verdict.** The gate FAILS the labeller when `p_floor >= 0.05`: no outcome that marginal permits could have
  rejected the null. Otherwise it PASSES.

The gate verdict is computed from the 300 item pilot ONLY. It never sees the training pool, the test set, or the
student.

## Data

SNLI (`stanfordnlp/snli` from the Hugging Face hub), recast to binary: `y = 1` iff the gold label is contradiction,
`y = 0` iff it is entailment or neutral, rows with no gold label dropped. Item text is
`premise + " [SEP] " + hypothesis`. If SNLI cannot be obtained the run falls back to a fully synthetic Gaussian
two class generative setup and the report says so; the arm structure and every rule below are unchanged. Which data
was used is stated in the artifact.

Three disjoint frames, all at 4 percent prevalence to match the clinical setting, all drawn with
`numpy.random.default_rng(42)`:

- **Pilot:** 300 items, 12 positives, 288 negatives, from the SNLI validation split.
- **Training pool:** 5000 items, 200 positives, 4800 negatives, from the SNLI train split.
- **Test set:** 10000 items, 400 positives, 9600 negatives, from the SNLI train split, disjoint by row index from
  the pool.

Seed 42 everywhere, recorded in the artifact.

## Labellers

### (i) Synthetic, the primary arm
A synthetic labeller is a pair `(TPR, FPR)`. It flags each true positive independently with probability TPR and
each true negative independently with probability FPR, drawn once for the pilot and once for the pool from the seed
42 stream. The grid, fixed now, 24 labellers:

- `TPR = 0.90`, `FPR` in {0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.87, 0.93, 0.97}. Twelve
  labellers spanning expected flag rates from about 0.046 to about 0.967, the top of the range covering the clinical
  regime in which 261 of 299 items were flagged, a flag rate of about 0.87.
- `TPR = 0.50`, `FPR` in {0.01, 0.05, 0.20, 0.50, 0.87}.
- `TPR = 0.10`, `FPR` in {0.01, 0.05, 0.20, 0.50}.
- `TPR = 0.04`, `FPR` in {0.03, 0.30, 0.90}. These are near chance labellers, included deliberately because a
  near chance labeller with a low flag rate is the case in which the gate is expected to pass a useless labeller.

### (ii) Real, secondary, reported separately
The local Ollama models under one plain contradiction prompt, temperature 0, seed 42, thinking disabled,
`num_predict` 8, asked for YES or NO. A flag is recorded iff the reply, stripped and upper cased, begins with YES;
anything else is a non flag and the count of replies matching neither YES nor NO is reported. Models attempted, in
this order: `qwen3:0.6b`, `qwen3:1.7b`, `qwen3:4b`, `llama3:8b`, `granite3.3:8b`, `qwen3:8b`, `qwen3:14b`. Each
model is run over the same 300 pilot items and the same 5000 pool items. Models that do not finish within the run
budget are reported as not run; no model is dropped after its numbers are seen.

## Student

One student per labeller, the same recipe for every cell: `TfidfVectorizer(lowercase=True, sublinear_tf=True,
min_df=2, ngram_range=(1,2))` fitted on the 5000 pool texts (labels do not enter the fit, so one vectorizer serves
every labeller) plus `LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=2000)` trained
on the pool texts with that labeller's pool labels. Outcome: the student's AUROC against TRUE labels on the 10000
item test set. A labeller whose pool labels are all one class yields no trainable student; its AUROC is recorded as
0.5 and it counts as a failed student. A small transformer student is run on a subset only if time allows and is
reported separately; it is not part of the reading rule.

## Reading rule (fixed now)

**Student failure** is test AUROC below 0.60 (primary) and, as a sensitivity analysis, below 0.70. The gate is
scored as a screening test in which "gate FAILS the labeller" is the positive test result and "student failed" is
the condition. Reported for the primary arm, then for the real arm, then pooled: the 2x2 cell counts, sensitivity,
specificity, positive predictive value, negative predictive value.

The primary reading is the synthetic arm at the 0.60 threshold.

- **Outcome A, validated.** Sensitivity >= 0.90 AND PPV >= 0.90, AND the gate's verdict is NOT reproducible by a
  single threshold on the raw flag rate ratio defined below.
- **Outcome B, partially validated.** Sensitivity >= 0.90 AND PPV >= 0.90, but either the verdict is exactly
  reproducible by a threshold on the raw ratio, or NPV is below 0.90, or both. A screen that works in one direction
  only, or that adds nothing over a simpler statistic, is recorded as partially validated and the specific
  limitation is named.
- **Outcome C, not validated.** Sensitivity < 0.90 or PPV < 0.90.
- **Outcome D, invalid, no reading recorded.** SNLI and the Gaussian fallback both fail; a frame cannot be built at
  the specified size and prevalence; the pilot does not contain exactly 12 positives; the frames are not disjoint.

## The honest limit, reported whatever the outcome

The thesis states that a passing gate certifies nothing. That is tested directly: among gate PASSING labellers, the
fraction whose student failed, which is `1 - NPV`, is reported with its count.

## The simple alternative predictor

`ratio = (n / N) / (K / N) = n / K`, the labeller's pilot flag rate divided by the pilot base rate. Reported:

1. The gate's FAIL region as a set of flag counts, derived in the script by evaluating `p_floor` at every
   `n` in `0..N`, so that the claim "the gate is a threshold rule on the flag count" is checked rather than asserted.
2. The screening statistics of the ratio predictor at the threshold that reproduces the gate's FAIL region, and
   whether the two verdict vectors are identical labeller by labeller.
3. The screening statistics of the ratio predictor swept over every threshold, with the best Youden J, so the
   question "could a better threshold on the raw ratio beat the gate" is answered rather than assumed.

If the gate's verdicts are identical to a single threshold on the raw ratio, the report says plainly that the gate
adds nothing over the raw ratio. That is an important negative finding and is reported as one.

Also reported, deciding nothing: Spearman correlation of ceiling MCC with student AUROC, and of `p_floor` with
student AUROC.

## Kill condition

No threshold (0.05 on `p_floor`, 0.60 and 0.70 on AUROC, 0.90 on sensitivity, PPV), no seed, no frame size, no
prevalence, no labeller grid entry, no student hyperparameter and no outcome definition is changed after any number
from this run is seen. If the outcome is C it is reported as C. No second student recipe, second seed, second
prevalence, second prompt or second grid is run in search of a different outcome. A different question would need
its own pre-registration and would not replace this outcome.

## Compliance

Public data only, no credentialed input, no hosted API, no en dashes or em dashes. Every number in the report comes
from `c135_run.py` into `results/accord/c135_gate_validation/c135.json`. Nothing under `accord/thesis/` is read or
written.

## Outcome (appended after the run; the text above is unchanged)

Run on 2026-09-26, seed 42, SNLI obtained successfully (no fallback needed). Frames as specified: pilot 300 with
exactly 12 positives, pool 5000 with 200, test 10000 with 400, pool and test disjoint. Artifact
`results/accord/c135_gate_validation/c135.json`. The real Ollama arm was not run under this pre-registration.

**Outcome C, not validated, AND the run is uninformative about the gate.** Both readings are recorded because the
rule above is not edited.

The pre-registered reading at the 0.60 threshold, synthetic arm, 24 labellers: cells 5 / 0 / 19 / 0 (gate fail and
student failed / gate fail and student useful / gate pass and student failed / gate pass and student useful),
sensitivity 0.208, PPV 1.000, specificity and NPV undefined and 0.000 respectively. Sensitivity 0.208 is below
0.90, so the pre-registered reading is Outcome C.

That reading cannot be trusted, and the run is recorded as uninformative, because the outcome variable has no
variance: **every one of the 24 students failed, and so did the clean-label reference student.** The diagnostic
reference added to the script after the grid was run, and computed only from true pool labels with no gate quantity
involved, is `clean_label_oracle_student_auroc = 0.6037`. A labeller with perfect labels therefore lands 0.004 above
the 0.60 failure line and below the 0.70 line. The pre-registered student, TF-IDF plus logistic regression on 5000
SNLI items at 4 percent prevalence, that is 200 positives, is too weak for the task, so no labeller in the grid
could have produced a useful student and no screening instrument of any kind could have been scored on this design.
Specificity is undefined for exactly this reason.

The design defect is in the student, not in the gate and not in the labeller grid. It is recorded here and carried
to a fresh pre-registration, C136, which keeps the seed, the frames, the labeller grid, the gate, the thresholds and
the reading rule unchanged and replaces the student's feature extractor with one that has headroom. The C136 student
choice was made from the clean-label reference alone; no gate verdict, screening statistic or 2x2 cell from this run
entered it.

One fact from this run is design independent, is arithmetic about the gate rather than about the data, and carries
forward: with N = 300 and K = 12, the gate's FAIL region is exactly the flag counts {0} together with
{235, 236, ..., 300}. It is the union of a single point at zero and one upper tail, so the gate is a threshold rule
on the flag count and nothing more. The gate fails a labeller only if that labeller flags at least 235 of 300 items,
a pilot flag rate of 0.783 or higher.
