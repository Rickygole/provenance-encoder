# C138: prevalence-matched thresholding of the weak labeller

Identifier. C135 and C136 were taken; C137 is claimed by `accord/audit_c137/`
and `results/accord/audit_c137/`. This correction is therefore **C138**, the
next free identifier in the shared sequence.

Committed before the first datum. The rule below is never edited. The outcome
is appended under OUTCOME, in this same file, after the run.

## 1. The claim under test

The ACCORD weak labeller is grossly miscalibrated in its MARGINAL: on the
1,183-row subject-disjoint training pool its hard positive rate is 0.8521,
against a gold base rate of 0.0401 on G299 (12 of 299). A student distilled
from those hard labels ranks the gold pairs at chance (C130 reproducible-label
control, G253 ensemble AUROC 0.5360, bootstrap CI 0.3454 to 0.7196).

C103 and C109 established that the same labeller's GRADED score does carry
rank signal: granite3.3:8b under the reversed wording `P_rev_agree` reaches
AUROC 0.7936 on G253 at permutation p = 0.0002
(`results/accord/prompt_sensitivity/c109/analysis.json::per_model_prompt.granite3.3:8b.P_rev_agree`).

Hypothesis: thresholding, not ranking, is what destroys the teacher's signal.
If the graded score is thresholded at the pilot prevalence rather than at the
model's own decision point, the distilled student should inherit the ranking.

## 2. Teacher configuration, fixed here

Exactly one configuration, chosen before the run as the strongest registered
cell in C109:

- model tag `granite3.3:8b` (local Ollama, loopback only)
- prompt `P_rev_agree`, imported from
  `accord/prompt_sensitivity/prompts_c109.py`, never retyped
- decoding identical to `accord/prompt_sensitivity/run_c109.py::call_chat`:
  `think=False`, `temperature=0`, `num_predict=1`, `logprobs=true`,
  `top_logprobs=8`, `num_ctx=4096`, `/api/chat`, system plus user message
- graded score = `flag_logodds` = `-(logprob(Yes) - logprob(No))`, the sign
  flip applied once, at the point of use, exactly as `run_c109.py` applies it
  for a `reversed` prompt. Higher means more conflict.
- score function imported from `run_c109.py` (`parse_and_score`), not
  reimplemented.

No other model, prompt, or decoding setting is run. If this configuration
cannot be run, the outcome is BLOCKED and no decision is recorded.

## 3. Label constructions compared

All arms label the SAME 1,183-row training pool from
`accord.control.data.build_splits().train` (subject-disjoint from gold, 291
subject-overlapping rows already excluded). Only the label column changes.

- **RAW**: the teacher's own hard label, `flag_label` from the same
  configuration. This is the same-teacher control and isolates thresholding
  from every other difference.
- **PM47** (PRIMARY): the top `n_pos` rows by `flag_logodds` are labelled
  positive, all others negative, with
  `n_pos = round(1183 * 0.04013377926421405) = 47`,
  where `0.04013377926421405` is the registered gold base rate on G299
  (`results/accord/prompt_sensitivity/analysis.json::gold_base_rate_299`).
  Ties at the threshold are broken by ASCENDING pool row index (the row's
  position in `accord_weak_supervision_labels_decontam.csv`), deterministically;
  the number of tied rows at the boundary is reported.
- **PM56** (sensitivity, no decision attached): `n_pos = round(1183 * 12/253) = 56`,
  the base rate of the G253 evaluation stratum itself.
- **PM95** (sensitivity, no decision attached): `n_pos = round(1183 * 0.0802675) = 95`,
  twice the pilot rate.

## 4. Student, fixed by reuse

`accord/control/train.py` is NOT edited. Its `build_splits` name is rebound to
a wrapper that returns the same `Splits` object with `train['y']` replaced, as
`accord/c130_reproducible_student/c130_run.py` does. Everything else runs
unchanged: fixed split seed `SPLIT_SEED = 1337`, AdamW at lr 2e-5, weight decay
0.01, batch 16, 3 epochs, max length 160, grad clip 1.0. Seeds 42, 123, 2024
per arm, one process per seed. The validation split keeps its ORIGINAL weak
labels in every arm, because validation loss is only monitored and never used
for selection; changing it would change nothing about the fitted model and
would make arms differ in a second place.

Evaluation: G253 (the 299 gold pairs minus the 46 pharmacist-ambiguous rows),
`pharm_pos = (pharm_label == "contradiction")`, 12 positives. Ensemble score is
the mean of the three seeds' sigmoid outputs, per pair. AUROC and average
precision, with 10,000-replicate nonparametric bootstrap CIs, reused from
`accord/control/report.py::bootstrap_auroc_ci`, `n_boot=10000`, `seed=42`.
G299 is reported as a secondary stratum.

## 5. DECISION RULE

**PRIMARY.** Prevalence-matched thresholding is declared to HELP if and only if
BOTH hold:

  (P1) the paired bootstrap 95% CI of
       `AUROC(PM47 ensemble) - AUROC(C130 reproducible-label student ensemble)`
       on the 253 shared gold rows excludes zero and its point estimate is
       positive; and
  (P2) the paired bootstrap 95% CI of
       `AUROC(PM47 ensemble) - AUROC(RAW ensemble)` on the same 253 rows
       excludes zero and its point estimate is positive.

Paired bootstrap: 10,000 resamples of the 253 gold rows with replacement,
`numpy.random.default_rng(43)`, replicates with a single class discarded and
redrawn, percentile interval, identical construction to
`c130_run.py::paired_diff_bootstrap`. C130's per-seed gold scores are read from
its persisted `scores_seed*.csv`, so the comparison is paired on pair_id, not
on two independently reported point estimates.

If (P1) and (P2) both hold: **Outcome A**, prevalence matching recovers the
teacher's ranking.
If either CI covers zero: **Outcome B**, no recoverable benefit demonstrated.
Outcome B is the expected result under the null and is reported as a negative
finding, not as a pending one.

**KILL CONDITION (Outcome C, invalid, no decision recorded).** Any of:
  (K1) `build_splits()` does not return 1,183 train rows and 291 excluded
       subject-overlapping rows;
  (K2) not every one of the 1,183 training rows has exactly one cached teacher
       score (join fan-out or a missing score);
  (K3) the teacher's parse-failure rate on the pool exceeds 0.05;
  (K4) fewer than 100 distinct `flag_logodds` values among the 1,183 rows (a
       degenerate graded score cannot be prevalence-thresholded meaningfully);
  (K5) the realised PM47 positive rate is not within 0.005 of 0.040134;
  (K6) any arm's seed fails to write both `scores_seed*.csv` and
       `metrics_seed*.json`.
A cell started a third time is refused, per the C109 convention.

No arm, seed, model, prompt, or value of k is added after the first datum. The
four arms above are the complete set.

## 6. Declared limitations, written before the run

**Double use of the gold set.** `k = 0.0401` is the base rate of the very gold
set the students are evaluated on. This is a real circularity and it is
declared here rather than discovered later. It is mild, for two reasons: k is a
single scalar, the marginal, and carries no per-row information about which
gold pairs are positive; and it is exactly the number a practitioner would hold
from an annotated pilot before deploying a labeller, so the procedure is one
that could actually be followed prospectively. It is not eliminated.

**Why k is not estimated from a held-out half of the gold set.** G253 contains
12 positives. A half of it carries about 6, so a half-sample estimate of k has
a binomial standard error near 0.0175 on a point estimate of 0.04: its 95%
interval runs roughly 0.006 to 0.075, a factor of twelve. A single held-out-half
k would therefore be an arbitrary draw from a very wide distribution and would
not be more honest than the full-sample k, only noisier and less reproducible.
The PM56 and PM95 arms are run instead, and the PM47/PM56/PM95 ladder brackets
that interval: if the conclusion is the same at 47, 56 and 95 positives, the
conclusion does not depend on knowing k precisely, and the held-out-half
objection is answered by the ladder rather than by one noisy split.

**Power.** G253 has 12 positives against 241 negatives. The corrected analytic
minimum detectable AUROC at alpha 0.05 and 80 percent power is **0.7382**
(`results/accord/audit_c137/a6_delong_mde.json::mde_alt_r4`), not the 0.7355
the thesis currently prints. Any true effect that lifts the student from 0.54 to
below about 0.74 is invisible here. Outcome B is therefore weak evidence of
absence, and must be reported as such: it rules out a large recoverable
benefit, not a modest one.

**MIMIC-III.** Credentialed data. Per-row teacher scores are cached only under
`~/mimic_local_only/`, as a pool row index plus numeric fields, never note
text. The committed artifact under `results/accord/` carries counts, rates,
AUROC, AP, CI bounds and booleans only: no note text, no HADM_ID, no
SUBJECT_ID, no pair_id. Local Ollama on loopback only; egress evidence
captured before the first call, as C103 and C109 do.

## OUTCOME

(appended after the run; the rule above is not edited)
