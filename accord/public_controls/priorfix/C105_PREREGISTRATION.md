# C105 pre-registration: does a prior-corrected soft target rescue distillation?

Committed before the first datum. The outcome is appended below after the run;
the rule above the outcome line is never edited (CLAUDE.md convention set by C44).

## Question

At the prevalence this project's clinical evaluation actually has (12/253),
does correcting a teacher's zero-shot log-odds for the mismatch between its
implicit training-time prior and the deployment prevalence -- rather than
retraining, rethresholding, or changing the student -- rescue a distilled
student's ranking of the positive class, on public data only.

## Data (public only, all SNLI)

- **Teacher pool**: Item 11's `llama3:8b` x `P1` cell (1,000 SNLI
  contradiction positives it already labelled at full scale) plus a top-up
  of 2,000 SNLI non-contradictions (1,000 entailment, 1,000 neutral) drawn
  from a partition disjoint from every other partition this project has
  used (Item 11's main/pilot/determinism draws and
  `accord/prevalence_replication`'s pools), labelled by `llama3:8b` under
  the same `P1` prompt, in the same call shape Item 11 used. Teacher pool:
  1,000 positives, 4,000 negatives, all SNLI.
- **10 replicates**, each independently drawing:
  - a training pool of 1,679 items at prevalence 12/253 (80 positives,
    1,599 negatives), sampled from the teacher pool without replacement
    within the replicate;
  - a gold-only sample of 253 items drawn i.i.d. at p = 12/253 from a
    disjoint, gold-labelled-only SNLI partition (never scored by the
    teacher for this purpose beyond computing `pi_hat`), giving
    `pi_hat = count / 253`;
  - all three draws (teacher top-up, training pool, gold-only sample) come
    from mutually disjoint SNLI partitions, asserted, not assumed, in
    `build_priorfix_partitions.py`.
  - a fixed evaluation set of 4,000 SNLI items (190 positives; prevalence
    ~4.75%, matching this project's registered `gold.base_rate_253`), drawn
    once (not per replicate) from data the teacher never labelled.

## Arms (same training pool, same student, four arms)

- **H**: hard teacher labels (argmax of teacher log-odds).
- **S1**: prior-corrected soft targets. The Saerens, Latinne and
  Decaestecker (2002) adjustment (`saerens2002adjusting`), implemented here
  as an intercept shift `b` on the teacher's log-odds `z`, solved by
  bisection so that `mean(sigmoid(z + b))` over the training pool equals
  `pi_hat`. **Disclosed explicitly**: this is the intercept-shift form of
  the correction, not the paper's iterative EM procedure, because a
  zero-shot LLM teacher has no known, estimable training-time class prior
  to initialize an EM iteration from; an intercept shift is the minimal
  correction that uses only the one quantity this setting actually supports
  (the deployment-side `pi_hat`) and recovers the paper's calibration
  target as its fixed point.
- **T**: hard labels for the top `round(pi_hat * 1679)` items by teacher
  z-score (rank-based pseudo-labelling at the estimated prevalence).
- **S0**: uncorrected `sigmoid(z)` as a soft target (no prior correction),
  the ablation that isolates what the intercept shift buys over the
  teacher's raw calibration.

## Student

`accord/control/model.py::PairClassifier` on BiomedBERT
(`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`), with
`accord/control/train.py`'s defaults: 3 epochs, batch size 16, lr 2e-5,
weight decay 0.01, max length 128, seed = the replicate index. Checked
before use: importing `accord.control.model` and `accord.control.train`
reads no MIMIC file at import time (both confirmed by inspection: `model.py`
has no data path at all; `train.py`'s only MIMIC-shaped path, `OUT_DIR`, is a
module-level string, not a read, and every actual file read lives inside
functions called from `train.py`'s own `main()`, which this pre-registration
never calls). This run reimplements the training loop directly against SNLI
data in `run_priorfix.py` rather than importing `accord.control.train`'s
`main()`, so no MIMIC-shaped path is ever constructed, read, or written by
this run. `accord.control.data` is not imported at all.

Training runs on MPS and must not overlap any Ollama model being loaded;
every Ollama model is unloaded (`OLLAMA_MAX_LOADED_MODELS=1` on this machine
already enforces at most one loaded model, but this run additionally
confirms no `ollama ps` entry is loaded before starting a training process).

## Primary

Delta-AP (S1 minus H) on the evaluation set, plus delta-AUROC. Intervals by
two-level percentile bootstrap (replicates, then items within each
replicate's evaluation-set predictions; B=10,000, seed 42).

- **Outcome A, fix works**: mean delta-AP > 0 with the interval's lower
  bound > 0, AND mean delta-AUROC > -0.01.
- **Outcome B, anything else**: reported as the fix not rescuing
  distillation, with the numbers.
- **Outcome C, invalid**: teacher marginal on a training pool below 2x
  `pi_hat`; teacher degeneracy above 10%; or a replicate incomplete.

## Secondary

- Student flag rate at threshold 0.5, per arm, against `pi_hat` and the
  teacher's own marginal on that replicate's training pool.
- The resampled 12/253 C98-style rejection rate per arm: for each
  bootstrap replicate's evaluation-set predictions, apply C98's rule
  (AUROC >= 0.7355 and one-sided Mann-Whitney p < 0.05, reimplemented via
  `accord/public_controls/evaluation.py::rule_c98`, imported, not
  retyped) and report the fraction of resamples that would have rejected.

## Kill condition

No other correction method, threshold, student architecture, or teacher is
tried after results are in. An interruption restarts all 40 runs (10
replicates x 4 arms), never resumed partway.

## Artifacts

`results/accord/public_controls/priorfix/*`. Register script prefix
`priorfix`, written but not run by this agent; the coordinator runs it.

## Outcome

_Recorded below after the run completes; not yet run as of this commit._
