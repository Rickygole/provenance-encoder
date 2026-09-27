# Pre-registration: data-randomisation control for the plain control detector

Committed BEFORE the first training step of this run. The decision rule below
is never edited; the outcome is recorded in the "Outcome" section at the
bottom of this same file afterwards.

## Why

Chapter 2 reviews Adebayo and colleagues' sanity checks, the data
randomisation test among them, and Chapter 3 then records that none of those
designs was run. The data randomisation test is the one that is cheap here:
retrain the detector on labels whose association with the inputs has been
destroyed, and see whether it performs any differently. If it does not, the
real labels carried nothing the model used.

## Design

Exactly the C130 configuration, with one change: the training-pool label
vector is randomly permuted across rows before training.

- Trainer: `accord/control/train.py`, unedited, invoked the way
  `accord/c130_reproducible_student/c130_run.py` invokes it (the
  `build_splits` name it imported is replaced by a wrapper; every other line
  of `train.py` runs unchanged).
- Training pool: `accord.control.data.build_splits().train`, the 1,183-row
  subject-disjoint pool, split seed `data.SPLIT_SEED = 1337`, fixed.
  Gate: `n_train == 1183` and `n_leaked_rows_excluded == 291`.
- Labels before permutation: the reproducible llama3:8b replication labels
  (`pool_replication_labels_llama3_8b.csv`), the SAME label vector C130
  trained on, so this run differs from C130 only by the permutation.
- Permutation: one draw, `numpy.random.default_rng(PERMUTATION_SEED).permutation`
  over the 1,183 training rows, with `PERMUTATION_SEED = 137` recorded here
  before the run. ONE permutation, held FIXED across the three model seeds,
  so that seed-to-seed spread reflects model stochasticity only, exactly as
  in the control protocol (`accord/control/data.py`: "only model
  stochasticity varies by seed"). The permutation preserves the label
  marginal exactly; that is gated.
- Validation labels are left as C130 left them, and are not permuted. They
  enter nothing but the printed loss: there is no early stopping and no model
  selection (three fixed epochs).
- Hyperparameters, all from the existing control run: AdamW, lr 2e-5, weight
  decay 0.01, batch size 16, 3 epochs, max length 160, grad-norm clip 1.0.
- Seeds: 42, 123, 2024.
- Scoring: the same 299 gold pairs, `G_253` = the 241 confirmed negatives plus
  the 12 confirmed contradictions, ambiguous dropped. Ensemble = mean of the
  three per-seed sigmoid scores, as in `accord/control/report.py`.

## Decision rule

Reference, the real-label run this mirrors:
`c130.g253.ensemble.auroc = 0.5359612724757953`, with percentile bootstrap
95 percent interval `[0.345434996612753, 0.7195712309820194]`
(B = 10000, seed 42), AP `0.08298569987933005`.

- **Outcome A, the labels carried nothing the model used.** The permuted-label
  ensemble AUROC on `G_253` lies inside the real-label run's bootstrap 95
  percent interval, AND the paired bootstrap 95 percent interval for
  (real ensemble AUROC minus permuted ensemble AUROC) on the same 253 pairs
  contains zero.
- **Outcome B, the labels carried something.** The paired bootstrap interval
  for the difference excludes zero in the real run's favour.
- **Outcome C, invalid, no decision recorded.** Any of the kill conditions
  below fires.

## Kill conditions

1. `len(splits.train) != 1183` or `n_leaked_rows_excluded != 291`.
2. The pool-label join is not one-to-one, or any training row lacks a
   replication label.
3. The permutation does not preserve the label marginal exactly (positive
   count before != positive count after).
4. Any seed fails to write both `scores_seed{N}.csv` and
   `metrics_seed{N}.json`.
5. The gold row order of any seed's score file disagrees with this run's own
   `build_splits().gold` order, or disagrees across seeds.
6. The permuted training pool ends up single-class (nothing to train on).

## What this design cannot establish

Stated before the run so it cannot be softened after it. There are 12
positives in `G_253`. The real-label run's own interval is
`[0.345, 0.720]`, which is 0.375 AUROC wide and contains 0.5. An interval
that wide cannot distinguish "the labels carried nothing" from "the labels
carried an effect this study is too small to see". Outcome A is therefore a
statement that the real-label run is INDISTINGUISHABLE from label-destroyed
training at this sample size, and is not a measurement that the real labels
are uninformative. One permutation draw is also not a permutation null
distribution: it establishes that one label-destroying draw performs
comparably, not the distribution over draws.

## Outcome

Recorded after the run. The decision rule above was not edited. Artifact:
`results/accord/labeller_interval_closure/permuted_label_control.json`.

All three gates passed. Three seeds trained on MPS, the run having waited for
two other processes to release the GPU first.

**Outcome A.** The permuted-label ensemble AUROC on `G_253` is
0.4810 with AP
0.0570 and bootstrap 95 percent interval
[0.3238, 0.633], against the
real-label run's 0.5360 / AP
0.0830 and
[0.3454, 0.7196]. The permuted
value lies inside the real-label interval. The paired bootstrap difference on
the same 253 pairs, real minus permuted, is 0.0561 with 95 percent
interval [-0.1108, 0.2255], which contains zero. Both
halves of the Outcome A criterion are met.

Per seed on `G_253`: 0.5041, 0.5062, 0.4333.

The permutation moved 100 of 1183
training rows (0.0845), because the
replication label vector is 1132/1183
positive. That is a property of the weak labels, not of the permutation, and
it is recorded here because it limits how much the test could ever have
shown: an almost constant label vector has very little association with the
inputs to destroy. A follow-up arm that draws labels independently of the
inputs rather than permuting them would destroy the label fully, and is not
run here because it was not pre-registered.
