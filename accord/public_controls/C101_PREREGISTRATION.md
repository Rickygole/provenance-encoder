# C101 pre-registration: a positive control at the evaluation's own size

Committed before the first datum. The outcome is appended below after the run; the
rule above the outcome line is never edited.

## Why

The 253-pair, 12-positive gold instrument (G_253) that the thesis reads every local
score against has a registered minimum detectable AUROC of 0.7355 at 80% power
(`mde_recomputed.gold_ambiguous_excluded.minimum_detectable_auc`). That is an
analytic property of the instrument's size, derived and separately verified by
simulation on synthetic binormal data (`accord/stats/injection_sensitivity.py`). What
has never been checked is whether a REAL, independently-known-good discriminator,
scored at exactly this instrument's size, behaves the way the instrument's own power
curve says it should. This run builds that positive control: a model with no
connection to ACCORD, no connection to MIMIC, and a large, human-labelled evaluation
set of its own, scored once, then subsampled down to G_253's exact shape (12
positive slots, 241 negative slots) many times over a grid of constructed target
AUROCs, to trace out the instrument's OWN empirical power curve on a real score
distribution rather than a synthetic binormal one.

This is not a claim about ACCORD's detector. It is a check on the evaluation
instrument, run once, independent of anything ACCORD has ever scored.

## What runs

**Score source.** `FacebookAI/roberta-large-mnli` (Liu et al., 2019, RoBERTa),
downloaded by a separate agent to `~/models_public/roberta-large-mnli/` with a
manifest at `results/accord/differential/nli_model_manifest.json` recording the
commit SHA and a sha256 per file. Loaded offline only: `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, `local_files_only=True`, CPU
only. `id2label` is read from the loaded config and asserted to be
`{0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}` before any score is computed; the run
aborts if it is not, rather than silently using a different mapping.

**Score.** For every item, `log p(CONTRADICTION) - log(1 - p(CONTRADICTION))`, where
`p(CONTRADICTION)` is the softmax probability at the CONTRADICTION index read from
`id2label`.

**Data.** Every gold-labelled (`label != -1`) item in
`accord/prevalence_replication/data/snli_validation.parquet` and
`snli_test.parquet`, EXCLUDING every `(premise, hypothesis)` pair that appears in any
file under `accord/prevalence_replication/data/pools/` (the sweep and
distillation-supplement pools already used elsewhere in this thesis for a different
purpose). Gold label convention, matching `build_pools.py`: entailment=0,
neutral=1, contradiction=2; `is_contradiction = (label == 2)`.

**Storage.** Scores are stored keyed by a source index and a
`sha256(premise + "\x1f" + hypothesis)[:16]` hash, with the split, the gold label,
and the score; no premise or hypothesis text is written anywhere. Summary statistics
(A0, counts, exclusion count, model manifest sha256) go to
`results/accord/public_controls/sizectl/roberta_scores_meta.json`.

**A0.** The empirical AUROC of this score against the excluded-pool gold set,
computed once, is called A0 throughout. A0 >= 0.90 is a validity precondition
(Outcome C below); this is a real natural-language-inference model on its own
benchmark's dev/test split, so A0 is expected to be well above that floor.

## Construction of target AUROCs

Grid: 0.50, 0.60, 0.65, 0.70, 0.72, 0.7355, 0.75, 0.78, 0.80, 0.85, 0.90.

Every replicate has the instrument's exact shape: 12 positive slots, 241 negative
slots (n=253), matching G_253. Negative slots are always drawn, with replacement,
from the true empirical negative-score pool (label != contradiction, pool-excluded).

**Label-mix (primary).** Each of the 12 positive slots is drawn from the true
positive-score pool with probability `1 - lambda`, else from the true negative-score
pool (a slot mislabelled positive at score-generation time). `lambda = (A0 -
target) / (A0 - 0.5)`, clipped to `[0, 1]`. This construction's population AUROC is
exact: for the mixture score `X` (positive slot) against `Y` (negative slot),
`P(X > Y) = (1 - lambda) * P(Pos > Neg) + lambda * P(Neg' > Neg) = (1 - lambda) * A0
+ lambda * 0.5`, because the second mixture component is drawn from the same
distribution as `Y` and so beats it exactly half the time in expectation (ties
broken at 0.5, as usual for near-continuous scores). Solving for `lambda` at a given
target gives the formula above. No target above A0 is achievable by this
construction (lambda would be negative); the grid is read against the achieved A0
and any grid point above A0 is reported as capped at A0, not silently extrapolated.

**Noise-mix (secondary).** Scores are z-scored using the full excluded-pool
population's mean and standard deviation. For a target AUROC, a noise standard
deviation `sigma` is found by bisection (AUROC is monotonically non-increasing in
`sigma`, from A0 at `sigma=0` to 0.5 as `sigma` grows) so that a Monte Carlo estimate
of the population AUROC (20,000 simulated positive and 20,000 simulated negative
draws, iid `N(0, sigma^2)` noise added to a with-replacement resample of the
z-scored pool) is within 0.002 of the target. Each replicate then resamples 12
positive-slot and 241 negative-slot scores from the z-scored pools with replacement
and adds fresh iid `N(0, sigma^2)` noise.

## Decision rules, replicate counts, seeds

Six rules, each a boolean "fires" on one constructed replicate (y = 12 ones then 241
zeros; s = that replicate's drawn scores):

- `mw_one_sided`: one-sided (greater) Mann-Whitney U, p < 0.05.
- `mw_two_sided`: two-sided Mann-Whitney U, p < 0.05.
- `c98_rule`: AUROC >= 0.7355 AND `mw_one_sided` fires. This is C98's rule, read
  verbatim off its own pre-registration and reimplemented for measurement only;
  `accord/differential/C98_PREREGISTRATION.md` and `graded_score_probe.py` are never
  edited or re-run by this work.
- `delong_single_one_sided`: one-sided DeLong z-test of H0: population AUROC = 0.5,
  using the DeLong variance of the single AUROC estimate (Sun and Xu's fast DeLong
  algorithm, via `accord/control/delong.py`, reused not reimplemented).
- `delong_paired_vs_null`: paired DeLong test (same module) of the replicate's score
  against a fresh, iid `N(0,1)` null score evaluated on the identical 253 slots,
  one-sided in the direction that the real score beats the null.
- `bootstrap_lower_excludes_chance`: the one bootstrap-type rule. Stratified
  percentile bootstrap, B=10,000, the exact RNG order pinned against
  `accord/stats/intervals.py`'s own bootstrap loop (reimplemented, not imported, in
  `accord/public_controls/evaluation.py`, because `intervals.py` reads MIMIC at
  import time). Fires if the bootstrap distribution's 2.5th percentile excludes 0.5.

Replicate counts per grid point, per construction: 4,000 for the first five rules;
1,000 for `bootstrap_lower_excludes_chance` (its own B=10,000 inner loop makes 4,000
outer replicates too slow to run at this instrument's size within this work's
budget; 1,000 is still fifty times the 20-replicate floor the rest of this project's
planted-defect work uses to call a check validated).

Seeds: `numpy.random.SeedSequence(42)`, spawned once per (rule, grid point,
construction) so no seed is reused across cells, and the same spawn tree is used for
every rerun (see Kill condition).

## Decision rule (pre-registered)

**Primary:** `mw_two_sided` under the label-mix construction, read against the
registered key `mde_recomputed.gold_ambiguous_excluded.minimum_detectable_auc`
(0.7355).

- **Outcome A:** the interpolated AUROC at which `mw_two_sided`'s empirical power
  (label-mix) first reaches 0.80 is within 0.02 of 0.7355, AND the empirical
  rejection rate at true AUROC 0.50 is in `[0.03, 0.07]` (a sane false-positive rate
  around the nominal 0.05). Reported as: this instrument's own power curve, measured
  on a real score distribution at its own size, agrees with the analytic and
  synthetic-binormal MDE.
- **Outcome B:** the crossing is more than 0.02 from 0.7355. Reported as: the
  direction of the disagreement (empirical crossing above or below the analytic
  value) and the empirical crossing itself, stated as the operative limit for this
  instrument rather than the analytic figure.
- **Outcome C, invalid:** the empirical rejection rate at true AUROC 0.50 falls
  outside `[0.02, 0.08]`, OR any constructed population AUROC (checked by a large,
  n=200,000-per-side Monte Carlo estimate, held out from the replicate loop itself)
  is off its target by more than 0.005, OR A0 < 0.90, OR the run does not complete
  every grid point x construction x rule cell. Reported as not run; no crossing is
  quoted.

**Always reported, regardless of outcome:** the interpolated 80%-power crossing for
every one of the six rules under both constructions, and `c98_rule`'s empirical
rejection rate at exactly AUROC = 0.7355 (this is the number the KEY FINDINGS
paragraph calls "about half power at the MDE"; it is measured here, not assumed).

## Kill condition

If a run is interrupted at any grid point, rule, or construction, it restarts from
scratch using the same `SeedSequence(42)` spawn tree; no partial result is spliced in
with a different seed. The score source (`FacebookAI/roberta-large-mnli`, the exact
manifest sha256 above) is never switched once any score from it has been seen or
used in a computation; a different model would need a new pre-registration.

## Artifacts

- `results/accord/public_controls/sizectl/roberta_scores_meta.json` (A0, counts,
  exclusions, model manifest reference; no item text or per-item score).
- `results/accord/public_controls/sizectl/roberta_scores.json` (per-item scores,
  keyed by hash and source index only, no text).
- `results/accord/public_controls/sizectl/power_curves.json` (the full grid x
  construction x rule power table, both crossings, `c98_rule` at 0.7355).
- Register script: `accord/public_controls/sizectl/register.py`, prefix `sizectl.`.
  Written but NOT run by this agent; the coordinator runs it once the concurrent
  register write elsewhere is done.

## Outcome

Recorded below after the run; the rule above is unchanged.
