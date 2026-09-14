# C108 pre-registration: corrected rerun of C101 after a disclosed calibration bug

Committed before the first datum of this rerun. The rule below is never edited
after that commit; the outcome is appended afterwards.

## Why this exists

C101 (`accord/public_controls/C101_PREREGISTRATION.md`) is Outcome C, invalid,
and **stays** Outcome C: it is not replaced, retracted, or overwritten by this
document or by anything this document's run produces. C101's own outcome
section already names the bug and refuses to patch-and-silently-rerun on top
of a result already seen:

> `bisect_sigma_for_target` (in `accord/public_controls/evaluation.py`) has an
> early-return: if a single 20,000-per-side Monte Carlo draw at the search's
> upper sigma bound (50 z-scored standard deviations of noise) already reads
> below the target, it accepts that bound without bisecting further. At
> target 0.50 that single draw came back 0.5113, just above 0.50 by ordinary
> Monte Carlo noise ... tripped the early return, and the sigma search
> stopped at the cap without ever entering the bisection loop that checks the
> 0.002 tolerance. ... This is a bug in the calibration helper, not a change
> to a decision rule or a threshold, and per the kill condition it is NOT
> patched and silently rerun on top of a result already seen: this run stands
> as Outcome C, invalid, exactly as computed. A corrected
> `bisect_sigma_for_target` (removing the noisy single-draw early return, or
> widening its own margin before accepting a boundary value) is a new run
> under a new pre-registration, not an edit to this one.

This is that new pre-registration. C108 is a corrected rerun of C101's
positive-control measurement, after fixing exactly the calibration bug C101's
own outcome diagnosed, and nothing else.

## What is reused verbatim from C101 (quoted, not paraphrased)

Everything below is copied character-for-character from
`accord/public_controls/C101_PREREGISTRATION.md` as committed at `8ff9767`
("C101 Outcome C: noise-mix sigma search under-calibrated at target 0.50").
None of it is re-derived or re-justified here; it is reused as-is.

**Score source and score formula** (C101, "What runs"):

> **Score source.** `FacebookAI/roberta-large-mnli` (Liu et al., 2019,
> RoBERTa), downloaded by a separate agent to `~/models_public/roberta-large-mnli/`
> with a manifest at `results/accord/differential/nli_model_manifest.json`
> recording the commit SHA and a sha256 per file. Loaded offline only:
> `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`,
> `local_files_only=True`, CPU only. `id2label` is read from the loaded
> config and asserted to be `{0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}`
> before any score is computed; the run aborts if it is not, rather than
> silently using a different mapping.
>
> **Score.** For every item, `log p(CONTRADICTION) - log(1 - p(CONTRADICTION))`,
> where `p(CONTRADICTION)` is the softmax probability at the CONTRADICTION
> index read from `id2label`.
>
> **Data.** Every gold-labelled (`label != -1`) item in
> `accord/prevalence_replication/data/snli_validation.parquet` and
> `snli_test.parquet`, EXCLUDING every `(premise, hypothesis)` pair that
> appears in any file under `accord/prevalence_replication/data/pools/` (the
> sweep and distillation-supplement pools already used elsewhere in this
> thesis for a different purpose). Gold label convention, matching
> `build_pools.py`: entailment=0, neutral=1, contradiction=2;
> `is_contradiction = (label == 2)`.

**A0 validity precondition** (C101, "What runs"):

> **A0.** The empirical AUROC of this score against the excluded-pool gold
> set, computed once, is called A0 throughout. A0 >= 0.90 is a validity
> precondition (Outcome C below); this is a real natural-language-inference
> model on its own benchmark's dev/test split, so A0 is expected to be well
> above that floor.

**Grid and both constructions** (C101, "Construction of target AUROCs"):

> Grid: 0.50, 0.60, 0.65, 0.70, 0.72, 0.7355, 0.75, 0.78, 0.80, 0.85, 0.90.
>
> Every replicate has the instrument's exact shape: 12 positive slots, 241
> negative slots (n=253), matching G_253. Negative slots are always drawn,
> with replacement, from the true empirical negative-score pool (label !=
> contradiction, pool-excluded).
>
> **Label-mix (primary).** Each of the 12 positive slots is drawn from the
> true positive-score pool with probability `1 - lambda`, else from the true
> negative-score pool (a slot mislabelled positive at score-generation
> time). `lambda = (A0 - target) / (A0 - 0.5)`, clipped to `[0, 1]`. This
> construction's population AUROC is exact: for the mixture score `X`
> (positive slot) against `Y` (negative slot), `P(X > Y) = (1 - lambda) *
> P(Pos > Neg) + lambda * P(Neg' > Neg) = (1 - lambda) * A0 + lambda * 0.5`,
> because the second mixture component is drawn from the same distribution
> as `Y` and so beats it exactly half the time in expectation (ties broken
> at 0.5, as usual for near-continuous scores). Solving for `lambda` at a
> given target gives the formula above. No target above A0 is achievable by
> this construction (lambda would be negative); the grid is read against
> the achieved A0 and any grid point above A0 is reported as capped at A0,
> not silently extrapolated.
>
> **Noise-mix (secondary).** Scores are z-scored using the full
> excluded-pool population's mean and standard deviation. For a target
> AUROC, a noise standard deviation `sigma` is found by bisection (AUROC is
> monotonically non-increasing in `sigma`, from A0 at `sigma=0` to 0.5 as
> `sigma` grows) so that ... a Monte Carlo estimate of the population AUROC
> (20,000 simulated positive and 20,000 simulated negative draws, iid
> `N(0, sigma^2)` noise added to a with-replacement resample of the z-scored
> pool) is within 0.002 of the target. Each replicate then resamples 12
> positive-slot and 241 negative-slot scores from the z-scored pools with
> replacement and adds fresh iid `N(0, sigma^2)` noise.

**Decision rules, replicate counts, seeds** (C101, "Decision rules, replicate
counts, seeds"):

> Six rules, each a boolean "fires" on one constructed replicate (y = 12
> ones then 241 zeros; s = that replicate's drawn scores):
>
> - `mw_one_sided`: one-sided (greater) Mann-Whitney U, p < 0.05.
> - `mw_two_sided`: two-sided Mann-Whitney U, p < 0.05.
> - `c98_rule`: AUROC >= 0.7355 AND `mw_one_sided` fires. This is C98's rule,
>   read verbatim off its own pre-registration and reimplemented for
>   measurement only; `accord/differential/C98_PREREGISTRATION.md` and
>   `graded_score_probe.py` are never edited or re-run by this work.
> - `delong_single_one_sided`: one-sided DeLong z-test of H0: population
>   AUROC = 0.5, using the DeLong variance of the single AUROC estimate
>   (Sun and Xu's fast DeLong algorithm, via `accord/control/delong.py`,
>   reused not reimplemented).
> - `delong_paired_vs_null`: paired DeLong test (same module) of the
>   replicate's score against a fresh, iid `N(0,1)` null score evaluated on
>   the identical 253 slots, one-sided in the direction that the real score
>   beats the null.
> - `bootstrap_lower_excludes_chance`: the one bootstrap-type rule.
>   Stratified percentile bootstrap, B=10,000, the exact RNG order pinned
>   against `accord/stats/intervals.py`'s own bootstrap loop (reimplemented,
>   not imported, in `accord/public_controls/evaluation.py`, because
>   `intervals.py` reads MIMIC at import time). Fires if the bootstrap
>   distribution's 2.5th percentile excludes 0.5.
>
> Replicate counts per grid point, per construction: 4,000 for the first
> five rules; 1,000 for `bootstrap_lower_excludes_chance` (its own B=10,000
> inner loop makes 4,000 outer replicates too slow to run at this
> instrument's size within this work's budget; 1,000 is still fifty times
> the 20-replicate floor the rest of this project's planted-defect work uses
> to call a check validated).
>
> Seeds: `numpy.random.SeedSequence(42)`, spawned once per (rule, grid
> point, construction) so no seed is reused across cells, and the same
> spawn tree is used for every rerun (see Kill condition).

**Decision rule** (C101, "Decision rule (pre-registered)"):

> **Primary:** `mw_two_sided` under the label-mix construction, read against
> the registered key `mde_recomputed.gold_ambiguous_excluded.minimum_detectable_auc`
> (0.7355).
>
> - **Outcome A:** the interpolated AUROC at which `mw_two_sided`'s
>   empirical power (label-mix) first reaches 0.80 is within 0.02 of
>   0.7355, AND the empirical rejection rate at true AUROC 0.50 is in
>   `[0.03, 0.07]` (a sane false-positive rate around the nominal 0.05).
>   Reported as: this instrument's own power curve, measured on a real
>   score distribution at its own size, agrees with the analytic and
>   synthetic-binormal MDE.
> - **Outcome B:** the crossing is more than 0.02 from 0.7355. Reported as:
>   the direction of the disagreement (empirical crossing above or below
>   the analytic value) and the empirical crossing itself, stated as the
>   operative limit for this instrument rather than the analytic figure.
> - **Outcome C, invalid:** the empirical rejection rate at true AUROC 0.50
>   falls outside `[0.02, 0.08]`, OR any constructed population AUROC
>   (checked by a large, n=200,000-per-side Monte Carlo estimate, held out
>   from the replicate loop itself) is off its target by more than 0.005,
>   OR A0 < 0.90, OR the run does not complete every grid point x
>   construction x rule cell. Reported as not run; no crossing is quoted.
>
> **Always reported, regardless of outcome:** the interpolated 80%-power
> crossing for every one of the six rules under both constructions, and
> `c98_rule`'s empirical rejection rate at exactly AUROC = 0.7355 (this is
> the number the KEY FINDINGS paragraph calls "about half power at the
> MDE"; it is measured here, not assumed).

**Kill condition** (C101, "Kill condition"):

> If a run is interrupted at any grid point, rule, or construction, it
> restarts from scratch using the same `SeedSequence(42)` spawn tree; no
> partial result is spliced in with a different seed. The score source
> (`FacebookAI/roberta-large-mnli`, the exact manifest sha256 above) is
> never switched once any score from it has been seen or used in a
> computation; a different model would need a new pre-registration.

C108 adopts the identical kill condition: **interrupted runs restart from
scratch** (same `SeedSequence(42)` spawn tree, from the beginning); **no
change is made to any rule, grid, threshold, replicate count, or seed after
any datum from this rerun is produced.**

## The ONLY change: the calibration fix

C101's diagnosis names the bug precisely: `bisect_sigma_for_target`'s early
return accepted a single, noisy 20,000-per-side Monte Carlo draw at its sigma
cap without ever checking the 0.002 tolerance, at the noise-mix target-0.50
cell (achieved 0.5112, 0.0112 off target). The fix, applied in
`accord/public_controls/evaluation.py` (shared by C101 and C102; C101's own
files, including `accord/public_controls/sizectl/power_curves.py` and
`C101_PREREGISTRATION.md`, are untouched by this fix and remain exactly as
they were when C101's outcome was recorded):

1. `bisect_sigma_for_target` no longer estimates the noise-mix construction's
   population AUROC by Monte Carlo sampling at all. A new function,
   `noise_mix_population_auroc`, computes it EXACTLY (deterministically, no
   RNG) on the full z-scored pools, using the closed form
   `P(X > Y) = mean over all (i,j) pairs of Phi((pos_z[i] - neg_z[j]) / (sigma*sqrt(2)))`
   (X, Y independent draws with fresh iid `N(0, sigma^2)` noise, so
   `X - Y = (pos_z[i]-neg_z[j]) + (N_i - N_j)` with `N_i - N_j ~ N(0, 2*sigma^2)`).
   This has zero sampling error, so there is no standard error for an unlucky
   draw to hide behind.
2. `target == 0.50` is handled explicitly, not searched for: as sigma grows
   without bound, `Phi(x/(sigma*sqrt(2))) -> Phi(0) = 0.5` for every finite
   `x`, so the construction's population AUROC approaches 0.50 only in the
   sigma -> infinity limit, never reaching it at any finite sigma for these
   pools. `bisect_sigma_for_target` returns a dedicated `INFINITE_SIGMA`
   sentinel with an exact achieved AUROC of 0.5 for this one grid point;
   `draw_noise_mix_replicate` realizes that limit exactly, not
   approximately, by drawing both the positive-slot and negative-slot scores
   from the SAME pooled (positive+negative) z-scored distribution -- pure
   label permutation -- which has population AUROC exactly 0.5 by symmetry.
3. Every remaining code path raises `ValueError` rather than returning a
   boundary value if the achieved AUROC is not within the pre-registered
   0.002 tolerance of the target. There is no early return left that can
   accept an unchecked value.

Nothing about the grid, the six rules, the replicate counts, the seeds, the
label-mix construction, the decision rule, or the outcome thresholds changes.
The noise-mix construction's DEFINITION does not change either (still: resample
the z-scored pools with replacement, add fresh iid Gaussian noise); only HOW
the sigma that realizes a given target AUROC is found, and how the target-0.50
edge case is realized, changes.

A unit test (`tests/test_sizectl_calibration.py`) pins that every grid target
is hit within the pre-registered 0.002 tolerance, directly on the same real
SNLI/RoBERTa pools C101's bug manifested on, for both constructions.

## What is different about this run, mechanically

- **Artifacts.** Written to `results/accord/public_controls/sizectl_c108/`,
  never `results/accord/public_controls/sizectl/` (C101's own artifacts are
  not touched, read, or overwritten by this run).
- **Score rerun, not reuse.** Rather than copying C101's already-scored
  `roberta_scores.json`, this rerun re-executes the identical scoring
  procedure (`accord/public_controls/sizectl_c108/score_roberta.py`, a
  byte-for-byte copy of C101's `score_roberta.py` except for the output
  path) against the same model manifest and the same data, so C108 has its
  own, independently-produced score artifact under its own path. The score
  source is not switched (same `FacebookAI/roberta-large-mnli`, same
  manifest sha256s, verified by an assertion in the script itself), per the
  kill condition quoted above.
- **Register script.** `accord/public_controls/sizectl_c108/register.py`,
  prefix `sizectl108.` (not `sizectl.`), so it can never collide with or
  overwrite C101's `sizectl.*` register keys. Written but NOT run by this
  agent, matching C101's own convention.

## Artifacts

- `results/accord/public_controls/sizectl_c108/roberta_scores_meta.json`
- `results/accord/public_controls/sizectl_c108/roberta_scores.json`
- `results/accord/public_controls/sizectl_c108/power_curves.json`
- Register script: `accord/public_controls/sizectl_c108/register.py`, prefix
  `sizectl108.`. Written but NOT run.

## Outcome

Recorded after the run; the rule above is unchanged.
