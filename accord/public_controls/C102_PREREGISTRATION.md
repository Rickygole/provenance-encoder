# C102 pre-registration: a planted-defect benchmark for this project's own checks

Committed before the first datum. The outcome is appended below after each stage
runs; the rule above the outcome line is never edited. Templates for any text used
by any stage are committed with their sha256 in this file, in the section for the
stage that uses them, before that stage runs; the reduced scope below uses no
generated note text at all (see Stage 1).

## Why

`accord/control/verify_rq3_falsifications.py` hardcodes `"falsified": True` for its
three checks; it cannot report anything else regardless of what it is given, so it
is not a check and nothing in this thesis may treat it as one. Item 10 replaces the
question "does this check pass" with the question this project has not asked of any
of its own checks: does it actually FIRE on a known defect, and does it stay quiet on
a clean run? A check that always says the same thing either way is measured here as
what it is.

## General verdict rule (applies to every stochastic check below)

A stochastic check is **validated** if, over 20 replicates on its designated plant,
it fires in >= 18 of 20, AND on EVERY clean arm registered for it, it fires in <= 2
of 20. It is **unvalidated_false_alarm** if it fires in > 5 of 20 on any clean arm,
**specificity_marginal** if 3-5 of 20 on any clean arm, and
**not_validated_sensitivity** if it fires in < 18 of 20 on its plant. Deterministic
(arithmetic) checks -- ones whose output is a fixed number given a fixed seed and
fixed inputs, not a count over independent replicates -- are reported separately,
as their own numbers, never folded into a validated-check count. No result in this
work is ever reported as "N checks validated" as a single number; each check is
named with its own counts. Alongside every count, the binomial probability of
observing that count's verdict is reported at true underlying rates of 0.5, 0.7, 0.9,
and 0.95, so a count of 18/20 can be read against how likely that count is if the
check's true firing rate were each of those values.

## Kill condition (applies to every stage)

An interrupted stage restarts from scratch with the same seeds; no partial stage
result is spliced with a different seed. Thresholds fixed in this document are never
changed after a replicate has been run at them, including on pilot seeds.

## Reduced scope for this build

Per the coordinator's instruction, the multi-mention synthetic-text generator and the
deterministic S1a check are dropped. Stage 1 is simulated at the candidate level
only: no note text, template, or generator is built or committed for Stage 1, so
there is no template hash to record for it. Stages 3 and 4 need a frozen BiomedBERT
feature cache and MPS; this document pre-registers their checks and thresholds now,
but only a 2-tiny-replicate CPU smoke test is run for Stage 3 in this build, and
Stage 4 is not run at all (it needs Stage 3's output).

---

## Stage 1 (reduced): S1b, defect-free-stratum disagreement

**Substrate.** G_253-sized (n=253) candidate population, split into the four gold
rule categories with counts read read-only from the register key
`goldstrata.join_feasibility.gold_rule_counts.{1,2,3,4}` (134, 31, 79, 55; sum 299),
apportioned down to n=253 by Hamilton's largest-remainder method so the four
category sizes stay proportional to the registered counts and sum exactly to 253.
Twelve true positives (matching this project's registered `n_positives_total`, 12)
are apportioned across the same four categories by the same method, so the baseline
positive rate is uniform across every category before any defect is planted (see
below): this is what makes the clean arm clean.

Rules 1-2 are pooled as group A; rules 3-4 (the "defect-free" stratum used
throughout this thesis, e.g. `intervals.py`'s E3) are pooled as group B.

**Annotator model.** The synthetic annotator confirms a candidate if and only if it
is a true positive AND its exemplar "carries" the conflict. Every true positive in
group A always has its exemplar carry the conflict (failure rate 0, always). Every
true positive in group B has its exemplar fail to carry the conflict independently
with probability `r` (the planted rate); a failed exemplar means that true positive
is NOT confirmed. No non-positive candidate is ever confirmed. At `r=0`, every true
positive in both groups is confirmed, so the confirmed rate in each group equals its
apportioned positive rate, which is uniform by construction: this is the clean arm,
and the expected confirmed count is exactly 12 of 253.

**Plant.** Exemplar failure rate `r` applied to group B only, at `r = 0.25` and `r =
0.47`.

**Check.** One-sided Fisher exact test (`alternative="greater"`) on the 2x2 table
`[[confirmed_A, not_confirmed_A], [confirmed_B, not_confirmed_B]]`, testing whether
group A's confirmed odds exceed group B's. Fires if p < 0.05.

**Replication.** For each rate in `{0, 0.25, 0.47}`: one designated "resample 0"
(the first draw from a `numpy.random.SeedSequence(42)` child stream reserved for
this stage), reported as the headline p-value and fire/no-fire; and 1,000
independent resamples from the same stream (of which the first 20 also serve the
general validated/unvalidated verdict above) reporting the empirical fire rate.

**Read as:** `r=0` is the clean arm (target false-alarm rate <= 2/20, i.e. <=10%,
consistent with alpha=0.05 read loosely); `r=0.25` and `r=0.47` are two plant
severities, each read against the >=18/20 sensitivity bar independently, so the
report states at which of the two rates (if either) the check is validated.

---

## Stage 2: latent score, marginal/feasibility and bootstrap checks

**Substrate.** n=253, base rate 12/253 (matching G_253). Latent label `y` is
Bernoulli(12/253). Score `s = y * delta + N(0,1)`, with `delta = sqrt(2) *
Phi^{-1}(0.90)` (the equal-variance binormal construction used identically by
`accord/stats/injection_sensitivity.py`, giving population AUROC(y, s) = 0.90
exactly under that model).

**Clean.** Symmetric label flips applied to `y` (each label flipped independently
with the SAME probability regardless of its value) so the expected marginal
(positive rate) is unchanged at 12/253 in expectation, and the flip is independent
of `s`'s generating value of `y`... concretely: `y_obs` starts as the true `y`, and
independently with probability `f` each entry is redrawn as a fresh
Bernoulli(12/253) draw (a "flip to a fresh coin toss", which is a marginal-preserving
corruption, not a directional one). `f = 0.10` for the clean arm.

**L1 (plant, "flag high scores").** The observed positive call is
"top 85% of candidates by score are flagged positive", regardless of `y_obs`: this
recovers most of the true signal (since `s` correlates with the true `y`) but at an
extreme, uninformative flag rate (85% flagged) that a rate-based labeller with no
real threshold-setting would produce.

**L2 (plant, "flag at random").** The observed positive call flags a uniformly
random 85% of candidates, independent of `s` and of `y`: same flag rate as L1, zero
information.

**Checks.**
- **S2a, marginal feasibility floor (deterministic).** `attainable_floor_p_one_sided`
  from `accord/stats/marginal_feasibility_gate.py` (reused, not reimplemented), given
  `n=253`, `n_pos=12`, and the arm's flag count. Check: floor > 0.05. This is
  arithmetic, not a replicate count; reported as its own number per arm, not folded
  into a validated/unvalidated count.
- **S2b, bootstrap ratio (stochastic, B=10,000 inner).** Stratified percentile
  bootstrap (the same pinned reimplementation as C101's `bootstrap_lower_excludes_chance`)
  of the ratio (observed marginal positive rate) / (base rate 12/253); fires if the
  bootstrap distribution's lower 2.5th percentile exceeds 2.0.
- **S2c, `c98_rule` on the score.** AUROC(y_obs, s) >= 0.7355 AND one-sided
  Mann-Whitney p < 0.05, using `y_obs` (the arm's observed/corrupted label) against
  the continuous score `s`. Designed to fire on clean and on L1 (both retain real
  signal in `s` relative to `y_obs`) and NOT fire on L2 (flags are independent of
  `s`, but note `y_obs` is still the corrupted-`y`-based label for clean/L1, while for
  L2 the label IS the random flag itself, so `c98_rule` there is testing `s` against
  pure noise and should not fire above its false-alarm rate).

**Replication.** 20 replicates per arm for S2b and S2c (stochastic); S2a is computed
once per arm (deterministic, exact given the arm's flag count).

---

## Stage 3: detector-defect checks (code and thresholds pre-registered; CPU smoke test only in this build)

Needs a frozen BiomedBERT feature cache and MPS for the full, validated run; not run
in this build beyond a 2-tiny-replicate CPU smoke test that proves the code executes
and returns finite numbers. The smoke test does NOT produce a validated/unvalidated
verdict; that determination is deferred to the coordinator's MPS run.

Classes are the real, frozen `AuthorityTemporalConflictGraph` and
`GuidelineGroundedHarmEstimator` from `accord/detector/accord_phase3_v2_ablations.py`,
loaded by `accord/public_controls/frozen_ablations_loader.py` (ast-parsed,
sha256-pinned, only the allow-listed ClassDef/Assign nodes executed; the file's
`pip install`s and Google Drive mount are never reached). Smoke-test dimensions are
tiny (`node_dim=hidden_dim=8`, `num_heads=2`) and untrained (random init); the
FULL run substitutes the frozen file's real dimensions and the trained checkpoint's
weights, and is gated by the coordinator releasing the GPU.

- **S3a, authority-scalar normalisation order.** Displacement (mean absolute
  difference in `conflict_score`) between a forward pass using
  `AuthorityTemporalConflictGraph.authority_matrix` as-is (the frozen file's own
  behaviour: an unnormalized scalar multiplies the attention scores) and a forward
  pass with a row-normalized copy of the same matrix substituted in, same input,
  same other weights. Fixed threshold: displacement < 0.02. Deterministic given a
  fixed seed and fixed weights.
- **S3b, auxiliary rule-id head gradient-RMS ratio.** With `prior_vector`'s
  rule-derivable feature set so the rule id target is an exact function of it (the
  plant: a shortcut the auxiliary head can solve without using the shared trunk) vs.
  a rule id target drawn independently of every engineered feature (the clean arm,
  standing in for "a text-carried template family" the engineered prior cannot see):
  the ratio of the gradient RMS the rule-id loss contributes to
  `GuidelineGroundedHarmEstimator.input_projection`'s shared weights, over the
  gradient RMS the contradiction loss contributes to the same weights. Fixed
  threshold: ratio < 0.05 (the shortcut plant should route almost no gradient
  through the shared trunk). Deterministic given a fixed seed and fixed weights.
- **S3c, macro-F1 drop when the rule-id feature is zeroed.** Macro-F1 of
  `rule_logits` argmax vs. the rule-id target, computed with the shortcut feature
  present vs. zeroed out in `prior_vector`, same weights, same other inputs. Fixed
  threshold: drop >= 0.20 on the shortcut plant.
- **S3d, harm-head weight-change ratio.** L2 norm of the change in
  `harm_head`'s weights after one optimizer step, divided by the L2 norm of its
  initial weights; computed once with the harm loss included in the backward pass
  (clean: the head should update) and once with it excluded (plant: "harm head
  outside the loss", the head should not update at all). Fixed threshold: ratio <
  1e-6 on the excluded-plant arm.

---

## Stage 4: large synthetic gold, real detector scores (code pre-registered; not run, needs Stage 3's trained output)

**Substrate.** Synthetic gold at 12/253 (matching this project's own basis).
**Plant.** The real, trained detector's scores against this synthetic gold (from
Stage 3's eventual full run). **Clean.** The same scores against an independent
permutation of the synthetic gold labels (breaking any true association).

**Check S4.** `c98_rule` (AUROC >= 0.7355 AND one-sided Mann-Whitney p < 0.05) on
resample 0, plus the fire rate over 1,000 resamples, on both plant and clean.

Not run in this build; `accord/public_controls/plantbench/stage4.py` defines the
function and a `main()` that exits, printing that it needs Stage 3's artifact, rather
than fabricating an input.

## Artifacts

- `results/accord/plantbench/stage1_s1b.json`
- `results/accord/plantbench/stage2_s2abc.json`
- `results/accord/plantbench/stage3_smoke.json` (smoke test only, explicitly labelled
  not a verdict)
- `results/accord/plantbench/stage4_*.json` -- not produced in this build.
- Register script: `accord/public_controls/plantbench/register.py`, prefix
  `plantbench.`. Written but NOT run by this agent; the coordinator runs it once the
  concurrent register write elsewhere is done.

## Outcome

Recorded below after each stage; the rule above is unchanged.

**Outcome letters, restated:** **A**, every stochastic check registered here is
validated. **B**, at least one is not, named explicitly with its counts; the thesis
may not rely on an unvalidated check. **C**, invalid: any stage's calibration fails
after the fallback described for it above, or a stage is incomplete.
