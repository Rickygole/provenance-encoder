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
- `results/accord/plantbench/stage3_smoke.json` (calibration smoke test, explicitly
  labelled not a verdict)
- `results/accord/plantbench/stage3_registered.json` (the registered Stage 3 run:
  real dims, MPS, 20 replicates per arm per check)
- `results/accord/plantbench/stage3_detector_scores_for_stage4.json` (Stage 3's
  bridge output, consumed by Stage 4)
- `results/accord/plantbench/stage4_s4.json`
- Register script: `accord/public_controls/plantbench/register.py`, prefix
  `plantbench.`. Written but NOT run by this agent; the coordinator runs it once the
  concurrent register write elsewhere is done.

## Outcome

Recorded 2026-09-14 after all four stages; the rule above is unchanged.

**Overall outcome: B.** At least one registered stochastic check is not validated,
named below with its counts; nothing in this thesis may rely on an unvalidated
check. No stage was incomplete and no stage's calibration fell back and then
failed, so the run does not qualify as C; every stochastic check that ran is
named with real counts, not "N checks validated" as a single number.

### Stage 1, S1b (candidate-level, no note text)

`not_validated_sensitivity` at both registered plant rates. Clean (r=0) is
deterministic in this design (no exemplar ever fails, so the table is fixed by
the apportioned counts): 0/20 false alarms. Plant r=0.25: 0/20 fires (rate over
1,000 resamples 0.001). Plant r=0.47: 0/20 fires (rate over 1,000 resamples
0.028). The defect-free stratum apportions to only 5 true positives at G_253
scale (rules 3-4, Hamilton apportionment of the registered rule counts), too few
for the one-sided Fisher test to reliably separate rules 1-2 from rules 3-4 even
at the higher planted rate.

### Stage 2 (S2a deterministic, S2b/S2c stochastic, 20 replicates each)

S2a: deterministic, reported per arm, not a fire-count verdict. Clean's flag
rate (near base rate) gives a near-zero feasibility floor; L1 and L2 (both flag
85%) give an identical, higher floor (0.135) regardless of accuracy, exactly the
scope this check documents for itself.

S2c (`c98_rule` on the score): `validated`. Clean 20/20 fires, L1 20/20 fires,
L2 0/20 fires -- matches the pre-registration's own stated expected pattern
exactly.

S2b (bootstrap ratio): the committed rule states the fire condition but not
which arms are its plant vs its clean arm. Both readings consistent with the
committed text are reported, neither chosen after seeing the data: reading
"plant = over-called arms (L1, L2), clean = clean" gives `validated` (L1 20/20,
L2 20/20, clean 0/20); reading "plant = signal arms (clean, L1), clean = L2"
gives `unvalidated_false_alarm` (L2 fires 20/20, over the 5/20 ceiling). S2b is
therefore named as **not validated under at least one reading** and the thesis
may not rely on it without first tightening this ambiguity in a new
pre-registration.

### Stage 3 (real frozen-file dimensions, MPS, 20 replicates per arm per check, untrained)

GPU/CPU freedom verified immediately before running (`curl .../api/ps` empty,
no training `pgrep`), per the coordinator's gate. Calibration (the original
2-tiny-replicate CPU smoke test) re-run first, unchanged in outcome, before any
registered-run datum.

- **S3a, authority-normalisation displacement: `not_validated_sensitivity`.**
  Plant (frozen file's actual, non-row-normalized `authority_matrix` init)
  0/20 fires, displacement 0.00025-0.00055 across replicates, never near the
  0.02 threshold even at real (256-dim) scale. Clean (pre-row-normalized
  starting matrix, added when this run was built since the smoke test never
  had a clean arm) 0/20 fires, displacement exactly 0.0 every replicate.
- **S3b, rule-id shortcut gradient-RMS ratio: `not_validated_sensitivity`.**
  Plant 0/20 fires, ratio 0.90-1.09; clean 0/20 fires (not applicable to a
  false-alarm count since neither arm fires), ratio 0.91-1.15. Plant and
  clean are statistically indistinguishable: an untrained model has not
  learned a shortcut through a single forward/backward pass, so there is
  nothing yet for this check to detect.
- **S3c, rule-feature-ablation macro-F1 drop: `not_validated_sensitivity`.**
  Plant 0/20 fires, drop -0.021 to +0.019 (noise around zero); clean 0/20
  fires, drop -0.021 to +0.017. An untrained `rule_type_head` has not learned
  to rely on any feature, so zeroing one feature changes nothing.
- **S3d, harm-head weight-change ratio: `validated`.** Plant (harm loss
  excluded from backward) 20/20 fires, ratio exactly 0.0 every replicate (no
  gradient path exists at all when the term is omitted). Clean (harm loss
  included) 0/20 fires, ratio 0.0031-0.0043, always far above the 1e-6
  threshold. This check's signal is a computational-graph connectivity
  property and does not depend on training.

S3a, S3b and S3c all need trained weights to exhibit the effect they are
designed to detect (a real normalization sensitivity, a real shortcut, a real
learned feature-reliance); none is in this build's scope, which never trains
anything and has no BiomedBERT feature cache. This is reported as the honest
registered-run finding: **3 of 4 Stage 3 checks are not validated at this
build's available scope**, not patched around by loosening a threshold or
inventing training that was never authorized.

### Stage 4 (S4, on Stage 3's bridge output)

`not_validated_sensitivity`. Input: the real, frozen
`AuthorityTemporalConflictGraph` + `GuidelineGroundedHarmEstimator` pipeline's
raw score on 253 synthetic items (untrained, real dims, MPS), paired with an
independent 12/253 synthetic gold (not constructed to correlate with the
score, since there is no trained detector in this build's scope). Plant
(case-bootstrap of the true pairing): resample 0 does not fire, fire rate
0.002 over 1,000 bootstrap resamples, 0/20 in the first 20. Clean (permutation
null): resample 0 does not fire, fire rate 0.001 over 1,000 resamples, 0/20 in
the first 20. `c98_rule` essentially never fires on either arm, which is the
expected, calibration-appropriate behavior for scores that carry no trained
relationship to the gold, not a defect in the check.

### Deterministic vs stochastic, restated

Deterministic (arithmetic, one number per arm, no replicate count): S2a. Every
other check above (S1b, S2b, S2c, S3a-d, S4) is stochastic, reported as a
fire-count out of 20 (or out of 1,000/1,000 where the pre-registration asked
for both) with its own binomial-probability table, never collapsed into a
single "N checks validated" figure.

### What failed and what did not

Nothing crashed or ran incomplete: every stage produced the artifact its
pre-registration named. What "failed" in the ordinary sense is a finding, not
an error: three of Stage 3's four checks, Stage 4, Stage 1's S1b, and one
reading of Stage 2's S2b are not validated at the scope this build actually
has (no trained detector, no BiomedBERT cache, small defect-free strata at
G_253 size). Validated: S2c, S3d, and S2b under one of its two readings.

Artifacts: `results/accord/plantbench/stage3_registered.json`,
`stage3_detector_scores_for_stage4.json`, `stage4_s4.json` (added to the list
below). Register script (not run): `accord/public_controls/plantbench/register.py`,
prefix `plantbench.*`.

## Correction to the Outcome above (2026-09-14, coordinator review)

The rule above is unchanged. The Outcome recorded above for Stages 3 and 4 is
corrected, not replaced, so both remain visible.

The pre-registration committed at 22bd7ec specifies that the full Stage 3 run
"substitutes the frozen file's real dimensions and the trained checkpoint's weights",
and that Stage 4's plant is "the real, trained detector's scores". The Stage 3 and
Stage 4 runs recorded above used the real dimensions with untrained, randomly
initialised weights, and Stage 4 consumed those untrained scores. The runs therefore
did not complete Stages 3 and 4 as registered.

Consequences under the rule as written:
- S3a, S3b and S3c are defined on behaviour a trained model can exhibit (sensitivity
  to normalisation order, a learned shortcut's gradient routing, reliance on a
  feature). Untrained weights cannot exhibit it, so their 0 of 20 firings are not
  evidence that the checks lack sensitivity. They are recorded as NOT RUN AS
  REGISTERED, with no verdict.
- S4 received scores carrying no signal by construction, so its 0 of 20 is likewise
  not evidence about the check. NOT RUN AS REGISTERED, no verdict.
- S3d tests whether any gradient reaches the harm head when its loss is excluded
  versus included. That property holds for untrained weights, so its result (fires on
  the plant 20 of 20, 0 of 20 on the clean arm) stands, reported with that caveat.
- Stages 1 and 2 are unaffected.

Overall C102 outcome under the rule: **C for Stages 3 and 4** (stages incomplete as
registered). The verdicts that stand are S1b not validated, S2c validated, S2b under
both readings, S2a deterministic, and S3d validated. The thesis may rely on S2c and
S3d only, and may not describe S3a, S3b, S3c or S4 as validated or as not validated.
