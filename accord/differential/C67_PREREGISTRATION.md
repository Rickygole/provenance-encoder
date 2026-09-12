# C67 capability sweep — pre-registered decision rule

> **OUTCOME RECORDED 2026-09-11: OUTCOME A.** Haiku 4.5's G_299 zero-shot
> over-calling ratio is 6.2500, strictly inside the 5.3333 to 15.4167 window the
> rule below required. The rule was committed at 05d9cdd before any MIMIC pair
> reached the model and was not edited after the result. Full result in
> CORRECTIONS.md **C71**, which also records that this line pointed at the wrong
> entry twice before landing there: the experiment labels C67 and C68 were
> assigned before the correction record reached those numbers, and C67 there is
> the detector-source recovery. The rule text below was never edited; only this
> pointer moved.

**Written 2026-09-11, BEFORE the Haiku runs were started.** The credential check on a synthetic
sentence pair has been run and returned HTTP 200; no MIMIC pair has been sent to this model. Do not
edit after the first number lands; if the rule turns out to be badly specified, record that as a
correction rather than revising this file.

## Why this exists

C65 withdrew the GPT-4o rows. That left the capability axis with three 8B labellers on the gold set
at 15.4167 to 23.7500 and a frontier labeller at 4.5000 and 5.3333, and nothing between them.

An adversarial review then made an objection this project has to take seriously: adding a
mid-capability point *after* seeing that the gap weakens the claim, with no committed prediction, is
the same move `tab:m-prereg` exists to prevent, one level up. Its exact form was that a claim which
cannot be embarrassed by its own next datum is not a finding. It is right, and the answer is to
write the prediction down first.

## What is being tested

Chapter 4 currently says over-calling "attenuates with capability" and Chapter 5 that the shift
"tracks capability." Both are interpolations across a gap the design never sampled. Within the 8B
cluster, at roughly fixed scale, the ratio already spans 15.4167 to 23.7500 — a factor of 1.54 with
capability held approximately constant — so capability is demonstrably not the only thing moving
this number.

The run: Claude Haiku 4.5 through the same script, prompt, exemplars, pairs, parser and evaluation
as the Sonnet runs. `ANTHROPIC_MODEL` is the only variable that changes. Both prompt conditions,
because the withdrawn GPT-4o evidence put the within-model prompt effect at 1.99x, larger than the
entire across-model span at 8B, so a single condition cannot bound it.

**Capability proxy, fixed here and not after.** Neither Haiku's nor Sonnet's parameter count is
public, so no numeric scale axis is available and none will be invented. The proxy is the vendor's
own published tier ordering: Haiku below Sonnet, both above an 8B open-weights model. That is an
ordinal claim and only ordinal predictions are made from it. This also retires "two orders of
magnitude" at `msu_ch5_discussion.tex:241`, which asserts a parameter-count ratio the register
cannot source.

## The decision rule

Read the over-calling ratio on G_299, zero-shot, against the three 8B labellers (15.4167, 18.5000,
23.7500) and Sonnet (5.3333).

### Outcome A — the ordering claim survives

> **Haiku's G_299 zero-shot ratio falls strictly between 5.3333 and 15.4167.**

Then the axis has an ordered three-tier sequence measured through one channel under one protocol,
and "over-calling attenuates with capability" is supported by a sampled gradient rather than an
interpolation. Chapter 4 keeps the sentence and reports the new row.

### Outcome B — the ordering claim is retired

> **Haiku's ratio is at or below Sonnet's 5.3333, or at or above the 8B minimum of 15.4167.**

Then capability tier does not order this quantity, and the words "attenuates with capability" and
"tracks capability" come out of Chapters 4 and 5 and are replaced by the two-cluster statement:
frontier labellers over-call far less than 8B labellers on this pool, and what separates them is
not identified by this design. The pool attribution stays withdrawn either way, because that rests
on the Sonnet rows alone and does not need a gradient.

### Outcome C — inconclusive

> **Haiku lands inside the 8B cluster's own spread**, that is within 15.4167 to 23.7500.

Then the result is reported as measured and no ordering claim is made in either direction, because
the 8B cluster's internal 1.54x span already shows that something other than tier moves this
number at fixed scale.

## What is NOT being tested, and cannot be by this run

Vendor. Every hosted row in this thesis is Anthropic, and a second Anthropic model deepens that
rather than fixing it. Whatever the outcome, the limitation stands and is stated in the same
paragraph as the contamination caveat: the frontier attenuation may be a property of one vendor's
training recipe, and this design cannot separate that from capability.

Channel. Every low-capability point is local Ollama and every high-capability point is a hosted
API, so compliance and the confound are the same variable. The only fix is a larger open-weights
model run locally, which 16GB of unified memory cannot reach above roughly 14B. Recorded as an
obligation on any continuation, not closed here.

## Cost and stopping

600 calls across both conditions. The synthetic check returned 65 input and 4 output tokens, so the
run is expected under two dollars. It is run once. If it fails partway, it is restarted from the
beginning and that is recorded; partial runs are not spliced.
