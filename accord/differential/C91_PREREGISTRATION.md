# C91 third-vendor discrimination replication — pre-registered decision rule

**Written 2026-09-12, BEFORE any Gemini run was started.** No MIMIC pair has been sent to
this model or this vendor. Do not edit after the first number lands; if the rule turns out to
be badly specified, record that as a correction rather than revising this file.

## Why this exists

The strongest claim in this thesis is that frontier labellers given one prompt and no training
separate the pharmacist's confirmed cases on the non-ambiguous core, where every system this
pipeline built ranks at chance. Four Fisher exact tests carry it, and all four are Anthropic:
Sonnet 5 and Haiku 4.5, each in two prompt conditions. The Bedrock runs add a second vendor to
the *over-calling* axis but not to this one; they were pre-registered as capability and channel
controls and produce no discrimination test on G_253.

So the headline rests on one maker. Adversarial review put the objection directly: the
attenuation and the discrimination may both be properties of one vendor's training recipe, and
the design cannot separate that from capability. C67 recorded the same limitation and could not
close it, because a second Anthropic model deepens the confound rather than fixing it.

A third vendor from a different family, on the same 253 rows, under the same prompt and the same
payload, is the arm that closes it. PhysioNet's guidance of 18 April 2023 names Google Gemini via
Vertex AI as one of four services it suggests for credentialed data, and unlike Azure it asks for
no opt-out form, so the route is available without a further prerequisite.

## What is being tested

Whether the discrimination result replicates on a vendor outside Anthropic.

The run: Gemini via Vertex AI in the author's own Google Cloud project, through the same script,
the same recovered prompt, the same synthetic demonstrations, the same 300 pairs, the same parser
and the same evaluation as the Sonnet and Haiku runs. The provider is the only variable that
changes. Both prompt conditions, 600 prompts, for the reason C67 gives: the within-model prompt
effect at 1.6800 is larger than the channel effect at 1.0962, so a single condition cannot bound
it.

## The decision rule

Read Gemini's G_253 **zero-shot** Fisher exact p against the Bonferroni threshold the enlarged
family carries, and its Matthews correlation bootstrap interval against zero.

### Outcome A — the claim generalises beyond one vendor

> **Gemini's G_253 zero-shot Fisher exact p clears the corrected threshold AND its Matthews
> interval excludes zero at the conventional level.**

Then the discrimination result holds across two makers and three models. Chapters 4, 5 and 6 may
say "frontier labellers" rather than "Anthropic models", and the one-vendor limitation C67 left
open is closed and recorded as closed.

### Outcome B — the claim narrows to one vendor

> **Gemini's zero-shot Fisher exact p fails to clear the corrected threshold, OR its Matthews
> interval contains zero.**

Then the discrimination result does not replicate outside Anthropic on this design. Every
sentence in this thesis that says a frontier labeller, or frontier labellers generally, separates
these cases is narrowed to name Anthropic models specifically, and the failure to replicate is
reported in the same paragraph as the original result rather than in a limitations list. The
comparison against the pipeline survives in the narrowed form; the generalisation does not.

### Outcome C — split between conditions

> **The zero-shot and few-shot conditions disagree about whether the interval excludes zero.**

Then the result is reported as measured under both conditions and no generalisation claim is made
in either direction, because the Sonnet rows already show the two conditions can disagree and this
thesis reads zero-shot as the dependent condition.

## The multiplicity consequence, stated before the run and not after

Two Fisher exact tests enter the hypothesis family whatever the outcome, because this chapter will
report their p values. That takes k from 26 to 28 and the Bonferroni threshold from 0.0019231 to
0.0017857.

**This costs a rejection the thesis currently holds, regardless of what Gemini returns.** Claude
Haiku 4.5's few-shot ambiguity gradient sits at p = 0.0018909. It clears 0.0019231 and does not
clear 0.0017857. It is recorded here, before the run, that this test leaves the clearing set as a
direct consequence of running this experiment, so that it can never be presented afterwards as a
surprise, as a reason not to report the new tests, or as evidence that the family was enlarged in
a direction chosen after the fact. C88 and C90 are the record of this project getting exactly that
wrong in both directions.

## What is NOT being tested, and cannot be by this run

Channel. Every high-capability point in this thesis remains a hosted API and every low-capability
point remains local serving, so compliance and the capability confound are still the same
variable. A third vendor does not touch it. The obligation on any continuation is unchanged: a
larger open-weights model served locally, which 16GB of unified memory cannot reach above roughly
14B.

Capability ordering. Gemini's parameter count is not public, so it contributes no numeric scale
point and none will be invented. It is a vendor replication, not a capability rung, and no row of
the capability ladder is recomputed from it.

Contamination. Whether any frontier model has seen MIMIC-III in pretraining is not knowable from
outside and is not claimed either way for any vendor here.

## Cost and stopping

600 calls across both conditions, the same payload as the five runs already enumerated in the
methodology chapter: one mining-extracted sentence per note side, no admission identifier, no
subject identifier, no free note body. Expected under two dollars. It is run once. If it fails
partway, it is restarted from the beginning and that is recorded; partial runs are not spliced.

## Outcome

> **NOT YET RUN.** This line is filled in after the result lands and the rule above is not edited.
