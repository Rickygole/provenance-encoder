# C72 within-channel capability test — pre-registered decision rule

**Written 2026-09-12, BEFORE any MIMIC pair was sent to this model.** A synthetic
sentence pair has confirmed the endpoint answers: `meta.llama3-8b-instruct-v1:0`
returned `'Contradiction.'` in 5 generated tokens, stop reason `stop`. Do not edit
after the first number lands; if the rule turns out to be badly specified, record
that as a correction rather than revising this file.

## Why this exists

C68 removed the vendor from the over-calling comparison and left the serving
channel in it. Its 8B arm is `llama3:8b` served locally through Ollama and its 70B
arm is `meta.llama3-70b-instruct-v1:0` served through AWS Bedrock, so capability
and channel still move together across the step the test reported. Chapter 4 states
that limitation in the note to the capability table and Chapter 6 names closing it
as the cheapest open item in the thesis. This closes it.

The confound is not hypothetical. A hosted endpoint differs from a local one in
ways that could plausibly move a decision rate: a different quantisation, a
different default sampling configuration, a different chat template applied
server-side, and a system prompt the caller does not see. Any of those could
produce the 2.2619x step C68 attributed to scale.

## What is being tested

`meta.llama3-8b-instruct-v1:0` on AWS Bedrock, in the author's own account, through
the same script, prompt, exemplars, pairs, parser and evaluation as every other
run in this comparison. `BEDROCK_MODEL` is the only variable that changes from the
C68 run. That gives both capability tiers of one family on one channel, which is
the comparison C68 could not make.

Reference values, all on G_299 zero-shot, all already committed:

| labeller | channel | ratio |
| --- | --- | --- |
| `llama3:8b` | local | 23.7500 |
| `qwen3:8b` | local | 15.4167 |
| `meta.llama3-70b-instruct-v1:0` | Bedrock | 10.5000 |

The local 8B cluster spans 15.4167 to 23.7500. The question is whether a hosted 8B
lands in that cluster or down near the hosted 70B.

## The decision rule

Read the over-calling ratio on G_299, zero-shot.

### Outcome A — capability orders the quantity, channel does not

> **The ratio is at or above 15.4167**, that is, inside the range the three local
> 8B labellers occupy.

Then serving channel does not account for the step, the within-family comparison of
C68 is a capability effect, and Chapter 4's capability claim and Chapter 5's reading
both stand as written. The limitation note in the capability table is replaced by
this result and Chapter 6's fourth future-work item is struck.

### Outcome B — the step is part capability and part channel

> **The ratio falls strictly between 10.5000 and 15.4167.**

Then hosting alone moved an 8B model out of the local 8B cluster without changing
its scale, so the 2.2619x step of C68 is a compound of capability and channel and
cannot be reported as capability alone. Every sentence attributing the within-family
step to capability is rewritten to name both factors, and the step is reported as an
upper bound on the capability component.

### Outcome C — the step is channel, not capability

> **The ratio is at or below 10.5000**, that is, a hosted 8B behaves like the
> hosted 70B rather than like its own local twin.

Then the within-family capability claim of C68 is refuted: what looked like a
capability ordering across a factor of roughly nine in parameters is the serving
path. Outcome A of C68 is withdrawn, the capability claim comes out of Chapters 1,
4, 5 and 6, and what survives is that hosted labellers over-call less than locally
served ones, which is a different and more troubling finding because it would mean
this thesis's own instrument was reading the channel.

## What this run still cannot settle

It does not control quantisation, which is the most likely mechanism if Outcome B or
C lands. Bedrock does not publish the serving precision for this model and the local
Ollama build is a 4-bit quantisation, so a difference between them is expected on
priors; this run measures whether it reaches the decision rate, not why.

It does not control the prompt. C68 found a 1.6800x prompt effect within the 70B,
larger than the capability step it was measuring. Both conditions are run here for
that reason, but two conditions bound a prompt effect rather than measure it, and
the rule above reads zero-shot only, fixed in advance.

It is one draw per pair at temperature 0.0. Bedrock accepts a temperature parameter
where Sonnet does not, so this arm is reproducible in a way the Anthropic arms are
not, and that asymmetry is a property of the comparison rather than of the models.

## Cost and stopping

600 calls across both conditions. The synthetic check returned 5 generated tokens
against a 16-token cap, so the run is expected to cost under a dollar. It is run
once. If it fails partway it is restarted from the beginning and that is recorded;
partial runs are not spliced.

## Where the outcome goes

CORRECTIONS.md C72, written after the numbers land, in the convention of C44. The
experiment label and the correction number coincide this time; C67 and C68 did not,
and C71 records why.
