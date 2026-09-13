# C94 local prompt control — pre-registered decision rule

**Written 2026-09-12, BEFORE any local run under the frontier template was started.** The three
models are already installed and have been used in this project; no new model is downloaded and
nothing is transmitted. Do not edit after the first number lands; if the rule turns out to be
badly specified, record that as a correction rather than revising this file.

## Why this exists

`capability_ladder.py` discloses, twice, that the Local `llama3:8b` row was run under
`config_id=llama_baseline` while the other four rows were run under the frontier template. The
consequence is stated in the thesis's own words: the channel effect of 1.0962 "is an upper bound
confounded with an uncontrolled prompt difference, not a clean channel comparison."

That confound is not small relative to what it contaminates. The measured prompt effect is
1.6800, which is larger than the channel effect it sits inside and comparable to some capability
steps. A three-factor decomposition whose smallest factor is the one carrying an uncontrolled
version of its largest factor is not a decomposition.

The fix needs no new data, no new model, no hosted API and no approval. It is the same three 8B
models, already on this machine, under the template the other rows used.

## What is being run

The three installed 8B models, `llama3:8b`, `qwen3:8b` and `granite3.3:8b`, on the same 300
annotated pairs, under the frontier zero-shot and few-shot templates, through the same script,
parser and evaluation as every other row. The serving path is local Ollama. Nothing leaves the
machine. The only variable that changes against the existing local row is the prompt.

**A recorded deviation, fixed in advance.** `qwen3:8b` returns an empty string under the archived
`max_tokens` of 50: the budget is consumed by a thinking block and the parser sees nothing, which
was confirmed by a single synthetic call before this file was written. The token ceiling is
raised for that model and the deviation is recorded in its output, exactly as the same failure was
recorded for the reasoning models on the hosted routes. A run whose `failed_parses` is not zero is
discarded and re-run, not reported.

## The decision rule

Recompute the channel effect at 8B as the ratio of the local 8B over-calling ratio to the hosted
8B over-calling ratio, both now under the frontier template, on the candidate pool.

### Outcome A, the ordering survives

> **The recomputed channel effect is below 1.6800, the measured prompt effect.**

Then the three factors keep the order the thesis states, capability first at 2.0635, prompt
second at 1.6800, channel third. Chapter 4 keeps its ordering sentence and reports the clean
channel value in place of the bounded one, and the disclosure that the old value was confounded
becomes a correction entry rather than a live caveat.

### Outcome B, the ordering flips and the chapter changes

> **The recomputed channel effect is at or above 1.6800.**

Then serving channel moves ahead of prompt and the sentence "capability first, prompt second,
channel third" is wrong as written. It is replaced by the measured order, and the thesis states
that the ordering it previously reported rested on a confounded estimate of the smallest factor.
No result is withheld on the ground that it reverses a published ordering. This outcome is named
here, before the run, precisely so that it cannot later look like a finding chosen after the fact.

### Outcome C, inconclusive

> **The recomputed channel effect lands within the spread the three 8B models already show at
> fixed capability**, which is 15.4167 to 23.7500 on the pool, a factor of 1.54.

Then the channel term is not separable from between-model variation at this scale and no ordering
claim is made about it in either direction. The thesis reports the three factors with channel
marked as not resolved by this design.

## What is NOT being tested, and cannot be by this run

Capability at the high end. These are three 8B models. The hosted rows remain the only
high-capability points and this run does not replace them, does not retire them, and is not
evidence about them. Any reading of this experiment as a substitute for the hosted ladder is
wrong: the measured hardware ceiling on this machine is roughly 14B, and a 70B model needs about
four times the available memory and more than the free disk.

Compliance for runs already made. 1,320 prompts were sent to the Anthropic API and 1,200 to
Bedrock. A local run does not un-send them. This experiment changes what happens prospectively
and changes nothing retrospectively, and the disclosure obligation for the completed
transmissions is unaffected by anything measured here.

Quantisation. All three local models are 4-bit builds and the hosted precisions are unpublished,
so the recomputed channel term still bounds channel together with precision. That confound is
narrowed by this run, not removed, and the thesis says so either way.

## Cost and stopping

1,800 prompts, three models in two prompt conditions each, at roughly 0.3 seconds per prompt
warm. No money, no transmission, no approval. It is run once per model. If a model's run fails
partway it is restarted from the beginning and that is recorded; partial runs are not spliced.

## Outcome

> **OUTCOME A, recorded 2026-09-12.** The prompt-matched channel effect is 1.0038, below the
> 1.6800 prompt effect the rule set as the threshold, so the ordering stands: capability first at
> 2.0635, prompt second at 1.6800, channel third at 1.0038. The rule was committed at 0140c77
> before the first call and was not edited after the result.
>
> The outcome is cleaner than Outcome A anticipated. The rule asked only whether the channel term
> stayed below the prompt term. It did, and it also collapsed: from 1.0962, an upper bound
> confounded with the prompt, to four parts in a thousand. Almost the whole of the apparent
> serving-channel effect was the prompt difference the ladder disclosed against itself.
>
> Two departures from the plan, both recorded rather than tidied. The qwen3 run did not complete:
> the machine's memory manager stopped it twice, once inside the batch and once running alone, and
> its rung is recorded as absent in the generated ladder rather than dropped from it. No partial
> run was spliced. And the raised token ceiling for that model, pre-recorded here, was applied.
> Nothing in the decision rule depends on the missing rung: the channel step is one model against
> itself across two serving paths, and both of those runs completed with zero unparseable pairs.
