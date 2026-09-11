# C68 within-vendor capability test — pre-registered decision rule

**Written 2026-09-11, BEFORE any MIMIC pair was sent to this model.** A synthetic
sentence pair has been used to confirm the endpoint answers (4 generated tokens,
stop reason `stop`). Do not edit after the first number lands; if the rule turns
out to be badly specified, record that as a correction rather than revising this
file.

## Why this exists

C67 established an ordering across capability tiers, but every low-capability
point in that comparison is an open-weights model run locally through Ollama and
every high-capability point is a hosted API from a single vendor. Capability,
vendor and serving channel therefore vary together, and the design cannot say
which of the three orders the quantity. A reviewer can attribute the 2.4667x step
to Anthropic's post-training or to the serving path, and this project has no
answer.

This run removes the vendor from the comparison. `meta.llama3-70b-instruct-v1:0`
on AWS Bedrock is the same model family, same maker and same instruction-tuning
lineage as `llama3:8b`, which is already measured locally at 23.7500 on the gold
set. Bedrock is one of the routes PhysioNet's guidance of 18 April 2023 names as
acceptable, and the account is the author's own.

So the comparison becomes: one vendor, one model family, two capability tiers,
two serving channels. If the ordering holds here it is not a vendor artifact, and
the channel is varied in the direction that would expose a channel artifact if
one existed.

## The decision rule

Read the over-calling ratio on G_299, zero-shot, against llama3:8b's 23.7500 and
against the frontier cluster's 5.3333 to 6.2500.

### Outcome A — capability orders the quantity within a vendor

> **Llama 3 70B's G_299 zero-shot ratio falls strictly between 6.2500 and
> 15.4167**, the gap C67 left unsampled.

Then the ordering survives holding vendor constant and varying channel, the
capability reading is supported by a within-family comparison, and Chapter 4 may
say so.

### Outcome B — the ordering is a vendor or channel artifact

> **The ratio is at or above 15.4167**, that is, indistinguishable from the 8B
> cluster despite an order of magnitude more parameters.

Then capability does not order this quantity within a vendor, the C67 ordering is
attributable to vendor or channel rather than capability, and every sentence
asserting that over-calling attenuates with capability comes out of Chapters 1, 4,
5 and 6. What survives is the two-cluster separation and nothing more.

### Outcome C — the frontier cluster is a channel effect

> **The ratio is at or below 6.2500**, that is, a hosted 70B behaves like the
> hosted frontier models rather than like its own 8B sibling.

Then the separation tracks the serving channel rather than capability, which is a
more interesting finding than the one this thesis currently reports, and it
replaces the capability claim rather than supporting it.

## What this run still cannot settle

Prompt sensitivity. The withdrawn GPT-4o rows put the within-model prompt effect
at 1.99x, larger than the entire across-model span at 8B. Both prompt conditions
are run here for that reason, but two conditions bound it rather than measure it.

The 8B arm is local and the 70B arm is hosted, so channel is varied, not
controlled. A within-channel capability test would need both tiers on Bedrock,
which is possible and is named as an obligation on any continuation rather than
done here.

## Cost and stopping

600 calls across both conditions. The synthetic check returned 4 generated tokens
against a 16-token cap, so the run is expected to cost a few dollars. It is run
once. If it fails partway it is restarted from the beginning and that is
recorded; partial runs are not spliced.

---

## OUTCOME, recorded 2026-09-11 after both conditions closed

**Recorded in the convention of C44: the rule above was not edited, and this
section was written after the numbers landed.**

Both conditions ran once, to completion, with no restart and no splice: 300 pairs
each, `0` unparseable draws and `0` pairs with no usable draw in either. Scored by
`accord/differential/frontier_metrics.py` with
`ACCORD_FRONTIER_PROVIDER=bedrock`, to
`results/accord/differential/frontier_bedrock_metrics_metallama370binstructv1.json`.

### The decision quantity

> G_299, zero-shot over-calling ratio: **10.5000**

That is strictly between 6.2500 and 15.4167. **Outcome A.** The ordering survives
holding vendor and model family constant while varying the serving channel, so it
is not attributable to vendor, and Chapter 4 may say that over-calling attenuates
with capability within a family.

The within-family step is the point of the test. On G_299 zero-shot:

| labeller | channel | G_299 zero-shot ratio |
| --- | --- | --- |
| llama3:8b | local, Ollama | 23.7500 |
| meta.llama3-70b-instruct-v1:0 | AWS Bedrock | 10.5000 |
| Claude Haiku 4.5 | Anthropic API | 6.2500 |
| Claude Sonnet 5 | Anthropic API | 5.3333 |

The 8B to 70B step within one family and one maker is 2.2619x, against the
2.4667x step C67 measured across families and vendors. The two are close enough
that vendor is not carrying the earlier result.

### What the few-shot arm does to the reading, which Outcome A does not cover

The rule reads zero-shot only, and it was right to fix that in advance, but the
few-shot arm has to be reported because it is the more awkward number. The 70B's
few-shot G_299 ratio is **6.2500**, which is not merely inside the frontier band,
it is exactly Haiku's zero-shot value and above Sonnet's. So a prompt change moves
this model from 10.5000 to 6.2500, a factor of 1.6800, which is a large fraction
of the whole capability span it was meant to probe. The pre-registration named
prompt sensitivity as the thing this run could not settle, and the result confirms
that warning rather than escaping it: capability orders the quantity, and prompt
moves it about as far, so neither factor can be reported without the other.

### The ambiguity gradient, across every labeller that has been asked

This was not part of the decision rule and is recorded because the run measured
it. Fisher one-sided, ambiguous against clear negatives, all six tests that now
exist:

| labeller | prompt | odds ratio | p |
| --- | --- | --- | --- |
| Sonnet 5 | few-shot | 3.0500 | 0.0029 |
| Haiku 4.5 | few-shot | 2.9046 | 0.0019 |
| Sonnet 5 | zero-shot | 2.7600 | 0.0046 |
| Haiku 4.5 | zero-shot | 1.7587 | 0.0790 |
| Llama 3 70B | few-shot | 1.6756 | 0.0996 |
| Llama 3 70B | zero-shot | 1.2257 | 0.3190 |

The reading is mixed and has to be reported as mixed. The gradient does replicate:
Haiku's few-shot odds ratio of 2.9046 at p = 0.0019 is close to Sonnet's 3.0500 and
is the smallest p value of the six. So the gradient is not a single model's quirk.
But it weakens monotonically with capability within each prompt condition, and on
the Llama 70B neither condition reaches significance. Two factors move it and both
have to be named: the same model shows a stronger gradient under few-shot than
zero-shot in all three cases, and at any fixed prompt the odds ratio falls as the
labeller gets weaker.

That still cuts against the item-property reading Section~4 leans on. If the
pharmacist's ambiguous category tracked evidential difficulty, a weaker labeller
should find those pairs at least as hard, and the gradient should not fade with
capability. It fades. What the six tests support is that the more capable labellers
concentrate their false positives on the ambiguous pairs and the weaker one does
not, which is a statement about labellers, not about items, and Chapter 4's claim
from it, already only correlational, has to be scoped that way.

It also moves the multiplicity arithmetic, which is the first time the register
built for C68 has been exercised by a new result. Reporting these four further
Fisher tests, two for Haiku and two for the 70B, takes the family from sixteen to
twenty and the Bonferroni threshold from 0.0031 to 0.0025. At that threshold two
tests clear: the anisotropy Welch test, which Section RQ3 already declines to read
as support for the mechanism, and Haiku's few-shot gradient at 0.0019. Sonnet's
few-shot gradient at 0.0029 clears sixteen and does not clear twenty, and Sonnet's
zero-shot gradient at 0.0046 clears neither. So the surviving evidence for the
gradient is one test on the model the thesis does not lead with, and the headline
figure does not survive correction over a family that includes its own replication
attempts. Recorded as C70.
