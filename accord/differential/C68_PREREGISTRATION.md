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
