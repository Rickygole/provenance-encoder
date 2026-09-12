"""The over-calling ladder across labellers, and the two ratios read from it.

WHY THIS EXISTS
---------------
Section sec:r-capability compares five labellers on one denominator and quotes
three derived ratios: the within-family step from llama3:8b to the Bedrock 70B,
the across-family step C67 measured, and the prompt effect within the 70B. Those
are quotients of numbers that live in four different artifacts, which is exactly
the shape of number that gets typed once and then never moves again. They are
computed here instead, and the register keys are written from this artifact.

The local 8B and the public-text ceiling are read from the committed sweep; the
three hosted labellers from their own metrics artifacts, each of which the runner
and the metrics script keep under a separate model tag.

CAVEAT ON ROW 1, READ BEFORE QUOTING "one prompt condition"
-------------------------------------------------------------
Four of the five rows (the three hosted labellers plus the hosted 8B added by
C72) are read through ratio() against the frontier zero-shot template, so those
four really do sit on one denominator and one prompt. The fifth, llama3:8b run
locally, is not: its ratio comes from results/accord/multillm/per_model_results.json,
whose own prompt_source field records config_id=llama_baseline, a different
prompt than the frontier template. This was found by checking that field rather
than assumed, and it is not a small print difference: this same script measures
a prompt effect within the 70B (zero-shot against few-shot, still one model, one
channel) of about 1.68x, larger than the 1.10x channel effect at 8B this ladder
also reports, which is computed by comparing the local llama3:8b row against the
hosted Bedrock 8B row, i.e. exactly the two rows that differ in prompt as well as
in channel. The channel effect above is therefore an upper bound confounded with
an uncontrolled prompt difference, not a clean channel comparison, and the local
row is kept in the ladder rather than dropped, because deleting the discrepant
row would hide the very confound this note exists to disclose.

Writes: results/accord/differential/capability_ladder.json
        and the capability.* block of results/accord/VERIFIED_FACTS.json

    python3 accord/differential/capability_ladder.py
"""

import collections
import json
import os

BASIS = "G_299"
STRAT = "zero_shot"
FACTS = "results/accord/VERIFIED_FACTS.json"
OUT = "results/accord/differential/capability_ladder.json"
PREFIX = "capability"

# label -> (artifact, channel). One prompt condition and one denominator, so the
# rows are comparable; mixing either is the defect tab:denoms exists to prevent.
HOSTED = [
    ("Claude Sonnet 5", "results/accord/differential/frontier_anthropic_metrics.json",
     "Hosted API"),
    ("Claude Haiku 4.5",
     "results/accord/differential/frontier_anthropic_metrics_claudehaiku4520251001.json",
     "Hosted API"),
    ("meta.llama3-70b-instruct-v1:0",
     "results/accord/differential/frontier_bedrock_metrics_metallama370binstructv1.json",
     "AWS Bedrock"),
    ("meta.llama3-8b-instruct-v1:0",
     "results/accord/differential/frontier_bedrock_metrics_metallama38binstructv1.json",
     "AWS Bedrock"),
]
SWEEP = "results/accord/prevalence_replication/sweep_results.json"
LOCAL_8B = "llama3:8b"


def ratio(path, strat=STRAT):
    return json.load(open(path))["by_strategy"][strat][BASIS]["over_calling_ratio"]


def main():
    rows = [{"labeller": name, "channel": chan, "ratio": ratio(path),
             "source": path} for name, path, chan in HOSTED]

    # The local 8B arm. Its ratio is already a registered fact of the sweep, so
    # it is read from the register rather than recomputed under a second
    # convention: two computations of one quantity is how two numbers for it end
    # up in one thesis.
    facts = json.load(open(FACTS), object_pairs_hook=collections.OrderedDict)
    local_key = "overcall.llama3_8b.gold.ratio"
    local = float(facts[local_key]["value"])
    # This row's prompt is read from its own source artifact rather than assumed
    # to match the other four: it does not. See the module docstring's caveat.
    local_prompt_source = json.load(open(facts[local_key]["artifact"]))["prompt_source"]
    rows.append({"labeller": LOCAL_8B, "channel": "Local", "ratio": local,
                 "source": f"{FACTS}::{local_key}", "prompt_source": local_prompt_source})
    rows.sort(key=lambda r: -r["ratio"])

    b70 = next(r["ratio"] for r in rows if r["labeller"].startswith("meta.llama3-70b"))
    # C72 added the hosted 8B, which turns two confounded factors into two
    # separately measurable ones: the step between tiers ON one channel, and the
    # step between channels AT one tier.
    b8 = next(r["ratio"] for r in rows if r["labeller"].startswith("meta.llama3-8b"))
    # The across-family step C67 reported, for the comparison the within-family
    # step is read against: the lowest of the three local 8B labellers against
    # the highest of the hosted cluster. Recomputed from the same two registered
    # ratios rather than carried over as a typed figure.
    step_8b_min = float(facts["overcall.qwen3_8b.gold.ratio"]["value"])
    haiku = next(r["ratio"] for r in rows if r["labeller"].startswith("Claude Haiku"))
    b70_fs = ratio(HOSTED[2][1], "few_shot")

    # Computed once, ahead of the artifact dict, so the confound note below can
    # quote the two figures it is about rather than restate them by hand.
    channel_effect_at_8b = local / b8
    prompt_effect_within_70b = b70 / b70_fs

    out = {
        "provenance": {
            "generated_by": "accord/differential/capability_ladder.py",
            "basis": BASIS, "strategy": STRAT,
            # Four of the five rows share one prompt condition. The fifth is the
            # Local llama3:8b row, whose own artifact records prompt_source
            # "prompts_existing.csv, config_id=llama_baseline", not the frontier
            # zero-shot template the other four use; see each row's own
            # "prompt_source" field for the hosted/local split. This is not a
            # footnote: the prompt effect measured within the 70B below is
            # larger than the channel effect at 8B, and the channel effect is
            # computed from the one pair of rows (Local and hosted 8B) where
            # prompt also differs, so it is an upper bound, not a clean
            # channel-only comparison.
            "note": ("One denominator across all five rows. Four of the five "
                     "share one prompt condition, the frontier zero-shot "
                     "template; the fifth (Local llama3:8b) does not, and its "
                     "own prompt_source is recorded on its row. The ratios "
                     "below are quotients of the ladder, computed here so that "
                     "none of them is typed into prose."),
        },
        "ladder": rows,
        "within_family_step_8b_to_70b": local / b70,
        "within_channel_step_8b_to_70b": b8 / b70,
        "channel_effect_at_8b_local_over_hosted": channel_effect_at_8b,
        "across_family_step_c67": step_8b_min / haiku,
        "prompt_effect_within_70b": prompt_effect_within_70b,
        "llama70b_few_shot_ratio": b70_fs,
        # Written into the artifact, not only the docstring, so any script or
        # sentence that loads this file and quotes channel_effect_at_8b_local_
        # over_hosted inherits the caveat rather than having to know to look
        # for it. local vs b8 is exactly the Local-row-vs-hosted-8B-row pair
        # that differs in prompt as well as in channel (see each row's
        # prompt_source), so this ratio bundles both effects together.
        "channel_effect_confound": {
            "confounded_with_prompt": True,
            "reason": ("channel_effect_at_8b_local_over_hosted divides the "
                       "Local llama3:8b row by the hosted Bedrock 8B row, and "
                       "those two rows differ in prompt_source as well as in "
                       "channel."),
            "prompt_effect_within_70b_same_channel": prompt_effect_within_70b,
            "channel_effect_at_8b": channel_effect_at_8b,
            "prompt_effect_exceeds_channel_effect": prompt_effect_within_70b > channel_effect_at_8b,
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)

    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]
    for key in ("within_family_step_8b_to_70b", "within_channel_step_8b_to_70b",
                "channel_effect_at_8b_local_over_hosted", "across_family_step_c67",
                "prompt_effect_within_70b", "llama70b_few_shot_ratio"):
        facts[f"{PREFIX}.{key}"] = collections.OrderedDict(
            [("artifact", OUT), ("value", out[key])])
    for r in rows:
        slug = (r["labeller"].replace(":", "").replace(".", "")
                .replace(" ", "_").replace("-", "").lower())
        facts[f"{PREFIX}.ladder.{slug}.ratio"] = collections.OrderedDict(
            [("artifact", OUT), ("value", r["ratio"])])
    json.dump(facts, open(FACTS, "w"), indent=1)

    print(f"wrote {OUT}")
    for r in rows:
        print(f"  {r['ratio']:>8.4f}  {r['labeller']:<34} {r['channel']}")
    print(f"\n  within-family 8B to 70B step : {out['within_family_step_8b_to_70b']:.4f}")
    print(f"  across-family step (C67)     : {out['across_family_step_c67']:.4f}")
    print(f"  WITHIN-CHANNEL 8B to 70B step: {out['within_channel_step_8b_to_70b']:.4f}"
          f"  (both on Bedrock)")
    print(f"  CHANNEL effect at 8B         : "
          f"{out['channel_effect_at_8b_local_over_hosted']:.4f}"
          f"  (local {local:.4f} vs hosted {b8:.4f})")
    print(f"  prompt effect within the 70B : {out['prompt_effect_within_70b']:.4f}"
          f"  ({b70:.4f} zero-shot to {b70_fs:.4f} few-shot)")


if __name__ == "__main__":
    main()
