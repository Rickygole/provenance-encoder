"""The over-calling ladder across labellers, and the two ratios read from it.

WHY THIS EXISTS
---------------
Section sec:r-capability compares four labellers on one denominator and quotes
three derived ratios: the within-family step from llama3:8b to the Bedrock 70B,
the across-family step C67 measured, and the prompt effect within the 70B. Those
are quotients of numbers that live in four different artifacts, which is exactly
the shape of number that gets typed once and then never moves again. They are
computed here instead, and the register keys are written from this artifact.

The local 8B and the public-text ceiling are read from the committed sweep; the
three hosted labellers from their own metrics artifacts, each of which the runner
and the metrics script keep under a separate model tag.

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
    rows.append({"labeller": LOCAL_8B, "channel": "Local", "ratio": local,
                 "source": f"{FACTS}::{local_key}"})
    rows.sort(key=lambda r: -r["ratio"])

    b70 = next(r["ratio"] for r in rows if r["labeller"].startswith("meta.llama3-70b"))
    # The across-family step C67 reported, for the comparison the within-family
    # step is read against: the lowest of the three local 8B labellers against
    # the highest of the hosted cluster. Recomputed from the same two registered
    # ratios rather than carried over as a typed figure.
    step_8b_min = float(facts["overcall.qwen3_8b.gold.ratio"]["value"])
    haiku = next(r["ratio"] for r in rows if r["labeller"].startswith("Claude Haiku"))
    b70_fs = ratio(HOSTED[2][1], "few_shot")

    out = {
        "provenance": {
            "generated_by": "accord/differential/capability_ladder.py",
            "basis": BASIS, "strategy": STRAT,
            "note": ("One denominator and one prompt condition across all four "
                     "rows. The ratios below are quotients of the ladder, "
                     "computed here so that none of them is typed into prose."),
        },
        "ladder": rows,
        "within_family_step_8b_to_70b": local / b70,
        "across_family_step_c67": step_8b_min / haiku,
        "prompt_effect_within_70b": b70 / b70_fs,
        "llama70b_few_shot_ratio": b70_fs,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)

    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]
    for key in ("within_family_step_8b_to_70b", "across_family_step_c67",
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
    print(f"  prompt effect within the 70B : {out['prompt_effect_within_70b']:.4f}"
          f"  ({b70:.4f} zero-shot to {b70_fs:.4f} few-shot)")


if __name__ == "__main__":
    main()
