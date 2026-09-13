"""The over-calling ladder across labellers, and the two ratios read from it.

WHY THIS EXISTS
---------------
Section sec:r-capability compares labellers on one denominator and quotes
several derived ratios: the within-family step from llama3:8b to the Bedrock
70B, the across-family step C67 measured, and the prompt effect within the
70B. Those are quotients of numbers that live in several different artifacts,
which is exactly the shape of number that gets typed once and then never moves
again. They are computed here instead, and the register keys are written from
this artifact.

Every row, hosted and local alike, is read through ratio() against its own
metrics artifact under the frontier template. There is no special case left
for any row: see the C94 note below for why there used to be one.

C94: THE LOCAL ROWS ARE NOW PROMPT-MATCHED
-------------------------------------------
Before C94, four of five rows (the three hosted labellers plus the hosted 8B
added by C72) were read through ratio() against the frontier zero-shot
template, and the fifth, llama3:8b run locally, was not: its ratio came from
results/accord/multillm/per_model_results.json under config_id=llama_baseline,
a different prompt. That confound was not small: this script measures a
prompt effect within the 70B (zero-shot against few-shot, one model, one
channel) of about 1.68x, larger than the 1.10x channel effect at 8B that the
confounded local-vs-hosted comparison produced.

C94 reran all three installed 8B models (llama3:8b, granite3.3:8b, qwen3:8b)
locally through Ollama under the same frontier zero-shot and few-shot
templates as the hosted rows, scored them with the same
accord/differential/frontier_metrics.py used for every hosted row (via
ACCORD_FRONTIER_PROVIDER=ollama and a per-model ACCORD_FRONTIER_MODEL_TAG; no
new script), and the three resulting metrics artifacts are read here through
ratio() exactly like the hosted rows, in the LOCAL list below. The register
read that used to stand in for the local row is gone.

The superseded llama_baseline-prompt local ratio and the channel effect it
produced are not deleted: they are recomputed from the still-live
overcall.llama3_8b.gold.ratio fact (untouched by this script, owned by the
per_model_results generator) and written under withdrawn.* every run, so the
old number stays in the register on record rather than silently vanishing
when this file's own capability.* prefix is rewritten. See
accord/differential/C94_PREREGISTRATION.md for the decision rule this run is
scored against, and accord/CORRECTIONS.md C94 for the outcome once recorded.

Writes: results/accord/differential/capability_ladder.json
        and the capability.* block of results/accord/VERIFIED_FACTS.json
        and two withdrawn.* keys (additive; never deletes withdrawn.* itself)

    python3 accord/differential/capability_ladder.py
"""

import collections
import json
import pathlib
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

# C94: the three installed 8B models, rerun locally through Ollama under the
# same frontier zero-shot/few-shot templates as the hosted rows, scored by the
# same accord/differential/frontier_metrics.py (no new metrics script needed;
# it already takes provider and model tag from the environment). Filenames
# follow the runner's own tagging convention in run_frontier_compliant.py
# (model label, lowercased, "-" and "." stripped, before ":"): llama3:8b ->
# _llama3, granite3.3:8b -> _granite33, qwen3:8b -> _qwen3.
LOCAL = [
    ("llama3:8b",
     "results/accord/differential/frontier_ollama_metrics_llama3.json", "Local"),
    ("granite3.3:8b",
     "results/accord/differential/frontier_ollama_metrics_granite33.json", "Local"),
    ("qwen3:8b",
     "results/accord/differential/frontier_ollama_metrics_qwen3.json", "Local"),
]


def ratio(path, strat=STRAT):
    return json.load(open(path))["by_strategy"][strat][BASIS]["over_calling_ratio"]


def main():
    # Every row, hosted and local, is read the same way: one function, one
    # prompt condition, one denominator. No special case for any row.
    # A local model whose run did not complete is OMITTED and RECORDED, not
    # substituted and not silently dropped. The machine that served these models
    # has 16GB of unified memory and was already deep in swap; one model's run
    # was killed by the memory manager twice. Reporting a ladder that quietly
    # lost a rung, or splicing a partial run into it, are both worse than a
    # ladder that says which rung is missing and why. The pre-registration
    # requires a restart from the beginning rather than a splice.
    missing = [name for name, path, _ in LOCAL if not pathlib.Path(path).exists()]
    present = [(n, p_, c) for n, p_, c in LOCAL if pathlib.Path(p_).exists()]
    rows = [{"labeller": name, "channel": chan, "ratio": ratio(path),
             "source": path} for name, path, chan in HOSTED + present]
    rows.sort(key=lambda r: -r["ratio"])

    # Still needed below for the across-family step and for reconstructing the
    # withdrawn llama_baseline-prompt channel effect; this key is owned by the
    # per_model_results generator, not by this script, and is not touched by
    # the capability.* rewrite at the end of main().
    facts = json.load(open(FACTS), object_pairs_hook=collections.OrderedDict)

    b70 = next(r["ratio"] for r in rows if r["labeller"].startswith("meta.llama3-70b"))
    # C72 added the hosted 8B, which turns two confounded factors into two
    # separately measurable ones: the step between tiers ON one channel, and the
    # step between channels AT one tier.
    b8 = next(r["ratio"] for r in rows if r["labeller"].startswith("meta.llama3-8b"))
    # The local counterpart to b8: same model family (llama3 8B) as the hosted
    # 8B row above, now under the same frontier zero-shot prompt, so this is a
    # channel-only comparison rather than the channel-and-prompt comparison it
    # was before C94.
    local = next(r["ratio"] for r in rows
                 if r["labeller"] == LOCAL_8B and r["channel"] == "Local")
    # The across-family step C67 reported, for the comparison the within-family
    # step is read against: the lowest of the three local 8B labellers against
    # the highest of the hosted cluster. Recomputed from the same two registered
    # ratios rather than carried over as a typed figure.
    step_8b_min = float(facts["overcall.qwen3_8b.gold.ratio"]["value"])
    haiku = next(r["ratio"] for r in rows if r["labeller"].startswith("Claude Haiku"))
    b70_fs = ratio(HOSTED[2][1], "few_shot")

    # Computed once, ahead of the artifact dict, so the note below can quote
    # the two figures it is about rather than restate them by hand.
    channel_effect_at_8b = local / b8
    prompt_effect_within_70b = b70 / b70_fs

    # C94: the llama_baseline-prompt local ratio and the channel effect it
    # produced, recomputed (not typed) from the still-live
    # overcall.llama3_8b.gold.ratio fact and the current hosted 8B row, so the
    # withdrawn value tracks its own inputs rather than being frozen prose.
    # This is the number the module docstring's pre-C94 caveat described as an
    # "upper bound confounded with an uncontrolled prompt difference."
    withdrawn_local_llama_baseline_ratio = float(
        facts["overcall.llama3_8b.gold.ratio"]["value"])
    withdrawn_channel_effect_at_8b_llama_baseline = (
        withdrawn_local_llama_baseline_ratio / b8)

    out = {
        "provenance": {
            "generated_by": "accord/differential/capability_ladder.py",
            "basis": BASIS, "strategy": STRAT,
            "note": ("One denominator, one prompt condition (the frontier "
                     "zero-shot template) across every row, hosted and local "
                     "alike. Before C94 the Local llama3:8b row was read from "
                     "a different pipeline under config_id=llama_baseline; "
                     "that value and the channel effect it produced are kept "
                     "under withdrawn.* in the register rather than deleted. "
                     "The ratios below are quotients of the ladder, computed "
                     "here so that none of them is typed into prose."),
        },
        "local_runs_not_completed": missing,
        "ladder": rows,
        "within_family_step_8b_to_70b": local / b70,
        "within_channel_step_8b_to_70b": b8 / b70,
        "channel_effect_at_8b_local_over_hosted": channel_effect_at_8b,
        "across_family_step_c67": step_8b_min / haiku,
        "prompt_effect_within_70b": prompt_effect_within_70b,
        "llama70b_few_shot_ratio": b70_fs,
        # C94 closed the confound the pre-C94 version of this block disclosed:
        # both rows in channel_effect_at_8b_local_over_hosted now share the
        # frontier zero-shot prompt (see each row's source in "ladder" above),
        # so this ratio is a channel-only comparison. The superseded,
        # confounded figure is not deleted; it is recomputed here and written
        # to withdrawn.* below so it stays quotable as a correction record.
        "channel_effect_confound": {
            "confounded_with_prompt": False,
            "resolved_by": "C94",
            "reason": ("Before C94, channel_effect_at_8b_local_over_hosted "
                       "divided the Local llama3:8b row (config_id="
                       "llama_baseline) by the hosted Bedrock 8B row "
                       "(frontier zero-shot), which differed in prompt as "
                       "well as in channel. Both rows now come from "
                       "ratio() against the frontier zero-shot template; "
                       "see the LOCAL and HOSTED source paths in the ladder."),
            "prompt_effect_within_70b_same_channel": prompt_effect_within_70b,
            "channel_effect_at_8b": channel_effect_at_8b,
            "prompt_effect_exceeds_channel_effect": prompt_effect_within_70b > channel_effect_at_8b,
            "withdrawn_llama_baseline_prompt_channel_effect_at_8b":
                withdrawn_channel_effect_at_8b_llama_baseline,
            "withdrawn_register_key": (
                "withdrawn.capability_channel_effect_at_8b_local_over_hosted_"
                "llama_baseline_c94"),
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

    # C94 withdrawal. Additive only: this never deletes anything under
    # withdrawn.*, unlike the "for k in facts if k.split('.')[0] == PREFIX"
    # loop above, which is why these two keys live outside PREFIX ("capability")
    # rather than inside it. See the CLAUDE.md rule that every generator deletes
    # its own prefix wholesale and would eat anything else parked there.
    facts["withdrawn.capability_ladder_local_llama38b_ratio_llama_baseline_c94"] = (
        collections.OrderedDict([
            ("artifact", "withdrawn: the Local llama3:8b over-calling ratio "
             "under config_id=llama_baseline (results/accord/multillm/"
             "per_model_results.json via the pre-C94 register key "
             "overcall.llama3_8b.gold.ratio, which itself remains live and "
             "unchanged), superseded as the ladder's local 8B row by the "
             "frontier-zero-shot-prompt-matched value at "
             "capability.ladder.llama38b.ratio once C94's Ollama rerun "
             "landed. See accord/differential/C94_PREREGISTRATION.md."),
            ("value", withdrawn_local_llama_baseline_ratio)]))
    facts["withdrawn.capability_channel_effect_at_8b_local_over_hosted_llama_baseline_c94"] = (
        collections.OrderedDict([
            ("artifact", "withdrawn: channel_effect_at_8b_local_over_hosted "
             "as computed before C94, dividing the llama_baseline-prompt "
             "Local llama3:8b ratio above by the frontier-zero-shot hosted "
             "Bedrock 8B ratio, i.e. confounded with an uncontrolled prompt "
             "difference as the pre-C94 module docstring disclosed. "
             "Superseded by the clean channel-only value at "
             "capability.channel_effect_at_8b_local_over_hosted. See "
             "accord/differential/C94_PREREGISTRATION.md."),
            ("value", withdrawn_channel_effect_at_8b_llama_baseline)]))

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
    print(f"\n  [withdrawn, C94] llama_baseline-prompt local ratio  : "
          f"{withdrawn_local_llama_baseline_ratio:.4f}")
    print(f"  [withdrawn, C94] llama_baseline-prompt channel effect: "
          f"{withdrawn_channel_effect_at_8b_llama_baseline:.4f}")


if __name__ == "__main__":
    main()
