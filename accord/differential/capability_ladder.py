"""The over-calling ladder across LOCAL labellers, and the ratios read from it.

WHY THIS EXISTS
---------------
Section sec:r-capability compares labellers on one denominator. Those
comparisons are quotients of numbers that live in several different
artifacts, which is exactly the shape of number that gets typed once and then
never moves again. They are computed here instead, and the register keys are
written from this artifact.

LOCAL ONLY, SINCE THE HOSTED WITHDRAWAL
-----------------------------------------
Every hosted row (Claude Sonnet 5, Claude Haiku 4.5, the two Bedrock Llama 3
rows) and every quantity computed FROM one (within_family_step_8b_to_70b,
within_channel_step_8b_to_70b, channel_effect_at_8b_local_over_hosted,
across_family_step_c67, prompt_effect_within_70b, llama70b_few_shot_ratio,
and the channel_effect_confound block) were withdrawn wholesale from
VERIFIED_FACTS.json by `accord/stats/withdraw_hosted.py` and now live under
withdrawn.hosted.capability.*. Their source artifacts
(frontier_anthropic_metrics.json, frontier_anthropic_metrics_*.json,
frontier_bedrock_metrics_*.json) are no longer even on disk: this script
cannot read them and does not try to. Nothing here recomputes a hosted
quantity; the withdrawn values already say what they were and why.

The C94 withdrawal is a SEPARATE, additive pair of keys
(withdrawn.capability_ladder_local_llama38b_ratio_llama_baseline_c94 and
withdrawn.capability_channel_effect_at_8b_local_over_hosted_llama_baseline_c94)
recording the pre-C94 llama_baseline-prompt local ratio and the confounded
channel effect it produced. This script does NOT rewrite either: they are
historical, additive, and the hosted withdrawal above already removed the
hosted quantity (channel_effect_at_8b_local_over_hosted) that the second one
was a superseded version of. Recomputing them here would need the hosted 8B
ratio this script no longer reads.

Every LOCAL row is read through ratio() against its own frontier-template
metrics artifact, exactly as under C94: one prompt condition (frontier
zero-shot, or few-shot where asked for explicitly), one denominator (G_299),
across every row. A content assertion on each entry checks the artifact
actually is the (model, think) pair the entry claims, so a filename that
outran its own content cannot be read as something it is not.

Writes: results/accord/differential/capability_ladder.json
        and the capability.* block of results/accord/VERIFIED_FACTS.json
        (additive withdrawn.capability_*_c94 keys are never touched here)

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

# (label, think, path). think is checked against the artifact's own top-level
# "think" field (None for the pre-C94/C94-era llama3:8b and granite3.3:8b
# artifacts, which predate this project recording think at all; False for the
# C99 qwen3:8b/qwen3:14b reruns, which record it explicitly). "Local" is the
# only channel left in this file since the hosted withdrawal, so it is no
# longer a per-row field.
LOCAL = [
    ("llama3:8b", None,
     "results/accord/differential/frontier_ollama_metrics_llama3.json"),
    ("granite3.3:8b", None,
     "results/accord/differential/frontier_ollama_metrics_granite33.json"),
    ("qwen3:8b", False,
     "results/accord/differential/frontier_ollama_metrics_qwen3_8b_nothink.json"),
    ("qwen3:14b", False,
     "results/accord/differential/frontier_ollama_metrics_qwen3_14b_nothink.json"),
]
# The three 8B models, used for the over-calling spread across one capability
# tier, and the label of the within-family qwen3 scale step.
LOCAL_8B_LABELS = ("llama3:8b", "granite3.3:8b", "qwen3:8b")


def content(path):
    return json.load(open(path))


def ratio(art, strat=STRAT):
    return art["by_strategy"][strat][BASIS]["over_calling_ratio"]


def main():
    missing = [label for label, _think, path in LOCAL
              if not pathlib.Path(path).exists()]
    present = [(label, think, path) for label, think, path in LOCAL
              if pathlib.Path(path).exists()]

    rows = []
    for label, think, path in present:
        art = content(path)
        # Content assertion: the artifact opened at this path really is the
        # (model, think) pair this entry claims, not a filename that outran
        # its own content. Model is read per-strategy (frontier_metrics.py has
        # always copied raw["model"] into by_strategy[strat]["model"], for
        # every version of that script); think is read at the artifact's top
        # level, which only the current frontier_metrics.py writes, and is
        # therefore absent (None) on the pre-C99 llama3:8b/granite3.3:8b
        # artifacts this file still reads, matching their think=None entries.
        art_model = art.get("by_strategy", {}).get(STRAT, {}).get("model")
        if art_model != label:
            raise SystemExit(f"{path}: model {art_model!r} != "
                             f"expected {label!r}")
        if art.get("think") != think:
            raise SystemExit(f"{path}: think {art.get('think')!r} != "
                             f"expected {think!r}")
        rows.append({"labeller": label, "think": think, "channel": "Local",
                     "ratio": ratio(art), "source": path})
    rows.sort(key=lambda r: -r["ratio"])

    ratio_by_label = {r["labeller"]: r["ratio"] for r in rows}

    spread_8b = None
    if all(l in ratio_by_label for l in LOCAL_8B_LABELS):
        vals = [ratio_by_label[l] for l in LOCAL_8B_LABELS]
        spread_8b = max(vals) / min(vals)

    step_qwen3_8b_to_14b = None
    if "qwen3:8b" in ratio_by_label and "qwen3:14b" in ratio_by_label:
        step_qwen3_8b_to_14b = (ratio_by_label["qwen3:8b"]
                                / ratio_by_label["qwen3:14b"])

    out = {
        "provenance": {
            "generated_by": "accord/differential/capability_ladder.py",
            "basis": BASIS, "strategy": STRAT,
            "note": ("Local only. Every hosted row and every quantity "
                     "computed from one was withdrawn wholesale by "
                     "accord/stats/withdraw_hosted.py and lives under "
                     "withdrawn.hosted.capability.*; their source artifacts "
                     "are no longer on disk and this script does not read "
                     "them. The two withdrawn.capability_*_c94 keys "
                     "(the pre-C94 llama_baseline-prompt local ratio and the "
                     "confounded channel effect it produced) are historical "
                     "and additive; this script does not rewrite them."),
        },
        "local_runs_not_completed": missing,
        "ladder": rows,
        "spread_8b_frontier_zero_shot_g299": spread_8b,
        "spread_8b_labels": list(LOCAL_8B_LABELS),
        "within_family_step_qwen3_8b_to_14b": step_qwen3_8b_to_14b,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)

    facts = json.load(open(FACTS), object_pairs_hook=collections.OrderedDict)
    # Additive-only: this deletes ONLY this script's own capability.* prefix,
    # never withdrawn.* (checked explicitly, not just by prefix match, so a
    # future rename of PREFIX cannot accidentally start eating withdrawn.*
    # the way CLAUDE.md's labelnoise note warns against).
    for k in [k for k in facts
             if k.split(".")[0] == PREFIX and not k.startswith("withdrawn.")]:
        del facts[k]

    def put(key, value):
        facts[f"{PREFIX}.{key}"] = collections.OrderedDict(
            [("artifact", OUT), ("value", value)])

    if spread_8b is not None:
        put("spread_8b_frontier_zero_shot_g299", spread_8b)
    if step_qwen3_8b_to_14b is not None:
        put("within_family_step_qwen3_8b_to_14b", step_qwen3_8b_to_14b)
    for r in rows:
        slug = (r["labeller"].replace(":", "").replace(".", "")
                .replace(" ", "_").replace("-", "").lower())
        put(f"ladder.{slug}.ratio", r["ratio"])

    json.dump(facts, open(FACTS, "w"), indent=1)

    print(f"wrote {OUT}")
    for r in rows:
        print(f"  {r['ratio']:>8.4f}  {r['labeller']:<16} think={r['think']!r:<6} "
              f"{r['channel']}")
    if missing:
        print(f"  missing (not completed): {missing}")
    if spread_8b is not None:
        print(f"\n  spread across the three local 8B models (G_299, "
              f"frontier zero-shot): {spread_8b:.4f}x")
    else:
        print(f"\n  spread across the three local 8B models: not computed "
              f"({[l for l in LOCAL_8B_LABELS if l not in ratio_by_label]} "
              f"missing)")
    if step_qwen3_8b_to_14b is not None:
        print(f"  within-family step, qwen3 8B to 14B: "
              f"{step_qwen3_8b_to_14b:.4f}x")
    else:
        print(f"  within-family step, qwen3 8B to 14B: not computed "
              f"(qwen3:14b not yet run)")


if __name__ == "__main__":
    main()
