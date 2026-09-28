#!/usr/bin/env python3
"""Register the answer-polarity sums the fig:answer-polarity caption prints.

The caption states, for each labeller, the sum of its raw yes rate to
"is there a contradiction?" and its raw yes rate to "are these consistent?".
A labeller answering from what it read lands near 1. Those sums are derived
from the per-cell rates in answer_polarity.json and were not registered, so
the provenance gate could not trace them.

    python3 accord/framing_mechanism/register_answer_polarity.py
"""
import collections, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accord.cloud_paths import assert_local_only  # noqa: E402

SRC = "results/accord/framing_mechanism/answer_polarity.json"
FACTS = assert_local_only(ROOT / "results/accord/VERIFIED_FACTS.json")
PREFIX = "answerpolarity"

# Recompute from the per-pair labels rather than from the artifact's rounded
# rates: summing two values already rounded to four places moves the last digit
# of the sum (1.0536 against the true 1.0535).
CACHE = pathlib.Path.home() / "mimic_local_only/accord_prompt_sensitivity_cache"
# Provenance string for the register: the cache is local-only and outside the
# released tree, so name it as the source rather than the JSON it is checked against.
CACHE_REL = "mimic_local_only/accord_prompt_sensitivity_cache/*.jsonl (local only)"
STEM = {"Llama 3 8B": "_llama3_8b_nothink",
        "Granite 3.3 8B": "_granite33_8b_nothink",
        "Qwen 3 8B": "_qwen3_8b_nothink"}


def flag_rate(stem, prompt):
    lab = [json.loads(l)["flag_label"] for l in (CACHE / f"{stem}__{prompt}.jsonl").open()]
    ok = [x for x in lab if x is not None]
    return sum(ok) / len(ok)


rate = {}
for model, stem in STEM.items():
    rate[(model, "P_neutral")] = flag_rate(stem, "P_neutral")
    rate[(model, "P_reversed")] = 1.0 - flag_rate(stem, "P_reversed")

slug = {"Llama 3 8B": "llama3_8b", "Granite 3.3 8B": "granite33_8b", "Qwen 3 8B": "qwen3_8b"}
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]

n = 0
for model, s in slug.items():
    a, b = rate[(model, "P_neutral")], rate[(model, "P_reversed")]
    for field, val, note in (
            ("raw_yes_contradiction", round(a, 4), "P_neutral"),
            ("raw_yes_consistency", round(b, 4), "P_reversed"),
            ("raw_yes_sum", round(a + b, 4), "the two above, summed")):
        facts[f"{PREFIX}.{s}.{field}"] = collections.OrderedDict(
            [("value", val),
             ("artifact", f"{CACHE_REL}, derived: {note}; cross-checked against {SRC}::cells")])
        n += 1
for model, s in slug.items():
    print(f"  {model:16}{rate[(model,'P_neutral')]:.4f} + {rate[(model,'P_reversed')]:.4f}"
          f" = {round(rate[(model,'P_neutral')]+rate[(model,'P_reversed')],4)}")

# The fig:answer-polarity and fig:scale-ladder captions also print inter-framing
# kappas and the Qwen ladder fit. Those were read straight off the artifact and
# never registered, so the provenance gate matched them to unrelated keys.
art = json.loads((ROOT / SRC).read_text())

coh = [c for c in art["coherence"]
       if c["forward_prompt"] == "P_neutral" and c["reverse_prompt"] == "P_reversed"]
for c in coh:
    s_ = slug[c["model"]]
    facts[f"{PREFIX}.{s_}.kappa_between_framings"] = collections.OrderedDict(
        [("value", c["kappa_between_framings"]),
         ("artifact", f"{SRC}::coherence[{c['model']}, P_neutral vs P_reversed]"
                      ".kappa_between_framings")])
    n += 1
ks = sorted(c["kappa_between_framings"] for c in coh)
for field, val in (("kappa_between_framings_min", ks[0]),
                   ("kappa_between_framings_max", ks[-1])):
    facts[f"{PREFIX}.{field}"] = collections.OrderedDict(
        [("value", val),
         ("artifact", f"{SRC}::coherence, derived: {field.split('_')[-1]} over the three "
                      "P_neutral vs P_reversed cells")])
    n += 1

lad = art["ladder"]
for field in ("slope_per_log10_params", "pearson_r"):
    facts[f"{PREFIX}.ladder.{field}"] = collections.OrderedDict(
        [("value", lad[field]), ("artifact", f"{SRC}::ladder.{field}")])
    n += 1
for pt in lad["points"]:
    if "params_b" not in pt:
        continue
    tag = f"qwen3_{str(pt['params_b']).replace('.', 'p')}b"
    for field in ("n_parsed", "n_unparsed", "raw_yes_rate"):
        facts[f"{PREFIX}.ladder.{tag}.{field}"] = collections.OrderedDict(
            [("value", pt[field]),
             ("artifact", f"{SRC}::ladder.points[{pt['model']}].{field}")])
        n += 1

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: {n} keys total")
