#!/usr/bin/env python3
"""Register scalar, print-precision forms of quantities the chapters print.

Two cases need this and neither is a new measurement.

1. Interval ends stored as a two-element list. The provenance gate matches
   scalars, so an interval printed as two numbers in prose has no scalar key to
   match even though the pair is registered.

2. p values whose fifth decimal is exactly 5. The project's convention is round
   half up, but `f"{v:.4f}"` is half to even and returns the other neighbour, so
   the gate generates a form the convention rejects. Registering the half-up
   form makes the printed digit traceable to a generator rather than to a hand
   correction.

    python3 accord/stats/register_printed_forms.py
"""
import collections, json, pathlib, sys
from decimal import Decimal, ROUND_HALF_UP

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accord.cloud_paths import assert_local_only  # noqa: E402

FACTS = assert_local_only(ROOT / "results/accord/VERIFIED_FACTS.json")
PREFIX = "printedform"

# (new key suffix, source key, index into a list source or None)
INTERVALS = [
    ("labellerintervals.promptconfirm.granite33_8b.P_rev_agree.g253.bootstrap95_auroc", "lo", 0),
    ("labellerintervals.promptconfirm.granite33_8b.P_rev_agree.g253.bootstrap95_auroc", "hi", 1),
]
# p values printed at four decimals under round half up
HALF_UP = [
    "disclosed_superseded.c114.task_a.nli_minus_detector_ensemble.p_two_sided",
    "c116.t1_tfidf_minus_detector.p_permutation_two_sided",
]

facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]

n = 0
for src, end, idx in INTERVALS:
    pair = facts[src]["value"]
    facts[f"{PREFIX}.{src}_{end}"] = collections.OrderedDict(
        [("value", round(pair[idx], 4)),
         ("artifact", f"{facts[src]['artifact']}, derived: element {idx} at print precision")])
    n += 1

for src in HALF_UP:
    v = facts[src]["value"]
    hu = float(Decimal(repr(v)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
    facts[f"{PREFIX}.{src}_4dp_half_up"] = collections.OrderedDict(
        [("value", hu),
         ("artifact", f"{facts[src]['artifact']}, derived: rounded to four decimals, half up")])
    n += 1

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: {n} keys")
for k in facts:
    if k.startswith(PREFIX):
        print(f"  {facts[k]['value']}  <- {k}")
