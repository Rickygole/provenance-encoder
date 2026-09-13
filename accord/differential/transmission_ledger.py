"""How many prompts each compliant run actually sent, counted from its artifacts.

WHY THIS EXISTS
---------------
Section 3.9 enumerates every transmission of sentence pairs to a hosted API. For
a day it enumerated one of three, because it was written when there was one and
the experiments moved without it. A disclosure section that lags the experiments
is worse than no table, so the count is computed here from the committed
prediction artifacts rather than asserted in prose: a run that leaves an artifact
in this directory appears in the ledger, and a run that does not is not disclosed
by accident either.

The count is prompts, not pairs: each run covers the annotated pairs under two
prompt conditions, and the disclosed quantity is the number of requests that
carried a sentence pair off this machine.

Reads:  results/accord/differential/frontier_<provider><tag>_<strategy>.json
Writes: results/accord/differential/transmission_ledger.json
        and the transmission.* block of results/accord/VERIFIED_FACTS.json

    python3 accord/differential/transmission_ledger.py
"""

import collections
import glob
import json
import os
import re

FACTS = "results/accord/VERIFIED_FACTS.json"
OUT = "results/accord/differential/transmission_ledger.json"
PREFIX = "transmission"
# Every key under PREFIX is deleted before the rewrite, so a value that must
# survive but is NOT produced here cannot live under this prefix. A withdrawn
# total was parked at transmission.total_prompts_superseded_c71 and vanished the
# next time this ran, taking a number Chapter 3 quotes with it. Withdrawn and
# superseded values belong under withdrawn.*, which no generator owns.
# The trailing wildcard matters: replicate artifacts end _zero_shot_rep3,
# and the first version of this glob was frontier_*_shot.json, which
# silently excluded them. A ledger that misses a run is the defect it was
# written to fix, so the pattern is deliberately loose and the regex below
# does the discrimination.
PATTERN = "results/accord/differential/frontier_*_shot*.json"


def _per_channel(runs):
    """Prompts per channel.

    The 2025 guidance judges a route by whether the service retains, and the two
    routes used here fall on opposite sides of that test, so the total is no
    longer the quantity the compliance argument needs. The per-channel split is
    computed from the same artifacts the total is, and never typed.
    """
    out = {}
    for r in runs:
        out[r["channel"]] = out.get(r["channel"], 0) + r["prompts"]
    return out


def main():
    runs = collections.defaultdict(lambda: {"prompts": 0, "conditions": [],
                                            "artifacts": []})
    for path in sorted(glob.glob(PATTERN)):
        # Replicated runs carry a _repN suffix and are additional draws over the
        # same pairs, so they are counted separately rather than folded in.
        m = re.match(r"frontier_(.+?)_(zero_shot|few_shot)(_rep\d+)?\.json$",
                     os.path.basename(path))
        if not m:
            continue
        who, strategy, rep = m.group(1), m.group(2), m.group(3) or ""
        d = json.load(open(path))
        key = who + rep
        r = runs[key]
        r["prompts"] += len(d["predictions"]) * int(d.get("replicates_per_pair", 1))
        r["conditions"].append(strategy)
        r["artifacts"].append(path)
        r["channel"] = d.get("channel")
        r["model"] = d.get("model")
        r["n_pairs"] = len(d["predictions"])
        r["replicates_per_pair"] = int(d.get("replicates_per_pair", 1))

    ledger = []
    for key in sorted(runs):
        r = dict(runs[key])
        r["conditions"] = sorted(r["conditions"])
        r["run"] = key
        ledger.append(r)

    out = {
        "provenance": {
            "generated_by": "accord/differential/transmission_ledger.py",
            "counted_from": PATTERN,
            "unit": ("prompts, that is requests that carried a sentence pair off "
                     "this machine, not pairs. Each run covers the annotated "
                     "pairs under each prompt condition it lists."),
        },
        "runs": ledger,
        "total_prompts": sum(r["prompts"] for r in ledger),
        "per_channel": _per_channel(ledger),
        "n_runs": len(ledger),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)

    facts = json.load(open(FACTS), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]
    for r in ledger:
        slug = r["run"].replace("-", "").replace(".", "").replace(":", "").lower()
        for field in ("prompts", "n_pairs", "replicates_per_pair"):
            facts[f"{PREFIX}.{slug}.{field}"] = collections.OrderedDict(
                [("artifact", OUT), ("value", r[field])])
    for field in ("total_prompts", "n_runs"):
        facts[f"{PREFIX}.{field}"] = collections.OrderedDict(
            [("artifact", OUT), ("value", out[field])])
    json.dump(facts, open(FACTS, "w"), indent=1)

    print(f"wrote {OUT}\n")
    for r in ledger:
        print(f"  {r['prompts']:>5} prompts  {r['run']:<42} "
              f"{r['channel']}  {'+'.join(r['conditions'])}")
    print(f"\n  {out['total_prompts']} prompts across {out['n_runs']} runs")


if __name__ == "__main__":
    main()
