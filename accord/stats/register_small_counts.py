#!/usr/bin/env python3
"""Register small counts the thesis prints that had no register entry of their own.

The numeric gate matches a printed number against any register value, so a
single digit passes whether or not the thing it counts is registered. An audit
listed the small counts that were being matched to unrelated keys. Each one
below is read from the committed artifact that holds it, so the count can be
traced to its real source and not to a coincidence.
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "smallcounts"


def main():
    facts = json.loads(FACTS.read_text(),
                       object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    report = "results/accord/control/final_report.json"
    per_seed = json.loads((ROOT / report).read_text())["top15_tp_per_seed"]
    for stratum, values in per_seed.items():
        for i, v in enumerate(values):
            facts[f"{PREFIX}.control.top15_tp_per_seed.{stratum}.seed{i}"] = {
                "artifact": f"{report}::top15_tp_per_seed.{stratum}[{i}]", "value": v}

    inj = "results/accord/injection_sensitivity.json"
    reps = json.loads((ROOT / inj).read_text())["n_reps_per_grid_point"]
    facts[f"{PREFIX}.injection.n_reps_per_grid_point"] = {
        "artifact": f"{inj}::n_reps_per_grid_point", "value": reps}

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"   registered {sum(1 for k in facts if k.startswith(PREFIX + '.'))} small counts")


if __name__ == "__main__":
    main()
