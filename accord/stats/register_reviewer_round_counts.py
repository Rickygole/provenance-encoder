#!/usr/bin/env python3
"""Register the Wilson intervals behind the gate validation's two perfect cells.

Wilson 95 percent intervals on the gate validation's perfect cells. Chapter 4
   reports specificity and positive predictive value of 1.0000 for the marginal
   feasibility gate. Both rest on single-digit denominators: 5 condemned
   labellers for the positive predictive value and 7 useful students for the
   specificity. A reviewer asked for the interval rather than the point, so the
   interval is computed here from the counts the gate validation already stores.

Run:  python3 accord/stats/register_reviewer_round_counts.py
"""
import collections
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "revround"

C136 = "results/accord/c136_gate_validation/c136.json"


def wilson(k, n, z=1.959963984540054):
    centre = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    half = (z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n))
            / (1 + z * z / n))
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    facts = json.loads(FACTS.read_text(),
                       object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    c136 = json.loads((ROOT / C136).read_text())
    arm = c136["synthetic_minilm"]["screening"]["auroc_lt_0.60"]["gate"]
    # positive predictive value: of the labellers the gate condemned, how many
    # produced a failing student.
    ppv_k = arm["cell_gatefail_studentfailed"]
    ppv_n = ppv_k + arm["cell_gatefail_studentuseful"]
    # specificity: of the labellers whose student was useful, how many the gate
    # passed.
    spec_k = arm["cell_gatepass_studentuseful"]
    spec_n = spec_k + arm["cell_gatefail_studentuseful"]
    for name, (k, n) in [("ppv", (ppv_k, ppv_n)), ("specificity", (spec_k, spec_n))]:
        lo, hi = wilson(k, n)
        base = f"{PREFIX}.c136.synthetic.minilm.thr0p60.{name}"
        art = f"{C136}::synthetic_minilm.screening.auroc_lt_0.60.gate"
        facts[base + ".k"] = {"artifact": art, "value": k}
        facts[base + ".n"] = {"artifact": art, "value": n}
        facts[base + ".wilson_lo"] = {"artifact": art, "value": lo}
        facts[base + ".wilson_hi"] = {"artifact": art, "value": hi}
        print(f"   {name}: {k} of {n}, Wilson {lo:.4f} to {hi:.4f}")

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"   registered {sum(1 for k in facts if k.startswith(PREFIX + '.'))} entries")


if __name__ == "__main__":
    main()
