#!/usr/bin/env python3
"""Register the plain control's intervals and the verdict its own report coded.

accord/control/report.py, written before any G_253 criterion existed, reads the
plain control's ensemble interval on G_299 and on the Rules 3 and 4 stratum
against a two-part rule: the interval includes 0.5 and its upper end is below a
ceiling of 0.6. Anything wider is recorded as INCONCLUSIVE_WIDE_CI, and that is
what the committed report records. The thesis later fixed its bounded-null
ceiling at the simulated Mann-Whitney floor, 0.7466, so it has to disclose the
earlier, tighter ceiling and what that ceiling would say. The interval ends
were never registered because the register's flattening skipped list values.

Everything here is read from the committed report or from the rule's source
line; nothing is typed.
"""
import collections
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
REPORT = "results/accord/control/final_report.json"
RULE_SRC = "accord/control/report.py"
PREFIX = "controlverdict"


def main():
    rep = json.loads((ROOT / REPORT).read_text())
    src = (ROOT / RULE_SRC).read_text()
    ceilings = set(re.findall(r"ens_ci(?:_df)?\[1\] < ([0-9.]+)", src))
    assert len(ceilings) == 1, f"expected one coded ceiling, found {ceilings}"
    ceiling = float(ceilings.pop())

    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    def put(name, value, where):
        facts[f"{PREFIX}.{name}"] = {"artifact": where, "value": value}
        print(f"   {PREFIX}.{name} = {value}")

    ens = rep["ensemble_mean_of_seed_scores"]
    for stratum, tag in (("gold_all", "g299"), ("gold_defect_free_rule34", "g134")):
        lo, hi = ens[stratum]["bootstrap"]["ci95"]
        base = f"{REPORT}::ensemble_mean_of_seed_scores.{stratum}.bootstrap.ci95"
        put(f"{tag}.ci95_lo", lo, base + ".0")
        put(f"{tag}.ci95_hi", hi, base + ".1")
        put(f"{tag}.upper_end_below_coded_ceiling", hi < ceiling, base + ".1")
    put("interpretation", rep["interpretation"], f"{REPORT}::interpretation")
    put("coded_ceiling", ceiling, f"{RULE_SRC}::ci_excludes_point6_all")
    FACTS.write_text(json.dumps(facts, indent=1) + "\n")


if __name__ == "__main__":
    main()
