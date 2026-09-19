#!/usr/bin/env python3
"""Make two superseded test results citable so the thesis can disclose them.

The register files a superseded value under `withdrawn.superseded_*`, and the
numeric gate refuses to match a printed number against anything under
`withdrawn.`. That is right for a value that must not be reported as a result.
It is wrong for a value the thesis has to DISCLOSE: the first pre-registered
paired average-precision test separated the bag-of-words baseline from the
detector, a flaw in its statistic was then found (it swapped raw scores between
systems whose scales differ), and the rank-based replacement was written after
that result had been seen. A thesis that prints only the replacement's null has
hidden a rejecting test, which is the defect its own multiplicity section exists
to prevent.

This script registers the two superseded p values under
`disclosed_superseded.*`, read from the committed artifacts, so they can be
printed beside the statement that they are superseded and why. The entries under
`withdrawn.superseded_c116.*` stay where they are: the values remain unusable as
results, and these keys exist only so the disclosure passes the gate.
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "disclosed_superseded"

SOURCES = [
    ("c111.task_b.tfidf_minus_detector_ensemble.p_two_sided",
     "results/accord/c111_robustness/c111_robustness.json",
     ("task_b_paired", "tfidf_minus_detector_ensemble", "p_permutation_two_sided")),
    ("c114.task_a.nli_minus_detector_ensemble.p_two_sided",
     "results/accord/c114_committee_checks/c114.json",
     ("task_a_nli_minus_detector_ensemble_g253", "p_permutation_two_sided")),
]


def dig(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def main():
    facts = json.loads(FACTS.read_text(),
                       object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]
    for name, artifact, path in SOURCES:
        value = dig(json.loads((ROOT / artifact).read_text()), path)
        facts[f"{PREFIX}.{name}"] = {
            "artifact": artifact + "::" + ".".join(path),
            "value": value,
        }
        print(f"   {PREFIX}.{name} = {value}")
    FACTS.write_text(json.dumps(facts, indent=1) + "\n")


if __name__ == "__main__":
    main()
