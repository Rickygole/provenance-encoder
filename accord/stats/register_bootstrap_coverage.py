"""Register results/accord/c134_reanalysis/c134a_bootstrap_coverage.json under c134a.*

Chapter 3 used to say that nothing here measured how often the percentile
bootstrap's upper end covers the true value at 12 positives, so the calibration
of the bounded null's defining interval was assumed rather than shown. It has
since been measured, and the measurement has a consequence the chapters have to
state: a nominal 95 percent interval covers about 92 percent, so a calibrated
interval would be wider and its upper end higher, which is the end the bounded
null's criterion rests on. A number a chapter must print needs a register key,
or pass 2 of verify_thesis.py cannot resolve it.

The prefix is deleted wholesale before rewriting, the convention every generator
in this directory follows, and every numeric and boolean leaf is rewritten with
an artifact::dotted.path pointer. String leaves are prose describing the design
and stay in the artifact.

Run:
    python3 accord/stats/register_bootstrap_coverage.py
"""

import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT = (ROOT / "results" / "accord" / "c134_reanalysis"
            / "c134a_bootstrap_coverage.json")
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "c134a"

SKIP_LEAF_NAMES = {"what", "addresses", "machinery", "verdict", "file",
                   "runtime_seconds"}


def sanitize(key):
    return (str(key).replace("::", "_").replace(":", "").replace(" ", "_")
            .replace("(", "").replace(")", "").replace(",", ""))


def flatten(prefix, node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in SKIP_LEAF_NAMES:
                continue
            sub = sanitize(k)
            flatten(f"{prefix}.{sub}" if prefix else sub, v, out)
    elif isinstance(node, list):
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in node):
            for i, x in enumerate(node):
                out[f"{prefix}.{i}"] = x
    elif isinstance(node, bool) or isinstance(node, (int, float)):
        out[prefix] = node


def main():
    artifact = json.loads(ARTIFACT.read_text())
    leaves = {}
    flatten("", artifact, leaves)

    design = artifact["design"]
    assert design["n_positives"] == 12, design
    assert design["n_negatives"] == 241, design
    assert design["nominal_coverage"] == 0.95, design

    # The declared rule is the one the thesis's own intervals use; the range the
    # chapters quote is that rule's coverage across the five true values, and the
    # one-sided figure they quote is the error on the upper end specifically.
    declared = {t: v["methods"]["declared_unstratified"]
                for t, v in artifact["per_target"].items()}
    covs = [m["coverage"] for m in declared.values()]
    assert len(covs) == 5, covs
    assert abs(min(covs) - 0.9165) < 5e-5, min(covs)
    assert abs(max(covs) - 0.9345) < 5e-5, max(covs)
    at_floor = declared["0.7466"]
    assert abs(at_floor["miss_rate_interval_entirely_below_truth"] - 0.02) < 5e-5, at_floor

    leaves["declared_unstratified.coverage_min_over_targets"] = round(min(covs), 4)
    leaves["declared_unstratified.coverage_max_over_targets"] = round(max(covs), 4)

    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    rel = ARTIFACT.relative_to(ROOT)
    derived = {"declared_unstratified.coverage_min_over_targets",
               "declared_unstratified.coverage_max_over_targets"}
    n_written = 0
    for path, value in sorted(leaves.items()):
        art = (f"derived: minimum and maximum of {rel}::per_target.*."
               "methods.declared_unstratified.coverage"
               if path in derived else f"{rel}::{path}")
        facts[f"{PREFIX}.{path}"] = collections.OrderedDict(
            [("artifact", art), ("value", value)])
        n_written += 1

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {n_written} keys under {PREFIX}.*")


if __name__ == "__main__":
    main()
