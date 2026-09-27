"""Register results/accord/clinical_core/displayed_exemplar_lower_bound.json under dispexlb.*

The displayed-exemplar lower bound was measured and drawn in fig-exemplar and
fig-pipeline-status before it had a register key, so its rate (72 of 165) could
not be printed in prose without failing pass 2 of verify_thesis.py. A figure
number that no chapter may state is the same defect as a chapter number no
artifact backs, read from the other side. This registers the analysis the same
way accord/stats/register_overcall_ambiguity_readings.py registers its own
artifact: the prefix is deleted wholesale, then every numeric and boolean leaf
is rewritten with an artifact::dotted.path pointer.

String leaves are not registered. The artifact carries annotation vocabulary
(action-label pairs, synthetic pair identifiers, drug names) that is the
annotation's own, not this analysis's measurement, and it stays in the artifact.

Run:
    python3 accord/stats/register_displayed_exemplar_lower_bound.py
"""

import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT = (ROOT / "results" / "accord" / "clinical_core"
            / "displayed_exemplar_lower_bound.json")
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "dispexlb"

SKIP_LEAF_NAMES = {
    "analysis", "generated_by", "what_is_measured", "why_this_is_a_lower_bound",
    "thesis_claim_this_bears_on", "READ_THIS_BEFORE_THE_TABLE",
    "what_the_comparison_supports", "data_agreement", "note", "notes", "why",
    "rule_short", "trigger_is", "defect_interpretation", "caveat",
    "what_is_independent", "calibration_source", "artifact", "what",
}
SKIP_SUBTREES = {"inputs", "displayed_action_pairs_on_the_defective_rows"}
# One row per annotated pair, carrying synthetic pair identifiers, drug names and
# the pharmacist's danger rating. The counts above it are the measurement; the
# rows are the evidence and stay in the artifact only.
SKIP_SUBTREES_ANY_DEPTH = {"rows"}


def sanitize(key):
    return (str(key).replace("::", "_").replace(":", "").replace(" ", "_")
            .replace("(", "").replace(")", "").replace(",", ""))


def flatten(prefix, node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if prefix == "" and k in SKIP_SUBTREES:
                continue
            if k in SKIP_LEAF_NAMES or k in SKIP_SUBTREES_ANY_DEPTH:
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

    pooled = artifact["rules_1_and_2"]["pooled"]
    assert pooled["n_pairs"] == 165, pooled["n_pairs"]
    assert pooled["n_displayed_pair_is_not_opposite"] == 72, pooled
    assert abs(pooled["rate_not_opposite"]
               - pooled["n_displayed_pair_is_not_opposite"] / pooled["n_pairs"]) < 5e-5, (
        "the artifact's stored rate is not its own two counts")

    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    rel = ARTIFACT.relative_to(ROOT)
    n_written = 0
    for path, value in sorted(leaves.items()):
        facts[f"{PREFIX}.{path}"] = collections.OrderedDict(
            [("artifact", f"{rel}::{path}"), ("value", value)])
        n_written += 1

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {n_written} keys under {PREFIX}.*")


if __name__ == "__main__":
    main()
