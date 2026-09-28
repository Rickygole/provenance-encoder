"""Register the 12 confirmed contradictions and their table under clincore.*

The characterisation of the 12 pharmacist-confirmed contradictions has existed as
an artifact since accord/clinical_core/characterise_confirmed_contradictions.py
ran, but it had no register prefix, so no count from it could be printed without
failing pass 2 of verify_thesis.py. Three reviews have asked for the 12 to be
shown in a table; a table of numbers a chapter may not print is not a table it can
carry. This registers both the characterisation and the table body derived from
it, under one prefix.

Only numeric and boolean leaves are registered. Every string leaf stays in the
artifact, which matters more here than elsewhere: the artifact's string leaves are
the annotation's own vocabulary, including the synthetic pair identifiers, the
per-side action labels and the LaTeX table body itself. The per-pair rows are the
evidence, not the measurement, and the subtree holding them is skipped outright.

Run:
    python3 accord/clinical_core/register_confirmed_contradictions.py
"""

import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
CC = ROOT / "results" / "accord" / "clinical_core"
CHARACTERISATION = CC / "confirmed_contradictions.json"
TABLE = CC / "confirmed_contradictions_table.json"
PREFIX = "clincore"

SKIP_LEAF_NAMES = {
    "analysis", "generated_by", "what", "addresses", "source_artifact",
    "data_agreement", "note", "notes", "why", "caveat", "consequence",
    "column_alignment", "danger_rating_is", "latex_table_body",
    "latex_table_note",
}
# the 12 per-pair rows carry synthetic pair identifiers and per-side action
# labels; they are the evidence and stay in the artifact only
SKIP_SUBTREES = {"confirmed_contradictions", "inputs", "rule_definitions",
                 "rule_short_names", "publishability_confirmation", "columns",
                 "distinct_drugs", "distinct_classes"}


def sanitize(key):
    return (str(key).replace("::", "_").replace(":", "").replace(" ", "_")
            .replace("(", "").replace(")", "").replace(",", ""))


def flatten(prefix, node, out, depth=0):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in SKIP_LEAF_NAMES:
                continue
            if depth == 0 and k in SKIP_SUBTREES:
                continue
            sub = sanitize(k)
            flatten(f"{prefix}.{sub}" if prefix else sub, v, out, depth + 1)
    elif isinstance(node, list):
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in node):
            for i, x in enumerate(node):
                out[f"{prefix}.{i}"] = x
    elif isinstance(node, bool) or isinstance(node, (int, float)):
        out[prefix] = node


def main():
    char = json.loads(CHARACTERISATION.read_text())
    table = json.loads(TABLE.read_text())

    assert len(char["confirmed_contradictions"]) == 12, char["gold_label_distribution"]
    assert char["n_gold_pairs"] == 299, char["n_gold_pairs"]
    assert char["gold_label_distribution"] == {"not_contradiction": 241,
                                               "ambiguous": 46,
                                               "contradiction": 12}, \
        char["gold_label_distribution"]
    assert sum(char["per_rule_positives"].values()) == 12, char["per_rule_positives"]
    assert sum(char["danger_distribution"].values()) == 12, char["danger_distribution"]
    assert char["thesis_crosscheck_all_agree"] is True, char["thesis_crosscheck"]
    assert table["n_rows"] == 12, table["n_rows"]
    pub = table["publishability"]
    assert pub["drug_names_all_in_ALL_DRUGS"] is True, pub
    assert pub["classes_all_in_DRUG_CLASSES_or_CLASS_SUPPLEMENT"] is True, pub
    assert pub["no_pair_identifier_in_body"] is True, pub
    assert "PAIR_" not in table["latex_table_body"], "a pair identifier reached the body"

    leaves = {}
    flatten("", char, leaves)
    table_leaves = {}
    flatten("", table, table_leaves)

    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    n = 0
    for src, sub, tree in ((CHARACTERISATION, "", leaves),
                           (TABLE, "table", table_leaves)):
        rel = src.relative_to(ROOT)
        for dotted, value in sorted(tree.items()):
            key = f"{PREFIX}.{sub}.{dotted}" if sub else f"{PREFIX}.{dotted}"
            facts[key] = collections.OrderedDict(
                [("artifact", f"{rel}::{dotted}"), ("value", value)])
            n += 1

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {n} keys under {PREFIX}.*; register now holds {len(facts)}")


if __name__ == "__main__":
    main()
