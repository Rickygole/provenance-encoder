"""Register results/accord/stats/pvalue_precision_audit.json under pvalprec.*

The p-value precision audit counts how many p values the thesis prints, how many
are not at four decimals, and how many change when rounded to four. Those counts
belong in the correction record and in the chapter that states the reporting
convention, so they need register keys.

Only numeric and boolean leaves are registered. The audit's string leaves are the
printed p values themselves, their locating context, and their register keys; they
stay in the artifact. The per-value list is skipped outright: it is one row per
printed occurrence and its line numbers move whenever a chapter is edited, which
would churn the register on every unrelated edit. The summary counts are what a
chapter can print.

Run:
    python3 accord/stats/register_pvalue_precision.py
"""

import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
ARTIFACT = ROOT / "results" / "accord" / "stats" / "pvalue_precision_audit.json"
PREFIX = "pvalprec"

SKIP_LEAF_NAMES = {"what", "addresses", "edits_made", "rounding_convention",
                   "files_scanned", "thresholds_checked"}
SKIP_SUBTREES = {"all_p_values", "not_at_four_decimals",
                 "flagged_rounding_to_four_changes_how_it_reads",
                 "trace_rows_whose_value_disagrees_with_their_register_key"}


def sanitize(key):
    return (str(key).replace("::", "_").replace(":", "").replace(" ", "_")
            .replace("(", "").replace(")", "").replace(",", ""))


def flatten(prefix, node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in SKIP_LEAF_NAMES:
                continue
            flatten(f"{prefix}.{sanitize(k)}" if prefix else sanitize(k), v, out)
    elif isinstance(node, list):
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in node):
            for i, x in enumerate(node):
                out[f"{prefix}.{i}"] = x
    elif isinstance(node, bool) or isinstance(node, (int, float)):
        out[prefix] = node


def main():
    artifact = json.loads(ARTIFACT.read_text())
    trimmed = {k: v for k, v in artifact.items() if k not in SKIP_SUBTREES}

    assert artifact["edits_made"].startswith("none"), artifact["edits_made"]
    assert artifact["n_p_values"] == sum(
        int(v) for v in artifact["count_by_printed_decimals"].values()), artifact
    assert artifact["n_not_at_four_decimals"] == len(
        artifact["not_at_four_decimals"]), artifact

    leaves = {}
    flatten("", trimmed, leaves)

    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    rel = ARTIFACT.relative_to(ROOT)
    for dotted, value in sorted(leaves.items()):
        facts[f"{PREFIX}.{dotted}"] = collections.OrderedDict(
            [("artifact", f"{rel}::{dotted}"), ("value", value)])

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {len(leaves)} keys under {PREFIX}.*; "
          f"register now holds {len(facts)}")


if __name__ == "__main__":
    main()
