"""Register results/accord/stats/overcall_ambiguity_readings.json under overcallamb.*

NOT RUN BY THE GENERATOR. Committed but deliberately not executed here; the
coordinator runs all three register scripts serially once the concurrent
write to VERIFIED_FACTS.json is done.

Same convention as accord/stats/register_overcall_decomposition.py: deletes
every existing "overcallamb." key, then writes one entry per numeric/boolean
leaf in the artifact with an artifact::dotted.path pointer. Prose fields and
lists of string identifiers are left in the artifact only.

Run (not yet -- coordinator only):
    python3 accord/stats/register_overcall_ambiguity_readings.py
"""

import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "results" / "accord" / "stats" / "overcall_ambiguity_readings.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "overcallamb"

SKIP_LEAF_NAMES = {
    "method", "note", "notes", "conclusion", "why_not", "formula",
    "register_source", "public_max_source", "generated_by",
}
SKIP_SUBTREES = {"provenance"}
# label_clean_counts_per_pair is an aggregate dict of category -> count read
# from clinical annotation data; the counts themselves are safe aggregate
# numbers and already cross-checked against the register's 241/46/12, but the
# category strings are annotation vocabulary, not this analysis's own
# measurement, so they are left in the artifact rather than duplicated here.
SKIP_SUBTREES_ANY_DEPTH = {"local_register_cross_check"}


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
            sub_key = sanitize(k)
            flatten(f"{prefix}.{sub_key}" if prefix else sub_key, v, out)
    elif isinstance(node, list):
        if all(isinstance(x, (int, float, bool)) or x is None for x in node):
            for i, x in enumerate(node):
                out[f"{prefix}.{i}"] = x
    else:
        out[prefix] = node


def main():
    artifact = json.loads(ARTIFACT.read_text())
    leaves = {}
    flatten("", artifact, leaves)

    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    rel = ARTIFACT.relative_to(ROOT)
    n_written = 0
    for path, value in leaves.items():
        if value is None:
            continue
        facts[f"{PREFIX}.{path}"] = collections.OrderedDict(
            [("artifact", f"{rel}::{path}"), ("value", value)])
        n_written += 1

    # Registered separately, explicitly, because the cross-check result (does
    # local recomputation from clinical annotation data match the register's
    # 241/46/12 split) is the one fact from local_register_cross_check this
    # analysis actually depends on.
    cc = artifact.get("local_register_cross_check", {})
    if "matches_register_241_46_12" in cc:
        facts[f"{PREFIX}.local_cross_check_matches_register"] = collections.OrderedDict(
            [("artifact", f"{rel}::local_register_cross_check.matches_register_241_46_12"),
             ("value", bool(cc["matches_register_241_46_12"]))])
        n_written += 1

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {n_written} keys under {PREFIX}.*")


if __name__ == "__main__":
    main()
