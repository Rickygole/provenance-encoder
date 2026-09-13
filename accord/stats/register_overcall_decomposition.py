"""Register results/accord/stats/overcall_decomposition.json under overcalldecomp.*

NOT RUN BY THE GENERATOR. Per the coordination rule for this round (another
agent is writing VERIFIED_FACTS.json concurrently), this script is committed
but deliberately not executed here; the coordinator runs all three register
scripts serially once the concurrent write is done.

Behaviour, matching the existing convention (e.g. accord/stats/
marginal_feasibility_gate.py): deletes every key already under the
"overcalldecomp." prefix, then writes one register entry per numeric/boolean
leaf in the artifact, each carrying an artifact::dotted.path pointer. Long
prose fields (method/note/conclusion notes, filenames, lists of string cell
keys) are deliberately not registered as "facts"; they stay in the artifact
only, matching the existing register's convention of short scalar values.

Run (not yet -- coordinator only):
    python3 accord/stats/register_overcall_decomposition.py
"""

import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "results" / "accord" / "stats" / "overcall_decomposition.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "overcalldecomp"

# Leaf key NAMES excluded from registration wherever they occur: prose,
# filenames, or lists of string identifiers rather than measured values.
SKIP_LEAF_NAMES = {
    "method", "note", "notes", "conclusion", "why_not", "formula",
    "three_way_snli_label_note", "per_item_files_checked",
    "cells_failing_identity_nominal_p", "cells_failing_identity_realised_p",
    "register_source", "source_artifact", "public_max_source",
    "target_source", "generated_by",
}
# Top-level subtrees excluded wholesale.
SKIP_SUBTREES = {"provenance"}


def sanitize(key):
    return (str(key).replace("::", "_").replace(":", "").replace(" ", "_")
            .replace("(", "").replace(")", "").replace(",", ""))


def flatten(prefix, node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in SKIP_SUBTREES and prefix == "":
                continue
            if k in SKIP_LEAF_NAMES:
                continue
            sub_key = sanitize(k)
            flatten(f"{prefix}.{sub_key}" if prefix else sub_key, v, out)
    elif isinstance(node, list):
        if all(isinstance(x, (int, float, bool)) or x is None for x in node):
            for i, x in enumerate(node):
                out[f"{prefix}.{i}"] = x
        # lists of strings (e.g. cell-key lists) are not flattened: their
        # counts are already separate scalar fields in the artifact.
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
            continue  # register omits Nones the same way "not recoverable" fields already say so in prose
        facts[f"{PREFIX}.{path}"] = collections.OrderedDict(
            [("artifact", f"{rel}::{path}"), ("value", value)])
        n_written += 1

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {n_written} keys under {PREFIX}.*")


if __name__ == "__main__":
    main()
