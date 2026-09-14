"""Register results/accord/stats/gold_tipping_points.json under goldtipping.*

NOT RUN BY THE GENERATOR. Committed but deliberately not executed here; the
coordinator runs all three register scripts serially once the concurrent
write to VERIFIED_FACTS.json is done.

Same convention as the other two register_*.py scripts in this batch:
deletes every existing "goldtipping." key, then writes one entry per
numeric/boolean leaf in the artifact with an artifact::dotted.path pointer.
Prose (method/conclusion/why_not) and the AUROC trace-by-flip-count list are
kept in the artifact and are NOT duplicated into short scalar register
entries other than the trace's own indexed points, which are numeric and are
registered.

Run (not yet -- coordinator only):
    python3 accord/stats/register_gold_tipping_points.py
"""

import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "results" / "accord" / "stats" / "gold_tipping_points.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "goldtipping"

SKIP_LEAF_NAMES = {
    "method", "note", "notes", "conclusion", "why_not", "formula",
    "target_source", "generated_by", "precondition_checked_first", "scope",
}
SKIP_SUBTREES = {"provenance"}


def sanitize(key):
    return (str(key).replace("::", "_").replace(":", "").replace(" ", "_")
            .replace("(", "").replace(")", "").replace(",", ""))


def flatten(prefix, node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if prefix == "" and k in SKIP_SUBTREES:
                continue
            if k in SKIP_LEAF_NAMES:
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

    # Precondition gate: refuse to register a tipping-point analysis whose
    # own AUROC precondition failed. The generator already aborts in this
    # case and would not have produced a matching artifact, but this is
    # checked again here because the register step must never publish a
    # number computed downstream of a failed precondition.
    if not artifact.get("c_auroc_precondition", {}).get("matches_to_4dp"):
        raise SystemExit(
            "ABORT: c_auroc_precondition.matches_to_4dp is not true in the "
            "artifact; refusing to register goldtipping.* on top of a "
            "failed precondition.")

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

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {n_written} keys under {PREFIX}.*")


if __name__ == "__main__":
    main()
