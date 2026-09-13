"""Register results/accord/plantbench/*.json under plantbench.*

NOT RUN BY THE GENERATOR. Committed but deliberately not executed here; the
coordinator runs this once the concurrent write to VERIFIED_FACTS.json
elsewhere is done. Same flatten convention as
accord/stats/register_gold_tipping_points.py: deletes every existing
"plantbench." key, then writes one entry per numeric/boolean leaf, with an
artifact::dotted.path pointer. `per_replicate` (Stage 2) and `results`
(Stage 3, explicitly a smoke test, not a verdict) are intentionally not
exploded -- only Stage 3's top-level smoke-test summary is registered, never
its per-check numbers, so nothing under `plantbench.stage3.*` can be
mistaken for a validated result.

Registers whichever of stage1_s1b.json / stage2_s2abc.json / stage3_smoke.json
/ stage4_s4.json exist; Stage 4 is expected to be absent in this build.

Run (not yet -- coordinator only):
    python3 accord/public_controls/plantbench/register.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
DIR = ROOT / "results" / "accord" / "plantbench"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "plantbench"

SKIP_LEAF_NAMES = {"method", "note", "notes", "params", "disclosed_deviation_s2b_bootstrap_mechanism",
                  "disclosed_ambiguity_s2b_plant_clean_mapping", "note_s2c_design"}
SKIP_SUBTREES_BY_STAGE = {
    "stage1_s1b": set(),
    "stage2_s2abc": {"per_replicate"},
    "stage3_smoke": {"results"},  # smoke test only; never registered as a verdict
    "stage4_s4": set(),
}


def sanitize(key):
    return (str(key).replace("::", "_").replace(":", "").replace(" ", "_")
            .replace("(", "").replace(")", "").replace(",", ""))


def flatten(prefix, node, out, skip_subtrees, depth=0):
    if isinstance(node, dict):
        for k, v in node.items():
            if depth == 0 and k in skip_subtrees:
                continue
            if k in SKIP_LEAF_NAMES:
                continue
            sub_key = sanitize(k)
            flatten(f"{prefix}.{sub_key}" if prefix else sub_key, v, out, skip_subtrees, depth + 1)
    elif isinstance(node, list):
        if all(isinstance(x, (int, float, bool)) or x is None for x in node):
            for i, x in enumerate(node):
                out[f"{prefix}.{i}"] = x
    else:
        out[prefix] = node


STAGE_FILES = {
    "stage1_s1b": DIR / "stage1_s1b.json",
    "stage2_s2abc": DIR / "stage2_s2abc.json",
    "stage3_smoke": DIR / "stage3_smoke.json",
    "stage4_s4": DIR / "stage4_s4.json",
}


def main():
    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    n_written = 0
    any_found = False
    for stage_name, path in STAGE_FILES.items():
        if not path.exists():
            continue
        any_found = True
        artifact = path.relative_to(ROOT)
        node = json.loads(path.read_text())
        leaves = {}
        flatten(stage_name, node, leaves, SKIP_SUBTREES_BY_STAGE[stage_name])
        for k, v in leaves.items():
            if v is None:
                continue
            facts[f"{PREFIX}.{k}"] = collections.OrderedDict(
                [("artifact", f"{artifact}::{k}"), ("value", v)])
            n_written += 1

    if not any_found:
        raise SystemExit(
            f"ABORT: none of {list(STAGE_FILES.values())} exist. Run at "
            "least one stage first; refusing to register an empty prefix.")

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {n_written} keys under {PREFIX}.*")


if __name__ == "__main__":
    main()
