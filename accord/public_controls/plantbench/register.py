"""Register results/accord/plantbench/*.json under plantbench.*

NOT RUN BY THE GENERATOR. Committed but deliberately not executed here; the
coordinator runs this once the concurrent write to VERIFIED_FACTS.json
elsewhere is done. Same flatten convention as
accord/stats/register_gold_tipping_points.py: deletes every existing
"plantbench." key, then writes one entry per numeric/boolean leaf, with an
artifact::dotted.path pointer. `per_replicate` (Stage 2) and `results`
(Stage 3 smoke test, not a verdict) are intentionally not exploded -- only
Stage 3 smoke's top-level summary is registered, never its per-check
numbers, so nothing under `plantbench.stage3_smoke.*` can be mistaken for a
validated result.

Registers whichever of stage1_s1b.json / stage2_s2abc.json / stage3_smoke.json
/ stage3_registered.json / stage4_s4.json exist.

Stage 3 (full, real dimensions) and Stage 4 both exist twice over in this
build: as `stage3_smoke.json` (an explicit smoke test, never a verdict, per
above) and as `stage3_registered.json` / `stage4_s4.json`, the runs the
pre-registration's rule was actually applied to. Per the Correction appended
to accord/public_controls/C102_PREREGISTRATION.md (2026-09-14, coordinator
review): both of the latter used untrained, randomly initialised weights
where the pre-registration specifies the trained checkpoint, so S3a, S3b, S3c
(stage3_registered.json) and S4 (stage4_s4.json) did NOT run as registered
and carry no verdict -- their firings are recorded but are not evidence about
the checks, because an untrained model cannot exhibit the trained-model
behaviour each check is defined on. S3d is the one exception: it tests
whether any gradient reaches the harm head at all, a property that holds
under untrained weights too, so its result stands as the one validated
Stage 3 check. Every key this script writes under stage3_registered.* and
stage4_s4.* carries that caveat in its own artifact string (see
_stage3_stage4_caveat below), not only in a document a reader might not
reach, so the register itself cannot be quoted as a verdict on S3a, S3b, S3c
or S4.

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
    "stage3_registered": set(),
    "stage4_s4": set(),
}

CORRECTION_SOURCE = ("accord/public_controls/C102_PREREGISTRATION.md "
                     "(Correction, 2026-09-14, coordinator review)")


def _stage3_stage4_caveat(stage_name, key):
    """The per-key caveat suffix required for stage3_registered and stage4_s4.

    stage3_registered's checks are S3a, S3b, S3c (untrained, NOT RUN AS
    REGISTERED, no verdict) and S3d (untrained but the check's own registered
    property still holds, so it stands as validated). Every OTHER key in that
    artifact (status, dims, seed_root, and so on) describes the run those
    checks came from, which is the same untrained run, so it gets the general
    correction note too. stage4_s4 is untrained throughout: every key in it
    gets the S4 caveat.
    """
    if stage_name == "stage3_registered":
        if key.startswith("checks.S3d_"):
            return (f" [{CORRECTION_SOURCE}: S3d tests whether any gradient "
                    "reaches the harm head at all, which holds under "
                    "untrained weights too; this result stands and is the "
                    "one validated Stage 3 check.]")
        if (key.startswith("checks.S3a_") or key.startswith("checks.S3b_")
                or key.startswith("checks.S3c_")):
            return (f" [{CORRECTION_SOURCE}: run with untrained, randomly "
                    "initialised weights, not the trained checkpoint the "
                    "pre-registration specifies; an untrained model cannot "
                    "exhibit the trained-model behaviour this check is "
                    "defined on. NOT RUN AS REGISTERED, no verdict.]")
        return (f" [{CORRECTION_SOURCE}: this run used untrained, randomly "
                "initialised weights, not the trained checkpoint the "
                "pre-registration specifies (see checks.S3a/S3b/S3c/S3d for "
                "the per-check consequence).]")
    if stage_name == "stage4_s4":
        return (f" [{CORRECTION_SOURCE}: Stage 4 consumed Stage 3 scores "
                "produced by that same untrained run, carrying no signal by "
                "construction; NOT RUN AS REGISTERED, no verdict.]")
    return ""


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
    "stage3_registered": DIR / "stage3_registered.json",
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
            # k is prefixed with the stage name (e.g. "stage3_registered.checks.
            # S3a_....verdict"); strip that stage prefix back off before
            # matching it against the per-check caveat logic, which works in
            # terms of the key's position inside its OWN artifact (e.g.
            # "checks.S3a_....verdict").
            inner_key = k[len(stage_name) + 1:] if k.startswith(stage_name + ".") else k
            caveat = _stage3_stage4_caveat(stage_name, inner_key)
            facts[f"{PREFIX}.{k}"] = collections.OrderedDict(
                [("artifact", f"{artifact}::{k}{caveat}"), ("value", v)])
            n_written += 1

    if not any_found:
        raise SystemExit(
            f"ABORT: none of {list(STAGE_FILES.values())} exist. Run at "
            "least one stage first; refusing to register an empty prefix.")

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {n_written} keys under {PREFIX}.*")


if __name__ == "__main__":
    main()
