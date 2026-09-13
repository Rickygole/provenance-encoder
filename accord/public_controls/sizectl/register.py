"""Register results/accord/public_controls/sizectl/{roberta_scores_meta,
power_curves}.json under sizectl.*

NOT RUN BY THE GENERATOR. Committed but deliberately not executed here; the
coordinator runs this once the concurrent write to VERIFIED_FACTS.json
elsewhere is done, matching the convention of
accord/stats/register_gold_tipping_points.py and its siblings: deletes every
existing "sizectl." key, then writes one entry per numeric/boolean leaf in
each artifact with an artifact::dotted.path pointer. Prose fields and the
per-grid-point fire-rate/params dictionaries are kept in the artifacts and
are not exploded into thousands of register entries; only the leaves listed
in FLATTEN_TOP_LEVEL_ONLY below are registered from `power_curves.json`, to
keep the register's size proportional to what the thesis would ever cite.

Run (not yet -- coordinator only):
    python3 accord/public_controls/sizectl/register.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
META = ROOT / "results" / "accord" / "public_controls" / "sizectl" / "roberta_scores_meta.json"
CURVES = ROOT / "results" / "accord" / "public_controls" / "sizectl" / "power_curves.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "sizectl"

SKIP_LEAF_NAMES = {"method", "note", "notes", "params", "scores_artifact"}


def sanitize(key):
    return (str(key).replace("::", "_").replace(":", "").replace(" ", "_")
            .replace("(", "").replace(")", "").replace(",", ""))


def flatten(prefix, node, out, max_depth=None, depth=0):
    if isinstance(node, dict):
        if max_depth is not None and depth >= max_depth:
            return
        for k, v in node.items():
            if k in SKIP_LEAF_NAMES:
                continue
            sub_key = sanitize(k)
            flatten(f"{prefix}.{sub_key}" if prefix else sub_key, v, out,
                   max_depth, depth + 1)
    elif isinstance(node, list):
        if all(isinstance(x, (int, float, bool)) or x is None for x in node):
            for i, x in enumerate(node):
                out[f"{prefix}.{i}"] = x
    else:
        out[prefix] = node


# power_curves.json's per-grid-point "grid" lists are intentionally not
# exploded (they are large and better read from the artifact directly); only
# these top-level summary keys are registered from it.
CURVES_TOP_LEVEL_KEYS = (
    "A0", "mde_analytic", "n_pos_slots", "n_neg_slots", "n_reps_cheap_rules",
    "n_reps_bootstrap_rule", "primary_rule", "primary_construction",
    "primary_crossing_power_0_80", "primary_rejection_rate_at_0_50",
    "c98_rule_rejection_rate_at_0_7355", "max_population_auroc_off_target",
    "outcome", "wall_clock_seconds",
)
# and these per-construction summaries (crossings and rejection-at-0.50, not
# every grid point).
CURVES_PER_CONSTRUCTION_KEYS = ("crossings_at_power_0_80", "validity")


def main():
    if not META.exists() or not CURVES.exists():
        raise SystemExit(
            f"ABORT: {META} and/or {CURVES} do not exist. Run "
            "score_roberta.py then power_curves.py first; refusing to "
            "register a prefix on top of artifacts that were never produced.")

    meta = json.loads(META.read_text())
    curves = json.loads(CURVES.read_text())

    leaves = {}
    flatten("roberta_scores_meta", meta, leaves)
    for k in CURVES_TOP_LEVEL_KEYS:
        if k in curves:
            flatten(f"power_curves.{sanitize(k)}", curves[k], leaves)
    for construction in ("label_mix", "noise_mix"):
        if construction not in curves:
            continue
        for k in CURVES_PER_CONSTRUCTION_KEYS:
            if k in curves[construction]:
                flatten(f"power_curves.{construction}.{sanitize(k)}",
                       curves[construction][k], leaves)

    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    meta_rel = META.relative_to(ROOT)
    curves_rel = CURVES.relative_to(ROOT)
    n_written = 0
    for path, value in leaves.items():
        if value is None:
            continue
        artifact = str(meta_rel) if path.startswith("roberta_scores_meta") else str(curves_rel)
        facts[f"{PREFIX}.{path}"] = collections.OrderedDict(
            [("artifact", f"{artifact}::{path}"), ("value", value)])
        n_written += 1

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {n_written} keys under {PREFIX}.*")


if __name__ == "__main__":
    main()
