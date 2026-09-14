"""Register C114 from its artifact. Deletes the c114 prefix wholesale and re-adds.

    python3 accord/c114_committee_checks/register_c114.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/c114_committee_checks/c114.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "c114"

d = json.loads((ROOT / SRC).read_text())
assert d["outcome"] == "valid", d
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]

paths = {
    "task_a.outcome": "task_a_outcome",
    "task_a.ap_nli": "task_a_nli_minus_detector_ensemble_g253.observed_ap_system1",
    "task_a.ap_detector_ensemble": "task_a_nli_minus_detector_ensemble_g253.observed_ap_system2",
    "task_a.observed_diff": "task_a_nli_minus_detector_ensemble_g253.observed_diff",
    "task_a.p_two_sided": "task_a_nli_minus_detector_ensemble_g253.p_permutation_two_sided",
    "task_a.B": "task_a_nli_minus_detector_ensemble_g253.B",
    "task_b.baseline_auc": "task_b_tipping_from_decontam_ensemble_g253.baseline_auc",
    "task_b.search_cap": "task_b_tipping_from_decontam_ensemble_g253.search_cap",
    "task_c.n": "task_c_bootstrap_rules34_ambiguous_dropped.n",
    "task_c.n_pos": "task_c_bootstrap_rules34_ambiguous_dropped.n_pos",
    "task_c.stratum_auroc": "task_c_bootstrap_rules34_ambiguous_dropped.stratum_auroc",
    "task_c.ci95_lo": "task_c_bootstrap_rules34_ambiguous_dropped.ci95.0",
    "task_c.ci95_hi": "task_c_bootstrap_rules34_ambiguous_dropped.ci95.1",
    "task_c.n_boot_valid": "task_c_bootstrap_rules34_ambiguous_dropped.n_boot_valid",
}
for name, v in d["task_b_tipping_from_decontam_ensemble_g253"]["targets"].items():
    for f in ("target_auc", "n_flips_greedy", "final_auc", "reached_within_cap"):
        paths[f"task_b.{name}.{f}"] = f"task_b_tipping_from_decontam_ensemble_g253.targets.{name}.{f}"
    for f in v["flip_directions"]:
        paths[f"task_b.{name}.{f}"] = f"task_b_tipping_from_decontam_ensemble_g253.targets.{name}.flip_directions.{f}"


def get(obj, dotted):
    for part in dotted.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


# Task A's raw-score paired test is superseded by C116 (scores on different scales).
SUPERSEDED = "withdrawn.superseded_c116"
for k in [k for k in facts if k.startswith(f"{SUPERSEDED}.{PREFIX}.")]:
    del facts[k]
for k, path in paths.items():
    key = f"{PREFIX}.{k}"
    if k.startswith("task_a."):
        key = f"{SUPERSEDED}.{key}"
    facts[key] = {"artifact": f"{SRC}::{path}", "value": get(d, path)}
FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {len(paths)} c114 keys; total {len(facts)}")
