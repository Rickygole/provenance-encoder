"""Emits the register-ready key / artifact / value triples for the two
artifacts this closure produced, WITHOUT writing to
results/accord/VERIFIED_FACTS.json -- another process owns that file and a
concurrent write would corrupt it. The registering agent can consume
results/accord/labeller_interval_closure/REGISTER_THESE.json directly.

Every value is read out of the artifact, never typed here.

Prefixes used, so a future regeneration can delete its own prefix wholesale
per the project's register convention:
    labellerintervals.*      (all 17 configurations, Tasks 1 and 2)
    permlabelctrl.*          (the data-randomisation control, Task 3)
"""
from __future__ import annotations

import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIR = ROOT / "results/accord/labeller_interval_closure"
INTERVALS = DIR / "intervals_17.json"
PERMLABEL = DIR / "permuted_label_control.json"
OUT = DIR / "REGISTER_THESE.json"

REL_I = "results/accord/labeller_interval_closure/intervals_17.json"
REL_P = "results/accord/labeller_interval_closure/permuted_label_control.json"


def short(cell_key: str) -> str:
    family, model, prompt = cell_key.split("::")
    m = model.replace(":", "_").replace(".", "")
    return f"{family}.{m}.{prompt}"


def main():
    keys = collections.OrderedDict()

    def put(key, rel, pointer, value):
        keys[key] = {"artifact": f"{rel}::{pointer}", "value": value}

    d = json.loads(INTERVALS.read_text())
    put("labellerintervals.n_configurations", REL_I, "n_configurations", d["n_configurations"])
    put("labellerintervals.g253.n", REL_I, "g253_n", d["g253_n"])
    put("labellerintervals.g253.n_pos", REL_I, "g253_n_pos", d["g253_n_pos"])
    put("labellerintervals.bootstrap_draws", REL_I, "method.bootstrap_draws", d["method"]["bootstrap_draws"])
    put("labellerintervals.bootstrap_seed", REL_I, "method.bootstrap_seed", d["method"]["bootstrap_seed"])

    s = d["summary"]
    for name in ("n_cells_with_bootstrap_lower_bound_above_0_5_on_g253",
                 "n_confounded_cells",
                 "n_configurations_rejecting_chance_at_0_05",
                 "bonferroni_alpha_over_17",
                 "n_surviving_bonferroni_at_registered_B_20000",
                 "n_surviving_bonferroni_at_B_200000"):
        put(f"labellerintervals.summary.{name}", REL_I, f"summary.{name}", s[name])
    put("labellerintervals.summary.n_cells_meeting_full_mirror_on_g253", REL_I,
        "summary.n_cells_meeting_the_full_mirror_of_the_bounded_null_on_g253",
        s["n_cells_meeting_the_full_mirror_of_the_bounded_null_on_g253"])
    put("labellerintervals.summary.n_cells_interval_above_floor_any_stratum", REL_I,
        "summary.cells_whose_interval_begins_above_the_floor_on_any_stratum (length)",
        len(s["cells_whose_interval_begins_above_the_floor_on_any_stratum"]))
    put("labellerintervals.summary.n_confounded_direction_free", REL_I,
        "summary.confounded_cells_direction_free_same_thresholds (length)",
        len(s["confounded_cells_direction_free_same_thresholds"]))
    put("labellerintervals.summary.n_cells_with_rule_cue_either_direction", REL_I,
        "summary.cells_with_a_significant_rule_type_cue_in_either_direction (length)",
        len(s["cells_with_a_significant_rule_type_cue_in_either_direction"]))

    for cell_key, v in d["cells"].items():
        if not v.get("cache_present"):
            continue
        sk = short(cell_key)
        g = v["g253"]
        put(f"labellerintervals.{sk}.g253.auroc", REL_I, f"cells.{cell_key}.g253.auroc", g["auroc"])
        put(f"labellerintervals.{sk}.g253.permutation_p", REL_I,
            f"cells.{cell_key}.g253.permutation_p", g["permutation_p"])
        put(f"labellerintervals.{sk}.g253.permutation_p_b200000", REL_I,
            f"cells.{cell_key}.g253.permutation_p_high_precision", g["permutation_p_high_precision"])
        put(f"labellerintervals.{sk}.g253.bootstrap95_auroc", REL_I,
            f"cells.{cell_key}.g253.bootstrap.ci95", g["bootstrap"]["ci95"])
        put(f"labellerintervals.{sk}.g253.bootstrap_n_discarded", REL_I,
            f"cells.{cell_key}.g253.bootstrap.n_resamples_discarded_single_class",
            g["bootstrap"]["n_resamples_discarded_single_class"])
        put(f"labellerintervals.{sk}.g253.bootstrap_lower_above_half", REL_I,
            f"cells.{cell_key}.g253.bootstrap.lower_bound_above_half",
            g["bootstrap"]["lower_bound_above_half"])
        put(f"labellerintervals.{sk}.g253.floor_applied", REL_I,
            f"cells.{cell_key}.g253.floor_applied", g["floor_applied"])
        put(f"labellerintervals.{sk}.g253.bootstrap_lower_above_floor", REL_I,
            f"cells.{cell_key}.g253.bootstrap_lower_above_applied_floor",
            g["bootstrap_lower_above_applied_floor"])
        put(f"labellerintervals.{sk}.g253.mirror_of_bounded_null_fully_met", REL_I,
            f"cells.{cell_key}.g253.mirror_of_bounded_null_fully_met",
            g["mirror_of_bounded_null_fully_met"])
        put(f"labellerintervals.{sk}.g253.lopo_auroc_min", REL_I,
            f"cells.{cell_key}.g253.lopo_auroc_min", g["lopo_auroc_min"])
        put(f"labellerintervals.{sk}.g253.lopo_auroc_max", REL_I,
            f"cells.{cell_key}.g253.lopo_auroc_max", g["lopo_auroc_max"])
        for st, nm in (("stratum_rules_1_2", "rules_1_2"), ("stratum_rules_3_4", "rules_3_4")):
            b = v[st]
            if b["auroc"] is None:
                continue
            put(f"labellerintervals.{sk}.{nm}.auroc", REL_I, f"cells.{cell_key}.{st}.auroc", b["auroc"])
            put(f"labellerintervals.{sk}.{nm}.permutation_p", REL_I,
                f"cells.{cell_key}.{st}.permutation_p", b["permutation_p"])
            put(f"labellerintervals.{sk}.{nm}.bootstrap95_auroc", REL_I,
                f"cells.{cell_key}.{st}.bootstrap.ci95", b["bootstrap"]["ci95"])
            put(f"labellerintervals.{sk}.{nm}.bootstrap_n_discarded", REL_I,
                f"cells.{cell_key}.{st}.bootstrap.n_resamples_discarded_single_class",
                b["bootstrap"]["n_resamples_discarded_single_class"])
            put(f"labellerintervals.{sk}.{nm}.floor_applied", REL_I,
                f"cells.{cell_key}.{st}.floor_applied", b["floor_applied"])
            put(f"labellerintervals.{sk}.{nm}.bootstrap_lower_above_floor", REL_I,
                f"cells.{cell_key}.{st}.bootstrap_lower_above_applied_floor",
                b["bootstrap_lower_above_applied_floor"])
            put(f"labellerintervals.{sk}.{nm}.lopo_auroc_min", REL_I,
                f"cells.{cell_key}.{st}.lopo_auroc_min", b["lopo_auroc_min"])
            put(f"labellerintervals.{sk}.{nm}.lopo_auroc_max", REL_I,
                f"cells.{cell_key}.{st}.lopo_auroc_max", b["lopo_auroc_max"])
        c = v["confound_rule12_vs_rule34_among_confirmed_negatives"]
        base = f"cells.{cell_key}.confound_rule12_vs_rule34_among_confirmed_negatives"
        put(f"labellerintervals.{sk}.confound.auroc_rule12_vs_rule34", REL_I,
            f"{base}.auroc_rule12_vs_rule34", c["auroc_rule12_vs_rule34"])
        put(f"labellerintervals.{sk}.confound.permutation_p", REL_I,
            f"{base}.permutation_p", c["permutation_p"])
        put(f"labellerintervals.{sk}.confound.confounded", REL_I,
            f"{base}.confounded", c["confounded"])
        put(f"labellerintervals.{sk}.confound.direction_free_auroc", REL_I,
            f"{base}.direction_free_auroc_max_auc_1minusauc",
            c["direction_free_auroc_max_auc_1minusauc"])
        put(f"labellerintervals.{sk}.confound.direction_free_p_two_sided", REL_I,
            f"{base}.direction_free_permutation_p_two_sided",
            c["direction_free_permutation_p_two_sided"])
        put(f"labellerintervals.{sk}.confound.direction_free_confounded", REL_I,
            f"{base}.direction_free_confounded_same_thresholds",
            c["direction_free_confounded_same_thresholds"])
        put(f"labellerintervals.{sk}.confound.cue_direction", REL_I,
            f"{base}.cue_direction", c["cue_direction"])

    if PERMLABEL.exists():
        q = json.loads(PERMLABEL.read_text())
        put("permlabelctrl.outcome", REL_P, "outcome", q["outcome"])
        put("permlabelctrl.permutation_seed", REL_P, "design.permutation_seed",
            q["design"]["permutation_seed"])
        pg = q["gates"]["permutation"]
        for nm in ("n_rows", "n_positive_before_permutation",
                   "n_positive_after_permutation", "n_rows_whose_label_changed",
                   "fraction_of_rows_whose_label_changed"):
            put(f"permlabelctrl.permutation.{nm}", REL_P, f"gates.permutation.{nm}", pg[nm])
        for basis in ("g299", "g253"):
            for cell, blk in q[basis].items():
                if blk["auroc"] is None:
                    continue
                put(f"permlabelctrl.{basis}.{cell}.auroc", REL_P, f"{basis}.{cell}.auroc", blk["auroc"])
                put(f"permlabelctrl.{basis}.{cell}.ap", REL_P, f"{basis}.{cell}.ap", blk["ap"])
                put(f"permlabelctrl.{basis}.{cell}.bootstrap95_auroc", REL_P,
                    f"{basis}.{cell}.bootstrap.ci95", blk["bootstrap"]["ci95"])
        for nm in ("g253_rules_3_4", "g253_rules_1_2"):
            blk = q[nm]
            if blk["auroc"] is not None:
                put(f"permlabelctrl.{nm}.ensemble.auroc", REL_P, f"{nm}.auroc", blk["auroc"])
                put(f"permlabelctrl.{nm}.ensemble.ap", REL_P, f"{nm}.ap", blk["ap"])
        cmp_ = q["comparison_vs_real_label_run"]
        for nm in ("real_label_ensemble_auroc_g253", "permuted_label_ensemble_auroc_g253",
                   "permuted_label_ensemble_ap_g253",
                   "permuted_auroc_inside_real_label_interval",
                   "auroc_difference_real_minus_permuted"):
            put(f"permlabelctrl.comparison.{nm}", REL_P,
                f"comparison_vs_real_label_run.{nm}", cmp_[nm])
        if cmp_["paired_bootstrap_difference_same_253_pairs"]:
            pb = cmp_["paired_bootstrap_difference_same_253_pairs"]
            put("permlabelctrl.comparison.paired_diff_mean", REL_P,
                "comparison_vs_real_label_run.paired_bootstrap_difference_same_253_pairs.mean",
                pb["mean"])
            put("permlabelctrl.comparison.paired_diff_ci95", REL_P,
                "comparison_vs_real_label_run.paired_bootstrap_difference_same_253_pairs.ci95",
                pb["ci95"])
            put("permlabelctrl.comparison.paired_diff_contains_zero", REL_P,
                "comparison_vs_real_label_run.paired_bootstrap_difference_same_253_pairs.contains_zero",
                pb["contains_zero"])

    OUT.write_text(json.dumps({
        "do_not_write_verified_facts_directly": (
            "another process owns results/accord/VERIFIED_FACTS.json; this file "
            "is the hand-off, not a register write"),
        "prefixes_to_delete_wholesale_on_regeneration": ["labellerintervals.", "permlabelctrl."],
        "n_keys": len(keys),
        "keys": keys,
    }, indent=2) + "\n")
    print(f"wrote {OUT} with {len(keys)} keys")


if __name__ == "__main__":
    main()
