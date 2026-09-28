#!/usr/bin/env python3
"""Register C139, the rule-identity prior comparison.

    python3 accord/c139_rule_prior/register_c139.py
"""
import collections, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accord.cloud_paths import assert_local_only  # noqa: E402

SRC = "results/accord/c139_rule_prior/c139.json"
FACTS = assert_local_only(ROOT / "results/accord/VERIFIED_FACTS.json")
PREFIX = "c139"

d = json.loads((ROOT / SRC).read_text())
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]

n = 0


def put(key, value, path):
    global n
    facts[f"{PREFIX}.{key}"] = collections.OrderedDict(
        [("value", value), ("artifact", f"{SRC}::{path}")])
    n += 1


rp = d["rule_prior"]
put("rule_prior.auroc", round(rp["auroc"], 4), "rule_prior.auroc")
put("rule_prior.ci95_lo", round(rp["bootstrap"]["ci95"][0], 4), "rule_prior.bootstrap.ci95[0]")
put("rule_prior.ci95_hi", round(rp["bootstrap"]["ci95"][1], 4), "rule_prior.bootstrap.ci95[1]")
put("rule_prior.permutation_p", round(rp["permutation_p"], 4), "rule_prior.permutation_p")
put("rule_prior.n", rp["n"], "rule_prior.n")
put("rule_prior.n_pos", rp["n_pos"], "rule_prior.n_pos")

put("bonferroni_alpha_over_17", d["bonferroni_alpha_over_n"], "bonferroni_alpha_over_n")
put("n_configurations", d["n_configurations"], "n_configurations")

for r, v in d["rule_strata"].items():
    put(f"rule{r}.n_pairs", v["n_pairs"], f"rule_strata.{r}.n_pairs")
    put(f"rule{r}.n_positives", v["n_positives"], f"rule_strata.{r}.n_positives")
    put(f"rule{r}.pharmacist_precision", round(v["pharmacist_precision"], 4),
        f"rule_strata.{r}.pharmacist_precision")

s = d["summary"]
put("strongest_cell.auroc_graded", round(s["strongest_cell_auroc"], 4), "summary.strongest_cell_auroc")
put("strongest_cell.auroc_rule_prior", round(s["rule_prior_auroc"], 4), "summary.rule_prior_auroc")
put("strongest_cell.auc_diff", round(s["strongest_cell_diff"], 4), "summary.strongest_cell_diff")
put("strongest_cell.p_two_sided", round(s["strongest_cell_p_two_sided"], 4),
    "summary.strongest_cell_p_two_sided")
put("n_cells_clearing_bonferroni", s["n_cells_clearing_bonferroni_graded_above_prior"],
    "summary.n_cells_clearing_bonferroni_graded_above_prior")
put("n_cells_uncorrected_p_below_0_05", s["n_cells_uncorrected_p_below_0_05"],
    "summary.n_cells_uncorrected_p_below_0_05")
put("n_cells_above_prior_pointwise", s["n_cells_with_graded_above_prior_pointwise"],
    "summary.n_cells_with_graded_above_prior_pointwise")
put("verdict", s["verdict"], "summary.verdict")

for r, v in d["secondary_within_rule_strata_descriptive"]["strata"].items():
    if v.get("auroc") is not None:
        put(f"within_rule{r}.auroc", round(v["auroc"], 4),
            f"secondary_within_rule_strata_descriptive.strata.{r}.auroc")
    put(f"within_rule{r}.n_pos", v["n_pos"],
        f"secondary_within_rule_strata_descriptive.strata.{r}.n_pos")

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: {n} keys; register now holds {len(facts)}")
