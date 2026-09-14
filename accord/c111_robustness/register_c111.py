"""Register the C111 robustness-check outcome from its artifact.
See C111_PREREGISTRATION.md.

WRITTEN, NOT RUN as part of this task (out of edit scope for
results/accord/VERIFIED_FACTS.json). Mirrors the pattern in
accord/tfidf_permutation/register_tfidfperm.py: delete this prefix wholesale,
re-add from the committed artifact only.

    python3 accord/c111_robustness/register_c111.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/c111_robustness/c111_robustness.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "c111"

d = json.loads((ROOT / SRC).read_text())
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]

entries = {
    "task_a.detector.seed0.ap": d["task_a"]["detector"]["seed0"]["observed_ap"],
    "task_a.detector.seed0.p_permutation_ap": d["task_a"]["detector"]["seed0"]["p_permutation_ap_one_sided"],
    "task_a.detector.seed1.ap": d["task_a"]["detector"]["seed1"]["observed_ap"],
    "task_a.detector.seed1.p_permutation_ap": d["task_a"]["detector"]["seed1"]["p_permutation_ap_one_sided"],
    "task_a.detector.seed2.ap": d["task_a"]["detector"]["seed2"]["observed_ap"],
    "task_a.detector.seed2.p_permutation_ap": d["task_a"]["detector"]["seed2"]["p_permutation_ap_one_sided"],
    "task_a.detector.ensemble.ap": d["task_a"]["detector"]["ensemble"]["observed_ap"],
    "task_a.detector.ensemble.p_permutation_ap": d["task_a"]["detector"]["ensemble"]["p_permutation_ap_one_sided"],
    "task_a.detector.contaminated_g299_descriptive.ap": d["task_a"]["detector"][
        "contaminated_accord_contra_prob_g299_descriptive_only"]["observed_ap"],
    "task_a.detector.contaminated_g299_descriptive.p_permutation_ap": d["task_a"]["detector"][
        "contaminated_accord_contra_prob_g299_descriptive_only"]["p_permutation_ap_one_sided"],
    "task_a.nli.g253.ap": d["task_a"]["nli"]["g253"]["observed_ap"],
    "task_a.nli.g253.p_permutation_ap": d["task_a"]["nli"]["g253"]["p_permutation_ap_one_sided"],
    "task_a.nli.g299.ap": d["task_a"]["nli"]["g299"]["observed_ap"],
    "task_a.nli.g299.p_permutation_ap": d["task_a"]["nli"]["g299"]["p_permutation_ap_one_sided"],
    "task_a.tfidf.g253.ap": d["task_a"]["tfidf"]["g253"]["observed_ap"],
    "task_a.tfidf.g253.p_permutation_ap": d["task_a"]["tfidf"]["g253"]["p_permutation_ap_one_sided"],
    "task_a.tfidf.g299.ap": d["task_a"]["tfidf"]["g299"]["observed_ap"],
    "task_a.tfidf.g299.p_permutation_ap": d["task_a"]["tfidf"]["g299"]["p_permutation_ap_one_sided"],
    "task_b.tfidf_minus_detector_ensemble.observed_diff": d["task_b_paired"][
        "tfidf_minus_detector_ensemble"]["observed_diff"],
    "task_b.tfidf_minus_detector_ensemble.p_two_sided": d["task_b_paired"][
        "tfidf_minus_detector_ensemble"]["p_permutation_two_sided"],
    "task_b.tfidf_minus_nli.observed_diff": d["task_b_paired"]["tfidf_minus_nli"]["observed_diff"],
    "task_b.tfidf_minus_nli.p_two_sided": d["task_b_paired"]["tfidf_minus_nli"]["p_permutation_two_sided"],
    "task_c.lopo_g253.ap_min": d["task_c_lopo"]["g253"]["ap_min"],
    "task_c.lopo_g253.ap_max": d["task_c_lopo"]["g253"]["ap_max"],
    "task_c.lopo_g253.p_min": d["task_c_lopo"]["g253"]["p_min"],
    "task_c.lopo_g253.p_max": d["task_c_lopo"]["g253"]["p_max"],
    "task_c.lopo_g253.n_drops_below_0.05": d["task_c_lopo"]["g253"]["n_drops_p_below_0.05"],
    "task_c.lopo_g299.ap_min": d["task_c_lopo"]["g299"]["ap_min"],
    "task_c.lopo_g299.ap_max": d["task_c_lopo"]["g299"]["ap_max"],
    "task_c.lopo_g299.p_min": d["task_c_lopo"]["g299"]["p_min"],
    "task_c.lopo_g299.p_max": d["task_c_lopo"]["g299"]["p_max"],
    "task_c.lopo_g299.n_drops_below_0.05": d["task_c_lopo"]["g299"]["n_drops_p_below_0.05"],
    "task_d.n_train_rows": d["task_d_disjointness"]["n_train_rows"],
    "task_d.n_gold_rows": d["task_d_disjointness"]["n_gold_rows"],
    "task_d.n_pair_overlap": d["task_d_disjointness"]["n_pair_overlap"],
    "task_d.n_subject_overlap": d["task_d_disjointness"]["n_subject_overlap"],
    "decision.outcome": d["decision"]["outcome"],
    "decision.failed_conditions": d["decision"]["failed_conditions"],
    "decision.nli_g253_significant": d["decision"]["nli_g253_significant"],
    "decision.nli_g299_significant": d["decision"]["nli_g299_significant"],
    "decision.nli_interpretation": d["decision"]["nli_interpretation"],
}
for field, value in entries.items():
    facts[f"{PREFIX}.{field}"] = collections.OrderedDict(
        [("artifact", f"{SRC}::{field}"), ("value", value)]
    )

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: outcome={d['decision']['outcome']} "
      f"nli_g253_significant={d['decision']['nli_g253_significant']}")
