"""Register the C100 off-the-shelf NLI baseline outcome. See C100_PREREGISTRATION.md."""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/differential/nli_roberta_large_mnli.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "nlibaseline"

d = json.loads((ROOT / SRC).read_text())
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]


def put(key, value):
    facts[f"{PREFIX}.{key}"] = collections.OrderedDict(
        [("artifact", SRC), ("value", value)])


for field in ("n_total", "n_pairs_scored", "n_over_token_limit", "max_tokens_seen"):
    put(field, d[field])
for field in ("n", "n_positives", "auroc", "mannwhitney_p_one_sided", "mde_auroc"):
    put(f"primary_g253.{field}", d["primary_g253"][field])
for field in ("n", "n_positives", "auroc", "auroc_ci_lo", "auroc_ci_hi",
              "auroc_bootstrap_B", "auroc_bootstrap_seed", "average_precision",
              "top_k", "top_k_true_positives"):
    put(f"secondary_g299.{field}", d["secondary_g299"][field])
for field in ("auroc_direction1_nursing_premise", "auroc_direction2_discharge_premise"):
    put(f"per_direction_g299.{field}", d["per_direction_g299_descriptive_only"][field])
put("delong_vs_detector.detector_auroc_g253_recomputed",
    d["delong_vs_detector"]["detector_auroc_g253_recomputed"])
put("delong_vs_detector.detector_auroc_g299_recomputed",
    d["delong_vs_detector"]["detector_auroc_g299_recomputed"])
for basis in ("g253", "g299"):
    for field in ("auc_a", "auc_b", "auc_diff", "se_diff", "z", "p"):
        put(f"delong_vs_detector.{basis}.{field}", d["delong_vs_detector"][basis][field])
put("outcome", d["outcome"])

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: G_253 AUROC={d['primary_g253']['auroc']:.4f} "
      f"p={d['primary_g253']['mannwhitney_p_one_sided']:.4f} "
      f"outcome={d['outcome']}")
