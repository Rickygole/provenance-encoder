"""Register the C105 prior-fix distillation outcome from its artifact.
See C105_PREREGISTRATION.md.

WRITTEN, NOT RUN, by the builder agent per this task's edit-scope rules: the
coordinator runs register scripts serially.
"""
import collections
import json
import pathlib

# Three directories below the repository root (accord/public_controls/priorfix/);
# parents[2] resolved to accord/, the same defect that broke the nlifactorial
# register script.
ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = "results/accord/public_controls/priorfix/analysis.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "priorfix"


def main():
    d = json.loads((ROOT / SRC).read_text())
    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    def put(key, value, node):
        facts[f"{PREFIX}.{key}"] = collections.OrderedDict(
            [("artifact", f"{SRC}::{node}"), ("value", value)])

    put("n_replicates", d["n_replicates"], "n_replicates")
    put("outcome", d["outcome"], "outcome")
    put("delta_ap_mean", d["delta_ap_s1_minus_h"]["mean"], "delta_ap_s1_minus_h.mean")
    put("delta_ap_ci_lo", d["delta_ap_s1_minus_h"]["ci95"][0], "delta_ap_s1_minus_h.ci95.0")
    put("delta_ap_ci_hi", d["delta_ap_s1_minus_h"]["ci95"][1], "delta_ap_s1_minus_h.ci95.1")
    put("delta_auc_mean", d["delta_auc_s1_minus_h"]["mean"], "delta_auc_s1_minus_h.mean")
    put("delta_auc_ci_lo", d["delta_auc_s1_minus_h"]["ci95"][0], "delta_auc_s1_minus_h.ci95.0")
    put("delta_auc_ci_hi", d["delta_auc_s1_minus_h"]["ci95"][1], "delta_auc_s1_minus_h.ci95.1")
    for arm, rate in d["c98_rule_rejection_rate_bootstrap"].items():
        put(f"c98_rejection_rate.{arm}", rate, f"c98_rule_rejection_rate_bootstrap.{arm}")

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {PREFIX}: outcome={d['outcome']} "
          f"delta_ap_mean={d['delta_ap_s1_minus_h']['mean']:.4f}")


if __name__ == "__main__":
    main()
