"""Register the C109 outcome from its artifact. See C109_PREREGISTRATION.md.

WRITTEN, NOT RUN, by the builder agent per this task's edit-scope rules: the
coordinator runs register scripts serially. Follows the same
read-artifact / delete-own-prefix / rewrite convention as
accord/prompt_sensitivity/register_promptsens.py and
accord/differential/register_graded_score.py.
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/prompt_sensitivity/c109/analysis.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "promptconfirm"

MODELS = ["llama3:8b", "granite3.3:8b"]
MODEL_SLUGS = {"llama3:8b": "llama3_8b", "granite3.3:8b": "granite33_8b"}
NEW_PROMPTS = ["P_rev_agree", "P_rev_follow", "P_rev_compatible"]
ALL_PROMPTS = NEW_PROMPTS + ["P_reversed_rerun"]


def main():
    d = json.loads((ROOT / SRC).read_text())
    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    def put(key, value, node):
        facts[f"{PREFIX}.{key}"] = collections.OrderedDict(
            [("artifact", f"{SRC}::{node}"), ("value", value)])

    put("outcome", d["outcome"], "outcome")
    put("outcome_note", d["outcome_note"], "outcome_note")
    put("n_new_paraphrases_satisfying_llama3_8b",
        d["n_new_paraphrases_satisfying_llama3_8b"],
        "n_new_paraphrases_satisfying_llama3_8b")
    put("mde_auroc_253", d["mde_auroc_253"], "mde_auroc_253")
    put("mde_auroc_defect_free_rules_3_4", d["mde_auroc_defect_free_rules_3_4"],
        "mde_auroc_defect_free_rules_3_4")

    for prompt in NEW_PROMPTS:
        pv = d["primary_paraphrase_verdicts_llama3_8b"][prompt]
        for field in ("auroc_253", "permutation_p_253", "satisfies_outcome_a_criterion"):
            put(f"llama3_8b.{prompt}.{field}", pv[field],
                f"primary_paraphrase_verdicts_llama3_8b.{prompt}.{field}")

    for model in MODELS:
        slug = MODEL_SLUGS[model]
        det = d["determinism_check"][model]
        for field in ("n_common_scored", "n_label_mismatch", "max_score_diff", "passes"):
            if det.get(field) is not None:
                put(f"{slug}.determinism_check.{field}", det[field],
                    f"determinism_check.{model}.{field}")
        for prompt in ALL_PROMPTS:
            cell = d["per_model_prompt"][model][prompt]
            for field in ("auroc_253", "permutation_p_253", "parse_failure_rate",
                           "degenerate_rate", "n_truncated", "valid"):
                if cell.get(field) is not None:
                    put(f"{slug}.{prompt}.{field}", cell[field],
                        f"per_model_prompt.{model}.{prompt}.{field}")
            key = f"{model}::{prompt}"
            conf = d["confound_test"].get(key)
            if conf is not None:
                for field in ("auroc_rule12_vs_rule34", "permutation_p", "confounded"):
                    if conf.get(field) is not None:
                        put(f"{slug}.{prompt}.confound.{field}", conf[field],
                            f"confound_test.{key}.{field}")
            strat = d["stratum_report"].get(key)
            if strat is not None:
                for stratum_name in ("rules_1_2", "rules_3_4"):
                    s = strat[stratum_name]
                    for field in ("n", "n_pos", "auroc", "permutation_p"):
                        if s.get(field) is not None:
                            put(f"{slug}.{prompt}.stratum.{stratum_name}.{field}",
                                s[field],
                                f"stratum_report.{key}.{stratum_name}.{field}")

    for prompt in ALL_PROMPTS:
        sec = d["secondary_llama3_8b_lopo_bootstrap"][prompt]
        for field in ("lopo_auroc_min", "lopo_auroc_max", "bootstrap95_auroc"):
            if sec.get(field) is not None:
                put(f"llama3_8b.{prompt}.secondary.{field}", sec[field],
                    f"secondary_llama3_8b_lopo_bootstrap.{prompt}.{field}")

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {PREFIX}: outcome={d['outcome']} ({d['outcome_note']})")


if __name__ == "__main__":
    main()
