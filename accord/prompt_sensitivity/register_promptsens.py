"""Register the C103 prompt-sensitivity outcome from its artifact.
See C103_PREREGISTRATION.md.

WRITTEN, NOT RUN, by the builder agent per this task's edit-scope rules: the
coordinator runs register scripts serially. Follows the same
read-artifact / delete-own-prefix / rewrite convention as
accord/differential/register_graded_score.py and
accord/stats/marginal_feasibility_gate.py's own register block.
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/prompt_sensitivity/analysis.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "promptsens"

MODELS = ["llama3:8b", "granite3.3:8b", "qwen3:8b"]
MODEL_SLUGS = {"llama3:8b": "llama3_8b", "granite3.3:8b": "granite33_8b",
               "qwen3:8b": "qwen3_8b"}
PROMPTS = ["P_asserting", "P_neutral", "P_reversed"]


def main():
    d = json.loads((ROOT / SRC).read_text())
    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    def put(key, value, node):
        facts[f"{PREFIX}.{key}"] = collections.OrderedDict(
            [("artifact", f"{SRC}::{node}"), ("value", value)])

    put("outcome", d["outcome"], "outcome")
    put("n_models_satisfying_outcome_a", d["n_models_satisfying_outcome_a"],
        "n_models_satisfying_outcome_a")
    put("ceiling_ratio", d["ceiling_ratio"], "ceiling_ratio")
    put("mde_auroc_253", d["mde_auroc_253"], "mde_auroc_253")

    for model in MODELS:
        slug = MODEL_SLUGS[model]
        mv = d["model_verdicts"][model]
        for field in ("mcnemar_p_two_sided_exact", "positive_rate_asserting",
                       "positive_rate_neutral", "neutral_over_call_ratio",
                       "satisfies_outcome_a_criteria"):
            put(f"{slug}.verdict.{field}", mv[field], f"model_verdicts.{model}.{field}")
        for prompt in PROMPTS:
            cell = d["per_model_prompt"][model][prompt]
            for field in ("positive_rate_299", "auroc_253",
                           "mannwhitney_p_one_sided_253",
                           "marginal_feasibility_floor_p", "parse_failure_rate",
                           "valid"):
                if cell.get(field) is not None:
                    put(f"{slug}.{prompt}.{field}", cell[field],
                        f"per_model_prompt.{model}.{prompt}.{field}")

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {PREFIX}: outcome={d['outcome']} "
          f"n_satisfying={d['n_models_satisfying_outcome_a']}")


if __name__ == "__main__":
    main()
