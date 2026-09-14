"""Register the C104 public NLI factorial outcome from its artifact.
See C104_PREREGISTRATION.md.

WRITTEN, NOT RUN, by the builder agent per this task's edit-scope rules: the
coordinator runs register scripts serially.
"""
import collections
import json
import pathlib

# The script lives three directories below the repository root
# (accord/public_controls/nlifactorial/); parents[2] pointed at accord/ and made
# every artifact path resolve to accord/results/..., which does not exist.
ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = "results/accord/public_controls/nlifactorial/analysis.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "nlifactorial"


def _slug(model, prompt):
    return model.replace(":", "_").replace(".", "") + "_" + prompt


def main():
    d = json.loads((ROOT / SRC).read_text())
    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    def put(key, value, node):
        facts[f"{PREFIX}.{key}"] = collections.OrderedDict(
            [("artifact", f"{SRC}::{node}"), ("value", value)])

    put("n_cells_available", d["n_cells_available"], "n_cells_available")

    for contrast_key, out_key in (
        ("contrast_H_fpr_hard_minus_easy_P0", "contrast_H"),
        ("contrast_P_fpr_p1_minus_p0_medium", "contrast_P"),
    ):
        c = d[contrast_key]
        put(f"{out_key}.outcome", c["outcome"], f"{contrast_key}.outcome")
        put(f"{out_key}.n_models_valid", c["n_models_valid"],
            f"{contrast_key}.n_models_valid")
        put(f"{out_key}.n_models_interval_above_zero", c["n_models_interval_above_zero"],
            f"{contrast_key}.n_models_interval_above_zero")
        for model, v in c["per_model"].items():
            slug = model.replace(":", "_").replace(".", "")
            put(f"{out_key}.{slug}.point", v["point"], f"{contrast_key}.per_model.{model}.point")
            put(f"{out_key}.{slug}.interval_above_zero", v["interval_above_zero"],
                f"{contrast_key}.per_model.{model}.interval_above_zero")

    s = d["contrast_S_overcall_14b_minus_8b_p1_medium"]
    put("contrast_S.available", s.get("available", False),
        "contrast_S_overcall_14b_minus_8b_p1_medium.available")
    if s.get("available"):
        put("contrast_S.outcome", s["outcome"],
            "contrast_S_overcall_14b_minus_8b_p1_medium.outcome")
        put("contrast_S.point", s["point"],
            "contrast_S_overcall_14b_minus_8b_p1_medium.point")

    for cell_key, cell in d["cells"].items():
        slug = cell_key.replace("::", "_").replace(":", "_").replace(".", "")
        put(f"cell.{slug}.valid", cell["valid"], f"cells.{cell_key}.valid")
        put(f"cell.{slug}.tpr", cell["tpr"], f"cells.{cell_key}.tpr")
        for level, lv in cell["levels"].items():
            if lv.get("fpr") is not None:
                put(f"cell.{slug}.{level}.fpr", lv["fpr"], f"cells.{cell_key}.levels.{level}.fpr")
            if lv.get("auroc_vs_positives") is not None:
                put(f"cell.{slug}.{level}.auroc", lv["auroc_vs_positives"],
                    f"cells.{cell_key}.levels.{level}.auroc_vs_positives")

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {PREFIX}: H={d['contrast_H_fpr_hard_minus_easy_P0']['outcome']} "
          f"P={d['contrast_P_fpr_p1_minus_p0_medium']['outcome']} "
          f"S_available={s.get('available', False)}")


if __name__ == "__main__":
    main()
