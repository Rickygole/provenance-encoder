"""Register the C99 scale-step outcome. See C99_PREREGISTRATION.md."""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/differential/scale_step_c99.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "scalestep"

d = json.loads((ROOT / SRC).read_text())
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]

for field in ("S", "D", "n_discordant", "b_8b_flag_only", "c_14b_flag_only",
              "mcnemar_exact_two_sided_p", "n_pairs_8b", "n_pairs_14b",
              "failed_parses_8b", "failed_parses_14b",
              "n_thinking_nonempty_8b", "n_thinking_nonempty_14b",
              "n_prompt_truncated_8b", "n_prompt_truncated_14b",
              "n_joined_pairs", "outcome"):
    facts[f"{PREFIX}.{field}"] = collections.OrderedDict(
        [("artifact", SRC), ("value", d[field])])

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: S={d['S']:.4f} D={d['D']:.4f} "
      f"p={d['mcnemar_exact_two_sided_p']:.4f} outcome={d['outcome']}")
