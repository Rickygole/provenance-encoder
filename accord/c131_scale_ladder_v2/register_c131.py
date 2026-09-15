"""Register the C131 outcome from its artifact. See C131_PREREGISTRATION.md.

Mirrors accord/c130_reproducible_student/register_c130.py: delete the c131.*
prefix wholesale, re-add every scalar leaf from the committed artifact only,
pointing at artifact::dotted.path (accord/c127_scale_ladder/analyze_c127.py's
own inline registration convention, made a separate standalone step here to
match this task's requested layout: run_c131.py, analyze_c131.py,
register_c131.py as three separate stages).

    python3 accord/c131_scale_ladder_v2/register_c131.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/c131_scale_ladder_v2/c131.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "c131"

d = json.loads((ROOT / SRC).read_text())
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]

n_written = 0


def put(path_prefix, node):
    global n_written
    if isinstance(node, dict):
        for k, v in node.items():
            put(f"{path_prefix}.{k}" if path_prefix else k, v)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            put(f"{path_prefix}.{i}", v)
    elif node is None:
        return
    else:
        facts[f"{PREFIX}.{path_prefix}"] = collections.OrderedDict(
            [("artifact", f"{SRC}::{path_prefix}"), ("value", node)])
        n_written += 1


put("", d)
FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {n_written} {PREFIX}.* keys: attenuation_outcome="
      f"{d.get('attenuation_outcome')!r} prospective_check_outcome="
      f"{d.get('prospective_check_outcome')!r}")
