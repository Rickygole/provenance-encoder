"""Register the C132 outcome from its artifact.
See C132_PREREGISTRATION.md.

WRITTEN, NOT RUN when the pre-registration was committed: it has nothing to
read until c132_run.py's full (non-dry-run) mode has written
results/accord/c132_cluster_bootstrap/c132.json. Mirrors
accord/c130_reproducible_student/register_c130.py and
accord/c126_teacher_student/c126_run.py's register step: delete this prefix
wholesale, re-add from the committed artifact only. The artifact holds
aggregates only, so nothing registered here is a row-level value.

    python3 accord/c132_cluster_bootstrap/register_c132.py
"""
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accord.cloud_paths import assert_local_only  # noqa: E402

SRC = "results/accord/c132_cluster_bootstrap/c132.json"
FACTS = assert_local_only(ROOT / "results/accord/VERIFIED_FACTS.json")
PREFIX = "c132"

d = json.loads((ROOT / SRC).read_text(), object_pairs_hook=collections.OrderedDict)
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]

entries = collections.OrderedDict()


def put(path, obj):
    for k, v in obj.items():
        field = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            put(field, v)
        else:
            entries[field] = v


if d.get("outcome") == "C":
    # Invalid run: the reason, and Task C, which is arithmetic on fixed counts and touches no data.
    entries["outcome"] = "C"
    entries["abort_reason"] = d["abort_reason"]
    put("task_c", d["task_c"])
else:
    entries["outcome"] = d["decision"]["outcome"]
    for top in ("n_boot", "cluster_seed", "row_check_seed", "power_floor"):
        entries[top] = d[top]
    for section in ("task_a", "task_b", "decision", "task_c", "task_d"):
        put(section, d[section])

for field, value in entries.items():
    facts[f"{PREFIX}.{field}"] = collections.OrderedDict(
        [("artifact", f"{SRC}::{field}"), ("value", value)]
    )

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: outcome={entries['outcome']} ({len(entries)} keys)")
