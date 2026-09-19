"""Register the C133 outcome from its artifact.
See C133_PREREGISTRATION.md.

WRITTEN, NOT RUN when the pre-registration was committed: it has nothing to
read until c133_run.py's full (non-dry-run) mode has written
results/accord/c133_interval_tipping/c133.json. Mirrors
accord/c132_cluster_bootstrap/register_c132.py: delete this prefix wholesale,
re-add from the committed artifact only. The artifact holds aggregates only
and records a flip by its step number only, so nothing registered here is a
row-level value or an identifier.

    python3 accord/c133_interval_tipping/register_c133.py
"""
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accord.cloud_paths import assert_local_only  # noqa: E402

SRC = "results/accord/c133_interval_tipping/c133.json"
FACTS = assert_local_only(ROOT / "results/accord/VERIFIED_FACTS.json")
PREFIX = "c133"

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
    # Invalid run: the reason, and the floors at changed counts, which are arithmetic and touch no data.
    entries["outcome"] = "C"
    entries["abort_reason"] = d["abort_reason"]
    put("floors_neg_to_pos", d["floors_neg_to_pos"])
else:
    entries["outcome"] = d["decision"]["outcome"]
    for top in ("n_boot", "seed", "max_flips", "power_floor"):
        entries[top] = d[top]
    for section in ("validity_check", "task_a", "task_b", "decision", "floors_neg_to_pos"):
        put(section, d[section])

# On a valid run the top-level "outcome" entry is a copy of decision.outcome, and the artifact holds
# it only there, so its path must say so or a reader following the path finds nothing.
ALIAS = {} if d.get("outcome") == "C" else {"outcome": "decision.outcome"}
for field, value in entries.items():
    facts[f"{PREFIX}.{field}"] = collections.OrderedDict(
        [("artifact", f"{SRC}::{ALIAS.get(field, field)}"), ("value", value)]
    )

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: outcome={entries['outcome']} ({len(entries)} keys)")
