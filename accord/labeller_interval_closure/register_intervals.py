"""Insert the labeller interval closure's facts into the register.

register_manifest.py GENERATES the hand-off file
results/accord/labeller_interval_closure/REGISTER_THESE.json; it deliberately
does not touch the register. This script is the insert half. It reads that
hand-off and nothing else, so every value here still traces to
intervals_17.json or permuted_label_control.json through the artifact pointer
the manifest recorded. No value is typed here.

The hand-off's top-level "do_not_write_verified_facts_directly" key is a guard
string addressed to a reader, not data, and is never registered. The prefixes
to delete wholesale are read from the hand-off's own
"prefixes_to_delete_wholesale_on_regeneration" list, so the delete and the
re-add cannot drift apart.

Regenerate the hand-off first if the artifacts moved:

    python3 accord/labeller_interval_closure/register_manifest.py
    python3 accord/labeller_interval_closure/register_intervals.py
"""
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accord.cloud_paths import assert_local_only  # noqa: E402

SRC = "results/accord/labeller_interval_closure/REGISTER_THESE.json"
FACTS = assert_local_only(ROOT / "results/accord/VERIFIED_FACTS.json")

d = json.loads((ROOT / SRC).read_text(), object_pairs_hook=collections.OrderedDict)
handoff = d["keys"]

# The prefixes this generator owns, as the hand-off declares them. Stored there
# with the trailing dot; the register convention keys off the first path segment.
PREFIXES = [p.rstrip(".") for p in d["prefixes_to_delete_wholesale_on_regeneration"]]
if not PREFIXES:
    sys.exit("hand-off declares no prefixes to delete; refusing to write")

# Every hand-off key must fall under a declared prefix, or the wholesale delete
# would not cover what this script adds and a stale key could survive a rerun.
stray = sorted({k.split(".")[0] for k in handoff} - set(PREFIXES))
if stray:
    sys.exit(f"hand-off keys outside the declared prefixes: {stray}")

facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] in PREFIXES]:
    del facts[k]

for key, entry in handoff.items():
    facts[key] = collections.OrderedDict(
        [("artifact", entry["artifact"]), ("value", entry["value"])]
    )

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
counts = ", ".join(
    f"{p}={sum(1 for k in facts if k.split('.')[0] == p)}" for p in PREFIXES
)
print(f"registered from {SRC}: {len(handoff)} keys ({counts})")
