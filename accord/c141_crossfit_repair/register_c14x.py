"""Register C140, C141 and C142 (prefixes c140, c141, c142; deletes and
re-adds) from their committed per-experiment JSON.

    python3 -I accord/c141_crossfit_repair/register_c14x.py

Section 4.5 of papers/jbi_weak_supervision/main.tex quotes numbers that reach
prose through these three artifacts, and the register held no key under any of
the three prefixes, so the register gate scanned nothing for them. Every leaf of
each artifact is registered, walked generically. A list of scalars (or of lists
of scalars) is one list value, not flattened into fake keys. Subtrees excluded
on purpose are listed in EXCLUDED and printed in the summary; none are excluded
at present.

This script deletes the three prefixes and nothing else: other generators run
concurrently in this tree, and the project working agreement records that a generator which deletes
more than its own prefix eats whatever else is parked there. Every other key is
snapshotted before the write and compared value by value after it; the write is
rolled back rather than left half applied if anything else moved. Floats are
registered at full precision, never rounded.
"""
import collections
import json
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
SOURCES = collections.OrderedDict([
    ("c140", "results/accord/c140_rival_adequacy/c140.json"),
    ("c141", "results/accord/c141_crossfit_repair/c141.json"),
    ("c142", "results/accord/c141_crossfit_repair/c142_delivery.json"),
])
PREFIXES = tuple(SOURCES)
# (prefix, json path) subtrees left out on purpose, printed in the summary.
EXCLUDED = ()

entries = collections.OrderedDict()
excluded_seen = []


def add(prefix, src, path, value):
    full = f"{prefix}.{path}"
    assert full not in entries, f"duplicate register key {full}"
    entries[full] = collections.OrderedDict(
        [("artifact", f"{src}::{path}"), ("value", value)]
    )


def is_scalar_list(x):
    return isinstance(x, list) and all(
        not isinstance(i, dict) and (not isinstance(i, list) or is_scalar_list(i))
        for i in x
    )


def walk(prefix, src, node, path):
    if (prefix, path) in EXCLUDED:
        excluded_seen.append(f"{prefix}:{path}")
        return
    if isinstance(node, dict):
        for k, v in node.items():
            walk(prefix, src, v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list) and not is_scalar_list(node):
        for i, v in enumerate(node):
            walk(prefix, src, v, f"{path}.{i}")
    else:
        add(prefix, src, path, node)


for prefix, src in SOURCES.items():
    walk(prefix, src, json.loads((ROOT / src).read_text()), "")
assert len(excluded_seen) == len(EXCLUDED), "an excluded path matched nothing"

# ------------------------------------------------------------------ the write
# Re-read immediately before writing: another generator may have written since
# this script started, and holding a stale copy across the work above is how a
# concurrent prefix gets eaten.
original_text = FACTS.read_text()
facts = json.loads(original_text, object_pairs_hook=collections.OrderedDict)
mine = [k for k in facts if k.split(".")[0] in PREFIXES]
mine_set = set(mine)
others_before = collections.OrderedDict(
    (k, v) for k, v in facts.items() if k not in mine_set)
for k in mine:
    del facts[k]
facts.update(entries)

assert len(facts) == len(others_before) + len(entries), (
    f"key accounting: {len(facts)} != {len(others_before)} others + "
    f"{len(entries)} new")
moved = [k for k, v in others_before.items() if k not in facts or facts[k] != v]
if moved:
    sys.exit(f"this script would have changed a foreign key: {moved[0]}; "
             "wrote nothing")

fd, tmp = tempfile.mkstemp(dir=str(FACTS.parent), suffix=".tmp")
try:
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(facts, indent=1) + "\n")
    os.replace(tmp, FACTS)
except BaseException:
    if os.path.exists(tmp):
        os.unlink(tmp)
    raise

back = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
if back != facts:
    FACTS.write_text(original_text)
    sys.exit("register changed under this script; restored it and wrote "
             "nothing")

counts = ", ".join(
    f"{p}={sum(1 for k in facts if k.split('.')[0] == p)}" for p in PREFIXES)
print(f"registered {counts}; excluded subtrees: "
      f"{', '.join(excluded_seen) if excluded_seen else 'none'}; "
      f"{len(facts)} keys total, was {len(others_before) + len(mine)}")
