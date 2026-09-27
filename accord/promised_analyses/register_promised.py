"""Register the four promised analyses from their committed artifacts.

Four prefixes, one per artifact, each deleted wholesale and re-added from the
artifact only, per the project's register convention:

    cicchetti.*      results/accord/promised_analyses/cicchetti_ppos_pneg.json
    bootdef.*        results/accord/promised_analyses/bootstrap_definitions.json
    tipstrat.*       results/accord/promised_analyses/stratified_tipping.json
    testfamrecount.* results/accord/promised_analyses/uncorrected_family_recount.json

Modelled on accord/c132_cluster_bootstrap/register_c132.py. Every value is read
out of the artifact; nothing is typed here. The walk is exhaustive rather than a
hand-picked list of fields, because a hand-picked list is how a number reaches
prose that the register does not hold: each scalar leaf of each artifact becomes
one key whose artifact pointer is the leaf's own JSON path, with list elements
addressed by index.

Register keys cannot carry the "::" the cell keys of these artifacts use, since
"::" separates the file from the pointer in an artifact string, so a cell key's
"::" becomes a "." in the register key and the pointer keeps the artifact's own
spelling. Collisions are checked, not assumed.

    python3 accord/promised_analyses/register_promised.py
"""
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accord.cloud_paths import assert_local_only  # noqa: E402

FACTS = assert_local_only(ROOT / "results/accord/VERIFIED_FACTS.json")

DIR = "results/accord/promised_analyses"
SOURCES = collections.OrderedDict([
    ("cicchetti", f"{DIR}/cicchetti_ppos_pneg.json"),
    ("bootdef", f"{DIR}/bootstrap_definitions.json"),
    ("tipstrat", f"{DIR}/stratified_tipping.json"),
    ("testfamrecount", f"{DIR}/uncorrected_family_recount.json"),
])


def key_segment(name: str) -> str:
    """A dict key as it appears in a register key.

    "::" is the cell-key separator these artifacts use and the artifact-string
    separator the register uses, so it becomes the register's own "." here. Any
    remaining ":" is a model tag's separator and becomes "_", the spelling the
    rest of this tree already uses for model tags (see register_c135.py).
    """
    return name.replace("::", ".").replace(":", "_")


def flatten(obj, key_path, ptr_path, out):
    """Append (register key, artifact pointer, value) for each scalar leaf."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, key_path + [key_segment(k)], ptr_path + [k], out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flatten(v, key_path + [str(i)], ptr_path + [str(i)], out)
    else:
        out.append((".".join(key_path), ".".join(ptr_path), obj))


facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] in SOURCES]:
    del facts[k]

per_prefix = collections.OrderedDict()
for prefix, src in SOURCES.items():
    path = ROOT / src
    if not path.exists():
        sys.exit(f"missing artifact {src}; refusing to register {prefix}")
    d = json.loads(path.read_text(), object_pairs_hook=collections.OrderedDict)
    leaves = []
    flatten(d, [], [], leaves)

    seen = collections.Counter(k for k, _, _ in leaves)
    dupes = sorted(k for k, n in seen.items() if n > 1)
    if dupes:
        sys.exit(f"{prefix}: register keys collide after sanitising: {dupes}")

    for field, pointer, value in leaves:
        facts[f"{prefix}.{field}"] = collections.OrderedDict(
            [("artifact", f"{src}::{pointer}"), ("value", value)]
        )
    per_prefix[prefix] = len(leaves)

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print("registered " + ", ".join(f"{p}={n} keys" for p, n in per_prefix.items()))
