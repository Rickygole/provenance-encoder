#!/usr/bin/env python3
"""Register the Mann-Whitney variants for the subject-overlap run on G_253.

The thesis printed 0.6418 on a row labelled "same 253 pairs". That value is the
G_299 comparison (12 positives against 287 negatives). No variant of the 12
against 241 test produces it. The four other p values in that column reproduce
under the asymptotic method with continuity correction, so this registers that
method's value for the same row, alongside the exact and no-correction variants
so a reader can see which convention is in use.

    python3 accord/mw_variants/register_mwvariants.py
"""
import collections, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accord.cloud_paths import assert_local_only  # noqa: E402

SRC = "results/accord/audit_c137/a4_mw_variants.json"
FACTS = assert_local_only(ROOT / "results/accord/VERIFIED_FACTS.json")
PREFIX = "mwvariants"

src = json.loads((ROOT / SRC).read_text())
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]

keep = {
    "subject_overlap_g253.asymptotic_continuity_corrected": "mw_asymp_cc_true",
    "subject_overlap_g253.asymptotic_no_correction": "mw_asymp_cc_false",
    "subject_overlap_g253.exact": "mw_exact",
    "subject_overlap_g253.ranksums": "ranksums",
}
n = 0
for field, key in keep.items():
    facts[f"{PREFIX}.{field}"] = collections.OrderedDict(
        [("value", src[key]), ("artifact", f"{SRC}::{key}")])
    n += 1
FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: {n} keys")
