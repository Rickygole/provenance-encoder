#!/usr/bin/env python3
"""Per-class confirmed rate on the annotated pairs, and dictionary density per class.

Chapter 4 observes that 5 of the 12 confirmed contradictions are an anticoagulant
and all 3 rated most dangerous are, then says a class-restricted detector would
face a base rate this project did not measure, and that the dictionary covers
anticoagulants more densely than most classes. Both are countable from what is
already stored, so they are counted here rather than asserted.

Drug and class are dictionary terms, not note content. No note text, sentence or
identifier is read or emitted.

    python3 accord/clinical_core/class_base_rates.py
"""
from __future__ import annotations

import ast
import collections
import json
import os
import pathlib
import re

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
MIMIC = pathlib.Path(os.environ.get("MIMIC_LOCAL_ONLY_DIR",
                                   pathlib.Path.home() / "mimic_local_only"))
GOLD = MIMIC / "from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv"
MINING = ROOT / "accord/phase3_mining.py"
OUT = ROOT / "results/accord/clinical_core/class_base_rates.json"


def drug_classes():
    """Read DRUG_CLASSES out of the mining module without importing it; the
    module's own imports do not resolve from this directory."""
    src = MINING.read_text()
    m = re.search(r"^DRUG_CLASSES\s*=\s*(\{.*?^\})", src, re.S | re.M)
    assert m, "DRUG_CLASSES literal not found"
    return ast.literal_eval(m.group(1))


def main():
    classes = drug_classes()
    to_class = {d: c for c, ds in classes.items() for d in ds}

    g = pd.read_csv(GOLD)
    assert len(g) == 299, len(g)

    # Rule 2 stores the drug field as "agent_a vs agent_b"; both sides are
    # dictionary agents of the SAME class, so either maps to the same class.
    def classes_for(drug_field):
        parts = [p.strip().lower() for p in re.split(r"\bvs\b", str(drug_field))]
        return sorted({to_class[p] for p in parts if p in to_class})

    per = collections.defaultdict(lambda: {"n_pairs": 0, "n_confirmed": 0,
                                           "n_ambiguous": 0})
    unmapped = 0
    for _, r in g.iterrows():
        cs = classes_for(r["drug"])
        if not cs:
            unmapped += 1
            continue
        for c in cs:
            per[c]["n_pairs"] += 1
            if r["label_clean"] == "contradiction":
                per[c]["n_confirmed"] += 1
            elif r["label_clean"] == "ambiguous":
                per[c]["n_ambiguous"] += 1

    rows = {}
    for c, v in sorted(per.items()):
        rows[c] = dict(v)
        rows[c]["confirmed_rate"] = round(v["n_confirmed"] / v["n_pairs"], 4)
        rows[c]["n_dictionary_agents"] = len(classes[c])

    out = {
        "what": "Per-class pair counts and confirmed rate on the 299 annotated pairs, "
                "with the number of dictionary agents per class. Counts only; drug and "
                "class are dictionary terms and no note content is read.",
        "post_hoc_descriptive": True,
        "n_annotated": int(len(g)),
        "n_pairs_with_no_dictionary_class": int(unmapped),
        "n_classes_in_dictionary": len(classes),
        "dictionary_agents_per_class": {c: len(ds) for c, ds in sorted(classes.items())},
        "per_class": rows,
    }
    ac = rows.get("anticoagulants", {})
    out["anticoagulants"] = {
        "n_pairs": ac.get("n_pairs"), "n_confirmed": ac.get("n_confirmed"),
        "confirmed_rate": ac.get("confirmed_rate"),
        "n_dictionary_agents": ac.get("n_dictionary_agents"),
        "is_largest_dictionary_class": bool(
            ac.get("n_dictionary_agents") == max(len(d) for d in classes.values())),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    print(f"annotated {out['n_annotated']}, unmapped {unmapped}")
    print(f"{'class':26}{'pairs':>6}{'conf':>6}{'rate':>9}{'agents':>8}")
    for c, v in rows.items():
        print(f"{c:26}{v['n_pairs']:>6}{v['n_confirmed']:>6}"
              f"{v['confirmed_rate']:>9.4f}{v['n_dictionary_agents']:>8}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
