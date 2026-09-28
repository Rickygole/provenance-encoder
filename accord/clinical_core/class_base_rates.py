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
SUPPLEMENT = ROOT / "accord/clinical_core/clinical_core_data.py"
OUT = ROOT / "results/accord/clinical_core/class_base_rates.json"


def drug_classes():
    """Read DRUG_CLASSES out of the mining module without importing it; the
    module's own imports do not resolve from this directory."""
    src = MINING.read_text()
    m = re.search(r"^DRUG_CLASSES\s*=\s*(\{.*?^\})", src, re.S | re.M)
    assert m, "DRUG_CLASSES literal not found"
    return ast.literal_eval(m.group(1))


def class_supplement():
    """The supplementary agent-to-class map. The project resolves a class as
    DRUG_TO_CLASS first and CLASS_SUPPLEMENT second (clinical_core_data.py), so
    counting on DRUG_CLASSES alone leaves aspirin and potassium unclassified and
    undercounts the confirmed contradictions by two."""
    src = SUPPLEMENT.read_text()
    m = re.search(r"^CLASS_SUPPLEMENT\s*=\s*(\{.*?^\})", src, re.S | re.M)
    assert m, "CLASS_SUPPLEMENT literal not found"
    return ast.literal_eval(m.group(1))


Z = 1.959963984540054


def wilson(k, n, z=Z):
    """Wilson score interval, the same closed form as c129_run.py."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, (center - half) / denom), min(1.0, (center + half) / denom))


def main():
    classes = drug_classes()
    to_class = {d: c for c, ds in classes.items() for d in ds}
    supp = class_supplement()
    # Project order: DRUG_TO_CLASS first, CLASS_SUPPLEMENT second.
    resolve = dict(supp)
    resolve.update(to_class)
    agents_per_class = collections.Counter()
    for c, ds in classes.items():
        agents_per_class[c] += len(ds)
    for d, c in supp.items():
        if d not in to_class:
            agents_per_class[c] += 1

    g = pd.read_csv(GOLD)
    assert len(g) == 299, len(g)

    # Rule 2 stores the drug field as "agent_a vs agent_b"; both sides are
    # dictionary agents of the SAME class, so either maps to the same class.
    def classes_for(drug_field):
        parts = [p.strip().lower() for p in re.split(r"\bvs\b", str(drug_field))]
        return sorted({resolve[p] for p in parts if p in resolve})

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
        rows[c]["n_dictionary_agents"] = int(agents_per_class[c])
        lo, hi = wilson(v["n_confirmed"], v["n_pairs"])
        rows[c]["confirmed_wilson95"] = [round(lo, 4), round(hi, 4)]

    out = {
        "what": "Per-class pair counts and confirmed rate on the 299 annotated pairs, "
                "with the number of dictionary agents per class. Counts only; drug and "
                "class are dictionary terms and no note content is read.",
        "post_hoc_descriptive": True,
        "n_annotated": int(len(g)),
        "n_pairs_with_no_dictionary_class": int(unmapped),
        "n_classes_in_dictionary": len(classes),
        "dictionary_agents_per_class": dict(sorted(agents_per_class.items())),
        "n_classes_in_dictionary_total": len(agents_per_class),
        "per_class": rows,
    }
    ac = rows.get("anticoagulants", {})
    out["anticoagulants"] = {
        "n_pairs": ac.get("n_pairs"), "n_confirmed": ac.get("n_confirmed"),
        "confirmed_rate": ac.get("confirmed_rate"),
        "n_dictionary_agents": ac.get("n_dictionary_agents"),
        "confirmed_wilson95": ac.get("confirmed_wilson95"),
        "selected_post_hoc_from_these_rows": True,
        "is_largest_dictionary_class": bool(
            ac.get("n_dictionary_agents") == max(agents_per_class.values())),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    print(f"annotated {out['n_annotated']}, unmapped {unmapped}")
    print(f"{'class':26}{'pairs':>6}{'conf':>6}{'rate':>9}{'agents':>8}")
    for c, v in rows.items():
        print(f"{c:26}{v['n_pairs']:>6}{v['n_confirmed']:>6}"
              f"{v['confirmed_rate']:>9.4f}{v['n_dictionary_agents']:>8}"
              f"   [{v['confirmed_wilson95'][0]:.4f}, {v['confirmed_wilson95'][1]:.4f}]")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
