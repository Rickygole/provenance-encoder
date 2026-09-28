#!/usr/bin/env python3
"""What did the pharmacist cite for the Rule 3 rows whose parsed dose gap exceeds 100 percent?

Table 6's note records that 3 of the 4 confirmed Rule 3 rows carry a percent
difference above 100 because a bolus figure is paired against a maintenance
figure, which invites the question of what was confirmed as a contradiction if
the dose gap itself is a parse artifact. The answer is in the pharmacist's
free-text justification.

This script NEVER emits that free text, and never emits note sentences or
identifiers. It reports only a tabulation against a fixed clinical vocabulary
fixed in this file before the column was read, plus counts. Output is an
aggregate, which is what the data use agreement permits to leave the machine.

    python3 accord/clinical_core/dose_justification_profile.py
"""
from __future__ import annotations

import json
import os
import pathlib
import re

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
MIMIC = pathlib.Path(os.environ.get("MIMIC_LOCAL_ONLY_DIR",
                                   pathlib.Path.home() / "mimic_local_only"))
GOLD = MIMIC / "from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv"
DERIVED = MIMIC / "derived/gold_standard_accord_results.csv"
OUT = ROOT / "results/accord/clinical_core/dose_justification_profile.json"

# Fixed before the column was read. Each theme is the clinical reason a
# pharmacist could give for calling a dose row a contradiction.
THEMES = {
    "route_or_form_change": r"\b(iv|intravenous|oral|po|subcut|subq|sc|drip|infusion|bolus)\b",
    "rate_versus_total": r"\b(rate|per hour|hourly|units/h|/h\b|continuous|maintenance)\b",
    "monitoring_or_titration": r"\b(titrat|monitor|inr|aptt|ptt|level|trough|goal)\b",
    "stopped_or_held": r"\b(held|hold|stopped|discontinu|d/c|ceased)\b",
    "restart_or_resume": r"\b(restart|resum|reinitiat)\b",
    "dose_magnitude": r"\b(dose|dosage|mg|mcg|units|increase|decrease|higher|lower)\b",
    "bleeding_or_harm_risk": r"\b(bleed|hemorrhag|haemorrhag|risk|harm|unsafe|danger)\b",
    "duplicate_therapy": r"\b(duplicat|overlap|both|two agents|bridge|bridging)\b",
}


def main():
    g = pd.read_csv(GOLD)
    dg = pd.read_csv(DERIVED)
    rule_by_pair = dict(zip(dg["pair_id"], dg["rule"]))

    g = g[g["label_clean"] == "contradiction"].copy()
    g["rule_joined"] = g["pair_id"].map(rule_by_pair)
    r3 = g[g["rule_joined"] == 3]

    rows = []
    for _, r in r3.iterrows():
        j = str(r.get("justification") or "")
        low = j.lower()
        rows.append({
            "danger_rating": None if pd.isna(r.get("danger_rating")) else int(r["danger_rating"]),
            "drug_class": str(r.get("drug_class") or ""),
            "justification_present": bool(j.strip()),
            "justification_char_len": len(j.strip()),
            "justification_word_count": len(j.split()),
            "themes_matched": sorted(k for k, pat in THEMES.items()
                                    if re.search(pat, low)),
        })

    out = {
        "what": "Theme tabulation of the pharmacist's free-text justification for the "
                "confirmed Rule 3 dose rows. No free text, note sentence or identifier "
                "is emitted; the vocabulary was fixed before the column was read.",
        "post_hoc_descriptive": True,
        "themes_vocabulary": {k: v for k, v in THEMES.items()},
        "n_confirmed_rule3": len(rows),
        "n_with_a_justification": sum(1 for r in rows if r["justification_present"]),
        "rows": sorted(rows, key=lambda r: (-(r["danger_rating"] or 0),
                                            r["drug_class"])),
        "theme_counts": {k: sum(1 for r in rows if k in r["themes_matched"])
                         for k in THEMES},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"confirmed Rule 3 rows: {out['n_confirmed_rule3']}, "
          f"with a justification: {out['n_with_a_justification']}")
    print("theme counts:")
    for k, v in out["theme_counts"].items():
        print(f"   {k:26} {v}")
    print("per row (no text):")
    for r in out["rows"]:
        print(f"   danger {r['danger_rating']}  {r['drug_class']:24} "
              f"words={r['justification_word_count']:3}  {', '.join(r['themes_matched'])}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
