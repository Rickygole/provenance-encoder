"""Emit the LaTeX table body for the 12 pharmacist-confirmed contradictions.

Three reviews have said the clinical heart of this thesis appears once, in
Chapter 1, and never again, and that a clinician on the committee will ask to see
the 12 confirmed contradictions themselves. The characterisation already exists in
results/accord/clinical_core/confirmed_contradictions.json, produced by
accord/clinical_core/characterise_confirmed_contradictions.py. This script turns
it into a table body and writes it to an artifact. It edits no .tex file.

Publishability is re-asserted here rather than taken on trust from the upstream
artifact's own confirmation note:

  - every normalised drug name is asserted to be a member of
    accord/dictionaries.py ALL_DRUGS, the extraction dictionary;
  - every pharmacologic class is asserted to be a key of
    accord/phase3_mining.py DRUG_CLASSES or a value of
    accord/clinical_core/clinical_core_data.py CLASS_SUPPLEMENT, both fixed maps
    of hand-written pharmacologic class terms that no note contributes to;
  - the emitted body is asserted to contain no pair identifier and no digit
    string that could be an admission or subject identifier.

Nothing else from the source rows is emitted. Pair identifiers, per-side action
labels and pharmacist confidence bands stay in the artifact. The danger rating is
the pharmacist's ordinal annotation metadata, not note content.

    python3 accord/clinical_core/emit_contradiction_table.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "accord")):
    if p not in sys.path:
        sys.path.insert(0, p)

from accord.dictionaries import ALL_DRUGS  # noqa: E402
from phase3_mining import DRUG_CLASSES  # noqa: E402
from clinical_core_data import CLASS_SUPPLEMENT  # noqa: E402

SRC = ROOT / "results" / "accord" / "clinical_core" / "confirmed_contradictions.json"
OUT = ROOT / "results" / "accord" / "clinical_core" / "confirmed_contradictions_table.json"

ALLOWED_CLASSES = set(DRUG_CLASSES) | set(CLASS_SUPPLEMENT.values())

# the four generating rules, short enough for a table column
RULE_COLUMN = {1: "R1 direct action", 2: "R2 drug class",
               3: "R3 dosage", 4: "R4 discharge-side absence"}


def pretty(term: str) -> str:
    """A dictionary term as a table cell: underscores are the dictionary's, not prose."""
    return term.replace("_", " ")


def main():
    src = json.loads(SRC.read_text())
    rows = src["confirmed_contradictions"]
    assert len(rows) == 12, f"expected 12 confirmed contradictions, found {len(rows)}"
    assert src["gold_label_distribution"]["contradiction"] == 12, src["gold_label_distribution"]
    assert src["thesis_crosscheck_all_agree"] is True, src["thesis_crosscheck"]

    for r in rows:
        assert r["drug_normalised"] in ALL_DRUGS, (
            f"{r['drug_normalised']} is not in accord/dictionaries.py ALL_DRUGS and is "
            "therefore not confirmed as a dictionary term; nothing is emitted")
        assert r["pharmacologic_class"] in ALLOWED_CLASSES, (
            f"{r['pharmacologic_class']} is in neither DRUG_CLASSES nor CLASS_SUPPLEMENT")
        assert isinstance(r["danger_rating"], int) and 1 <= r["danger_rating"] <= 5, r
        assert r["rule"] in RULE_COLUMN, r

    ordered = sorted(rows, key=lambda r: (r["rule"], -r["danger_rating"],
                                          r["drug_normalised"]))

    body_lines = []
    for r in ordered:
        body_lines.append(
            f"{RULE_COLUMN[r['rule']]} & {pretty(r['drug_normalised'])} & "
            f"{pretty(r['pharmacologic_class'])} & {r['danger_rating']} & "
            f"{pretty(r['conflict_type_coarse'])} \\\\")
    body = "\n".join(body_lines)

    dsum = src["danger_summary"]
    per_rule = src["per_rule_positives"]
    dose = src["dose_conflict_caveat"]
    note = (
        "All 12 pairs the pharmacist confirmed as contradictions, out of "
        f"{src['n_gold_pairs']} annotated pairs "
        f"({src['gold_label_distribution']['not_contradiction']} not a contradiction, "
        f"{src['gold_label_distribution']['ambiguous']} ambiguous). "
        "Rule is the generating rule that put the pair in the annotation pool: "
        "R1 same drug with opposite actions across the two notes, R2 two unequal sets of "
        "agents of one pharmacologic class, one continued and one ceased, R3 the same "
        "drug at doses differing by "
        "more than 20 percent, R4 a drug held in the nursing note and absent from the "
        "discharge summary. "
        f"Rule counts are {per_rule['1']}, {per_rule['2']}, {per_rule['3']} and "
        f"{per_rule['4']}. "
        "Drug and class are dictionary terms: the normalised drug name is a member of "
        "the extraction dictionary and the class is a fixed hand-written class map of "
        "agents, not a standard terminology, so "
        "neither is note content. "
        "Danger is the pharmacist's ordinal rating, 1 the least and 5 the most "
        f"clinically dangerous; the 12 ratings have median {dsum['median']} and mean "
        f"{dsum['mean']}, and {dsum['n_at_or_above_4']} of 12 are at 4 or above. "
        "Conflict type is the coarse reading of the two sides; of the "
        f"{src['conflict_type_distribution_coarse']['class substitution']} class "
        "substitutions, "
        f"{src['conflict_type_distribution']['class substitution (agent swapped within class)']}"
        " swaps the agent inside the class and "
        f"{src['conflict_type_distribution']['class substitution (class-level instruction versus same agent)']}"
        " read as a class-level instruction set against the same agent, which the rule "
        "admits because it requires the two sides' agent sets to be unequal rather than "
        "disjoint, so they may overlap on one agent. "
        f"Of the {per_rule['3']} dose rows, "
        f"{dose['n_confirmed_rule3_above_100_pct']} carry a percent difference above "
        "100 because a bolus figure is paired against a maintenance figure; those are "
        "unit and role mismatches in the parsed dose, not dose errors of that size. "
        "No clinical note text and no patient or admission identifier appears in this "
        "table."
    )

    # nothing that could be an identifier may reach the body
    assert "PAIR_" not in body and "pair_id" not in body, body
    assert not re.search(r"\d{5,}", body), "a long digit string reached the table body"
    for r in rows:
        assert r["pair_id"] not in body, r["pair_id"]

    out = {
        "what": "LaTeX table body and note for the 12 pharmacist-confirmed contradictions",
        "source_artifact": str(SRC.relative_to(ROOT)),
        "addresses": "three reviews: the 12 confirmed contradictions appear once in "
                     "Chapter 1 and are never shown",
        "columns": ["Rule", "Drug", "Class", "Danger", "Conflict type"],
        "column_alignment": "llllll -> {@{}llrl@{}} with Danger right-aligned",
        "n_rows": len(ordered),
        "publishability": {
            "drug_names_all_in_ALL_DRUGS": True,
            "n_drugs_checked": len({r["drug_normalised"] for r in rows}),
            "classes_all_in_DRUG_CLASSES_or_CLASS_SUPPLEMENT": True,
            "n_classes_checked": len({r["pharmacologic_class"] for r in rows}),
            "no_pair_identifier_in_body": True,
            "no_note_text_in_body": True,
            "danger_rating_is": "the pharmacist's ordinal annotation metadata",
        },
        "distinct_drugs": sorted({r["drug_normalised"] for r in rows}),
        "distinct_classes": sorted({r["pharmacologic_class"] for r in rows}),
        "danger_summary": dsum,
        "per_rule_positives": per_rule,
        "conflict_type_distribution_coarse": src["conflict_type_distribution_coarse"],
        "pharmacologic_class_distribution": src["pharmacologic_class_distribution"],
        "latex_table_body": body,
        "latex_table_note": note,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(body)
    print()
    print(note)
    print()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
