"""How many of the 19 double-coded overlap papers did the primary coder mark
strict-positive (audits_selection == yes)?

This is a join, not a new judgement call: the second coder's 19 paper_ids
(results/accord/survey/second_coder_table.csv) are looked up against the
primary coder's full 75-paper table (results/accord/survey/coding_table.csv)
on the strict field, audits_selection. It exists to settle a question the
thesis previously left unanswered: since the strict cell's chance-corrected
kappa is undefined on the 19-paper overlap, is that because the primary
coder's own marginal on the overlap is degenerate (all one value), or because
the record simply does not say? This script answers it directly rather than
asserting either.

Run:  python3 accord/stats/survey_overlap_strict_count.py
Writes: results/accord/survey/overlap_strict_count.json
"""

import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SURVEY = ROOT / "results" / "accord" / "survey"
OUT = SURVEY / "overlap_strict_count.json"


def load_table(path):
    with open(path) as f:
        return {row["paper_id"]: row for row in csv.DictReader(f)}


def main():
    primary = load_table(SURVEY / "coding_table.csv")
    overlap_ids = list(load_table(SURVEY / "second_coder_table.csv").keys())

    missing = [pid for pid in overlap_ids if pid not in primary]
    if missing:
        raise SystemExit(f"overlap ids not found in primary table: {missing}")

    strict_yes = [pid for pid in overlap_ids
                  if primary[pid]["audits_selection"] == "yes"]

    out = {
        "n_overlap_papers": len(overlap_ids),
        "n_overlap_ids_missing_from_primary_table": len(missing),
        "n_primary_coder_strict_positive_on_overlap": len(strict_yes),
        "strict_positive_paper_ids": strict_yes,
        "note": ("Primary coder's audits_selection (strict field) looked up for "
                 "each of the 19 second-coder overlap paper_ids. Determinable "
                 "directly from coding_table.csv and second_coder_table.csv; "
                 "no imputation."),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
