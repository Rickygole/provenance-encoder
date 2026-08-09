#!/usr/bin/env python3
"""Compute every reported number in this survey FROM coding_table.csv.

No count in coding_summary.json is typed by hand; all are derived here.

Usage:  python3 tabulate.py            (run from this directory)
Writes: coding_summary.json
"""
import csv
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "coding_table.csv")
SCREEN_PATH = os.path.join(HERE, "screening_log.csv")
OUT_PATH = os.path.join(HERE, "coding_summary.json")


def load(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def dist(rows, field):
    return dict(Counter(r[field] for r in rows))


def crosstab(rows, a, b):
    t = defaultdict(lambda: defaultdict(int))
    for r in rows:
        t[r[a]][r[b]] += 1
    return {k: dict(v) for k, v in t.items()}


def both(rows, sel_field):
    """audits_both = audits_model_on_pool == yes AND <sel_field> == yes."""
    return sum(1 for r in rows if r["audits_model_on_pool"] == "yes" and r[sel_field] == "yes")


def listfreq(rows, field):
    c = Counter()
    for r in rows:
        for x in (r[field] or "").split("|"):
            x = x.strip()
            if x:
                c[x] += 1
    return dict(c.most_common())


def main():
    rows = load(CSV_PATH)
    n = len(rows)
    filtered = [r for r in rows if r["cohort_is_filtered"] == "yes"]
    model_yes = [r for r in rows if r["audits_model_on_pool"] == "yes"]

    summary = {
        "n_included_papers": n,
        "year_range": [min(r["year"] for r in rows), max(r["year"] for r in rows)],
        "distributions": {
            "audits_model_on_pool": dist(rows, "audits_model_on_pool"),
            "cohort_is_filtered": dist(rows, "cohort_is_filtered"),
            "audits_selection_strict": dist(rows, "audits_selection"),
            "audits_selection_literal": dist(rows, "audits_selection_literal"),
            "audits_both_strict": dist(rows, "audits_both"),
            "audits_both_literal": dist(rows, "audits_both_literal"),
            "null_attribute_handling": dist(rows, "null_attribute_handling"),
            "year": dist(rows, "year"),
            "source": dist(rows, "source"),
        },
        "headline": {
            "audits_model_on_pool_yes": sum(1 for r in rows if r["audits_model_on_pool"] == "yes"),
            "cohort_is_filtered_yes": len(filtered),
            "audits_selection_yes_STRICT": sum(1 for r in rows if r["audits_selection"] == "yes"),
            "audits_selection_yes_LITERAL": sum(1 for r in rows if r["audits_selection_literal"] == "yes"),
            "audits_both_yes_STRICT": both(rows, "audits_selection"),
            "audits_both_yes_LITERAL": both(rows, "audits_selection_literal"),
            "model_yes_and_selection_no_STRICT": sum(
                1 for r in rows if r["audits_model_on_pool"] == "yes" and r["audits_selection"] == "no"),
            "model_yes_and_selection_no_LITERAL": sum(
                1 for r in rows if r["audits_model_on_pool"] == "yes" and r["audits_selection_literal"] == "no"),
        },
        "crosstab_model_x_selection_STRICT": crosstab(rows, "audits_model_on_pool", "audits_selection"),
        "crosstab_model_x_selection_LITERAL": crosstab(rows, "audits_model_on_pool", "audits_selection_literal"),
        "crosstab_filtered_x_selection_STRICT": crosstab(rows, "cohort_is_filtered", "audits_selection"),
        "restricted_to_filtered_cohorts": {
            "n": len(filtered),
            "audits_selection_STRICT": dist(filtered, "audits_selection"),
            "audits_selection_LITERAL": dist(filtered, "audits_selection_literal"),
            "audits_both_STRICT": both(filtered, "audits_selection"),
            "audits_both_LITERAL": both(filtered, "audits_selection_literal"),
        },
        "restricted_to_model_audits": {
            "n": len(model_yes),
            "audits_selection_STRICT": dist(model_yes, "audits_selection"),
            "audits_selection_LITERAL": dist(model_yes, "audits_selection_literal"),
        },
        "protected_attributes_frequency": listfreq(rows, "protected_attributes"),
        "fairness_metrics_frequency": listfreq(rows, "fairness_metrics"),
        "hard_cases": {
            "n_flagged_HARD_CASE": sum(1 for r in rows if r["notes"].startswith("HARD CASE")),
            "paper_ids": [r["paper_id"] for r in rows if r["notes"].startswith("HARD CASE")],
            "n_strict_no_but_literal_yes": sum(
                1 for r in rows if r["audits_selection"] == "no" and r["audits_selection_literal"] == "yes"),
            "paper_ids_strict_no_literal_yes": [
                r["paper_id"] for r in rows
                if r["audits_selection"] == "no" and r["audits_selection_literal"] == "yes"],
        },
    }

    if os.path.exists(SCREEN_PATH):
        s = load(SCREEN_PATH)
        summary["screening"] = {
            "n_records_screened_total": len(s),
            "by_disposition": dict(Counter(r["disposition"] for r in s)),
            "by_source": dict(Counter(r["source"] for r in s)),
            "n_included": sum(1 for r in s if r["disposition"] == "included"),
            "n_excluded_any_stage": sum(1 for r in s if r["disposition"].startswith("excluded")),
            "n_unretrievable": sum(1 for r in s if r["disposition"] == "unretrievable"),
            "n_fulltext_attempted": sum(
                1 for r in s if r["disposition"] in ("included", "excluded_at_fulltext", "unretrievable")),
        }

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
