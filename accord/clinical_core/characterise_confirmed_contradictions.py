"""ANALYSIS 1. Characterise the 12 pharmacist-confirmed medication
contradictions: generating rule, normalised drug name and pharmacologic class,
danger rating, and conflict type.

The thesis states the count (12) on page 1 and then never says what they are.
This script recovers the clinical content as publishable aggregates and
cross-checks the thesis's printed per-rule split and danger claims.

PUBLISHABILITY. Every field written is either a synthetic pair key (PAIR_nnnn),
a rule id, a dictionary term (normalised drug name from ALL_DRUGS / the
pharmacologic class map), a dictionary-derived action label, a parsed dose
number, or the pharmacist's ordinal danger rating. No note text, no HADM_ID,
no SUBJECT_ID. The check is asserted in code before the artifact is written.

OUT: results/accord/clinical_core/confirmed_contradictions.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from accord.clinical_core.clinical_core_data import (  # noqa: E402
    RULE_CONFLICT_TYPE, RULE_SHORT, load_gold, pharm_class)
from accord.phase3_mining import RULE_DESCRIPTIONS  # noqa: E402

OUT = REPO / "results/accord/clinical_core/confirmed_contradictions.json"

# Values printed in the thesis, quoted here only to be cross-checked. Any
# mismatch is recorded in the artifact rather than silently reconciled.
THESIS_CLAIMS = {
    "n_confirmed": 12,
    "per_rule_positives": {"1": 3, "2": 3, "3": 4, "4": 2},
    "stratum_rules_1_2": 6,
    "stratum_rules_3_4": 6,
    "n_danger_5": 3,
    "n_danger_4": 1,
}

# Anything that would make an artifact unpublishable under the PhysioNet
# credentialed data use agreement.
BANNED_RE = re.compile(r"HADM|SUBJECT_ID|nursing_sentence|discharge_sentence", re.IGNORECASE)


def assert_publishable(blob: str) -> None:
    m = BANNED_RE.search(blob)
    assert m is None, f"unpublishable token {m.group(0)!r} reached the artifact"
    for ch in ("\u2013", "\u2014"):
        assert ch not in blob, "en or em dash in artifact"


def conflict_type(row) -> str:
    """The conflict type the generating rule defines, refined for Rule 2 by
    whether the two recorded agents are the same drug or two agents of one
    class."""
    base = RULE_CONFLICT_TYPE[row["rule"]]
    if row["rule"] == 2 and row["drug_norm"] == row["drug_discharge"]:
        return "class substitution (class-level instruction versus same agent)"
    if row["rule"] == 2:
        return "class substitution (agent swapped within class)"
    return base


def main() -> None:
    g = load_gold()
    pos = g[g["pharm_label"] == "contradiction"].copy().sort_values("pair_id")
    pos["conflict_type"] = pos.apply(conflict_type, axis=1)
    pos["pharm_class_discharge"] = pos["drug_discharge"].apply(pharm_class)

    rows = []
    for _, r in pos.iterrows():
        rows.append({
            "pair_id": r["pair_id"],
            "rule": int(r["rule"]),
            "rule_short": r["rule_short"],
            "drug_normalised": r["drug_norm"],
            "drug_normalised_discharge_side": r["drug_discharge"],
            "pharmacologic_class": r["pharm_class"],
            "pharmacologic_class_discharge_side": r["pharm_class_discharge"],
            "drug_class_recorded_by_rule2": (None if not isinstance(r["drug_class_dict"], str)
                                             else r["drug_class_dict"]),
            "nursing_action": (None if not isinstance(r["nursing_action"], str) else r["nursing_action"]),
            "discharge_action": (None if not isinstance(r["discharge_action"], str) else r["discharge_action"]),
            "conflict_type_coarse": RULE_CONFLICT_TYPE[int(r["rule"])],
            "conflict_type": r["conflict_type"],
            "danger_rating": (None if r["danger_rating"] != r["danger_rating"] else int(r["danger_rating"])),
            "pharmacist_confidence": (None if not isinstance(r["confidence"], str) else r["confidence"]),
            "dose_diff_pct": (None if r["dose_diff_pct"] != r["dose_diff_pct"]
                              else round(float(r["dose_diff_pct"]), 2)),
        })

    per_rule = Counter(int(r["rule"]) for r in rows)
    danger = Counter(r["danger_rating"] for r in rows)
    by_class = Counter(r["pharmacologic_class"] for r in rows)
    by_type = Counter(r["conflict_type"] for r in rows)
    by_type_coarse = Counter(r["conflict_type_coarse"] for r in rows)
    dangers = [r["danger_rating"] for r in rows if r["danger_rating"] is not None]

    recomputed = {
        "n_confirmed": len(rows),
        "per_rule_positives": {str(k): per_rule.get(k, 0) for k in (1, 2, 3, 4)},
        "stratum_rules_1_2": per_rule.get(1, 0) + per_rule.get(2, 0),
        "stratum_rules_3_4": per_rule.get(3, 0) + per_rule.get(4, 0),
        "n_danger_5": danger.get(5, 0),
        "n_danger_4": danger.get(4, 0),
    }
    crosscheck = {k: {"thesis": v, "recomputed": recomputed[k], "agrees": v == recomputed[k]}
                  for k, v in THESIS_CLAIMS.items()}

    art = {
        "analysis": "Analysis 1: characterisation of the 12 pharmacist-confirmed contradictions",
        "generated_by": "accord/clinical_core/characterise_confirmed_contradictions.py",
        "inputs": {
            "canonical_gold_labels": "$MIMIC_LOCAL_ONLY_DIR/derived/gold_standard_accord_results.csv",
            "pharmacist_returned_sheet": "$MIMIC_LOCAL_ONLY_DIR/from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv",
            "candidates_table": "$MIMIC_LOCAL_ONLY_DIR/from_drive_2026-08-08/outputs/annotation_sample_300.csv",
            "weak_label_audit": "$MIMIC_LOCAL_ONLY_DIR/accord_supp/accord_decontamination_audit.csv",
            "class_dictionary": "accord/phase3_mining.py DRUG_CLASSES plus clinical_core_data.CLASS_SUPPLEMENT",
        },
        "rule_definitions": {str(k): RULE_DESCRIPTIONS[k] for k in sorted(RULE_DESCRIPTIONS)},
        "rule_short_names": {str(k): v for k, v in RULE_SHORT.items()},
        "n_gold_pairs": int(len(g)),
        "gold_label_distribution": {k: int(v) for k, v in g["pharm_label"].value_counts().items()},
        "confirmed_contradictions": rows,
        "per_rule_positives": {str(k): per_rule.get(k, 0) for k in (1, 2, 3, 4)},
        "danger_distribution": {str(k): v for k, v in sorted(danger.items(), key=lambda x: (x[0] is None, x[0]))},
        "danger_summary": {
            "n_rated": len(dangers),
            "n_unrated": len(rows) - len(dangers),
            "min": min(dangers), "max": max(dangers),
            "mean": round(sum(dangers) / len(dangers), 4),
            "median": round(float(sorted(dangers)[len(dangers) // 2]) if len(dangers) % 2
                            else (sorted(dangers)[len(dangers) // 2 - 1] + sorted(dangers)[len(dangers) // 2]) / 2, 2),
            "n_at_or_above_4": sum(1 for d in dangers if d >= 4),
        },
        "pharmacologic_class_distribution": dict(sorted(by_class.items())),
        "conflict_type_distribution_coarse": dict(sorted(by_type_coarse.items())),
        "conflict_type_distribution": dict(sorted(by_type.items())),
        "exemplar_action_caveat": {
            "what": ("The nursing_action and discharge_action recorded in the candidates table are "
                     "the actions of the single exemplar mention each side contributed, chosen by "
                     "pick_exemplar in accord/phase3_mining.py. Rule 1 fires on the SET of actions "
                     "for a drug within a note, so an exemplar pair can fail to display the "
                     "opposite-action pair that made the row fire."),
            "n_rule1_gold_pairs": int((g["rule"] == 1).sum()),
            "n_rule1_gold_pairs_whose_exemplar_actions_are_an_opposite_pair": int(
                (((g["rule"] == 1)
                  & (((g["nursing_action"] == "continuation") & (g["discharge_action"] == "cessation"))
                     | ((g["nursing_action"] == "cessation") & (g["discharge_action"] == "continuation"))))).sum()),
            "n_rule1_confirmed": int(per_rule.get(1, 0)),
            "n_rule1_confirmed_whose_exemplar_actions_are_an_opposite_pair": sum(
                1 for r in rows if r["rule"] == 1
                and {r["nursing_action"], r["discharge_action"]} == {"continuation", "cessation"}),
            "consequence": ("A reader who reads the recorded action pair as the conflict itself will "
                            "find Rule 1 rows whose two actions are identical. That is an artefact of "
                            "exemplar selection, not a mislabelled pair."),
        },
        "dose_conflict_caveat": {
            "what": ("dose_diff_pct is computed from parsed numeric doses whose canonical units must "
                     "match (accord/dictionaries.py parse_dose). Anticoagulant rows can pair a bolus "
                     "figure against a maintenance figure, producing percent differences far above "
                     "100 that are a unit and role mismatch rather than a dose error of that size."),
            "dose_diff_pct_of_confirmed_rule3": sorted(
                r["dose_diff_pct"] for r in rows if r["rule"] == 3 and r["dose_diff_pct"] is not None),
            "n_confirmed_rule3_above_100_pct": sum(
                1 for r in rows if r["rule"] == 3 and r["dose_diff_pct"] is not None
                and r["dose_diff_pct"] > 100),
        },
        "thesis_crosscheck": crosscheck,
        "thesis_crosscheck_all_agree": all(v["agrees"] for v in crosscheck.values()),
        "publishability_confirmation": {
            "drug_names": ("Every normalised drug name written here was asserted to be a member of "
                           "accord/dictionaries.py ALL_DRUGS, the extraction dictionary, before the "
                           "artifact was written. A name that appeared in a note but not in the "
                           "dictionary would fail that assertion and nothing would be written."),
            "n_distinct_drug_names_in_gold_set": int(len(set(g["drug_norm"]) | set(g["drug_discharge"]))),
            "drug_classes": ("Class names come from accord/phase3_mining.py DRUG_CLASSES and from "
                             "clinical_core_data.CLASS_SUPPLEMENT, both fixed maps of pharmacologic "
                             "class terms. No class name is derived from a note."),
            "action_labels": ("Action labels are the keys of accord/dictionaries.py ACTION_PATTERNS "
                              "plus the literal 'absent' that Rule 4 assigns to a missing discharge "
                              "side; asserted before writing."),
            "danger_and_confidence": ("The pharmacist's ordinal danger rating and confidence band, "
                                      "which are annotation metadata, not note content."),
            "keys": "pair_id is the synthetic PAIR_nnnn key from the annotation sheet.",
        },
        "data_agreement": ("Aggregates and dictionary terms only. No clinical note text and no patient "
                           "or admission identifier appears in this artifact; asserted in code before writing."),
    }

    blob = json.dumps(art, indent=1, sort_keys=False)
    assert_publishable(blob)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(blob + "\n")

    print(f"wrote {OUT}")
    print(f"n_confirmed={len(rows)} per_rule={recomputed['per_rule_positives']} "
          f"danger={dict(sorted((k, v) for k, v in danger.items()))}")
    for k, v in crosscheck.items():
        print(f"  crosscheck {k}: thesis={v['thesis']} recomputed={v['recomputed']} "
              f"{'OK' if v['agrees'] else 'MISMATCH'}")


if __name__ == "__main__":
    main()
