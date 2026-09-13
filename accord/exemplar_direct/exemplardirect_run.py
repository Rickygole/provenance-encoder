"""C106: direct exemplar check on the 165 Rule 1 / Rule 2 gold candidates.
See C106_PREREGISTRATION.md.

Applies accord/phase2_medication.py::expand_text and
::extract_mentions_from_doc, and accord/dictionaries.py's ABBREV/VARIANTS
normalization (all imported, none modified) to the DISPLAYED nursing_sentence
and discharge_sentence text in the annotation sheet, one sentence at a time,
to decide whether each displayed pair carries the opposite-action conflict its
rule asserts.

Local only. All stdout/stderr produced while the credentialed CSV is open is
redirected to a file under ~/mimic_local_only/logs. Only aggregate counts and
categorical action labels (continuation / cessation / initiation /
dose_modification / unclassified -- a five-value fixed vocabulary, not note
text) ever reach the written JSON or real stdout. No pair_id, no sentence
text, and no drug-class join key leaves this process.

    python3 accord/exemplar_direct/exemplardirect_run.py
"""
from __future__ import annotations

import contextlib
import json
import pathlib
import sys
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accord"))

from accord.cloud_paths import assert_local_only  # noqa: E402
from accord.stats._overcall_common import wilson_ci  # noqa: E402

LOG_DIR = pathlib.Path.home() / "mimic_local_only" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = assert_local_only(LOG_DIR / f"exemplardirect_{time.strftime('%Y%m%dT%H%M%S')}.log")

GOLD = assert_local_only(
    pathlib.Path.home() / "mimic_local_only/from_drive_2026-08-08/outputs/annotation_sheet_for_pharmacist_clean.csv"
)
OUT = ROOT / "results" / "accord" / "exemplar_direct" / "exemplar_direct.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"

REQUIRED_COLS = ["pair_id", "rule", "drug", "drug_class", "nursing_sentence", "discharge_sentence"]
CAP_CHARS = 200
EXISTING_ESTIMATE_KEYJOIN = 0.0
EXISTING_ESTIMATE_RECONSTRUCTION = 0.4745
MIN_DECIDABLE = 100


def _normalize_drug(name, abbrev, variants):
    d = str(name).strip().lower()
    n = abbrev.get(d, d)
    n = variants.get(n, n)
    return n


def _action_for_drug(sentence_text, drug_norm, note_role, expand_text, extract_mentions_from_doc):
    if not isinstance(sentence_text, str) or not sentence_text.strip():
        return None
    expanded = expand_text(sentence_text)
    sentences = [(expanded, 0, len(expanded))]
    mentions = extract_mentions_from_doc(sentences, note_role)
    matches = [m for m in mentions if m["drug"] == drug_norm]
    if not matches:
        return None
    return matches[0]["action"]


def run() -> dict:
    import phase2_medication as p2  # noqa: E402 (imported inside redirected scope; unmodified)
    from dictionaries import ABBREV, VARIANTS  # noqa: E402 (imported, unmodified)

    df = pd.read_csv(GOLD)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return {"outcome": "C", "reason": f"missing required columns: {missing}"}

    sub = df[df["rule"].isin([1, 2])].reset_index(drop=True)
    n_total = len(sub)

    decidable_flags = []
    correct_flags = []
    at_cap_flags = []
    undecidable_reasons = {}
    action_pair_counts = {}
    wrong_involves_unclassified = []
    by_rule_n = {1: 0, 2: 0}

    for _, r in sub.iterrows():
        rule = int(r["rule"])
        by_rule_n[rule] += 1
        disch_sent = r["discharge_sentence"]
        nurse_sent = r["nursing_sentence"]
        disch_len = len(str(disch_sent)) if isinstance(disch_sent, str) else 0
        at_cap = disch_len == CAP_CHARS
        at_cap_flags.append(at_cap)

        if rule == 1:
            d_norm = _normalize_drug(r["drug"], ABBREV, VARIANTS)
            drug_n, drug_d = d_norm, d_norm
        else:
            parts = str(r["drug"]).split(" vs ")
            if len(parts) != 2:
                decidable_flags.append(False)
                correct_flags.append(None)
                undecidable_reasons["rule2_drug_field_not_parseable"] = (
                    undecidable_reasons.get("rule2_drug_field_not_parseable", 0) + 1
                )
                continue
            drug_n = _normalize_drug(parts[0], ABBREV, VARIANTS)
            drug_d = _normalize_drug(parts[1], ABBREV, VARIANTS)

        action_n = _action_for_drug(nurse_sent, drug_n, "nursing", p2.expand_text, p2.extract_mentions_from_doc)
        action_d = _action_for_drug(disch_sent, drug_d, "discharge", p2.expand_text, p2.extract_mentions_from_doc)

        if action_n is None or action_d is None:
            decidable_flags.append(False)
            correct_flags.append(None)
            reasons = []
            if action_n is None:
                reasons.append("nursing_drug_not_detected")
            if action_d is None:
                reasons.append("discharge_drug_not_detected")
            key = "+".join(reasons)
            undecidable_reasons[key] = undecidable_reasons.get(key, 0) + 1
            continue

        decidable_flags.append(True)
        pair_key = "+".join(sorted([action_n, action_d]))
        action_pair_counts[pair_key] = action_pair_counts.get(pair_key, 0) + 1
        correct = {action_n, action_d} == {"continuation", "cessation"}
        correct_flags.append(bool(correct))
        wrong_involves_unclassified.append(
            (not correct) and ("unclassified" in (action_n, action_d))
        )

    n_decidable = sum(decidable_flags)
    n_undecidable = n_total - n_decidable
    n_wrong = sum(1 for f in correct_flags if f is False)
    n_correct = sum(1 for f in correct_flags if f is True)
    assert n_wrong + n_correct == n_decidable
    n_wrong_involves_unclassified = sum(1 for f in wrong_involves_unclassified if f)
    n_wrong_no_unclassified = n_wrong - n_wrong_involves_unclassified

    n_at_cap = sum(at_cap_flags)
    decidable_not_capped = [
        correct_flags[i] for i in range(n_total) if decidable_flags[i] and not at_cap_flags[i]
    ]
    n_decidable_excl_capped = len(decidable_not_capped)
    n_wrong_excl_capped = sum(1 for f in decidable_not_capped if f is False)

    if n_decidable < MIN_DECIDABLE:
        return {
            "outcome": "C",
            "reason": f"only {n_decidable} decidable rows, below the pre-registered floor of {MIN_DECIDABLE}",
            "n_total": n_total, "n_decidable": n_decidable,
        }

    defect_rate = n_wrong / n_decidable
    ci_lo, ci_hi = wilson_ci(n_wrong, n_decidable)

    defect_rate_excl_capped = (n_wrong_excl_capped / n_decidable_excl_capped) if n_decidable_excl_capped else None
    ci_excl_lo, ci_excl_hi = (
        wilson_ci(n_wrong_excl_capped, n_decidable_excl_capped) if n_decidable_excl_capped else (None, None)
    )

    excl_0 = not (ci_lo <= EXISTING_ESTIMATE_KEYJOIN <= ci_hi)
    excl_47 = not (ci_lo <= EXISTING_ESTIMATE_RECONSTRUCTION <= ci_hi)
    if excl_0 != excl_47:
        outcome = "A"
        excluded, retained = (
            ("0 (key-join estimate)", "0.4745 (reconstruction estimate)")
            if excl_0 else
            ("0.4745 (reconstruction estimate)", "0 (key-join estimate)")
        )
        outcome_note = f"interval excludes {excluded} and includes {retained}"
    else:
        outcome = "B"
        outcome_note = (
            "interval excludes both 0 and 0.4745" if excl_0 else "interval includes both 0 and 0.4745"
        )

    return {
        "outcome": outcome,
        "outcome_note": outcome_note,
        "n_total_rule1_rule2": n_total,
        "n_by_rule": {"rule1": by_rule_n[1], "rule2": by_rule_n[2]},
        "n_decidable": n_decidable,
        "n_undecidable": n_undecidable,
        "undecidable_reasons": undecidable_reasons,
        "n_wrong": n_wrong,
        "n_correct": n_correct,
        "n_wrong_involves_unclassified": n_wrong_involves_unclassified,
        "n_wrong_no_unclassified_both_sides_classified": n_wrong_no_unclassified,
        "defect_rate": defect_rate,
        "wilson_ci_95": [ci_lo, ci_hi],
        "n_at_cap_200_discharge": n_at_cap,
        "n_decidable_excl_capped": n_decidable_excl_capped,
        "n_wrong_excl_capped": n_wrong_excl_capped,
        "defect_rate_excl_capped": defect_rate_excl_capped,
        "wilson_ci_95_excl_capped": [ci_excl_lo, ci_excl_hi],
        "action_pair_counts_among_decidable": action_pair_counts,
        "existing_estimates": {
            "keyjoin_23_rows": EXISTING_ESTIMATE_KEYJOIN,
            "reconstruction_137_rows": EXISTING_ESTIMATE_RECONSTRUCTION,
        },
        "caveat": (
            "Measurement by an unvalidated instrument: the mention extractor and action "
            "classifier (accord/phase2_medication.py, accord/dictionaries.py) have never been "
            "validated against human annotation of action classification. This is extractor "
            "dependence, not ground truth. Row-level agreement with the 23 key-joined or 137 "
            "reconstructed rows could not be computed: neither subset's row identities "
            "(pair_id) are retained in any artifact found in this repo; only aggregate counts "
            "are available for those two instruments. See C106_PREREGISTRATION.md."
        ),
    }


def main():
    with LOG_PATH.open("w") as logf, contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
        print(f"exemplardirect_run starting {time.strftime('%Y-%m-%d %H:%M:%S')}")
        result = run()
        print("exemplardirect_run finished")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT}")
    print(f"log: {LOG_PATH}")
    if result.get("outcome") == "C":
        print(f"OUTCOME C, invalid: {result.get('reason')}")
    else:
        print(
            f"outcome {result['outcome']}: n={result['n_total_rule1_rule2']} "
            f"n_decidable={result['n_decidable']} defect_rate={result['defect_rate']:.4f} "
            f"wilson_ci={result['wilson_ci_95']} n_at_cap={result['n_at_cap_200_discharge']} "
            f"defect_rate_excl_capped={result['defect_rate_excl_capped']}"
        )


if __name__ == "__main__":
    main()
