"""ANALYSIS 3. What made 46 of the 299 gold pairs ambiguous.

The thesis reports the three-way gold distribution and then characterises only
the 12 positives. The 46 ambiguous pairs outnumber the positives nearly four to
one and are never described, so a reader cannot tell whether ambiguity was a
clinical disagreement, a rule artefact, or a documentation-sufficiency problem.

METHOD. The pharmacist wrote one free-text justification per pair. This script
codes each justification against a fixed, committed set of themed regular
expressions, reports (a) multi-label incidence, how many justifications touch
each theme, and (b) a primary category under a documented priority order, so
the primary counts sum to the number of pairs. The same coder is run on the
241 not-a-contradiction and 12 contradiction justifications as a contrast, so a
theme that is simply frequent in the pharmacist's writing style cannot be
mistaken for a driver of ambiguity. Rule and pharmacologic-class distributions
are reported alongside.

PUBLISHABILITY. The justifications are read in this process only. NOTHING
derived from their text other than a theme label from the fixed list below and
an integer count is written or printed. No note text, no identifier.

OUT: results/accord/clinical_core/ambiguous_pair_themes.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from accord.clinical_core.clinical_core_data import RULE_SHORT, load_gold  # noqa: E402

OUT = REPO / "results/accord/clinical_core/ambiguous_pair_themes.json"

BANNED_RE = re.compile(r"HADM|SUBJECT_ID|nursing_sentence|discharge_sentence", re.IGNORECASE)

# ---------------------------------------------------------------------------
# THE CODER. Fixed before the counts were looked at; every pattern is generic
# clinical-reasoning vocabulary, none of it lifted from a note.
# ---------------------------------------------------------------------------
THEMES: dict[str, str] = {
    # The pharmacist says the pair does not carry enough to decide at all.
    "insufficient_documentation": (
        r"informations?|insufficient|not sure|unsure|unclear|not clear|"
        r"cannot (?:tell|determine|say|conclude)|can'?t (?:tell|say)|"
        r"needs? (?:more|to be|further|additional)|provide|missing|"
        r"not (?:specif|mention|document|state)|no (?:detail|clear)|"
        r"more (?:detail|context|information)|why"),
    # The disagreement is about magnitude, units, frequency or titration,
    # not about whether the drug is on or off.
    "dose_or_frequency_uncertainty": (
        r"dosages?|doses?|dosed|dosing|titrat|adjust|increas|decreas|reduc|taper|"
        r"once|twice|three times|daily|frequency|divided|total daily|\bmg\b|units?"),
    # The two sides describe different care settings, so a change is expected
    # rather than contradictory.
    "care_setting_transition": r"inpatient|outpatient|discharg|\bhome\b|transition",
    # The two statements plausibly refer to different moments: a hold followed
    # by a restart, or an ordering the pair does not fix.
    "temporal_sequence": (
        r"restart|resum|held|hold|temporar|later|\btimes?\b|timing|sequence|"
        r"same day|day \d|earlier"),
    # Resolution depends on a laboratory or monitoring value the pair does not
    # contain.
    "monitoring_dependent": (
        r"\binr\b|\bptt\b|levels?|monitor|\blab\b|bleeding|platelet|creatinine|"
        r"potassium level|rate control|heart rate"),
    # A benign clinical reading exists, so the pair is defensible either way.
    "plausible_clinical_alternative": (
        r"likely|probabl|\bmay\b|could|possib|however|reflect|appropriate|"
        r"expected|indication|depends|standard|guideline"),
    # The conflict is between a class-level instruction and a specific agent,
    # or between two agents of the same class.
    "class_versus_agent": (
        r"\bclass\b|together|\bboth\b|different (?:drug|medication|agent|beta)|"
        r"substitut|blocker"),
}

# Priority for the single primary category. An explicit statement that the
# evidence is insufficient outranks any clinical content that follows it,
# because it is a claim about the pair rather than about the medicine; the
# remaining order runs from the most specific mechanism to the most general.
PRIORITY = [
    "insufficient_documentation",
    "monitoring_dependent",
    "dose_or_frequency_uncertainty",
    "temporal_sequence",
    "care_setting_transition",
    "class_versus_agent",
    "plausible_clinical_alternative",
]
assert set(PRIORITY) == set(THEMES)

THEME_DESCRIPTIONS = {
    "insufficient_documentation": (
        "The pharmacist states the two sentences do not carry enough to decide, and asks for "
        "information the pair does not contain."),
    "dose_or_frequency_uncertainty": (
        "The apparent conflict is about how much or how often, not about whether the drug is on "
        "or off, and the pair does not fix the amount."),
    "care_setting_transition": (
        "The two sides describe different phases of care, so a difference between them is what a "
        "transition looks like rather than an error."),
    "temporal_sequence": (
        "The two statements plausibly refer to different moments, typically a hold followed by a "
        "restart, and the pair does not fix the order."),
    "monitoring_dependent": (
        "Whether the pair is an error depends on a laboratory or monitoring value that is not in "
        "either sentence."),
    "plausible_clinical_alternative": (
        "A benign clinical reading of the pair exists, so the pharmacist declined to call it an "
        "error without committing to it being correct either."),
    "class_versus_agent": (
        "The conflict sits between a class-level instruction and a named agent, or between two "
        "agents of one class, so what counts as the same medication is itself in question."),
    "uncategorised": (
        "The justification matched none of the seven themed patterns."),
}


def code(text: str) -> tuple[list[str], str]:
    hits = [k for k, pat in COMPILED.items() if pat.search(text)]
    for k in PRIORITY:
        if k in hits:
            return hits, k
    return hits, "uncategorised"


COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in THEMES.items()}


def profile(sub) -> dict:
    multi = Counter()
    primary = Counter()
    n_themes = Counter()
    for t in sub["justification"].fillna("").astype(str):
        hits, prim = code(t)
        for h in hits:
            multi[h] += 1
        primary[prim] += 1
        n_themes[len(hits)] += 1
    return {
        "n_pairs": int(len(sub)),
        "n_with_justification": int(sub["justification"].notna().sum()),
        "themes_multilabel_incidence": {k: multi.get(k, 0) for k in THEMES},
        "themes_multilabel_share_of_pairs": {
            k: round(multi.get(k, 0) / len(sub), 4) for k in THEMES},
        "primary_category_counts": {k: primary.get(k, 0)
                                    for k in PRIORITY + ["uncategorised"] if primary.get(k, 0)},
        "n_themes_per_justification": {str(k): v for k, v in sorted(n_themes.items())},
        "rule_distribution": {str(k): int(v) for k, v in sorted(sub["rule"].value_counts().items())},
        "rule_share": {str(k): round(v / len(sub), 4)
                       for k, v in sorted(sub["rule"].value_counts().items())},
        "pharmacologic_class_distribution": {
            k: int(v) for k, v in sorted(sub["pharm_class"].value_counts().items())},
        "confidence_distribution": {
            str(k): int(v) for k, v in sorted(sub["confidence"].fillna("unrecorded")
                                              .astype(str).value_counts().items())},
    }


def main() -> None:
    g = load_gold()
    assert g["justification"].notna().all(), "a gold pair has no justification"

    amb = g[g["pharm_label"] == "ambiguous"]
    pos = g[g["pharm_label"] == "contradiction"]
    neg = g[g["pharm_label"] == "not_contradiction"]
    nonamb = g[g["pharm_label"] != "ambiguous"]

    strata = {
        "ambiguous": profile(amb),
        "contradiction": profile(pos),
        "not_contradiction": profile(neg),
        "non_ambiguous_combined": profile(nonamb),
        "all_gold": profile(g),
    }

    # Enrichment of each theme in the ambiguous stratum relative to the
    # non-ambiguous pairs. A ratio far above 1 means the theme is a driver of
    # ambiguity rather than a habit of the pharmacist's prose.
    enrich = {}
    for k in THEMES:
        a = strata["ambiguous"]["themes_multilabel_share_of_pairs"][k]
        n = strata["non_ambiguous_combined"]["themes_multilabel_share_of_pairs"][k]
        enrich[k] = {
            "share_ambiguous": a,
            "share_non_ambiguous": n,
            "ratio": (None if n == 0 else round(a / n, 3)),
            "note": "non-ambiguous share is zero, ratio undefined" if n == 0 else None,
        }

    # Rule-level ambiguity rate: of the pairs a rule generated, how many the
    # pharmacist could not call either way.
    rule_rates = {}
    for r in sorted(g["rule"].unique()):
        sub = g[g["rule"] == r]
        rule_rates[str(int(r))] = {
            "rule_short": RULE_SHORT[int(r)],
            "n_pairs": int(len(sub)),
            "n_ambiguous": int((sub["pharm_label"] == "ambiguous").sum()),
            "ambiguous_rate": round(float((sub["pharm_label"] == "ambiguous").mean()), 4),
            "n_confirmed": int((sub["pharm_label"] == "contradiction").sum()),
            "confirmed_rate": round(float((sub["pharm_label"] == "contradiction").mean()), 4),
        }

    class_rates = {}
    for c in sorted(g["pharm_class"].unique()):
        sub = g[g["pharm_class"] == c]
        class_rates[c] = {
            "n_pairs": int(len(sub)),
            "n_ambiguous": int((sub["pharm_label"] == "ambiguous").sum()),
            "ambiguous_rate": round(float((sub["pharm_label"] == "ambiguous").mean()), 4),
            "n_confirmed": int((sub["pharm_label"] == "contradiction").sum()),
        }

    # Cross-check the two ambiguous-dropped strata the thesis already uses
    # (C112) against this join, so the ambiguous set characterised here is
    # provably the same 46 pairs the thesis drops.
    c112 = json.loads((REPO / "results/accord/c112_decontam_defectfree/c112_defectfree.json"
                       ).read_text())["strata"]
    strata_gate = {}
    for name, rules in (("rules12_ambiguous_dropped", (1, 2)), ("rules34_ambiguous_dropped", (3, 4))):
        sub = nonamb[nonamb["rule"].isin(rules)]
        got = {"n": int(len(sub)), "n_pos": int((sub["pharm_label"] == "contradiction").sum())}
        reg = {"n": c112[name]["n"], "n_pos": c112[name]["n_pos"]}
        strata_gate[name] = {"registered": reg, "recomputed": got, "agrees": got == reg}
        assert got == reg, f"{name} does not reproduce the registered stratum: {got} vs {reg}"

    art = {
        "analysis": "Analysis 3: what made 46 of the 299 gold pairs ambiguous",
        "generated_by": "accord/clinical_core/ambiguous_pair_themes.py",
        "inputs": {
            "canonical_gold_labels": "$MIMIC_LOCAL_ONLY_DIR/derived/gold_standard_accord_results.csv",
            "pharmacist_returned_sheet": "$MIMIC_LOCAL_ONLY_DIR/from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv",
            "candidates_table": "$MIMIC_LOCAL_ONLY_DIR/from_drive_2026-08-08/outputs/annotation_sample_300.csv",
        },
        "method": (
            "One free-text justification per gold pair was coded against seven fixed regular "
            "expressions over generic clinical-reasoning vocabulary. Multi-label incidence counts "
            "every theme a justification touches; the primary category is the highest-priority "
            "theme it touches, under the priority order recorded here. The justification text is "
            "never written, printed or quoted."),
        "theme_definitions": THEME_DESCRIPTIONS,
        "theme_patterns": THEMES,
        "primary_priority_order": PRIORITY,
        "n_gold_pairs": int(len(g)),
        "gold_label_distribution": {k: int(v) for k, v in g["pharm_label"].value_counts().items()},
        "ambiguous_to_positive_ratio": round(len(amb) / len(pos), 3),
        "strata": strata,
        "theme_enrichment_ambiguous_vs_non_ambiguous": enrich,
        "ambiguity_rate_by_rule": rule_rates,
        "ambiguity_rate_by_pharmacologic_class": class_rates,
        "c112_ambiguous_dropped_strata_gate": strata_gate,
        "data_agreement": (
            "Category labels and integer counts only. No clinical note text, no free-text "
            "justification content, and no patient or admission identifier appears in this "
            "artifact; asserted in code before writing."),
    }

    blob = json.dumps(art, indent=1, sort_keys=False)
    m = BANNED_RE.search(blob)
    assert m is None, f"unpublishable token {m.group(0)!r} reached the artifact"
    for ch in ("\u2013", "\u2014"):
        assert ch not in blob, "en or em dash in artifact"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(blob + "\n")
    print(f"wrote {OUT}")
    print("ambiguous primary:", strata["ambiguous"]["primary_category_counts"])
    print("ambiguous multilabel:", strata["ambiguous"]["themes_multilabel_incidence"])
    print("enrichment:", {k: v["ratio"] for k, v in enrich.items()})
    print("rule rates:", {k: (v["n_pairs"], v["n_ambiguous"], v["ambiguous_rate"]) for k, v in rule_rates.items()})


if __name__ == "__main__":
    main()
