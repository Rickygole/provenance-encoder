"""ANALYSIS 4. A fourth instrument for the exemplar defect rate, and the only
one that is a directly measured lower bound on the annotated gold set.

WHAT IS MEASURED. For Rules 1 and 2 the trigger predicate is fixed in code
(accord/phase3_mining.py mine_rule1 and mine_rule2): the rule fires only when
one side of the pair carries `continuation` and the other carries `cessation`.
Both functions compute

    cond1 = "continuation" in n_actions and "cessation" in d_actions
    cond2 = "cessation" in n_actions and "continuation" in d_actions
    if not (cond1 or cond2): continue

so the firing mention pair is, by construction, an opposite action pair. The
candidates table stores the action label of the exemplar mention that was
DISPLAYED to the pharmacist on each side. Therefore:

    a displayed pair whose two action labels are not {continuation, cessation}
    provably cannot be the mention pair that fired the trigger.

That is a defect, and it is decidable from the candidates table's own stored
labels. No re-extraction, no key join, no reconstruction of which mention fired.

WHY IT IS A LOWER BOUND, NOT THE RATE. The converse does not hold. A displayed
pair that IS an opposite pair may still be the wrong pair: the admission can
contain several continuation mentions and several cessation mentions of the
same drug, and the one displayed may not be the one the predicate matched. Such
a row is undetectable by this test and is scored as non-defective here. The
measured proportion is therefore a floor on the displayed-exemplar defect rate,
and the true rate is at least this large.

WHAT THE THESIS CURRENTLY CLAIMS. Chapter 3 and Chapter 4 state that the rate
cannot be recovered because the candidate schema does not record which mention
fired. That is true of WHICH mention fired. It is not true of whether the
displayed mentions COULD have fired it, which is what this script measures.

THE OBVIOUS MISREADING, GUARDED AGAINST. This test is meaningful for Rules 1
and 2 ONLY. Rule 3 fires on a dose difference and Rule 4 on an absence from the
discharge side; neither trigger mentions an action label at all, so neither
requires the displayed actions to be an opposite pair. Their non-opposite rates
are reported below because a reader will ask for them, and every one of them is
tagged with defect_interpretation "none". A high Rule 3 or Rule 4 figure is the
expected behaviour of a rule that does not constrain actions, not evidence of
a defect.

DATA AGREEMENT. MIMIC-III credentialed, local only. Action labels are the
dictionary's own labels, pair_id is a synthetic key, everything written is a
count, a proportion or an interval. No note text, no identifiers.

OUT: results/accord/clinical_core/displayed_exemplar_lower_bound.json
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "accord" / "stats"))
from _overcall_common import wilson_ci  # noqa: E402

from accord.clinical_core.clinical_core_data import RULE_SHORT, load_gold  # noqa: E402

OUT = REPO / "results/accord/clinical_core/displayed_exemplar_lower_bound.json"
MINING_SRC = REPO / "accord" / "phase3_mining.py"
BANNED_RE = re.compile(r"HADM|SUBJECT_ID|nursing_sentence|discharge_sentence", re.IGNORECASE)

OPPOSITE = {"continuation", "cessation"}

# The three instruments the thesis prints side by side, plus the rebuilt pool's
# own design rate. Quoted here only to be compared against; each is read from
# its artifact below rather than trusted as typed.
INSTRUMENTS = {
    "key_join": {
        "value": 0.0,
        "n": 23,
        "artifact": "reported in accord/exemplar_direct/exemplar_direct.json::existing_estimates.keyjoin_23_rows",
        "what": "23 rows that could be key-joined back to a mentions snapshot",
    },
    "reconstruction": {
        "value": 0.4745,
        "n": 137,
        "artifact": "results/accord/exemplar/exemplar_correctness_recomputed.json::rule_1_2_pooled",
        "what": "reconstruction of the firing mention over the 137 of 165 rows it could resolve",
    },
    "direct_extractor": {
        "value": 0.9315,
        "n": 146,
        "artifact": "results/accord/exemplar_direct/exemplar_direct.json::defect_rate",
        "what": "re-running the mention extractor and action classifier over the pair text",
    },
    "rebuilt_pool_design_rate": {
        "value": 0.4140,
        "n": 1307,
        "artifact": "$MIMIC_LOCAL_ONLY_DIR/accord/phase3_mining_summary.json::headline_old_exemplar_correct_rate",
        "what": "one minus the rebuilt pool's old (.iloc[0]) exemplar correct rate, on the 1307 rebuilt action-conflict candidates, not on the annotated pairs",
    },
}


def assert_trigger_predicate_is_opposite_pair() -> dict:
    """Read the trigger out of the mining source by static parse, so the claim
    'the trigger requires an opposite action pair' is checked against the code
    rather than asserted in prose. The source is never executed."""
    tree = ast.parse(MINING_SRC.read_text())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("mine_rule1", "mine_rule2"):
            src = ast.get_source_segment(MINING_SRC.read_text(), node) or ""
            c1 = '"continuation" in n_actions and "cessation" in d_actions' in src
            c2 = '"cessation" in n_actions and "continuation" in d_actions' in src
            guard = "if not (cond1 or cond2)" in src
            assert c1 and c2 and guard, f"{node.name} no longer has the opposite-pair trigger this test relies on"
            found[node.name] = {
                "requires_continuation_and_cessation": True,
                "guard_present": True,
            }
    assert set(found) == {"mine_rule1", "mine_rule2"}, "could not locate both action-conflict rules"
    return found


def profile_rule(sub) -> dict:
    n = int(len(sub))
    pair = sub.apply(lambda r: {r["nursing_action"], r["discharge_action"]}, axis=1)
    non_opposite = pair.apply(lambda s: s != OPPOSITE)
    k = int(non_opposite.sum())
    lo, hi = wilson_ci(k, n)
    return {
        "n_pairs": n,
        "n_displayed_pair_is_opposite": n - k,
        "n_displayed_pair_is_not_opposite": k,
        "rate_not_opposite": round(k / n, 4),
        "wilson_ci95": [round(lo, 4), round(hi, 4)],
        "_mask": non_opposite,
    }


def main() -> None:
    trigger = assert_trigger_predicate_is_opposite_pair()
    g = load_gold()

    # Re-read each comparison instrument from its artifact instead of trusting
    # the typed value above.
    recon = json.loads((REPO / "results/accord/exemplar/exemplar_correctness_recomputed.json"
                        ).read_text())["rule_1_2_pooled"]
    direct = json.loads((REPO / "results/accord/exemplar_direct/exemplar_direct.json").read_text())
    instruments = {k: dict(v) for k, v in INSTRUMENTS.items()}
    instruments["reconstruction"]["read_back"] = round(
        recon["n_wrong"] / recon["n_resolved"], 4)
    instruments["direct_extractor"]["read_back"] = round(direct["defect_rate"], 4)
    instruments["key_join"]["read_back"] = round(
        direct["existing_estimates"]["keyjoin_23_rows"], 4)
    mining_summary = json.loads(
        (Path(__import__("os").environ.get("MIMIC_LOCAL_ONLY_DIR",
                                           Path.home() / "mimic_local_only"))
         / "accord/phase3_mining_summary.json").read_text())
    instruments["rebuilt_pool_design_rate"]["read_back"] = round(
        1 - mining_summary["headline_old_exemplar_correct_rate"], 4)
    for k, v in instruments.items():
        v["agrees_with_artifact"] = abs(v["read_back"] - v["value"]) <= 0.0002

    action_rules = g[g["rule"].isin((1, 2))]
    pooled = profile_rule(action_rules)
    pooled_mask = pooled.pop("_mask")

    per_rule = {}
    for r in (1, 2):
        p = profile_rule(g[g["rule"] == r])
        p.pop("_mask")
        per_rule[str(r)] = {"rule_short": RULE_SHORT[r],
                            "trigger_requires_opposite_action_pair": True,
                            "defect_interpretation": "lower bound on the displayed-exemplar defect rate",
                            **p}

    # Rules 3 and 4: reported, and explicitly NOT a defect measurement.
    not_applicable = {}
    for r in (3, 4):
        p = profile_rule(g[g["rule"] == r])
        p.pop("_mask")
        not_applicable[str(r)] = {
            "rule_short": RULE_SHORT[r],
            "trigger_requires_opposite_action_pair": False,
            "trigger_is": ("a dose difference above a percent threshold" if r == 3
                           else "absence of the drug from every discharge mention"),
            "defect_interpretation": "none",
            "why": ("This rule's trigger does not reference an action label, so the displayed "
                    "actions are under no obligation to be an opposite pair. The figure below is "
                    "descriptive only and must not be read as a defect rate."),
            **p,
        }

    # The action pairs actually displayed on the defective Rules 1 and 2 rows,
    # as dictionary labels. This says what the pharmacist saw instead.
    bad = action_rules[pooled_mask]
    displayed = {}
    for _, r in bad.iterrows():
        key = f'{r["nursing_action"]}+{r["discharge_action"]}'
        displayed[key] = displayed.get(key, 0) + 1

    # Positive-level breakdown: the confirmed contradictions whose displayed
    # evidence provably could not have fired the rule that produced them.
    pos = action_rules[action_rules["pharm_label"] == "contradiction"].copy()
    pos_rows = []
    for _, r in pos.sort_values("pair_id").iterrows():
        affected = {r["nursing_action"], r["discharge_action"]} != OPPOSITE
        pos_rows.append({
            "pair_id": r["pair_id"],
            "rule": int(r["rule"]),
            "drug_normalised": r["drug_norm"],
            "pharmacologic_class": r["pharm_class"],
            "displayed_actions": f'{r["nursing_action"]}+{r["discharge_action"]}',
            "displayed_pair_could_have_fired_the_trigger": not affected,
            "danger_rating": (None if r["danger_rating"] != r["danger_rating"] else int(r["danger_rating"])),
        })
    n_pos_affected = sum(1 for r in pos_rows if not r["displayed_pair_could_have_fired_the_trigger"])
    danger5_all = g[(g["pharm_label"] == "contradiction") & (g["danger_rating"] == 5)]
    danger5_affected = [r["pair_id"] for r in pos_rows
                        if r["danger_rating"] == 5 and not r["displayed_pair_could_have_fired_the_trigger"]]

    # Ambiguity link: is a defective display associated with the pharmacist
    # failing to reach a verdict? Reported as counts and rates, no test.
    amb_given_bad = float((bad["pharm_label"] == "ambiguous").mean())
    good = action_rules[~pooled_mask]
    amb_given_good = float((good["pharm_label"] == "ambiguous").mean())

    art = {
        "analysis": ("Analysis 4: a directly measured LOWER BOUND on the displayed-exemplar defect "
                     "rate for Rules 1 and 2 on the annotated gold set"),
        "generated_by": "accord/clinical_core/displayed_exemplar_lower_bound.py",
        "inputs": {
            "canonical_gold_labels": "$MIMIC_LOCAL_ONLY_DIR/derived/gold_standard_accord_results.csv",
            "candidates_table": "$MIMIC_LOCAL_ONLY_DIR/from_drive_2026-08-08/outputs/annotation_sample_300.csv",
            "trigger_predicate_source": "accord/phase3_mining.py mine_rule1, mine_rule2 (read by static parse, never executed)",
        },
        "what_is_measured": (
            "The proportion of annotated pairs whose DISPLAYED exemplar action labels are not "
            "{continuation, cessation}. For Rules 1 and 2 the trigger fires only on an opposite "
            "action pair, so such a display provably cannot be the mention pair that fired the "
            "rule."),
        "why_this_is_a_lower_bound": (
            "The test detects a display that could not have fired the trigger. It cannot detect a "
            "display that is an opposite pair but still the wrong mention pair, which happens "
            "whenever an admission has more than one continuation or more than one cessation "
            "mention of the drug. Those rows are scored non-defective here, so the true "
            "displayed-exemplar defect rate is at least the rate reported."),
        "trigger_predicate_verified_in_code": trigger,
        "thesis_claim_this_bears_on": (
            "Chapters 3 and 4 state the exemplar defect rate cannot be recovered because the "
            "candidate schema does not record which mention fired. That holds for the identity of "
            "the firing mention. It does not hold for whether the displayed mentions could have "
            "fired the trigger, which is decidable from the stored action labels alone and is "
            "measured here."),
        "n_gold_pairs": int(len(g)),
        "rules_1_and_2": {
            "pooled": {"trigger_requires_opposite_action_pair": True,
                       "defect_interpretation": "lower bound on the displayed-exemplar defect rate",
                       **pooled},
            "per_rule": per_rule,
            "displayed_action_pairs_on_the_defective_rows": dict(
                sorted(displayed.items(), key=lambda x: -x[1])),
        },
        "rules_3_and_4_not_a_defect_measurement": not_applicable,
        "READ_THIS_BEFORE_THE_TABLE": (
            "Only the Rules 1 and 2 figures carry a defect interpretation. Rules 3 and 4 are "
            "included for completeness and are tagged defect_interpretation none: their triggers "
            "are a dose difference and an absence, so an opposite action pair was never required "
            "and a high non-opposite rate is the expected behaviour of the rule."),
        "confirmed_positives_in_rules_1_and_2": {
            "n": len(pos_rows),
            "n_whose_displayed_evidence_could_not_have_fired_the_trigger": n_pos_affected,
            "rows": pos_rows,
            "n_danger_5_in_whole_gold_set": int(len(danger5_all)),
            "danger_5_positives_affected": danger5_affected,
            "n_danger_5_affected": len(danger5_affected),
        },
        "association_with_the_ambiguous_verdict": {
            "ambiguous_rate_when_display_could_not_have_fired": round(amb_given_bad, 4),
            "ambiguous_rate_when_display_could_have_fired": round(amb_given_good, 4),
            "n_defective": int(len(bad)),
            "n_non_defective": int(len(good)),
            "note": ("Descriptive only. No test is run and no causal claim is made; a defective "
                     "display is one of several reasons a pair could read as undecidable."),
        },
        "comparison_with_existing_instruments": instruments,
        "independence_of_the_agreeing_instruments": {
            "caveat": (
                "The three agreeing readings are not fully independent and should not be presented "
                "as if they were. The reconstruction reads a mentions-table snapshot and re-derives "
                "which mention sat at .iloc[0]; this instrument reads the action labels the "
                "candidates table stored. Those two substrates are strongly correlated: "
                "exemplar_direct.json's own calibration check matches the stored labels against the "
                "snapshot at 0.8934 on the nursing side and 0.9508 on the discharge side. So the "
                "reconstruction and this lower bound are two views of nearly the same labels, and "
                "their agreement is weaker evidence than two independent measurements would be."),
            "what_is_independent": (
                "The rebuilt-pool design rate is independent of both. It is computed on a freshly "
                "mined 1307-candidate pool by reproducing the .iloc[0] selection and comparing it "
                "against the corrected selection, touching neither the annotated candidates table "
                "nor the snapshot those two read. That it lands at 0.4139, inside this bound's "
                "interval, is the load-bearing part of the agreement."),
            "calibration_match_nursing_side": 0.8934,
            "calibration_match_discharge_side": 0.9508,
            "calibration_source": "results/accord/exemplar_direct/exemplar_direct.json::calibration_check",
        },
        "what_the_comparison_supports": (
            "Three of the four instruments now sit close together: the reconstruction at 0.4745 on "
            "137 rows, the rebuilt pool's own design rate at 0.4140 on 1307 rebuilt candidates, and "
            "this lower bound at 0.4364 on all 165 annotated Rules 1 and 2 pairs, whose 95 percent "
            "Wilson interval contains both of the other two. The key join's 0 sits outside that "
            "interval on 23 non-randomly selected rows, and the direct extractor's 0.9315 sits "
            "above it and rests on 69 of 136 rows whose defect is a missing action cue from an "
            "unvalidated extractor. The agreement of three instruments computed from three "
            "different substrates, one of them a floor that needs no reconstruction and no "
            "extractor, is evidence that the rate is near 0.4 rather than unrecoverable. Read with "
            "the independence caveat above: the strongest single claim this supports is that a "
            "rate of 0 is excluded on the annotated pairs themselves, by a measurement that needs "
            "no reconstruction, no key join and no extractor."),
        "data_agreement": (
            "Dictionary action labels, synthetic pair keys, counts, proportions and intervals only. "
            "No clinical note text and no patient or admission identifier appears in this "
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
    print(f"pooled Rules 1+2: {pooled['n_displayed_pair_is_not_opposite']}/{pooled['n_pairs']} "
          f"= {pooled['rate_not_opposite']} CI {pooled['wilson_ci95']}")
    for r, v in per_rule.items():
        print(f"  rule {r}: {v['n_displayed_pair_is_not_opposite']}/{v['n_pairs']} "
              f"= {v['rate_not_opposite']} CI {v['wilson_ci95']}")
    print(f"  positives affected: {n_pos_affected}/{len(pos_rows)}; danger-5 affected {danger5_affected}")
    for k, v in instruments.items():
        print(f"  instrument {k}: {v['value']} read back {v['read_back']} "
              f"{'OK' if v['agrees_with_artifact'] else 'MISMATCH'}")


if __name__ == "__main__":
    main()
