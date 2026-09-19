"""ACCORD Phase 3 rebuild: contradiction mining, WITH the exemplar bug fixed,
Rules 3+4 implemented, and the OLD-vs-NEW exemplar-correctness measurement
that is the actual deliverable of this rebuild.

THE BUG (from ACCORD_Phase2_Contradiction_Mining, cell 4, and identically in
cell 5 for Rule 2): the conflict is decided over the SET of actions seen
across every mention of a drug in a note --

    conflict = ('continuation' in n_actions and 'cessation' in d_actions) or (...)
    if conflict:
        n_row = nursing[nursing['drug'] == drug].iloc[0]
        d_row = discharge[discharge['drug'] == drug].iloc[0]

-- but the row shown to the pharmacist / LLM / model is the FIRST mention of
that drug in the note, which need not be the mention that carries the
action that made the set-conflict true. A drug can be mentioned five times
in a discharge summary with four "continuation" mentions and one
"cessation" mention buried at the end; `.iloc[0]` will grab a continuation
row, and the exemplar pair the pipeline calls a "contradiction" instead
depicts two notes agreeing.

THE FIX: select, on each side, the mention whose OWN `action` field is one
of the two actions that make up the conflict for that specific candidate
(not just "some mention of the drug"). If more than one mention on a side
qualifies, break the tie by taking the one nearest -- by character offset
within the note -- to any dose-bearing mention, on the theory that a mention
sitting next to a concrete dose is more likely to be a substantive clinical
instruction than a passing reference (e.g. in an allergy or problem list).

Four rules, all producing (nursing_row, discharge_row) candidate pairs per
admission:

  Rule 1 -- Direct Action Conflict. Same drug, `continuation` in one note's
    action set and `cessation` in the other's. (Reference had source code;
    exemplar selection is fixed here.)

  Rule 2 -- Drug Class Conflict. Different drugs in the same pharmacologic
    class (DRUG_CLASSES below, verbatim from the reference), one note's
    class-level action set says continuation, the other's says cessation.
    (Reference had source code; exemplar selection is fixed here.)

  Rule 3 -- Dosage Conflict (NEW, no reference source existed). Same drug
    mentioned with a parseable dose in both notes, and some (nursing dose,
    discharge dose) pair for that drug differs by more than 20% of the
    larger value, with both doses reduced to a common unit (mg, mEq, unit,
    or ml -- see dictionaries.parse_dose). The exemplar pair is the
    (nursing, discharge) dose mention pair with the LARGEST percent
    difference, so by construction it carries the conflicting dose; there is
    no legacy .iloc[0] bug to fix here, only a rule that never had code.

  Rule 4 -- Omission Conflict (NEW, no reference source existed). A drug
    receives a `cessation` action ("held") in the nursing note and is
    ABSENT from the discharge summary's extracted mentions entirely (not
    merely un-actioned -- checked against every mention the extractor found
    in that note, classified or not). The exemplar nursing row is the
    cessation mention (nearest-to-dose tie-break if more than one); there is
    no discharge exemplar because the drug is not mentioned, so
    discharge_sentence is empty and discharge_action is "absent".

Splits: admissions are assigned to train/val/test by SUBJECT_ID (not row, not
HADM_ID) with disjointness asserted in code -- the reference used a row-level
random split.

Local only. Prints aggregates only. Sentence text stays under $MIMIC_LOCAL_ONLY_DIR/accord/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dictionaries import dose_pct_diff

HERE = Path(__file__).resolve().parent
MIMIC_LOCAL_ONLY_DIR = os.environ.get("MIMIC_LOCAL_ONLY_DIR", os.path.expanduser("~/mimic_local_only"))
OUT = Path(MIMIC_LOCAL_ONLY_DIR) / "accord"

# ---------------------------------------------------------------------------
# VERBATIM from ACCORD_Phase2_Contradiction_Mining, cell 5
# ---------------------------------------------------------------------------
DRUG_CLASSES = {
    'ace_inhibitors': ['lisinopril', 'enalapril', 'ramipril', 'captopril'],
    'arbs': ['losartan', 'valsartan'],
    'beta_blockers': ['metoprolol', 'carvedilol', 'atenolol', 'bisoprolol', 'propranolol'],
    'loop_diuretics': ['furosemide', 'torsemide', 'bumetanide'],
    'anticoagulants': ['warfarin', 'heparin', 'enoxaparin', 'apixaban', 'rivaroxaban', 'dabigatran'],
    'statins': ['atorvastatin', 'simvastatin'],
    'insulins': ['insulin', 'glargine', 'lispro'],
    'nitrates': ['nitroglycerin', 'isosorbide'],
    'antiarrhythmics': ['amiodarone', 'digoxin'],
    'aldosterone_antagonists': ['spironolactone', 'eplerenone'],
}
DRUG_TO_CLASS = {drug: cls for cls, drugs in DRUG_CLASSES.items() for drug in drugs}

RULE_DESCRIPTIONS = {
    1: "Direct Action Conflict: same drug, opposite actions across notes",
    2: "Drug Class Conflict: class-level instruction conflicts with agent-level instruction",
    3: "Dosage Conflict: same drug, dose differs by more than 20 percent",
    4: "Omission Conflict: drug held in nursing note, absent from discharge summary",
}


def pick_exemplar(candidates: pd.DataFrame, note_mentions: pd.DataFrame) -> pd.Series:
    """Among mentions that already qualify (their action carries the
    conflict), break ties by nearest character offset to any dose-bearing
    mention in the same note; fall back to earliest sentence for determinism.
    """
    if len(candidates) == 1:
        return candidates.iloc[0]
    dose_positions = note_mentions.loc[note_mentions["dose_value"].notna(), "drug_start_char"].to_numpy()
    if len(dose_positions) == 0:
        return candidates.sort_values("sentence_idx").iloc[0]
    dists = candidates["drug_start_char"].apply(lambda p: np.abs(dose_positions - p).min())
    best_idx = dists.idxmin()
    return candidates.loc[best_idx]


def mine_rule1(classified: pd.DataFrame, mentions_by_note: dict) -> list[dict]:
    out = []
    for hadm_id, group in classified.groupby("HADM_ID"):
        nursing = group[group["note_role"] == "nursing"]
        discharge = group[group["note_role"] == "discharge"]
        if len(nursing) == 0 or len(discharge) == 0:
            continue
        for drug in set(nursing["drug"]) & set(discharge["drug"]):
            n_sub = nursing[nursing["drug"] == drug]
            d_sub = discharge[discharge["drug"] == drug]
            n_actions = set(n_sub["action"])
            d_actions = set(d_sub["action"])

            cond1 = "continuation" in n_actions and "cessation" in d_actions
            cond2 = "cessation" in n_actions and "continuation" in d_actions
            if not (cond1 or cond2):
                continue
            req_n, req_d = ("continuation", "cessation") if cond1 else ("cessation", "continuation")

            old_n = n_sub.iloc[0]
            old_d = d_sub.iloc[0]

            new_n_candidates = n_sub[n_sub["action"] == req_n]
            new_d_candidates = d_sub[d_sub["action"] == req_d]
            note_n = mentions_by_note.get((hadm_id, "nursing"))
            note_d = mentions_by_note.get((hadm_id, "discharge"))
            new_n = pick_exemplar(new_n_candidates, note_n)
            new_d = pick_exemplar(new_d_candidates, note_d)

            old_correct = (old_n["action"] == req_n) and (old_d["action"] == req_d)
            changed = (old_n.name != new_n.name) or (old_d.name != new_d.name)

            out.append(_row(1, hadm_id, drug, None, req_n, req_d, old_n, old_d, new_n, new_d, old_correct, changed))
    return out


def mine_rule2(classified: pd.DataFrame, mentions_by_note: dict) -> list[dict]:
    out = []
    c = classified.copy()
    c["drug_class"] = c["drug"].map(DRUG_TO_CLASS)
    c = c.dropna(subset=["drug_class"])
    for hadm_id, group in c.groupby("HADM_ID"):
        nursing = group[group["note_role"] == "nursing"]
        discharge = group[group["note_role"] == "discharge"]
        if len(nursing) == 0 or len(discharge) == 0:
            continue
        for cls in set(nursing["drug_class"]) & set(discharge["drug_class"]):
            n_sub = nursing[nursing["drug_class"] == cls]
            d_sub = discharge[discharge["drug_class"] == cls]
            if set(n_sub["drug"]) == set(d_sub["drug"]):
                continue  # same drug(s) both sides -> Rule 1 territory, skip
            n_actions = set(n_sub["action"])
            d_actions = set(d_sub["action"])
            cond1 = "continuation" in n_actions and "cessation" in d_actions
            cond2 = "cessation" in n_actions and "continuation" in d_actions
            if not (cond1 or cond2):
                continue
            req_n, req_d = ("continuation", "cessation") if cond1 else ("cessation", "continuation")

            old_n = n_sub.iloc[0]
            old_d = d_sub.iloc[0]

            new_n_candidates = n_sub[n_sub["action"] == req_n]
            new_d_candidates = d_sub[d_sub["action"] == req_d]
            note_n = mentions_by_note.get((hadm_id, "nursing"))
            note_d = mentions_by_note.get((hadm_id, "discharge"))
            new_n = pick_exemplar(new_n_candidates, note_n)
            new_d = pick_exemplar(new_d_candidates, note_d)

            old_correct = (old_n["action"] == req_n) and (old_d["action"] == req_d)
            changed = (old_n.name != new_n.name) or (old_d.name != new_d.name)

            drug_label = f'{new_n["drug"]} vs {new_d["drug"]}'
            row = _row(2, hadm_id, drug_label, cls, req_n, req_d, old_n, old_d, new_n, new_d, old_correct, changed)
            out.append(row)
    return out


def _row(rule, hadm_id, drug, drug_class, req_n, req_d, old_n, old_d, new_n, new_d, old_correct, changed):
    return {
        "HADM_ID": hadm_id,
        "rule": rule,
        "drug": drug,
        "drug_class": drug_class,
        "required_nursing_action": req_n,
        "required_discharge_action": req_d,
        "nursing_action": new_n["action"],
        "discharge_action": new_d["action"],
        "nursing_sentence": new_n["sentence"],
        "discharge_sentence": new_d["sentence"],
        "nursing_sentence_idx": new_n["sentence_idx"],
        "discharge_sentence_idx": new_d["sentence_idx"],
        "old_nursing_action": old_n["action"],
        "old_discharge_action": old_d["action"],
        "old_exemplar_correct": bool(old_correct),
        "new_exemplar_correct": bool((new_n["action"] == req_n) and (new_d["action"] == req_d)),
        "exemplar_changed_by_fix": bool(changed),
    }


def mine_rule3(all_mentions: pd.DataFrame) -> list[dict]:
    """Dosage Conflict. Same drug, dose differs by >20%, both sides reduced
    to a comparable unit. No .iloc[0] bug applicable -- this rule never
    existed in the reference, so the exemplar pair is simply the (nursing,
    discharge) dose-mention pair with the largest percent difference."""
    out = []
    dosed = all_mentions[all_mentions["dose_value"].notna()]
    for hadm_id, group in dosed.groupby("HADM_ID"):
        nursing = group[group["note_role"] == "nursing"]
        discharge = group[group["note_role"] == "discharge"]
        if len(nursing) == 0 or len(discharge) == 0:
            continue
        for drug in set(nursing["drug"]) & set(discharge["drug"]):
            n_sub = nursing[nursing["drug"] == drug]
            d_sub = discharge[discharge["drug"] == drug]
            best = None
            for _, nr in n_sub.iterrows():
                for _, dr in d_sub.iterrows():
                    if nr["dose_unit"] != dr["dose_unit"]:
                        continue
                    pct = dose_pct_diff(nr["dose_value"], dr["dose_value"])
                    if pct > 20.0 and (best is None or pct > best[0]):
                        best = (pct, nr, dr)
            if best is None:
                continue
            pct, nr, dr = best
            out.append(
                {
                    "HADM_ID": hadm_id,
                    "rule": 3,
                    "drug": drug,
                    "drug_class": DRUG_TO_CLASS.get(drug),
                    "nursing_action": nr["action"],
                    "discharge_action": dr["action"],
                    "nursing_sentence": nr["sentence"],
                    "discharge_sentence": dr["sentence"],
                    "nursing_sentence_idx": nr["sentence_idx"],
                    "discharge_sentence_idx": dr["sentence_idx"],
                    "nursing_dose": nr["dose"],
                    "discharge_dose": dr["dose"],
                    "dose_diff_pct": round(float(pct), 1),
                }
            )
    return out


def mine_rule4(classified: pd.DataFrame, all_mentions: pd.DataFrame, mentions_by_note: dict) -> list[dict]:
    """Omission Conflict. Drug held (cessation) in nursing note, absent from
    EVERY discharge mention (classified or not) for that admission."""
    out = []
    nursing_cessation = classified[(classified["note_role"] == "nursing") & (classified["action"] == "cessation")]
    discharge_drugs_by_hadm = all_mentions[all_mentions["note_role"] == "discharge"].groupby("HADM_ID")["drug"].apply(set)
    for hadm_id, group in nursing_cessation.groupby("HADM_ID"):
        discharge_drugs = discharge_drugs_by_hadm.get(hadm_id, set())
        for drug in set(group["drug"]):
            if drug in discharge_drugs:
                continue  # not an omission -- drug IS mentioned in discharge
            candidates = group[group["drug"] == drug]
            note_n = mentions_by_note.get((hadm_id, "nursing"))
            exemplar = pick_exemplar(candidates, note_n)
            out.append(
                {
                    "HADM_ID": hadm_id,
                    "rule": 4,
                    "drug": drug,
                    "drug_class": DRUG_TO_CLASS.get(drug),
                    "nursing_action": exemplar["action"],
                    "discharge_action": "absent",
                    "nursing_sentence": exemplar["sentence"],
                    "discharge_sentence": "",
                    "nursing_sentence_idx": exemplar["sentence_idx"],
                    "discharge_sentence_idx": None,
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    t0 = time.time()
    mentions = pd.read_csv(OUT / "hf_medication_mentions.csv.gz")
    print(f"Loaded mentions: {mentions.shape}")

    classified = mentions[mentions["action"] != "unclassified"].copy()
    print(f"Classified mentions: {len(classified):,}")
    print(f"Action distribution:\n{classified['action'].value_counts()}")

    mentions_by_note = {k: v for k, v in mentions.groupby(["HADM_ID", "note_role"])}

    print("\nMining Rule 1 (direct action conflict)...")
    r1 = mine_rule1(classified, mentions_by_note)
    print(f"  Rule 1 candidates: {len(r1)}")

    print("Mining Rule 2 (drug class conflict)...")
    r2 = mine_rule2(classified, mentions_by_note)
    print(f"  Rule 2 candidates: {len(r2)}")

    print("Mining Rule 3 (dosage conflict, NEW)...")
    r3 = mine_rule3(mentions)
    print(f"  Rule 3 candidates: {len(r3)}")

    print("Mining Rule 4 (omission conflict, NEW)...")
    r4 = mine_rule4(classified, mentions, mentions_by_note)
    print(f"  Rule 4 candidates: {len(r4)}")

    df1 = pd.DataFrame(r1)
    df2 = pd.DataFrame(r2)
    df3 = pd.DataFrame(r3)
    df4 = pd.DataFrame(r4)

    # ---- THE HEADLINE MEASUREMENT ----
    old_new = pd.concat([df1, df2], ignore_index=True) if (len(df1) or len(df2)) else pd.DataFrame()
    if len(old_new):
        old_correct_rate = old_new["old_exemplar_correct"].mean()
        new_correct_rate = old_new["new_exemplar_correct"].mean()
        n_changed = int(old_new["exemplar_changed_by_fix"].sum())
    else:
        old_correct_rate = new_correct_rate = float("nan")
        n_changed = 0

    print("\n" + "=" * 70)
    print("HEADLINE MEASUREMENT: exemplar correctness, OLD (.iloc[0]) vs NEW")
    print("(Rules 1+2 only -- the only rules with an .iloc[0] bug to compare)")
    print("=" * 70)
    print(f"Action-conflict candidates (rule 1 + rule 2): {len(old_new)}")
    print(f"OLD exemplar actually carries the conflict: {old_correct_rate*100:.1f}%")
    print(f"NEW exemplar actually carries the conflict: {new_correct_rate*100:.1f}%")
    print(f"Candidates where the fix changed the exemplar row: {n_changed} ({n_changed/len(old_new)*100:.1f}%)" if len(old_new) else "")

    all_candidates = pd.concat([df1, df2, df3, df4], ignore_index=True, sort=False)
    all_candidates["rule_description"] = all_candidates["rule"].map(RULE_DESCRIPTIONS)
    print(f"\nTotal candidates: {len(all_candidates)}")
    print(f"By rule:\n{all_candidates['rule'].value_counts().sort_index()}")
    print(f"Unique admissions: {all_candidates['HADM_ID'].nunique()}")

    all_candidates.to_csv(OUT / "contradiction_candidates.csv.gz", index=False)
    print(f"\nSaved {OUT / 'contradiction_candidates.csv.gz'}")

    # ---- Splits by SUBJECT_ID, disjoint, asserted ----
    hadm_to_subject = mentions.drop_duplicates("HADM_ID").set_index("HADM_ID")["SUBJECT_ID"].to_dict()
    all_candidates["SUBJECT_ID"] = all_candidates["HADM_ID"].map(hadm_to_subject)

    subjects = sorted(all_candidates["SUBJECT_ID"].dropna().unique().tolist())
    rng = random.Random(args.seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    train_subj = set(subjects[:n_train])
    val_subj = set(subjects[n_train:n_train + n_val])
    test_subj = set(subjects[n_train + n_val:])

    assert not (train_subj & val_subj), "subject leaked between train and val"
    assert not (train_subj & test_subj), "subject leaked between train and test"
    assert not (val_subj & test_subj), "subject leaked between val and test"
    assert train_subj | val_subj | test_subj == set(subjects), "split does not cover all subjects"

    def split_of(sid):
        if sid in train_subj:
            return "train"
        if sid in val_subj:
            return "val"
        if sid in test_subj:
            return "test"
        return None

    all_candidates["split"] = all_candidates["SUBJECT_ID"].map(split_of)
    all_candidates.to_csv(OUT / "contradiction_candidates_with_splits.csv.gz", index=False)

    split_counts = all_candidates["split"].value_counts().to_dict()
    admissions_per_split = all_candidates.groupby("split")["HADM_ID"].nunique().to_dict()
    subjects_per_split = {"train": len(train_subj), "val": len(val_subj), "test": len(test_subj)}

    print("\nSplits (group-disjoint by SUBJECT_ID, asserted in code):")
    print(f"  Subjects:   {subjects_per_split}")
    print(f"  Admissions: {admissions_per_split}")
    print(f"  Candidates: {split_counts}")
    print(f"  Reference used a row-level random split; 1,978 candidate pairs came from only "
          f"1,203 admissions there.")
    print(f"  This rebuild: {len(all_candidates)} candidates from {all_candidates['HADM_ID'].nunique()} "
          f"admissions across {len(subjects)} subjects.")

    summary = {
        "rule1_candidates": len(df1),
        "rule2_candidates": len(df2),
        "rule3_candidates": len(df3),
        "rule4_candidates": len(df4),
        "total_candidates": len(all_candidates),
        "unique_admissions": int(all_candidates["HADM_ID"].nunique()),
        "unique_subjects": len(subjects),
        "headline_old_exemplar_correct_rate": None if pd.isna(old_correct_rate) else round(float(old_correct_rate), 4),
        "headline_new_exemplar_correct_rate": None if pd.isna(new_correct_rate) else round(float(new_correct_rate), 4),
        "headline_n_action_conflict_candidates": len(old_new),
        "headline_n_exemplar_changed_by_fix": n_changed,
        "split_subjects": subjects_per_split,
        "split_admissions": {k: int(v) for k, v in admissions_per_split.items()},
        "split_candidates": {k: int(v) for k, v in split_counts.items()},
        "reference_rule1": 886,
        "reference_rule2": 202,
        "reference_rule3": 520,
        "reference_rule4": 370,
        "reference_total_raw": 1980,
        "reference_final_after_prefilter": 1978,
        "reference_admissions_for_1978_pairs": 1203,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(OUT / "phase3_mining_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
