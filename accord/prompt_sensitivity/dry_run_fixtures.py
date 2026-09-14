"""INVENTED sentence pairs shared by run_c109.py --dry-run and
analyze_c109.py --dry-run.

Authored for this smoke test only. Never derived from, and never joined
against, any MIMIC or gold file; carries no PHI and no MIMIC-derived text.
Field names mirror the real gold CSV's (nursing_sentence, discharge_sentence,
label_clean, rule) only so the dry run exercises the same shape the real
analyze_c109.py reads from pharmacist_labeled_clean.csv and
gold_standard_accord_results.csv.
"""

# pair_id, nursing_sentence, discharge_sentence, label_clean, rule
INVENTED_GOLD = [
    ("inv_r1_c1", "Nursing note: give invented-drug-A 10 mg by mouth twice daily.", "Discharge summary: hold invented-drug-A indefinitely.", "contradiction", 1),
    ("inv_r1_c2", "Nursing note: continue invented-drug-B 5 mg at bedtime.", "Discharge summary: stop invented-drug-B due to interaction.", "contradiction", 1),
    ("inv_r1_c3", "Nursing note: increase invented-drug-C to 40 mg daily.", "Discharge summary: invented-drug-C discontinued on discharge.", "contradiction", 1),
    ("inv_r1_n1", "Nursing note: continue invented-drug-D 20 mg daily.", "Discharge summary: continue invented-drug-D 20 mg daily.", "not_contradiction", 1),
    ("inv_r1_a1", "Nursing note: invented-drug-E dose adjusted per protocol.", "Discharge summary: invented-drug-E as previously dosed.", "ambiguous", 1),
    ("inv_r2_c1", "Nursing note: give invented-drug-F 2 puffs every 4 hours.", "Discharge summary: invented-drug-F discontinued, replaced by invented-drug-G.", "contradiction", 2),
    ("inv_r2_c2", "Nursing note: continue invented-drug-H 1 tablet nightly.", "Discharge summary: invented-drug-H held for interaction with invented-drug-I.", "contradiction", 2),
    ("inv_r2_c3", "Nursing note: invented-drug-J 100 mg twice daily.", "Discharge summary: invented-drug-J reduced to 25 mg once daily.", "contradiction", 2),
    ("inv_r2_n1", "Nursing note: continue invented-drug-K 50 mg daily.", "Discharge summary: continue invented-drug-K 50 mg daily.", "not_contradiction", 2),
    ("inv_r2_a1", "Nursing note: invented-drug-L per sliding scale.", "Discharge summary: invented-drug-L per prior sliding scale.", "ambiguous", 2),
    ("inv_r3_c1", "Nursing note: give invented-drug-M 4 mg IV every 6 hours.", "Discharge summary: invented-drug-M switched to oral, dose unspecified.", "contradiction", 3),
    ("inv_r3_n1", "Nursing note: continue invented-drug-N 10 mg daily.", "Discharge summary: continue invented-drug-N 10 mg daily.", "not_contradiction", 3),
    ("inv_r3_n2", "Nursing note: invented-drug-O 5 mg twice daily.", "Discharge summary: invented-drug-O 5 mg twice daily, unchanged.", "not_contradiction", 3),
    ("inv_r3_n3", "Nursing note: invented-drug-P as needed for pain.", "Discharge summary: invented-drug-P as needed for pain, unchanged.", "not_contradiction", 3),
    ("inv_r3_a1", "Nursing note: invented-drug-Q dose titrated overnight.", "Discharge summary: invented-drug-Q at discharge dose.", "ambiguous", 3),
    ("inv_r4_c1", "Nursing note: continue invented-drug-R 75 mg daily.", "Discharge summary: invented-drug-R contraindicated, discontinue.", "contradiction", 4),
    ("inv_r4_n1", "Nursing note: invented-drug-S 15 mg every morning.", "Discharge summary: invented-drug-S 15 mg every morning, unchanged.", "not_contradiction", 4),
    ("inv_r4_n2", "Nursing note: invented-drug-T 2 mg at bedtime.", "Discharge summary: invented-drug-T 2 mg at bedtime, unchanged.", "not_contradiction", 4),
    ("inv_r4_n3", "Nursing note: invented-drug-U weekly injection.", "Discharge summary: invented-drug-U weekly injection, unchanged.", "not_contradiction", 4),
    ("inv_r4_a1", "Nursing note: invented-drug-V per weight-based protocol.", "Discharge summary: invented-drug-V continued per protocol.", "ambiguous", 4),
]
