"""Shared loader for the clinical-core analyses (the 299-pair pharmacist gold
sheet joined to the candidates table and to every labeller's calls).

WHY THIS EXISTS. The pharmacist's three-way label, justification, confidence
and danger rating live in a hand-returned spreadsheet; the rule id, normalised
drug name, drug class, actions and doses live in the candidates table; the
labellers' calls live in three more files. Nothing in the repo joined all of
them, which is why the 12 confirmed contradictions, the false-positive
locations and the 46 ambiguous pairs were never characterised.

DATA AGREEMENT. MIMIC-III credentialed, local only. This module reads note
sentences and HADM_IDs (the join key) but the only things it returns to a
caller are pair_id (a synthetic key), rule, normalised drug name, drug class,
action, dose numbers, labels, confidence, danger rating and the pharmacist's
free-text justification. Callers MUST NOT write the justification to any
artifact; it is provided so a committed coder can reduce it to a category
label and a count. No caller may write HADM_ID, SUBJECT_ID or note text.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# phase3_mining.py does `from dictionaries import ...` (it is run as a script
# from inside accord/), so accord/ must be importable for that line to resolve.
if str(REPO / "accord") not in sys.path:
    sys.path.insert(0, str(REPO / "accord"))

from accord.dictionaries import ACTION_PATTERNS, ALL_DRUGS  # noqa: E402
from accord.phase3_mining import DRUG_TO_CLASS, RULE_DESCRIPTIONS  # noqa: E402

MIMIC = Path(os.environ.get("MIMIC_LOCAL_ONLY_DIR", os.path.expanduser("~/mimic_local_only")))

GOLD_PATH = MIMIC / "derived/gold_standard_accord_results.csv"
LABELLED_PATH = MIMIC / "from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv"
SAMP_PATH = MIMIC / "from_drive_2026-08-08/outputs/annotation_sample_300.csv"
AUDIT_PATH = MIMIC / "accord_supp/accord_decontamination_audit.csv"
REPL_DIR = MIMIC / "accord_multillm"

# Replication labellers, in the order run_multillm_replication.py ran them.
REPLICATION_MODELS = {
    "llama3:8b": REPL_DIR / "gold_replication_labels_llama3_8b.csv",
    "granite3.3:8b": REPL_DIR / "gold_replication_labels_granite33_8b.csv",
    "qwen3:8b": REPL_DIR / "gold_replication_labels_qwen3_8b.csv",
}

# Pharmacologic classes for the drugs the mining dictionary's DRUG_CLASSES map
# does not cover. DRUG_CLASSES exists to detect Rule 2 (within-class agent
# substitution) and so only enumerates classes with more than one HF agent in
# the dictionary; single-agent and supportive-therapy drugs are absent from it
# and would otherwise report as "unclassified". These are dictionary terms
# (ATC-level pharmacologic classes), not note content.
CLASS_SUPPLEMENT = {
    "aspirin": "antiplatelets",
    "potassium": "electrolyte_supplements",
    "magnesium": "electrolyte_supplements",
    "insulin": "insulins",
    "humalog": "insulins",
    "novolog": "insulins",
    "lantus": "insulins",
    "hydralazine": "vasodilators",
    "nitroprusside": "vasodilators",
    "amlodipine": "calcium_channel_blockers",
    "diltiazem": "calcium_channel_blockers",
    "nifedipine": "calcium_channel_blockers",
    "clonidine": "central_alpha2_agonists",
    "hydrochlorothiazide": "thiazide_diuretics",
    "chlorthalidone": "thiazide_diuretics",
    "metolazone": "thiazide_like_diuretics",
    "dobutamine": "inotropes",
    "dopamine": "inotropes",
    "milrinone": "inotropes",
    "morphine": "opioid_analgesics",
    "hydromorphone": "opioid_analgesics",
    "oxycodone": "opioid_analgesics",
    "acetaminophen": "non_opioid_analgesics",
    "ibuprofen": "nsaids",
    "prednisone": "corticosteroids",
    "methylprednisolone": "corticosteroids",
    "pantoprazole": "proton_pump_inhibitors",
    "omeprazole": "proton_pump_inhibitors",
    "lansoprazole": "proton_pump_inhibitors",
    "metformin": "biguanides",
    "glipizide": "sulfonylureas",
    "colchicine": "antigout_agents",
    "allopurinol": "antigout_agents",
    "vancomycin": "antibacterials",
    "ciprofloxacin": "antibacterials",
    "ceftriaxone": "antibacterials",
    "azithromycin": "antibacterials",
    "piperacillin": "antibacterials",
    "trimethoprim": "antibacterials",
    "sulfamethoxazole": "antibacterials",
    "fluconazole": "antifungals",
    "sacubitril-valsartan": "arni",
    "empagliflozin": "sglt2_inhibitors",
    "dapagliflozin": "sglt2_inhibitors",
    "ivabradine": "hcn_channel_blockers",
    "beta-blocker": "beta_blockers",
    "ace-inhibitor": "ace_inhibitors",
    "angiotensin-receptor-blocker": "arbs",
    "isosorbide": "nitrates",
    "nitroglycerin": "nitrates",
}

# The four mining rules, named as the thesis names them.
RULE_SHORT = {
    1: "direct action conflict",
    2: "drug class conflict",
    3: "dosage conflict",
    4: "discharge-side absence",
}

# Conflict type implied by the generating rule. Fixed by the rule's own
# definition (see accord/phase3_mining.py RULE_DESCRIPTIONS), not by reading
# any note.
RULE_CONFLICT_TYPE = {
    1: "held versus continued",
    2: "class substitution",
    3: "dose change",
    4: "omission",
}


def normalise_drug(raw: object) -> str:
    """The candidates table records Rule 2 rows as '<agent> vs <agent>'.
    Return the nursing-side normalised agent name (lowercase dictionary form)."""
    if not isinstance(raw, str):
        return "unknown"
    return raw.split(" vs ")[0].strip().lower()


def drug_pair(raw: object) -> tuple[str, str]:
    """(nursing-side agent, discharge-side agent); equal unless Rule 2 recorded
    a substitution."""
    if not isinstance(raw, str):
        return ("unknown", "unknown")
    parts = [p.strip().lower() for p in raw.split(" vs ")]
    return (parts[0], parts[1] if len(parts) > 1 else parts[0])


def pharm_class(drug: str) -> str:
    return DRUG_TO_CLASS.get(drug) or CLASS_SUPPLEMENT.get(drug) or "unclassified"


def load_gold() -> pd.DataFrame:
    """The 299 gold pairs, one row each, with every labeller's call attached.

    Columns returned: pair_id, rule, rule_short, rule_description, drug,
    drug_discharge, drug_class_dict, drug_class, pharm_label, danger_rating,
    confidence, justification, nursing_action, discharge_action, nursing_dose,
    discharge_dose, dose_diff_pct, weak_label, and pred_<model> per
    replication model that has been persisted.
    """
    gold = pd.read_csv(GOLD_PATH)
    assert len(gold) == 299, f"gold set is {len(gold)} rows, expected 299"

    lab = pd.read_csv(LABELLED_PATH).drop_duplicates("pair_id")
    assert lab["pair_id"].nunique() == 299

    samp = pd.read_csv(SAMP_PATH)
    audit = pd.read_csv(AUDIT_PATH)
    audit["HADM_ID"] = audit["HADM_ID"].astype(float)

    # pair_id <-> HADM_ID <-> weak label, the same join
    # run_multillm_replication.py performs.
    g2 = gold.merge(
        samp[["pair_id", "HADM_ID", "drug", "rule", "nursing_sentence", "discharge_sentence",
              "drug_class", "nursing_action", "discharge_action",
              "nursing_dose", "discharge_dose", "dose_diff_pct"]],
        on=["pair_id", "rule"], how="left")
    g2["HADM_ID"] = g2["HADM_ID"].astype(float)
    key = ["HADM_ID", "drug", "rule", "nursing_sentence", "discharge_sentence"]
    j = g2.merge(audit[key + ["weak_label", "weak_confidence"]], on=key, how="left")
    assert len(j) == 299, f"candidates join fanned out to {len(j)} rows"
    assert j["weak_label"].notna().all(), "weak labeller call missing for at least one gold pair"

    # The pharmacist's own returned sheet: free text, confidence, and the
    # cleaned three-way label, which must agree with the canonical gold.
    j = j.merge(lab[["pair_id", "label_clean", "confidence_clean", "justification"]],
                on="pair_id", how="left")
    assert (j["pharm_label"] == j["label_clean"]).all(), \
        "pharmacist sheet label disagrees with the canonical gold label"

    j["rule"] = j["rule"].astype(int)
    j["rule_short"] = j["rule"].map(RULE_SHORT)
    j["rule_description"] = j["rule"].map(RULE_DESCRIPTIONS)
    pairs = j["drug"].apply(drug_pair)
    j["drug_norm"] = [p[0] for p in pairs]
    j["drug_discharge"] = [p[1] for p in pairs]
    j["drug_class_dict"] = j["drug_class"]
    j["pharm_class"] = j["drug_norm"].apply(pharm_class)
    j["confidence"] = j["confidence_clean"]

    # PUBLISHABILITY GATE. Every drug name a caller may write must be a member
    # of the extraction dictionary (accord/dictionaries.py ALL_DRUGS) and every
    # action must be one of the dictionary's own action labels. That is what
    # makes these fields dictionary terms rather than strings copied out of a
    # note, and it is asserted rather than asserted-in-prose.
    unknown = sorted((set(j["drug_norm"]) | set(j["drug_discharge"])) - set(ALL_DRUGS))
    assert not unknown, f"drug name(s) not in the extraction dictionary, do not publish: {unknown}"
    allowed_actions = set(ACTION_PATTERNS) | {"absent", "unknown"}
    seen_actions = set(j["nursing_action"].dropna()) | set(j["discharge_action"].dropna())
    assert seen_actions <= allowed_actions, \
        f"action label(s) outside the dictionary's own labels: {sorted(seen_actions - allowed_actions)}"

    # Every column that could carry note text or an identifier is dropped here
    # so a caller cannot serialise one by accident.
    j = j.drop(columns=["HADM_ID", "nursing_sentence", "discharge_sentence", "label_clean",
                        "confidence_clean"])
    return j


def attach_replication(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Attach pred_<model> columns for each persisted replication labeller.
    Returns (df, models_attached, models_missing)."""
    attached, missing = [], []
    for model, path in REPLICATION_MODELS.items():
        if not path.exists():
            missing.append(f"{model} -> {path}")
            continue
        r = pd.read_csv(path)
        if "pair_id" not in r.columns:
            # persist_replication_labels.py wrote the llama3 file keyed by
            # HADM_ID, in the row order of accord.control.data.load_gold(),
            # which is the same pair_id order as the gold set. Recover pair_id
            # positionally, gated on the rule and drug sequences matching
            # row for row so a reordering cannot pass silently.
            assert len(r) == len(df), f"{model} file has {len(r)} rows, gold has {len(df)}"
            assert (r["rule"].astype(int).to_numpy() == df["rule"].to_numpy()).all(), \
                f"{model} rule sequence does not match the gold row order"
            assert (r["drug"].astype(str).to_numpy() == df["drug"].astype(str).to_numpy()).all(), \
                f"{model} drug sequence does not match the gold row order"
            r = r.assign(pair_id=df["pair_id"].to_numpy())
        r = r[["pair_id", "replication_label"]]
        r = r.drop_duplicates("pair_id").rename(columns={"replication_label": f"pred_{model}"})
        df = df.merge(r, on="pair_id", how="left")
        assert df[f"pred_{model}"].notna().all(), f"{model} call missing for a gold pair"
        attached.append(model)
    return df, attached, missing
