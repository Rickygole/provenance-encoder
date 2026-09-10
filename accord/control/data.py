"""
Defect-free control: data loading.

Builds the training pool from the ACCORD weak-supervision labels
(accord_weak_supervision_labels_decontam.csv, 1,679 rows, already
decontaminated against the gold set at the row level -- HADM_ID + drug +
rule + nursing_sentence + discharge_sentence), and the fixed 299-pair gold
evaluation set, using the SAME pair_id <-> HADM_ID join `run_experiments.py`
uses (annotation_sample_300.csv -> accord_decontamination_audit.csv).

ROW-LEVEL decontamination is not the same as SUBJECT-level disjointness: a
subject can have two admissions, one of which supplied a gold-evaluated pair
and the other of which supplied a training row. Checked directly: 158 of the
266 gold subjects also have at least one row in the 1,679-row weak-label
pool (291 of 1,679 rows). Correction C8 (row-level splits) is exactly this
class of mistake, so those 291 rows are EXCLUDED from the training pool here
and the exclusion is asserted, not just filtered silently.

Within what remains, a train/val split is drawn by SUBJECT_ID (not row),
reusing phase3_mining.py's approach (shuffle subjects with a dedicated RNG,
assign by prefix fraction, assert disjointness) -- fixed once
(SPLIT_SEED), independent of the per-run model seed, so that seed-to-seed
variance in the reported metrics reflects model init / training stochasticity
only, not different training data.

NEVER print or persist clinical note text outside this process's own
tokenization step. Only pair_id, rule, label, and numeric scores may leave
this module's callers.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

MENTIONS_PATH = "/Users/rickygole/mimic_local_only/accord/hf_medication_mentions.csv.gz"
WS_PATH = "/Users/rickygole/mimic_local_only/accord_supp/accord_weak_supervision_labels_decontam.csv"
GOLD_PATH = "/Users/rickygole/Downloads/gold_standard_accord_results.csv"
# Repointed off the Google Drive mount: this file joins verbatim note
# sentences to HADM_ID, which must not sit in cloud storage.
SAMP_PATH = "/Users/rickygole/mimic_local_only/from_drive_2026-08-08/outputs/annotation_sample_300.csv"
AUDIT_PATH = "/Users/rickygole/mimic_local_only/accord_supp/accord_decontamination_audit.csv"

SPLIT_SEED = 1337   # fixed train/val subject split, independent of model seed
TRAIN_FRAC = 0.85


def _hadm_to_subject() -> dict:
    mentions = pd.read_csv(MENTIONS_PATH, usecols=["HADM_ID", "SUBJECT_ID"])
    mentions = mentions.drop_duplicates("HADM_ID")
    mentions["HADM_ID"] = mentions["HADM_ID"].astype(float)
    return mentions.set_index("HADM_ID")["SUBJECT_ID"].to_dict()


def load_gold() -> pd.DataFrame:
    """Reuse run_experiments.py's pair_id <-> HADM_ID <-> weak_label join.
    Returns the 299 gold pairs with pharm_pos, rule, SUBJECT_ID, sentences."""
    gold = pd.read_csv(GOLD_PATH)
    samp = pd.read_csv(SAMP_PATH)
    audit = pd.read_csv(AUDIT_PATH)
    audit["HADM_ID"] = audit["HADM_ID"].astype(float)

    g2 = gold.merge(
        samp[["pair_id", "HADM_ID", "drug", "rule", "nursing_sentence", "discharge_sentence"]],
        on=["pair_id", "rule"], how="left",
    )
    g2["HADM_ID"] = g2["HADM_ID"].astype(float)
    key = ["HADM_ID", "drug", "rule", "nursing_sentence", "discharge_sentence"]
    gold_join = g2.merge(audit[key + ["weak_label", "weak_confidence", "removal_reason"]], on=key, how="left")

    assert gold_join["weak_label"].notna().sum() == 299, "gold<->weak-label join is not fully 1:1 -- STOP"
    assert (gold_join["removal_reason"] == "pharmacist_overlap").all()
    assert gold_join.shape[0] == 299

    hadm_to_subj = _hadm_to_subject()
    gold_join["SUBJECT_ID"] = gold_join["HADM_ID"].map(hadm_to_subj)
    assert gold_join["SUBJECT_ID"].notna().all(), "gold pair with no SUBJECT_ID match -- STOP"

    gold_join["pharm_pos"] = (gold_join["pharm_label"] == "contradiction").astype(int)
    gold_join["discharge_sentence"] = gold_join["discharge_sentence"].fillna("")
    return gold_join


def load_training_pool(gold: pd.DataFrame) -> pd.DataFrame:
    """The 1,679-row decontaminated weak-label set, with any row whose
    SUBJECT_ID also appears in the gold set removed (subject-level
    disjointness, asserted below)."""
    ws = pd.read_csv(WS_PATH)
    ws["HADM_ID"] = ws["HADM_ID"].astype(float)
    ws["discharge_sentence"] = ws["discharge_sentence"].fillna("")

    hadm_to_subj = _hadm_to_subject()
    ws["SUBJECT_ID"] = ws["HADM_ID"].map(hadm_to_subj)
    assert ws["SUBJECT_ID"].notna().all(), "training row with no SUBJECT_ID match -- STOP"

    gold_subjects = set(gold["SUBJECT_ID"].unique())
    leaked = ws["SUBJECT_ID"].isin(gold_subjects)
    n_leaked = int(leaked.sum())
    pool = ws[~leaked].reset_index(drop=True)

    assert set(pool["SUBJECT_ID"].unique()).isdisjoint(gold_subjects), (
        "subject leaked between training pool and gold set after filtering -- STOP"
    )
    assert n_leaked > 0, (
        "expected some row-level-decontaminated-but-subject-overlapping rows; "
        "got none -- check whether the leak check itself is broken"
    )

    pool["y"] = (pool["weak_label"] == "contradiction").astype(int)
    assert set(pool["weak_label"].unique()) <= {"contradiction", "not_contradiction"}

    return pool, n_leaked


def subject_train_val_split(pool: pd.DataFrame, train_frac: float = TRAIN_FRAC, split_seed: int = SPLIT_SEED):
    subjects = sorted(pool["SUBJECT_ID"].unique().tolist())
    rng = random.Random(split_seed)
    rng.shuffle(subjects)
    n_train = int(len(subjects) * train_frac)
    train_subj = set(subjects[:n_train])
    val_subj = set(subjects[n_train:])

    assert train_subj.isdisjoint(val_subj), "subject leaked between train and val"
    assert train_subj | val_subj == set(subjects)

    train_df = pool[pool["SUBJECT_ID"].isin(train_subj)].reset_index(drop=True)
    val_df = pool[pool["SUBJECT_ID"].isin(val_subj)].reset_index(drop=True)
    return train_df, val_df


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    gold: pd.DataFrame
    n_leaked_rows_excluded: int
    n_gold_subjects: int
    n_pool_subjects: int


def build_splits() -> Splits:
    gold = load_gold()
    pool, n_leaked = load_training_pool(gold)
    train_df, val_df = subject_train_val_split(pool)
    return Splits(
        train=train_df,
        val=val_df,
        gold=gold,
        n_leaked_rows_excluded=n_leaked,
        n_gold_subjects=gold["SUBJECT_ID"].nunique(),
        n_pool_subjects=pool["SUBJECT_ID"].nunique(),
    )


def tokenize_pairs(tokenizer, df: pd.DataFrame, max_length: int = 160):
    """[CLS] nursing_sentence [SEP] discharge_sentence [SEP], standard
    tokenizer pair encoding (token_type_ids marks the two segments)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enc = tokenizer(
            df["nursing_sentence"].astype(str).tolist(),
            df["discharge_sentence"].astype(str).tolist(),
            truncation=True, max_length=max_length, padding="max_length",
            return_tensors="pt",
        )
    return enc
