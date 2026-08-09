"""Loading + group-join helpers for the selection-experiment.

Reads MIMIC-III derived files under /Users/rickygole/mimic_local_only. Two
kinds of loaders are kept strictly separate:

  - load_split_features(): per-row hadm_id/subject_id/label plus DERIVED
    scalar text statistics (character count, segment count). Never returns
    the text itself, never caches it.
  - load_split_text(): per-row concatenated note text, for transient use by
    the TF-IDF vectorizer only. Callers must not print, log, or persist the
    return value -- it is discarded immediately after vectorization.

join_admissions() reads only HADM_ID/LANGUAGE/ETHNICITY/INSURANCE from
ADMISSIONS.csv (the quarantine copy -- see task brief) and never any note or
free-text clinical field from that table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TASK_DIR = Path("/Users/rickygole/mimic_local_only/task")
ADMISSIONS_CSV = Path(
    "/Users/rickygole/mimic_local_only/from_drive_2026-08-08/data/ADMISSIONS.csv"
)

UNKNOWN_ETHNICITY_VALUES = {
    "UNKNOWN/NOT SPECIFIED",
    "UNABLE TO OBTAIN",
    "PATIENT DECLINED TO ANSWER",
}


def load_split_features(name: str) -> pd.DataFrame:
    """hadm_id/subject_id/label/n_segments/total_chars for one split. No text."""
    rows = []
    with (TASK_DIR / f"{name}.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            segs = d["segments"]
            rows.append(
                dict(
                    hadm_id=d["hadm_id"],
                    subject_id=d["subject_id"],
                    label=int(d["label"]),
                    n_segments=len(segs),
                    total_chars=sum(len(s["text"]) for s in segs),
                )
            )
    df = pd.DataFrame(rows)
    df["split"] = name
    return df


def load_split_text(name: str) -> tuple[list[str], list[str]]:
    """(hadm_ids, concatenated_text) for one split, in file order. Transient
    use only (vectorize immediately, do not retain)."""
    hadm_ids, texts = [], []
    with (TASK_DIR / f"{name}.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            hadm_ids.append(d["hadm_id"])
            texts.append(" ".join(s["text"] for s in d["segments"]))
    return hadm_ids, texts


def load_admissions() -> pd.DataFrame:
    adm = pd.read_csv(
        ADMISSIONS_CSV, usecols=["HADM_ID", "LANGUAGE", "ETHNICITY", "INSURANCE"]
    )
    adm["HADM_ID"] = adm["HADM_ID"].astype(str)
    return adm


def join_admissions(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join group attributes onto df (must have a 'hadm_id' str column)."""
    adm = load_admissions()
    out = df.merge(adm, left_on="hadm_id", right_on="HADM_ID", how="left", validate="one_to_one")
    n_unmatched = out["HADM_ID"].isna().sum()
    if n_unmatched:
        raise AssertionError(
            f"{n_unmatched} of {len(out)} rows failed to join to ADMISSIONS.csv "
            f"on HADM_ID -- refusing to proceed with an incomplete group join."
        )
    out["lang_missing"] = out["LANGUAGE"].isna().astype(int)
    out["ethnicity_unknown"] = (
        out["ETHNICITY"].isin(UNKNOWN_ETHNICITY_VALUES) | out["ETHNICITY"].isna()
    ).astype(int)
    out = out.drop(columns=["HADM_ID"])
    return out


def build_population() -> pd.DataFrame:
    """val + test, joined with group attributes, feature-only (no text). This
    is the population the selection sweep and audit operate on; the
    classifier is fit separately, once, on train only."""
    pop = pd.concat(
        [load_split_features("val"), load_split_features("test")], ignore_index=True
    )
    pop = join_admissions(pop)
    return pop


def doc_feature(pop: pd.DataFrame) -> np.ndarray:
    """0.7*z(total_chars) + 0.3*z(n_segments): documentation volume (primary)
    and note-mention count (secondary), the real, outcome-predictive
    selection mechanism this experiment targets. No group term."""
    zc = (pop["total_chars"] - pop["total_chars"].mean()) / pop["total_chars"].std()
    zn = (pop["n_segments"] - pop["n_segments"].mean()) / pop["n_segments"].std()
    return (0.7 * zc + 0.3 * zn).to_numpy()
