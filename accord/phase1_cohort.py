"""ACCORD Phase 1 rebuild: heart-failure cohort + nursing/discharge note pairs.

Reproduces ACCORD_Phase1_Cohort_Extraction with two bugs fixed:

  BUG A (LOS filter silently dropped). The reference notebook computes
  `hf['LOS_HOURS'] > 0` on a copy (cell 4, giving 13,597 admissions) but then
  never uses that filtered frame again -- every downstream note join uses
  `hf_hadm`, the raw ICD-9 ^428 set from cell 3 (13,608 admissions), so the
  LOS>0 condition never actually constrains which notes get pulled. Here the
  LOS-filtered HADM_ID set is what everything downstream joins against.

  BUG B (discharge summary "first" is not chronological). The reference
  selects the discharge note with
  `.groupby('HADM_ID').first()` WITHOUT sorting first (cell 18). This looks
  like "first note charted" but CHARTTIME is null for 100% of Discharge
  summary rows in MIMIC-III (verified against NOTEEVENTS.csv.gz locally) --
  so ".first()" returns whatever row happened to be first in that chunk's
  concatenation order, not the first discharge summary in time. ~11% of HF
  admissions have >1 discharge-summary row (Report + Addendum), so this is
  not a corner case. Fixed here by sorting on CHARTDATE (the field that IS
  populated for discharge summaries), then ROW_ID as a deterministic
  tie-break, before taking the first.

Cohort definition (unchanged from reference): ICD-9 code matching ^428
(heart failure) in DIAGNOSES_ICD, LOS_HOURS > 0, admission has both a
nursing note (last CHARTTIME strictly before DISCHTIME) and a discharge
summary (first chronologically).

Local only. Prints aggregate counts. Writes note TEXT only under local_only/
(gitignored), never elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "local_only"

NOTEEVENTS = os.environ.get("MIMIC_NOTEEVENTS_PATH", "/Users/rickygole/mimic_local_only/raw_mimiciii/NOTEEVENTS.csv.gz")
ADMISSIONS = os.environ.get("MIMIC_ADMISSIONS_PATH", "/Users/rickygole/mimic_local_only/raw_mimiciii/ADMISSIONS.csv")
PATIENTS = os.environ.get("MIMIC_PATIENTS_PATH", "/Users/rickygole/mimic_local_only/raw_mimiciii/PATIENTS.csv")
DIAGNOSES = os.environ.get("MIMIC_DIAGNOSES_PATH", "/Users/rickygole/mimic_local_only/raw_mimiciii/DIAGNOSES_ICD.csv.gz")

KEEP_CATEGORIES = {"discharge summary", "nursing", "nursing/other"}


def assign_role(category: str) -> str:
    cat = str(category).strip().lower()
    if cat == "discharge summary":
        return "discharge"
    if cat in ("nursing", "nursing/other"):
        return "nursing"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    admissions = pd.read_csv(ADMISSIONS, parse_dates=["ADMITTIME", "DISCHTIME"])
    patients = pd.read_csv(PATIENTS, parse_dates=["DOB", "DOD"])
    diagnoses = pd.read_csv(DIAGNOSES, dtype={"ICD9_CODE": str})

    print(f"Admissions: {admissions.shape}")
    print(f"Patients:   {patients.shape}")
    print(f"Diagnoses:  {diagnoses.shape}")

    diagnoses["ICD9_CODE"] = diagnoses["ICD9_CODE"].fillna("").astype(str)
    hf_hadm_icd = set(diagnoses[diagnoses["ICD9_CODE"].str.match(r"^428")]["HADM_ID"])
    print(f"\nHF admissions by ICD-9 ^428 (pre-LOS filter): {len(hf_hadm_icd)}")

    hf = admissions[admissions["HADM_ID"].isin(hf_hadm_icd)].copy()
    hf = hf.merge(patients[["SUBJECT_ID", "GENDER", "DOB"]], on="SUBJECT_ID", how="left")
    hf["LOS_HOURS"] = (hf["DISCHTIME"] - hf["ADMITTIME"]).dt.total_seconds() / 3600
    hf_los = hf[hf["LOS_HOURS"] > 0].copy()

    print(f"HF admissions with LOS_HOURS > 0 (BUG A fixed, this set is used downstream): {len(hf_los)}")
    print(f"  (reference reports 13,597 here but then does not apply the filter downstream)")
    print(f"Unique patients: {hf_los['SUBJECT_ID'].nunique()}")
    print(f"Mean LOS (hrs):   {hf_los['LOS_HOURS'].mean():.1f}")
    print(f"Median LOS (hrs): {hf_los['LOS_HOURS'].median():.1f}")

    hf_hadm = set(hf_los["HADM_ID"])
    disch_lookup = hf_los.set_index("HADM_ID")["DISCHTIME"].to_dict()

    print("\nStreaming NOTEEVENTS...")
    keep_chunks = []
    n_rows_seen = 0
    for i, chunk in enumerate(
        pd.read_csv(
            NOTEEVENTS,
            parse_dates=["CHARTTIME"],
            chunksize=args.chunksize,
            low_memory=False,
        )
    ):
        n_rows_seen += len(chunk)
        chunk["CATEGORY"] = chunk["CATEGORY"].str.strip()
        cat_lower = chunk["CATEGORY"].str.lower()
        mask = chunk["HADM_ID"].isin(hf_hadm) & cat_lower.isin(KEEP_CATEGORIES)
        filtered = chunk[mask]
        if len(filtered) > 0:
            keep_chunks.append(
                filtered[["ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTDATE", "CHARTTIME", "CATEGORY", "TEXT"]].copy()
            )
        if i % 5 == 0:
            print(f"  ...{n_rows_seen:,} NOTEEVENTS rows scanned, {sum(len(c) for c in keep_chunks):,} kept")

    notes = pd.concat(keep_chunks, ignore_index=True)
    notes["CHARTDATE"] = pd.to_datetime(notes["CHARTDATE"])
    notes["NOTE_ROLE"] = notes["CATEGORY"].apply(assign_role)
    print(f"\nTotal NOTEEVENTS rows scanned: {n_rows_seen:,}")
    print(f"HF nursing/discharge notes kept: {len(notes):,}")
    print(f"Role distribution:\n{notes['NOTE_ROLE'].value_counts()}")

    role_per_admission = notes.groupby("HADM_ID")["NOTE_ROLE"].apply(set)
    has_discharge = role_per_admission.apply(lambda x: "discharge" in x).sum()
    has_nursing = role_per_admission.apply(lambda x: "nursing" in x).sum()
    has_both = role_per_admission.apply(lambda x: {"nursing", "discharge"}.issubset(x)).sum()

    print(f"\nHF admissions (LOS-filtered) total: {len(hf_hadm):,}")
    print(f"Have discharge summary:             {has_discharge:,} ({has_discharge/len(hf_hadm)*100:.1f}%)")
    print(f"Have nursing note:                  {has_nursing:,} ({has_nursing/len(hf_hadm)*100:.1f}%)")
    print(f"Have nursing + discharge (role coverage): {has_both:,} ({has_both/len(hf_hadm)*100:.1f}%)")
    print("  (reference reports 9,825 here, but against the unfiltered 13,608-admission universe)")

    notes["DISCHTIME"] = notes["HADM_ID"].map(disch_lookup)

    # Nursing: last CHARTTIME strictly before DISCHTIME.
    nursing_pool = notes[
        (notes["NOTE_ROLE"] == "nursing")
        & notes["CHARTTIME"].notna()
        & (notes["CHARTTIME"] < notes["DISCHTIME"])
    ].sort_values(["HADM_ID", "CHARTTIME"])
    nursing_sel = nursing_pool.groupby("HADM_ID").last().reset_index()

    # Discharge: first chronologically. CHARTTIME is null for discharge
    # summaries in MIMIC-III, so we sort on CHARTDATE with ROW_ID as a
    # deterministic tie-break (BUG B fix -- reference took an arbitrary row).
    discharge_pool = notes[notes["NOTE_ROLE"] == "discharge"].sort_values(["HADM_ID", "CHARTDATE", "ROW_ID"])
    discharge_sel = discharge_pool.groupby("HADM_ID").first().reset_index()

    print(f"\nNursing notes selected:   {len(nursing_sel):,}")
    print(f"Discharge notes selected: {len(discharge_sel):,}")

    valid_hadm = set(nursing_sel["HADM_ID"]) & set(discharge_sel["HADM_ID"])
    print(f"Valid admissions with both notes selected: {len(valid_hadm):,}")
    print("  (reference reports 9,783 valid admissions / 19,566 note-pair rows)")

    nursing_sel = nursing_sel[nursing_sel["HADM_ID"].isin(valid_hadm)].copy()
    discharge_sel = discharge_sel[discharge_sel["HADM_ID"].isin(valid_hadm)].copy()
    nursing_sel["FINAL_ROLE"] = "nursing"
    discharge_sel["FINAL_ROLE"] = "discharge"

    pairs = pd.concat([nursing_sel, discharge_sel], ignore_index=True)
    pairs = pairs[["HADM_ID", "SUBJECT_ID", "ROW_ID", "FINAL_ROLE", "CHARTTIME", "CHARTDATE", "TEXT"]]

    pairs.to_csv(OUT / "hf_note_pairs.csv.gz", index=False)
    print(f"\nSaved {OUT / 'hf_note_pairs.csv.gz'}  ({len(pairs):,} rows)")

    summary = {
        "hf_admissions_icd_only": len(hf_hadm_icd),
        "hf_admissions_los_filtered": len(hf_los),
        "hf_admissions_los_filtered_unique_subjects": int(hf_los["SUBJECT_ID"].nunique()),
        "notes_scanned_total": n_rows_seen,
        "hf_nursing_or_discharge_notes_kept": len(notes),
        "has_discharge_summary": int(has_discharge),
        "has_nursing_note": int(has_nursing),
        "has_both_roles": int(has_both),
        "nursing_notes_selected": len(nursing_sel),
        "discharge_notes_selected": len(discharge_sel),
        "valid_admissions_both_selected": len(valid_hadm),
        "total_note_pair_rows": len(pairs),
        "reference_hf_admissions_reported": 13597,
        "reference_hf_admissions_actually_used_for_notes": 13608,
        "reference_has_both_roles": 9825,
        "reference_valid_admissions_final": 9783,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(OUT / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT / 'phase1_summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
