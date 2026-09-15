"""Generality Experiment A -- build sepsis and AKI cohorts with the SAME
phase1 (cohort+note pairs) / phase2 (medication mentions) / phase3
(contradiction-candidate mining) logic used for the HF cohort in
accord/phase1_cohort.py, accord/phase2_medication.py, accord/phase3_mining.py,
generalized with a --icd9-prefix / --cohort parameter.

Cohort definitions (ICD-9 prefix match on DIAGNOSES_ICD, per
accord/thesis/ch3_data.tex Table 3.1 and CORRECTIONS.md L297, which already
report these exact target sizes from a prior run):
  sepsis: ICD-9 prefix "038" (septicemia)  -> target 4,339 note-pair admissions
  aki:    ICD-9 prefix "584" (acute kidney failure) -> target 7,579

Both cohorts are built in a SINGLE pass over NOTEEVENTS (expensive scan done
once, not twice). Mention extraction and candidate mining run per cohort
after, reusing the phase2/phase3 algorithms VERBATIM (same dictionaries,
same exemplar-selection fix, same 4 rules).

Local only. Intermediate per-admission files (with note TEXT, needed to
reproduce the mining pipeline exactly as the HF cohort was built) are written
under /Users/rickygole/mimic_local_only/accord_<cohort>/ -- gitignored,
never printed, mirroring the existing /Users/rickygole/mimic_local_only/accord/
(HF) layout. No note text is printed to stdout at any point.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import spacy

REPO_ACCORD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ACCORD))
from dictionaries import ABBREV, ACTION_PATTERNS, ALL_DRUGS, NEGATION, VARIANTS, parse_dose, resolve_action  # noqa: E402

MIMIC_ROOT = Path("/Users/rickygole/mimic_local_only")

NOTEEVENTS = os.environ.get("MIMIC_NOTEEVENTS_PATH", "/Users/rickygole/mimic_local_only/raw_mimiciii/NOTEEVENTS.csv.gz")
ADMISSIONS = os.environ.get("MIMIC_ADMISSIONS_PATH", "/Users/rickygole/mimic_local_only/raw_mimiciii/ADMISSIONS.csv")
PATIENTS = os.environ.get("MIMIC_PATIENTS_PATH", "/Users/rickygole/mimic_local_only/raw_mimiciii/PATIENTS.csv")
DIAGNOSES = os.environ.get("MIMIC_DIAGNOSES_PATH", "/Users/rickygole/mimic_local_only/raw_mimiciii/DIAGNOSES_ICD.csv.gz")

KEEP_CATEGORIES = {"discharge summary", "nursing", "nursing/other"}

COHORTS = {
    "sepsis": r"^038",
    "aki": r"^584",
}


def assign_role(category: str) -> str:
    cat = str(category).strip().lower()
    if cat == "discharge summary":
        return "discharge"
    if cat in ("nursing", "nursing/other"):
        return "nursing"
    return "other"


# ---------------------------------------------------------------------------
# PHASE 1 (generalized): cohort + note pairs, both cohorts in one NOTEEVENTS pass
# ---------------------------------------------------------------------------
def phase1_build_all(chunksize: int = 200_000) -> dict:
    t0 = time.time()
    admissions = pd.read_csv(ADMISSIONS, parse_dates=["ADMITTIME", "DISCHTIME"])
    patients = pd.read_csv(PATIENTS, parse_dates=["DOB", "DOD"])
    diagnoses = pd.read_csv(DIAGNOSES, dtype={"ICD9_CODE": str})
    diagnoses["ICD9_CODE"] = diagnoses["ICD9_CODE"].fillna("").astype(str)

    cohort_hadm = {}
    for name, prefix in COHORTS.items():
        icd_hadm = set(diagnoses[diagnoses["ICD9_CODE"].str.match(prefix)]["HADM_ID"])
        print(f"[{name}] admissions matching ICD-9 prefix {prefix}: {len(icd_hadm):,}")
        sub = admissions[admissions["HADM_ID"].isin(icd_hadm)].copy()
        sub = sub.merge(patients[["SUBJECT_ID", "GENDER", "DOB"]], on="SUBJECT_ID", how="left")
        sub["LOS_HOURS"] = (sub["DISCHTIME"] - sub["ADMITTIME"]).dt.total_seconds() / 3600
        sub_los = sub[sub["LOS_HOURS"] > 0].copy()
        print(f"[{name}] after LOS>0: {len(sub_los):,}")
        cohort_hadm[name] = sub_los

    union_hadm = set()
    disch_lookup = {}
    for name, sub_los in cohort_hadm.items():
        union_hadm |= set(sub_los["HADM_ID"])
        disch_lookup.update(sub_los.set_index("HADM_ID")["DISCHTIME"].to_dict())
    print(f"\nUnion of HADM_IDs to scan NOTEEVENTS for: {len(union_hadm):,}")

    print("Streaming NOTEEVENTS (single pass for both cohorts)...")
    keep_chunks = []
    n_rows_seen = 0
    for i, chunk in enumerate(
        pd.read_csv(NOTEEVENTS, parse_dates=["CHARTTIME"], chunksize=chunksize, low_memory=False)
    ):
        n_rows_seen += len(chunk)
        chunk["CATEGORY"] = chunk["CATEGORY"].str.strip()
        cat_lower = chunk["CATEGORY"].str.lower()
        mask = chunk["HADM_ID"].isin(union_hadm) & cat_lower.isin(KEEP_CATEGORIES)
        filtered = chunk[mask]
        if len(filtered) > 0:
            keep_chunks.append(
                filtered[["ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTDATE", "CHARTTIME", "CATEGORY", "TEXT"]].copy()
            )
        if i % 5 == 0:
            print(f"  ...{n_rows_seen:,} scanned, {sum(len(c) for c in keep_chunks):,} kept")

    notes = pd.concat(keep_chunks, ignore_index=True)
    notes["CHARTDATE"] = pd.to_datetime(notes["CHARTDATE"])
    notes["NOTE_ROLE"] = notes["CATEGORY"].apply(assign_role)
    notes["DISCHTIME"] = notes["HADM_ID"].map(disch_lookup)
    print(f"Total NOTEEVENTS rows scanned: {n_rows_seen:,}; kept {len(notes):,}; elapsed {time.time()-t0:.0f}s")

    summaries = {}
    for name, sub_los in cohort_hadm.items():
        hadm_set = set(sub_los["HADM_ID"])
        cnotes = notes[notes["HADM_ID"].isin(hadm_set)]

        nursing_pool = cnotes[
            (cnotes["NOTE_ROLE"] == "nursing") & cnotes["CHARTTIME"].notna() & (cnotes["CHARTTIME"] < cnotes["DISCHTIME"])
        ].sort_values(["HADM_ID", "CHARTTIME"])
        nursing_sel = nursing_pool.groupby("HADM_ID").last().reset_index()

        discharge_pool = cnotes[cnotes["NOTE_ROLE"] == "discharge"].sort_values(["HADM_ID", "CHARTDATE", "ROW_ID"])
        discharge_sel = discharge_pool.groupby("HADM_ID").first().reset_index()

        valid_hadm = set(nursing_sel["HADM_ID"]) & set(discharge_sel["HADM_ID"])
        nursing_sel = nursing_sel[nursing_sel["HADM_ID"].isin(valid_hadm)].copy()
        discharge_sel = discharge_sel[discharge_sel["HADM_ID"].isin(valid_hadm)].copy()
        nursing_sel["FINAL_ROLE"] = "nursing"
        discharge_sel["FINAL_ROLE"] = "discharge"

        pairs = pd.concat([nursing_sel, discharge_sel], ignore_index=True)
        pairs = pairs[["HADM_ID", "SUBJECT_ID", "ROW_ID", "FINAL_ROLE", "CHARTTIME", "CHARTDATE", "TEXT"]]

        out_dir = MIMIC_ROOT / f"accord_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(out_dir / f"{name}_note_pairs.csv.gz", index=False)

        summary = {
            "cohort": name,
            "icd_prefix": COHORTS[name],
            "admissions_icd_los_filtered": len(sub_los),
            "unique_subjects": int(sub_los["SUBJECT_ID"].nunique()),
            "valid_admissions_both_notes": len(valid_hadm),
            "note_pair_rows": len(pairs),
        }
        summaries[name] = summary
        print(f"[{name}] valid admissions with both notes: {len(valid_hadm):,} "
              f"(target from thesis Table 3.1: {'4,339' if name=='sepsis' else '7,579'})")
        with open(out_dir / f"{name}_phase1_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return summaries


# ---------------------------------------------------------------------------
# PHASE 2 (generalized): medication mention extraction -- VERBATIM logic
# ---------------------------------------------------------------------------
_ABBREV_RES = [(re.compile(r"\b" + re.escape(a) + r"\b", re.IGNORECASE), full) for a, full in ABBREV.items()]
_VARIANT_RES = [(re.compile(r"\b" + re.escape(v) + r"\b", re.IGNORECASE), correct) for v, correct in VARIANTS.items()]
_DRUG_RES = [(drug, re.compile(r"\b" + re.escape(drug) + r"\b", re.IGNORECASE)) for drug in ALL_DRUGS]


def expand_text(text: str) -> str:
    for pat, full in _ABBREV_RES:
        text = pat.sub(full, text)
    for pat, correct in _VARIANT_RES:
        text = pat.sub(correct, text)
    return text


def build_sentences(doc):
    out = []
    for sent in doc.sents:
        raw = sent.text
        stripped = raw.strip()
        if len(stripped) <= 10:
            continue
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        start = sent.start_char + lead
        end = sent.end_char - trail
        out.append((stripped, start, end))
    return out


def extract_mentions_from_doc(sentences, note_role):
    mentions = []
    texts = [s[0] for s in sentences]
    for i, (sent_text, sent_start, sent_end) in enumerate(sentences):
        window_start = max(0, i - 1)
        window_end = min(len(texts), i + 2)
        context = " ".join(texts[window_start:window_end])
        is_negated = bool(NEGATION.search(context))
        raw_action = "unclassified"
        for act, pattern in ACTION_PATTERNS.items():
            if pattern.search(context):
                raw_action = act
                break
        final_action = resolve_action(raw_action, is_negated)
        for drug, pat in _DRUG_RES:
            m = pat.search(sent_text)
            if not m:
                continue
            dose = ""
            dose_start = dose_end = None
            dose_pat = re.compile(re.escape(drug) + r"\s*(\d+(?:\.\d+)?\s*(?:mg|mcg|mEq|units?|ml|g))", re.IGNORECASE)
            dose_match = dose_pat.search(sent_text)
            if dose_match:
                dose = dose_match.group(1).strip()
                dose_start = sent_start + dose_match.start(1)
                dose_end = sent_start + dose_match.end(1)
            normalized = ABBREV.get(drug.lower(), drug.lower())
            normalized = VARIANTS.get(normalized, normalized)
            parsed = parse_dose(dose)
            dose_value, dose_unit = parsed if parsed else (None, None)
            mentions.append({
                "drug": normalized, "dose": dose, "dose_value": dose_value, "dose_unit": dose_unit,
                "action": final_action, "negated": is_negated, "sentence": sent_text, "sentence_idx": i,
                "sent_start_char": sent_start, "sent_end_char": sent_end,
                "drug_start_char": sent_start + m.start(), "drug_end_char": sent_start + m.end(),
                "dose_start_char": dose_start, "dose_end_char": dose_end, "note_role": note_role,
            })
    return mentions


def phase2_extract(cohort: str, batch_size: int = 200) -> None:
    t0 = time.time()
    out_dir = MIMIC_ROOT / f"accord_{cohort}"
    pairs = pd.read_csv(out_dir / f"{cohort}_note_pairs.csv.gz")
    print(f"[{cohort}] Loaded note pairs: {pairs.shape}")

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    raw_texts = pairs["TEXT"].fillna("").astype(str).tolist()
    expanded_texts = [expand_text(t[:100000]) for t in raw_texts]

    results = []
    errors = 0
    for i, doc in enumerate(nlp.pipe(expanded_texts, batch_size=batch_size)):
        try:
            sentences = build_sentences(doc)
            mentions = extract_mentions_from_doc(sentences, pairs["FINAL_ROLE"].iloc[i])
            for m in mentions:
                m["HADM_ID"] = pairs["HADM_ID"].iloc[i]
                m["SUBJECT_ID"] = pairs["SUBJECT_ID"].iloc[i]
                m["note_row_id"] = pairs["ROW_ID"].iloc[i]
            results.extend(mentions)
        except Exception:
            errors += 1
        if i % 2000 == 0:
            print(f"  [{cohort}] ...{i}/{len(pairs)} | mentions so far: {len(results):,} | {time.time()-t0:.0f}s")

    mentions_df = pd.DataFrame(results)
    print(f"[{cohort}] Done. Total mentions: {len(mentions_df):,}; errors: {errors}; {time.time()-t0:.0f}s")
    mentions_df.to_csv(out_dir / f"{cohort}_medication_mentions.csv.gz", index=False)

    classified = (mentions_df["action"] != "unclassified").sum() if len(mentions_df) else 0
    summary = {
        "cohort": cohort, "total_mentions": len(mentions_df), "classified_mentions": int(classified),
        "unique_admissions_with_mentions": int(mentions_df["HADM_ID"].nunique()) if len(mentions_df) else 0,
        "errors": errors, "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(out_dir / f"{cohort}_phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# PHASE 3 (generalized): contradiction mining -- VERBATIM logic, 4 rules
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


def pick_exemplar(candidates: pd.DataFrame, note_mentions: pd.DataFrame) -> pd.Series:
    if len(candidates) == 1:
        return candidates.iloc[0]
    dose_positions = note_mentions.loc[note_mentions["dose_value"].notna(), "drug_start_char"].to_numpy()
    if len(dose_positions) == 0:
        return candidates.sort_values("sentence_idx").iloc[0]
    dists = candidates["drug_start_char"].apply(lambda p: np.abs(dose_positions - p).min())
    best_idx = dists.idxmin()
    return candidates.loc[best_idx]


def mine_rule1(classified, mentions_by_note):
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
            new_n_candidates = n_sub[n_sub["action"] == req_n]
            new_d_candidates = d_sub[d_sub["action"] == req_d]
            note_n = mentions_by_note.get((hadm_id, "nursing"))
            note_d = mentions_by_note.get((hadm_id, "discharge"))
            new_n = pick_exemplar(new_n_candidates, note_n)
            new_d = pick_exemplar(new_d_candidates, note_d)
            out.append({"HADM_ID": hadm_id, "rule": 1, "drug": drug, "drug_class": None,
                        "nursing_action": new_n["action"], "discharge_action": new_d["action"],
                        "nursing_sentence": new_n["sentence"], "discharge_sentence": new_d["sentence"]})
    return out


def mine_rule2(classified, mentions_by_note):
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
                continue
            n_actions = set(n_sub["action"])
            d_actions = set(d_sub["action"])
            cond1 = "continuation" in n_actions and "cessation" in d_actions
            cond2 = "cessation" in n_actions and "continuation" in d_actions
            if not (cond1 or cond2):
                continue
            req_n, req_d = ("continuation", "cessation") if cond1 else ("cessation", "continuation")
            new_n_candidates = n_sub[n_sub["action"] == req_n]
            new_d_candidates = d_sub[d_sub["action"] == req_d]
            note_n = mentions_by_note.get((hadm_id, "nursing"))
            note_d = mentions_by_note.get((hadm_id, "discharge"))
            new_n = pick_exemplar(new_n_candidates, note_n)
            new_d = pick_exemplar(new_d_candidates, note_d)
            out.append({"HADM_ID": hadm_id, "rule": 2, "drug": f'{new_n["drug"]} vs {new_d["drug"]}',
                        "drug_class": cls, "nursing_action": new_n["action"], "discharge_action": new_d["action"],
                        "nursing_sentence": new_n["sentence"], "discharge_sentence": new_d["sentence"]})
    return out


def mine_rule3(all_mentions):
    from dictionaries import dose_pct_diff
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
            out.append({"HADM_ID": hadm_id, "rule": 3, "drug": drug, "drug_class": DRUG_TO_CLASS.get(drug),
                        "nursing_action": nr["action"], "discharge_action": dr["action"],
                        "nursing_sentence": nr["sentence"], "discharge_sentence": dr["sentence"]})
    return out


def mine_rule4(classified, all_mentions, mentions_by_note):
    out = []
    nursing_cessation = classified[(classified["note_role"] == "nursing") & (classified["action"] == "cessation")]
    discharge_drugs_by_hadm = all_mentions[all_mentions["note_role"] == "discharge"].groupby("HADM_ID")["drug"].apply(set)
    for hadm_id, group in nursing_cessation.groupby("HADM_ID"):
        discharge_drugs = discharge_drugs_by_hadm.get(hadm_id, set())
        for drug in set(group["drug"]):
            if drug in discharge_drugs:
                continue
            candidates = group[group["drug"] == drug]
            note_n = mentions_by_note.get((hadm_id, "nursing"))
            exemplar = pick_exemplar(candidates, note_n)
            out.append({"HADM_ID": hadm_id, "rule": 4, "drug": drug, "drug_class": DRUG_TO_CLASS.get(drug),
                        "nursing_action": exemplar["action"], "discharge_action": "absent",
                        "nursing_sentence": exemplar["sentence"], "discharge_sentence": ""})
    return out


def phase3_mine(cohort: str) -> None:
    t0 = time.time()
    out_dir = MIMIC_ROOT / f"accord_{cohort}"
    mentions = pd.read_csv(out_dir / f"{cohort}_medication_mentions.csv.gz")
    print(f"[{cohort}] Loaded mentions: {mentions.shape}")
    classified = mentions[mentions["action"] != "unclassified"].copy()
    mentions_by_note = {k: v for k, v in mentions.groupby(["HADM_ID", "note_role"])}

    r1 = mine_rule1(classified, mentions_by_note)
    r2 = mine_rule2(classified, mentions_by_note)
    r3 = mine_rule3(mentions)
    r4 = mine_rule4(classified, mentions, mentions_by_note)
    print(f"[{cohort}] rule1={len(r1)} rule2={len(r2)} rule3={len(r3)} rule4={len(r4)}")

    all_candidates = pd.DataFrame(r1 + r2 + r3 + r4)
    all_candidates.to_csv(out_dir / f"{cohort}_contradiction_candidates.csv.gz", index=False)
    summary = {
        "cohort": cohort, "rule1": len(r1), "rule2": len(r2), "rule3": len(r3), "rule4": len(r4),
        "total_candidates": len(all_candidates),
        "unique_admissions": int(all_candidates["HADM_ID"].nunique()) if len(all_candidates) else 0,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(out_dir / f"{cohort}_phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["phase1", "phase2", "phase3", "all"], default="all")
    ap.add_argument("--cohort", choices=["sepsis", "aki"], default=None,
                     help="restrict phase2/phase3 to one cohort")
    args = ap.parse_args()

    if args.stage in ("phase1", "all"):
        phase1_build_all()
    cohorts = [args.cohort] if args.cohort else list(COHORTS.keys())
    if args.stage in ("phase2", "all"):
        for c in cohorts:
            phase2_extract(c)
    if args.stage in ("phase3", "all"):
        for c in cohorts:
            phase3_mine(c)


if __name__ == "__main__":
    main()
