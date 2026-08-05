"""ACCORD Phase 2 rebuild: medication mention extraction, WITH offsets.

Reproduces extract_medications_v4 from ACCORD_Phase2_Medication_Extraction
(the version actually run over the full cohort -- 219,683 mentions) using the
same dictionaries (accord.dictionaries: ABBREV, VARIANTS, NEGATION,
ACTION_PATTERNS, HF_DRUGS / ALL_DRUGS, resolve_action) VERBATIM, with the
following changes only:

  1. Sentence segmentation uses spacy's rule-based `sentencizer` component
     instead of the full en_core_web_sm pipeline (tagger + parser + NER).
     The reference only used spacy for sentence boundaries; running the full
     parser to get them cost ~6,300s for 19,566 notes on Colab CPU. The
     sentencizer is punctuation-based and gets the same boundaries for this
     purpose at a small fraction of the cost. This does not touch any of the
     drug/action/negation dictionaries.

  2. For every mention we now additionally record: sentence index within the
     note (position in the filtered-sentence list, matching how the
     reference indexed its context window), and character offsets --
     sentence span and drug-match span -- in the abbreviation/variant
     -expanded note text. The reference discarded all of this, which is what
     made the .iloc[0] exemplar bug possible: nothing in the mentions table
     said *where* a mention was, so mining could not tell mentions apart.

  3. The sentence field is stored in full (not truncated to 200 chars as the
     reference did) since we need it as ground truth for the OLD-vs-NEW
     exemplar-correctness measurement in phase 3.

Local only. Prints aggregate counts. Writes note-derived sentence text only
under local_only/ (gitignored), never elsewhere.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import spacy

from dictionaries import ABBREV, ACTION_PATTERNS, ALL_DRUGS, NEGATION, VARIANTS, parse_dose, resolve_action

HERE = Path(__file__).resolve().parent
OUT = HERE / "local_only"

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
    """Filtered (len(stripped) > 10) sentence list with offsets INTO the
    expanded text, matching reference's `sentences = [sent.text.strip() for
    sent in doc.sents if len(sent.text.strip()) > 10]` but keeping spans."""
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
            dose_pat = re.compile(
                re.escape(drug) + r"\s*(\d+(?:\.\d+)?\s*(?:mg|mcg|mEq|units?|ml|g))", re.IGNORECASE
            )
            dose_match = dose_pat.search(sent_text)
            if dose_match:
                dose = dose_match.group(1).strip()
                dose_start = sent_start + dose_match.start(1)
                dose_end = sent_start + dose_match.end(1)

            normalized = ABBREV.get(drug.lower(), drug.lower())
            normalized = VARIANTS.get(normalized, normalized)

            parsed = parse_dose(dose)
            dose_value, dose_unit = parsed if parsed else (None, None)

            mentions.append(
                {
                    "drug": normalized,
                    "dose": dose,
                    "dose_value": dose_value,
                    "dose_unit": dose_unit,
                    "action": final_action,
                    "negated": is_negated,
                    "sentence": sent_text,
                    "sentence_idx": i,
                    "sent_start_char": sent_start,
                    "sent_end_char": sent_end,
                    "drug_start_char": sent_start + m.start(),
                    "drug_end_char": sent_start + m.end(),
                    "dose_start_char": dose_start,
                    "dose_end_char": dose_end,
                    "note_role": note_role,
                }
            )
    return mentions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    pairs = pd.read_csv(OUT / "hf_note_pairs.csv.gz")
    print(f"Loaded note pairs: {pairs.shape}")
    print(f"Roles: {pairs['FINAL_ROLE'].value_counts().to_dict()}")

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    raw_texts = pairs["TEXT"].fillna("").astype(str).tolist()
    expanded_texts = [expand_text(t[:100000]) for t in raw_texts]

    print("\nSegmenting sentences and extracting mentions...")
    results = []
    errors = 0
    for i, doc in enumerate(nlp.pipe(expanded_texts, batch_size=args.batch_size)):
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
            print(f"  ...{i}/{len(pairs)} notes | mentions so far: {len(results):,} | errors: {errors} | {time.time()-t0:.0f}s")

    mentions_df = pd.DataFrame(results)
    print(f"\nDone. Total mentions: {len(mentions_df):,}")
    print(f"Errors: {errors}")
    print(f"Time: {time.time()-t0:.0f}s")
    print(f"\nAction distribution:\n{mentions_df['action'].value_counts()}")

    mentions_df.to_csv(OUT / "hf_medication_mentions.csv.gz", index=False)
    print(f"\nSaved {OUT / 'hf_medication_mentions.csv.gz'}")

    classified = (mentions_df["action"] != "unclassified").sum()
    with_dose = mentions_df["dose_value"].notna().sum()
    summary = {
        "total_mentions": len(mentions_df),
        "classified_mentions": int(classified),
        "unclassified_mentions": int(len(mentions_df) - classified),
        "mentions_with_parsed_dose": int(with_dose),
        "unique_admissions_with_mentions": int(mentions_df["HADM_ID"].nunique()),
        "unique_drugs_found": int(mentions_df["drug"].nunique()),
        "errors": errors,
        "elapsed_sec": round(time.time() - t0, 1),
        "reference_total_mentions": 219683,
        "reference_classified_mentions": 88336,
        "reference_unique_drugs": 70,
        "reference_unique_admissions": 9732,
    }
    with open(OUT / "phase2_medication_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
