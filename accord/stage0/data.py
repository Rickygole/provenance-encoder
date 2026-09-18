"""
Stage 0 re-diagnosis: data pipeline.

Reproduces the ATCGSelfSupervisedDataset task (matched nursing/discharge note
pairs from the same HADM_ID vs. mismatched pairs from different admissions),
but fixes the two data-hygiene bugs present in the original notebook:

  1. The original split hf_sample by ROW POSITION after a random shuffle, with
     no regard for SUBJECT_ID. A subject with two admissions in the 20%
     subsample could have one admission in "train" and the other in "dev".
     We split by SUBJECT_ID instead, so no patient straddles the split.

  2. The original built a fresh ATCGSelfSupervisedDataset for dev and drew the
     negative (mismatched) partner with `random.choice` INSIDE __getitem__,
     which is called by the DataLoader every epoch. Since no seed was fixed
     per-example, the dev set's negative pairs were different every epoch --
     "dev accuracy" was evaluated against a moving target. We instead build a
     fixed list of (nursing_row, discharge_row, label) tuples once, with a
     dedicated RNG, and reuse that exact list for every epoch.

NEVER print or persist clinical note text from this module. Only lengths,
counts, ids and numeric aggregates may leave this process.
"""
import os
import random
import re
import warnings
from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset

MIMIC_LOCAL_ONLY_DIR = os.environ.get("MIMIC_LOCAL_ONLY_DIR", os.path.expanduser("~/mimic_local_only"))
DATA_CSV = os.path.join(MIMIC_LOCAL_ONLY_DIR, "accord/hf_note_pairs.csv.gz")
MAX_LENGTH = 256
SAMPLE_FRAC = 0.2
DEV_FRAC = 0.1
SAMPLE_SEED = 42        # matches the original notebook's random_state=42 for hf_sample
SPLIT_SEED = 1337        # subject-level train/dev split
PAIR_SEED = 2024          # fixed matched/mismatched pair construction (train)
DEV_PAIR_SEED = 2025      # fixed matched/mismatched pair construction (dev)

# Candidate section headers used by the 'meds' window strategy to locate a
# discharge-medication section inside a MIMIC-III discharge summary.
_MED_SECTION_PATTERNS = [
    r"discharge medications?\s*:",
    r"medications on discharge\s*:",
    r"discharge medication list\s*:",
]


def load_merged_pairs(csv_path: str = DATA_CSV) -> pd.DataFrame:
    """Load hf_note_pairs.csv.gz and merge nursing+discharge notes on HADM_ID,
    exactly as cell 15 of the reference notebook does."""
    df = pd.read_csv(csv_path)
    nursing = df[df["FINAL_ROLE"] == "nursing"].copy()
    discharge = df[df["FINAL_ROLE"] == "discharge"].copy()
    merged = nursing.merge(
        discharge, on=["HADM_ID", "SUBJECT_ID"], suffixes=("_nursing", "_discharge")
    )
    return merged[["HADM_ID", "SUBJECT_ID", "TEXT_nursing", "TEXT_discharge"]]


def build_subject_split(
    csv_path: str = DATA_CSV,
    sample_frac: float = SAMPLE_FRAC,
    dev_frac: float = DEV_FRAC,
    sample_seed: int = SAMPLE_SEED,
    split_seed: int = SPLIT_SEED,
):
    """Reproduce the 20% subsample (n=1956, same random_state as the
    reference notebook), then split by SUBJECT_ID (not by row) into
    train/dev pools of admissions."""
    merged = load_merged_pairs(csv_path)
    sample_size = int(len(merged) * sample_frac)
    sample = merged.sample(n=sample_size, random_state=sample_seed).reset_index(drop=True)

    subjects = sample["SUBJECT_ID"].unique().tolist()
    rng = random.Random(split_seed)
    rng.shuffle(subjects)

    target_dev_admissions = int(len(sample) * dev_frac)
    admissions_per_subject = sample["SUBJECT_ID"].value_counts().to_dict()

    dev_subjects = set()
    running = 0
    for s in subjects:
        if running >= target_dev_admissions:
            break
        dev_subjects.add(s)
        running += admissions_per_subject[s]

    is_dev = sample["SUBJECT_ID"].isin(dev_subjects)
    train_df = sample[~is_dev].reset_index(drop=True)
    dev_df = sample[is_dev].reset_index(drop=True)

    assert set(train_df["SUBJECT_ID"]).isdisjoint(set(dev_df["SUBJECT_ID"])), (
        "subject leakage between train and dev"
    )
    return train_df, dev_df


def build_fixed_pairs(n_rows: int, seed: int, neg_ratio: int = 1):
    """Build a FIXED (matched + mismatched) pair list for a split of n_rows
    admissions. Positive: (i, i, 1). Negative: (i, j, 0) with j != i, drawn
    once with a dedicated RNG -- not resampled per epoch."""
    rng = random.Random(seed)
    pairs = [(i, i, 1) for i in range(n_rows)]
    for i in range(n_rows):
        for _ in range(neg_ratio):
            j = rng.choice([k for k in range(n_rows) if k != i])
            pairs.append((i, j, 0))
    return pairs


def _tail_token_ids(tokenizer, text: str, max_length: int):
    """Tokenize the LAST max_length tokens of text (instead of the first),
    keeping [CLS] ... [SEP] special tokens."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ids = tokenizer(str(text), add_special_tokens=False, truncation=False)["input_ids"]
    budget = max_length - 2  # room for [CLS] and [SEP]
    tail_ids = ids[-budget:] if len(ids) > budget else ids
    input_ids = [tokenizer.cls_token_id] + tail_ids + [tokenizer.sep_token_id]
    attn = [1] * len(input_ids)
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        attn = attn + [0] * pad_len
    return input_ids[:max_length], attn[:max_length]


def _meds_window_text(text: str):
    """Locate a discharge-medications section header and return a character
    window starting there. Falls back to None if no header is found (caller
    should fall back to the tail-token strategy)."""
    s = str(text)
    for pat in _MED_SECTION_PATTERNS:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return s[m.start():]
    return None


def tokenize_window(tokenizer, text: str, max_length: int, window: str):
    """Return (input_ids, attention_mask) as python lists of length
    max_length, under one of three windowing strategies:
      'head' - the original strategy: tokenizer truncation=True keeps the
               FIRST max_length tokens.
      'tail' - keep the LAST max_length tokens of the note.
      'meds' - if a discharge-medications header is found, head-truncate
               from that point; else fall back to 'tail'.
    """
    if window == "head":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enc = tokenizer(
                str(text), max_length=max_length, truncation=True,
                padding="max_length", return_tensors=None,
            )
        return enc["input_ids"], enc["attention_mask"]
    elif window == "tail":
        return _tail_token_ids(tokenizer, text, max_length)
    elif window == "meds":
        snippet = _meds_window_text(text)
        if snippet is None:
            return _tail_token_ids(tokenizer, text, max_length)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enc = tokenizer(
                snippet, max_length=max_length, truncation=True,
                padding="max_length", return_tensors=None,
            )
        return enc["input_ids"], enc["attention_mask"]
    else:
        raise ValueError(f"unknown window strategy: {window}")


@dataclass
class TokenizedNotes:
    nursing_ids: torch.Tensor      # (N, L)
    nursing_mask: torch.Tensor     # (N, L)
    discharge_ids: torch.Tensor    # (N, L)
    discharge_mask: torch.Tensor   # (N, L)


def tokenize_split(df: pd.DataFrame, tokenizer, window: str, max_length: int = MAX_LENGTH) -> TokenizedNotes:
    """Pre-tokenize every admission in df ONCE under the given window
    strategy (avoids re-tokenizing identical text every epoch)."""
    n_ids, n_mask, d_ids, d_mask = [], [], [], []
    for _, row in df.iterrows():
        ni, nm = tokenize_window(tokenizer, row["TEXT_nursing"], max_length, window)
        di, dm = tokenize_window(tokenizer, row["TEXT_discharge"], max_length, window)
        n_ids.append(ni)
        n_mask.append(nm)
        d_ids.append(di)
        d_mask.append(dm)
    return TokenizedNotes(
        nursing_ids=torch.tensor(n_ids, dtype=torch.long),
        nursing_mask=torch.tensor(n_mask, dtype=torch.long),
        discharge_ids=torch.tensor(d_ids, dtype=torch.long),
        discharge_mask=torch.tensor(d_mask, dtype=torch.long),
    )


class FixedPairDataset(Dataset):
    """Wraps pre-tokenized notes + a FIXED pair list. No text ever touches
    this class; no randomness happens at __getitem__ time."""

    def __init__(self, tokenized: TokenizedNotes, pairs: list):
        self.tok = tokenized
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        n_idx, d_idx, label = self.pairs[idx]
        return {
            "nursing_input_ids": self.tok.nursing_ids[n_idx],
            "nursing_attention_mask": self.tok.nursing_mask[n_idx],
            "discharge_input_ids": self.tok.discharge_ids[d_idx],
            "discharge_attention_mask": self.tok.discharge_mask[d_idx],
            "label": torch.tensor(float(label), dtype=torch.float),
        }


def build_datasets(tokenizer, window: str, csv_path: str = DATA_CSV, max_length: int = MAX_LENGTH):
    """Full pipeline: subject-level split -> fixed pairs -> tokenize -> Dataset."""
    train_df, dev_df = build_subject_split(csv_path)
    train_pairs = build_fixed_pairs(len(train_df), seed=PAIR_SEED)
    dev_pairs = build_fixed_pairs(len(dev_df), seed=DEV_PAIR_SEED)

    train_tok = tokenize_split(train_df, tokenizer, window, max_length)
    dev_tok = tokenize_split(dev_df, tokenizer, window, max_length)

    train_ds = FixedPairDataset(train_tok, train_pairs)
    dev_ds = FixedPairDataset(dev_tok, dev_pairs)
    meta = {
        "n_train_admissions": len(train_df),
        "n_dev_admissions": len(dev_df),
        "n_train_pairs": len(train_pairs),
        "n_dev_pairs": len(dev_pairs),
        "n_train_subjects": int(train_df["SUBJECT_ID"].nunique()),
        "n_dev_subjects": int(dev_df["SUBJECT_ID"].nunique()),
    }
    return train_ds, dev_ds, meta
