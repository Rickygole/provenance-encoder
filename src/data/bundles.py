"""Bundle -> tensors. One packing function serves all four arms.

Token budget is IDENTICAL across arms: every segment gets 2 reserved slots.
Arm A2 fills them with [ROLE=x] [AGE=y] literal tokens; every other arm fills
them with [PAD] and masks them out. Without this the decisive baseline would
silently get fewer content positions than the model it is meant to test.

Positions are LOCAL WITHIN SEGMENT. Segment order inside a bundle is not
semantic, so a position index running across the bundle would encode an
ordering that does not exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from .roles import ROLE_NAMES

RESERVED_PER_SEGMENT = 2
AGE_BUCKETS = [(6, "0h"), (24, "6h"), (72, "24h"), (168, "72h"), (float("inf"), "7d")]


def age_token(hours: float) -> str:
    for hi, name in AGE_BUCKETS:
        if hours < hi:
            return f"[AGE={name}]"
    return "[AGE=gt7d]"


def auth_bucket(hours: float, n_buckets: int = 4) -> int:
    """Coarse recency bucket. NOT an authority ordinal -- see roles.py."""
    for i, (hi, _) in enumerate(AGE_BUCKETS[:n_buckets]):
        if hours < hi:
            return i
    return n_buckets - 1


class BundleDataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path, tokenizer: Tokenizer, *, arm: str,
                 max_seq_len: int = 512, max_tokens_per_segment: int = 96,
                 mlm_prob: float = 0.15, seed: int = 0):
        self.rows = [json.loads(l) for l in Path(path).open()]
        self.tok = tokenizer
        self.arm = arm
        self.max_seq_len = max_seq_len
        self.max_seg = max_tokens_per_segment
        self.mlm_prob = mlm_prob
        self.rng = np.random.default_rng(seed)
        v = tokenizer.get_vocab()
        self.pad_id, self.cls_id = v["[PAD]"], v["[CLS]"]
        self.sep_id, self.mask_id = v["[SEP]"], v["[MASK]"]
        self.vocab_size = tokenizer.get_vocab_size()
        self.role_tok = {i: v.get(f"[ROLE={n}]", v["[UNK]"]) for i, n in ROLE_NAMES.items()}
        self.age_tok = {k: v.get(k, v["[UNK]"]) for k in
                        [f"[AGE={n}]" for n in ("0h", "6h", "24h", "72h", "7d", "gt7d")]}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        b = self.rows[idx]
        S = self.max_seq_len
        ids = np.full(S, self.pad_id, dtype=np.int64)
        role = np.zeros(S, dtype=np.int64)
        lpos = np.zeros(S, dtype=np.int64)
        dt = np.zeros(S, dtype=np.float32)
        mask = np.zeros(S, dtype=np.float32)

        cur = 0
        ids[cur], mask[cur], cur = self.cls_id, 1.0, cur + 1

        for seg in b["segments"]:
            if cur + RESERVED_PER_SEGMENT + 8 >= S:
                break
            r = int(seg["role"])
            hours = float(seg["delta_t_hours"])

            # Reserved slots -- identical budget in every arm.
            for j, tid in enumerate((self.role_tok[r], self.age_tok[age_token(hours)])):
                if self.arm == "A2":
                    ids[cur], mask[cur] = tid, 1.0
                else:
                    ids[cur], mask[cur] = self.pad_id, 0.0
                role[cur], dt[cur], lpos[cur] = r, hours, j
                cur += 1

            enc = self.tok.encode(seg["text"]).ids[: self.max_seg]
            room = S - cur - 1
            enc = enc[:room]
            if not enc:
                break
            n = len(enc)
            ids[cur:cur + n] = enc
            role[cur:cur + n] = r
            dt[cur:cur + n] = hours
            lpos[cur:cur + n] = np.arange(RESERVED_PER_SEGMENT, RESERVED_PER_SEGMENT + n)
            mask[cur:cur + n] = 1.0
            cur += n

            if cur < S:
                ids[cur], role[cur], dt[cur], mask[cur] = self.sep_id, r, hours, 1.0
                lpos[cur] = RESERVED_PER_SEGMENT + n
                cur += 1

        labels = np.full(S, -100, dtype=np.int64)
        maskable = np.where((mask > 0) & (ids != self.cls_id) & (ids != self.sep_id))[0]
        # Never mask the A2 provenance tokens: predicting them is a different task.
        if self.arm == "A2":
            role_ids_set = set(self.role_tok.values()) | set(self.age_tok.values())
            maskable = np.array([i for i in maskable if ids[i] not in role_ids_set], dtype=np.int64)
        if maskable.size:
            k = max(1, int(round(self.mlm_prob * maskable.size)))
            pick = self.rng.choice(maskable, size=k, replace=False)
            labels[pick] = ids[pick]
            u = self.rng.random(k)
            ids[pick[u < 0.8]] = self.mask_id
            rnd = pick[(u >= 0.8) & (u < 0.9)]
            if rnd.size:
                ids[rnd] = self.rng.integers(5, self.vocab_size, size=rnd.size)

        return {
            "input_ids": torch.from_numpy(ids),
            "local_positions": torch.from_numpy(np.clip(lpos, 0, S - 1)),
            "attention_mask": torch.from_numpy(mask),
            "role_ids": torch.from_numpy(role),
            "auth_ids": torch.from_numpy(
                np.array([auth_bucket(x) for x in dt], dtype=np.int64)),
            "event_times": torch.from_numpy(dt),
            "mlm_labels": torch.from_numpy(labels),
        }


def n_distinct_roles(batch: dict) -> torch.Tensor:
    """Per-sequence count of distinct non-pad roles. Drives the inertness guard."""
    r = batch["role_ids"].clone()
    r[batch["attention_mask"] == 0] = 0
    return torch.tensor([len(set(row[row > 0].tolist())) for row in r])
