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

from .roles import N_ROLES, ROLE_NAMES

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
                 mlm_prob: float = 0.15, seed: int = 0, permute_roles: bool = False,
                 random_roles: bool = False, length_roles: bool = False,
                 segment_index_roles: bool = False, shuffle_segments: bool = False,
                 global_positions: bool = False, misaligned_blocks: bool = False):
        self.rows = [json.loads(l) for l in Path(path).open()]
        self.tok = tokenizer
        self.arm = arm
        # CONTROLS.
        #
        # WARNING on permute_roles: a GLOBAL derangement is VACUOUS. `A` is a free
        # KxK table, so under a global relabeling sigma the model simply learns
        # A'[sigma(r), sigma(r')] = A[r, r']. The loss surface is isomorphic and
        # the result is fixed at the uncorrupted value by construction. It has
        # zero statistical power. Kept only for the regression test that proves
        # this; never cite it as evidence. Use random_roles or segment_index_roles.
        #
        # random_roles assigns each segment an independent random role, which
        # destroys cross-example consistency and is a real (if weak) test.
        #
        # segment_index_roles is the decisive POSITIVE competing hypothesis: keys
        # the bias on ordinal position in the bundle, carrying zero provenance
        # meaning. It is only meaningful with shuffle_segments=True, because
        # bundles are built in chronological order and role therefore correlates
        # with position -- without shuffling, A5 is a re-run of A4.
        self.permute_roles = permute_roles
        self.random_roles = random_roles
        # length_roles replaces the role id with a note-length bucket: a per-segment
        # categorical that is genuinely informative but carries no authority
        # semantics. If this helps as much as real roles, the mechanism is "any
        # informative per-segment category", not document provenance.
        self.length_roles = length_roles
        self.segment_index_roles = segment_index_roles
        self.shuffle_segments = shuffle_segments
        # THE BASELINE FIX. Local-within-segment positions mean the vanilla arm has
        # NO global position coordinate: token 5 of segment 1 and token 5 of
        # segment 5 receive identical position embeddings. Since
        # local_position + segment_index = global_position, an arm keyed on
        # segment index hands the model back exactly the coordinate this deletes.
        # Every measured "segmentation" gain may therefore be a positionally
        # handicapped baseline buying position back. This flag restores standard
        # absolute positions so the baseline is not crippled by construction.
        self.global_positions = global_positions
        self.misaligned_blocks = misaligned_blocks
        # Must be a DERANGEMENT -- no fixed points. A permutation that maps the
        # most common role to itself leaves most tokens correctly labelled and is
        # not a control at all.
        _rng = np.random.default_rng(9876)
        _src = np.arange(1, N_ROLES)
        while True:
            _p = _rng.permutation(_src)
            if not (_p == _src).any():
                break
        self.role_perm = {0: 0, **{int(s): int(d) for s, d in zip(_src, _p)}}
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

        rr = np.random.default_rng(idx)  # deterministic per example
        segments = list(b["segments"])
        if self.shuffle_segments:
            # Bundles are assembled chronologically, so role correlates with
            # position. Shuffling decorrelates them; without this, a segment-index
            # control is indistinguishable from the role condition.
            segments = [segments[i] for i in rr.permutation(len(segments))]
        for seg_i, seg in enumerate(segments):
            if cur + RESERVED_PER_SEGMENT + 8 >= S:
                break
            r = int(seg["role"])
            if self.random_roles:
                r = int(rr.integers(1, N_ROLES))
            elif self.permute_roles:
                r = self.role_perm.get(r, r)
            elif self.length_roles:
                # 6 length buckets mapped onto role slots 1..6
                L = len(seg["text"])
                r = 1 + min(5, int(L // 250))
            elif self.segment_index_roles:
                # A5: ordinal position in the bundle. Zero provenance meaning.
                r = 1 + min(N_ROLES - 2, seg_i)
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
            if self.global_positions:
                lpos[cur:cur + n] = np.arange(cur, cur + n)
            else:
                lpos[cur:cur + n] = np.arange(RESERVED_PER_SEGMENT,
                                              RESERVED_PER_SEGMENT + n)
            mask[cur:cur + n] = 1.0
            cur += n

            if cur < S:
                ids[cur], role[cur], dt[cur], mask[cur] = self.sep_id, r, hours, 1.0
                lpos[cur] = RESERVED_PER_SEGMENT + n
                cur += 1

        if self.misaligned_blocks:
            # THE IDENTIFYING CONTROL. Same number of contiguous blocks, same
            # size distribution, but boundaries placed at random offsets that do
            # NOT coincide with document boundaries. Isolates *document*
            # structure from *any* coarse blocking of the sequence. If this ties
            # the aligned version, the word "document" comes out of every claim.
            n_real = max(1, len(segments))
            valid = int(mask.sum())
            cuts = sorted(rr.choice(np.arange(1, max(2, valid)),
                                    size=min(n_real - 1, max(1, valid - 1)),
                                    replace=False).tolist()) if n_real > 1 else []
            blk, prev = np.zeros(S, dtype=np.int64), 0
            for bi, c in enumerate(cuts + [valid]):
                blk[prev:c] = 1 + (bi % (N_ROLES - 2))
                prev = c
            role = np.where(mask > 0, blk, 0)

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
