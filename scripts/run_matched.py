"""Matched-protocol evaluation harness.

The existing comparison (scripts/finetune_ours.py vs scripts/finetune_pubmedbert.py)
is confounded on six axes at once: input budget (256/36 vs 512/110 tokens), epochs
(3 vs 1), model selection (best-of-3-epochs on the reported val set vs single final
epoch), learning rate (5e-5 vs 2e-5, no sweep), seeds (3 vs 1), and extra inputs
(role/segment/delta_t vs text-only). This script fixes the first five -- the sixth
(extra inputs) is architectural, not a protocol bug, and is reported rather than
erased.

Five systems, one protocol, one table:
  (a) metadata-only logistic regression   -- no text
  (b) TF-IDF + logistic regression        -- full text, the floor
  (c) our encoder, random init
  (d) our encoder, pretrained (block-mask MLM)
  (e) PubMedBERT + concatenation

Shared, explicit knobs for (c)/(d)/(e): --max-len (token budget, sequence length),
--chars-per-note (CHARACTER budget per note, applied identically before either
tokenizer runs), --epochs, --lr-grid, --seeds. Selection is identical for every
neural arm: best epoch by val AUROC, LR chosen by mean best-epoch val AUROC across
seeds, same epoch budget, same LR grid.

The floor rows are always computed and always printed first -- there is no code
path that emits a neural number without them.

Local only. Reads MIMIC-derived local_only/task/*.jsonl. Writes only scalars to
results/.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.bundles import RESERVED_PER_SEGMENT  # noqa: E402
from src.data.roles import N_ROLES  # noqa: E402
from src.model.config import EncoderConfig, preset  # noqa: E402
from src.model.encoder import ProvenanceEncoder  # noqa: E402

LOCAL = ROOT / "local_only"
RESULTS = ROOT / "results"
DEFAULT_CKPT = LOCAL / "runs" / "A3_10M_st3000_s0_hardblk_shuf"
ALL_SYSTEMS = ["metadata", "tfidf", "ours_random", "ours_pretrained", "pubmedbert"]


# ============================================================================
# Data
# ============================================================================

def load_rows(task_dir: Path, split: str, limit: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in (task_dir / f"{split}.jsonl").open()]
    if limit:
        rows = rows[:limit]
    return rows


def subsample_group_disjoint(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Subject-group-disjoint, prevalence-preserving subsample of `rows` down to
    ~n examples. Used by the label-budget ladder (scripts/run_label_ladder.sh):
    pretraining's benefit is supposed to live in the LOW-label regime, and that
    claim is only honest if the low-label subsamples are built with the same
    discipline as the original splits.

    Operates entirely WITHIN the split it is given (in practice, only ever
    called on the already-selected train split -- val/test are untouched), so
    it cannot by itself introduce cross-split leakage. Whole subjects are kept
    or dropped together (never split across the boundary of "in the
    subsample" / "not in the subsample"), and positive- and negative-carrying
    subjects are drawn from separately shuffled pools so the achieved positive
    rate tracks `rows`' positive rate rather than drifting with subsample size.
    """
    if n <= 0 or n >= len(rows):
        return rows
    by_subj: dict[str, list[dict]] = {}
    for r in rows:
        by_subj.setdefault(r["subject_id"], []).append(r)
    prevalence = sum(r["label"] for r in rows) / len(rows)
    pos_subjects = [s for s, rs in by_subj.items() if any(r["label"] for r in rs)]
    neg_subjects = [s for s, rs in by_subj.items() if not any(r["label"] for r in rs)]
    rng = random.Random(seed)
    rng.shuffle(pos_subjects)
    rng.shuffle(neg_subjects)
    target_pos = round(n * prevalence)
    target_neg = n - target_pos

    out: list[dict] = []
    n_pos = n_neg = 0
    for s in pos_subjects:
        if n_pos >= target_pos:
            break
        rs = by_subj[s]
        out.extend(rs)
        n_pos += sum(r["label"] for r in rs)
    for s in neg_subjects:
        if n_neg >= target_neg:
            break
        rs = by_subj[s]
        out.extend(rs)
        n_neg += sum(1 - r["label"] for r in rs)
    rng.shuffle(out)
    return out


def budget_rows(rows: list[dict], chars_per_note: int) -> list[dict]:
    """Truncate every segment's text to chars_per_note CHARACTERS.

    Applied identically before either tokenizer runs, so the two neural arms see
    the same raw information and differ only in how their own tokenizer chunks
    it. This is what makes token counts downstream comparable-but-different
    rather than silently unequal.
    """
    out = []
    for r in rows:
        segs = [{**s, "text": s["text"][:chars_per_note]} for s in r["segments"]]
        out.append({**r, "segments": segs})
    return out


# ============================================================================
# (a) metadata-only, (b) TF-IDF -- the floor
# ============================================================================

def meta_features(rows: list[dict]) -> np.ndarray:
    X = []
    for r in rows:
        segs = r["segments"]
        lens = [len(s["text"]) for s in segs]
        ts = [s["delta_t_hours"] for s in segs]
        counts = [0] * N_ROLES
        for s in segs:
            counts[s["role"]] += 1
        X.append(counts + [
            len(segs), sum(lens), float(np.mean(lens)), float(np.max(lens)),
            float(np.min(ts)), float(np.max(ts)), float(np.mean(ts)),
            float(np.max(ts) - np.min(ts)),
            float(np.mean(np.diff(sorted(ts)))) if len(ts) > 1 else 0.0,
            len(segs) / max(np.max(ts), 1e-3),
        ])
    return np.array(X, dtype=np.float64)


def run_metadata(train_rows, val_rows) -> dict:
    ytr = np.array([r["label"] for r in train_rows])
    yva = np.array([r["label"] for r in val_rows])
    sc = StandardScaler().fit(meta_features(train_rows))
    Xtr, Xva = sc.transform(meta_features(train_rows)), sc.transform(meta_features(val_rows))
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1, C=1.0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]
    return {"system": "metadata_lr", "family": "floor", "y": yva.tolist(), "p": p.tolist(),
            "extra_inputs": "counts/lengths/timing, NO TEXT", "params": int(Xtr.shape[1] + 1),
            "pretraining": "none", "note": "no text whatsoever"}


def text_of(rows: list[dict]) -> list[str]:
    return [" ".join(s["text"] for s in r["segments"]) for r in rows]


def run_tfidf(train_rows, val_rows) -> dict:
    ytr = np.array([r["label"] for r in train_rows])
    yva = np.array([r["label"] for r in val_rows])
    v = TfidfVectorizer(max_features=60_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xtr = v.fit_transform(text_of(train_rows))
    Xva = v.transform(text_of(val_rows))
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1, C=1.0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]
    return {"system": "tfidf_lr", "family": "floor", "y": yva.tolist(), "p": p.tolist(),
            "extra_inputs": "none (bag of words, full text)", "params": int(Xtr.shape[1] + 1),
            "pretraining": "none", "note": "full bundle text, no truncation"}


# ============================================================================
# (c)/(d) our encoder
# ============================================================================

class OursDataset(torch.utils.data.Dataset):
    """Char-budgeted bundle -> tensors for the from-scratch encoder.

    Unlike scripts/finetune_ours.py's TaskDS, the per-note cap is applied in
    CHARACTERS (via budget_rows, before this class ever sees the text) rather
    than tokens. This class does not truncate tokens per segment at all -- only
    the overall sequence length max_len bounds how much fits, exactly mirroring
    how OursBert/BertDataset below behaves under the same max_len.
    """

    def __init__(self, rows, tok: Tokenizer, max_len: int, pos_bound: int,
                 seg_tokens: int | None = None, semantic_roles: bool = False):
        self.rows, self.tok, self.max_len = rows, tok, max_len
        self.pos_bound = pos_bound  # nn.Embedding row count for local_positions
        # PRETRAINING-CONVENTION COMPATIBILITY. src/data/bundles.py caps each
        # segment at seg_tokens (default 36) and lays local positions in
        # [RESERVED_PER_SEGMENT, RESERVED_PER_SEGMENT + n). If fine-tuning uses
        # a different cap or starts positions at 0, a pretrained checkpoint is
        # asked to interpret position rows that never received a gradient --
        # which fully explains ours_pretrained scoring BELOW ours_random.
        self.seg_tokens = seg_tokens
        self.semantic_roles = semantic_roles
        v = tok.get_vocab()
        self.pad, self.cls, self.sep = v["[PAD]"], v["[CLS]"], v["[SEP]"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        S = self.max_len
        ids = np.full(S, self.pad, dtype=np.int64)
        role = np.zeros(S, dtype=np.int64)
        lpos = np.zeros(S, dtype=np.int64)
        dt = np.zeros(S, dtype=np.float32)
        mask = np.zeros(S, dtype=np.float32)
        cur = 0
        ids[cur], mask[cur] = self.cls, 1.0
        cur += 1
        n_tok, n_seg = 0, 0
        # Ration max_len ACROSS notes. Without a per-segment cap the first note
        # consumed the whole budget and the loop broke, so the neural arms saw
        # 1.39 of 6.78 notes while the floor saw all of them -- a 6x information
        # asymmetry that was reported as a "matched" protocol.
        per_seg_tok = max(8, (S - 2) // 8)
        if self.seg_tokens is not None:
            per_seg_tok = min(per_seg_tok, self.seg_tokens)
        for k, s in enumerate(r["segments"], start=1):
            enc = self.tok.encode(s["text"]).ids[:per_seg_tok]
            if not enc:
                continue
            if cur + len(enc) + 1 >= S:
                continue
            n = len(enc)
            ids[cur:cur + n] = enc
            role[cur:cur + n] = (int(s["role"]) if self.semantic_roles
                                 else 1 + min(N_ROLES - 2, k - 1))
            lpos[cur:cur + n] = np.arange(RESERVED_PER_SEGMENT,
                                          RESERVED_PER_SEGMENT + n)
            dt[cur:cur + n] = float(s["delta_t_hours"])
            mask[cur:cur + n] = 1.0
            cur += n
            ids[cur], role[cur] = self.sep, role[cur - 1]
            mask[cur], dt[cur] = 1.0, dt[cur - 1]
            lpos[cur] = RESERVED_PER_SEGMENT + n
            cur += 1
            n_tok += n + 1
            n_seg += 1
        lpos = np.clip(lpos, 0, self.pos_bound - 1)
        return {
            "input_ids": torch.from_numpy(ids),
            "local_positions": torch.from_numpy(lpos),
            "attention_mask": torch.from_numpy(mask),
            "role_ids": torch.from_numpy(role),
            "auth_ids": torch.zeros(S, dtype=torch.long),
            "event_times": torch.from_numpy(dt),
            "label": torch.tensor(float(r["label"])),
            "tokens_used": n_tok,
            "segs_used": n_seg,
            "segs_total": len(r["segments"]),
        }


class OursClf(nn.Module):
    def __init__(self, enc: ProvenanceEncoder):
        super().__init__()
        self.enc = enc
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(enc.cfg.d_model, 1)

    def forward(self, b):
        h = self.enc(b["input_ids"], b["local_positions"], b["attention_mask"],
                     b["role_ids"], b["auth_ids"], b["event_times"])
        return self.head(self.drop(h[:, 0])).squeeze(-1)


def read_convention(init: str, ckpt: Path | None) -> dict:
    """The input convention a checkpoint was TRAINED under.

    Needed in main() where the datasets are built, which is a different scope
    from build_ours_model(). Kept as one function so the two paths cannot drift.
    """
    if init != "pretrained":
        return {"seg_tokens": None, "semantic_roles": False}
    ck = torch.load(Path(ckpt) / "model.pt", map_location="cpu", weights_only=False)
    conv = ck.get("input_convention")
    if conv is None:
        raise SystemExit(
            f"{ckpt}/model.pt has no input_convention block -- it predates the "
            "fix that records how it was trained. Re-pretrain with the current "
            "scripts/pretrain.py; evaluating under a guessed convention is how "
            "ours_pretrained scored below ours_random.")
    return conv


def build_ours_model(init: str, ckpt: Path | None, max_len: int, device,
                      tokenizer_dir: Path = LOCAL / "tokenizer"):
    tok = Tokenizer.from_file(str(Path(tokenizer_dir) / "wordpiece.json"))
    if init == "pretrained":
        ck = torch.load(Path(ckpt) / "model.pt", map_location="cpu", weights_only=False)
        cfg = EncoderConfig(**{k: v for k, v in ck["cfg"].items()
                               if k in EncoderConfig.__dataclass_fields__ and
                               k not in ("use_role_embed", "use_authority_bias",
                                         "use_time_bias", "use_prefix_tokens")})
        for k in ("alibi_bias", "rpe_bias", "hard_block_mask"):
            setattr(cfg, k, ck["cfg"].get(k, False))
        enc = ProvenanceEncoder(cfg)
        enc.load_state_dict(ck["model"])
        pos_bound = cfg.max_seq_len  # pos_embed table size actually loaded from ckpt
        pretraining_desc = f"block-mask MLM, {ckpt}"
        # Read the convention this checkpoint was TRAINED under. Never hardcode:
        # a checkpoint pretrained with different --seg-tokens or a role control
        # would otherwise be silently mis-evaluated.
        conv = ck.get("input_convention")
        if conv is None:
            raise SystemExit(
                f"{ckpt}/model.pt has no input_convention block -- it predates "
                "the fix that records how it was trained. Re-pretrain with the "
                "current scripts/pretrain.py; evaluating it under a guessed "
                "convention is how ours_pretrained scored below ours_random.")
    else:
        cfg = EncoderConfig(arm="A3", vocab_size=tok.get_vocab_size(), d_model=384,
                            n_layers=6, n_heads=6, d_ff=1536, max_seq_len=max_len)
        cfg.hard_block_mask = True
        enc = ProvenanceEncoder(cfg)
        pos_bound = max_len
        pretraining_desc = "none"
    if init != "pretrained":
        # Random init defines its own convention; match pretraining's so the two
        # arms remain comparable to each other as well as to the checkpoint.
        conv = {"seg_tokens": None, "semantic_roles": False}
    model = OursClf(enc).to(device)
    return model, tok, pos_bound, cfg.arm, pretraining_desc, conv


@torch.no_grad()
def eval_ours(model, dl, dev):
    model.eval()
    ps, ys = [], []
    for b in dl:
        y = b.pop("label")
        b = {k: v.to(dev) for k, v in b.items()
             if k in ("input_ids", "local_positions", "attention_mask", "role_ids",
                       "auth_ids", "event_times")}
        ps += torch.sigmoid(model(b)).float().cpu().tolist()
        ys += y.tolist()
    model.train()
    return roc_auc_score(ys, ps), average_precision_score(ys, ps), ps, ys


def train_ours_one(init, ckpt, train_ds, val_ds, epochs, lr, batch, seed, device,
                    tokenizer_dir: Path = LOCAL / "tokenizer"):
    torch.manual_seed(seed)
    model, tok, pos_bound, arm, pretraining_desc, conv = build_ours_model(
        init, ckpt, train_ds.max_len, device, tokenizer_dir)
    g = torch.Generator().manual_seed(seed)
    dtr = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True, generator=g)
    dva = torch.utils.data.DataLoader(val_ds, batch_size=batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_pos = sum(r["label"] for r in train_ds.rows)
    prev = n_pos / max(len(train_ds.rows), 1)
    pos_w = torch.tensor((1 - prev) / max(prev, 1e-6), device=device)

    history, best = [], None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for b in dtr:
            y = b.pop("label").to(device)
            bb = {k: b[k].to(device) for k in ("input_ids", "local_positions", "attention_mask",
                                                "role_ids", "auth_ids", "event_times")}
            loss = F.binary_cross_entropy_with_logits(model(bb), y, pos_weight=pos_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        auroc, auprc, ps, ys = eval_ours(model, dva, device)
        history.append({"epoch": ep, "auroc": auroc, "auprc": auprc})
        if best is None or auroc > best["auroc"]:
            best = {"epoch": ep, "auroc": auroc, "auprc": auprc, "p": ps, "y": ys,
                    "state": copy.deepcopy(model.state_dict())}
    minutes = (time.time() - t0) / 60
    return {"history": history, "best": best, "minutes": minutes, "arm": arm,
            "non_emb_params": model.enc.n_params(non_embedding=True),
            "pretraining": pretraining_desc}


# ============================================================================
# (e) PubMedBERT
# ============================================================================

MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


class BertDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tok, max_len: int):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids = [self.tok.cls_token_id]
        seg = [0]
        n_tok, n_seg = 0, 0
        per_seg_tok = max(8, (self.max_len - 2) // 8)
        for k, s in enumerate(r["segments"], start=1):
            enc = self.tok.encode(s["text"], add_special_tokens=False)[:per_seg_tok]
            if not enc:
                continue
            if len(ids) + len(enc) + 1 > self.max_len:
                continue
            ids += enc + [self.tok.sep_token_id]
            seg += [k] * (len(enc) + 1)
            n_tok += len(enc) + 1
            n_seg += 1
        pad = self.max_len - len(ids)
        ids += [self.tok.pad_token_id] * pad
        seg += [-1] * pad
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor([1] * (self.max_len - pad) + [0] * pad),
            "label": torch.tensor(float(r["label"])),
            "tokens_used": n_tok,
            "segs_used": n_seg,
            "segs_total": len(r["segments"]),
        }


class BertClf(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.bb = backbone
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(backbone.config.hidden_size, 1)

    def forward(self, ids, mask):
        out = self.bb(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.head(self.drop(out[:, 0])).squeeze(-1)


@torch.no_grad()
def eval_bert(model, dl, dev):
    model.eval()
    ps, ys = [], []
    for b in dl:
        ids, mask = b["input_ids"].to(dev), b["attention_mask"].to(dev)
        logits = model(ids, mask)
        ps += torch.sigmoid(logits).float().cpu().tolist()
        ys += b["label"].tolist()
    model.train()
    return roc_auc_score(ys, ps), average_precision_score(ys, ps), ps, ys


def train_bert_one(train_ds, val_ds, epochs, lr, batch, accum, seed, device):
    import math
    import os
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(seed)
    os.environ.setdefault("HF_HOME", str(LOCAL / "hf"))
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = BertClf(AutoModel.from_pretrained(MODEL_ID)).to(device)

    g = torch.Generator().manual_seed(seed)
    dtr = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True, generator=g)
    dva = torch.utils.data.DataLoader(val_ds, batch_size=batch)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = math.ceil(len(dtr) / accum) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(total, 1), pct_start=0.1)
    n_pos = sum(r["label"] for r in train_ds.rows)
    prev = n_pos / max(len(train_ds.rows), 1)
    pos_w = torch.tensor((1 - prev) / max(prev, 1e-6), device=device)

    history, best = [], None
    t0, step = time.time(), 0
    for ep in range(epochs):
        model.train()
        for i, b in enumerate(dtr):
            ids, mask = b["input_ids"].to(device), b["attention_mask"].to(device)
            y = b["label"].to(device)
            loss = F.binary_cross_entropy_with_logits(model(ids, mask), y, pos_weight=pos_w) / accum
            loss.backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
        auroc, auprc, ps, ys = eval_bert(model, dva, device)
        history.append({"epoch": ep, "auroc": auroc, "auprc": auprc})
        if best is None or auroc > best["auroc"]:
            best = {"epoch": ep, "auroc": auroc, "auprc": auprc, "p": ps, "y": ys,
                    "state": None}  # 110M state dicts are too large to keep 3x in memory
    minutes = (time.time() - t0) / 60
    n_par = sum(p.numel() for p in model.bb.parameters())
    return {"history": history, "best": best, "minutes": minutes, "arm": "concat",
            "non_emb_params": n_par, "pretraining": "PubMed abstracts+fulltext, 3.1B words"}


# ============================================================================
# Token-budget accounting (item 2: report the asymmetry, don't hide it)
# ============================================================================

def budget_stats(ds, n_sample: int = 300) -> dict:
    idx = np.random.default_rng(0).choice(len(ds), size=min(n_sample, len(ds)), replace=False)
    tok_used, seg_used, seg_total = [], [], []
    for i in idx:
        ex = ds[int(i)]
        tok_used.append(ex["tokens_used"])
        seg_used.append(ex["segs_used"])
        seg_total.append(ex["segs_total"])
    tok_used, seg_used, seg_total = map(np.array, (tok_used, seg_used, seg_total))
    return {
        "n_sample": int(len(idx)),
        "mean_tokens_used": float(tok_used.mean()),
        "mean_seq_fill_frac": float(tok_used.mean() / ds.max_len),
        "mean_segs_used": float(seg_used.mean()),
        "mean_segs_total": float(seg_total.mean()),
        "mean_seg_truncation_frac": float(1 - (seg_used / np.maximum(seg_total, 1)).mean()),
    }


# ============================================================================
# Statistics: DeLong paired test + bootstrap CIs
# ============================================================================

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Sun & Xu (2014) fast DeLong. preds: (k, n) predictions for k classifiers,
    n examples ordered [positives..., negatives...]. m = number of positives."""
    k = preds.shape[0]
    n = preds.shape[1] - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def delong_paired_test(y: list[float], p_a: list[float], p_b: list[float]) -> dict:
    """Paired DeLong test on AUROC(a) - AUROC(b), same val examples for both."""
    y = np.asarray(y)
    order = np.argsort(-y, kind="stable")  # positives first
    m = int(y.sum())
    preds = np.vstack([np.asarray(p_a)[order], np.asarray(p_b)[order]])
    aucs, cov = _fast_delong(preds, m)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        z, pval = float("nan"), float("nan")
    else:
        se = np.sqrt(var)
        z = diff / se
        from scipy.stats import norm
        pval = float(2 * (1 - norm.cdf(abs(z))))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff,
            "se": float(np.sqrt(var)) if var > 0 else None, "z": float(z) if var > 0 else None,
            "p_value": pval}


def bootstrap_ci(y: list[float], p: list[float], metric, n_boot: int = 2000, seed: int = 0,
                  alpha: float = 0.05) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    rng = np.random.default_rng(seed)
    point = float(metric(y, p))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        if yy.sum() == 0 or yy.sum() == n:  # degenerate resample, AUROC undefined
            continue
        vals.append(metric(yy, pp))
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_boot_used": int(len(vals))}


def bootstrap_ci_diff(y: list[float], p_a: list[float], p_b: list[float], metric,
                       n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired bootstrap on metric(a) - metric(b), resampling the same indices for both."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    n = len(y)
    rng = np.random.default_rng(seed)
    point = float(metric(y, p_a) - metric(y, p_b))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == n:
            continue
        vals.append(metric(yy, p_a[idx]) - metric(yy, p_b[idx]))
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_boot_used": int(len(vals))}


# ============================================================================
# LR-grid + seed selection, identical procedure for every neural arm
# ============================================================================

def select_neural(name: str, runs_by_lr_seed: dict[tuple[float, int], dict]) -> dict:
    """Same rule for every neural arm: pick the LR with the highest MEAN
    best-epoch val AUROC across seeds; report mean/std over seeds at that LR.
    """
    lrs = sorted({lr for lr, _ in runs_by_lr_seed})
    per_lr = {}
    for lr in lrs:
        seed_runs = {s: r for (l, s), r in runs_by_lr_seed.items() if l == lr}
        aurocs = [r["best"]["auroc"] for r in seed_runs.values()]
        per_lr[lr] = {"mean_auroc": float(np.mean(aurocs)), "seed_runs": seed_runs}
    chosen_lr = max(per_lr, key=lambda lr: per_lr[lr]["mean_auroc"])
    seed_runs = per_lr[chosen_lr]["seed_runs"]
    aurocs = [r["best"]["auroc"] for r in seed_runs.values()]
    auprcs = [r["best"]["auprc"] for r in seed_runs.values()]
    # Pool predictions from every seed's best epoch for the pooled-val stats table.
    # The FIRST seed's predictions are used for the paired DeLong/bootstrap tests
    # against other systems (pairing requires a single fixed prediction vector).
    first_seed = sorted(seed_runs)[0]
    ref = seed_runs[first_seed]["best"]
    return {
        "system": name, "chosen_lr": chosen_lr,
        "lr_grid_means": {lr: v["mean_auroc"] for lr, v in per_lr.items()},
        "seeds": sorted(seed_runs),
        "best_epochs": {s: r["best"]["epoch"] for s, r in seed_runs.items()},
        "auroc_mean": float(np.mean(aurocs)), "auroc_std": float(np.std(aurocs)),
        "auprc_mean": float(np.mean(auprcs)), "auprc_std": float(np.std(auprcs)),
        "y": ref["y"], "p": ref["p"],  # reference seed for pairwise tests
        "minutes_per_run": float(np.mean([r["minutes"] for r in seed_runs.values()])),
    }


# ============================================================================
# Orchestration
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", default=",".join(ALL_SYSTEMS),
                    help=f"comma list, subset of {ALL_SYSTEMS}")
    ap.add_argument("--max-len", type=int, default=256,
                    help="shared sequence-length budget (tokens) for BOTH neural arms")
    ap.add_argument("--chars-per-note", type=int, default=600,
                    help="shared per-note CHARACTER budget for BOTH neural arms, applied "
                         "before either tokenizer runs")
    ap.add_argument("--epochs", type=int, default=2, help="shared epoch budget")
    ap.add_argument("--lr-grid", default="2e-5,5e-5", help="comma list, searched on val, shared")
    ap.add_argument("--seeds", default="0,1,2", help="comma list of fine-tuning seeds")
    ap.add_argument("--batch-ours", type=int, default=16)
    ap.add_argument("--batch-bert", type=int, default=4)
    ap.add_argument("--accum-bert", type=int, default=4)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="pretrained checkpoint dir for ours_pretrained")
    ap.add_argument("--limit-train", type=int, default=0, help="0 = full train set")
    ap.add_argument("--limit-eval", type=int, default=0, help="0 = full val set")
    ap.add_argument("--subsample-train", type=int, default=0,
                    help="if >0, subsample the TRAIN split (only) to ~N examples, "
                         "subject-group-disjoint and prevalence-preserving (see "
                         "subsample_group_disjoint). Drives the label-budget ladder "
                         "(scripts/run_label_ladder.sh). val/test are never subsampled "
                         "or otherwise touched by this flag.")
    ap.add_argument("--subsample-seed", type=int, default=0)
    ap.add_argument("--task-dir", default=str(LOCAL / "task"),
                    help="override for smoke-testing without touching the real task dir")
    ap.add_argument("--tokenizer-dir", default=str(LOCAL / "tokenizer"),
                    help="override for smoke-testing against a scratch tokenizer that "
                         "matches a scratch --ckpt's vocab (a --ckpt pretrained with a "
                         "different tokenizer than this one will fail on vocab_size)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="")
    ap.add_argument("--test", action="store_true",
                    help="spend the locked test set once, on the already-val-selected models")
    ap.add_argument("--protocol-frozen", action="store_true",
                    help="required alongside --test; asserts the protocol above is final")
    args = ap.parse_args()

    if args.test and not args.protocol_frozen:
        raise SystemExit(
            "--test requires --protocol-frozen. The test set has never been touched and "
            "must be spent once, on a protocol that is not still being tuned. Re-run with "
            "--protocol-frozen once max-len/chars-per-note/epochs/lr-grid/seeds are final."
        )

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    for s in systems:
        if s not in ALL_SYSTEMS:
            raise SystemExit(f"unknown system {s!r}; choose from {ALL_SYSTEMS}")
    lr_grid = [float(x) for x in args.lr_grid.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    task_dir = Path(args.task_dir)
    tokenizer_dir = Path(args.tokenizer_dir)

    train_rows = load_rows(task_dir, "train", args.limit_train)
    val_rows = load_rows(task_dir, "val", args.limit_eval)

    subsample_info = None
    if args.subsample_train:
        before = len(train_rows)
        train_rows = subsample_group_disjoint(train_rows, args.subsample_train, args.subsample_seed)
        subsample_info = {
            "target_n": args.subsample_train, "before_n": before, "achieved_n": len(train_rows),
            "achieved_prevalence": sum(r["label"] for r in train_rows) / max(len(train_rows), 1),
            "seed": args.subsample_seed,
        }

    protocol = {
        "max_len": args.max_len, "chars_per_note": args.chars_per_note,
        "epochs": args.epochs, "lr_grid": lr_grid, "seeds": seeds,
        "batch_ours": args.batch_ours, "batch_bert": args.batch_bert,
        "accum_bert": args.accum_bert, "effective_batch_bert": args.batch_bert * args.accum_bert,
        "ckpt": args.ckpt, "limit_train": args.limit_train, "limit_eval": args.limit_eval,
        "subsample_train": subsample_info,
        "task_dir": str(task_dir), "tokenizer_dir": str(tokenizer_dir),
        "device": str(dev), "systems": systems,
        "selection_rule": "best epoch by val AUROC; LR chosen by mean best-epoch val AUROC "
                           "across seeds; identical for every neural arm",
        "argv": sys.argv[1:],
    }
    print("PROTOCOL (shared across every neural arm)")
    print(json.dumps(protocol, indent=2))
    print()
    if subsample_info:
        print(f"train subsampled (subject-group-disjoint, prevalence-preserving): "
              f"{subsample_info['before_n']:,} -> {subsample_info['achieved_n']:,} "
              f"(target {subsample_info['target_n']:,}), "
              f"prevalence {subsample_info['achieved_prevalence']:.2%}\n")

    print(f"train={len(train_rows):,} val={len(val_rows):,} "
          f"val_positives={sum(r['label'] for r in val_rows):.0f}\n")

    results: dict[str, dict] = {}

    # ---- (a) metadata-only ----
    if "metadata" in systems:
        print(">> metadata-only logistic regression (no text)")
        results["metadata_lr"] = run_metadata(train_rows, val_rows)
        print(f"   AUROC {roc_auc_score(results['metadata_lr']['y'], results['metadata_lr']['p']):.4f}")

    # ---- (b) TF-IDF ----
    if "tfidf" in systems:
        print(">> TF-IDF + logistic regression (full text)")
        results["tfidf_lr"] = run_tfidf(train_rows, val_rows)
        print(f"   AUROC {roc_auc_score(results['tfidf_lr']['y'], results['tfidf_lr']['p']):.4f}")

    # ---- shared char-budgeted rows for the two neural arms ----
    need_neural = any(s in systems for s in ("ours_random", "ours_pretrained", "pubmedbert"))
    if need_neural:
        btr = budget_rows(train_rows, args.chars_per_note)
        bva = budget_rows(val_rows, args.chars_per_note)

    # ---- (c)/(d) our encoder ----
    all_ours_runs: dict[str, dict] = {}  # key -> {(lr, seed): run}, kept for --test reuse
    all_ours_pos_bound: dict[str, int] = {}
    for init, key in (("random", "ours_random"), ("pretrained", "ours_pretrained")):
        if key not in systems:
            continue
        print(f">> our encoder, {init} init  (max_len={args.max_len} tokens, "
              f"chars_per_note={args.chars_per_note})")
        tok = Tokenizer.from_file(str(tokenizer_dir / "wordpiece.json"))
        pos_bound = args.max_len
        if init == "pretrained":
            ck = torch.load(Path(args.ckpt) / "model.pt", map_location="cpu", weights_only=False)
            pos_bound = ck["cfg"]["max_seq_len"]
        conv = read_convention(init, args.ckpt)
        train_ds = OursDataset(btr, tok, args.max_len, pos_bound,
                               seg_tokens=conv["seg_tokens"],
                               semantic_roles=conv["semantic_roles"])
        val_ds = OursDataset(bva, tok, args.max_len, pos_bound,
                             seg_tokens=conv["seg_tokens"],
                             semantic_roles=conv["semantic_roles"])
        stats = budget_stats(val_ds)
        print(f"   token budget actually consumed: {json.dumps(stats)}")

        runs = {}
        for lr in lr_grid:
            for seed in seeds:
                t0 = time.time()
                r = train_ours_one(init, args.ckpt, train_ds, val_ds, args.epochs, lr,
                                    args.batch_ours, seed, dev, tokenizer_dir)
                runs[(lr, seed)] = r
                print(f"   lr={lr:.0e} seed={seed}: best AUROC {r['best']['auroc']:.4f} "
                      f"AUPRC {r['best']['auprc']:.4f} @epoch{r['best']['epoch']} "
                      f"({r['minutes']:.1f}m)")
        sel = select_neural(key, runs)
        sel.update({"family": "neural", "extra_inputs": "role_ids, segment position, delta_t",
                    "params": runs[list(runs)[0]]["non_emb_params"],
                    "pretraining": runs[list(runs)[0]]["pretraining"],
                    "budget_stats": stats})
        results[key] = sel
        all_ours_runs[key] = runs
        all_ours_pos_bound[key] = pos_bound
        print(f"   SELECTED lr={sel['chosen_lr']:.0e}  AUROC {sel['auroc_mean']:.4f}"
              f"±{sel['auroc_std']:.4f}  AUPRC {sel['auprc_mean']:.4f}±{sel['auprc_std']:.4f}\n")

    # ---- (e) PubMedBERT ----
    if "pubmedbert" in systems:
        print(f">> PubMedBERT + concatenation  (max_len={args.max_len} tokens, "
              f"chars_per_note={args.chars_per_note})")
        import os
        os.environ.setdefault("HF_HOME", str(LOCAL / "hf"))
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        train_ds = BertDataset(btr, tok, args.max_len)
        val_ds = BertDataset(bva, tok, args.max_len)
        stats = budget_stats(val_ds)
        print(f"   token budget actually consumed: {json.dumps(stats)}")

        runs = {}
        for lr in lr_grid:
            for seed in seeds:
                r = train_bert_one(train_ds, val_ds, args.epochs, lr, args.batch_bert,
                                   args.accum_bert, seed, dev)
                runs[(lr, seed)] = r
                print(f"   lr={lr:.0e} seed={seed}: best AUROC {r['best']['auroc']:.4f} "
                      f"AUPRC {r['best']['auprc']:.4f} @epoch{r['best']['epoch']} "
                      f"({r['minutes']:.1f}m)")
        sel = select_neural("pubmedbert", runs)
        sel.update({"family": "neural", "extra_inputs": "text only",
                    "params": runs[list(runs)[0]]["non_emb_params"],
                    "pretraining": runs[list(runs)[0]]["pretraining"],
                    "budget_stats": stats})
        results["pubmedbert"] = sel
        print(f"   SELECTED lr={sel['chosen_lr']:.0e}  AUROC {sel['auroc_mean']:.4f}"
              f"±{sel['auroc_std']:.4f}  AUPRC {sel['auprc_mean']:.4f}±{sel['auprc_std']:.4f}\n")

    # ================================================================
    # Table -- floor rows first, always present, never omittable.
    # ================================================================
    order = [s for s in ["metadata_lr", "tfidf_lr", "ours_random", "ours_pretrained", "pubmedbert"]
             if s in results]
    print("=" * 100)
    print("RESULTS -- floor rows are never omitted")
    print("=" * 100)
    header = (f"{'system':<18}{'family':<8}{'AUROC':>9}{'AUROC CI':>18}{'AUPRC':>9}"
              f"{'AUPRC CI':>18}{'extra inputs':<38}")
    print(header)
    print("-" * len(header))
    n_boot = args.n_boot
    ci_cache = {}
    for name in order:
        r = results[name]
        y, p = r["y"], r["p"]
        auroc = roc_auc_score(y, p)
        auprc = average_precision_score(y, p)
        auroc_ci = bootstrap_ci(y, p, roc_auc_score, n_boot=n_boot)
        auprc_ci = bootstrap_ci(y, p, average_precision_score, n_boot=n_boot)
        ci_cache[name] = {"auroc_ci": auroc_ci, "auprc_ci": auprc_ci}
        fam = r.get("family", "floor")
        extra = r.get("extra_inputs", "")
        print(f"{name:<18}{fam:<8}{auroc:>9.4f}{'['+f'{auroc_ci['lo']:.4f},{auroc_ci['hi']:.4f}'+']':>18}"
              f"{auprc:>9.4f}{'['+f'{auprc_ci['lo']:.4f},{auprc_ci['hi']:.4f}'+']':>18}{extra:<38}")
    n_val_pos = int(sum(val_rows[i]["label"] for i in range(len(val_rows)))) if val_rows else 0
    print(f"\nval n={len(val_rows)}, positives={n_val_pos}. With ~375 positives the bootstrap "
          f"AUROC CI half-width is roughly +/-0.03 -- effects smaller than that are not resolved "
          f"by this val set.")

    # ---- pairwise paired stats: every neural arm vs the strongest floor row ----
    floor_names = [n for n in ("tfidf_lr", "metadata_lr") if n in results]
    neural_names = [n for n in ("ours_random", "ours_pretrained", "pubmedbert") if n in results]
    pairwise = []
    if floor_names and neural_names:
        best_floor = max(floor_names, key=lambda n: roc_auc_score(results[n]["y"], results[n]["p"]))
        print(f"\nPAIRED STATS vs strongest floor ({best_floor}), DeLong test + bootstrap CI on the "
              f"paired AUROC difference (same val examples for both models):")
        yf, pf = results[best_floor]["y"], results[best_floor]["p"]
        for nn_ in neural_names:
            yn, pn = results[nn_]["y"], results[nn_]["p"]
            if yn != yf:
                print(f"  {nn_} vs {best_floor}: SKIPPED (different val rows under --limit-eval / "
                      f"different seed reference; not paired)")
                continue
            dl = delong_paired_test(yf, pn, pf)
            bs = bootstrap_ci_diff(yf, pn, pf, roc_auc_score, n_boot=n_boot)
            pairwise.append({"a": nn_, "b": best_floor, "delong": dl, "bootstrap_diff": bs})
            print(f"  {nn_:<18} AUROC {dl['auc_a']:.4f} vs {best_floor} {dl['auc_b']:.4f}  "
                  f"diff {dl['diff']:+.4f}  DeLong p={dl['p_value']:.4f}  "
                  f"bootstrap diff CI [{bs['lo']:+.4f}, {bs['hi']:+.4f}]")

    # ---- the confound this harness does NOT and cannot remove ----
    print("\nREMAINING (unremovable) CONFOUND: extra inputs.")
    print("  ours receives role_ids, segment position and delta_t as separate channels;")
    print("  PubMedBERT receives text only. This is architectural, not a protocol bug --")
    print("  every other axis (input budget, epochs, selection, LR, seeds) is now matched.")

    out = {
        "protocol": protocol,
        "n_train": len(train_rows), "n_val": len(val_rows), "n_val_pos": n_val_pos,
        "systems": {name: {k: v for k, v in r.items() if k not in ("y", "p")}
                    for name, r in results.items()},
        "pointwise": {name: {"auroc": roc_auc_score(results[name]["y"], results[name]["p"]),
                              "auprc": average_precision_score(results[name]["y"], results[name]["p"]),
                              **ci_cache[name]}
                      for name in order},
        "pairwise_vs_floor": pairwise,
        "spent_test_set": False,
    }

    # ================================================================
    # --test: locked, one-shot, only with --protocol-frozen
    # ================================================================
    if args.test:
        print("\n" + "=" * 100)
        print("LOCKED TEST-SET EVALUATION (--protocol-frozen). Spending the test set once.")
        print("=" * 100)
        test_rows = load_rows(task_dir, "test", args.limit_eval)
        n_test_pos = int(sum(r["label"] for r in test_rows))
        print(f"test n={len(test_rows):,} positives={n_test_pos}")
        test_results = {}
        for name in ("metadata_lr", "tfidf_lr"):
            if name not in results:
                continue
            # Re-fit on train (identical procedure), score once on test.
            if name == "metadata_lr":
                r = run_metadata(train_rows, test_rows)
            else:
                r = run_tfidf(train_rows, test_rows)
            test_results[name] = r
            print(f"  {name}: test AUROC {roc_auc_score(r['y'], r['p']):.4f} "
                  f"AUPRC {average_precision_score(r['y'], r['p']):.4f}")

        btest = budget_rows(test_rows, args.chars_per_note) if need_neural else None
        for key in ("ours_random", "ours_pretrained"):
            if key not in results:
                continue
            # Reuse the reference seed's ALREADY-TRAINED, SELECTED (best-lr,
            # best-epoch) in-memory weights -- the test set scores the exact
            # model chosen on val, not a freshly retrained one, and no extra
            # training touches the test-selection decision.
            lr = results[key]["chosen_lr"]
            seed0 = results[key]["seeds"][0]
            state = all_ours_runs[key][(lr, seed0)]["best"]["state"]
            pos_bound = all_ours_pos_bound[key]
            init = "pretrained" if key == "ours_pretrained" else "random"
            print(f"  scoring {key} on test with the val-selected weights "
                  f"(lr={lr:.0e} seed={seed0}, no retraining)")
            tok = Tokenizer.from_file(str(tokenizer_dir / "wordpiece.json"))
            conv = read_convention(init, args.ckpt)
            test_ds = OursDataset(btest, tok, args.max_len, pos_bound,
                                  seg_tokens=conv["seg_tokens"],
                                  semantic_roles=conv["semantic_roles"])
            model, _, _, _, _, conv = build_ours_model(init, args.ckpt, args.max_len, dev, tokenizer_dir)
            model.load_state_dict(state)
            dte = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_ours)
            auroc, auprc, ps, ys = eval_ours(model, dte, dev)
            test_results[key] = {"auroc": auroc, "auprc": auprc, "y": ys, "p": ps}
            print(f"  {key}: test AUROC {auroc:.4f} AUPRC {auprc:.4f}")

        if "pubmedbert" in results:
            print("  pubmedbert: SKIPPED in --test. train_bert_one() deliberately does not keep "
                  "in-memory copies of the 110M-parameter state dict across (lr, seed) (memory "
                  "budget on a 16GB machine); scoring test would require retraining at the "
                  "selected lr/seed, which this harness does not do automatically to avoid a "
                  "silent extra multi-tens-of-minutes run. Call train_bert_one() once more "
                  "explicitly at the selected lr/seed and evaluate it on test before reporting "
                  "a locked PubMedBERT test number.")
            print(f"  {key}: test AUROC {auroc:.4f} AUPRC {auprc:.4f}")

        out["test"] = {name: {k: v for k, v in r.items() if k not in ("y", "p")}
                       for name, r in test_results.items()}
        out["test_pointwise"] = {name: {"auroc": roc_auc_score(r["y"], r["p"]),
                                        "auprc": average_precision_score(r["y"], r["p"])}
                                 for name, r in test_results.items()}
        out["spent_test_set"] = True
        out["protocol_frozen"] = True

    RESULTS.mkdir(exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = RESULTS / f"matched_protocol{tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
