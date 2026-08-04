"""Does intra-note attention masking beat standard concatenation on PubMedBERT?

Standard practice for multi-note clinical prediction is to concatenate notes and
feed them to a biomedical encoder, which discards document structure entirely.
This compares that against the mechanism measured at 1.08 nats in the pretraining
study: restrict attention so tokens attend within their own note, with [CLS]
global so information still crosses.

Both arms use the SAME backbone, the SAME weights, the SAME token budget and the
SAME data order. The only difference is the attention connectivity, so any
difference is attributable to it and nothing else.

transformers accepts a 3-D (B, S, S) attention mask, so the block structure is
imposed without modifying any model code.

Local only. MIMIC never leaves the machine; the backbone is public and
MIT-licensed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "local_only"
TASK = LOCAL / "task"
MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


class BundleDS(torch.utils.data.Dataset):
    def __init__(self, path, tok, max_len=512, per_seg=110, limit=0):
        self.rows = [json.loads(l) for l in Path(path).open()]
        if limit:
            self.rows = self.rows[:limit]
        self.tok, self.max_len, self.per_seg = tok, max_len, per_seg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids = [self.tok.cls_token_id]
        seg = [0]                      # 0 = CLS, global
        for k, s in enumerate(r["segments"], start=1):
            enc = self.tok.encode(s["text"], add_special_tokens=False)[: self.per_seg]
            if not enc or len(ids) + len(enc) + 1 > self.max_len:
                break
            ids += enc + [self.tok.sep_token_id]
            seg += [k] * (len(enc) + 1)
        pad = self.max_len - len(ids)
        ids += [self.tok.pad_token_id] * pad
        seg += [-1] * pad
        return {
            "input_ids": torch.tensor(ids),
            "seg": torch.tensor(seg),
            "label": torch.tensor(float(r["label"])),
        }


def build_mask(seg: torch.Tensor, block: bool) -> torch.Tensor:
    """(B,S) segment ids -> (B,1,S,S) BOOL mask. True = visible.

    transformers 5.x rejects 3-D masks and float masks containing zeros; a 4-D
    boolean mask is the format that works and it is verified to change the model
    output rather than being silently ignored.
    """
    valid = (seg >= 0)
    S = seg.size(1)
    if not block:
        vis = valid.unsqueeze(-2).expand(-1, S, -1) & valid.unsqueeze(-1)
    else:
        same = seg.unsqueeze(-1) == seg.unsqueeze(-2)       # within-note
        cls_row = (seg == 0).unsqueeze(-1)                  # CLS attends everywhere
        cls_col = (seg == 0).unsqueeze(-2)                  # everyone sees CLS
        vis = (same | cls_row | cls_col) & valid.unsqueeze(-2) & valid.unsqueeze(-1)
    # Every row must have at least one visible key. A fully-masked row makes
    # softmax average over all -inf and emit NaN, which then propagates through
    # the whole batch. Padding rows are fully masked in BOTH arms, so the
    # diagonal is added to both and the comparison stays symmetric. Padding
    # outputs are discarded anyway -- only [CLS] feeds the classifier.
    eye = torch.eye(S, dtype=torch.bool, device=seg.device).unsqueeze(0)
    return (vis | eye).unsqueeze(1)                          # (B, 1, S, S)


class Classifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.bb = backbone
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(backbone.config.hidden_size, 1)

    def forward(self, ids, mask3d):
        out = self.bb(input_ids=ids, attention_mask=mask3d).last_hidden_state
        return self.head(self.drop(out[:, 0])).squeeze(-1)


@torch.no_grad()
def evaluate(model, dl, dev, block):
    model.eval()
    ps, ys = [], []
    for b in dl:
        ids, seg = b["input_ids"].to(dev), b["seg"].to(dev)
        logits = model(ids, build_mask(seg, block))
        ps += torch.sigmoid(logits).float().cpu().tolist()
        ys += b["label"].tolist()
    model.train()
    return (roc_auc_score(ys, ps), average_precision_score(ys, ps),
            float(np.mean(ys)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", action="store_true", help="intra-note attention masking")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-eval", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.environ.setdefault("HF_HOME", str(LOCAL / "hf"))
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = Classifier(AutoModel.from_pretrained(MODEL_ID)).to(dev)

    tr = BundleDS(TASK / "train.jsonl", tok, args.max_len, limit=args.limit_train)
    va = BundleDS(TASK / "val.jsonl", tok, args.max_len, limit=args.limit_eval)
    g = torch.Generator().manual_seed(args.seed)
    dtr = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True, generator=g)
    dva = torch.utils.data.DataLoader(va, batch_size=args.batch)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = math.ceil(len(dtr) / args.accum) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(total, 1),
                                                pct_start=0.1)
    pos_w = torch.tensor((1 - 0.102) / 0.102, device=dev)

    arm = "block" if args.block else "concat"
    print(f"arm={arm} device={dev} train={len(tr):,} val={len(va):,} "
          f"seq={args.max_len} eff_batch={args.batch*args.accum}")

    t0, step = time.time(), 0
    model.train()
    for ep in range(args.epochs):
        for i, b in enumerate(dtr):
            ids, seg = b["input_ids"].to(dev), b["seg"].to(dev)
            y = b["label"].to(dev)
            logits = model(ids, build_mask(seg, args.block))
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w) / args.accum
            loss.backward()
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    print(f"  ep{ep} step {step}/{total} loss {float(loss)*args.accum:.4f} "
                          f"{(time.time()-t0)/60:.1f}m", flush=True)

    auroc, auprc, prev = evaluate(model, dva, dev, args.block)
    out = {"arm": arm, "auroc": auroc, "auprc": auprc, "prevalence": prev,
           "seed": args.seed, "epochs": args.epochs, "max_len": args.max_len,
           "n_train": len(tr), "n_val": len(va), "minutes": (time.time()-t0)/60}
    d = LOCAL / "task_runs"; d.mkdir(exist_ok=True)
    (d / f"{arm}_s{args.seed}.json").write_text(json.dumps(out, indent=2))
    print(f"\nFINAL arm={arm} AUROC={auroc:.4f} AUPRC={auprc:.4f} "
          f"(prevalence {prev:.4f}) {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
