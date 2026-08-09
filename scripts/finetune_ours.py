"""Fine-tune the from-scratch encoder on multi-note mortality prediction.

No backbone. The encoder is the one pretrained by scripts/pretrain.py on MIMIC
note bundles with MLM -- ~10M non-embedding parameters and roughly 3M tokens of
pretraining, against PubMedBERT's 110M and 3.1B words.

The honest framing this supports is not "beats PubMedBERT" but "recovers most of
a 110M biomedical model's performance on multi-note input at ~1% of the
pretraining data, from an architecture trained entirely on one laptop."

Local only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tokenizers import Tokenizer

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.lineage import check_task_lineage, task_dir_lineage  # noqa: E402
from src.data.roles import N_ROLES  # noqa: E402
from src.model.config import EncoderConfig  # noqa: E402
from src.model.encoder import ProvenanceEncoder  # noqa: E402

LOCAL = ROOT / "local_only"
TASK = LOCAL / "task"
DEFAULT_LINEAGE_PIN = ROOT / "results" / "task_lineage_pin.json"


class TaskDS(torch.utils.data.Dataset):
    def __init__(self, path, tok, max_len=256, per_seg=36, limit=0):
        rows = []
        with Path(path).open() as f:
            for line in f:
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
        self.rows = rows
        self.tok, self.max_len, self.per_seg = tok, max_len, per_seg
        v = tok.get_vocab()
        self.pad, self.cls, self.sep = v["[PAD]"], v["[CLS]"], v["[SEP]"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        S = self.max_len
        ids = np.full(S, self.pad, dtype=np.int64)
        role = np.zeros(S, dtype=np.int64)
        segid = np.full(S, -1, dtype=np.int64)
        lpos = np.zeros(S, dtype=np.int64)
        dt = np.zeros(S, dtype=np.float32)
        mask = np.zeros(S, dtype=np.float32)
        cur = 0
        ids[cur], mask[cur], segid[cur] = self.cls, 1.0, 0
        cur += 1
        for k, s in enumerate(r["segments"], start=1):
            enc = self.tok.encode(s["text"]).ids[: self.per_seg]
            if not enc or cur + len(enc) + 1 >= S:
                break
            n = len(enc)
            ids[cur:cur + n] = enc
            # Segment id drives the block bias -- this is the winning arm's signal.
            role[cur:cur + n] = 1 + min(N_ROLES - 2, k - 1)
            segid[cur:cur + n] = k
            lpos[cur:cur + n] = np.arange(n)
            dt[cur:cur + n] = float(s["delta_t_hours"])
            mask[cur:cur + n] = 1.0
            cur += n
            ids[cur], role[cur], segid[cur] = self.sep, role[cur - 1], k
            mask[cur], dt[cur], lpos[cur] = 1.0, dt[cur - 1], n
            cur += 1
        return {
            "input_ids": torch.from_numpy(ids),
            "local_positions": torch.from_numpy(lpos),
            "attention_mask": torch.from_numpy(mask),
            "role_ids": torch.from_numpy(role),
            "auth_ids": torch.zeros(S, dtype=torch.long),
            "event_times": torch.from_numpy(dt),
            "label": torch.tensor(float(r["label"])),
        }


class Clf(nn.Module):
    def __init__(self, enc: ProvenanceEncoder):
        super().__init__()
        self.enc = enc
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(enc.cfg.d_model, 1)

    def forward(self, b):
        h = self.enc(b["input_ids"], b["local_positions"], b["attention_mask"],
                     b["role_ids"], b["auth_ids"], b["event_times"])
        return self.head(self.drop(h[:, 0])).squeeze(-1)


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    ps, ys = [], []
    for b in dl:
        y = b.pop("label")
        b = {k: v.to(dev) for k, v in b.items()}
        ps += torch.sigmoid(model(b)).float().cpu().tolist()
        ys += y.tolist()
    model.train()
    return roc_auc_score(ys, ps), average_precision_score(ys, ps), float(np.mean(ys))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run dir containing model.pt, or 'none'")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit-train", type=int, default=0,
                    help="truncate the train split to this many rows (smoke-testing)")
    ap.add_argument("--limit-eval", type=int, default=0,
                    help="truncate the val split to this many rows (smoke-testing)")
    ap.add_argument("--task-dir", default=str(TASK),
                    help="downstream task split to train on (lineage guard checks "
                         "this dir); override only for smoke-testing")
    ap.add_argument("--lineage-pin", default=str(DEFAULT_LINEAGE_PIN),
                    help="pinned task-split reference the lineage guard compares "
                         "against; override only for smoke-testing")
    ap.add_argument("--repin-task-lineage", action="store_true",
                    help="accept the current local_only/task split as the new "
                         "pinned reference if it disagrees with the stored one "
                         "(use after an intentional rebuild; see "
                         "accord/CORRECTIONS.md C32)")
    args = ap.parse_args()

    task_dir = Path(args.task_dir)
    data_lineage = task_dir_lineage(task_dir)
    check_task_lineage(data_lineage, Path(args.lineage_pin), repin=args.repin_task_lineage)

    torch.manual_seed(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = Tokenizer.from_file(str(LOCAL / "tokenizer" / "wordpiece.json"))

    if args.ckpt != "none":
        ck = torch.load(Path(args.ckpt) / "model.pt", map_location="cpu", weights_only=False)
        cfg = EncoderConfig(**{k: v for k, v in ck["cfg"].items()
                               if k in EncoderConfig.__dataclass_fields__ and
                               k not in ("use_role_embed", "use_authority_bias",
                                         "use_time_bias", "use_prefix_tokens")})
        for k in ("alibi_bias", "rpe_bias", "hard_block_mask"):
            setattr(cfg, k, ck["cfg"].get(k, False))
        enc = ProvenanceEncoder(cfg)
        enc.load_state_dict(ck["model"])
        init = "pretrained"
    else:
        cfg = EncoderConfig(arm="A3", vocab_size=tok.get_vocab_size(), d_model=384,
                            n_layers=6, n_heads=6, d_ff=1536, max_seq_len=args.max_len)
        cfg.hard_block_mask = True
        enc = ProvenanceEncoder(cfg)
        init = "random"
    cfg.max_seq_len = args.max_len
    model = Clf(enc).to(dev)

    tr = TaskDS(task_dir / "train.jsonl", tok, args.max_len, limit=args.limit_train)
    va = TaskDS(task_dir / "val.jsonl", tok, args.max_len, limit=args.limit_eval)
    g = torch.Generator().manual_seed(args.seed)
    dtr = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True, generator=g)
    dva = torch.utils.data.DataLoader(va, batch_size=args.batch)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    pos_w = torch.tensor((1 - 0.102) / 0.102, device=dev)
    n_par = enc.n_params(non_embedding=True)
    print(f"init={init} arm={cfg.arm} hard_block={cfg.hard_block_mask} "
          f"non_emb_params={n_par:,} device={dev}")
    print(f"train={len(tr):,} val={len(va):,} seq={args.max_len}")

    t0 = time.time()
    model.train()
    best = None
    for ep in range(args.epochs):
        for i, b in enumerate(dtr):
            y = b.pop("label").to(dev)
            b = {k: v.to(dev) for k, v in b.items()}
            loss = F.binary_cross_entropy_with_logits(model(b), y, pos_weight=pos_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if (i + 1) % 300 == 0:
                print(f"  ep{ep} {i+1}/{len(dtr)} loss {float(loss):.4f} "
                      f"{(time.time()-t0)/60:.1f}m", flush=True)
        auroc, auprc, prev = evaluate(model, dva, dev)
        print(f"  [ep{ep}] AUROC {auroc:.4f} AUPRC {auprc:.4f}", flush=True)
        if best is None or auroc > best[0]:
            best = (auroc, auprc, ep)

    out = {"init": init, "arm": cfg.arm, "hard_block": cfg.hard_block_mask,
           "non_emb_params": n_par, "auroc": best[0], "auprc": best[1],
           "best_epoch": best[2], "prevalence": prev, "seed": args.seed,
           "minutes": (time.time() - t0) / 60, "tag": args.tag,
           # n_train/n_val: this is the field that was previously absent from
           # every ours_* run and let concat_s0/concat_s1 pass as seed
           # variance (accord/CORRECTIONS.md C32).
           "n_train": len(tr), "n_val": len(va),
           "n_test": data_lineage["test_n"], "data_lineage": data_lineage}
    d = LOCAL / "task_runs"; d.mkdir(exist_ok=True)
    (d / f"ours_{args.tag or init}_s{args.seed}.json").write_text(json.dumps(out, indent=2))
    print(f"\nFINAL ours({args.tag or init}) AUROC={best[0]:.4f} AUPRC={best[1]:.4f} "
          f"params={n_par:,} {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
