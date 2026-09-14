"""
Defect-free control: training driver. One process per seed (run via
run_all.sh), so memory from one MPS/torch session never carries into the
next.

set_seed() is called INSIDE main(), at the top of each run -- not once at
import time. Correction C7 was that reported seeds were actually 2 distinct
seeds plus a hardcoded row (the seed wasn't really controlling 3 independent
runs). Here every seed argument controls model init, dropout, and
DataLoader shuffle order, freshly, per process.

Writes, per seed, to /Users/rickygole/mimic_local_only/accord_control/:
  scores_seed{N}.csv   -- pair_id, rule, pharm_pos, score (299 gold rows;
                           NO note text)
  metrics_seed{N}.json -- AUROC, AP, top-15 TP count, defect-free-subset
                           versions of same, train/val loss history

Train/val DATA split is fixed across seeds (data.SPLIT_SEED); only model
stochasticity varies by seed.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from data import build_splits, tokenize_pairs
from model import PairClassifier, MODEL_NAME

OUT_DIR = "/Users/rickygole/mimic_local_only/accord_control"
MAX_LENGTH = 160


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EncodedDataset(Dataset):
    def __init__(self, enc, y):
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc["token_type_ids"]
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "label": self.y[idx],
        }


def run_eval_scores(model, enc, device, batch_size=64):
    """Return sigmoid(score) for every row in enc, in order, no shuffling."""
    model.eval()
    n = enc["input_ids"].shape[0]
    scores = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ids = enc["input_ids"][start:end].to(device)
            mask = enc["attention_mask"][start:end].to(device)
            tt = enc["token_type_ids"][start:end].to(device)
            _, logit = model(ids, mask, tt)
            scores.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(scores)


def evaluate_loss(model, loader, device):
    model.eval()
    total_loss, n_seen = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tt = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            loss, _ = model(ids, mask, tt, labels)
            total_loss += loss.item() * labels.size(0)
            n_seen += labels.size(0)
    return total_loss / n_seen


def top_k_tp(scores: np.ndarray, y: np.ndarray, k: int) -> int:
    order = np.argsort(-scores)
    top = order[:k]
    return int(y[top].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    # Scores written here are derived from credentialed data: refuse any output
    # directory that resolves into cloud-synced storage.
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from accord.cloud_paths import assert_local_only
    args.out_dir = str(assert_local_only(args.out_dir))

    set_seed(args.seed)
    device = get_device()
    print(f"[seed={args.seed}] device={device}")

    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME)
    model = PairClassifier(backbone).to(device)

    splits = build_splits()
    print(f"[seed={args.seed}] train={len(splits.train)} val={len(splits.val)} gold=299 "
          f"leaked_rows_excluded={splits.n_leaked_rows_excluded} "
          f"gold_subjects={splits.n_gold_subjects} pool_subjects={splits.n_pool_subjects}")
    print(f"[seed={args.seed}] train label positive rate: {splits.train['y'].mean():.4f} "
          f"val label positive rate: {splits.val['y'].mean():.4f}")

    train_enc = tokenize_pairs(tokenizer, splits.train, MAX_LENGTH)
    val_enc = tokenize_pairs(tokenizer, splits.val, MAX_LENGTH)
    gold_enc = tokenize_pairs(tokenizer, splits.gold, MAX_LENGTH)

    train_ds = EncodedDataset(train_enc, splits.train["y"].to_numpy())
    val_ds = EncodedDataset(val_enc, splits.val["y"].to_numpy())

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss, n_seen = 0.0, 0
        t0 = time.time()
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tt = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss, _ = model(ids, mask, tt, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            n_seen += labels.size(0)

        train_loss = total_loss / n_seen
        val_loss = evaluate_loss(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"[seed={args.seed}] epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} ({elapsed:.0f}s)")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "elapsed_sec": elapsed})

    gold_scores = run_eval_scores(model, gold_enc, device)

    from sklearn.metrics import roc_auc_score, average_precision_score

    y_gold = splits.gold["pharm_pos"].to_numpy()
    rule = splits.gold["rule"].to_numpy()
    defect_free_mask = np.isin(rule, [3, 4])

    def block(mask):
        yy = y_gold[mask]
        ss = gold_scores[mask]
        n, pos = int(mask.sum()), int(yy.sum())
        out = {"n": n, "pos": pos}
        if pos == 0 or pos == n:
            out["auroc"] = None
            out["ap"] = None
        else:
            out["auroc"] = float(roc_auc_score(yy, ss))
            out["ap"] = float(average_precision_score(yy, ss))
        out["top15_tp"] = top_k_tp(ss, yy, min(15, n))
        return out

    metrics = {
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_length": MAX_LENGTH,
        "n_train": len(splits.train),
        "n_val": len(splits.val),
        "n_leaked_rows_excluded": splits.n_leaked_rows_excluded,
        "n_gold_subjects": splits.n_gold_subjects,
        "n_pool_subjects": splits.n_pool_subjects,
        "train_pos_rate": float(splits.train["y"].mean()),
        "val_pos_rate": float(splits.val["y"].mean()),
        "history": history,
        "gold_all": block(np.ones_like(defect_free_mask, dtype=bool)),
        "gold_defect_free_rule34": block(defect_free_mask),
        "gold_rules12": block(~defect_free_mask),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    scores_df = pd.DataFrame({
        "pair_id": splits.gold["pair_id"].to_numpy(),
        "rule": rule,
        "pharm_pos": y_gold,
        "score": gold_scores,
    })
    scores_path = os.path.join(args.out_dir, f"scores_seed{args.seed}.csv")
    scores_df.to_csv(scores_path, index=False)

    metrics_path = os.path.join(args.out_dir, f"metrics_seed{args.seed}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[seed={args.seed}] wrote {scores_path}")
    print(f"[seed={args.seed}] wrote {metrics_path}")
    print(f"[seed={args.seed}] gold_all AUROC={metrics['gold_all']['auroc']} "
          f"gold_defect_free_rule34 AUROC={metrics['gold_defect_free_rule34']['auroc']}")


if __name__ == "__main__":
    main()
