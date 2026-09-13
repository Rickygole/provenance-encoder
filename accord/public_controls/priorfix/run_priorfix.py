"""C105: 10 replicates x 4 arms (H, S1, T, S0), BiomedBERT student, SNLI only.
See C105_PREREGISTRATION.md. No MIMIC access anywhere in this script.

Reuses accord.control.model.PairClassifier (checked: no MIMIC read at import
time) with accord.control.train.py's defaults (3 epochs, batch 16, lr 2e-5,
weight decay 0.01, max length 128, seed = replicate index), but the training
loop itself is written here directly against SNLI data rather than importing
accord.control.train's main(), which builds MIMIC-shaped paths.

Must not run while an Ollama model is loaded on this machine (training uses
MPS; `ollama ps` is checked empty before the first replicate).
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from accord.control.model import PairClassifier, MODEL_NAME  # noqa: E402
from accord.public_controls.nlifactorial.build_manifest import (  # noqa: E402
    reproduce_snli_pool_cursors,
)

HERE = pathlib.Path(__file__).resolve().parent
PARTITIONS_PATH = HERE / "partitions.json"
NLIFACTORIAL_RAW = (ROOT / "results" / "accord" / "public_controls"
                     / "nlifactorial" / "raw" / "llama3_8b__P1.jsonl")
TOPUP_RAW = (ROOT / "results" / "accord" / "public_controls" / "priorfix"
             / "raw" / "teacher_topup.jsonl")
OUT_DIR = ROOT / "results" / "accord" / "public_controls" / "priorfix"
MAX_LENGTH = 128
N_REPLICATES = 10
TRAIN_POOL_N = 1679
TRAIN_POOL_POS = 80          # round(1679 * 12/253) = 80
GOLD_N = 253
GOLD_P = 12 / 253
ARMS = ["H", "S1", "T", "S0"]


def assert_no_ollama_model_loaded():
    r = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
    lines = [l for l in r.stdout.strip().splitlines()[1:] if l.strip()]
    if lines:
        raise SystemExit(f"refusing to train: {len(lines)} Ollama model(s) "
                          f"still loaded; unload them first")


def load_teacher_pool():
    """Returns (rows_df) with columns row, class(0/1 is_contradiction),
    logodds, degenerate, source. Positives from Item 11's own llama3:8b/P1
    cell raw jsonl (item_id 'snli:<row>'); negatives from this item's own
    teacher_topup.jsonl."""
    pos_recs = []
    for line in NLIFACTORIAL_RAW.read_text().splitlines():
        if not line:
            continue
        r = json.loads(line)
        if r["hardness"] != "positives":
            continue
        row = int(r["item_id"].split(":")[1])
        pos_recs.append({"row": row, "class": 1, "logodds": r["logodds"],
                          "degenerate": r["degenerate"], "hard_label": r["hard_label"]})
    neg_recs = []
    for line in TOPUP_RAW.read_text().splitlines():
        if not line:
            continue
        r = json.loads(line)
        neg_recs.append({"row": r["row"], "class": 0, "logodds": r["logodds"],
                          "degenerate": r["degenerate"], "hard_label": r["hard_label"]})
    df = pd.DataFrame(pos_recs + neg_recs)
    assert (df["class"] == 1).sum() == 1000
    assert (df["class"] == 0).sum() == 2000, (df["class"] == 0).sum()
    return df


def bisect_intercept_shift(z, pi_hat, tol=1e-4, max_iter=100):
    """Saerens/Latinne/Decaestecker intercept-shift form: find b such that
    mean(sigmoid(z + b)) == pi_hat. sigmoid is monotone increasing in b, so
    bisection applies directly."""
    z = np.asarray(z, dtype=float)

    def mean_sig(b):
        return float(np.mean(1.0 / (1.0 + np.exp(-(z + b)))))

    lo, hi = -30.0, 30.0
    f_lo, f_hi = mean_sig(lo), mean_sig(hi)
    if pi_hat <= f_lo:
        return lo
    if pi_hat >= f_hi:
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = mean_sig(mid)
        if abs(f_mid - pi_hat) < tol:
            return mid
        if f_mid < pi_hat:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_training_pool(teacher_df, rng):
    pos = teacher_df[teacher_df["class"] == 1]
    neg = teacher_df[teacher_df["class"] == 0]
    pos_idx = rng.choice(len(pos), TRAIN_POOL_POS, replace=False)
    neg_idx = rng.choice(len(neg), TRAIN_POOL_N - TRAIN_POOL_POS, replace=False)
    pool = pd.concat([pos.iloc[pos_idx], neg.iloc[neg_idx]]).reset_index(drop=True)
    return pool.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)


def draw_gold_sample(gold_pos_rows, gold_neg_rows, rng):
    """i.i.d. Bernoulli(GOLD_P) draw, WITH replacement from each fixed pool."""
    is_pos = rng.random(GOLD_N) < GOLD_P
    n_pos = int(is_pos.sum())
    n_neg = GOLD_N - n_pos
    pos_rows = rng.choice(gold_pos_rows, n_pos, replace=True) if n_pos else np.array([], dtype=int)
    neg_rows = rng.choice(gold_neg_rows, n_neg, replace=True) if n_neg else np.array([], dtype=int)
    pi_hat = n_pos / GOLD_N
    return pi_hat, n_pos


def arm_targets(pool, arm, pi_hat):
    z = pool["logodds"].to_numpy()
    if arm == "H":
        return (pool["class"].to_numpy() == 1).astype(float)
    if arm == "S0":
        return 1.0 / (1.0 + np.exp(-z))
    if arm == "S1":
        b = bisect_intercept_shift(z, pi_hat)
        return 1.0 / (1.0 + np.exp(-(z + b)))
    if arm == "T":
        k = max(0, min(len(z), round(pi_hat * len(z))))
        order = np.argsort(-z)
        y = np.zeros(len(z))
        y[order[:k]] = 1.0
        return y
    raise ValueError(arm)


class SNLIPairDataset(Dataset):
    def __init__(self, enc, y):
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc["token_type_ids"]
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx],
                "token_type_ids": self.token_type_ids[idx], "label": self.y[idx]}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def tokenize(tokenizer, df, snli_df):
    premises = [str(snli_df.loc[r, "premise"]) for r in df["row"]]
    hyps = [str(snli_df.loc[r, "hypothesis"]) for r in df["row"]]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enc = tokenizer(premises, hyps, truncation=True, max_length=MAX_LENGTH,
                         padding="max_length", return_tensors="pt")
    return enc


def train_one_arm(tokenizer, backbone_state, pool, targets, eval_enc, eval_y, device, seed, epochs, batch_size, lr):
    set_seed(seed)
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(MODEL_NAME)
    model = PairClassifier(backbone).to(device)

    train_enc = tokenize(tokenizer, pool, backbone_state["snli_df"])
    ds = SNLIPairDataset(train_enc, targets)
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    for epoch in range(epochs):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tt = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss, _ = model(ids, mask, tt, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    scores = []
    with torch.no_grad():
        n = eval_enc["input_ids"].shape[0]
        for start in range(0, n, 64):
            end = min(start + 64, n)
            ids = eval_enc["input_ids"][start:end].to(device)
            mask = eval_enc["attention_mask"][start:end].to(device)
            tt = eval_enc["token_type_ids"][start:end].to(device)
            _, logit = model(ids, mask, tt)
            scores.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=N_REPLICATES)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    assert_no_ollama_model_loaded()
    partitions = json.loads(PARTITIONS_PATH.read_text())
    teacher_df = load_teacher_pool()
    snli_df, *_ = reproduce_snli_pool_cursors()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = get_device()
    print(f"device={device}")

    eval_rows = pd.DataFrame(
        [{"row": r, "class": 1} for r in partitions["eval_pos_rows"]]
        + [{"row": r, "class": 0} for r in partitions["eval_neg_rows"]])
    eval_enc = tokenize(tokenizer, eval_rows, snli_df)
    eval_y = eval_rows["class"].to_numpy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "replicates").mkdir(exist_ok=True)

    for rep in range(args.replicates):
        rep_path = OUT_DIR / "replicates" / f"rep{rep}.json"
        if rep_path.exists():
            print(f"skip replicate {rep}: already exists")
            continue
        rng = np.random.default_rng(1000 + rep)
        pool = build_training_pool(teacher_df, rng)
        teacher_marginal = float(pool["class"].mean())  # NOT used as target; diagnostic only
        teacher_pos_rate_hard = float(((pool["logodds"] > 0).astype(int)).mean())
        pi_hat, n_pos_gold = draw_gold_sample(
            np.array(partitions["gold_pos_pool_rows"]),
            np.array(partitions["gold_neg_pool_rows"]), rng)

        rep_out = {"replicate": rep, "pi_hat": pi_hat, "n_pos_gold": n_pos_gold,
                   "teacher_marginal_on_pool": teacher_marginal,
                   "teacher_hard_positive_rate_on_pool": teacher_pos_rate_hard,
                   "arms": {}}
        for arm in ARMS:
            targets = arm_targets(pool, arm, pi_hat)
            t0 = time.time()
            scores = train_one_arm(
                tokenizer, {"snli_df": snli_df}, pool, targets, eval_enc, eval_y,
                device, seed=rep, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
            elapsed = time.time() - t0
            rep_out["arms"][arm] = {
                "scores": scores.tolist(), "elapsed_sec": round(elapsed, 1),
                "flag_rate_at_0_5": float((scores >= 0.5).mean()),
            }
            print(f"[rep {rep}] arm {arm}: pi_hat={pi_hat:.4f} "
                  f"flag_rate@0.5={rep_out['arms'][arm]['flag_rate_at_0_5']:.4f} "
                  f"({elapsed:.0f}s)")
        rep_out["eval_y"] = eval_y.tolist()
        rep_path.write_text(json.dumps(rep_out) + "\n")
        print(f"wrote {rep_path}")


if __name__ == "__main__":
    main()
