"""
Stage 0 re-diagnosis: training driver.

Runs one of four configurations (frozen BioMedBERT backbone, matching the
original 592,647-trainable-parameter run):

  --arch bottleneck --window head   baseline (reproduction of the published run)
  --arch direct     --window head   H1 ablation (fix the 1-D bottleneck only)
  --arch bottleneck --window tail   H2 ablation (fix the truncation window only)
  --arch direct     --window tail   both fixed

Writes an aggregate-only JSON results file (loss/accuracy per epoch, param
counts, split sizes). NEVER writes note text.

Usage:
  python train.py --arch bottleneck --window head --out /path/to/out.json
"""
import argparse
import json
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from data import build_datasets
from model import ATCGPretrainer, DirectHeadPretrainer

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, device, backbone_frozen):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.no_grad()
    with ctx:
        for batch in loader:
            n_ids = batch["nursing_input_ids"].to(device)
            n_mask = batch["nursing_attention_mask"].to(device)
            d_ids = batch["discharge_input_ids"].to(device)
            d_mask = batch["discharge_attention_mask"].to(device)
            labels = batch["label"].to(device)

            loss, logits, _ = model(n_ids, n_mask, d_ids, d_mask, labels)
            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["bottleneck", "direct"], required=True)
    ap.add_argument("--window", choices=["head", "tail", "meds"], required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dev-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"[config] arch={args.arch} window={args.window} device={device}")

    print("Loading tokenizer + backbone...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME)

    print("Building datasets (subject-level split, fixed pairs)...")
    train_ds, dev_ds, meta = build_datasets(tokenizer, window=args.window)
    print(f"[data] {meta}")

    if args.arch == "bottleneck":
        pretrainer = ATCGPretrainer(backbone)
    else:
        pretrainer = DirectHeadPretrainer(backbone)
    pretrainer = pretrainer.to(device)

    # Frozen backbone, matching the original 592,647-trainable-param config.
    for p in pretrainer.backbone.parameters():
        p.requires_grad = False
    for p in pretrainer.node_projection.parameters():
        p.requires_grad = True
    if args.arch == "bottleneck":
        for p in pretrainer.atcg.parameters():
            p.requires_grad = True
        for p in pretrainer.coherence_head.parameters():
            p.requires_grad = True
    else:
        for p in pretrainer.classifier.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in pretrainer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in pretrainer.parameters())
    print(f"[params] trainable={trainable:,} total={total_params:,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.dev_batch_size, shuffle=False)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, pretrainer.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    history = []
    best_dev_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        pretrainer.train()
        total_train_loss = 0.0
        n_seen = 0
        t0 = time.time()

        for batch in train_loader:
            n_ids = batch["nursing_input_ids"].to(device)
            n_mask = batch["nursing_attention_mask"].to(device)
            d_ids = batch["discharge_input_ids"].to(device)
            d_mask = batch["discharge_attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss, logits, _ = pretrainer(n_ids, n_mask, d_ids, d_mask, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item() * labels.size(0)
            n_seen += labels.size(0)

        avg_train_loss = total_train_loss / n_seen
        dev_loss, dev_acc = evaluate(pretrainer, dev_loader, device, backbone_frozen=True)
        elapsed = time.time() - t0

        print(f"epoch {epoch+1:>2}  train_loss={avg_train_loss:.4f}  "
              f"dev_loss={dev_loss:.4f}  dev_acc={dev_acc:.4f}  ({elapsed:.0f}s)")

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "dev_loss": dev_loss,
            "dev_acc": dev_acc,
            "elapsed_sec": elapsed,
        })
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch + 1

    result = {
        "arch": args.arch,
        "window": args.window,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "trainable_params": trainable,
        "total_params": total_params,
        "data_meta": meta,
        "history": history,
        "final_train_loss": history[-1]["train_loss"],
        "final_dev_loss": history[-1]["dev_loss"],
        "final_dev_acc": history[-1]["dev_acc"],
        "best_dev_acc": best_dev_acc,
        "best_epoch": best_epoch,
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
