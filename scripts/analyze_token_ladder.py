"""Assemble the token-budget ladder table: MLM validation loss and downstream
AUROC side by side at each token-budget rung, with the non-neural floor
(metadata-only LR, TF-IDF) always alongside every row. See
scripts/run_token_ladder.sh, which produces the inputs this reads.

Reads scalars only: results/matched_protocol_<tag-prefix>_b*.json (written by
scripts/run_matched.py) plus each rung's local_only/runs/<ckpt>/summary.json
for the MLM loss (this script runs locally, same as pretrain.py -- it does
not touch note text, only the numeric summary). Writes
results/token_ladder.json, scalars only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def fmt_mean_std(mean, std) -> str:
    if mean is None:
        return "n/a"
    return f"{mean:.4f}+/-{std:.4f}" if std is not None else f"{mean:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag-prefix", default="tokladder")
    args = ap.parse_args()

    files = sorted(RESULTS.glob(f"matched_protocol_{args.tag_prefix}_b*.json"))
    if not files:
        raise SystemExit(
            f"no results/matched_protocol_{args.tag_prefix}_b*.json found -- "
            f"run scripts/run_token_ladder.sh first."
        )

    rows = []
    for f in files:
        d = json.loads(f.read_text())
        ckpt = Path(d["protocol"]["ckpt"])
        summ_path = ckpt / "summary.json"
        mlm_tokens = mlm_loss = None
        if summ_path.exists():
            summ = json.loads(summ_path.read_text())
            mlm_tokens, mlm_loss = summ.get("tokens"), summ.get("final_val_loss")
        else:
            print(f"WARNING: {summ_path} not found (checkpoint deleted after fine-tuning?) "
                  f"-- {f.name} will report AUROC with no matching MLM loss")
        sysd, pw = d.get("systems", {}), d.get("pointwise", {})
        rows.append({
            "tag": f.stem,
            "ckpt": str(ckpt),
            "mlm_tokens": mlm_tokens,
            "mlm_val_loss": mlm_loss,
            "metadata_auroc": pw.get("metadata_lr", {}).get("auroc"),
            "tfidf_auroc": pw.get("tfidf_lr", {}).get("auroc"),
            "ours_random_auroc_mean": sysd.get("ours_random", {}).get("auroc_mean"),
            "ours_random_auroc_std": sysd.get("ours_random", {}).get("auroc_std"),
            "ours_pretrained_auroc_mean": sysd.get("ours_pretrained", {}).get("auroc_mean"),
            "ours_pretrained_auroc_std": sysd.get("ours_pretrained", {}).get("auroc_std"),
        })

    rows.sort(key=lambda r: (r["mlm_tokens"] is None, r["mlm_tokens"] or 0))

    header = (f"{'tokens':>12}  {'mlm_val_loss':>13}  {'metadata':>9}  {'tfidf':>9}  "
              f"{'ours_random':>18}  {'ours_pretrained':>20}")
    print("=" * len(header))
    print("TOKEN-BUDGET LADDER -- MLM loss vs downstream AUROC, floor always alongside")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        tok = f"{r['mlm_tokens']:,}" if r["mlm_tokens"] is not None else "n/a"
        loss = f"{r['mlm_val_loss']:.4f}" if r["mlm_val_loss"] is not None else "n/a"
        meta = f"{r['metadata_auroc']:.4f}" if r["metadata_auroc"] is not None else "n/a"
        tfidf = f"{r['tfidf_auroc']:.4f}" if r["tfidf_auroc"] is not None else "n/a"
        rand = fmt_mean_std(r["ours_random_auroc_mean"], r["ours_random_auroc_std"])
        pre = fmt_mean_std(r["ours_pretrained_auroc_mean"], r["ours_pretrained_auroc_std"])
        print(f"{tok:>12}  {loss:>13}  {meta:>9}  {tfidf:>9}  {rand:>18}  {pre:>20}")

    if len(rows) >= 2:
        span = [r for r in rows if r["ours_pretrained_auroc_mean"] is not None]
        if len(span) >= 2:
            lo, hi = span[0], span[-1]
            d_loss = (lo["mlm_val_loss"] - hi["mlm_val_loss"]
                       if lo["mlm_val_loss"] is not None and hi["mlm_val_loss"] is not None else None)
            d_auroc = hi["ours_pretrained_auroc_mean"] - lo["ours_pretrained_auroc_mean"]
            print(f"\nlowest -> highest rung: "
                  f"{'MLM loss dropped ' + f'{d_loss:.4f} nats, ' if d_loss is not None else ''}"
                  f"ours_pretrained AUROC changed {d_auroc:+.4f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "token_ladder.json").write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nwrote {RESULTS / 'token_ladder.json'}")


if __name__ == "__main__":
    main()
