"""Assemble the label-budget ladder table: AUROC by training-set size for the
non-neural floor (metadata-only LR, TF-IDF), our random-init encoder, and our
pretrained encoder -- with the pretrained-minus-random gap called out
explicitly, since that gap (not the raw AUROC) is the number that answers
"does pretraining help in the low-label regime." See
scripts/run_label_ladder.sh, which produces the inputs this reads.

Reads scalars only: results/matched_protocol_<tag-prefix>_*.json. Writes
results/label_ladder.json, scalars only.
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
    ap.add_argument("--tag-prefix", default="labelladder")
    args = ap.parse_args()

    files = sorted(RESULTS.glob(f"matched_protocol_{args.tag_prefix}_*.json"))
    if not files:
        raise SystemExit(
            f"no results/matched_protocol_{args.tag_prefix}_*.json found -- "
            f"run scripts/run_label_ladder.sh first."
        )

    prefix = f"matched_protocol_{args.tag_prefix}_"
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        budget_tag = f.stem[len(prefix):]
        sub = d["protocol"].get("subsample_train")
        n_train = sub["achieved_n"] if sub else d.get("n_train")
        prevalence = sub["achieved_prevalence"] if sub else None
        sysd, pw = d.get("systems", {}), d.get("pointwise", {})
        rand_mean = sysd.get("ours_random", {}).get("auroc_mean")
        pre_mean = sysd.get("ours_pretrained", {}).get("auroc_mean")
        rows.append({
            "budget_tag": budget_tag,
            "n_train": n_train,
            "prevalence": prevalence,
            "metadata_auroc": pw.get("metadata_lr", {}).get("auroc"),
            "tfidf_auroc": pw.get("tfidf_lr", {}).get("auroc"),
            "ours_random_auroc_mean": rand_mean,
            "ours_random_auroc_std": sysd.get("ours_random", {}).get("auroc_std"),
            "ours_pretrained_auroc_mean": pre_mean,
            "ours_pretrained_auroc_std": sysd.get("ours_pretrained", {}).get("auroc_std"),
            "pretrain_minus_random": (pre_mean - rand_mean)
                                      if pre_mean is not None and rand_mean is not None else None,
        })

    def sort_key(r):
        try:
            return (0, int(r["n_train"] or 0))
        except (TypeError, ValueError):
            return (1, 0)
    rows.sort(key=sort_key)

    header = (f"{'budget':>10}  {'n_train':>8}  {'metadata':>9}  {'tfidf':>9}  "
              f"{'ours_random':>18}  {'ours_pretrained':>20}  {'pretrain-random':>16}")
    print("=" * len(header))
    print("LABEL-BUDGET LADDER -- floor always alongside; pretrain-random is the number "
          "that matters")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        ntr = f"{r['n_train']:,}" if r["n_train"] is not None else "n/a"
        meta = f"{r['metadata_auroc']:.4f}" if r["metadata_auroc"] is not None else "n/a"
        tfidf = f"{r['tfidf_auroc']:.4f}" if r["tfidf_auroc"] is not None else "n/a"
        rand = fmt_mean_std(r["ours_random_auroc_mean"], r["ours_random_auroc_std"])
        pre = fmt_mean_std(r["ours_pretrained_auroc_mean"], r["ours_pretrained_auroc_std"])
        gap = f"{r['pretrain_minus_random']:+.4f}" if r["pretrain_minus_random"] is not None else "n/a"
        print(f"{r['budget_tag']:>10}  {ntr:>8}  {meta:>9}  {tfidf:>9}  "
              f"{rand:>18}  {pre:>20}  {gap:>16}")
    print("\nseed noise measured elsewhere on this task is +/-0.0079 AUROC -- a "
          "pretrain-random gap smaller than that, at any budget, is not resolved by "
          "the seeds used here.")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "label_ladder.json").write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nwrote {RESULTS / 'label_ladder.json'}")


if __name__ == "__main__":
    main()
