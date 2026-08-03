"""How hard is it to HIDE note provenance from a linear model?

The role probe showed 97.5% recoverability after marker-scrubbing and
length-matching. The obvious rebuttal is "you scrubbed the wrong markers." This
answers that empirically: iteratively delete the most role-discriminative
n-grams and refit. If accuracy survives deleting thousands of features, the
signal is diffuse and irreducible rather than carried by a few give-away tokens.

Caches its sample to local_only/ so repeat runs skip the 1.1GB scan.
Local only; emits aggregates, never note text.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from role_probe import (  # noqa: E402
    NOTEEVENTS,
    OUT_DIR,
    length_matched,
    sample_notes,
    scrub,
)

CACHE = OUT_DIR / "probe_sample.pkl"


def get_buckets(path: str, per_role: int, chunk: int, seed: int, refresh: bool):
    if CACHE.exists() and not refresh:
        print(f"loading cached sample from {CACHE}")
        with CACHE.open("rb") as f:
            return pickle.load(f)
    buckets = sample_notes(path, per_role, chunk, seed)
    OUT_DIR.mkdir(exist_ok=True)
    with CACHE.open("wb") as f:
        pickle.dump(buckets, f)
    print(f"cached sample to {CACHE}")
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=NOTEEVENTS)
    ap.add_argument("--per-role", type=int, default=6000)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    buckets = get_buckets(args.path, args.per_role, args.chunk, args.seed, args.refresh)
    print("sampled:", {k: len(v) for k, v in buckets.items()})

    lm = length_matched(buckets, args.seed)
    if lm is None:
        raise SystemExit("length matching failed")
    print("length-matched:", {k: len(v) for k, v in lm.items()})

    texts, labels, groups = [], [], []
    for role, items in lm.items():
        for sid, txt in items:
            texts.append(scrub(txt, 2))  # hardest scrub level
            labels.append(role)
            groups.append(sid)
    texts = np.array(texts, dtype=object)
    labels = np.array(labels)
    groups = np.array(groups)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    tr, te = next(gss.split(texts, labels, groups))
    assert not (set(groups[tr]) & set(groups[te]))

    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xtr_full = vec.fit_transform(texts[tr])
    Xte_full = vec.transform(texts[te])
    feat_names = np.array(vec.get_feature_names_out())
    print(f"vocabulary: {len(feat_names):,} features")

    # Rank features by maximum |coefficient| across classes on the full model.
    base = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)
    base.fit(Xtr_full, labels[tr])
    importance = np.abs(base.coef_).max(axis=0)
    order = np.argsort(-importance)

    schedule = [0, 10, 50, 100, 500, 1_000, 5_000, 10_000, 20_000, 40_000]
    schedule = [k for k in schedule if k < len(feat_names)]

    rows = []
    for k in schedule:
        banned = set(order[:k].tolist())
        keep = np.array([i for i in range(len(feat_names)) if i not in banned])
        Xtr, Xte = Xtr_full[:, keep], Xte_full[:, keep]
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)
        clf.fit(Xtr, labels[tr])
        pred = clf.predict(Xte)
        acc = accuracy_score(labels[te], pred)
        mf1 = f1_score(labels[te], pred, average="macro")
        rows.append({"removed": int(k), "kept": int(len(keep)), "accuracy": float(acc), "macro_f1": float(mf1)})
        print(f"  removed top {k:>6,} features -> acc {acc:.4f}  macro_f1 {mf1:.4f}  (kept {len(keep):,})")

    out = OUT_DIR / "probe_robustness.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")

    # Verdict must be read off the SHAPE of the curve, not the endpoint. Deleting
    # 93% of the vocabulary will lower any classifier; that says nothing about
    # whether the signal was concentrated. The diagnostic question is how much
    # accuracy the FIRST few thousand features were carrying.
    chance = 1.0 / len(set(labels))
    by_k = {r["removed"]: r["accuracy"] for r in rows}
    base = by_k[0]
    at_1k = by_k.get(1_000, base)
    frac_vocab_to_halve = None
    for r in rows:
        if r["accuracy"] <= base - (base - chance) / 2:
            frac_vocab_to_halve = r["removed"] / len(feat_names)
            break

    print("\n" + "=" * 68)
    print(f"chance = {chance:.1%}   full-vocab accuracy = {base:.1%}")
    print(f"cost of removing the top 1,000 features: {base - at_1k:+.1%}")
    if frac_vocab_to_halve is None:
        print("removing every scheduled feature never halved the above-chance margin")
    else:
        print(f"had to remove {frac_vocab_to_halve:.0%} of the vocabulary to halve "
              f"the above-chance margin")

    if base - at_1k < 0.05:
        print("\nSignal is DIFFUSE. The top 1,000 discriminative n-grams carry almost")
        print("none of it, so role cannot be hidden by scrubbing markers -- it is")
        print("distributed across the whole vocabulary. Architectural provenance")
        print("conditioning supplies the model nothing it would not extract itself.")
    else:
        print("\nSignal concentrates in a small feature set; targeted scrubbing is viable.")


if __name__ == "__main__":
    main()
