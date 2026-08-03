"""Two checks that decide what the probe result actually means.

P1 -- WHAT is carrying the signal? Print the top discriminative n-grams per
      class. If they are template scaffolding and register ("service", "d/c",
      "npo", header fragments) rather than clinical content, then the honest
      claim is "role is recoverable from formatting register", which is weaker
      and less interesting than "notes genuinely discuss different things".

P2 -- Does redundancy hold AT THE GRANULARITY THE MODEL SEES? The encoder
      conditions attention over a 512-token window. Whole-note TF-IDF over
      thousands of tokens is a strictly easier problem. If role is 97% from a
      whole note but 70% from a random mid-note 128-token window, the
      redundancy argument collapses for most tokens in the sequence and the
      negative claim is unsupported.

Also reports the Fano upper bound on H(role | text), which upper-bounds what
ANY architecture conditioning on that label could gain.

Local only. Prints corpus-level vocabulary statistics and aggregates; never
whole notes, never records.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from role_probe import OUT_DIR, scrub  # noqa: E402

CACHE = OUT_DIR / "probe_sample.pkl"
WINDOWS = [64, 128, 256, 512, None]  # None = whole note
SEED = 0


def fano_bound(p_err: float, n_classes: int) -> float:
    """Upper bound on H(label | text) in bits, from an achievable error rate."""
    if p_err <= 0:
        return 0.0
    hb = -p_err * math.log2(p_err) - (1 - p_err) * math.log2(1 - p_err)
    return hb + p_err * math.log2(n_classes - 1)


def fit_eval(texts, labels, groups, seed=SEED, max_features=50_000, return_model=False):
    texts = np.array(texts, dtype=object)
    labels = np.array(labels)
    groups = np.array(groups)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(gss.split(texts, labels, groups))
    assert not (set(groups[tr]) & set(groups[te])), "subject leaked"

    # IDF fit on the TRAIN fold only -- otherwise test statistics leak in.
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xtr = vec.fit_transform(texts[tr])
    Xte = vec.transform(texts[te])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr, labels[tr])
    pred = clf.predict(Xte)
    out = {
        "accuracy": float(accuracy_score(labels[te], pred)),
        "macro_f1": float(f1_score(labels[te], pred, average="macro")),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_test_subjects": int(len(set(groups[te]))),
    }
    if return_model:
        return out, clf, vec
    return out


def windows_from(text: str, size: int | None, rng, n_draw: int = 1):
    toks = text.split()
    if size is None or len(toks) <= size:
        return [" ".join(toks)]
    outs = []
    for _ in range(n_draw):
        start = int(rng.integers(0, len(toks) - size))
        outs.append(" ".join(toks[start:start + size]))
    return outs


def main() -> None:
    rng = np.random.default_rng(SEED)
    with CACHE.open("rb") as f:
        buckets = pickle.load(f)
    n_classes = len(buckets)
    print(f"loaded {sum(len(v) for v in buckets.values()):,} notes, {n_classes} roles\n")

    # ---------------- P1: what carries the signal ----------------
    texts, labels, groups = [], [], []
    for role, items in buckets.items():
        for sid, txt in items:
            texts.append(scrub(txt, 2))
            labels.append(role)
            groups.append(sid)
    res_full, clf, vec = fit_eval(texts, labels, groups, return_model=True)
    feats = np.array(vec.get_feature_names_out())
    print("=" * 72)
    print("P1  Top discriminative n-grams per role (corpus vocabulary statistics)")
    print("=" * 72)
    top_terms = {}
    for ci, cls in enumerate(clf.classes_):
        coef = clf.coef_[ci] if clf.coef_.shape[0] > 1 else clf.coef_[0]
        idx = np.argsort(-coef)[:25]
        terms = [feats[i] for i in idx]
        top_terms[str(cls)] = terms
        print(f"\n  {cls}:")
        print("    " + ", ".join(terms))

    # ---------------- P2: granularity ----------------
    print("\n" + "=" * 72)
    print("P2  Recoverability at the granularity the model actually consumes")
    print("=" * 72)
    print(f"\n{'window (tokens)':>16}{'accuracy':>11}{'macro_f1':>11}"
          f"{'H(role|text) <=':>18}{'of 1.585 bits':>15}")
    print("-" * 72)
    rows = []
    for size in WINDOWS:
        wt, wl, wg = [], [], []
        for role, items in buckets.items():
            for sid, txt in items:
                s = scrub(txt, 2)
                for w in windows_from(s, size, rng, n_draw=1):
                    if len(w.split()) < 16:
                        continue
                    wt.append(w)
                    wl.append(role)
                    wg.append(sid)
        r = fit_eval(wt, wl, wg)
        bound = fano_bound(1.0 - r["accuracy"], n_classes)
        marginal = math.log2(n_classes)
        r.update({"window": size, "fano_bits": bound, "frac_of_marginal": bound / marginal})
        rows.append(r)
        label = "whole note" if size is None else str(size)
        print(f"{label:>16}{r['accuracy']:>11.4f}{r['macro_f1']:>11.4f}"
              f"{bound:>18.3f}{bound / marginal:>14.1%}")

    payload = {"full_note_scrubbed": res_full, "top_terms": top_terms, "windows": rows}
    out = OUT_DIR / "probe_granularity.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")

    smallest = [r for r in rows if r["window"] == 64][0]
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  whole note : {rows[-1]['accuracy']:.1%}   H(role|text) <= {rows[-1]['fano_bits']:.3f} bits")
    print(f"  64 tokens  : {smallest['accuracy']:.1%}   H(role|text) <= {smallest['fano_bits']:.3f} bits")
    if smallest["accuracy"] > 0.90:
        print("\n  Redundancy HOLDS at model granularity. The provenance label supplies")
        print("  almost no information even in a short window. The negative claim is")
        print("  airtight for the information-channel framing.")
    elif smallest["accuracy"] > 0.75:
        print("\n  Redundancy WEAKENS at short windows. Some tokens sit in genuinely")
        print("  ambiguous context. The negative claim must be scoped by window size.")
    else:
        print("\n  Redundancy COLLAPSES at short windows. Most tokens the model sees are")
        print("  provenance-ambiguous, and the negative claim is NOT supported at the")
        print("  granularity that matters. The architecture may still have a role.")


if __name__ == "__main__":
    main()
