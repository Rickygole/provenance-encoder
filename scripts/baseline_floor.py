"""The floor. How much of this task needs an encoder at all?

Three baselines on the identical bundles and identical group-disjoint splits:

  1. METADATA ONLY -- no text whatsoever. Note counts per role, lengths, timing.
     If this scores near the neural models, the task is a severity-proxy counting
     problem and every transformer number is decoration.
  2. TF-IDF on full bundle text.
  3. TF-IDF truncated to the SAME 36 tokens/note the neural model actually reads,
     which is the only text baseline matched to what our encoder sees.

Also reports the leak diagnostic: our task builder drops notes charted after
DEATHTIME, and DEATHTIME is non-null only for positives, so that filter is
label-conditioned. This measures how much signal that leaves in the metadata.

Local only; aggregates out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.roles import N_ROLES, ROLE_NAMES  # noqa: E402

TASK = ROOT / "local_only" / "task"


def load(split):
    return [json.loads(l) for l in (TASK / f"{split}.jsonl").open()]


def meta_features(rows):
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
            len(segs) / max(np.max(ts), 1e-3),          # notes per hour
        ])
    return np.array(X, dtype=np.float64)


def text_of(rows, per_note_tokens=None):
    out = []
    for r in rows:
        parts = []
        for s in r["segments"]:
            t = s["text"]
            if per_note_tokens:
                t = " ".join(t.split()[:per_note_tokens])
            parts.append(t)
        out.append(" ".join(parts))
    return out


def score(name, Xtr, ytr, Xva, yva, sparse=False):
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             n_jobs=-1, C=1.0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]
    a, ap = roc_auc_score(yva, p), average_precision_score(yva, p)
    print(f"  {name:<44}AUROC {a:.4f}   AUPRC {ap:.4f}")
    return {"name": name, "auroc": float(a), "auprc": float(ap)}


def main() -> None:
    tr, va = load("train"), load("val")
    ytr = np.array([r["label"] for r in tr])
    yva = np.array([r["label"] for r in va])
    print(f"train {len(tr):,} ({ytr.mean():.2%} pos) | val {len(va):,} ({yva.mean():.2%} pos)\n")
    res = []

    print("FLOOR BASELINES (no encoder)")
    Mtr, Mva = meta_features(tr), meta_features(va)
    sc = StandardScaler().fit(Mtr)
    res.append(score("metadata only (counts/lengths/timing, NO text)",
                     sc.transform(Mtr), ytr, sc.transform(Mva), yva))

    for tag, cap in (("full text", None), ("truncated to 36 tok/note (what ours reads)", 36)):
        v = TfidfVectorizer(max_features=60_000, ngram_range=(1, 2), min_df=3,
                            sublinear_tf=True)
        Xt = v.fit_transform(text_of(tr, cap))
        Xv = v.transform(text_of(va, cap))
        res.append(score(f"TF-IDF + LR, {tag}", Xt, ytr, Xv, yva))

    print("\nNEURAL (3 seeds, same val split)")
    print(f"  {'ours random init':<44}AUROC 0.8003   AUPRC 0.3242")
    print(f"  {'ours block-mask pretrained':<44}AUROC 0.7956   AUPRC 0.3330")
    print(f"  {'PubMedBERT + concat (1 run, seq 512)':<44}AUROC 0.8223   AUPRC 0.3878")

    (ROOT / "results" / "baseline_floor.json").write_text(json.dumps(res, indent=2))

    best_floor = max(r["auroc"] for r in res)
    meta_only = res[0]["auroc"]
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  best non-neural floor : {best_floor:.4f}")
    print(f"  ours (best arm)       : 0.8003")
    print(f"  PubMedBERT            : 0.8223")
    print(f"  headroom above floor  : {0.8223 - best_floor:+.4f} AUROC")
    if meta_only > 0.75:
        print(f"\n  METADATA ALONE reaches {meta_only:.4f} with NO TEXT. Much of this task")
        print("  is note-counting, not language. Any encoder comparison on it is")
        print("  measuring very little.")
    if best_floor >= 0.79:
        print("\n  The floor is at or above our neural arms. This benchmark CANNOT")
        print("  resolve encoder quality. No pretraining budget or architecture")
        print("  change will show a difference here. The task must change.")
    elif 0.8223 - best_floor < 0.05:
        print("\n  Headroom under 0.05 AUROC. The task has little dynamic range;")
        print("  effects will sit inside seed noise.")
    else:
        print("\n  Real headroom above the floor. The task can resolve models.")


if __name__ == "__main__":
    main()
