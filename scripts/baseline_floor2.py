"""Same floor baselines as scripts/baseline_floor.py, generalized to run on
any of the three tasks built on the same first-48h MIMIC-III bundles
(mortality, 30-day readmission, prolonged LOS). No new logic: identical
feature construction, identical LogisticRegression config, identical three
arms (metadata-only, TF-IDF full text, TF-IDF truncated to 36 tok/note).

  1. METADATA ONLY -- no text whatsoever. Note counts per role, lengths, timing.
  2. TF-IDF on full bundle text.
  3. TF-IDF truncated to the same 36 tokens/note the neural model reads.

Run once per task directory (local_only/task, local_only/task_readmit30,
local_only/task_los7). Splits and rows are read as already built by
build_task.py / build_task2.py -- no re-splitting, no re-filtering, so the
group-disjoint / label-independence guarantees from the builder carry over
unchanged.

Local only; aggregates out, no note text printed or written.
"""

from __future__ import annotations

import argparse
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


def load(task_dir: Path, split):
    return [json.loads(l) for l in (task_dir / f"{split}.jsonl").open()]


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


def score(name, Xtr, ytr, Xva, yva) -> dict:
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             n_jobs=-1, C=1.0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]
    a, ap = roc_auc_score(yva, p), average_precision_score(yva, p)
    print(f"  {name:<44}AUROC {a:.4f}   AUPRC {ap:.4f}")
    return {"name": name, "auroc": float(a), "auprc": float(ap)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True,
                     help="e.g. local_only/task, local_only/task_readmit30, local_only/task_los7")
    ap.add_argument("--task-name", required=True, help="label for the results file / printout")
    ap.add_argument("--out", default=None, help="results json path; default results/baseline_floor_<task-name>.json")
    args = ap.parse_args()

    task_dir = Path(args.task_dir)
    tr, va = load(task_dir, "train"), load(task_dir, "val")
    ytr = np.array([r["label"] for r in tr])
    yva = np.array([r["label"] for r in va])

    print("=" * 72)
    print(f"TASK: {args.task_name}  ({task_dir})")
    print("=" * 72)
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

    out_path = Path(args.out) if args.out else ROOT / "results" / f"baseline_floor_{args.task_name}.json"
    out_path.write_text(json.dumps({
        "task": args.task_name,
        "task_dir": str(task_dir),
        "n_train": len(tr), "n_val": len(va),
        "train_prevalence": float(ytr.mean()), "val_prevalence": float(yva.mean()),
        "results": res,
    }, indent=2))

    best_floor = max(r["auroc"] for r in res)
    meta_only = res[0]["auroc"]
    print("\nVERDICT")
    print(f"  best non-neural floor : {best_floor:.4f}")
    if meta_only > 0.75:
        print(f"  METADATA ALONE reaches {meta_only:.4f} with NO TEXT -- much of this")
        print("  task is note-counting, not language.")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
