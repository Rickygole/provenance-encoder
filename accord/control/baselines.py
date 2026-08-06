"""
Defect-free control: cheap floor baselines (majority-class, TF-IDF +
logistic regression) on the SAME subject-disjoint split as the vanilla
BioMedBERT control, trained on the identical weak-label training pool (no
rebalancing, no weak_confidence use -- like-for-like with the main model).

Writes /Users/rickygole/mimic_local_only/accord_control/baselines.json.
NEVER persists note text; TF-IDF vectorizer is fit in-process and discarded.
"""
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from data import build_splits

OUT_DIR = "/Users/rickygole/mimic_local_only/accord_control"
BASELINE_SEED = 42  # LogisticRegression solver seed; deterministic given fixed data, one run suffices


def top_k_tp(scores: np.ndarray, y: np.ndarray, k: int) -> int:
    order = np.argsort(-scores)
    return int(y[order[:k]].sum())


def block_metrics(y: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> dict:
    yy, ss = y[mask], scores[mask]
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


def main():
    splits = build_splits()
    y_train = splits.train["y"].to_numpy()
    y_gold = splits.gold["pharm_pos"].to_numpy()
    rule = splits.gold["rule"].to_numpy()
    defect_free_mask = np.isin(rule, [3, 4])
    all_mask = np.ones_like(defect_free_mask, dtype=bool)

    results = {}

    # ---- majority-class baseline ----
    # Constant score for every gold row = the training-pool positive rate
    # (i.e. "always predict the majority training class"). With a constant
    # score, roc_auc_score returns exactly 0.5 (all ties -> no discrimination)
    # and average_precision_score reduces to the achieved positive rate in
    # the scored set -- reported explicitly rather than assumed.
    maj_rate = float(y_train.mean())
    const_scores = np.full_like(y_gold, fill_value=maj_rate, dtype=float)
    results["majority_class"] = {
        "description": "constant score = training-pool positive rate; predicts the majority weak-label class (contradiction, 85.2% of training rows) for every gold pair",
        "train_majority_rate": maj_rate,
        "gold_all": block_metrics(y_gold, const_scores, all_mask),
        "gold_defect_free_rule34": block_metrics(y_gold, const_scores, defect_free_mask),
    }

    # ---- TF-IDF + logistic regression ----
    train_text = (splits.train["nursing_sentence"].astype(str) + " [SEP] " + splits.train["discharge_sentence"].astype(str)).tolist()
    gold_text = (splits.gold["nursing_sentence"].astype(str) + " [SEP] " + splits.gold["discharge_sentence"].astype(str)).tolist()

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train = vec.fit_transform(train_text)
    X_gold = vec.transform(gold_text)

    clf = LogisticRegression(max_iter=2000, random_state=BASELINE_SEED)
    clf.fit(X_train, y_train)
    tfidf_scores = clf.predict_proba(X_gold)[:, 1]

    results["tfidf_logreg"] = {
        "description": "TF-IDF (1-2gram, max_features=5000, min_df=2) over 'nursing [SEP] discharge' concatenated text + LogisticRegression, trained on the identical weak-label pool, no class rebalancing",
        "max_features": 5000,
        "ngram_range": [1, 2],
        "gold_all": block_metrics(y_gold, tfidf_scores, all_mask),
        "gold_defect_free_rule34": block_metrics(y_gold, tfidf_scores, defect_free_mask),
    }

    results["n_train"] = len(splits.train)
    results["n_gold"] = len(splits.gold)
    results["train_pos_rate"] = maj_rate

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "baselines.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
