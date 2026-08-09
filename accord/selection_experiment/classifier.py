"""ONE group-blind classifier, fit once on the unfiltered train split.

TF-IDF + logistic regression -- deliberately not a concession, per the task
brief: the object under study is the AUDIT, not the model. What matters is
that it is genuinely informative (not the C28 near-constant classifier that
made demographic parity trivial), and that it never sees group labels.

Text is read, vectorized, and discarded within this module. It is never
printed, logged, or written to disk.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from data_prep import load_split_text

TARGET_FPR = 0.15  # moderate review-burden operating point, same convention
                    # as accord/denominator/simulate.py's fixed threshold.

NON_DEGENERACY_MIN_AUROC = 0.65
NON_DEGENERACY_PRED_POS_RANGE = (0.05, 0.50)
NON_DEGENERACY_MIN_LIFT = 1.5


def fit_reference_classifier(verbose: bool = True):
    """Fit TF-IDF + LogisticRegression on train only. Returns (vec, clf,
    threshold, info). Threshold is chosen from val's ROC curve at
    TARGET_FPR, then frozen."""
    t0 = time.time()
    train_ids, train_texts = load_split_text("train")
    train_labels = _labels_for("train")
    vec = TfidfVectorizer(
        max_features=50_000, ngram_range=(1, 1), min_df=5, max_df=0.9,
        sublinear_tf=True, stop_words="english",
    )
    X_train = vec.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, train_labels)
    fit_seconds = time.time() - t0
    del train_texts, X_train  # never retained

    val_ids, val_texts = load_split_text("val")
    val_labels = _labels_for("val")
    X_val = vec.transform(val_texts)
    scores_val = clf.predict_proba(X_val)[:, 1]
    del val_texts, X_val

    auc = roc_auc_score(val_labels, scores_val)
    fpr, tpr, thr = roc_curve(val_labels, scores_val)
    idx = int(np.argmin(np.abs(fpr - TARGET_FPR)))
    threshold = float(thr[idx])
    pred_val = (scores_val >= threshold).astype(int)
    base_rate = float(val_labels.mean())
    tp = int(((val_labels == 1) & (pred_val == 1)).sum())
    fp = int(((val_labels == 0) & (pred_val == 1)).sum())
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")

    info = dict(
        train_n=len(train_ids), train_pos=int(train_labels.sum()),
        val_n=len(val_ids), val_pos=int(val_labels.sum()), val_base_rate=base_rate,
        val_auroc=float(auc), operating_threshold=threshold,
        operating_fpr=float(fpr[idx]), operating_tpr=float(tpr[idx]),
        val_pred_pos_rate=float(pred_val.mean()), val_ppv=ppv,
        val_precision_lift_over_base_rate=float(ppv / base_rate) if base_rate else float("nan"),
        fit_seconds=round(fit_seconds, 2), vocab_size=len(vec.vocabulary_),
    )
    if verbose:
        print("Reference classifier (TF-IDF + LogisticRegression, fit on train only):")
        for k, v in info.items():
            print(f"  {k}: {v}")
    return vec, clf, threshold, info


def _labels_for(split_name: str) -> np.ndarray:
    import json
    from pathlib import Path
    labels = []
    with (Path("/Users/rickygole/mimic_local_only/task") / f"{split_name}.jsonl").open() as f:
        for line in f:
            labels.append(int(json.loads(line)["label"]))
    return np.array(labels)


def score_population(vec, clf, hadm_ids: list[str], texts: list[str]) -> np.ndarray:
    """Vectorize+score a list of texts (population order preserved). Text is
    not retained after this call returns."""
    X = vec.transform(texts)
    scores = clf.predict_proba(X)[:, 1]
    return scores


def check_non_degeneracy(info: dict) -> tuple[bool, list[str]]:
    """Pre-registered bar. Returns (ok, list_of_failures)."""
    failures = []
    if info["val_auroc"] < NON_DEGENERACY_MIN_AUROC:
        failures.append(
            f"AUROC {info['val_auroc']:.4f} < bar {NON_DEGENERACY_MIN_AUROC}"
        )
    lo, hi = NON_DEGENERACY_PRED_POS_RANGE
    if not (lo < info["val_pred_pos_rate"] < hi):
        failures.append(
            f"predicted-positive rate {info['val_pred_pos_rate']:.4f} outside "
            f"({lo}, {hi}) -- near-constant classifier risk (cf. C28's 87.3%)"
        )
    if info["val_precision_lift_over_base_rate"] < NON_DEGENERACY_MIN_LIFT:
        failures.append(
            f"precision lift {info['val_precision_lift_over_base_rate']:.3f}x "
            f"< bar {NON_DEGENERACY_MIN_LIFT}x"
        )
    return (len(failures) == 0), failures
