"""Group-blind reference classifier for ACSIncome: OneHotEncoder(categorical)
+ passthrough-scaled numeric features -> LogisticRegression. Fit ONCE on the
train split. Threshold chosen on val's ROC curve at a target FPR, then
frozen -- same convention as accord/selection_experiment/classifier.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import data_prep

TARGET_FPR = 0.15
SEED = 42


def build_pipeline(seed: int = SEED) -> Pipeline:
    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), data_prep.CATEGORICAL_FEATURES),
            ("num", StandardScaler(), data_prep.NUMERIC_FEATURES),
        ]
    )
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    return Pipeline([("prep", ct), ("clf", clf)])


def fit_reference_classifier(splits: dict, seed: int = SEED, verbose: bool = False):
    tr, va = splits["train"], splits["val"]
    pipe = build_pipeline(seed=seed)
    pipe.fit(tr["X"], tr["y"])

    val_score = pipe.predict_proba(va["X"])[:, 1]
    fpr_arr, tpr_arr, thr_arr = roc_curve(va["y"], val_score)
    # smallest threshold achieving FPR <= TARGET_FPR (roc_curve fpr is
    # increasing in decreasing threshold order)
    ok = np.where(fpr_arr <= TARGET_FPR)[0]
    threshold = float(thr_arr[ok[-1]]) if len(ok) else float(thr_arr[0])

    val_pred = (val_score >= threshold).astype(int)
    auroc = roc_auc_score(va["y"], val_score)
    pred_pos_rate = val_pred.mean()
    base_rate = va["y"].mean()
    precision = precision_score(va["y"], val_pred, zero_division=0)
    precision_lift = precision / base_rate if base_rate > 0 else float("nan")
    achieved_fpr = float(fpr_arr[ok[-1]]) if len(ok) else float(fpr_arr[0])

    info = dict(
        n_train=int(len(tr["y"])), n_val=int(len(va["y"])),
        train_base_rate=float(tr["y"].mean()), val_base_rate=float(base_rate),
        threshold=threshold, target_fpr=TARGET_FPR, achieved_val_fpr=achieved_fpr,
        val_auroc=float(auroc), val_pred_pos_rate=float(pred_pos_rate),
        val_precision=float(precision), val_precision_lift=float(precision_lift),
    )
    if verbose:
        print(f"train n={info['n_train']} base_rate={info['train_base_rate']:.4f}")
        print(f"val   n={info['n_val']} base_rate={info['val_base_rate']:.4f}")
        print(f"threshold={threshold:.4f} (target FPR={TARGET_FPR}, achieved={achieved_fpr:.4f})")
        print(f"val AUROC={auroc:.4f}  pred_pos_rate={pred_pos_rate:.4f}  "
              f"precision={precision:.4f}  precision_lift={precision_lift:.3f}x")
    return pipe, threshold, info


def check_non_degeneracy(info: dict):
    failures = []
    if info["val_auroc"] < 0.65:
        failures.append(f"AUROC {info['val_auroc']:.4f} < 0.65")
    if not (0.05 < info["val_pred_pos_rate"] < 0.50):
        failures.append(f"predicted-positive rate {info['val_pred_pos_rate']:.4f} "
                         "not strictly between 0.05 and 0.50")
    if info["val_precision_lift"] < 1.5:
        failures.append(f"precision lift {info['val_precision_lift']:.3f}x < 1.5x")
    return (len(failures) == 0), failures


def score_split(pipe: Pipeline, X) -> np.ndarray:
    return pipe.predict_proba(X)[:, 1]
