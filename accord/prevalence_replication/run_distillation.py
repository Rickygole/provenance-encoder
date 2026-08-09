"""
Distillation half of the prevalence replication (see task point 6): does a
small student trained on the LLM's labels reproduce the TEACHER's marginal
positive rate, or the TRUE prevalence?

This is the actual "prevalence inheritance" claim from the clinical
pipeline (student reproduced teacher's per-rule predicted-positive rate to
within 1.5pp) and it has never been tested outside the clinical setting.

Student: TF-IDF (word 1-2 grams) + logistic regression, exactly as
specified in the task ("TF-IDF + logistic regression is fine").

Training target: the LLM's own decision (contradiction=1 / not_contradiction=0)
on the DECIDED rows of a given (model, prompt, prevalence-level) combo.
ABSTAIN and UNPARSEABLE rows are dropped from student training (they carry
no usable teacher label) and the drop count is reported.

Evaluation: 5-fold (or fewer, if a class has <5 members) stratified
cross-validated out-of-fold predictions, so the reported student marginal
is never an in-sample number. If a fold split isn't feasible (e.g. only 2
positives total, matching the 2%-prevalence pool with N=100), that is
reported explicitly rather than silently faked.

Reads only the raw per-combo CSVs already written by run_prevalence_sweep.py
(public SNLI text + already-elicited LLM labels). No network calls.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "results", "raw_predictions")
RESULTS_DIR = os.path.join(HERE, "results")
SEED = 42

FNAME_RE = re.compile(r"^(?P<model>.+)__(?P<prompt>A_bare|B_baserate_abstain)__p(?P<level>[0-9.]+)\.csv$")


def fit_student(df):
    """
    df has columns premise, hypothesis, is_contradiction, pred.
    Returns a dict of distillation metrics, or a dict with fit=False and a
    reason if the fit isn't feasible.
    """
    decided = df[df["pred"].isin(["contradiction", "not_contradiction"])].copy()
    n_dropped = len(df) - len(decided)
    if len(decided) < 10:
        return {"fit": False, "reason": f"only {len(decided)} decided rows (<10)",
                "n_dropped_abstain_or_unparseable": n_dropped}

    decided["y_teacher"] = (decided["pred"] == "contradiction").astype(int)
    decided["y_true"] = decided["is_contradiction"].astype(int)
    X_text = (decided["premise"].fillna("") + " [SEP] " + decided["hypothesis"].fillna("")).tolist()
    y = decided["y_teacher"].to_numpy()

    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    if n_pos < 2 or n_neg < 2:
        # Teacher predicted (essentially) a single class -- report the
        # degenerate marginal directly rather than attempting CV.
        teacher_marginal = float(y.mean())
        true_marginal = float(decided["y_true"].mean())
        return {
            "fit": False,
            "reason": f"teacher label near-degenerate on decided rows (n_pos={n_pos}, n_neg={n_neg}); "
                      f"student would trivially predict the majority class",
            "n_decided": len(decided), "n_dropped_abstain_or_unparseable": n_dropped,
            "teacher_marginal": teacher_marginal, "true_prevalence_decided": true_marginal,
            "student_marginal_oof": teacher_marginal,  # a constant-class fit reproduces the teacher marginal exactly
        }

    k = min(5, n_pos, n_neg)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=3000, sublinear_tf=True)
    X = vec.fit_transform(X_text)
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    oof_pred = cross_val_predict(clf, X, y, cv=skf, method="predict")
    oof_proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]

    teacher_marginal = float(y.mean())
    true_marginal = float(decided["y_true"].mean())
    student_marginal_hard = float(oof_pred.mean())
    student_marginal_soft = float(oof_proba.mean())  # mean OOF P(contradiction); not
    # threshold-dependent, so it doesn't collapse to 0 the way the hard 0.5-threshold
    # prediction can at small n (see REPORT.md -- this happens at N=100 in this project).

    # accuracy of student oof preds vs teacher label (fidelity) and vs true gold
    fidelity_to_teacher = float((oof_pred == y).mean())
    fidelity_to_true = float((oof_pred == decided["y_true"].to_numpy()).mean())
    degenerate_hard = (len(set(oof_pred.tolist())) == 1)

    def closer(x, a, b):
        if abs(x - a) < abs(x - b):
            return "teacher"
        if abs(x - b) < abs(x - a):
            return "true_prevalence"
        return "tied"

    return {
        "fit": True, "k_folds": k, "degenerate_hard_threshold": degenerate_hard,
        "n_decided": len(decided), "n_dropped_abstain_or_unparseable": n_dropped,
        "teacher_marginal": teacher_marginal,
        "true_prevalence_decided": true_marginal,
        "student_marginal_oof": student_marginal_hard,
        "student_marginal_oof_soft": student_marginal_soft,
        "abs_diff_student_vs_teacher": abs(student_marginal_hard - teacher_marginal),
        "abs_diff_student_vs_true": abs(student_marginal_hard - true_marginal),
        "closer_to": closer(student_marginal_hard, teacher_marginal, true_marginal),
        "closer_to_soft": closer(student_marginal_soft, teacher_marginal, true_marginal),
        "oof_accuracy_vs_teacher_label": fidelity_to_teacher,
        "oof_accuracy_vs_true_gold": fidelity_to_true,
    }


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        print("No raw prediction CSVs found -- run run_prevalence_sweep.py first.")
        return

    results = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        m = FNAME_RE.match(fname)
        if not m:
            print(f"  [skip, unrecognized filename] {fname}")
            continue
        model, prompt, level_s = m.group("model"), m.group("prompt"), m.group("level")
        model = model.replace("_", ":", 1) if ":" not in model else model
        # filenames replace ':' with '_' once (see run_prevalence_sweep.py); model
        # names in this project are like "llama3:8b" -> "llama3_8b"
        level = float(level_s)

        df = pd.read_csv(fpath)
        metrics = fit_student(df)
        results.setdefault(model, {}).setdefault(prompt, {})[level] = metrics

        if metrics["fit"]:
            print(f"{model:16s} {prompt:20s} p={level:.2f}  "
                  f"teacher={metrics['teacher_marginal']:.3f}  "
                  f"student_oof={metrics['student_marginal_oof']:.3f}  "
                  f"true={metrics['true_prevalence_decided']:.3f}  "
                  f"closer_to={metrics['closer_to']}")
        else:
            print(f"{model:16s} {prompt:20s} p={level:.2f}  NOT FIT -- {metrics['reason']}")

    with open(os.path.join(RESULTS_DIR, "distillation_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved distillation_results.json to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
