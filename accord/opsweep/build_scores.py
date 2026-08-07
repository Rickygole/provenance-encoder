"""
ACCORD operating-point sweep -- Step 0: assemble the per-pair scoring table.

Builds, for the 299 gold pairs, a table with:
  pair_id, rule, pharm_pos (ground truth), language_group3,
  accord_score      (accord_contra_prob, from gold_standard_accord_results.csv)
  control_score      (mean of vanilla subject-disjoint control, seeds 42/123/2024)
  tfidf_score        (TF-IDF + LogisticRegression, retrained here on the identical
                       subject-disjoint training pool used by accord/control/baselines.py)

No clinical note text is written anywhere. Only pair_id, a demographic group label,
rule id, ground-truth label, and three numeric scores leave this script.

Output: /Users/rickygole/mimic_local_only/accord_opsweep/gold_scores_language.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "/Users/rickygole/provenance-encoder/accord/control")
from data import build_splits  # noqa: E402

OUT_DIR = "/Users/rickygole/mimic_local_only/accord_opsweep"
os.makedirs(OUT_DIR, exist_ok=True)

CONTROL_SCORE_PATHS = [
    "/Users/rickygole/mimic_local_only/accord_control/scores_seed42.csv",
    "/Users/rickygole/mimic_local_only/accord_control/scores_seed123.csv",
    "/Users/rickygole/mimic_local_only/accord_control/scores_seed2024.csv",
]

TFIDF_SEED = 42  # matches accord/control/baselines.py BASELINE_SEED


def lang_group(x):
    if pd.isna(x):
        return "MISSING"
    if x == "ENGL":
        return "ENGLISH"
    return "RECORDED_NON_ENGLISH"


def build_demo():
    adm = pd.read_csv("/Users/rickygole/Downloads/ADMISSIONS.csv")
    adm["HADM_ID"] = adm["HADM_ID"].astype(float)
    demo = adm[["HADM_ID", "LANGUAGE"]].drop_duplicates("HADM_ID")
    demo["language_group3"] = demo["LANGUAGE"].apply(lang_group)
    return demo[["HADM_ID", "language_group3"]]


def main():
    splits = build_splits()
    gold = splits.gold.copy()  # 299 rows, has HADM_ID, pair_id, rule, pharm_pos, accord_contra_prob
    train = splits.train

    demo = build_demo()
    gold = gold.merge(demo, on="HADM_ID", how="left")
    assert gold["language_group3"].notna().all(), "unmatched demo join -- STOP"

    vc = gold["language_group3"].value_counts()
    print("Language group3 counts on gold (299):")
    print(vc.to_string())
    assert set(vc.index) == {"ENGLISH", "MISSING", "RECORDED_NON_ENGLISH"}

    # ---- TF-IDF + logistic regression, retrained on the identical subject-disjoint pool ----
    train_text = (train["nursing_sentence"].astype(str) + " [SEP] " + train["discharge_sentence"].astype(str)).tolist()
    gold_text = (gold["nursing_sentence"].astype(str) + " [SEP] " + gold["discharge_sentence"].astype(str)).tolist()
    y_train = train["y"].to_numpy()

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train = vec.fit_transform(train_text)
    X_gold = vec.transform(gold_text)
    clf = LogisticRegression(max_iter=2000, random_state=TFIDF_SEED)
    clf.fit(X_train, y_train)
    gold["tfidf_score"] = clf.predict_proba(X_gold)[:, 1]

    # sanity check against the saved baselines.json AUROC
    from sklearn.metrics import roc_auc_score, average_precision_score
    y_gold = gold["pharm_pos"].to_numpy()
    auroc = roc_auc_score(y_gold, gold["tfidf_score"])
    ap = average_precision_score(y_gold, gold["tfidf_score"])
    print(f"TF-IDF+LR rebuild check: AUROC={auroc:.4f} AP={ap:.4f} (expect AUROC~0.576, AP~0.157)")

    # ---- accord score (already on gold df) ----
    gold["accord_score"] = gold["accord_contra_prob"]

    # ---- vanilla subject-disjoint control, mean of 3 seeds ----
    ctrl_scores = []
    for p in CONTROL_SCORE_PATHS:
        df = pd.read_csv(p)[["pair_id", "score"]].rename(columns={"score": os.path.basename(p)})
        ctrl_scores.append(df)
    ctrl = ctrl_scores[0]
    for df in ctrl_scores[1:]:
        ctrl = ctrl.merge(df, on="pair_id", how="inner")
    assert len(ctrl) == 299
    score_cols = [c for c in ctrl.columns if c != "pair_id"]
    ctrl["control_score"] = ctrl[score_cols].mean(axis=1)

    gold = gold.merge(ctrl[["pair_id", "control_score"]], on="pair_id", how="left")
    assert gold["control_score"].notna().all()

    out = gold[["pair_id", "rule", "pharm_pos", "language_group3",
                "accord_score", "control_score", "tfidf_score"]].copy()

    out_path = os.path.join(OUT_DIR, "gold_scores_language.csv")
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  shape={out.shape}")
    print("\nGround-truth positives by language group (pharm_pos, n=12 total):")
    print(out.groupby("language_group3")["pharm_pos"].agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
