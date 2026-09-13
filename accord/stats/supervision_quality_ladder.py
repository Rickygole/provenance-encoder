"""Does supervision quality change the outcome, holding features and learner fixed?

WHY THIS EXISTS
---------------
The thesis claims the pipeline fails, and varies exactly one stage. Every
reviewer of it has made the same objection: the weak labeller emits hard labels at
a 0.85 marginal on a rare event, which caps inheritable agreement before training
begins, so stages 1 and 3 were never given a fair test and the architectural
findings are unfalsifiable rather than falsified. The missing arm is a detector
trained on supervision that discriminates.

The obvious way to run it, relabelling the 1,679-row pool through a hosted model,
requires transmitting credentialed sentences. PhysioNet's guidance of 24 September
2025 prohibits sharing access to the data with third parties "including sending it
through APIs", and strongly recommends locally deployed models instead. That
reading is available and this project does not resolve it, so no pair is sent.

This runs the same question on labels already in hand. One feature
representation, one learner, three supervision conditions, evaluated against the
pharmacist on held-out folds every time:

  weak    the deployed labeller's own labels, what the pipeline actually trained on
  frontier  Claude Sonnet 5 zero-shot labels, a labeller whose own agreement with
            the pharmacist is +0.3189
  gold    the pharmacist's labels, the ceiling any supervision could reach

If frontier and gold beat weak, the failure is located in the supervision rather
than in the features or the learner, and the architectural nulls are confounded.
If all three sit at chance, the supervision was never the binding constraint and
the thesis's central claim is stronger than it currently argues.

Nothing is transmitted. Note text is read locally and never printed or written.

Run:  python3 accord/stats/supervision_quality_ladder.py
Writes: results/accord/stats/supervision_quality_ladder.json
"""

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOCAL = pathlib.Path.home() / "mimic_local_only" / "from_drive_2026-08-08" / "outputs"
GOLD = pathlib.Path.home() / "mimic_local_only" / "derived" / "gold_standard_accord_results.csv"
DIFF = ROOT / "results" / "accord" / "differential"
OUT = ROOT / "results" / "accord" / "stats" / "supervision_quality_ladder.json"
SEEDS = (0, 1, 2)
FOLDS = 5


def main():
    sheet = pd.read_csv(LOCAL / "annotation_sheet_for_pharmacist.csv", low_memory=False)
    gold = pd.read_csv(GOLD)
    son = {p["pair_id"]: p["pred"] for p in
           json.loads((DIFF / "frontier_anthropic_zero_shot.json").read_text())["predictions"]}
    merged = sheet.merge(gold[["pair_id", "pharm_label"]], on="pair_id", how="inner")
    merged = merged[merged.pharm_label != "ambiguous"].copy()
    merged["y"] = (merged.pharm_label == "contradiction").astype(int)
    merged["frontier"] = merged.pair_id.map(son)
    merged = merged[merged.frontier.notna()].copy()
    merged["frontier"] = merged.frontier.astype(int)

    wl = pd.read_csv(LOCAL / "pharmacist_llama_merged.csv", low_memory=False)
    wmap = (wl.groupby("pair_id")["weak_label"]
              .agg(lambda s: int((s == "contradiction").any())).to_dict())
    merged["weak"] = merged.pair_id.map(wmap)
    merged = merged[merged.weak.notna()].copy()
    merged["weak"] = merged["weak"].astype(int)

    text = (merged.nursing_sentence.fillna("") + " [SEP] "
            + merged.discharge_sentence.fillna("")).to_numpy()
    y = merged.y.to_numpy()

    out = {"n": int(len(merged)), "n_positives": int(y.sum()),
           "folds": FOLDS, "seeds": list(SEEDS),
           "note": ("One TF-IDF representation and one logistic regression throughout. "
                    "Only the TRAINING labels change. Every evaluation is against the "
                    "pharmacist on held-out folds, so the three conditions are scored on "
                    "exactly the same target."),
           "transmission": "none; every input was already on this machine",
           "conditions": {}}

    for cond in ("weak", "frontier", "gold"):
        train_lab = y if cond == "gold" else merged[cond].to_numpy()
        aur, aps, mccs = [], [], []
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
            sc = np.zeros(len(y)); pr = np.zeros(len(y), dtype=int)
            for tr, te in skf.split(text, y):
                if len(set(train_lab[tr])) < 2:
                    sc[te] = 0.5; pr[te] = 0; continue
                v = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)
                Xtr = v.fit_transform(text[tr]); Xte = v.transform(text[te])
                m = LogisticRegression(max_iter=2000, class_weight="balanced")
                m.fit(Xtr, train_lab[tr])
                sc[te] = m.predict_proba(Xte)[:, 1]; pr[te] = m.predict(Xte)
            aur.append(roc_auc_score(y, sc)); aps.append(average_precision_score(y, sc))
            mccs.append(matthews_corrcoef(y, pr))
        out["conditions"][cond] = {
            "train_label_positive_rate": round(float(np.mean(train_lab)), 4),
            "auroc_mean": round(float(np.mean(aur)), 4), "auroc_sd": round(float(np.std(aur)), 4),
            "ap_mean": round(float(np.mean(aps)), 4),
            "mcc_mean": round(float(np.mean(mccs)), 4), "mcc_sd": round(float(np.std(mccs)), 4),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"n={out['n']} positives={out['n_positives']}  {FOLDS}-fold x {len(SEEDS)} seeds, "
          f"evaluated against the pharmacist throughout\n")
    for c, v in out["conditions"].items():
        print(f"  train on {c:9s} (pos rate {v['train_label_positive_rate']:.4f})  "
              f"AUROC {v['auroc_mean']:.4f} +/- {v['auroc_sd']:.4f}   "
              f"AP {v['ap_mean']:.4f}   MCC {v['mcc_mean']:+.4f}")


if __name__ == "__main__":
    main()
