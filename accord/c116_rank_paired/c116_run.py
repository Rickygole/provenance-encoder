"""C116: C111/C114 paired AP permutation tests on within-system ranks.
See C116_PREREGISTRATION.md. Local only; aggregates only.

    python3 accord/c116_rank_paired/c116_run.py
"""
import contextlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accord" / "control"))
sys.path.insert(0, str(ROOT / "accord" / "c111_robustness"))

import c111_run as c  # noqa: E402  (reuses paths, constants and the paired test)

LOG = c.LOG_DIR / f"c116_run_{time.strftime('%Y%m%dT%H%M%S')}.log"
OUT = ROOT / "results" / "accord" / "c116_rank_paired" / "c116.json"


class AbortC(RuntimeError):
    pass


def build_frame():
    from data import build_splits
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    splits = build_splits()
    gold = splits.gold.copy()
    gold["y"] = (gold["pharm_label"] == "contradiction").astype(int)
    gold["g253"] = gold["pharm_label"] != "ambiguous"
    nli = pd.read_csv(c.NLI_PATH)
    dec = pd.read_csv(c.DECONTAM_PRED_PATH)

    y_train = splits.train["y"].to_numpy()
    tr = (splits.train["nursing_sentence"].astype(str) + " [SEP] " + splits.train["discharge_sentence"].astype(str)).tolist()
    gt = (gold["nursing_sentence"].astype(str) + " [SEP] " + gold["discharge_sentence"].astype(str)).tolist()
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(vec.fit_transform(tr), y_train)
    tfidf = clf.predict_proba(vec.transform(gt))[:, 1]

    facts = json.loads(c.FACTS.read_text())
    y_all, m = gold["y"].to_numpy(), gold["g253"].to_numpy()
    if round(float(average_precision_score(y_all, tfidf)), 4) != round(facts["tfidfperm.task_a.g299.ap"]["value"], 4) or \
       round(float(average_precision_score(y_all[m], tfidf[m])), 4) != round(facts["tfidfperm.task_a.g253.ap"]["value"], 4):
        raise AbortC("TF-IDF reproduction failed")

    f = gold[["pair_id", "y", "g253"]].copy()
    f["tfidf"] = tfidf
    f = f.merge(nli[["pair_id", "contra_prob_mean"]], on="pair_id", how="left")
    d = dec[["pair_id"]].copy()
    d["det"] = dec[["prob_seed0", "prob_seed1", "prob_seed2"]].to_numpy().mean(axis=1)
    f = f.merge(d, on="pair_id", how="left")
    g = f[f.g253].reset_index(drop=True)
    if len(g) != 253 or g[["contra_prob_mean", "det"]].isna().any().any():
        raise AbortC("G_253 frame malformed")
    if round(float(average_precision_score(g["y"], g["det"])), 6) != round(facts["c111.task_a.detector.ensemble.ap"]["value"], 6):
        raise AbortC("detector ensemble AP reproduction failed")
    return g


def run():
    g = build_frame()
    y = g["y"].to_numpy()
    raw = {"tfidf": g["tfidf"].to_numpy(), "nli": g["contra_prob_mean"].to_numpy(), "det": g["det"].to_numpy()}
    rk = {k: rankdata(v, method="average") / len(v) for k, v in raw.items()}
    scale = {k: {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())} for k, v in raw.items()}
    for k in raw:
        if abs(average_precision_score(y, raw[k]) - average_precision_score(y, rk[k])) > 1e-12:
            raise AbortC(f"rank transform changed AP for {k}")
    out = {"preregistration": "accord/c116_rank_paired/C116_PREREGISTRATION.md", "raw_score_scale_g253": scale}
    for name, (a, b) in {"t1_tfidf_minus_detector": ("tfidf", "det"),
                         "t2_nli_minus_detector": ("nli", "det"),
                         "t3_tfidf_minus_nli": ("tfidf", "nli")}.items():
        out[name] = c.paired_swap_perm_ap(y, rk[a], rk[b], c.B_PERM, c.SEED)
    t1, t2 = out["t1_tfidf_minus_detector"], out["t2_nli_minus_detector"]
    out["t1_keep_tfidf_claim"] = bool(t1["p_permutation_two_sided"] < 0.05 and t1["observed_diff"] > 0)
    out["t2_nli_claim_allowed"] = bool(t2["p_permutation_two_sided"] < 0.05 and t2["observed_diff"] > 0)
    out["outcome"] = "valid"
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
        try:
            res = run()
        except AbortC as e:
            OUT.write_text(json.dumps({"outcome": "C", "abort_reason": str(e)}, indent=2))
            raise SystemExit("Outcome C; reason in output JSON")
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    for k in ("t1_tfidf_minus_detector", "t2_nli_minus_detector", "t3_tfidf_minus_nli"):
        print(k, f"diff={res[k]['observed_diff']:.4f}", f"p={res[k]['p_permutation_two_sided']:.5f}")
    print("keep_tfidf_claim", res["t1_keep_tfidf_claim"], "nli_claim", res["t2_nli_claim_allowed"])
    print("scales", json.dumps(res["raw_score_scale_g253"]))


if __name__ == "__main__":
    main()
