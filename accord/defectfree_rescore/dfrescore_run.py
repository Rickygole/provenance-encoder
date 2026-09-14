"""C110: is the ranking null concentrated in the exemplar-defect rows (Rules 1/2)
or does it persist on the defect-free strata (Rules 3/4)? See
C110_PREREGISTRATION.md. Pre-registration committed before this script was run.

Three systems, all from EXISTING scores -- no new model runs:
  - detector: accord_contra_prob (gold_standard_accord_results.csv)
  - NLI baseline: contra_prob_mean (nli_scores_roberta_large_mnli.csv)
  - TF-IDF logreg: recomputed deterministically via accord.control.data.build_splits,
    exactly as accord/tfidf_permutation/tfidfperm_run.py does, with a mandatory
    reproduction check against the registered G_299 baseline before any stratum
    number from it is used.

Local only. Any code path that touches ~/mimic_local_only redirects stdout/stderr
to a log file under ~/mimic_local_only/logs; only the aggregate JSON this script
writes under results/accord/defectfree_rescore/ and a short numeric summary ever
reach real stdout. No pair_id, sentence, or row is printed.

    python3 accord/defectfree_rescore/dfrescore_run.py
"""
from __future__ import annotations

import contextlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accord" / "control"))

from accord.cloud_paths import assert_local_only  # noqa: E402

LOG_DIR = pathlib.Path.home() / "mimic_local_only" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = assert_local_only(LOG_DIR / f"dfrescore_{time.strftime('%Y%m%dT%H%M%S')}.log")

OUT = ROOT / "results" / "accord" / "defectfree_rescore" / "dfrescore_strata.json"

DETECTOR_PATH = pathlib.Path.home() / "mimic_local_only" / "derived" / "gold_standard_accord_results.csv"
NLI_PATH = pathlib.Path.home() / "mimic_local_only" / "derived" / "nli_scores_roberta_large_mnli.csv"
PHARM_PATH = (
    pathlib.Path.home() / "mimic_local_only" / "from_drive_2026-08-08" / "outputs" / "pharmacist_labeled_clean.csv"
)
BASELINES_PATH = ROOT / "results" / "accord" / "control" / "baselines.json"
FACTS_PATH = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"

SEED = 42
B_PERM = 20000

STRATA_DEFS = {
    "rules12_ambiguous_dropped": {"rules": (1, 2), "ambiguous": "drop"},
    "rules12_ambiguous_as_negative": {"rules": (1, 2), "ambiguous": "negative"},
    "rules34_ambiguous_dropped": {"rules": (3, 4), "ambiguous": "drop"},
    "rules34_ambiguous_as_negative": {"rules": (3, 4), "ambiguous": "negative"},
}


def perm_p(y: np.ndarray, s: np.ndarray, B: int, seed_seq) -> dict:
    """One-sided label-permutation null holding scores s fixed. Returns
    observed AUROC/AP and permutation p for each (upper-tail: null >= observed)."""
    rng = np.random.default_rng(seed_seq)
    n = len(y)
    obs_auc = float(roc_auc_score(y, s))
    obs_ap = float(average_precision_score(y, s))
    idx = np.arange(n)
    null_auc = np.empty(B)
    null_ap = np.empty(B)
    for b in range(B):
        yp = y[rng.permutation(idx)]
        null_auc[b] = roc_auc_score(yp, s)
        null_ap[b] = average_precision_score(yp, s)
    return {
        "n": int(n),
        "n_pos": int(y.sum()),
        "auroc": obs_auc,
        "ap": obs_ap,
        "p_permutation_auroc": float((null_auc >= obs_auc).mean()),
        "p_permutation_ap": float((null_ap >= obs_ap).mean()),
        "B": int(B),
    }


def mask_for(rule_arr: np.ndarray, ambiguous_arr: np.ndarray, rules, ambig_mode) -> np.ndarray:
    rule_mask = np.isin(rule_arr, rules)
    if ambig_mode == "drop":
        return rule_mask & (~ambiguous_arr)
    return rule_mask  # ambiguous handled at label level (counted as negative)


def build_label_join() -> pd.DataFrame:
    """pair_id -> rule, label_clean, is_ambiguous, is_pos(contradiction).
    label agreement between gold_standard_accord_results.csv's own pharm_label
    and pharmacist_labeled_clean.csv's label_clean is checked, not assumed."""
    det = pd.read_csv(DETECTOR_PATH)
    pharm = pd.read_csv(PHARM_PATH)

    j = det[["pair_id", "rule", "accord_contra_prob", "pharm_label"]].merge(
        pharm[["pair_id", "label_clean"]], on="pair_id", how="inner"
    )
    if j.shape[0] != 299:
        raise SystemExit(f"ABORT: detector<->pharmacist join is not 299 rows (got {j.shape[0]})")
    if j["label_clean"].isna().any():
        raise SystemExit("ABORT: unjoined pair_id against pharmacist_labeled_clean.csv")

    n_label_mismatch = int((j["pharm_label"] != j["label_clean"]).sum())
    j["is_ambiguous"] = (j["label_clean"] == "ambiguous")
    j["is_pos"] = (j["label_clean"] == "contradiction").astype(int)
    j["label_mismatch_count"] = n_label_mismatch  # constant column, carried for reporting
    return j


def run() -> dict:
    labels = build_label_join()
    label_mismatch = int(labels["label_mismatch_count"].iloc[0])
    rule_arr = labels["rule"].to_numpy()
    ambig_arr = labels["is_ambiguous"].to_numpy()
    y_arr = labels["is_pos"].to_numpy()

    # ---- detector ----
    det_scores = labels["accord_contra_prob"].to_numpy()

    # ---- NLI baseline: join contra_prob_mean, rule/labels reused from `labels` ----
    nli = pd.read_csv(NLI_PATH)
    nli_j = labels[["pair_id"]].merge(nli[["pair_id", "contra_prob_mean"]], on="pair_id", how="left")
    if nli_j["contra_prob_mean"].isna().any():
        raise SystemExit("ABORT: unjoined pair_id against nli_scores_roberta_large_mnli.csv")
    nli_scores = nli_j["contra_prob_mean"].to_numpy()

    # ---- TF-IDF logreg: reproduce tfidfperm_run.py's task (a) fit, mandatory check ----
    from data import build_splits  # noqa: E402 (imported inside redirected scope)

    splits = build_splits()
    y_train = splits.train["y"].to_numpy()
    gold = splits.gold
    train_text = (
        splits.train["nursing_sentence"].astype(str) + " [SEP] " + splits.train["discharge_sentence"].astype(str)
    ).tolist()
    gold_text = (
        gold["nursing_sentence"].astype(str) + " [SEP] " + gold["discharge_sentence"].astype(str)
    ).tolist()

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train = vec.fit_transform(train_text)
    X_gold = vec.transform(gold_text)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    scores_gold_by_pairid = pd.Series(
        clf.predict_proba(X_gold)[:, 1], index=gold["pair_id"].to_numpy()
    )

    registered = json.loads(BASELINES_PATH.read_text())["tfidf_logreg"]["gold_all"]
    y_gold_pos = gold["pharm_pos"].to_numpy()
    auroc_299 = float(roc_auc_score(y_gold_pos, scores_gold_by_pairid.reindex(gold["pair_id"]).to_numpy()))
    ap_299 = float(average_precision_score(y_gold_pos, scores_gold_by_pairid.reindex(gold["pair_id"]).to_numpy()))
    tfidf_reproduction_ok = (
        round(auroc_299, 4) == round(registered["auroc"], 4)
        and round(ap_299, 4) == round(registered["ap"], 4)
    )

    tfidf_scores = None
    if tfidf_reproduction_ok:
        aligned = scores_gold_by_pairid.reindex(labels["pair_id"].to_numpy())
        if aligned.isna().any():
            raise SystemExit("ABORT: TF-IDF gold scores did not align to the label join by pair_id")
        tfidf_scores = aligned.to_numpy()

    # ---- G_253 AUROC per system (descriptive ordering comparison only) ----
    g253_mask = ~ambig_arr
    g253 = {
        "detector_auroc": float(roc_auc_score(y_arr[g253_mask], det_scores[g253_mask])),
        "nli_auroc": float(roc_auc_score(y_arr[g253_mask], nli_scores[g253_mask])),
        "tfidf_auroc": (
            float(roc_auc_score(y_arr[g253_mask], tfidf_scores[g253_mask])) if tfidf_scores is not None else None
        ),
        "detector_auroc_registered_source": "results/accord/VERIFIED_FACTS.json::signal.contra_prob_vs_pharm_label__ambiguous_excluded_n253.auroc",
        "detector_auroc_registered_value": 0.4595,
    }

    ss = np.random.SeedSequence(SEED)
    stratum_names = list(STRATA_DEFS.keys())
    n_systems = 3  # detector, nli, tfidf
    children = ss.spawn(len(stratum_names) * n_systems)
    child_idx = 0

    strata_results = {}
    for sname, sdef in STRATA_DEFS.items():
        m = mask_for(rule_arr, ambig_arr, sdef["rules"], sdef["ambiguous"])
        n = int(m.sum())
        n_pos = int(y_arr[m].sum())

        entry = {"n": n, "n_pos": n_pos, "rules": list(sdef["rules"]), "ambiguous_handling": sdef["ambiguous"]}

        entry["detector"] = perm_p(y_arr[m], det_scores[m], B_PERM, children[child_idx]) if n_pos >= 1 else None
        child_idx += 1
        entry["nli"] = perm_p(y_arr[m], nli_scores[m], B_PERM, children[child_idx]) if n_pos >= 1 else None
        child_idx += 1
        if tfidf_scores is not None and n_pos >= 1:
            entry["tfidf"] = perm_p(y_arr[m], tfidf_scores[m], B_PERM, children[child_idx])
        else:
            entry["tfidf"] = None
        child_idx += 1

        strata_results[sname] = entry

    decision_cell = strata_results["rules34_ambiguous_dropped"]
    detector_reg_auroc = 0.4595
    outcome = None
    reasons = []

    if not tfidf_reproduction_ok:
        outcome = "C"
        reasons.append("TF-IDF reproduction check failed against results/accord/control/baselines.json")
    for check_name in ("rules34_ambiguous_dropped", "rules34_ambiguous_as_negative"):
        if strata_results[check_name]["n_pos"] < 5:
            outcome = "C"
            reasons.append(f"{check_name} has fewer than 5 positives ({strata_results[check_name]['n_pos']})")

    if outcome is None:
        det_cell = decision_cell["detector"]
        cond_p = det_cell["p_permutation_auroc"] >= 0.05
        cond_close = abs(det_cell["auroc"] - detector_reg_auroc) <= 0.10
        outcome = "A" if (cond_p and cond_close) else "B"

    def ordering(auroc_map):
        # descending order of system names by AUROC, None-safe
        items = [(k, v) for k, v in auroc_map.items() if v is not None]
        items.sort(key=lambda kv: -kv[1])
        return [k for k, _ in items]

    g253_order = ordering(
        {"detector": g253["detector_auroc"], "nli": g253["nli_auroc"], "tfidf": g253["tfidf_auroc"]}
    )
    ordering_comparison = {}
    for sname, entry in strata_results.items():
        auroc_map = {
            "detector": entry["detector"]["auroc"] if entry["detector"] else None,
            "nli": entry["nli"]["auroc"] if entry["nli"] else None,
            "tfidf": entry["tfidf"]["auroc"] if entry["tfidf"] else None,
        }
        this_order = ordering(auroc_map)
        ordering_comparison[sname] = {
            "order_desc_auroc": this_order,
            "matches_g253_order": this_order == g253_order,
        }

    result = {
        "seed": SEED,
        "B_perm": B_PERM,
        "label_join_mismatch_count_pharm_label_vs_label_clean": label_mismatch,
        "tfidf_reproduction_check": {
            "auroc_g299_recomputed": auroc_299,
            "ap_g299_recomputed": ap_299,
            "auroc_g299_registered": registered["auroc"],
            "ap_g299_registered": registered["ap"],
            "passed": tfidf_reproduction_ok,
        },
        "g253_reference": g253,
        "g253_order_desc_auroc": g253_order,
        "strata": strata_results,
        "ordering_comparison_vs_g253": ordering_comparison,
        "decision": {
            "outcome": outcome,
            "reasons": reasons,
            "decision_cell": "rules34_ambiguous_dropped",
            "detector_g253_auroc_reference": detector_reg_auroc,
            "detector_stratum_auroc": decision_cell["detector"]["auroc"] if decision_cell["detector"] else None,
            "detector_stratum_p_permutation_auroc": (
                decision_cell["detector"]["p_permutation_auroc"] if decision_cell["detector"] else None
            ),
        },
    }
    return result


def main():
    with LOG_PATH.open("w") as logf, contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
        print(f"dfrescore_run starting {time.strftime('%Y-%m-%d %H:%M:%S')}")
        result = run()
        print("dfrescore_run finished")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT}")
    print(f"log: {LOG_PATH}")
    print(f"tfidf reproduction check passed: {result['tfidf_reproduction_check']['passed']}")
    print(f"decision outcome: {result['decision']['outcome']}")
    if result["decision"]["reasons"]:
        print(f"reasons: {result['decision']['reasons']}")


if __name__ == "__main__":
    main()
