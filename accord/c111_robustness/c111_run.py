"""C111: detector- and NLI- label-permutation tests for average precision, a
paired permutation comparison of TF-IDF against the detector ensemble and
against NLI, leave-one-positive-out fragility of TF-IDF's AP, and
training-set disjointness. See C111_PREREGISTRATION.md. No new model is
trained or run; every score used here already exists on disk except the
TF-IDF refit, which reproduces `tfidfperm_run.py`'s task (a) exactly.

Local only. Reads credentialed gold/weak-label/detector/NLI data. All
stdout/stderr produced while that data is in scope is redirected to a file
under ~/mimic_local_only/logs; only the aggregate JSON this script writes
under results/ and a short numeric summary ever reach real stdout. No
pair_id, HADM_ID, SUBJECT_ID, or sentence text is ever printed or written to
the output JSON -- only counts, scores already reduced to floats, and the
integer drop-index of the leave-one-positive-out loop.

    python3 accord/c111_robustness/c111_run.py
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

LOG_DIR = pathlib.Path.home() / "mimic_local_only" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"c111_run_{time.strftime('%Y%m%dT%H%M%S')}.log"

OUT = ROOT / "results" / "accord" / "c111_robustness" / "c111_robustness.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
DECONTAM_METRICS = ROOT / "results" / "accord" / "decontam" / "corrected_gold_metrics.json"

NLI_PATH = pathlib.Path.home() / "mimic_local_only" / "derived" / "nli_scores_roberta_large_mnli.csv"
DECONTAM_PRED_PATH = (pathlib.Path.home() / "mimic_local_only" / "from_drive_2026-09-13"
                       / "outputs" / "gold_eval_decontam_predictions.csv")
CONTAMINATED_DETECTOR_PATH = pathlib.Path.home() / "mimic_local_only" / "derived" / "gold_standard_accord_results.csv"

B_PERM = 20000
SEED = 42
N_CHILDREN = 35


class AbortC(RuntimeError):
    """Raised to signal Outcome C: invalid, no datum written beyond the abort reason."""


def perm_null_ap(y: np.ndarray, s: np.ndarray, B: int, seed_child) -> dict:
    """One-sided label-permutation null holding scores s fixed. Reports AP
    (decision-relevant) and AUROC (descriptive only, no permutation test)."""
    rng = np.random.default_rng(seed_child)
    n = len(y)
    obs_ap = float(average_precision_score(y, s))
    obs_auroc = float(roc_auc_score(y, s))
    null_ap = np.empty(B)
    idx = np.arange(n)
    for b in range(B):
        yp = y[rng.permutation(idx)]
        null_ap[b] = average_precision_score(yp, s)
    return {
        "n": int(n),
        "n_pos": int(y.sum()),
        "observed_ap": obs_ap,
        "observed_auroc_descriptive_only": obs_auroc,
        "null_ap_mean": float(null_ap.mean()),
        "p_permutation_ap_one_sided": float((null_ap >= obs_ap).mean()),
        "B": B,
    }


def paired_swap_perm_ap(y: np.ndarray, s1: np.ndarray, s2: np.ndarray, B: int, seed_child) -> dict:
    """Two-sided paired-permutation test on AP difference: per pair, swap
    which system's score occupies 'slot 1' vs 'slot 2' with probability 0.5,
    independently per pair, holding y fixed. Null statistic is
    AP(slot1) - AP(slot2) under the swap; observed is AP(s1) - AP(s2)
    unswapped."""
    rng = np.random.default_rng(seed_child)
    n = len(y)
    obs_ap1 = float(average_precision_score(y, s1))
    obs_ap2 = float(average_precision_score(y, s2))
    obs_diff = obs_ap1 - obs_ap2
    null_diff = np.empty(B)
    for b in range(B):
        swap = rng.integers(0, 2, size=n).astype(bool)
        slot1 = np.where(swap, s2, s1)
        slot2 = np.where(swap, s1, s2)
        null_diff[b] = average_precision_score(y, slot1) - average_precision_score(y, slot2)
    return {
        "n": int(n),
        "n_pos": int(y.sum()),
        "observed_ap_system1": obs_ap1,
        "observed_ap_system2": obs_ap2,
        "observed_diff": obs_diff,
        "null_diff_mean": float(null_diff.mean()),
        "p_permutation_two_sided": float((np.abs(null_diff) >= abs(obs_diff)).mean()),
        "B": B,
    }


def lopo_drops(y_full: np.ndarray, s_full: np.ndarray, B: int, seed_children) -> dict:
    """Drop each of the positives in y_full (in the order they appear) one at
    a time, recompute AP and its permutation p over the remaining rows using
    the SAME fixed scores (no refit). seed_children: list of 12 independent
    SeedSequence children, one per drop."""
    pos_idx = np.flatnonzero(y_full == 1)
    n_pos = len(pos_idx)
    assert n_pos == 12, f"expected 12 positives, got {n_pos}"
    assert len(seed_children) == n_pos
    drops = []
    for i, drop_row in enumerate(pos_idx):
        keep = np.ones(len(y_full), dtype=bool)
        keep[drop_row] = False
        y_sub, s_sub = y_full[keep], s_full[keep]
        r = perm_null_ap(y_sub, s_sub, B, seed_children[i])
        r["drop_index"] = int(i)
        drops.append(r)
    aps = [d["observed_ap"] for d in drops]
    ps = [d["p_permutation_ap_one_sided"] for d in drops]
    return {
        "drops": drops,
        "ap_min": float(min(aps)),
        "ap_max": float(max(aps)),
        "p_min": float(min(ps)),
        "p_max": float(max(ps)),
        "n_drops_p_below_0.05": int(sum(p < 0.05 for p in ps)),
        "n_drops_total": n_pos,
    }


def run() -> dict:
    from data import build_splits  # noqa: E402 (imported inside redirected scope)

    # ---- load everything ----
    splits = build_splits()
    gold = splits.gold.copy()
    gold["y"] = (gold["pharm_label"] == "contradiction").astype(int)
    gold["g253"] = gold["pharm_label"] != "ambiguous"

    nli = pd.read_csv(NLI_PATH)
    dec = pd.read_csv(DECONTAM_PRED_PATH)
    cont = pd.read_csv(CONTAMINATED_DETECTOR_PATH)

    # ---- validity gate: label agreement across sources (counts only) ----
    m_nli = gold[["pair_id", "pharm_label", "y"]].merge(
        nli[["pair_id", "label_clean"]], on="pair_id", how="inner")
    n_label_mismatch_nli = int(((m_nli["pharm_label"] == "contradiction").astype(int)
                                 != (m_nli["label_clean"] == "contradiction").astype(int)).sum())
    m_dec = gold[["pair_id", "y"]].merge(dec[["pair_id", "y_true"]], on="pair_id", how="inner")
    n_label_mismatch_dec = int((m_dec["y"] != m_dec["y_true"]).sum())
    if n_label_mismatch_nli != 0 or n_label_mismatch_dec != 0 or len(m_nli) != 299 or len(m_dec) != 253:
        raise AbortC(
            f"label agreement gate failed: nli_mismatch={n_label_mismatch_nli} "
            f"(n={len(m_nli)}), dec_mismatch={n_label_mismatch_dec} (n={len(m_dec)})"
        )

    # ---- validity gate: exactly 253 common pair_ids across all three G_253 sources ----
    set_gold253 = set(gold.loc[gold.g253, "pair_id"])
    set_nli = set(nli["pair_id"])
    set_dec = set(dec["pair_id"])
    common253 = set_gold253 & set_nli & set_dec
    if len(set_gold253) != 253 or len(set_dec) != 253 or len(common253) != 253:
        raise AbortC(
            f"G_253 alignment gate failed: gold253={len(set_gold253)}, "
            f"dec={len(set_dec)}, common={len(common253)}"
        )

    # ---- reproduction gate: detector per-seed/ensemble vs corrected_gold_metrics.json ----
    dec_y = dec["y_true"].to_numpy()
    dec_seed_scores = {s: dec[f"prob_seed{s}"].to_numpy() for s in range(3)}
    dec_ensemble = np.mean([dec_seed_scores[s] for s in range(3)], axis=0)
    reg_metrics = json.loads(DECONTAM_METRICS.read_text())
    for s in range(3):
        recomputed_auroc = round(float(roc_auc_score(dec_y, dec_seed_scores[s])), 4)
        registered_auroc = round(reg_metrics["per_seed"][f"seed{s}"]["auroc"], 4)
        if recomputed_auroc != registered_auroc:
            raise AbortC(
                f"detector seed{s} AUROC reproduction failed: recomputed={recomputed_auroc}, "
                f"registered={registered_auroc}"
            )
    recomputed_ens_auroc = round(float(roc_auc_score(dec_y, dec_ensemble)), 4)
    registered_ens_auroc = round(reg_metrics["aggregate"]["ensemble_auroc"], 4)
    if recomputed_ens_auroc != registered_ens_auroc:
        raise AbortC(
            f"detector ensemble AUROC reproduction failed: recomputed={recomputed_ens_auroc}, "
            f"registered={registered_ens_auroc}"
        )

    # ---- TF-IDF refit, exactly as tfidfperm_run.py task (a) step 1 ----
    y_train = splits.train["y"].to_numpy()
    train_text = (splits.train["nursing_sentence"].astype(str) + " [SEP] "
                  + splits.train["discharge_sentence"].astype(str)).tolist()
    gold_text = (gold["nursing_sentence"].astype(str) + " [SEP] "
                 + gold["discharge_sentence"].astype(str)).tolist()

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train = vec.fit_transform(train_text)
    X_gold = vec.transform(gold_text)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    tfidf_scores = clf.predict_proba(X_gold)[:, 1]

    facts = json.loads(FACTS.read_text())
    reg_ap_299 = facts["tfidfperm.task_a.g299.ap"]["value"]
    reg_ap_253 = facts["tfidfperm.task_a.g253.ap"]["value"]
    y_all = gold["y"].to_numpy()
    g253_mask = gold["g253"].to_numpy()
    recomputed_ap_299 = float(average_precision_score(y_all, tfidf_scores))
    recomputed_ap_253 = float(average_precision_score(y_all[g253_mask], tfidf_scores[g253_mask]))
    if round(recomputed_ap_299, 4) != round(reg_ap_299, 4) or round(recomputed_ap_253, 4) != round(reg_ap_253, 4):
        raise AbortC(
            f"TF-IDF reproduction failed: recomputed g299_ap={recomputed_ap_299:.6f} "
            f"(registered {reg_ap_299:.6f}), recomputed g253_ap={recomputed_ap_253:.6f} "
            f"(registered {reg_ap_253:.6f})"
        )

    # ---- build the master G_253-aligned frame (pair_id used only as a join key, never emitted) ----
    master = gold[["pair_id", "y", "g253"]].copy()
    master["tfidf_score"] = tfidf_scores
    master = master.merge(nli[["pair_id", "contra_prob_mean"]], on="pair_id", how="left")
    master = master.merge(
        pd.DataFrame({"pair_id": dec["pair_id"], "det_seed0": dec_seed_scores[0],
                       "det_seed1": dec_seed_scores[1], "det_seed2": dec_seed_scores[2],
                       "det_ensemble": dec_ensemble}),
        on="pair_id", how="left",
    )
    if master["contra_prob_mean"].isna().any():
        raise AbortC("NLI score missing for a gold pair after join")
    g253 = master[master.g253].reset_index(drop=True)
    if len(g253) != 253 or g253["det_ensemble"].isna().any():
        raise AbortC(f"G_253 frame malformed after join: n={len(g253)}, "
                      f"det_na={int(g253['det_ensemble'].isna().sum())}")

    # ---- seed stream: 35 independent children, fixed order per pre-registration ----
    ss = np.random.SeedSequence(SEED)
    children = ss.spawn(N_CHILDREN)
    c_det_seed0, c_det_seed1, c_det_seed2, c_det_ens, c_det_cont_g299 = children[0:5]
    c_nli_253, c_nli_299 = children[5:7]
    c_tfidf_253, c_tfidf_299 = children[7:9]
    c_pair_tfidf_det, c_pair_tfidf_nli = children[9:11]
    c_lopo_253 = children[11:23]
    c_lopo_299 = children[23:35]

    y253 = g253["y"].to_numpy()

    # (a) detector, per seed and ensemble, G_253
    det_a = {
        "seed0": perm_null_ap(y253, g253["det_seed0"].to_numpy(), B_PERM, c_det_seed0),
        "seed1": perm_null_ap(y253, g253["det_seed1"].to_numpy(), B_PERM, c_det_seed1),
        "seed2": perm_null_ap(y253, g253["det_seed2"].to_numpy(), B_PERM, c_det_seed2),
        "ensemble": perm_null_ap(y253, g253["det_ensemble"].to_numpy(), B_PERM, c_det_ens),
        "g299_note": ("not computable: no per-pair score from the decontaminated retrain "
                       "was retained for the 46 ambiguous pairs (msu_ch4_results.tex, "
                       "paragraph around line 220-221)"),
    }
    # descriptive-only supplementary: contaminated single-run column, G_299
    cont_y299 = (cont["pharm_label"] == "contradiction").astype(int).to_numpy()
    cont_s299 = cont["accord_contra_prob"].to_numpy()
    det_a["contaminated_accord_contra_prob_g299_descriptive_only"] = perm_null_ap(
        cont_y299, cont_s299, B_PERM, c_det_cont_g299)

    # (a) NLI, G_253 and G_299
    y299 = master["y"].to_numpy()
    s299_nli = master["contra_prob_mean"].to_numpy()
    nli_a = {
        "g253": perm_null_ap(y253, g253["contra_prob_mean"].to_numpy(), B_PERM, c_nli_253),
        "g299": perm_null_ap(y299, s299_nli, B_PERM, c_nli_299),
    }

    # (a) TF-IDF, G_253 and G_299
    s299_tfidf = master["tfidf_score"].to_numpy()
    tfidf_a = {
        "g253": perm_null_ap(y253, g253["tfidf_score"].to_numpy(), B_PERM, c_tfidf_253),
        "g299": perm_null_ap(y299, s299_tfidf, B_PERM, c_tfidf_299),
    }

    # (b) paired comparisons, G_253
    paired_b = {
        "tfidf_minus_detector_ensemble": paired_swap_perm_ap(
            y253, g253["tfidf_score"].to_numpy(), g253["det_ensemble"].to_numpy(),
            B_PERM, c_pair_tfidf_det),
        "tfidf_minus_nli": paired_swap_perm_ap(
            y253, g253["tfidf_score"].to_numpy(), g253["contra_prob_mean"].to_numpy(),
            B_PERM, c_pair_tfidf_nli),
    }

    # (c) leave-one-positive-out, TF-IDF, G_253 and G_299
    lopo_c = {
        "g253": lopo_drops(y253, g253["tfidf_score"].to_numpy(), B_PERM, c_lopo_253),
        "g299": lopo_drops(y299, s299_tfidf, B_PERM, c_lopo_299),
    }

    # (d) training-set disjointness
    key = ["HADM_ID", "drug", "rule", "nursing_sentence", "discharge_sentence"]
    train_keys = set(map(tuple, splits.train[key].astype(str).values.tolist()))
    gold_keys = set(map(tuple, splits.gold[key].astype(str).values.tolist()))
    n_pair_overlap = len(train_keys & gold_keys)
    n_subject_overlap = len(set(splits.train["SUBJECT_ID"].unique())
                             & set(splits.gold["SUBJECT_ID"].unique()))
    disjointness_d = {
        "n_train_rows": int(len(splits.train)),
        "n_gold_rows": int(len(splits.gold)),
        "n_pair_overlap": int(n_pair_overlap),
        "n_subject_overlap": int(n_subject_overlap),
    }

    # ---- decision rule ----
    tfidf_g253_significant = tfidf_a["g253"]["p_permutation_ap_one_sided"] < 0.05
    lopo_survives = lopo_c["g253"]["n_drops_p_below_0.05"] >= 10
    detector_ensemble_not_significant = det_a["ensemble"]["p_permutation_ap_one_sided"] >= 0.05
    disjoint = disjointness_d["n_pair_overlap"] == 0 and disjointness_d["n_subject_overlap"] == 0

    failed = []
    if not tfidf_g253_significant:
        failed.append("tfidf_g253_ap_permutation_p_not_below_0.05")
    if not lopo_survives:
        failed.append("lopo_g253_fewer_than_10_of_12_drops_below_0.05")
    if not detector_ensemble_not_significant:
        failed.append("detector_ensemble_g253_ap_permutation_p_below_0.05")
    if not disjoint:
        failed.append("training_set_not_disjoint_from_gold")

    outcome_letter = "A" if not failed else "B"

    nli_g253_significant = nli_a["g253"]["p_permutation_ap_one_sided"] < 0.05
    nli_g299_significant = nli_a["g299"]["p_permutation_ap_one_sided"] < 0.05
    if nli_g253_significant:
        nli_interpretation = ("NLI's AP permutation p is significant on G_253 (the shared stratum): "
                               "per the registered interpretation, whatever the outcome letter, the "
                               "signal cannot be attributed to the weak labels specifically and must "
                               "be described as a surface lexical signal that a label-free model also finds.")
    else:
        nli_interpretation = ("NLI's AP permutation p is NOT significant on G_253: the registered "
                               "interpretation's condition does not trigger; the outcome letter stands "
                               "on its own terms.")
    if nli_g299_significant != nli_g253_significant:
        nli_interpretation += (f" Note: G_299 disagrees with G_253 (G_299 significant={nli_g299_significant}, "
                                f"G_253 significant={nli_g253_significant}); both are reported, not reconciled.")

    result = {
        "seed": SEED,
        "B_perm": B_PERM,
        "n_children_spawned": N_CHILDREN,
        "reproduction_checks": {
            "tfidf_g299_ap": {"recomputed": recomputed_ap_299, "registered": reg_ap_299},
            "tfidf_g253_ap": {"recomputed": recomputed_ap_253, "registered": reg_ap_253},
            "detector_per_seed_auroc": {f"seed{s}": round(float(roc_auc_score(dec_y, dec_seed_scores[s])), 4)
                                         for s in range(3)},
            "detector_ensemble_auroc": recomputed_ens_auroc,
            "label_agreement": {"nli_mismatches": n_label_mismatch_nli, "dec_mismatches": n_label_mismatch_dec},
            "g253_common_pair_ids": len(common253),
        },
        "task_a": {"detector": det_a, "nli": nli_a, "tfidf": tfidf_a},
        "task_b_paired": paired_b,
        "task_c_lopo": lopo_c,
        "task_d_disjointness": disjointness_d,
        "decision": {
            "tfidf_g253_ap_permutation_p": tfidf_a["g253"]["p_permutation_ap_one_sided"],
            "tfidf_g253_significant": tfidf_g253_significant,
            "lopo_g253_n_drops_below_0.05": lopo_c["g253"]["n_drops_p_below_0.05"],
            "lopo_survives_at_least_10_of_12": lopo_survives,
            "detector_ensemble_g253_ap_permutation_p": det_a["ensemble"]["p_permutation_ap_one_sided"],
            "detector_ensemble_not_significant": detector_ensemble_not_significant,
            "training_set_disjoint": disjoint,
            "failed_conditions": failed,
            "outcome": outcome_letter,
            "nli_g253_significant": nli_g253_significant,
            "nli_g299_significant": nli_g299_significant,
            "nli_interpretation": nli_interpretation,
        },
    }
    return result


def main():
    with LOG_PATH.open("w") as logf, contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
        print(f"c111_run starting {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            result = run()
        except AbortC as e:
            print(f"ABORT (Outcome C): {e}")
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps({"outcome": "C", "abort_reason": str(e)}, indent=2))
            print(f"wrote {OUT} (Outcome C, no data)")
            raise SystemExit(f"ABORT (Outcome C): {e}")
        print("c111_run finished")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT}")
    print(f"log: {LOG_PATH}")
    d = result["decision"]
    print(f"outcome: {d['outcome']}  failed_conditions={d['failed_conditions']}")
    print(f"tfidf_g253_ap_p={d['tfidf_g253_ap_permutation_p']:.4f}  "
          f"lopo_g253_below_0.05={d['lopo_g253_n_drops_below_0.05']}/12  "
          f"detector_ensemble_g253_ap_p={d['detector_ensemble_g253_ap_permutation_p']:.4f}  "
          f"training_disjoint={d['training_set_disjoint']}")
    print(f"nli_g253_significant={d['nli_g253_significant']}  nli_g299_significant={d['nli_g299_significant']}")


if __name__ == "__main__":
    main()
