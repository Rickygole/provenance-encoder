"""C134d: threshold sweep on the DECONTAMINATED ensemble.

accord/stats/reviewer_free_computations.py answers "a different cutoff would
rescue it" for the released, subject-overlap run only, and says so in its own
docstring: "Computed on the released score file, the same file behind the AUROC
finding, not on the three-seed run behind the operating table." So the objection
is unanswered for the decontaminated ensemble that Table tab:operating reports
at the configured threshold of 0.5 (kappa below zero in every seed). This script
answers it, for the ensemble and for each of the three seeds.

Method is byte-identical in shape to the released-run computation it must be
comparable with:
  - sweep every distinct score as a cutoff with the rule s >= t, keep the max MCC,
  - report the MCC at the configured threshold 0.5 alongside it,
  - score the maximised statistic against
    accord/stats/reviewer_free_computations.py::oracle_threshold_null, the same
    max-over-threshold null on uniform random scores at n = 253, 12 positives,
    B = 4000, seed = 1, which is what produced the printed p = 0.8195 for the
    released run's 0.0554. Recomputed here, not copied, and the printed 0.8195
    and 0.0554 are reproduced as a check that the two runs are comparable.
  - a second, tighter null is added and labelled: a label-permutation null that
    holds the ensemble's own score vector fixed and permutes the 12 positives,
    B = 10000, so the null carries the real score distribution's ties and shape.

Local only. Reads credentialed per-pair scores from
~/mimic_local_only/from_drive_2026-09-13/outputs/gold_eval_decontam_predictions.csv
(pair_id, y_true, prob_seed0..2 only; no note text, no HADM_ID, no SUBJECT_ID).
stdout/stderr while that frame is in scope go to a log under ~/mimic_local_only/logs.
The JSON written under results/ holds counts and statistics only.

    python3 accord/c134_reanalysis/c134d_decontam_threshold_sweep.py
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accord.cloud_paths import assert_local_only  # noqa: E402
from accord.stats.reviewer_free_computations import cells, mcc  # noqa: E402

MIMIC_LOCAL_ONLY_DIR = pathlib.Path(os.environ.get(
    "MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
PRED = MIMIC_LOCAL_ONLY_DIR / "from_drive_2026-09-13" / "outputs" / "gold_eval_decontam_predictions.csv"
LOG_DIR = MIMIC_LOCAL_ONLY_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG = assert_local_only(LOG_DIR / f"c134d_{time.strftime('%Y%m%dT%H%M%S')}.log")
OUT = ROOT / "results" / "accord" / "c134_reanalysis" / "c134d_decontam_threshold_sweep.json"
RELEASED = ROOT / "results" / "accord" / "stats" / "reviewer_free_computations.json"

N, NPOS = 253, 12
B_ORACLE, ORACLE_SEED = 4000, 1
B_PERM, PERM_SEED = 10000, 134


def sweep(y: np.ndarray, s: np.ndarray) -> dict:
    """Max MCC over cutoffs, rule s >= t, exactly as the released-run sweep does."""
    best = (-2.0, None)
    for t in sorted(set(s.tolist())):
        tp, tn, fp, fn = cells(y, s >= t)
        m = mcc(tp, tn, fp, fn)
        if m > best[0]:
            best = (m, (float(t), tp, tn, fp, fn))
    t, tp, tn, fp, fn = best[1]
    ctp, ctn, cfp, cfn = cells(y, s >= 0.5)
    return {
        "threshold_optimal": {"mcc": round(best[0], 4), "mcc_full_precision": best[0],
                              "threshold": round(t, 6), "tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "at_configured_0.5": {"mcc": round(mcc(ctp, ctn, cfp, cfn), 4),
                              "tp": ctp, "tn": ctn, "fp": cfp, "fn": cfn},
        "n_distinct_cutoffs_swept": len(set(s.tolist())),
    }


def max_mcc_over_thresholds_vec(y_sorted_desc: np.ndarray, npos: int, n: int) -> float:
    ctp = np.cumsum(y_sorted_desc)
    k = np.arange(1, n + 1)
    cfp = k - ctp
    cfn = npos - ctp
    ctn = (n - npos) - cfp
    den = np.sqrt((ctp + cfp) * (ctp + cfn) * (ctn + cfp) * (ctn + cfn), dtype=float)
    val = np.where(den == 0, 0.0, (ctp * ctn - cfp * cfn) / np.where(den == 0, 1, den))
    return float(val.max())


def oracle_null_random_scores(B=B_ORACLE, seed=ORACLE_SEED) -> np.ndarray:
    """Verbatim reimplementation of reviewer_free_computations.py's null:
    uniform random scores, permuted labels, max MCC over all cutoffs."""
    rng = np.random.default_rng(seed)
    y0 = np.zeros(N, bool)
    y0[:NPOS] = True
    best = []
    for _ in range(B):
        s_ = rng.random(N)
        y = y0[rng.permutation(N)]
        best.append(max_mcc_over_thresholds_vec(y[np.argsort(-s_)], NPOS, N))
    return np.array(best)


def label_permutation_null(s: np.ndarray, B=B_PERM, seed=PERM_SEED) -> np.ndarray:
    """Tighter null: this score vector held fixed, the 12 positives permuted."""
    rng = np.random.default_rng(seed)
    order = np.argsort(-s)
    y0 = np.zeros(N, bool)
    y0[:NPOS] = True
    out = np.empty(B)
    for b in range(B):
        out[b] = max_mcc_over_thresholds_vec(y0[rng.permutation(N)][order], NPOS, N)
    return out


def main():
    t0 = time.time()
    released = json.loads(RELEASED.read_text())
    with LOG.open("w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
        d = pd.read_csv(PRED)
        assert len(d) == N and int(d.y_true.sum()) == NPOS, "unexpected geometry"
        y = d.y_true.to_numpy().astype(bool)
        scores = {f"seed{i}": d[f"prob_seed{i}"].to_numpy(dtype=float) for i in range(3)}
        scores["ensemble_mean_of_three_seeds"] = np.mean(
            [scores[f"seed{i}"] for i in range(3)], axis=0)
        swept = {k: sweep(y, v) for k, v in scores.items()}
        perm = {k: label_permutation_null(v) for k, v in scores.items()}
        point = {k: swept[k]["threshold_optimal"]["mcc"] for k in scores}
        perm_p = {k: float((perm[k] >= point[k]).mean()) for k in scores}
        perm_desc = {k: {"median": float(np.median(perm[k])),
                         "p05": float(np.percentile(perm[k], 5)),
                         "p95": float(np.percentile(perm[k], 95))} for k in scores}

    onull = oracle_null_random_scores()
    oracle_p = {k: float("%.4g" % float((onull >= point[k]).mean())) for k in scores}

    # comparability check: reproduce the printed released-run figures with this code
    rel = released["detector_threshold_optimal_mcc"]["G_253"]["accord_contra_prob"]
    rel_mcc = rel["threshold_optimal"]["mcc"]
    rel_p = released["oracle_threshold_null"]["p_null_ge"]["accord_contra_prob"]
    check = {
        "released_run_printed_mcc": rel_mcc,
        "released_run_printed_p": rel_p,
        "released_run_p_recomputed_against_this_null": float("%.4g" % float((onull >= rel_mcc).mean())),
        "oracle_null_median_recomputed": round(float(np.median(onull)), 4),
        "oracle_null_median_printed": released["oracle_threshold_null"]["null_median"],
    }
    check["p_reproduces"] = abs(check["released_run_p_recomputed_against_this_null"] - rel_p) < 5e-4
    check["null_median_reproduces"] = abs(
        check["oracle_null_median_recomputed"] - check["oracle_null_median_printed"]) < 5e-5

    out = {
        "what": "threshold sweep and max-over-threshold MCC null for the decontaminated "
                "three-seed detector on G_253, the run Table tab:operating reports",
        "closes": "accord/stats/reviewer_free_computations.py docstring: the threshold-rescue "
                  "objection was answered for the released subject-overlap run only, "
                  "'not on the three-seed run behind the operating table'",
        "geometry": {"n": N, "n_positives": NPOS, "n_negatives": N - NPOS,
                     "source": "gold_eval_decontam_predictions.csv (local only)"},
        "nulls": {
            "oracle_random_scores": {"B": B_ORACLE, "seed": ORACLE_SEED,
                                     "description": "uniform random scores, permuted labels, "
                                                    "max MCC over cutoffs; identical to the null "
                                                    "behind the printed p = 0.8195",
                                     "median": round(float(np.median(onull)), 4),
                                     "p05": round(float(np.percentile(onull, 5)), 4),
                                     "p95": round(float(np.percentile(onull, 95)), 4)},
            "label_permutation_holding_real_scores": {"B": B_PERM, "seed": PERM_SEED},
        },
        "comparability_check_against_released_run": check,
        "per_configuration": {
            k: {**swept[k],
                "p_vs_oracle_random_score_null": oracle_p[k],
                "p_vs_label_permutation_null": perm_p[k],
                "label_permutation_null": perm_desc[k]}
            for k in scores},
        "headline": {
            "best_mcc_attainable_at_any_cutoff_ensemble":
                swept["ensemble_mean_of_three_seeds"]["threshold_optimal"]["mcc"],
            "mcc_at_configured_threshold_ensemble":
                swept["ensemble_mean_of_three_seeds"]["at_configured_0.5"]["mcc"],
            "p_comparable_to_printed_0_8195":
                oracle_p["ensemble_mean_of_three_seeds"],
            "released_run_for_reference": {"mcc": rel_mcc, "p": rel_p},
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    for k in scores:
        print(f"{k:32s} best MCC {swept[k]['threshold_optimal']['mcc']:+.4f} at t="
              f"{swept[k]['threshold_optimal']['threshold']:.6f}  "
              f"MCC@0.5 {swept[k]['at_configured_0.5']['mcc']:+.4f}  "
              f"p_oracle {oracle_p[k]:.4f}  p_perm {perm_p[k]:.4f}")
    print("comparability:", json.dumps(check))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
