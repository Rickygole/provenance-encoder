"""C112: re-run C110's decision rule -- is the ranking null concentrated in the
exemplar-defect rows (Rules 1/2), or does it persist on the defect-free strata
(Rules 3/4) -- on the decontaminated three-seed ensemble detector, the
configuration the thesis actually reports, instead of C110's contaminated
accord_contra_prob column. See C112_PREREGISTRATION.md, committed before this
script was run. C110 is not edited or withdrawn by this run.

No new model runs. All detector scores read from EXISTING files:
  - ensemble / per-seed: prob_seed0, prob_seed1, prob_seed2
    (gold_eval_decontam_predictions.csv, 253 non-ambiguous gold pairs)
  - rule labels: rule column (gold_standard_accord_results.csv, 299 rows)
  - gold labels: label_clean (pharmacist_labeled_clean.csv)

Local only. Any code path that touches ~/mimic_local_only redirects
stdout/stderr to a log file under ~/mimic_local_only/logs; only the aggregate
JSON this script writes under results/accord/c112_decontam_defectfree/ and a
short numeric summary ever reach real stdout. No pair_id, sentence, or row is
printed.

    python3 accord/c112_decontam_defectfree/c112_run.py
"""
from __future__ import annotations

import contextlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accord.cloud_paths import assert_local_only  # noqa: E402

LOG_DIR = pathlib.Path.home() / "mimic_local_only" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = assert_local_only(LOG_DIR / f"c112_run_{time.strftime('%Y%m%dT%H%M%S')}.log")

OUT = ROOT / "results" / "accord" / "c112_decontam_defectfree" / "c112_defectfree.json"

DETECTOR_RULE_PATH = pathlib.Path.home() / "mimic_local_only" / "derived" / "gold_standard_accord_results.csv"
PHARM_PATH = (
    pathlib.Path.home() / "mimic_local_only" / "from_drive_2026-08-08" / "outputs" / "pharmacist_labeled_clean.csv"
)
DECONTAM_PRED_PATH = (
    pathlib.Path.home() / "mimic_local_only" / "from_drive_2026-09-13" / "outputs" / "gold_eval_decontam_predictions.csv"
)
DECONTAM_METRICS_PATH = ROOT / "results" / "accord" / "decontam" / "corrected_gold_metrics.json"

SEED = 42
B_PERM = 20000

STRATA_DEFS = {
    "rules34_ambiguous_dropped": {"rules": (3, 4), "decision_relevant": True},
    "rules12_ambiguous_dropped": {"rules": (1, 2), "decision_relevant": False},
}
SYSTEMS = ("ensemble", "seed0", "seed1", "seed2")


class AbortC(RuntimeError):
    """Raised to signal Outcome C: no stratum datum written beyond the reason."""


def perm_p(y: np.ndarray, s: np.ndarray, B: int, seed_seq) -> dict:
    """One-sided label-permutation null holding scores s fixed. Upper-tail p:
    fraction of permuted-label draws whose statistic is >= the observed one."""
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


def run() -> dict:
    det = pd.read_csv(DETECTOR_RULE_PATH)
    pharm = pd.read_csv(PHARM_PATH)
    dec = pd.read_csv(DECONTAM_PRED_PATH)

    # ---- label-agreement check 1: pharm_label vs label_clean, over the 299-row join ----
    j299 = det[["pair_id", "rule", "pharm_label"]].merge(
        pharm[["pair_id", "label_clean"]], on="pair_id", how="inner"
    )
    if j299.shape[0] != 299:
        raise AbortC(f"detector<->pharmacist join is not 299 rows (got {j299.shape[0]})")
    if j299["label_clean"].isna().any():
        raise AbortC("unjoined pair_id against pharmacist_labeled_clean.csv")
    n_mismatch_pharm_vs_clean = int((j299["pharm_label"] != j299["label_clean"]).sum())

    # ---- label-agreement check 2: decontam y_true vs label_clean-derived label, over the 253-row join ----
    j253 = j299.merge(
        dec[["pair_id", "y_true", "prob_seed0", "prob_seed1", "prob_seed2"]], on="pair_id", how="inner"
    )
    if j253.shape[0] != 253:
        raise AbortC(f"detector<->decontam-predictions join is not 253 rows (got {j253.shape[0]})")
    j253 = j253.copy()
    j253["is_pos"] = (j253["label_clean"] == "contradiction").astype(int)
    n_mismatch_ytrue_vs_clean = int((j253["y_true"] != j253["is_pos"]).sum())

    if n_mismatch_pharm_vs_clean != 0 or n_mismatch_ytrue_vs_clean != 0:
        raise AbortC(
            "label-agreement check failed: "
            f"pharm_label_vs_label_clean_mismatches={n_mismatch_pharm_vs_clean} (n=299), "
            f"y_true_vs_label_clean_mismatches={n_mismatch_ytrue_vs_clean} (n=253)"
        )

    # ---- reproduction check: per-seed and ensemble AUROC on all 253 pairs vs corrected_gold_metrics.json ----
    y_arr = j253["is_pos"].to_numpy()
    seed_scores = {s: j253[f"prob_seed{s}"].to_numpy() for s in range(3)}
    ensemble_scores = np.mean([seed_scores[0], seed_scores[1], seed_scores[2]], axis=0)

    reg = json.loads(DECONTAM_METRICS_PATH.read_text())
    recomputed_seed_auroc = {s: round(float(roc_auc_score(y_arr, seed_scores[s])), 4) for s in range(3)}
    registered_seed_auroc = {s: round(float(reg["per_seed"][f"seed{s}"]["auroc"]), 4) for s in range(3)}
    recomputed_ens_auroc = round(float(roc_auc_score(y_arr, ensemble_scores)), 4)
    registered_ens_auroc = round(float(reg["aggregate"]["ensemble_auroc"]), 4)

    seed_checks_ok = all(recomputed_seed_auroc[s] == registered_seed_auroc[s] for s in range(3))
    ens_check_ok = recomputed_ens_auroc == registered_ens_auroc
    reproduction_ok = seed_checks_ok and ens_check_ok

    reproduction_check = {
        "recomputed_seed_auroc": recomputed_seed_auroc,
        "registered_seed_auroc": registered_seed_auroc,
        "recomputed_ensemble_auroc_g253": recomputed_ens_auroc,
        "registered_ensemble_auroc_g253": registered_ens_auroc,
        "recomputed_ensemble_auroc_g253_full_precision": float(roc_auc_score(y_arr, ensemble_scores)),
        "passed": reproduction_ok,
    }

    if not reproduction_ok:
        raise AbortC(
            "reproduction check failed against results/accord/decontam/corrected_gold_metrics.json: "
            f"seed_match={seed_checks_ok}, ensemble_match={ens_check_ok}"
        )

    g253_ensemble_auroc_reference = reproduction_check["recomputed_ensemble_auroc_g253_full_precision"]

    # ---- strata ----
    rule_arr = j253["rule"].to_numpy()

    strata_results = {}
    n_pos_decision_cell = None
    for sname, sdef in STRATA_DEFS.items():
        m = np.isin(rule_arr, sdef["rules"])
        n = int(m.sum())
        n_pos = int(y_arr[m].sum())
        strata_results[sname] = {
            "n": n,
            "n_pos": n_pos,
            "rules": list(sdef["rules"]),
            "decision_relevant": sdef["decision_relevant"],
            "systems": {},
        }
        if sdef["decision_relevant"]:
            n_pos_decision_cell = n_pos

    if n_pos_decision_cell is not None and n_pos_decision_cell < 5:
        raise AbortC(
            f"Rules 3+4, ambiguous-dropped stratum has fewer than 5 positives ({n_pos_decision_cell})"
        )

    scores_by_system = {
        "ensemble": ensemble_scores,
        "seed0": seed_scores[0],
        "seed1": seed_scores[1],
        "seed2": seed_scores[2],
    }

    ss = np.random.SeedSequence(SEED)
    n_children = len(STRATA_DEFS) * len(SYSTEMS)
    children = ss.spawn(n_children)
    child_idx = 0
    for sname, sdef in STRATA_DEFS.items():
        m = np.isin(rule_arr, sdef["rules"])
        for system in SYSTEMS:
            strata_results[sname]["systems"][system] = perm_p(
                y_arr[m], scores_by_system[system][m], B_PERM, children[child_idx]
            )
            child_idx += 1

    # ---- decision rule ----
    decision_cell = strata_results["rules34_ambiguous_dropped"]
    ens_cell = decision_cell["systems"]["ensemble"]
    cond_p = ens_cell["p_permutation_auroc"] >= 0.05
    gap = abs(ens_cell["auroc"] - g253_ensemble_auroc_reference)
    cond_close = gap <= 0.10
    outcome = "A" if (cond_p and cond_close) else "B"

    result = {
        "seed": SEED,
        "B_perm": B_PERM,
        "label_agreement_check": {
            "pharm_label_vs_label_clean_mismatches_n299": n_mismatch_pharm_vs_clean,
            "y_true_vs_label_clean_mismatches_n253": n_mismatch_ytrue_vs_clean,
        },
        "reproduction_check": reproduction_check,
        "g253_ensemble_auroc_reference": g253_ensemble_auroc_reference,
        "g253_ensemble_auroc_reference_source": "recomputed on this run's own 253-pair join, "
        "matches results/accord/decontam/corrected_gold_metrics.json aggregate.ensemble_auroc",
        "contaminated_g253_reference_for_context_only": {
            "value": 0.4595,
            "source": "results/accord/VERIFIED_FACTS.json::"
            "signal.contra_prob_vs_pharm_label__ambiguous_excluded_n253.auroc (C110's reference, unchanged)",
        },
        "strata": strata_results,
        "decision": {
            "outcome": outcome,
            "decision_cell": "rules34_ambiguous_dropped",
            "ensemble_stratum_auroc": ens_cell["auroc"],
            "ensemble_stratum_p_permutation_auroc": ens_cell["p_permutation_auroc"],
            "ensemble_g253_auroc_reference": g253_ensemble_auroc_reference,
            "auroc_gap_vs_g253": gap,
            "cond_p_permutation_ge_0.05": cond_p,
            "cond_within_0.10_of_g253": cond_close,
        },
    }
    return result


def main():
    with LOG_PATH.open("w") as logf, contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
        print(f"c112_run starting {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            result = run()
        except AbortC as e:
            print(f"ABORT (Outcome C): {e}")
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps({"outcome": "C", "abort_reason": str(e)}, indent=2))
            print(f"wrote {OUT} (Outcome C, no data)")
            raise SystemExit(f"ABORT (Outcome C): {e}")
        print("c112_run finished")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT}")
    print(f"log: {LOG_PATH}")
    print(f"reproduction check passed: {result['reproduction_check']['passed']}")
    print(f"label agreement: {result['label_agreement_check']}")
    d = result["decision"]
    print(f"decision outcome: {d['outcome']}")
    print(
        f"decision cell auroc={d['ensemble_stratum_auroc']:.4f} "
        f"p_perm={d['ensemble_stratum_p_permutation_auroc']:.4f} "
        f"g253_ref={d['ensemble_g253_auroc_reference']:.4f} gap={d['auroc_gap_vs_g253']:.4f}"
    )


if __name__ == "__main__":
    main()
