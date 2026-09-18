"""C114: paired AP test NLI vs decontaminated detector ensemble (G_253), gold
tipping point from the decontaminated ensemble, and a bootstrap interval on the
Rules 3+4 ambiguous-dropped stratum. See C114_PREREGISTRATION.md, committed
before this script was run.

Local only. All stdout/stderr while credentialed data is in scope goes to a log
under ~/mimic_local_only/logs. The output JSON holds counts and aggregate floats
only; no pair_id, row, or sentence is printed or written.

    python3 accord/c114_committee_checks/c114_run.py
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
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accord" / "c111_robustness"))
sys.path.insert(0, str(ROOT / "accord" / "stats"))
sys.path.insert(0, str(ROOT / "accord" / "control"))

from accord.cloud_paths import assert_local_only  # noqa: E402

MIMIC_LOCAL_ONLY_DIR = pathlib.Path(os.environ.get("MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
LOG_DIR = MIMIC_LOCAL_ONLY_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = assert_local_only(LOG_DIR / f"c114_run_{time.strftime('%Y%m%dT%H%M%S')}.log")

OUT = ROOT / "results" / "accord" / "c114_committee_checks" / "c114.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"

DECONTAM_PRED_PATH = (MIMIC_LOCAL_ONLY_DIR / "from_drive_2026-09-13"
                      / "outputs" / "gold_eval_decontam_predictions.csv")
NLI_PATH = MIMIC_LOCAL_ONLY_DIR / "derived" / "nli_scores_roberta_large_mnli.csv"
RULE_PATH = MIMIC_LOCAL_ONLY_DIR / "derived" / "gold_standard_accord_results.csv"

B_PERM = 20000
SEED = 42
TARGETS = {"mde_0_7355": 0.7355, "c108_mannwhitney_0_7466": 0.7466, "c108_c98_rule_0_8084": 0.8084}


class AbortC(RuntimeError):
    pass


def run() -> dict:
    from c111_run import paired_swap_perm_ap
    from gold_tipping_points import greedy_flip_to_target, MAX_FLIPS
    from report import bootstrap_auroc_ci

    facts = json.loads(FACTS.read_text())
    fv = lambda k: facts[k]["value"] if isinstance(facts[k], dict) else facts[k]

    dec = pd.read_csv(DECONTAM_PRED_PATH)
    nli = pd.read_csv(NLI_PATH)
    rules = pd.read_csv(RULE_PATH)
    if len(dec) != 253:
        raise AbortC(f"decontam rows {len(dec)} != 253")
    j = dec[["pair_id", "y_true", "prob_seed0", "prob_seed1", "prob_seed2"]].merge(
        nli[["pair_id", "contra_prob_mean", "label_clean"]], on="pair_id", how="inner").merge(
        rules[["pair_id", "rule"]], on="pair_id", how="inner")
    n_mis = int((j["y_true"] != (j["label_clean"] == "contradiction").astype(int)).sum())
    if len(j) != 253 or n_mis != 0:
        raise AbortC(f"join gate failed: n={len(j)} mismatches={n_mis}")

    y = j["y_true"].to_numpy().astype(int)
    ens = j[["prob_seed0", "prob_seed1", "prob_seed2"]].to_numpy().mean(axis=1)
    nli_s = j["contra_prob_mean"].to_numpy()
    ens_auc = float(roc_auc_score(y, ens))
    if round(ens_auc, 4) != round(fv("decontam.aggregate.ensemble_auroc"), 4):
        raise AbortC(f"ensemble AUROC reproduction failed: {ens_auc:.4f}")

    m34 = np.isin(j["rule"].to_numpy(), (3, 4))
    y34, e34 = y[m34], ens[m34]
    auc34 = float(roc_auc_score(y34, e34))
    if int(m34.sum()) != 108 or int(y34.sum()) != 6 or \
            round(auc34, 4) != round(fv("c112.rules34_ambiguous_dropped.ensemble.auroc"), 4):
        raise AbortC(f"stratum gate failed: n={int(m34.sum())} pos={int(y34.sum())} auc={auc34:.4f}")

    # Task A
    ta = paired_swap_perm_ap(y, nli_s, ens, B_PERM, SEED)
    a1 = bool(ta["p_permutation_two_sided"] < 0.05 and ta["observed_diff"] > 0)

    # Task B
    tb = {"baseline_auc": ens_auc, "search_cap": MAX_FLIPS, "targets": {}}
    for name, t in TARGETS.items():
        n_flips, final_auc, directions, reached, _trace = greedy_flip_to_target(y, ens, t)
        tb["targets"][name] = {"target_auc": t, "n_flips_greedy": int(n_flips),
                               "final_auc": float(final_auc), "reached_within_cap": bool(reached),
                               "flip_directions": directions}

    # Task C
    tc = bootstrap_auroc_ci(y34, e34, n_boot=10_000, seed=SEED)
    tc.update({"n": int(m34.sum()), "n_pos": int(y34.sum()), "stratum_auroc": auc34})

    return {
        "preregistration": "accord/c114_committee_checks/C114_PREREGISTRATION.md",
        "gates": {"n_joined": int(len(j)), "label_mismatches": n_mis,
                  "ensemble_auroc_g253": ens_auc, "stratum_n": int(m34.sum()),
                  "stratum_pos": int(y34.sum()), "stratum_auroc": auc34},
        "task_a_nli_minus_detector_ensemble_g253": ta,
        "task_a_outcome": "A1" if a1 else "A2",
        "task_b_tipping_from_decontam_ensemble_g253": tb,
        "task_c_bootstrap_rules34_ambiguous_dropped": tc,
        "outcome": "valid",
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w") as logf, contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
        try:
            result = run()
        except AbortC as e:
            OUT.write_text(json.dumps({"outcome": "C", "abort_reason": str(e)}, indent=2))
            raise SystemExit("ABORT (Outcome C); reason written to the output JSON")
    OUT.write_text(json.dumps(result, indent=2))
    ta = result["task_a_nli_minus_detector_ensemble_g253"]
    print(f"task A {result['task_a_outcome']}: diff={ta['observed_diff']:.4f} p={ta['p_permutation_two_sided']:.5f}")
    for k, v in result["task_b_tipping_from_decontam_ensemble_g253"]["targets"].items():
        print(f"task B {k}: flips={v['n_flips_greedy']} reached={v['reached_within_cap']}")
    tc = result["task_c_bootstrap_rules34_ambiguous_dropped"]
    print(f"task C: ci95={tc['ci95']} n_valid={tc['n_boot_valid']}")


if __name__ == "__main__":
    main()
