"""C138c: reconcile the two bootstrap definitions the bounded null mixes.

The thesis's headline G_253 interval, 0.3210 to 0.7034, is an UNSTRATIFIED row
bootstrap (accord/control/report.py::bootstrap_auroc_ci resamples n rows with
replacement and discards single-class draws). The 0.7466 floor it is compared
against comes from the C108 power simulation, whose synthetic replicates hold
the class counts fixed at 12 positives and 241 negatives and whose one
bootstrap-type decision rule is an explicitly STRATIFIED percentile bootstrap
(accord/public_controls/evaluation.py::rule_bootstrap_lower_excludes_chance).
The thesis never states that the two sides of that comparison use different
resampling conventions.

This script computes BOTH intervals for BOTH ensembles at the same seed 42 and
the same B = 10,000, and reports the four upper ends side by side against the
0.7466 floor and against the C132 cluster-bootstrap margin of 0.0236.

Inputs are per-row scores already on disk. No model is trained or scored, no
hosted API is called. Aggregates only: no pair_id, no HADM_ID, no SUBJECT_ID
and no note text is written or printed.
"""
from __future__ import annotations

import collections
import json
import os
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
MIMIC = pathlib.Path(os.environ.get("MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
DECONTAM_PRED = ROOT / "local_only/from_drive_2026-09-13/outputs/gold_eval_decontam_predictions.csv"
C130_DIR = MIMIC / "accord_c130_reproducible_student"
C130_SEEDS = (42, 123, 2024)

OUT_DIR = ROOT / "results/accord/promised_analyses"
OUT = OUT_DIR / "c138c_bootstrap_definitions.json"

SEED, B = 42, 10_000
POWER_FLOOR = 0.7466              # sizectl108 mw_two_sided crossing at power 0.80
BOOTSTRAP_RULE_FLOOR = 0.7766     # the same simulation's bootstrap-rule crossing
CLUSTER_MARGIN = 0.0236           # c132.decision.min_margin_below_floor, 4 dp


def boot_unstratified(y, s, seed=SEED, B=B):
    """The thesis convention: resample n rows, discard single-class draws."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out, dropped = [], 0
    while len(out) < B and (len(out) + dropped) < B * 20:
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == n:
            dropped += 1
            continue
        out.append(roc_auc_score(yy, s[idx]))
    a = np.array(out)
    return a, dropped


def boot_stratified(y, s, seed=SEED, B=B):
    """The power simulation's convention: class counts held fixed."""
    rng = np.random.default_rng(seed)
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for _ in range(B):
        idx = np.r_[rng.choice(ip, len(ip), True), rng.choice(ineg, len(ineg), True)]
        out.append(roc_auc_score(y[idx], s[idx]))
    return np.array(out), 0


def interval(a, dropped):
    return collections.OrderedDict([
        ("ci95_lo", float(np.percentile(a, 2.5))),
        ("ci95_hi", float(np.percentile(a, 97.5))),
        ("n_boot_valid", int(len(a))),
        ("n_draws_dropped_single_class", int(dropped)),
        ("boot_mean", float(a.mean())),
        ("boot_sd", float(a.std(ddof=1))),
        ("width", float(np.percentile(a, 97.5) - np.percentile(a, 2.5))),
    ])


def load_decontam():
    df = pd.read_csv(DECONTAM_PRED)
    y = df["y_true"].to_numpy()
    s = np.mean([df[f"prob_seed{k}"].to_numpy() for k in (0, 1, 2)], axis=0)
    return y, s, set(df["pair_id"])


def load_c130(g253_pair_ids):
    frames = {k: pd.read_csv(C130_DIR / f"scores_seed{k}.csv") for k in C130_SEEDS}
    ref = frames[C130_SEEDS[0]]["pair_id"].to_numpy()
    for k in C130_SEEDS:
        assert (frames[k]["pair_id"].to_numpy() == ref).all(), "pair_id order mismatch across C130 seeds"
    ens = np.vstack([frames[k]["score"].to_numpy() for k in C130_SEEDS]).mean(axis=0)
    y299 = frames[C130_SEEDS[0]]["pharm_pos"].to_numpy()
    mask = frames[C130_SEEDS[0]]["pair_id"].isin(g253_pair_ids).to_numpy()
    assert mask.sum() == 253, f"G_253 mask selected {int(mask.sum())} rows"
    return y299[mask], ens[mask]


def main():
    y_d, s_d, g253_ids = load_decontam()
    assert len(y_d) == 253 and int(y_d.sum()) == 12
    y_c, s_c = load_c130(g253_ids)
    assert len(y_c) == 253 and int(y_c.sum()) == 12

    res = collections.OrderedDict()
    res["provenance"] = {
        "generated_by": "accord/promised_analyses/c138c_bootstrap_definitions.py",
        "seed": SEED,
        "n_boot": B,
        "basis": "G_253, 12 positives and 241 negatives",
        "unstratified_rule": "accord/control/report.py::bootstrap_auroc_ci, n rows resampled, single-class draws discarded",
        "stratified_rule": "accord/public_controls/evaluation.py::stratified_bootstrap_auroc, class counts held fixed, the rule used by the C108 power simulation",
        "power_floor_mw_two_sided": POWER_FLOOR,
        "power_floor_bootstrap_rule": BOOTSTRAP_RULE_FLOOR,
        "c132_min_margin_below_floor": CLUSTER_MARGIN,
        "no_model_inference": True,
    }

    ensembles = collections.OrderedDict([
        ("decontam_ensemble", (y_d, s_d)),
        ("c130_ensemble", (y_c, s_c)),
    ])
    for name, (y, s) in ensembles.items():
        blk = collections.OrderedDict()
        blk["auroc"] = float(roc_auc_score(y, s))
        blk["n"] = int(len(y))
        blk["n_pos"] = int(y.sum())
        a, d = boot_unstratified(y, s)
        blk["unstratified"] = interval(a, d)
        a, d = boot_stratified(y, s)
        blk["stratified"] = interval(a, d)
        blk["upper_end_shift_stratified_minus_unstratified"] = round(
            blk["stratified"]["ci95_hi"] - blk["unstratified"]["ci95_hi"], 6)
        blk["lower_end_shift_stratified_minus_unstratified"] = round(
            blk["stratified"]["ci95_lo"] - blk["unstratified"]["ci95_lo"], 6)
        for conv in ("unstratified", "stratified"):
            blk[conv]["margin_below_power_floor"] = round(POWER_FLOOR - blk[conv]["ci95_hi"], 6)
            blk[conv]["hi_below_power_floor"] = bool(blk[conv]["ci95_hi"] < POWER_FLOOR)
            blk[conv]["contains_0_5"] = bool(blk[conv]["ci95_lo"] <= 0.5 <= blk[conv]["ci95_hi"])
            blk[conv]["margin_below_bootstrap_rule_floor"] = round(BOOTSTRAP_RULE_FLOOR - blk[conv]["ci95_hi"], 6)
        res[name] = blk

    ue = collections.OrderedDict([
        ("decontam_unstratified_hi", res["decontam_ensemble"]["unstratified"]["ci95_hi"]),
        ("decontam_stratified_hi", res["decontam_ensemble"]["stratified"]["ci95_hi"]),
        ("c130_unstratified_hi", res["c130_ensemble"]["unstratified"]["ci95_hi"]),
        ("c130_stratified_hi", res["c130_ensemble"]["stratified"]["ci95_hi"]),
    ])
    res["four_upper_ends"] = ue
    res["decision"] = collections.OrderedDict([
        ("all_four_below_power_floor", all(v < POWER_FLOOR for v in ue.values())),
        ("max_upper_end", max(ue.values())),
        ("argmax_upper_end", max(ue, key=ue.get)),
        ("min_margin_below_power_floor", round(POWER_FLOOR - max(ue.values()), 6)),
        ("all_four_contain_0_5", all(res[n][c]["contains_0_5"] for n in ensembles for c in ("unstratified", "stratified"))),
        ("stratified_is_tighter_at_upper_end_both_ensembles",
         all(res[n]["upper_end_shift_stratified_minus_unstratified"] < 0 for n in ensembles)),
        ("bounded_null_verdict_unchanged_under_stratified_definition", all(v < POWER_FLOOR for v in ue.values())),
    ])

    # Reproduction checks against the registered row-level values, so a drift in
    # either score file shows up here rather than silently changing a verdict.
    reg = json.loads((ROOT / "results/accord/c132_cluster_bootstrap/c132.json").read_text())
    res["reproduction_checks"] = {
        "decontam_auroc_matches_c132_4dp": round(res["decontam_ensemble"]["auroc"], 4) == round(reg["task_b"]["decontam_ensemble"]["auroc"], 4),
        "c130_auroc_matches_c132_4dp": round(res["c130_ensemble"]["auroc"], 4) == round(reg["task_b"]["c130_ensemble"]["auroc"], 4),
        "decontam_unstratified_hi_matches_registered_4dp": round(res["decontam_ensemble"]["unstratified"]["ci95_hi"], 4) == round(reg["task_b"]["decontam_ensemble"]["row_level_seed42_check"]["ci95_hi"], 4),
        "c130_unstratified_hi_matches_registered_4dp": round(res["c130_ensemble"]["unstratified"]["ci95_hi"], 4) == round(reg["task_b"]["c130_ensemble"]["row_level_seed42_check"]["ci95_hi"], 4),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=1))


if __name__ == "__main__":
    main()
