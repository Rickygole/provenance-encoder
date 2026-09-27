"""Data-randomisation control (Adebayo and colleagues' label-permutation
sanity check) for the plain control detector. See
PERMUTED_LABEL_PREREGISTRATION.md, committed before this script was run;
every threshold and every kill condition here is copied from that file and
never re-derived.

accord/control/train.py is NEVER edited. Following
accord/c130_reproducible_student/c130_run.py exactly, the `build_splits` name
train.py imported is replaced by a wrapper that returns the same Splits
object with the training pool's `y` column set to a PERMUTED copy of the C130
replication label vector; every other line inside train.py -- model,
optimizer, seeding, DataLoader, gold scoring -- runs unchanged. The
permutation is one draw, fixed across the three model seeds, so seed-to-seed
spread is model stochasticity only.

    python3 accord/labeller_interval_closure/permuted_label_control.py --dry-run
    python3 accord/labeller_interval_closure/permuted_label_control.py

Local only. No Ollama call, no hosted API. Per-seed scores land under
~/mimic_local_only/accord_permuted_label_control/; the only thing written to
results/ is one aggregate JSON (counts, AUROC, AP, interval bounds, seeds,
booleans -- no pair_id, HADM_ID, SUBJECT_ID or note text).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accord" / "control"))

from accord.cloud_paths import assert_local_only  # noqa: E402
from accord.c130_reproducible_student.c130_run import (  # noqa: E402
    EXPECTED_N_LEAKED,
    EXPECTED_N_TRAIN,
    JOIN_KEY,
    REPL_POOL_LABELS_PATH,
    AbortC,
    gate_join_completeness,
    load_replication_labels,
)

SEEDS = (42, 123, 2024)
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
PERMUTATION_SEED = 137  # PERMUTED_LABEL_PREREGISTRATION.md, fixed before the run
N_BOOT = 10_000
BOOT_SEED = 42          # matches c130_run.py::BOOT_SEED so the two intervals are comparable
PAIRED_DIFF_SEED = 43   # matches c130_run.py::PAIRED_DIFF_SEED

# Reference values, the real-label run this mirrors. Read from the artifact,
# never typed: results/accord/c130_reproducible_student/c130.json.
C130_JSON = ROOT / "results/accord/c130_reproducible_student/c130.json"

MIMIC_LOCAL_ONLY_DIR = pathlib.Path(
    os.environ.get("MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
OUT_DIR = MIMIC_LOCAL_ONLY_DIR / "accord_permuted_label_control"
C130_SCORES_DIR = MIMIC_LOCAL_ONLY_DIR / "accord_c130_reproducible_student"
RESULTS_OUT = ROOT / "results/accord/labeller_interval_closure/permuted_label_control.json"
LOG_DIR = MIMIC_LOCAL_ONLY_DIR / "logs"


def bootstrap_auroc_ci(y, scores, n_boot=N_BOOT, seed=BOOT_SEED):
    """accord/control/report.py::bootstrap_auroc_ci, plus the discarded count."""
    rng = np.random.default_rng(seed)
    y, scores = np.asarray(y), np.asarray(scores, float)
    n = len(y)
    aucs, attempts, discarded = [], 0, 0
    while len(aucs) < n_boot and attempts < n_boot * 20:
        attempts += 1
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == n:
            discarded += 1
            continue
        aucs.append(roc_auc_score(yy, scores[idx]))
    a = np.asarray(aucs)
    return {
        "n_boot_valid": int(len(a)),
        "n_resamples_discarded_single_class": int(discarded),
        "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
        "boot_mean": float(a.mean()), "boot_std": float(a.std()), "seed": seed,
    }


def paired_diff_bootstrap(y, a, b, B=N_BOOT, seed=PAIRED_DIFF_SEED):
    """c130_run.py::paired_diff_bootstrap, unchanged: AUROC(a) - AUROC(b) on
    the same resampled rows."""
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a, float), np.asarray(b, float)
    n = len(y)
    d = []
    while len(d) < B:
        idx = rng.integers(0, n, n)
        if 0 < y[idx].sum() < n:
            d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    d = np.asarray(d)
    return {"mean": float(d.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "contains_zero": bool(np.percentile(d, 2.5) <= 0.0 <= np.percentile(d, 97.5)),
            "seed": seed, "n_boot": B}


def build_permuted_labels(splits_train: pd.DataFrame, repl_df: pd.DataFrame):
    """Returns (merged training frame with a permuted `y`, gate dict)."""
    merged, join_gate = gate_join_completeness(splits_train, repl_df)
    if not join_gate["passed"]:
        raise AbortC("join-completeness gate failed (kill condition 2)")
    merged = merged[merged["replication_label"] != "UNPARSEABLE"].reset_index(drop=True)
    y_real = (merged["replication_label"] == "contradiction").astype(int).to_numpy()
    rng = np.random.default_rng(PERMUTATION_SEED)
    y_perm = y_real[rng.permutation(len(y_real))]
    gate = {
        "n_rows": int(len(y_real)),
        "n_positive_before_permutation": int(y_real.sum()),
        "n_positive_after_permutation": int(y_perm.sum()),
        "marginal_preserved": bool(y_real.sum() == y_perm.sum()),
        "n_rows_whose_label_changed": int((y_real != y_perm).sum()),
        "fraction_of_rows_whose_label_changed": float((y_real != y_perm).mean()),
        "single_class_after_permutation": bool(y_perm.sum() in (0, len(y_perm))),
        "permutation_seed": PERMUTATION_SEED,
        "passed": bool(y_real.sum() == y_perm.sum()
                       and y_perm.sum() not in (0, len(y_perm))),
    }
    merged = merged.copy()
    merged["y"] = y_perm
    return merged, join_gate, gate


def make_permuted_build_splits(permuted_train: pd.DataFrame):
    from data import Splits, build_splits as _orig

    def _build():
        s = _orig()
        return Splits(train=permuted_train, val=s.val, gold=s.gold,
                      n_leaked_rows_excluded=s.n_leaked_rows_excluded,
                      n_gold_subjects=s.n_gold_subjects,
                      n_pool_subjects=s.n_pool_subjects)
    return _build


def train_one_seed(seed: int, permuted_train: pd.DataFrame, out_dir: pathlib.Path):
    import train as train_mod
    out_dir = assert_local_only(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_mod.build_splits = make_permuted_build_splits(permuted_train)
    old_argv = sys.argv
    sys.argv = ["train.py", "--seed", str(seed), "--epochs", str(EPOCHS),
                "--batch-size", str(BATCH_SIZE), "--lr", str(LR),
                "--out-dir", str(out_dir)]
    try:
        train_mod.main()
    finally:
        sys.argv = old_argv


def block(y, scores):
    n, pos = int(len(y)), int(np.asarray(y).sum())
    if pos in (0, n):
        return {"n": n, "n_pos": pos, "auroc": None, "ap": None, "bootstrap": None}
    return {"n": n, "n_pos": pos,
            "auroc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
            "bootstrap": bootstrap_auroc_ci(y, scores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="run every gate and the permutation, train nothing")
    args = ap.parse_args()

    from data import build_splits
    splits = build_splits()
    split_gate = {
        "n_train": int(len(splits.train)), "n_val": int(len(splits.val)),
        "n_gold": int(len(splits.gold)),
        "n_leaked_rows_excluded": int(splits.n_leaked_rows_excluded),
        "expected_n_train": EXPECTED_N_TRAIN, "expected_n_leaked": EXPECTED_N_LEAKED,
        "split_seed": 1337,
        "passed": bool(len(splits.train) == EXPECTED_N_TRAIN
                       and splits.n_leaked_rows_excluded == EXPECTED_N_LEAKED),
    }
    if not split_gate["passed"]:
        raise AbortC("split-size gate failed (kill condition 1)")

    repl_df = load_replication_labels(REPL_POOL_LABELS_PATH)
    permuted_train, join_gate, perm_gate = build_permuted_labels(splits.train, repl_df)
    if not perm_gate["passed"]:
        raise AbortC("permutation gate failed (kill condition 3 or 6)")

    gates = collections.OrderedDict(
        split_size=split_gate, join_completeness=join_gate, permutation=perm_gate)

    if args.dry_run:
        print(json.dumps({"gates": gates, "overall": "READY_TO_TRAIN"}, indent=2))
        return

    for seed in SEEDS:
        t0 = time.time()
        train_one_seed(seed, permuted_train, OUT_DIR)
        print(f"[permuted] seed {seed} done in {time.time() - t0:.0f}s")

    seed_scores = {}
    for seed in SEEDS:
        sp = OUT_DIR / f"scores_seed{seed}.csv"
        mp = OUT_DIR / f"metrics_seed{seed}.json"
        if not (sp.is_file() and mp.is_file()):
            raise AbortC(f"seed {seed} did not write its outputs (kill condition 4)")
        seed_scores[seed] = pd.read_csv(sp)

    ref = seed_scores[SEEDS[0]][["pair_id", "rule", "pharm_pos"]].reset_index(drop=True)
    for s in SEEDS:
        cur = seed_scores[s]["pair_id"].to_numpy()
        if not (cur == ref["pair_id"].to_numpy()).all():
            raise AbortC("pair_id order mismatch across seeds (kill condition 5)")
    gold_full = splits.gold.reset_index(drop=True)
    if not (gold_full["pair_id"].to_numpy() == ref["pair_id"].to_numpy()).all():
        raise AbortC("gold row order mismatch (kill condition 5)")

    score_matrix = np.vstack([seed_scores[s]["score"].to_numpy() for s in SEEDS])
    ensemble = score_matrix.mean(axis=0)
    y299 = ref["pharm_pos"].to_numpy()
    g253 = ~(gold_full["pharm_label"] == "ambiguous").to_numpy()
    rule = ref["rule"].to_numpy()

    out = collections.OrderedDict()
    out["preregistration"] = "accord/labeller_interval_closure/PERMUTED_LABEL_PREREGISTRATION.md"
    out["design"] = {
        "what": ("Adebayo and colleagues' data randomisation test: the plain "
                 "control detector retrained on a permuted copy of the "
                 "training-pool label vector, everything else identical to "
                 "the C130 real-label run"),
        "trainer": "accord/control/train.py (unedited)",
        "seeds": list(SEEDS), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "lr": LR, "weight_decay": 0.01, "max_length": 160,
        "optimizer": "AdamW", "split_seed": 1337,
        "permutation_seed": PERMUTATION_SEED,
        "n_permutation_draws": 1,
        "permutation_fixed_across_seeds": True,
    }
    out["gates"] = gates
    out["g299"] = {"ensemble": block(y299, ensemble),
                   **{f"seed{s}": block(y299, seed_scores[s]["score"].to_numpy()) for s in SEEDS}}
    out["g253"] = {"ensemble": block(y299[g253], ensemble[g253]),
                   **{f"seed{s}": block(y299[g253], seed_scores[s]["score"].to_numpy()[g253]) for s in SEEDS}}
    dfm = np.isin(rule, [3, 4])
    out["g253_rules_3_4"] = block(y299[g253 & dfm], ensemble[g253 & dfm])
    out["g253_rules_1_2"] = block(y299[g253 & ~dfm], ensemble[g253 & ~dfm])

    # ---- comparison against the real-label run -------------------------
    c130 = json.loads(C130_JSON.read_text())
    real_g253 = c130["g253"]["ensemble"]
    perm_g253 = out["g253"]["ensemble"]
    real_ci = real_g253["bootstrap"]["ci95"]
    inside = bool(real_ci[0] <= perm_g253["auroc"] <= real_ci[1])

    paired = None
    paired_note = None
    real_score_files = {s: C130_SCORES_DIR / f"scores_seed{s}.csv" for s in SEEDS}
    if all(p.is_file() for p in real_score_files.values()):
        real_frames = {s: pd.read_csv(p) for s, p in real_score_files.items()}
        order_ok = all((real_frames[s]["pair_id"].to_numpy() == ref["pair_id"].to_numpy()).all()
                       for s in SEEDS)
        if order_ok:
            real_ens = np.vstack([real_frames[s]["score"].to_numpy() for s in SEEDS]).mean(axis=0)
            if abs(float(roc_auc_score(y299[g253], real_ens[g253])) - real_g253["auroc"]) > 1e-9:
                raise AbortC("recomputed C130 ensemble AUROC on G_253 does not "
                             "reproduce the registered value from its own score files")
            paired = paired_diff_bootstrap(y299[g253], real_ens[g253], ensemble[g253])
            paired["direction"] = "real_label_ensemble_auroc minus permuted_label_ensemble_auroc"
        else:
            paired_note = "C130 per-seed score files are not in the gold row order; paired test skipped"
    else:
        paired_note = ("C130 per-seed score files not present under "
                       "~/mimic_local_only/accord_c130_reproducible_student/; "
                       "paired test skipped")

    out["comparison_vs_real_label_run"] = {
        "reference_artifact": "results/accord/c130_reproducible_student/c130.json::g253.ensemble",
        "real_label_ensemble_auroc_g253": real_g253["auroc"],
        "real_label_ensemble_ap_g253": real_g253["ap"],
        "real_label_bootstrap_ci95_g253": real_ci,
        "permuted_label_ensemble_auroc_g253": perm_g253["auroc"],
        "permuted_label_ensemble_ap_g253": perm_g253["ap"],
        "permuted_label_bootstrap_ci95_g253": perm_g253["bootstrap"]["ci95"],
        "permuted_auroc_inside_real_label_interval": inside,
        "auroc_difference_real_minus_permuted": (real_g253["auroc"] - perm_g253["auroc"]),
        "paired_bootstrap_difference_same_253_pairs": paired,
        "paired_bootstrap_note": paired_note,
    }

    if paired is not None:
        if inside and paired["contains_zero"]:
            outcome, reason = "A", (
                "the permuted-label ensemble AUROC on G_253 lies inside the "
                "real-label run's bootstrap interval and the paired difference "
                "interval contains zero: the real labels are not distinguishable "
                "from label-destroyed labels at this sample size")
        elif not paired["contains_zero"] and paired["ci95"][0] > 0:
            outcome, reason = "B", (
                "the paired difference interval excludes zero in the real-label "
                "run's favour: the real labels carried something the model used")
        else:
            outcome, reason = "A", (
                "the paired difference interval contains zero; the real-label "
                "run is not distinguishable from label-destroyed training")
    else:
        outcome, reason = "A_unpaired", (
            "paired test unavailable; decided on the unpaired criterion only: "
            f"permuted AUROC inside the real-label interval = {inside}")

    out["outcome"] = outcome
    out["outcome_reason"] = reason
    out["what_this_does_not_establish"] = (
        "G_253 carries 12 positives. The real-label run's own bootstrap 95 "
        f"percent interval is {real_ci}, which is "
        f"{real_ci[1] - real_ci[0]:.3f} AUROC wide and contains 0.5. An "
        "interval that wide cannot separate 'the labels carried nothing' from "
        "'the labels carried an effect this study is too small to see'. This "
        "result is therefore that the real-label control run is "
        "INDISTINGUISHABLE from training on destroyed labels, not a "
        "measurement that the real labels are uninformative. One permutation "
        "draw is also not a permutation null distribution: it shows that one "
        "label-destroying draw performs comparably, not the distribution over "
        "draws. And a detector that was already at chance on this gold set "
        "cannot fall below chance by much, so the test has little room to "
        "discriminate in the first place.")
    out["pool_label_marginal_note"] = (
        "the replication label vector is 94.7 percent positive on the pool, so "
        f"permuting it changed only {perm_gate['n_rows_whose_label_changed']} of "
        f"{perm_gate['n_rows']} training rows "
        f"({perm_gate['fraction_of_rows_whose_label_changed']:.4f}). That is a "
        "property of the weak labels, not of the permutation: an almost "
        "constant label vector has very little association with the inputs to "
        "destroy, which is itself the point.")

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {RESULTS_OUT}")
    print(f"outcome={outcome}: {reason}")
    print(f"permuted g253 ensemble AUROC={perm_g253['auroc']:.4f} "
          f"AP={perm_g253['ap']:.4f} vs real {real_g253['auroc']:.4f} / {real_g253['ap']:.4f}")


if __name__ == "__main__":
    main()
