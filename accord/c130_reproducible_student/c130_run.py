"""C130: retrain the defect-free control detector's configuration with labels
from the reproducible llama3:8b weak-labelling replication
(accord/differential/run_multillm_replication.py) instead of the lost deployed
run, on the SAME subject-disjoint 1,183-row training pool
(accord.control.data.build_splits().train), and check whether the detector's
chance-level ranking of the gold pairs reproduces. See C130_PREREGISTRATION.md,
committed before this script was run.

accord/control/train.py is never edited. The label substitution is made by
replacing the `build_splits` name train.py imported ("from data import
build_splits") with a wrapper that returns the same Splits object with the
training pool's `y` column swapped for the replication run's label, then
calling train.main() once per seed with argv set for this run's own output
directory. Every other line inside train.py -- model, optimizer, seeding,
DataLoader, gold-scoring -- runs unchanged.

    python3 accord/c130_reproducible_student/c130_run.py --dry-run
    python3 accord/c130_reproducible_student/c130_run.py            # full run; NOT executed by this task

Local only. No Ollama call, no hosted API. Any code path that loads
accord.control.data or reads ~/mimic_local_only redirects stdout/stderr to a
log file under ~/mimic_local_only/logs; that log is never read back into this
task. --dry-run writes nothing to results/; it writes its gate summary to the
log only, and prints a short counts-and-booleans-only summary to real stdout.
The full run writes exactly one aggregate JSON,
results/accord/c130_reproducible_student/c130.json (counts, AUROC, AP, CI
bounds, seeds, booleans only -- no pair_id, HADM_ID, SUBJECT_ID, or note
text).
"""
from __future__ import annotations

import argparse
import collections
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
sys.path.insert(0, str(ROOT / "accord" / "control"))

from accord.cloud_paths import assert_local_only  # noqa: E402

JOIN_KEY = ["HADM_ID", "drug", "rule", "nursing_sentence", "discharge_sentence"]
SEEDS = (42, 123, 2024)
EPOCHS = 3
EXPECTED_N_TRAIN = 1183
EXPECTED_N_LEAKED = 291

REPL_POOL_LABELS_PATH = pathlib.Path.home() / "mimic_local_only" / "accord_multillm" / "pool_replication_labels_llama3_8b.csv"
REPL_GOLD_LABELS_PATH = pathlib.Path.home() / "mimic_local_only" / "accord_multillm" / "gold_replication_labels_llama3_8b.csv"
WS_PATH = pathlib.Path.home() / "mimic_local_only" / "accord_supp" / "accord_weak_supervision_labels_decontam.csv"
PER_MODEL_RESULTS = ROOT / "results" / "accord" / "multillm" / "per_model_results.json"
FACTS_PATH = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
REGISTERED_MARGINAL_KEY = "overcall.llama3_8b.pool.positive_rate"

C130_OUT_DIR = pathlib.Path.home() / "mimic_local_only" / "accord_c130_reproducible_student"
RESULTS_OUT = ROOT / "results" / "accord" / "c130_reproducible_student" / "c130.json"

LOG_DIR = pathlib.Path.home() / "mimic_local_only" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = assert_local_only(LOG_DIR / f"c130_run_{time.strftime('%Y%m%dT%H%M%S')}.log")

N_BOOT = 10_000
BOOT_SEED = 42          # ensemble AUROC CI, per report.py's bootstrap_auroc_ci, seed fixed by this task
PAIRED_DIFF_SEED = 43   # student-minus-teacher, matching c126_run.py::paired_diff
POWER_FLOOR_G253 = 0.7466  # sizectl108.power_curves.primary_crossing_power_0_80


class AbortC(RuntimeError):
    """Raised to signal Outcome C (invalid run): no decision is recorded."""


# ---------------------------------------------------------------------------
# Gates. All five run under the LOG redirect; only counts/booleans leave it.
# ---------------------------------------------------------------------------

def gate_split_size():
    from data import build_splits
    splits = build_splits()
    ok = len(splits.train) == EXPECTED_N_TRAIN and splits.n_leaked_rows_excluded == EXPECTED_N_LEAKED
    return splits, {
        "n_train": int(len(splits.train)),
        "n_val": int(len(splits.val)),
        "n_gold": int(len(splits.gold)),
        "n_leaked_rows_excluded": int(splits.n_leaked_rows_excluded),
        "expected_n_train": EXPECTED_N_TRAIN,
        "expected_n_leaked": EXPECTED_N_LEAKED,
        "passed": bool(ok),
    }


def gate_label_file_exists(path: pathlib.Path):
    return {"path": str(path), "exists": bool(path.is_file())}


def load_replication_labels(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_cols = [c for c in JOIN_KEY + ["replication_label"] if c not in df.columns]
    if missing_cols:
        raise AbortC(f"replication label file missing required columns: {missing_cols}")
    df["HADM_ID"] = df["HADM_ID"].astype(float)
    df["discharge_sentence"] = df["discharge_sentence"].fillna("")
    df["nursing_sentence"] = df["nursing_sentence"].fillna("")
    return df


def gate_label_file_uniqueness(repl_df: pd.DataFrame):
    n_dup = int(repl_df.duplicated(subset=JOIN_KEY).sum())
    return {"n_rows": int(len(repl_df)), "n_duplicate_join_key_rows": n_dup, "passed": n_dup == 0}


def gate_join_completeness(train_df: pd.DataFrame, repl_df: pd.DataFrame):
    n_train_dup = int(train_df.duplicated(subset=JOIN_KEY).sum())
    merged = train_df.merge(
        repl_df[JOIN_KEY + ["replication_label"]], on=JOIN_KEY, how="left"
    )
    n_train = int(len(train_df))
    n_matched_rows = int(len(merged))
    n_missing_label = int(merged["replication_label"].isna().sum())
    no_fanout = n_matched_rows == n_train
    no_missing = n_missing_label == 0
    return merged, {
        "n_train_rows": n_train,
        "n_train_duplicate_join_key_rows": n_train_dup,
        "n_matched_rows_after_join": n_matched_rows,
        "n_pool_rows_missing_a_replication_label": n_missing_label,
        "passed": bool(n_train_dup == 0 and no_fanout and no_missing),
    }


def gate_marginal_match(repl_df: pd.DataFrame, facts: dict):
    ws = pd.read_csv(WS_PATH)
    ws["HADM_ID"] = ws["HADM_ID"].astype(float)
    ws["discharge_sentence"] = ws["discharge_sentence"].fillna("")
    ws["nursing_sentence"] = ws["nursing_sentence"].fillna("")
    n_pool = int(len(ws))
    j = ws[JOIN_KEY].merge(repl_df[JOIN_KEY + ["replication_label"]], on=JOIN_KEY, how="left")
    n_matched = int(j["replication_label"].notna().sum())
    parseable = j[j["replication_label"] != "UNPARSEABLE"]
    n_parseable = int(len(parseable))
    computed_rate = float((parseable["replication_label"] == "contradiction").mean()) if n_parseable else None
    registered_rate = facts.get(REGISTERED_MARGINAL_KEY, {}).get("value")
    rounds_match = (
        computed_rate is not None
        and registered_rate is not None
        and round(computed_rate, 4) == round(float(registered_rate), 4)
    )
    return {
        "n_pool_rows": n_pool,
        "n_pool_rows_matched_in_label_file": n_matched,
        "n_pool_rows_parseable": n_parseable,
        "computed_pool_positive_rate": computed_rate,
        "registered_pool_positive_rate": registered_rate,
        "registered_key": REGISTERED_MARGINAL_KEY,
        "passed": bool(rounds_match and n_matched == n_pool),
    }


def run_dry_run_gates() -> dict:
    facts = json.loads(FACTS_PATH.read_text())
    splits, split_gate = gate_split_size()
    gates = collections.OrderedDict()
    gates["split_size"] = split_gate
    if not split_gate["passed"]:
        raise AbortC("split-size gate failed: build_splits() no longer reproduces the 1,183-row training pool")

    exists_gate = gate_label_file_exists(REPL_POOL_LABELS_PATH)
    gates["label_file_exists"] = exists_gate
    if not exists_gate["exists"]:
        gates["label_file_uniqueness"] = None
        gates["join_completeness"] = None
        gates["marginal_match"] = None
        return {
            "gates": gates,
            "overall": "BLOCKED_MISSING_INPUT",
            "reason": (
                "replication label file not found at "
                f"{REPL_POOL_LABELS_PATH}; per-row llama3:8b pool labels have not been "
                "persisted (run_multillm_replication.py discards them, keeping only "
                "aggregate counts) -- see C130_PREREGISTRATION.md"
            ),
        }

    repl_df = load_replication_labels(REPL_POOL_LABELS_PATH)
    uniq_gate = gate_label_file_uniqueness(repl_df)
    gates["label_file_uniqueness"] = uniq_gate
    if not uniq_gate["passed"]:
        raise AbortC("label-file-uniqueness gate failed: duplicate join-key rows in the replication label file")

    _, join_gate = gate_join_completeness(splits.train, repl_df)
    gates["join_completeness"] = join_gate
    if not join_gate["passed"]:
        raise AbortC("join-completeness gate failed: not every training-pool row has exactly one replication label")

    marg_gate = gate_marginal_match(repl_df, facts)
    gates["marginal_match"] = marg_gate
    if not marg_gate["passed"]:
        raise AbortC(
            "marginal-match gate failed: replication pool positive rate does not match "
            f"the registered {REGISTERED_MARGINAL_KEY}"
        )

    return {"gates": gates, "overall": "READY_TO_TRAIN", "reason": None}


# ---------------------------------------------------------------------------
# Teacher signal on the gold set.
# ---------------------------------------------------------------------------

def teacher_g299_from_registered_confusion() -> dict:
    d = json.loads(PER_MODEL_RESULTS.read_text())
    conf = d["per_model"]["llama3:8b"]["gold"]["confusion"]
    tp, fp, fn, tn = conf["TP"], conf["FP"], conf["FN"], conf["TN"]
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    auroc = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    return {
        "n": tp + fp + fn + tn, "n_pos": tp + fn,
        "confusion": conf, "recall": recall, "specificity": specificity,
        "auroc_binary_score": auroc,
        "source": f"{PER_MODEL_RESULTS.relative_to(ROOT)}::per_model.llama3:8b.gold.confusion",
    }


def teacher_g253_if_available(gold_df: pd.DataFrame):
    if not REPL_GOLD_LABELS_PATH.is_file():
        return None, (
            f"no per-row llama3:8b replication prediction file at {REPL_GOLD_LABELS_PATH}; "
            "G_253 teacher AUROC is not derivable from the registered G_299 confusion cells alone "
            "(ambiguous rows cannot be excluded from an aggregate confusion matrix)"
        )
    gl = load_replication_labels(REPL_GOLD_LABELS_PATH)
    j = gold_df.merge(gl[JOIN_KEY + ["replication_label"]], on=JOIN_KEY, how="left")
    if j["replication_label"].isna().any():
        return None, "gold<->replication-label join incomplete"
    j = j[j["pharm_label"] != "ambiguous"].copy()
    j = j[j["replication_label"] != "UNPARSEABLE"].copy()
    y = (j["pharm_label"] == "contradiction").astype(int).to_numpy()
    score = (j["replication_label"] == "contradiction").astype(float).to_numpy()
    return {"y": y, "score": score, "n": int(len(y)), "n_pos": int(y.sum())}, None


# ---------------------------------------------------------------------------
# Bootstrap helpers (identical construction to accord/control/report.py and
# accord/c126_teacher_student/c126_run.py).
# ---------------------------------------------------------------------------

def paired_diff_bootstrap(y, a, b, B=N_BOOT, seed=PAIRED_DIFF_SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    while len(d) < B:
        idx = rng.integers(0, n, n)
        if 0 < y[idx].sum() < n:
            d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


# ---------------------------------------------------------------------------
# Full run (label-overridden training + scoring). Not executed by this task.
# ---------------------------------------------------------------------------

def make_overridden_build_splits(repl_df: pd.DataFrame):
    """Returns a zero-argument function with the same contract as
    accord.control.data.build_splits(), except splits.train['y'] is replaced
    by the replication label, and rows the replication file marks
    UNPARSEABLE are dropped from the training pool (never coerced to a
    class)."""
    from data import build_splits as _orig_build_splits, Splits

    def _build_splits_c130():
        splits = _orig_build_splits()
        merged, join_gate = gate_join_completeness(splits.train, repl_df)
        if not join_gate["passed"]:
            raise AbortC("join-completeness gate failed inside training wrapper")
        merged = merged[merged["replication_label"] != "UNPARSEABLE"].reset_index(drop=True)
        merged["y"] = (merged["replication_label"] == "contradiction").astype(int)
        return Splits(
            train=merged,
            val=splits.val,
            gold=splits.gold,
            n_leaked_rows_excluded=splits.n_leaked_rows_excluded,
            n_gold_subjects=splits.n_gold_subjects,
            n_pool_subjects=splits.n_pool_subjects,
        )

    return _build_splits_c130


def train_one_seed(seed: int, repl_df: pd.DataFrame, out_dir: pathlib.Path):
    import train as train_mod

    out_dir = assert_local_only(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_mod.build_splits = make_overridden_build_splits(repl_df)
    old_argv = sys.argv
    sys.argv = [
        "train.py", "--seed", str(seed), "--epochs", str(EPOCHS), "--out-dir", str(out_dir),
    ]
    try:
        train_mod.main()
    finally:
        sys.argv = old_argv


def block_metrics(y, scores):
    from report import bootstrap_auroc_ci
    n, pos = int(len(y)), int(np.asarray(y).sum())
    if pos == 0 or pos == n:
        return {"n": n, "n_pos": pos, "auroc": None, "ap": None, "bootstrap": None}
    return {
        "n": n, "n_pos": pos,
        "auroc": float(roc_auc_score(y, scores)),
        "ap": float(average_precision_score(y, scores)),
        "bootstrap": bootstrap_auroc_ci(y, scores, n_boot=N_BOOT, seed=BOOT_SEED),
    }


def run_full(label_path: pathlib.Path):
    facts = json.loads(FACTS_PATH.read_text())
    repl_df = load_replication_labels(label_path)
    uniq_gate = gate_label_file_uniqueness(repl_df)
    if not uniq_gate["passed"]:
        raise AbortC("label-file-uniqueness gate failed")
    marg_gate = gate_marginal_match(repl_df, facts)
    if not marg_gate["passed"]:
        raise AbortC("marginal-match gate failed")

    from data import build_splits
    splits = build_splits()
    if len(splits.train) != EXPECTED_N_TRAIN:
        raise AbortC("split-size gate failed")
    _, join_gate = gate_join_completeness(splits.train, repl_df)
    if not join_gate["passed"]:
        raise AbortC("join-completeness gate failed")

    seed_scores = {}
    for seed in SEEDS:
        train_one_seed(seed, repl_df, C130_OUT_DIR)
        scores_path = C130_OUT_DIR / f"scores_seed{seed}.csv"
        metrics_path = C130_OUT_DIR / f"metrics_seed{seed}.json"
        if not (scores_path.is_file() and metrics_path.is_file()):
            raise AbortC(f"seed {seed} failed to train: expected output files not written")
        seed_scores[seed] = pd.read_csv(scores_path)

    ref = seed_scores[SEEDS[0]][["pair_id", "rule", "pharm_pos"]].reset_index(drop=True)
    for s in SEEDS:
        cur = seed_scores[s][["pair_id", "rule", "pharm_pos"]].reset_index(drop=True)
        if not (cur["pair_id"].to_numpy() == ref["pair_id"].to_numpy()).all():
            raise AbortC("pair_id order mismatch across seeds")

    score_matrix = np.vstack([seed_scores[s]["score"].to_numpy() for s in SEEDS])
    ensemble_scores = score_matrix.mean(axis=0)

    gold_full = splits.gold.reset_index(drop=True)
    if not (gold_full["pair_id"].to_numpy() == ref["pair_id"].to_numpy()).all():
        raise AbortC("gold row order mismatch between this run's own build_splits() and the trained seeds' scores_seed*.csv")
    ambiguous_mask = (gold_full["pharm_label"] == "ambiguous").to_numpy()
    g253_mask = ~ambiguous_mask
    y299 = ref["pharm_pos"].to_numpy()

    out = collections.OrderedDict()
    out["seeds"] = list(SEEDS)
    out["g299"] = {
        "ensemble": block_metrics(y299, ensemble_scores),
        **{f"seed{s}": block_metrics(y299, seed_scores[s]["score"].to_numpy()) for s in SEEDS},
    }
    out["g253"] = {
        "ensemble": block_metrics(y299[g253_mask], ensemble_scores[g253_mask]),
        **{f"seed{s}": block_metrics(y299[g253_mask], seed_scores[s]["score"].to_numpy()[g253_mask]) for s in SEEDS},
    }

    out["teacher_g299"] = teacher_g299_from_registered_confusion()
    teacher_g253, teacher_g253_unavailable_reason = teacher_g253_if_available(gold_full)
    if teacher_g253 is not None:
        student_g253 = ensemble_scores[g253_mask]
        diff = float(roc_auc_score(teacher_g253["y"], student_g253) - roc_auc_score(teacher_g253["y"], teacher_g253["score"]))
        lo, hi = paired_diff_bootstrap(teacher_g253["y"], student_g253, teacher_g253["score"])
        out["teacher_g253"] = {
            "n": teacher_g253["n"], "n_pos": teacher_g253["n_pos"],
            "teacher_auroc": float(roc_auc_score(teacher_g253["y"], teacher_g253["score"])),
            "student_minus_teacher": {"diff": diff, "ci95_lo": lo, "ci95_hi": hi},
        }
    else:
        out["teacher_g253"] = {"available": False, "reason": teacher_g253_unavailable_reason}

    ens_g253 = out["g253"]["ensemble"]
    if ens_g253["bootstrap"] is None:
        decision = "C"
        decision_reason = "G_253 ensemble AUROC bootstrap CI undefined (single-class)"
    else:
        lo, hi = ens_g253["bootstrap"]["ci95"]
        if lo <= 0.5 <= hi and hi < POWER_FLOOR_G253:
            decision = "REPRODUCES"
            decision_reason = "CI includes 0.5 and upper bound is below the power floor"
        elif lo > 0.5:
            decision = "DOES_NOT_REPRODUCE"
            decision_reason = "CI excludes 0.5 on the high side"
        else:
            decision = "INCONCLUSIVE"
            decision_reason = "CI includes 0.5 but upper bound is at or above the power floor"
    out["decision"] = {
        "stratum": "g253", "power_floor": POWER_FLOOR_G253,
        "outcome": decision, "reason": decision_reason,
        "ci95_used": ens_g253.get("bootstrap", {}).get("ci95") if ens_g253["bootstrap"] else None,
    }
    out["gates_passed"] = {"label_file_uniqueness": uniq_gate, "marginal_match": marg_gate, "join_completeness": join_gate}
    return out


def main():
    global REPL_POOL_LABELS_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--label-file", default=str(REPL_POOL_LABELS_PATH))
    args = ap.parse_args()

    REPL_POOL_LABELS_PATH = pathlib.Path(args.label_file)

    with LOG_PATH.open("w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
        print(f"c130_run starting {time.strftime('%Y-%m-%d %H:%M:%S')} dry_run={args.dry_run}")
        try:
            if args.dry_run:
                result = run_dry_run_gates()
            else:
                result = run_full(REPL_POOL_LABELS_PATH)
        except AbortC as e:
            print(f"ABORT (Outcome C): {e}")
            if not args.dry_run:
                RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
                RESULTS_OUT.write_text(json.dumps({"outcome": "C", "abort_reason": str(e)}, indent=2) + "\n")
            print("finished (Outcome C)")
            print(json.dumps({"outcome": "C", "abort_reason": str(e)}, indent=2))
            raise SystemExit(f"ABORT (Outcome C): {e}")
        print("c130_run finished")

    if args.dry_run:
        print(json.dumps(result, indent=2))
        print(f"log: {LOG_PATH}")
        return

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {RESULTS_OUT}")
    print(f"log: {LOG_PATH}")
    print(json.dumps({"decision": result["decision"], "g253_ensemble": result["g253"]["ensemble"]}, indent=2))


if __name__ == "__main__":
    main()
