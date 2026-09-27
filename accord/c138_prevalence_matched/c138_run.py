"""C138: prevalence-matched thresholding of the weak labeller, then distil.

See accord/c138_prevalence_matched/C138_PREREGISTRATION.md, committed before
the first datum. This script implements that file and nothing else; the four
arms, the value of k, the seeds, the decision rule and the kill condition are
all fixed there.

accord/control/train.py is NEVER edited. As in
accord/c130_reproducible_student/c130_run.py, the `build_splits` name that
train.py imported is rebound to a wrapper returning the same Splits object with
train['y'] swapped for this arm's label, and train.main() is called once per
seed in-process with argv set for that arm's own output directory. Every other
line in train.py -- model, optimizer, seeding, DataLoader, gold scoring -- runs
unchanged.

Local only. Writes per-seed scores under ~/mimic_local_only/accord_c138/<arm>/
and exactly one aggregate JSON,
results/accord/c138_prevalence_matched/c138.json: counts, rates, AUROC, AP, CI
bounds, booleans. No note text, no HADM_ID, no SUBJECT_ID, no pair_id.

  python3 accord/c138_prevalence_matched/c138_run.py --gates-only
  python3 accord/c138_prevalence_matched/c138_run.py
"""
from __future__ import annotations

import argparse
import collections
import contextlib
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

SEEDS = (42, 123, 2024)
EPOCHS = 3
EXPECTED_N_TRAIN = 1183
EXPECTED_N_LEAKED = 291
N_BOOT = 10_000
BOOT_SEED = 42
PAIRED_DIFF_SEED = 43

# Pre-registered constants. Do not change; a change here is a new registration.
GOLD_BASE_RATE_299 = 0.04013377926421405
K_BY_ARM = collections.OrderedDict([
    ("PM47", GOLD_BASE_RATE_299),
    ("PM56", 12.0 / 253.0),
    ("PM95", 2.0 * GOLD_BASE_RATE_299),
])
PM47_RATE_TOLERANCE = 0.005
MAX_PARSE_FAILURE_RATE = 0.05
MIN_DISTINCT_LOGODDS = 100
MDE_AUROC_253 = 0.7382  # results/accord/audit_c137/a6_delong_mde.json::mde_alt_r4

MIMIC_LOCAL_ONLY_DIR = pathlib.Path(
    os.environ.get("MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
CACHE_PATH = MIMIC_LOCAL_ONLY_DIR / "accord_c138_cache" / "pool_granite33_8b_nothink__P_rev_agree.jsonl"
WS_PATH = MIMIC_LOCAL_ONLY_DIR / "accord_supp" / "accord_weak_supervision_labels_decontam.csv"
C138_OUT_ROOT = MIMIC_LOCAL_ONLY_DIR / "accord_c138"
C130_SCORE_DIR = MIMIC_LOCAL_ONLY_DIR / "accord_c130_reproducible_student"
LOG_DIR = MIMIC_LOCAL_ONLY_DIR / "logs"
RESULTS_OUT = ROOT / "results" / "accord" / "c138_prevalence_matched" / "c138.json"
C109_ANALYSIS = ROOT / "results" / "accord" / "prompt_sensitivity" / "c109" / "analysis.json"
C130_JSON = ROOT / "results" / "accord" / "c130_reproducible_student" / "c130.json"

JOIN_KEY = ["HADM_ID", "drug", "rule", "nursing_sentence", "discharge_sentence"]


class AbortC(RuntimeError):
    """Outcome C: invalid run, no decision recorded."""


# ---------------------------------------------------------------------------
# Teacher scores, joined onto the training pool by pool row index.
# ---------------------------------------------------------------------------

def load_teacher_scores() -> pd.DataFrame:
    if not CACHE_PATH.is_file():
        raise AbortC(f"no pool teacher-score cache at {CACHE_PATH}; "
                     "run score_pool_granite.py first")
    recs = [json.loads(l) for l in CACHE_PATH.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(recs)
    if df["ws_row"].duplicated().any():
        raise AbortC("duplicate ws_row in the teacher-score cache")
    return df


def attach_teacher_to_pool(teacher: pd.DataFrame) -> pd.DataFrame:
    """Return the WS pool with ws_row and the teacher's columns attached."""
    ws = pd.read_csv(WS_PATH)
    ws["ws_row"] = np.arange(len(ws))
    ws["HADM_ID"] = ws["HADM_ID"].astype(float)
    ws["nursing_sentence"] = ws["nursing_sentence"].fillna("").astype(str)
    ws["discharge_sentence"] = ws["discharge_sentence"].fillna("").astype(str)
    merged = ws.merge(teacher, on="ws_row", how="left", validate="one_to_one")
    return merged


def gates(pool_with_teacher: pd.DataFrame, train_df: pd.DataFrame):
    """Every pre-registered kill condition, as a dict of booleans and counts."""
    g = collections.OrderedDict()

    g["K1_split_size"] = {
        "n_train": int(len(train_df)),
        "expected_n_train": EXPECTED_N_TRAIN,
        "passed": bool(len(train_df) == EXPECTED_N_TRAIN),
    }

    # K2: every training row gets exactly one teacher score, joined on the
    # full row key (not the row index, which train_df does not carry).
    tkeep = pool_with_teacher[JOIN_KEY + ["ws_row", "flag_label", "flag_logodds",
                                          "parse_failure", "call_failed", "degenerate"]]
    n_dup_key = int(tkeep.duplicated(subset=JOIN_KEY).sum())
    merged = train_df.merge(tkeep, on=JOIN_KEY, how="left")
    n_missing = int(merged["flag_logodds"].isna().sum())
    g["K2_join_completeness"] = {
        "n_train_rows": int(len(train_df)),
        "n_rows_after_join": int(len(merged)),
        "n_duplicate_join_key_rows_in_pool": n_dup_key,
        "n_train_rows_missing_a_teacher_score": n_missing,
        "passed": bool(len(merged) == len(train_df) and n_missing == 0 and n_dup_key == 0),
    }

    scored = pool_with_teacher[pool_with_teacher["flag_logodds"].notna()]
    n_parse = int(pool_with_teacher["parse_failure"].fillna(True).astype(bool).sum())
    rate = n_parse / len(pool_with_teacher)
    g["K3_parse_failure_rate"] = {
        "n_pool_rows": int(len(pool_with_teacher)),
        "n_parse_failures": n_parse,
        "parse_failure_rate": round(rate, 4),
        "max_allowed": MAX_PARSE_FAILURE_RATE,
        "passed": bool(rate <= MAX_PARSE_FAILURE_RATE),
    }

    distinct = int(merged["flag_logodds"].round(12).nunique())
    g["K4_graded_score_not_degenerate"] = {
        "n_distinct_logodds_on_train": distinct,
        "min_required": MIN_DISTINCT_LOGODDS,
        "passed": bool(distinct >= MIN_DISTINCT_LOGODDS),
    }
    return merged, g


def prevalence_matched_labels(merged: pd.DataFrame, k: float):
    """Top round(n*k) rows by flag_logodds labelled 1, ties at the boundary
    broken by ascending ws_row, exactly as pre-registered."""
    n = len(merged)
    n_pos = int(np.floor(n * k + 0.5))  # round-half-up, pre-registered
    order = merged.sort_values(["flag_logodds", "ws_row"],
                               ascending=[False, True]).index.to_numpy()
    y = np.zeros(n, dtype=int)
    pos_idx = order[:n_pos]
    y[merged.index.get_indexer(pos_idx)] = 1
    lo = merged.loc[order[n_pos - 1], "flag_logodds"] if n_pos > 0 else None
    n_tied_at_boundary = int((merged["flag_logodds"].round(12) == round(float(lo), 12)).sum()) if lo is not None else 0
    return y, {
        "k": k,
        "n_train": n,
        "n_pos_target": n_pos,
        "n_pos_realised": int(y.sum()),
        "positive_rate": float(y.mean()),
        "threshold_logodds": float(lo) if lo is not None else None,
        "n_rows_tied_at_threshold_value": n_tied_at_boundary,
    }


# ---------------------------------------------------------------------------
# Training: rebind train.py's build_splits, never edit train.py.
# ---------------------------------------------------------------------------

def make_overridden_build_splits(y_by_key: pd.DataFrame):
    from data import build_splits as _orig, Splits

    def _build():
        splits = _orig()
        m = splits.train.merge(y_by_key, on=JOIN_KEY, how="left")
        if len(m) != len(splits.train) or m["y_c138"].isna().any():
            raise AbortC("label join failed inside the training wrapper")
        m = m.drop(columns=["y"]).rename(columns={"y_c138": "y"})
        m["y"] = m["y"].astype(int)
        return Splits(train=m.reset_index(drop=True), val=splits.val, gold=splits.gold,
                      n_leaked_rows_excluded=splits.n_leaked_rows_excluded,
                      n_gold_subjects=splits.n_gold_subjects,
                      n_pool_subjects=splits.n_pool_subjects)
    return _build


def train_one_seed(seed: int, y_by_key: pd.DataFrame, out_dir: pathlib.Path):
    import train as train_mod
    out_dir = assert_local_only(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    orig = train_mod.build_splits
    train_mod.build_splits = make_overridden_build_splits(y_by_key)
    old_argv = sys.argv
    sys.argv = ["train.py", "--seed", str(seed), "--epochs", str(EPOCHS),
                "--out-dir", str(out_dir)]
    try:
        train_mod.main()
    finally:
        sys.argv = old_argv
        train_mod.build_splits = orig


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------

def block_metrics(y, scores):
    from report import bootstrap_auroc_ci
    y = np.asarray(y)
    n, pos = int(len(y)), int(y.sum())
    if pos == 0 or pos == n:
        return {"n": n, "n_pos": pos, "auroc": None, "ap": None, "bootstrap": None}
    return {"n": n, "n_pos": pos,
            "auroc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
            "bootstrap": bootstrap_auroc_ci(y, scores, n_boot=N_BOOT, seed=BOOT_SEED)}


def paired_diff(y, a, b, B=N_BOOT, seed=PAIRED_DIFF_SEED):
    """AUROC(a) - AUROC(b), paired bootstrap, identical construction to
    c130_run.py::paired_diff_bootstrap."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    while len(d) < B:
        idx = rng.integers(0, n, n)
        if 0 < y[idx].sum() < n:
            d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"diff": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "ci95_lo": lo, "ci95_hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def load_arm_scores(out_dir: pathlib.Path, ref_pair_ids):
    per_seed = {}
    for s in SEEDS:
        sp = out_dir / f"scores_seed{s}.csv"
        mp = out_dir / f"metrics_seed{s}.json"
        if not (sp.is_file() and mp.is_file()):
            raise AbortC(f"K6: {out_dir.name} seed {s} did not write both outputs")
        df = pd.read_csv(sp)
        if not (df["pair_id"].to_numpy() == ref_pair_ids).all():
            raise AbortC(f"pair_id order mismatch in {out_dir.name} seed {s}")
        per_seed[s] = df["score"].to_numpy()
    ens = np.vstack([per_seed[s] for s in SEEDS]).mean(axis=0)
    return per_seed, ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates-only", action="store_true")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = assert_local_only(LOG_DIR / f"c138_run_{time.strftime('%Y%m%dT%H%M%S')}.log")
    out = collections.OrderedDict()
    out["preregistration"] = "accord/c138_prevalence_matched/C138_PREREGISTRATION.md"
    out["teacher"] = {
        "model": "granite3.3:8b", "prompt": "P_rev_agree",
        "graded_score": "flag_logodds = -(logprob(Yes) - logprob(No)), higher = more conflict",
        "decoding": {"think": False, "temperature": 0, "num_predict": 1,
                     "logprobs": True, "top_logprobs": 8, "num_ctx": 4096,
                     "endpoint": "127.0.0.1/api/chat"},
        "generated_by": "accord/c138_prevalence_matched/score_pool_granite.py",
        "pool_graded_scores_existed_before": False,
        "gold_g253_auroc_of_this_config": json.loads(C109_ANALYSIS.read_text())[
            "per_model_prompt"]["granite3.3:8b"]["P_rev_agree"]["auroc_253"],
    }
    out["mde_auroc_253"] = MDE_AUROC_253

    with open(log_path, "a") as log_fh, contextlib.redirect_stdout(log_fh), \
            contextlib.redirect_stderr(log_fh):
        from data import build_splits
        splits = build_splits()
        teacher = load_teacher_scores()
        pool_t = attach_teacher_to_pool(teacher)
        merged, gate_report = gates(pool_t, splits.train)

    out["gates"] = gate_report
    failed = [k for k, v in gate_report.items() if not v["passed"]]
    if failed:
        out["outcome"] = "C"
        out["outcome_reason"] = f"kill condition triggered: {failed}"
        RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_OUT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({"outcome": "C", "failed_gates": failed}, indent=2))
        return

    # Arm label constructions.
    arms = collections.OrderedDict()
    raw_y = merged["flag_label"].astype(int).to_numpy()
    arms["RAW"] = (raw_y, {"construction": "teacher hard flag_label",
                           "n_train": int(len(raw_y)),
                           "n_pos_realised": int(raw_y.sum()),
                           "positive_rate": float(raw_y.mean())})
    for name, k in K_BY_ARM.items():
        y, info = prevalence_matched_labels(merged, k)
        info["construction"] = "top round(n*k) by flag_logodds, ties by ascending ws_row"
        arms[name] = (y, info)

    pm47_rate = arms["PM47"][1]["positive_rate"]
    k5 = abs(pm47_rate - GOLD_BASE_RATE_299) <= PM47_RATE_TOLERANCE
    out["gates"]["K5_pm47_rate_matches_pilot"] = {
        "realised_positive_rate": pm47_rate,
        "target": GOLD_BASE_RATE_299,
        "tolerance": PM47_RATE_TOLERANCE,
        "passed": bool(k5),
    }
    out["arms"] = {n: info for n, (y, info) in arms.items()}
    out["label_agreement"] = {
        n: {"n_positive_in_both_raw_and_this": int(((raw_y == 1) & (y == 1)).sum()),
            "n_positive_here_only": int(((raw_y == 0) & (y == 1)).sum())}
        for n, (y, info) in arms.items() if n != "RAW"
    }
    if not k5:
        out["outcome"] = "C"
        out["outcome_reason"] = "kill condition K5 triggered"
        RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_OUT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({"outcome": "C", "failed_gates": ["K5"]}, indent=2))
        return

    if args.gates_only:
        print(json.dumps({"gates": out["gates"], "arms": out["arms"],
                          "label_agreement": out["label_agreement"]}, indent=2))
        return

    # Train every arm, every seed.
    key_cols = JOIN_KEY
    for name, (y, info) in arms.items():
        y_by_key = merged[key_cols].copy()
        y_by_key["y_c138"] = y
        arm_dir = C138_OUT_ROOT / name
        for seed in SEEDS:
            if (arm_dir / f"metrics_seed{seed}.json").is_file():
                continue
            with open(log_path, "a") as fh, contextlib.redirect_stdout(fh), \
                    contextlib.redirect_stderr(fh):
                train_one_seed(seed, y_by_key, arm_dir)
            print(f"trained {name} seed {seed}", flush=True)

    # Evaluate.
    with open(log_path, "a") as log_fh, contextlib.redirect_stdout(log_fh), \
            contextlib.redirect_stderr(log_fh):
        from data import build_splits
        splits = build_splits()
    gold = splits.gold.reset_index(drop=True)
    ref_pair_ids = gold["pair_id"].to_numpy()
    y299 = (gold["pharm_label"] == "contradiction").astype(int).to_numpy()
    m253 = (gold["pharm_label"] != "ambiguous").to_numpy()

    arm_scores = {}
    for name in arms:
        per_seed, ens = load_arm_scores(C138_OUT_ROOT / name, ref_pair_ids)
        arm_scores[name] = (per_seed, ens)
        out["arms"][name]["g253"] = {
            "ensemble": block_metrics(y299[m253], ens[m253]),
            **{f"seed{s}": block_metrics(y299[m253], per_seed[s][m253]) for s in SEEDS},
        }
        out["arms"][name]["g299"] = {
            "ensemble": block_metrics(y299, ens),
            **{f"seed{s}": block_metrics(y299, per_seed[s]) for s in SEEDS},
        }

    # C130 reproducible-label student, paired on the same gold rows.
    c130_per_seed = {}
    for s in SEEDS:
        df = pd.read_csv(C130_SCORE_DIR / f"scores_seed{s}.csv")
        if not (df["pair_id"].to_numpy() == ref_pair_ids).all():
            raise AbortC("C130 scores_seed pair_id order does not match this run's gold order")
        c130_per_seed[s] = df["score"].to_numpy()
    c130_ens = np.vstack([c130_per_seed[s] for s in SEEDS]).mean(axis=0)
    c130_reg = json.loads(C130_JSON.read_text())["g253"]["ensemble"]
    c130_block = block_metrics(y299[m253], c130_ens[m253])
    out["reference_c130_reproducible_label_student"] = {
        "recomputed_here": c130_block,
        "registered": {"auroc": c130_reg["auroc"], "ci95": c130_reg["bootstrap"]["ci95"]},
        "matches_registered_4dp": bool(
            round(c130_block["auroc"], 4) == round(c130_reg["auroc"], 4)),
        "teacher": "llama3:8b hard replication labels, pool positive rate 0.9470",
    }

    yy = y299[m253]
    out["paired_comparisons_g253"] = {
        "P1_PM47_minus_c130": paired_diff(yy, arm_scores["PM47"][1][m253], c130_ens[m253]),
        "P2_PM47_minus_RAW": paired_diff(yy, arm_scores["PM47"][1][m253],
                                         arm_scores["RAW"][1][m253]),
        "PM56_minus_RAW": paired_diff(yy, arm_scores["PM56"][1][m253],
                                      arm_scores["RAW"][1][m253]),
        "PM95_minus_RAW": paired_diff(yy, arm_scores["PM95"][1][m253],
                                      arm_scores["RAW"][1][m253]),
        "RAW_minus_c130": paired_diff(yy, arm_scores["RAW"][1][m253], c130_ens[m253]),
    }

    p1 = out["paired_comparisons_g253"]["P1_PM47_minus_c130"]
    p2 = out["paired_comparisons_g253"]["P2_PM47_minus_RAW"]
    p1_ok = p1["ci95_lo"] > 0
    p2_ok = p2["ci95_lo"] > 0
    out["decision"] = {
        "P1_satisfied": bool(p1_ok), "P2_satisfied": bool(p2_ok),
        "rule": ("Outcome A iff both paired CIs exclude zero with a positive point "
                 "estimate; otherwise Outcome B"),
    }
    out["outcome"] = "A" if (p1_ok and p2_ok) else "B"
    out["outcome_reason"] = (
        "prevalence-matched labels beat both the reproducible-label student and the "
        "same-teacher hard-label student, each with a paired 95% CI excluding zero"
        if out["outcome"] == "A" else
        "at least one pre-registered paired 95% CI covers zero; no recoverable "
        "benefit from prevalence-matched thresholding is demonstrated at this power"
    )
    out["power_note"] = (
        f"G253 carries 12 positives against 241 negatives. The corrected analytic "
        f"minimum detectable AUROC at alpha 0.05 and 80 percent power is "
        f"{MDE_AUROC_253} (results/accord/audit_c137/a6_delong_mde.json::mde_alt_r4), "
        f"not the 0.7355 the thesis prints. Outcome B here rules out a large "
        f"recoverable benefit, not a modest one."
    )
    out["double_use_disclosure"] = (
        "k = 0.040134 is the base rate of the same gold set the students are scored "
        "on. The circularity is real and declared in the pre-registration: k is a "
        "single scalar, the marginal, and carries no per-row information about which "
        "gold pairs are positive, and it is the number a practitioner would hold "
        "from an annotated pilot. It is not eliminated. A k estimated from a "
        "held-out half of G253 would rest on about 6 positives, a 95 percent "
        "interval of roughly 0.006 to 0.075; the PM47/PM56/PM95 ladder brackets "
        "that interval instead, so the conclusion can be checked for dependence on "
        "k without relying on one noisy split."
    )

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {RESULTS_OUT.relative_to(ROOT)}")
    print(json.dumps({
        "outcome": out["outcome"],
        "g253_auroc": {n: out["arms"][n]["g253"]["ensemble"]["auroc"] for n in arms},
        "c130_reference": c130_block["auroc"],
        "P1": p1, "P2": p2,
    }, indent=2))


if __name__ == "__main__":
    main()
