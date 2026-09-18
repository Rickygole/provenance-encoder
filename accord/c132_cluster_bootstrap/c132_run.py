"""C132: cluster bootstrap of the two G_253 ensemble AUROC intervals the bounded
null rests on, resampling whole admissions and, separately, whole subjects,
instead of treating the 253 gold pairs as independent. See
C132_PREREGISTRATION.md, committed before this script's full mode was run.

No model is trained or scored. The per-row scores are the ones already on
disk: the decontaminated three-seed ensemble exactly as
accord/stats/recompute_decontam_aggregate.py loads it (same PRED path, same
mean of prob_seed0..2) and the reproducible-label retrain exactly as
accord/c130_reproducible_student/c130_run.py loads it (same C130_OUT_DIR, same
SEEDS, same mean of the three seeds' score column, same pair_id order checks).
Cluster identifiers come from accord.control.data.load_gold(), the project's
one pair_id <-> HADM_ID <-> SUBJECT_ID join. They are used only to group row
positions inside this process; no identifier, pair_id or per-row value is
written, printed or logged anywhere.

    python3 accord/c132_cluster_bootstrap/c132_run.py --dry-run
    python3 accord/c132_cluster_bootstrap/c132_run.py            # full run

Local only. No hosted API, no Ollama call. Everything that touches
~/mimic_local_only runs with stdout/stderr redirected to a file under
~/mimic_local_only/logs; that log is never read back. --dry-run checks that
every input exists and that the cluster join covers all 253 pairs, computes no
AUROC, no cluster count and no interval, writes nothing to results/, and
prints counts and booleans only. The full run writes exactly one aggregate
JSON, results/accord/c132_cluster_bootstrap/c132.json. register_c132.py adds
the c132.* register keys from that artifact.
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
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accord" / "control"))
sys.path.insert(0, str(ROOT / "accord" / "stats"))
sys.path.insert(0, str(ROOT / "accord" / "c130_reproducible_student"))

from accord.cloud_paths import assert_local_only  # noqa: E402
# The same path objects the two source scripts use, imported rather than
# retyped, so this run cannot drift onto a different per-row score file.
from recompute_decontam_aggregate import PRED as DECONTAM_PRED  # noqa: E402
from c130_run import C130_OUT_DIR, SEEDS as C130_SEEDS  # noqa: E402
from recompute_mde import hanley_mcneil_mde  # noqa: E402

MIMIC_LOCAL_ONLY_DIR = pathlib.Path(os.environ.get("MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
LOG_DIR = MIMIC_LOCAL_ONLY_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = assert_local_only(LOG_DIR / f"c132_run_{time.strftime('%Y%m%dT%H%M%S')}.log")
RESULTS_OUT = assert_local_only(ROOT / "results" / "accord" / "c132_cluster_bootstrap" / "c132.json")
FACTS_PATH = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"

N_BOOT = 10_000
CLUSTER_SEED = 20260806   # accord/control/report.py::BOOT_SEED, the project's bootstrap seed
ROWCHECK_SEED = 42        # the seed both registered row-level intervals were recorded with
MAX_ATTEMPT_FACTOR = 20   # as accord/control/report.py::bootstrap_auroc_ci

N_G253, N_POS_G253, N_NEG_G253 = 253, 12, 241
N_G299, N_POS_G299_AMBIG_POS, N_NEG_G299_AMBIG_POS = 299, 58, 241

# Registered values this check is read against, with the 4 dp figure the
# pre-registration quotes. The register must still hold them at run time.
REGISTERED = collections.OrderedDict([
    ("power_floor", ("sizectl108.power_curves.primary_crossing_power_0_80", 0.7466)),
    ("decontam_ci95_lo", ("decontam.aggregate.ensemble_ci95_lo", 0.3210)),
    ("decontam_ci95_hi", ("decontam.aggregate.ensemble_ci95_hi", 0.7034)),
    ("decontam_auroc", ("decontam.aggregate.ensemble_auroc", 0.5145)),
    ("c130_ci95_lo", ("c130.g253.ensemble.ci95_lo", 0.3454)),
    ("c130_ci95_hi", ("c130.g253.ensemble.ci95_hi", 0.7196)),
    ("c130_auroc", ("c130.g253.ensemble.auroc", 0.5360)),
])

# Task C: post-flip class counts of the C114 tipping-point check (5, 6 and 7
# positive-to-negative flips from 12 positives and 241 negatives).
TASK_C_COUNTS = ((7, 246), (6, 247), (5, 248))


class AbortC(RuntimeError):
    """Raised to signal Outcome C (invalid run): no decision is recorded.
    Messages are static text and counts only, never a row value."""


# ---------------------------------------------------------------------------
# Inputs and gates. Everything here runs under the log redirect.
# ---------------------------------------------------------------------------

def input_paths() -> collections.OrderedDict:
    import data as control_data
    paths = collections.OrderedDict()
    paths["decontam_predictions"] = DECONTAM_PRED
    for s in C130_SEEDS:
        paths[f"c130_scores_seed{s}"] = C130_OUT_DIR / f"scores_seed{s}.csv"
    paths["gold_results"] = pathlib.Path(control_data.GOLD_PATH)
    paths["annotation_sample"] = pathlib.Path(control_data.SAMP_PATH)
    paths["decontamination_audit"] = pathlib.Path(control_data.AUDIT_PATH)
    paths["medication_mentions"] = pathlib.Path(control_data.MENTIONS_PATH)
    paths["verified_facts"] = FACTS_PATH
    return paths


def load_registered() -> dict:
    facts = json.loads(FACTS_PATH.read_text())
    out = {}
    for name, (key, quoted) in REGISTERED.items():
        if key not in facts:
            raise AbortC(f"register key missing: {key}")
        value = float(facts[key]["value"])
        if round(value, 4) != round(quoted, 4):
            raise AbortC(f"register key {key} no longer rounds to the pre-registered {quoted:.4f}")
        out[name] = value
    return out


def load_inputs():
    """Returns (ens_i, ens_ii, g299_ii, gate). ens_* are dicts of aligned numpy
    arrays (y, s, hadm, subj) on the 253 non-ambiguous pairs, each in its own
    source file's row order; g299_ii is the retrain ensemble on all 299 pairs
    with the pharmacist label kept. gate holds counts and booleans only."""
    from data import load_gold
    gate = collections.OrderedDict()

    gold = load_gold().reset_index(drop=True)
    gate["n_gold_rows"] = int(len(gold))
    gate["gold_pair_id_unique"] = bool(gold["pair_id"].is_unique)
    gate["n_gold_rows_missing_admission_id"] = int(gold["HADM_ID"].isna().sum())
    gate["n_gold_rows_missing_subject_id"] = int(gold["SUBJECT_ID"].isna().sum())
    non_ambiguous = gold[gold["pharm_label"] != "ambiguous"]
    gate["n_gold_non_ambiguous"] = int(len(non_ambiguous))
    gate["n_gold_ambiguous"] = int(len(gold) - len(non_ambiguous))
    if len(gold) != N_G299 or not gate["gold_pair_id_unique"]:
        raise AbortC("gold join is not 299 unique pairs")

    # (i) decontaminated subject-disjoint ensemble, loaded as recompute_decontam_aggregate.py does.
    dec = pd.read_csv(DECONTAM_PRED)
    gate["decontam_n_rows"] = int(len(dec))
    gate["decontam_pair_id_unique"] = bool(dec["pair_id"].is_unique)
    ids = gold[["pair_id", "HADM_ID", "SUBJECT_ID", "pharm_label", "pharm_pos"]]
    m = dec[["pair_id", "y_true"]].merge(ids, on="pair_id", how="left")
    gate["decontam_n_rows_after_join"] = int(len(m))
    gate["decontam_n_rows_missing_cluster_id"] = int((m["HADM_ID"].isna() | m["SUBJECT_ID"].isna()).sum())
    gate["decontam_n_label_disagreements"] = int((m["y_true"].astype(int) != m["pharm_pos"].fillna(-1).astype(int)).sum())
    gate["decontam_n_ambiguous_rows"] = int((m["pharm_label"] == "ambiguous").sum())
    y_i = dec["y_true"].to_numpy().astype(int)
    seeds_i = {k: dec[f"prob_seed{k}"].to_numpy() for k in range(3)}
    s_i = np.mean([seeds_i[k] for k in range(3)], axis=0)
    gate["decontam_n_pos"] = int(y_i.sum())
    gate["decontam_n_neg"] = int(len(y_i) - y_i.sum())
    ens_i = {"y": y_i, "s": s_i, "hadm": m["HADM_ID"].to_numpy(), "subj": m["SUBJECT_ID"].to_numpy()}

    # (ii) reproducible-label retrain ensemble, loaded as c130_run.py::run_full does.
    seed_scores = {s: pd.read_csv(C130_OUT_DIR / f"scores_seed{s}.csv") for s in C130_SEEDS}
    ref = seed_scores[C130_SEEDS[0]][["pair_id", "rule", "pharm_pos"]].reset_index(drop=True)
    gate["c130_n_rows"] = int(len(ref))
    same_order = all(
        len(seed_scores[s]) == len(ref)
        and bool((seed_scores[s]["pair_id"].to_numpy() == ref["pair_id"].to_numpy()).all())
        for s in C130_SEEDS
    )
    gate["c130_pair_id_order_same_across_seeds"] = bool(same_order)
    gold_order = len(ref) == len(gold) and bool((gold["pair_id"].to_numpy() == ref["pair_id"].to_numpy()).all())
    gate["c130_pair_id_order_matches_gold"] = bool(gold_order)
    if not (same_order and gold_order):
        raise AbortC("c130 score files do not share the gold row order")
    gate["c130_n_label_disagreements"] = int((ref["pharm_pos"].to_numpy().astype(int) != gold["pharm_pos"].to_numpy().astype(int)).sum())
    ensemble_ii = np.vstack([seed_scores[s]["score"].to_numpy() for s in C130_SEEDS]).mean(axis=0)
    g253_mask = (gold["pharm_label"] != "ambiguous").to_numpy()
    y299 = ref["pharm_pos"].to_numpy().astype(int)
    hadm299, subj299 = gold["HADM_ID"].to_numpy(), gold["SUBJECT_ID"].to_numpy()
    ens_ii = {"y": y299[g253_mask], "s": ensemble_ii[g253_mask], "hadm": hadm299[g253_mask], "subj": subj299[g253_mask]}
    gate["c130_g253_n_rows"] = int(g253_mask.sum())
    gate["c130_g253_n_pos"] = int(ens_ii["y"].sum())
    gate["c130_g253_n_neg"] = int(g253_mask.sum() - ens_ii["y"].sum())
    gate["c130_g253_n_rows_missing_cluster_id"] = int((pd.isna(ens_ii["hadm"]) | pd.isna(ens_ii["subj"])).sum())

    # Task D reading: the 46 ambiguous pairs counted as POSITIVE.
    y299_ambig_pos = gold["pharm_label"].isin(["contradiction", "ambiguous"]).to_numpy().astype(int)
    g299_ii = {"y": y299_ambig_pos, "s": ensemble_ii, "hadm": hadm299, "subj": subj299}
    gate["task_d_n_rows"] = int(len(y299_ambig_pos))
    gate["task_d_n_pos"] = int(y299_ambig_pos.sum())
    gate["task_d_n_neg"] = int(len(y299_ambig_pos) - y299_ambig_pos.sum())
    gate["task_d_n_rows_missing_subject_id"] = int(pd.isna(subj299).sum())
    # The decontaminated predictions file scores only the non-ambiguous pairs,
    # so ensemble (i) has no score for the 46 pairs Task D would relabel.
    gate["task_d_decontam_ensemble_scores_all_299"] = bool(len(dec) == N_G299)

    gate["same_253_pairs_in_both_ensembles"] = bool(
        set(dec["pair_id"].tolist()) == set(gold.loc[g253_mask, "pair_id"].tolist())
    )

    join_ok = (
        gate["decontam_n_rows"] == N_G253 and gate["decontam_pair_id_unique"]
        and gate["decontam_n_rows_after_join"] == N_G253
        and gate["decontam_n_rows_missing_cluster_id"] == 0
        and gate["decontam_n_label_disagreements"] == 0
        and gate["decontam_n_ambiguous_rows"] == 0
        and gate["decontam_n_pos"] == N_POS_G253 and gate["decontam_n_neg"] == N_NEG_G253
        and gate["c130_g253_n_rows"] == N_G253
        and gate["c130_g253_n_rows_missing_cluster_id"] == 0
        and gate["c130_n_label_disagreements"] == 0
        and gate["c130_g253_n_pos"] == N_POS_G253 and gate["c130_g253_n_neg"] == N_NEG_G253
        and gate["same_253_pairs_in_both_ensembles"]
    )
    gate["cluster_join_covers_all_253_pairs"] = bool(join_ok)
    gate["task_d_counts_as_preregistered"] = bool(
        gate["task_d_n_rows"] == N_G299 and gate["task_d_n_pos"] == N_POS_G299_AMBIG_POS
        and gate["task_d_n_neg"] == N_NEG_G299_AMBIG_POS and gate["task_d_n_rows_missing_subject_id"] == 0
    )
    return ens_i, ens_ii, g299_ii, gate


def run_dry_run() -> dict:
    paths = input_paths()
    exists = collections.OrderedDict((name, bool(p.is_file())) for name, p in paths.items())
    out = collections.OrderedDict([("inputs_exist", exists), ("all_inputs_exist", bool(all(exists.values())))])
    if not out["all_inputs_exist"]:
        out["overall"] = "BLOCKED_MISSING_INPUT"
        return out
    load_registered()
    out["registered_values_present_and_round_to_preregistered"] = True
    _, _, _, gate = load_inputs()
    out["gate"] = gate
    ready = gate["cluster_join_covers_all_253_pairs"] and gate["task_d_counts_as_preregistered"]
    out["overall"] = "READY_TO_RUN" if ready else "WOULD_ABORT_OUTCOME_C"
    return out


# ---------------------------------------------------------------------------
# The bootstrap. One function for the row-level and both cluster levels.
# ---------------------------------------------------------------------------

def cluster_bootstrap(y, s, groups, B=N_BOOT, seed=CLUSTER_SEED) -> dict:
    """Percentile bootstrap of the AUROC resampling whole clusters with
    replacement: each resample draws G cluster indices (G = number of distinct
    clusters) from numpy.random.default_rng(seed) and takes every row of each
    drawn cluster. Clusters are numbered by first appearance in row order. A
    resample with no positive or no negative is redrawn and counted.

    With one cluster per row (groups = arange(n)) the drawn cluster indices
    ARE the row indices, so this is draw-for-draw the row-level bootstrap of
    recompute_decontam_aggregate.py::boot_default_rng and
    accord/control/report.py::bootstrap_auroc_ci. The row-level reproduction
    check calls this same function that way."""
    y, s = np.asarray(y), np.asarray(s)
    codes, uniques = pd.factorize(np.asarray(groups), sort=False)
    G = len(uniques)
    members = [np.flatnonzero(codes == g) for g in range(G)]
    rng = np.random.default_rng(seed)
    aucs, n_redrawn, attempts = [], 0, 0
    while len(aucs) < B:
        if attempts >= B * MAX_ATTEMPT_FACTOR:
            raise AbortC("bootstrap attempt cap reached before B valid resamples")
        attempts += 1
        draw = rng.integers(0, G, G)
        idx = np.concatenate([members[g] for g in draw])
        pos = y[idx].sum()
        if pos == 0 or pos == len(idx):
            n_redrawn += 1
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return collections.OrderedDict([
        ("n_clusters", int(G)), ("seed", int(seed)), ("n_boot_valid", int(len(aucs))),
        ("n_redrawn_single_class", int(n_redrawn)), ("redraw_rate", float(n_redrawn / attempts)),
        ("ci95_lo", float(np.percentile(aucs, 2.5))), ("ci95_hi", float(np.percentile(aucs, 97.5))),
    ])


def cluster_size_summary(groups) -> dict:
    sizes = pd.Series(np.asarray(groups)).value_counts().to_numpy()
    return collections.OrderedDict([
        ("n_clusters", int(len(sizes))), ("max_pairs_per_cluster", int(sizes.max())),
        ("n_clusters_with_more_than_one_pair", int((sizes > 1).sum())),
    ])


def task_c() -> dict:
    out = collections.OrderedDict([("alpha", 0.05), ("power", 0.80), ("method", "accord/stats/recompute_mde.py::hanley_mcneil_mde")])
    for n_pos, n_neg in TASK_C_COUNTS:
        mde = round(hanley_mcneil_mde(n_pos, n_neg), 4)
        out[f"pos{n_pos}_neg{n_neg}"] = collections.OrderedDict([
            ("n_pos", n_pos), ("n_neg", n_neg), ("minimum_detectable_auc", mde), ("gap_above_chance", round(mde - 0.5, 4)),
        ])
    return out


def run_full() -> dict:
    registered = load_registered()
    missing = [name for name, p in input_paths().items() if not p.is_file()]
    if missing:
        raise AbortC(f"input file(s) missing: {', '.join(missing)}")
    ens_i, ens_ii, g299_ii, gate = load_inputs()
    if not gate["cluster_join_covers_all_253_pairs"]:
        raise AbortC("cluster identifiers could not be joined for every one of the 253 pairs")
    if not gate["task_d_counts_as_preregistered"]:
        raise AbortC("Task D class counts are not the pre-registered 58 positives and 241 negatives on 299 pairs")

    # Row-level reproduction first. Nothing clustered is computed unless both pass.
    blocks = collections.OrderedDict()
    for name, ens, reg_prefix in (("decontam_ensemble", ens_i, "decontam"), ("c130_ensemble", ens_ii, "c130")):
        n = len(ens["y"])
        row = cluster_bootstrap(ens["y"], ens["s"], np.arange(n), seed=ROWCHECK_SEED)
        reg_lo, reg_hi = registered[f"{reg_prefix}_ci95_lo"], registered[f"{reg_prefix}_ci95_hi"]
        row["registered_ci95_lo"], row["registered_ci95_hi"] = reg_lo, reg_hi
        row["reproduces_registered_4dp"] = bool(
            round(row["ci95_lo"], 4) == round(reg_lo, 4) and round(row["ci95_hi"], 4) == round(reg_hi, 4)
        )
        auroc = float(roc_auc_score(ens["y"], ens["s"]))
        blocks[name] = collections.OrderedDict([
            ("n", int(n)), ("n_pos", int(ens["y"].sum())), ("auroc", auroc),
            ("auroc_reproduces_registered_4dp", bool(round(auroc, 4) == round(registered[f"{reg_prefix}_auroc"], 4))),
            ("row_level_seed42_check", row),
        ])
    if not all(b["row_level_seed42_check"]["reproduces_registered_4dp"] for b in blocks.values()):
        raise AbortC("row-level check did not reproduce the registered intervals to four decimals")

    out = collections.OrderedDict()
    out["preregistration"] = "accord/c132_cluster_bootstrap/C132_PREREGISTRATION.md"
    out["n_boot"] = N_BOOT
    out["cluster_seed"] = CLUSTER_SEED
    out["row_check_seed"] = ROWCHECK_SEED
    out["power_floor"] = registered["power_floor"]

    out["task_a"] = collections.OrderedDict([
        ("n_pairs", N_G253), ("n_pos", N_POS_G253), ("n_neg", N_NEG_G253),
        ("admission", cluster_size_summary(ens_i["hadm"])), ("subject", cluster_size_summary(ens_i["subj"])),
    ])

    for name, ens in (("decontam_ensemble", ens_i), ("c130_ensemble", ens_ii)):
        n = len(ens["y"])
        blocks[name]["row_level_seed20260806"] = cluster_bootstrap(ens["y"], ens["s"], np.arange(n))
        blocks[name]["admission"] = cluster_bootstrap(ens["y"], ens["s"], ens["hadm"])
        blocks[name]["subject"] = cluster_bootstrap(ens["y"], ens["s"], ens["subj"])
    out["task_b"] = blocks

    floor = registered["power_floor"]
    uppers = collections.OrderedDict(
        (f"{name}_{level}", blocks[name][level]["ci95_hi"])
        for name in ("decontam_ensemble", "c130_ensemble") for level in ("admission", "subject")
    )
    at_or_above = [k for k, v in uppers.items() if v >= floor]
    out["decision"] = collections.OrderedDict([
        ("outcome", "B" if at_or_above else "A"),
        ("power_floor", floor),
        ("upper_ends", uppers),
        ("n_upper_ends_at_or_above_floor", len(at_or_above)),
        ("upper_ends_at_or_above_floor", at_or_above),
        ("max_upper_end", float(max(uppers.values()))),
        ("min_margin_below_floor", float(floor - max(uppers.values()))),
    ])

    out["task_c"] = task_c()

    n299 = len(g299_ii["y"])
    out["task_d"] = collections.OrderedDict([
        ("reading", "46 ambiguous pairs counted as positive; descriptive only"),
        ("c130_ensemble", collections.OrderedDict([
            ("n", int(n299)), ("n_pos", int(g299_ii["y"].sum())), ("n_neg", int(n299 - g299_ii["y"].sum())),
            ("auroc", float(roc_auc_score(g299_ii["y"], g299_ii["s"]))),
            ("row_level", cluster_bootstrap(g299_ii["y"], g299_ii["s"], np.arange(n299))),
            ("subject", cluster_bootstrap(g299_ii["y"], g299_ii["s"], g299_ii["subj"])),
        ])),
        ("decontam_ensemble", collections.OrderedDict([
            ("available", False),
            ("reason", "the decontaminated predictions file scores only the 253 non-ambiguous pairs; "
                       "no score exists for the 46 ambiguous pairs and none is substituted"),
        ])),
    ])
    out["gate"] = gate
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    failure = None
    with LOG_PATH.open("w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
        print(f"c132_run starting {time.strftime('%Y-%m-%d %H:%M:%S')} dry_run={args.dry_run}")
        try:
            result = run_dry_run() if args.dry_run else run_full()
        except AbortC as e:
            print(f"ABORT (Outcome C): {e}")
            failure = ("C", str(e))
        except Exception as e:  # traceback goes to the local log only; real stderr gets the type name
            traceback.print_exc()
            failure = ("ERROR", type(e).__name__)
        print("c132_run finished")

    if failure and failure[0] == "ERROR":
        raise SystemExit(f"c132_run failed with {failure[1]}; traceback is in the local log (not read back): {LOG_PATH}")
    if failure:
        if not args.dry_run:
            RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
            RESULTS_OUT.write_text(json.dumps(
                collections.OrderedDict([("outcome", "C"), ("abort_reason", failure[1]), ("task_c", task_c())]), indent=2) + "\n")
        raise SystemExit(f"ABORT (Outcome C): {failure[1]}")

    if args.dry_run:
        print(json.dumps(result, indent=2))
        print(f"log: {LOG_PATH}")
        return

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {RESULTS_OUT}")
    print(f"log: {LOG_PATH}")
    print(json.dumps({"decision": result["decision"], "task_a": result["task_a"]}, indent=2))


if __name__ == "__main__":
    main()
