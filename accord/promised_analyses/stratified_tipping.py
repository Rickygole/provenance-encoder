"""The C133 interval tipping-point check recomputed under the
STRATIFIED bootstrap definition, the one the C108 power simulation uses.

bootstrap_definitions.py shows the stratified upper end sits 0.0097 to 0.0135 below the
unstratified one on G_253. The bounded null's own verdict survives that shift
with room to spare, but the C133 tipping counts do not obviously survive it:
the C130 ensemble's k = 1 upper end clears the 0.7466 floor by only 0.0125
under the unstratified rule, which is smaller than the shift. This script
re-runs the same greedy relabelling path, reusing C133's own greedy_step,
pair_kernel and score loader so the relabelled rows are identical, and reports
the tipping count under each convention.

Aggregates only. No row identity, no pair_id, no HADM_ID, no SUBJECT_ID, no
note text. No model inference.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accord" / "c132_cluster_bootstrap"))
sys.path.insert(0, str(ROOT / "accord" / "c133_interval_tipping"))

import c132_run as c132  # noqa: E402
from c133_run import DrawPool, auroc, greedy_step, pair_kernel, percentile_interval  # noqa: E402

OUT_DIR = ROOT / "results/accord/promised_analyses"
OUT = OUT_DIR / "stratified_tipping.json"

SEED, B, MAX_FLIPS = 42, 10_000, 8
FLOOR = 0.7466


def stratified_interval(y, K2, seed=SEED, B=B):
    """Class counts held fixed, the C108 power simulation's convention.

    Vectorised the same way C133's percentile_interval is: each resample is
    held as a vector of per-row multiplicities, and the exact 2U statistic of
    all B resamples is one pair of matrix products against the same pairwise
    kernel, so this differs from percentile_interval only in the draw rule.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    ip, ineg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    # The index stream is drawn replicate by replicate, positives then
    # negatives, the same order and the same two rng.choice calls per
    # replicate as bootstrap_definitions.py::boot_stratified and as
    # accord/audit_c137/a3_decontam.py::boot_ci(strat=True), so the k = 0
    # interval this returns is the one those two independently report.
    # Only the AUROC evaluation is batched.
    idx = np.empty((B, n), dtype=np.int64)
    for b in range(B):
        idx[b, :len(ip)] = rng.choice(ip, len(ip), True)
        idx[b, len(ip):] = rng.choice(ineg, len(ineg), True)
    flat = (idx + np.arange(B)[:, None] * n).ravel()
    cnt = np.bincount(flat, minlength=B * n).reshape(B, n).astype(float)
    c_pos, c_neg = cnt * y, cnt * (1.0 - y)
    u2 = ((c_pos @ K2) * c_neg).sum(axis=1)
    aucs = u2 / (2.0 * c_pos.sum(axis=1) * c_neg.sum(axis=1))
    return collections.OrderedDict([
        ("seed", seed), ("n_boot_valid", int(B)), ("n_redrawn_single_class", 0),
        ("ci95_lo", float(np.percentile(aucs, 2.5))),
        ("ci95_hi", float(np.percentile(aucs, 97.5))),
    ])


def main():
    ens_i, ens_ii, _, gate = c132.load_inputs()
    ensembles = collections.OrderedDict([("decontam_ensemble", ens_i), ("c130_ensemble", ens_ii)])
    kernels = {k: pair_kernel(v["s"]) for k, v in ensembles.items()}

    res = collections.OrderedDict()
    res["provenance"] = {
        "generated_by": "accord/promised_analyses/stratified_tipping.py",
        "reuses": "accord/c133_interval_tipping/c133_run.py greedy_step, pair_kernel, percentile_interval; accord/c132_cluster_bootstrap/c132_run.py load_inputs",
        "seed": SEED, "n_boot": B, "max_flips": MAX_FLIPS, "power_floor": FLOOR,
        "direction": "pos_to_neg, the direction C133 reports the tipping counts for",
        "no_model_inference": True,
    }

    for name, ens in ensembles.items():
        K2 = kernels[name]
        pool = DrawPool(len(ens["y"]))
        y = np.asarray(ens["y"]).astype(int).copy()
        # k = 0 validity, both conventions.
        blk = collections.OrderedDict()
        k0u = percentile_interval(y, K2, pool)
        k0s = stratified_interval(y, K2)
        blk["k0"] = {"auroc": float(auroc(y, K2)),
                     "unstratified_ci95_hi": k0u["ci95_hi"],
                     "stratified_ci95_hi": k0s["ci95_hi"]}
        steps = collections.OrderedDict()
        first_u = first_s = None
        for k in range(1, MAX_FLIPS + 1):
            y, n_tied = greedy_step(y, K2, "pos_to_neg")
            iu = percentile_interval(y, K2, pool)
            is_ = stratified_interval(y, K2)
            rec = collections.OrderedDict([
                ("k", k), ("n_pos", int(y.sum())), ("auroc", float(auroc(y, K2))),
                ("n_candidates_tied_for_best", n_tied),
                ("unstratified_ci95_hi", iu["ci95_hi"]),
                ("stratified_ci95_hi", is_["ci95_hi"]),
                ("shift_stratified_minus_unstratified", round(is_["ci95_hi"] - iu["ci95_hi"], 6)),
                ("unstratified_at_or_above_floor", bool(iu["ci95_hi"] >= FLOOR)),
                ("stratified_at_or_above_floor", bool(is_["ci95_hi"] >= FLOOR)),
            ])
            if first_u is None and rec["unstratified_at_or_above_floor"]:
                first_u = k
            if first_s is None and rec["stratified_at_or_above_floor"]:
                first_s = k
            steps[f"k{k}"] = rec
        blk["steps"] = steps
        blk["k_first_at_or_above_floor_unstratified"] = first_u
        blk["k_first_at_or_above_floor_stratified"] = first_s
        blk["tipping_count_changes_under_stratified_definition"] = bool(first_u != first_s)
        res[name] = blk

    # Validity: the k = 0 unstratified interval must reproduce the registered
    # row-level interval, and the k = 0 stratified interval must reproduce the
    # one bootstrap_definitions.py reports, or nothing below can be read.
    reg = json.loads((ROOT / "results/accord/c132_cluster_bootstrap/c132.json").read_text())["task_b"]
    bd = json.loads((ROOT / "results/accord/promised_analyses/bootstrap_definitions.json").read_text())
    res["reproduction_checks"] = {
        f"{name}_k0_unstratified_hi_matches_registered_4dp": bool(
            round(res[name]["k0"]["unstratified_ci95_hi"], 4)
            == round(reg[name]["row_level_seed42_check"]["ci95_hi"], 4))
        for name in ensembles
    }
    res["reproduction_checks"].update({
        f"{name}_k0_stratified_hi_matches_bootstrap_definitions_6dp": bool(
            round(res[name]["k0"]["stratified_ci95_hi"], 6)
            == round(bd[name]["stratified"]["ci95_hi"], 6))
        for name in ensembles
    })
    res["decision"] = collections.OrderedDict([
        ("unstratified_tipping_counts", {k: res[k]["k_first_at_or_above_floor_unstratified"] for k in ensembles}),
        ("stratified_tipping_counts", {k: res[k]["k_first_at_or_above_floor_stratified"] for k in ensembles}),
        ("any_tipping_count_changes", any(res[k]["tipping_count_changes_under_stratified_definition"] for k in ensembles)),
    ])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=1))


if __name__ == "__main__":
    main()
