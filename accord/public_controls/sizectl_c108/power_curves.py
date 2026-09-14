"""C108: power curves for a positive control at G_253's own size -- corrected
rerun of C101 after the disclosed calibration bug. See
../C108_PREREGISTRATION.md.

Byte-for-byte the same grid, rules, replicate counts, decision rule, outcome
thresholds and seed tree as C101's own
`accord/public_controls/sizectl/power_curves.py` (reused verbatim per the
C108 pre-registration). The ONLY difference from that script:

1. Input/output paths point at this rerun's own artifact tree
   (`results/accord/public_controls/sizectl_c108/`), never C101's.
2. The noise-mix sigma calibration call no longer passes a `seed` (the fixed
   `evaluation.bisect_sigma_for_target` is deterministic: no RNG, no Monte
   Carlo draw, exact full-pool computation) and its return value may be the
   `INFINITE_SIGMA` sentinel at target 0.50, which downstream code (replicate
   drawing, and this script's own held-out population-AUROC check) must
   handle explicitly rather than passing to `rng.normal(0, sigma, ...)`.

Pure simulation on top of the already-scored SNLI pool: no model load, no
network, no MIMIC. Reuses accord/public_controls/evaluation.py throughout.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accord.public_controls import evaluation as ev  # noqa: E402

META = ROOT / "results" / "accord" / "public_controls" / "sizectl_c108" / "roberta_scores_meta.json"
SCORES = ROOT / "results" / "accord" / "public_controls" / "sizectl_c108" / "roberta_scores.json"
OUT = ROOT / "results" / "accord" / "public_controls" / "sizectl_c108" / "power_curves.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"

GRID = [0.50, 0.60, 0.65, 0.70, 0.72, 0.7355, 0.75, 0.78, 0.80, 0.85, 0.90]
N_POS_SLOTS, N_NEG_SLOTS = 12, 241
N_REPS_CHEAP = 4000
N_REPS_BOOTSTRAP = 1000
POP_CHECK_N = 200_000  # held-out Monte Carlo check of achieved population AUROC
POP_CHECK_TOL = 0.005
MASTER_SEED = 42

RULES_CHEAP = ["mw_one_sided", "mw_two_sided", "c98_rule",
              "delong_single_one_sided", "delong_paired_vs_null"]
RULE_BOOTSTRAP = "bootstrap_lower_excludes_chance"
ALL_RULES = RULES_CHEAP + [RULE_BOOTSTRAP]


def fire(rule, y, s, extra_seed=None):
    if rule == "mw_one_sided":
        return ev.rule_mw_one_sided(y, s)
    if rule == "mw_two_sided":
        return ev.rule_mw_two_sided(y, s)
    if rule == "c98_rule":
        return ev.rule_c98(y, s)
    if rule == "delong_single_one_sided":
        return ev.rule_delong_single_one_sided(y, s)
    if rule == "delong_paired_vs_null":
        return ev.rule_delong_paired_vs_null(y, s, seed=extra_seed)
    if rule == RULE_BOOTSTRAP:
        return ev.rule_bootstrap_lower_excludes_chance(y, s, seed=extra_seed, B=10_000)
    raise ValueError(rule)


def load_pools():
    meta = json.loads(META.read_text())
    records = json.loads(SCORES.read_text())
    is_contra = np.array([r["is_contradiction"] for r in records.values()])
    scores = np.array([r["score"] for r in records.values()])
    pos_pool = scores[is_contra == 1]
    neg_pool = scores[is_contra == 0]
    a0 = float(roc_auc_score(is_contra, scores))
    assert abs(a0 - meta["A0_auroc"]) < 1e-3, "recomputed A0 does not match meta"
    mu = float(scores.mean())
    sd = float(scores.std())
    pos_z = (pos_pool - mu) / sd
    neg_z = (neg_pool - mu) / sd
    return meta, pos_pool, neg_pool, pos_z, neg_z, a0


def check_population_auroc(kind, target, params, pos_pool, neg_pool, pos_z, neg_z, rng):
    """Held-out large-N Monte Carlo estimate of the achieved population AUROC
    for this construction/target, NOT counted among the pre-registered
    replicates (this is a validity check, not a power-curve data point).
    Deliberately independent of the exact calibration machinery in
    `evaluation.py` -- this is a sampled cross-check, kept exactly as C101
    ran it, except for handling the `INFINITE_SIGMA` sentinel (target 0.50)
    by the same pure label-permutation realization
    `draw_noise_mix_replicate` uses, instead of passing `sigma=inf` to
    `rng.normal`."""
    n = POP_CHECK_N
    if kind == "label_mix":
        lam = params["lambda"]
        from_neg = rng.random(n) < min(max(lam, 0.0), 1.0)
        pos_side = np.where(from_neg, rng.choice(neg_pool, n, True), rng.choice(pos_pool, n, True))
        neg_side = rng.choice(neg_pool, n, True)
    else:
        sigma = params["sigma"]
        if sigma == ev.INFINITE_SIGMA:
            pooled = np.concatenate([pos_z, neg_z])
            pos_side = rng.choice(pooled, n, True)
            neg_side = rng.choice(pooled, n, True)
        else:
            pos_side = rng.choice(pos_z, n, True) + rng.normal(0, sigma, n)
            neg_side = rng.choice(neg_z, n, True) + rng.normal(0, sigma, n)
    y = np.r_[np.ones(n, int), np.zeros(n, int)]
    s = np.r_[pos_side, neg_side]
    return float(roc_auc_score(y, s))


def run_construction(kind, construction_seed_seq, pos_pool, neg_pool, pos_z, neg_z, a0):
    """`construction_seed_seq` is this construction's own node in the single
    shared `SeedSequence(42)` tree rooted in `main()` (spawned once per
    (rule, grid point, construction), per the C101 pre-registration, reused
    verbatim under C108): one `.spawn(len(GRID))` call here gives each grid
    point its own child, in GRID's fixed order, and each grid point then
    spawns its own pop-check and rule children in one further fixed-order
    call. The sigma calibration itself no longer consumes a seed (it is
    deterministic under the fix), so the seed tree structure for every
    OTHER draw (pop check, six rules) is unchanged from C101."""
    grid_seed_seqs = construction_seed_seq.spawn(len(GRID))
    grid_results = []
    validity = {"rejection_rate_at_0_50": {}, "population_auroc_off_target": {}}

    for target, cell_seed_seq in zip(GRID, grid_seed_seqs):
        # Fixed order, one call: [pop_check, rule_0, rule_1, ..., rule_last].
        cell_children = cell_seed_seq.spawn(1 + len(ALL_RULES))
        pop_check_seed_seq, rule_seeds = cell_children[0], cell_children[1:]

        if kind == "label_mix":
            lam = ev.label_mix_lambda(a0, target)
            params = {"lambda": lam}
        else:
            sigma, achieved = ev.bisect_sigma_for_target(pos_z, neg_z, target)
            params = {"sigma": sigma, "bisection_achieved": achieved}

        pop_check_rng = np.random.default_rng(pop_check_seed_seq)
        pop_auroc = check_population_auroc(kind, target, params, pos_pool, neg_pool,
                                           pos_z, neg_z, pop_check_rng)
        off_target = abs(pop_auroc - target)
        validity["population_auroc_off_target"][f"{target}"] = round(off_target, 5)

        fire_rates = {}
        for rule, rule_seed_seq in zip(ALL_RULES, rule_seeds):
            n_reps = N_REPS_BOOTSTRAP if rule == RULE_BOOTSTRAP else N_REPS_CHEAP
            rng = np.random.default_rng(rule_seed_seq)
            fires = 0
            for rep in range(n_reps):
                if kind == "label_mix":
                    y, s = ev.draw_label_mix_replicate(pos_pool, neg_pool, N_POS_SLOTS,
                                                       N_NEG_SLOTS, params["lambda"], rng)
                else:
                    y, s = ev.draw_noise_mix_replicate(pos_z, neg_z, N_POS_SLOTS, N_NEG_SLOTS,
                                                       params["sigma"], rng)
                extra_seed = int(rng.integers(0, 2**31 - 1))
                if fire(rule, y, s, extra_seed=extra_seed):
                    fires += 1
            fire_rates[rule] = fires / n_reps
            if target == 0.50:
                validity["rejection_rate_at_0_50"][rule] = fires / n_reps

        grid_results.append({"target_auroc": target, "params": params,
                             "population_auroc_check": pop_auroc,
                             "fire_rates": fire_rates})

    crossings = {}
    for rule in ALL_RULES:
        power = [g["fire_rates"][rule] for g in grid_results]
        crossings[rule] = ev.interpolate_crossing(GRID, power, target=0.80)

    return {"construction": kind, "grid": grid_results, "crossings_at_power_0_80": crossings,
           "validity": validity}


def main():
    if not META.exists() or not SCORES.exists():
        print("waiting on RoBERTa scores (score_roberta.py has not produced "
              f"{META} / {SCORES} yet); power_curves.py not run.")
        return

    t0 = time.time()
    meta, pos_pool, neg_pool, pos_z, neg_z, a0 = load_pools()

    valid = a0 >= 0.90
    # One shared SeedSequence(42) tree for the whole run, per the C101
    # pre-registration ("Seeds from numpy SeedSequence(42).spawn"), reused
    # verbatim under C108: construction is the first spawn dimension, in the
    # fixed order [label_mix, noise_mix].
    root_seed_seq = np.random.SeedSequence(MASTER_SEED)
    label_mix_seed_seq, noise_mix_seed_seq = root_seed_seq.spawn(2)
    label_mix = run_construction("label_mix", label_mix_seed_seq,
                                 pos_pool, neg_pool, pos_z, neg_z, a0)
    noise_mix = run_construction("noise_mix", noise_mix_seed_seq,
                                 pos_pool, neg_pool, pos_z, neg_z, a0)

    facts = json.loads(FACTS.read_text())
    mde_key = "mde_recomputed.gold_ambiguous_excluded.minimum_detectable_auc"
    mde = facts[mde_key]["value"]

    primary_crossing = label_mix["crossings_at_power_0_80"]["mw_two_sided"]
    rej_050 = label_mix["validity"]["rejection_rate_at_0_50"]["mw_two_sided"]
    off_targets = list(label_mix["validity"]["population_auroc_off_target"].values()) + \
        list(noise_mix["validity"]["population_auroc_off_target"].values())
    max_off_target = max(off_targets) if off_targets else None

    invalid = (not valid) or (rej_050 < 0.02 or rej_050 > 0.08) or \
        (max_off_target is not None and max_off_target > POP_CHECK_TOL)

    if invalid:
        outcome = "C"
    elif primary_crossing is not None and abs(primary_crossing - mde) <= 0.02 and \
            0.03 <= rej_050 <= 0.07:
        outcome = "A"
    else:
        outcome = "B"

    c98_at_mde = None
    for g in label_mix["grid"]:
        if abs(g["target_auroc"] - mde) < 1e-6 or abs(g["target_auroc"] - 0.7355) < 1e-6:
            c98_at_mde = g["fire_rates"]["c98_rule"]

    out = {
        "preregistration": "accord/public_controls/C108_PREREGISTRATION.md",
        "A0": a0, "mde_analytic": mde,
        "n_pos_slots": N_POS_SLOTS, "n_neg_slots": N_NEG_SLOTS,
        "n_reps_cheap_rules": N_REPS_CHEAP, "n_reps_bootstrap_rule": N_REPS_BOOTSTRAP,
        "primary_rule": "mw_two_sided", "primary_construction": "label_mix",
        "primary_crossing_power_0_80": primary_crossing,
        "primary_rejection_rate_at_0_50": rej_050,
        "c98_rule_rejection_rate_at_0_7355": c98_at_mde,
        "max_population_auroc_off_target": max_off_target,
        "outcome": outcome,
        "label_mix": label_mix, "noise_mix": noise_mix,
        "wall_clock_seconds": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("label_mix", "noise_mix")},
                     indent=2))


if __name__ == "__main__":
    main()
