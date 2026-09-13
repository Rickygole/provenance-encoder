"""C101: power curves for a positive control at G_253's own size.
See ../C101_PREREGISTRATION.md. Requires roberta_scores.json /
roberta_scores_meta.json from score_roberta.py to exist first.

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

META = ROOT / "results" / "accord" / "public_controls" / "sizectl" / "roberta_scores_meta.json"
SCORES = ROOT / "results" / "accord" / "public_controls" / "sizectl" / "roberta_scores.json"
OUT = ROOT / "results" / "accord" / "public_controls" / "sizectl" / "power_curves.json"
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
    replicates (this is a validity check, not a power-curve data point)."""
    n = POP_CHECK_N
    if kind == "label_mix":
        lam = params["lambda"]
        from_neg = rng.random(n) < min(max(lam, 0.0), 1.0)
        pos_side = np.where(from_neg, rng.choice(neg_pool, n, True), rng.choice(pos_pool, n, True))
        neg_side = rng.choice(neg_pool, n, True)
    else:
        sigma = params["sigma"]
        pos_side = rng.choice(pos_z, n, True) + rng.normal(0, sigma, n)
        neg_side = rng.choice(neg_z, n, True) + rng.normal(0, sigma, n)
    y = np.r_[np.ones(n, int), np.zeros(n, int)]
    s = np.r_[pos_side, neg_side]
    return float(roc_auc_score(y, s))


def run_construction(kind, pos_pool, neg_pool, pos_z, neg_z, a0, master_seed):
    seed_seq = np.random.SeedSequence(master_seed)
    grid_results = []
    validity = {"rejection_rate_at_0_50": {}, "population_auroc_off_target": {}}

    for target in GRID:
        cell_seed_seq = seed_seq.spawn(1)[0]
        rng_setup = np.random.default_rng(cell_seed_seq)

        if kind == "label_mix":
            lam = ev.label_mix_lambda(a0, target)
            params = {"lambda": lam}
        else:
            sigma, achieved = ev.bisect_sigma_for_target(pos_z, neg_z, target,
                                                          seed=int(cell_seed_seq.generate_state(1)[0]))
            params = {"sigma": sigma, "bisection_achieved": achieved}

        pop_check_rng = np.random.default_rng(cell_seed_seq.spawn(1)[0])
        pop_auroc = check_population_auroc(kind, target, params, pos_pool, neg_pool,
                                           pos_z, neg_z, pop_check_rng)
        off_target = abs(pop_auroc - target)
        validity["population_auroc_off_target"][f"{target}"] = round(off_target, 5)

        fire_rates = {}
        rule_seeds = seed_seq.spawn(len(ALL_RULES))
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
    label_mix = run_construction("label_mix", pos_pool, neg_pool, pos_z, neg_z, a0, MASTER_SEED)
    noise_mix = run_construction("noise_mix", pos_pool, neg_pool, pos_z, neg_z, a0,
                                 MASTER_SEED + 1)

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
        "preregistration": "accord/public_controls/C101_PREREGISTRATION.md",
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
