"""C102 Stage 2: latent score, marginal/feasibility and bootstrap checks.
See ../C102_PREREGISTRATION.md.

Pure simulation. No MIMIC data, no note text. Reuses
`accord/stats/marginal_feasibility_gate.py` (attainable_floor_p, imported
without side effects) and `accord/public_controls/evaluation.py`'s pinned
bootstrap and rule functions.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy import stats as sp_stats

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accord.public_controls import evaluation as ev  # noqa: E402

OUT = ROOT / "results" / "accord" / "plantbench" / "stage2_s2abc.json"

SEED = 42
N = 253
N_POS = 12
TARGET_AUROC = 0.90
FLAG_RATE = 0.85
FLIP_PROB_CLEAN = 0.10
N_REPLICATES = 20
ALPHA = 0.05


def mu_for_auroc(auroc: float) -> float:
    """Equal-variance binormal mean shift for a target population AUROC,
    same construction as accord/stats/injection_sensitivity.py."""
    return float(np.sqrt(2.0) * sp_stats.norm.ppf(auroc))


def draw_latent(rng):
    y = (rng.random(N) < (N_POS / N)).astype(int)
    delta = mu_for_auroc(TARGET_AUROC)
    s = y * delta + rng.normal(0, 1, N)
    return y, s


def arm_clean(y, rng):
    """Symmetric flips: each label independently redrawn from a fresh
    Bernoulli(base rate) with probability FLIP_PROB_CLEAN, so the expected
    marginal is unchanged and the corruption does not depend on the label's
    current value."""
    fresh = (rng.random(N) < (N_POS / N)).astype(int)
    redraw = rng.random(N) < FLIP_PROB_CLEAN
    return np.where(redraw, fresh, y)


def arm_l1_top_flagged(s):
    """Flag the top FLAG_RATE fraction of candidates by score."""
    n_flag = round(N * FLAG_RATE)
    order = np.argsort(-s)
    y_obs = np.zeros(N, dtype=int)
    y_obs[order[:n_flag]] = 1
    return y_obs, n_flag


def arm_l2_random_flagged(rng):
    """Flag a uniformly random FLAG_RATE fraction, independent of s or y."""
    n_flag = round(N * FLAG_RATE)
    order = rng.permutation(N)
    y_obs = np.zeros(N, dtype=int)
    y_obs[order[:n_flag]] = 1
    return y_obs, n_flag


def s2a_marginal_floor(n_flag):
    return ev.attainable_floor_p(N, N_POS, n_flag)


def s2b_bootstrap_ratio_excludes_2(y_obs, seed, B=10_000):
    """Percentile bootstrap of (observed positive rate) / (base rate); fires
    if the lower 2.5th percentile exceeds 2.0.

    DEVIATION FROM THE COMMITTED PRE-REGISTRATION TEXT, disclosed rather than
    silently fixed: C102_PREREGISTRATION.md describes this as "stratified
    percentile bootstrap (the same pinned reimplementation as C101's
    bootstrap_lower_excludes_chance)". That reimplementation bootstraps an
    AUROC from a (y, s) pair by resampling the positive and negative strata
    separately; S2b's quantity is a single proportion (one binary array, no
    second array to stratify against), so "the same reimplementation" does
    not apply literally. What is implemented here -- and is the only reading
    that computes the stated quantity at all -- is a plain (non-stratified)
    percentile bootstrap of the count, same B=10,000, same fixed-seed
    `default_rng`. The fire condition, threshold (2.0) and B are exactly as
    committed; only the bootstrap's stratification mechanism differs from
    the parenthetical's (inapplicable) description."""
    rng = np.random.default_rng(seed)
    n = len(y_obs)
    base_rate = N_POS / N
    ratios = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        ratios[b] = y_obs[i].mean() / base_rate
    lo = float(np.percentile(ratios, 2.5))
    return lo, bool(lo > 2.0)


def s2c_c98_on_score(y_obs, s):
    return ev.rule_c98(y_obs, s)


def run_arm(name, y_obs, s, n_flag, seed):
    floor = s2a_marginal_floor(n_flag)
    lo_ratio, s2b_fires = s2b_bootstrap_ratio_excludes_2(y_obs, seed)
    s2c_fires = s2c_c98_on_score(y_obs, s)
    return {
        "arm": name, "n_flagged": int(n_flag), "flag_rate": round(n_flag / N, 4),
        "s2a_marginal_floor_p": floor, "s2a_floor_exceeds_0_05": bool(floor > 0.05),
        "s2b_bootstrap_lower_ratio": lo_ratio, "s2b_fires": s2b_fires,
        "s2c_c98_fires": s2c_fires,
    }


def main():
    seed_seq = np.random.SeedSequence(SEED)
    children = seed_seq.spawn(N_REPLICATES)

    per_arm_reps = {"clean": [], "L1": [], "L2": []}
    for rep_seed in children:
        rng = np.random.default_rng(rep_seed)
        y, s = draw_latent(rng)

        y_clean = arm_clean(y, rng)
        n_flag_clean = int(y_clean.sum())
        per_arm_reps["clean"].append(run_arm("clean", y_clean, s, n_flag_clean, rep_seed))

        y_l1, n_flag_l1 = arm_l1_top_flagged(s)
        per_arm_reps["L1"].append(run_arm("L1_top_flagged", y_l1, s, n_flag_l1, rep_seed))

        y_l2, n_flag_l2 = arm_l2_random_flagged(rng)
        per_arm_reps["L2"].append(run_arm("L2_random_flagged", y_l2, s, n_flag_l2, rep_seed))

    summary = {}
    for arm, reps in per_arm_reps.items():
        summary[arm] = {
            "n_replicates": len(reps),
            "s2a_floor_mean": float(np.mean([r["s2a_marginal_floor_p"] for r in reps])),
            "s2a_floor_exceeds_0_05_count": int(sum(r["s2a_floor_exceeds_0_05"] for r in reps)),
            "s2b_fire_count": int(sum(r["s2b_fires"] for r in reps)),
            "s2c_fire_count": int(sum(r["s2c_c98_fires"] for r in reps)),
        }

    def verdict_for(check_key, plant_arms, clean_arms):
        clean_fire_counts = [summary[a][check_key] for a in clean_arms]
        if any(c > 5 for c in clean_fire_counts):
            return "unvalidated_false_alarm"
        plant_fire_counts = [summary[a][check_key] for a in plant_arms]
        best = max(plant_fire_counts) if plant_fire_counts else 0
        if best >= 18:
            if all(c <= 2 for c in clean_fire_counts):
                return "validated"
            return "specificity_marginal"
        if any(3 <= c <= 5 for c in clean_fire_counts):
            return "specificity_marginal"
        return "not_validated_sensitivity"

    # The C102 pre-registration states S2c's expected pattern explicitly
    # ("should fire on clean and L1, not L2"), so its plant/clean arms are
    # unambiguous. It does NOT state which arms are plant vs clean for S2b
    # (only the fire condition itself: lower bootstrap bound of marginal /
    # base-rate > 2). That gap was not noticed until after this stage had
    # already been run once; per the kill condition, the committed rule text
    # is NOT edited post hoc to add a reading that was not there before the
    # first datum. Both readings consistent with the committed text are
    # reported below, neither chosen as "the" verdict:
    #   reading 1: plant = over-called arms (L1, L2, both flag 85%),
    #              clean = the near-base-rate "clean" arm.
    #   reading 2: plant = arms with real signal (clean, L1), matching S2c's
    #              own mapping, clean = L2.
    verdicts = {
        "S2a": "deterministic (reported as a per-arm floor, not a fire-count verdict)",
        "S2b_reading_over_calling_plant_L1_L2_clean_arm_clean":
            verdict_for("s2b_fire_count", plant_arms=["L1", "L2"], clean_arms=["clean"]),
        "S2b_reading_signal_plant_clean_L1_clean_arm_L2":
            verdict_for("s2b_fire_count", plant_arms=["clean", "L1"], clean_arms=["L2"]),
        "S2c": verdict_for("s2c_fire_count", plant_arms=["clean", "L1"], clean_arms=["L2"]),
    }

    out = {
        "preregistration": "accord/public_controls/C102_PREREGISTRATION.md",
        "stage": "2", "checks": ["S2a", "S2b", "S2c"],
        "target_population_auroc": TARGET_AUROC, "base_rate": N_POS / N,
        "flag_rate": FLAG_RATE, "n_replicates": N_REPLICATES, "seed": SEED,
        "disclosed_deviation_s2b_bootstrap_mechanism": (
            "committed text calls this a stratified bootstrap reusing C101's "
            "AUROC-bootstrap function; S2b's quantity is a single proportion "
            "with no second array to stratify against, so a plain percentile "
            "bootstrap of the count is what is implemented (same B, same "
            "threshold, same fire condition). See s2b_bootstrap_ratio_excludes_2's "
            "docstring."),
        "disclosed_ambiguity_s2b_plant_clean_mapping": (
            "the committed pre-registration states S2b's fire condition but "
            "not which arms are its plant vs its clean arm; both readings "
            "consistent with the committed text are reported under "
            "verdicts.S2b_reading_*, neither chosen as the verdict."),
        "note_s2c_design": (
            "S2c is designed to fire on clean and L1 (both retain real signal "
            "between the score and the observed label) and not fire on L2 "
            "(the observed label is independent random noise there); clean "
            "and L1 are therefore each read as this check's PLANT-side arms "
            "(it should fire), and L2 as its CLEAN arm (it should not)."),
        "per_arm_summary": summary,
        "per_replicate": per_arm_reps,
        "verdicts": verdicts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"per_arm_summary": summary, "verdicts": verdicts}, indent=2))


if __name__ == "__main__":
    main()
