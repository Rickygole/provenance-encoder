"""C102 Stage 4: large synthetic gold, real detector scores.
See ../C102_PREREGISTRATION.md.

Runs once Stage 3's output exists (`stage3.stage3_to_stage4_bridge`); before
that, `main()` checks for the artifact and exits without fabricating an
input. Stage 3's output in this build is the real, frozen
AuthorityTemporalConflictGraph + GuidelineGroundedHarmEstimator pipeline's
raw score on synthetic inputs, UNTRAINED (no training is in this build's
scope) -- so `y` (the 12/253 synthetic gold) and `s` are independent by
construction, and a null Stage 4 result is the expected, calibration-
appropriate finding, not evidence anything is broken.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accord.public_controls import evaluation as ev  # noqa: E402

# Where Stage 3's full (MPS) run is expected to write the trained detector's
# scores against the Stage 4 synthetic gold, once it exists. Not produced by
# this build.
STAGE3_DETECTOR_SCORES = (ROOT / "results" / "accord" / "plantbench"
                          / "stage3_detector_scores_for_stage4.json")
OUT = ROOT / "results" / "accord" / "plantbench" / "stage4_s4.json"

N = 253
N_POS = 12
N_RESAMPLES = 1000
SEED = 42


def s4_c98_power(y, s, plant_seed_seq, clean_seed_seq, n_resamples=N_RESAMPLES):
    """`c98_rule` on resample 0, plus the fire rate over `n_resamples`
    resamples, on BOTH arms:

    PLANT: the real (y, s) pairing, unpermuted. resample 0 is that original
    pairing itself; the 1,000-resample table is a case (row) bootstrap --
    resampling the 253 (y_i, s_i) pairs together, WITH replacement, keeping
    each label matched to its own score -- characterizing how the plant
    reading's fire/no-fire holds up under resampling variation.

    CLEAN: `s` fixed, `y` independently permuted (breaking the pairing).
    resample 0 is the first permutation drawn from `clean_seed_seq`; the
    1,000-resample table is the fire rate over 1,000 independent
    permutations, characterizing the null (no true relationship) rate.
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    n = len(y)

    plant_resample_0_fires = bool(ev.rule_c98(y, s))
    rng_plant = np.random.default_rng(plant_seed_seq)
    plant_fires = []
    for _ in range(n_resamples):
        i = rng_plant.integers(0, n, n)
        plant_fires.append(ev.rule_c98(y[i], s[i]))

    rng_clean = np.random.default_rng(clean_seed_seq)
    clean_fires = []
    clean_resample_0_fires = None
    for rep in range(n_resamples):
        y_perm = rng_clean.permutation(y)
        fired = ev.rule_c98(y_perm, s)
        if rep == 0:
            clean_resample_0_fires = bool(fired)
        clean_fires.append(fired)

    return {
        "plant": {
            "resample_0_fires": plant_resample_0_fires,
            "fire_rate_over_1000_bootstrap_resamples": float(np.mean(plant_fires)),
            "fire_count_first_20_bootstrap_resamples": int(sum(plant_fires[:20])),
            "n_resamples": n_resamples,
        },
        "clean": {
            "resample_0_fires": clean_resample_0_fires,
            "fire_rate_over_1000_permutation_resamples": float(np.mean(clean_fires)),
            "fire_count_first_20_permutation_resamples": int(sum(clean_fires[:20])),
            "n_resamples": n_resamples,
        },
    }


def build_synthetic_gold(seed):
    """Large synthetic gold at 12/253, i.e. the SAME rate as G_253, used only
    to give Stage 4's permutation-based clean construction a population to
    permute; this is NOT G_253 itself."""
    rng = np.random.default_rng(seed)
    y = np.zeros(N, dtype=int)
    y[rng.choice(N, N_POS, replace=False)] = 1
    return y


def main():
    if not STAGE3_DETECTOR_SCORES.exists():
        print(f"Stage 4 needs Stage 3's trained-detector output at "
              f"{STAGE3_DETECTOR_SCORES}, which does not exist (Stage 3 in "
              "this build is a CPU smoke test only, not a trained run). "
              "Stage 4 is not run.")
        return
    payload = json.loads(STAGE3_DETECTOR_SCORES.read_text())
    y = np.asarray(payload["y"])
    s = np.asarray(payload["s"], dtype=float)
    assert len(y) == N and int(y.sum()) == N_POS, (
        "Stage 3's detector-score artifact is not at the pre-registered "
        "12/253 synthetic-gold size")
    root = np.random.SeedSequence(SEED)
    plant_seed_seq, clean_seed_seq = root.spawn(2)
    result = s4_c98_power(y, s, plant_seed_seq, clean_seed_seq)

    plant_fires_20 = result["plant"]["fire_count_first_20_bootstrap_resamples"]
    clean_fires_20 = result["clean"]["fire_count_first_20_permutation_resamples"]
    if clean_fires_20 > 5:
        verdict = "unvalidated_false_alarm"
    elif plant_fires_20 >= 18:
        verdict = "validated" if clean_fires_20 <= 2 else "specificity_marginal"
    elif 3 <= clean_fires_20 <= 5:
        verdict = "specificity_marginal"
    else:
        verdict = "not_validated_sensitivity"

    out = {"preregistration": "accord/public_controls/C102_PREREGISTRATION.md",
          "stage": "4", "check": "S4", "source_artifact": str(STAGE3_DETECTOR_SCORES.relative_to(ROOT)),
          "source_note": payload.get("note"),
          **result, "verdict": verdict}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
