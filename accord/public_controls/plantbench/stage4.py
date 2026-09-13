"""C102 Stage 4: large synthetic gold, real detector scores.
See ../C102_PREREGISTRATION.md.

NOT RUN by this build: needs Stage 3's trained-detector output, which does
not exist yet (Stage 3 here is a CPU smoke test only; the full, trained run
is the coordinator's, on MPS, with the real BiomedBERT feature cache).
`main()` checks for that artifact and exits without fabricating an input.
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


def s4_c98_power(y, s, seed_seq, n_resamples=N_RESAMPLES):
    """resample 0 (the true, unpermuted gold) plus the c98_rule fire rate
    over `n_resamples` independent permutations of the gold labels (holding
    `s` fixed), which is this stage's CLEAN construction. `y`, `s` on the
    unpermuted labels is the PLANT reading (real detector scores against the
    real synthetic gold)."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    resample_0_fires = ev.rule_c98(y, s)

    rng = np.random.default_rng(seed_seq)
    fires = []
    for _ in range(n_resamples):
        y_perm = rng.permutation(y)
        fires.append(ev.rule_c98(y_perm, s))
    return {
        "resample_0_on_true_gold_fires": bool(resample_0_fires),
        "fire_rate_over_permuted_gold_resamples": float(np.mean(fires)),
        "n_resamples": n_resamples,
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
    result = s4_c98_power(y, s, np.random.SeedSequence(SEED))
    out = {"preregistration": "accord/public_controls/C102_PREREGISTRATION.md",
          "stage": "4", "check": "S4", **result}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
