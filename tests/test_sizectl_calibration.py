"""Regression test for the C101 calibration bug (see C101_PREREGISTRATION.md's
"Diagnosis, not a rule change" section) and its fix in
`accord/public_controls/evaluation.py`, applied under C108
(`accord/public_controls/C108_PREREGISTRATION.md`).

C101's `bisect_sigma_for_target` accepted a single, noisy 20,000-per-side
Monte Carlo draw at its sigma cap without ever checking the 0.002 tolerance,
at the noise-mix target-0.50 grid point (achieved population AUROC 0.5112,
0.0112 off target, over the 0.005 held-out bar). The fixed version computes
the noise-mix construction's population AUROC EXACTLY (no Monte Carlo, no
RNG, no standard error) on the full pools, handles target == 0.50 as the
construction's own sigma -> infinity limit rather than searching for a
finite sigma that cannot exist, and raises instead of silently returning an
out-of-tolerance boundary value.

Uses the real, already-scored, public SNLI/RoBERTa pools C101 produced
(`results/accord/public_controls/sizectl/roberta_scores{,_meta}.json`) --
the same pools the bug actually manifested on -- rather than a synthetic
stand-in, so this is a direct regression test against the failure, not just
a test of the formula in isolation.

Run: python3 -m pytest tests/test_sizectl_calibration.py -q
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accord.public_controls import evaluation as ev  # noqa: E402
from accord.public_controls.sizectl.power_curves import GRID  # noqa: E402

META = ROOT / "results" / "accord" / "public_controls" / "sizectl" / "roberta_scores_meta.json"
SCORES = ROOT / "results" / "accord" / "public_controls" / "sizectl" / "roberta_scores.json"

TOL = 0.002  # the pre-registered noise-mix bisection tolerance (C101, unchanged by C108).

pytestmark = pytest.mark.skipif(
    not (META.exists() and SCORES.exists()),
    reason=f"needs the real C101 pools at {SCORES} (score_roberta.py output); "
           "not present in this checkout")


@pytest.fixture(scope="module")
def pools():
    meta = json.loads(META.read_text())
    records = json.loads(SCORES.read_text())
    is_contra = np.array([r["is_contradiction"] for r in records.values()])
    scores = np.array([r["score"] for r in records.values()])
    pos_pool = scores[is_contra == 1]
    neg_pool = scores[is_contra == 0]
    a0 = float(roc_auc_score(is_contra, scores))
    assert abs(a0 - meta["A0_auroc"]) < 1e-3
    mu, sd = float(scores.mean()), float(scores.std())
    pos_z = (pos_pool - mu) / sd
    neg_z = (neg_pool - mu) / sd
    return pos_pool, neg_pool, pos_z, neg_z, a0


def test_grid_is_the_registered_c101_grid():
    """Guards against this test silently checking a different grid than the
    one C101/C108 actually run (the grid is reused verbatim under C108)."""
    assert GRID == [0.50, 0.60, 0.65, 0.70, 0.72, 0.7355, 0.75, 0.78, 0.80, 0.85, 0.90]


def test_every_grid_target_hit_within_tolerance_noise_mix(pools):
    """The bug: one grid cell (noise-mix, target 0.50) was off by 0.0112 with
    the old, single-Monte-Carlo-draw early return. The fix must hit every
    grid target within the pre-registered 0.002 tolerance, including 0.50."""
    _, _, pos_z, neg_z, _ = pools
    off_targets = {}
    for target in GRID:
        sigma, achieved = ev.bisect_sigma_for_target(pos_z, neg_z, target, tol=TOL)
        off_targets[target] = abs(achieved - target)
        assert off_targets[target] <= TOL, (
            f"target {target}: achieved {achieved}, off by {off_targets[target]} "
            f"(tol {TOL})")
    # The exact failure mode from C101: target 0.50 specifically.
    assert off_targets[0.50] <= TOL


def test_target_0_50_returns_infinite_sigma_sentinel_not_a_finite_cap(pools):
    """C101's bug was a finite-sigma-cap value silently accepted at target
    0.50. The fix must not return any finite sigma for this target: the
    noise-mix construction's population AUROC approaches 0.50 only in the
    sigma -> infinity limit, never reaching it at finite sigma for these
    pools, so pretending a finite cap "is" the target is exactly the bug."""
    _, _, pos_z, neg_z, _ = pools
    sigma, achieved = ev.bisect_sigma_for_target(pos_z, neg_z, 0.50)
    assert sigma == ev.INFINITE_SIGMA
    assert achieved == 0.5


def test_noise_mix_population_auroc_is_exact_not_sampled(pools):
    """Calling the deterministic AUROC function twice at the same sigma must
    give bit-identical results (no RNG state, no sampling variance) -- the
    property the old Monte Carlo estimate did not have, which is what let a
    single unlucky draw slip past the tolerance check undetected."""
    _, _, pos_z, neg_z, _ = pools
    a1 = ev.noise_mix_population_auroc(pos_z, neg_z, 3.0)
    a2 = ev.noise_mix_population_auroc(pos_z, neg_z, 3.0)
    assert a1 == a2


def test_noise_mix_population_auroc_matches_independent_monte_carlo(pools):
    """Cross-check the exact deterministic formula (used for calibration)
    against a fully independent, large-N Monte Carlo estimate (the same kind
    of check C101's own held-out validity gate uses), to catch a bug in the
    Phi-based formula itself, not just internal self-consistency."""
    _, _, pos_z, neg_z, _ = pools
    rng = np.random.default_rng(12345)
    for sigma in (0.5, 1.5, 4.0):
        exact = ev.noise_mix_population_auroc(pos_z, neg_z, sigma)
        mc = ev.estimate_noise_mix_auroc(pos_z, neg_z, sigma, rng, n_mc=200_000)
        # SE of an AUROC MC estimate at n=200,000/side is on the order of
        # 0.001; 0.01 is a generous, non-flaky margin for a cross-check.
        assert abs(exact - mc) < 0.01, f"sigma={sigma}: exact={exact}, mc={mc}"


def test_draw_noise_mix_replicate_at_infinite_sigma_gives_auroc_near_half(pools):
    """Sanity that the pure-label-permutation realization of the sigma ->
    infinity limit (used per-replicate at target 0.50) actually produces
    scores whose empirical AUROC is close to 0.5 at a large sample size, not
    just that the calibration function SAYS 0.5."""
    _, _, pos_z, neg_z, _ = pools
    rng = np.random.default_rng(99)
    y, s = ev.draw_noise_mix_replicate(pos_z, neg_z, 50_000, 50_000, ev.INFINITE_SIGMA, rng)
    auc = roc_auc_score(y, s)
    assert abs(auc - 0.5) < 0.01


def test_bisect_raises_rather_than_silently_capping_for_unreachable_target(pools):
    """A target above the sigma=0 (no-noise) achieved AUROC cannot be reached
    by ADDING noise (that only reduces separation); the fixed function must
    raise, not silently return the sigma=0 boundary the way the old code did
    for its (different) boundary."""
    _, _, pos_z, neg_z, a0 = pools
    unreachable = min(a0 + 0.05, 0.999)
    with pytest.raises(ValueError):
        ev.bisect_sigma_for_target(pos_z, neg_z, unreachable)


def test_every_grid_target_hit_within_tolerance_label_mix(pools):
    """The label-mix construction's population AUROC is exact BY
    CONSTRUCTION (`lambda = (A0 - target) / (A0 - 0.5)` makes
    `(1-lambda)*A0 + lambda*0.5 == target` an algebraic identity, not a
    search result), so this pins the identity itself directly (tolerance
    1e-9, no sampling involved) for every registered grid point, then adds a
    loose Monte Carlo sanity cross-check (tol 0.01, wide enough not to be
    flaky at n=200,000/side; a tight MC-vs-target comparison here would just
    be testing this test's own sampling noise, not the construction)."""
    pos_pool, neg_pool, _, _, a0 = pools
    rng = np.random.default_rng(4242)
    for target in GRID:
        lam = ev.label_mix_lambda(a0, target)
        lam_clipped = min(max(lam, 0.0), 1.0)
        exact = (1 - lam_clipped) * a0 + lam_clipped * 0.5
        assert abs(exact - target) <= 1e-9, (
            f"label-mix target {target}: exact identity gives {exact}")

        from_neg = rng.random(200_000) < lam_clipped
        pos_side = np.where(from_neg, rng.choice(neg_pool, 200_000, True),
                            rng.choice(pos_pool, 200_000, True))
        neg_side = rng.choice(neg_pool, 200_000, True)
        y = np.r_[np.ones(200_000, int), np.zeros(200_000, int)]
        s = np.r_[pos_side, neg_side]
        achieved = float(roc_auc_score(y, s))
        assert abs(achieved - target) <= 0.01, (
            f"label-mix target {target}: MC sanity check achieved {achieved}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
