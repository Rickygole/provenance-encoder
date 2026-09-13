"""Shared helpers for the three over-calling / gold-tipping-point analyses.

Not a generator itself; imported by:
    accord/stats/overcall_decomposition.py
    accord/stats/overcall_ambiguity_readings.py
    accord/stats/gold_tipping_points.py

Kept tiny and dependency-light on purpose: one Wilson interval, one register
reader, one register-key resolver. Nothing here touches ~/mimic_local_only.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FACTS_PATH = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"


def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion.

    Copied from accord/thesis/figures/style.py:wilson_ci (not imported, to
    keep this module import-free of the figures package) with the same
    clamping behaviour: the bound is clamped to [0, 1] and to the point
    estimate itself, because at k = 0 or k = n the exact bound is 0 or 1 and
    the floating point arithmetic can land a few parts in 1e18 on the wrong
    side of it.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    adj = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    lo = min(p, max(0.0, (centre - adj) / denom))
    hi = max(p, min(1.0, (centre + adj) / denom))
    return (lo, hi)


def load_facts():
    """Read-only load of the register. Never written by these scripts."""
    return json.loads(FACTS_PATH.read_text())


def reg(facts, key):
    """Resolve a live (non-withdrawn) register key to its value.

    Raises rather than silently returning None: a key that moved or was
    withdrawn should break the generator, not produce a quietly wrong number.
    """
    if key not in facts:
        raise KeyError(f"register key not found: {key}")
    if key.startswith("withdrawn."):
        raise KeyError(f"register key is withdrawn: {key}")
    node = facts[key]
    return node["value"] if isinstance(node, dict) and "value" in node else node


def read_json(path):
    return json.loads(pathlib.Path(path).read_text())
