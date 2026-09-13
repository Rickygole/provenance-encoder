"""C102 Stage 1 (reduced scope): S1b, defect-free-stratum disagreement.
See ../C102_PREREGISTRATION.md.

Candidate-level simulation only: no note text, no template, nothing MIMIC-
derived. Reads two register keys read-only (never writes them):
`goldstrata.join_feasibility.gold_rule_counts.{1,2,3,4}` and
`exemplar.power_analysis.n_positives_total`.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.stats import fisher_exact, binom

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
OUT = ROOT / "results" / "accord" / "plantbench" / "stage1_s1b.json"

SEED = 42
RATES = [0.0, 0.25, 0.47]
N_RESAMPLES = 1000
ALPHA = 0.05


def apportion(total: int, weights) -> np.ndarray:
    """Hamilton's largest-remainder method: integer counts summing exactly to
    `total`, proportional to `weights`."""
    weights = np.asarray(weights, dtype=float)
    shares = weights / weights.sum() * total
    base = np.floor(shares).astype(int)
    remainder = int(total - base.sum())
    frac = shares - base
    order = np.argsort(-frac)
    for i in range(remainder):
        base[order[i]] += 1
    return base


def read_register_counts():
    facts = json.loads(FACTS.read_text())
    rule_counts = [facts[f"goldstrata.join_feasibility.gold_rule_counts.{r}"]["value"]
                  for r in (1, 2, 3, 4)]
    n_pos_total = facts["exemplar.power_analysis.n_positives_total"]["value"]
    return rule_counts, n_pos_total


def build_strata(n_target=253):
    rule_counts, n_pos_total = read_register_counts()
    n_by_rule = apportion(n_target, rule_counts)
    pos_by_rule = apportion(n_pos_total, rule_counts)
    n_a, n_b = int(n_by_rule[0] + n_by_rule[1]), int(n_by_rule[2] + n_by_rule[3])
    pos_a, pos_b = int(pos_by_rule[0] + pos_by_rule[1]), int(pos_by_rule[2] + pos_by_rule[3])
    assert n_a + n_b == n_target
    assert pos_a + pos_b == n_pos_total
    return {
        "rule_counts": rule_counts, "n_positives_total": n_pos_total,
        "n_by_rule": n_by_rule.tolist(), "pos_by_rule": pos_by_rule.tolist(),
        "n_a": n_a, "n_b": n_b, "pos_a": pos_a, "pos_b": pos_b,
    }


def one_replicate(pos_a, n_a, pos_b, n_b, rate, rng):
    """Group A's true positives are always confirmed (failure rate 0). Group
    B's true positives are confirmed independently with probability
    (1 - rate); the plant applies `rate` only to group B (the defect-free
    stratum). Returns (p_value, fires)."""
    confirmed_a = pos_a  # deterministic: rate is always 0 for group A
    confirmed_b = int(rng.binomial(pos_b, 1.0 - rate)) if pos_b > 0 else 0
    table = [[confirmed_a, n_a - confirmed_a], [confirmed_b, n_b - confirmed_b]]
    _, p = fisher_exact(table, alternative="greater")
    return float(p), bool(p < ALPHA)


def run_rate(strata, rate, seed_seq):
    rng = np.random.default_rng(seed_seq)
    p0, fires0 = one_replicate(strata["pos_a"], strata["n_a"],
                               strata["pos_b"], strata["n_b"], rate, rng)
    # "resample 0" above consumed the first draw of this rate's stream; the
    # 1000-resample rate below uses a FRESH rng from the same seed so the
    # headline resample-0 result is reproducible independent of how many
    # more resamples are run after it.
    rng2 = np.random.default_rng(seed_seq)
    fires = []
    ps = []
    for _ in range(N_RESAMPLES):
        p, f = one_replicate(strata["pos_a"], strata["n_a"],
                             strata["pos_b"], strata["n_b"], rate, rng2)
        ps.append(p)
        fires.append(f)
    fire_rate_1000 = float(np.mean(fires))
    fire_count_20 = int(np.sum(fires[:20]))
    return {
        "rate": rate,
        "resample_0": {"p_value": p0, "fires": fires0},
        "fire_count_first_20": fire_count_20,
        "fire_rate_over_1000_resamples": fire_rate_1000,
        "n_resamples": N_RESAMPLES,
    }


def binomial_probability_table(fire_count_20):
    """P(X = fire_count_20) for X ~ Binomial(20, true_rate), at a few true
    rates, so a count like 18/20 can be read against how likely it is under
    each assumed true firing rate."""
    return {str(r): float(binom.pmf(fire_count_20, 20, r)) for r in (0.5, 0.7, 0.9, 0.95)}


def verdict(clean_result, plant_results):
    """General C102 verdict rule applied to S1b: validated needs the clean
    arm's fire rate (first 20) <= 2/20 AND at least one plant rate's fire
    count (first 20) >= 18/20; otherwise named per the plan's categories."""
    clean_fires = clean_result["fire_count_first_20"]
    if clean_fires > 5:
        return "unvalidated_false_alarm", f"clean fired {clean_fires}/20"
    best_plant = max(plant_results, key=lambda r: r["fire_count_first_20"])
    if best_plant["fire_count_first_20"] >= 18:
        if clean_fires <= 2:
            return "validated", (f"clean {clean_fires}/20, best plant "
                                 f"(rate={best_plant['rate']}) {best_plant['fire_count_first_20']}/20")
        return "specificity_marginal", f"clean {clean_fires}/20 (3-5 range)"
    if 3 <= clean_fires <= 5:
        return "specificity_marginal", f"clean {clean_fires}/20"
    return "not_validated_sensitivity", (f"best plant (rate={best_plant['rate']}) "
                                         f"only {best_plant['fire_count_first_20']}/20")


def main():
    strata = build_strata()
    seed_seq = np.random.SeedSequence(SEED)
    children = seed_seq.spawn(len(RATES))
    results = [run_rate(strata, rate, child) for rate, child in zip(RATES, children)]
    clean = next(r for r in results if r["rate"] == 0.0)
    plants = [r for r in results if r["rate"] != 0.0]
    verdict_label, verdict_note = verdict(clean, plants)

    for r in results:
        r["binomial_probability_of_first_20_count_at_true_rate"] = \
            binomial_probability_table(r["fire_count_first_20"])

    out = {
        "preregistration": "accord/public_controls/C102_PREREGISTRATION.md",
        "stage": "1", "check": "S1b",
        "strata": strata,
        "results_by_rate": results,
        "verdict": verdict_label,
        "verdict_note": verdict_note,
        "seed": SEED,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
