"""C134a: measured coverage of the nominal 95 percent percentile bootstrap
interval for AUROC at 12 positives and 241 negatives.

Chapter 3 states, of the bounded-null definition: "nothing here measures how
often the percentile bootstrap's upper end covers the true value at 12
positives. The definition rests on that upper end, so its calibration at this
size is assumed rather than shown." This script measures it.

Reuses the pre-registered C108 power-simulation machinery verbatim:
  - the already-scored public RoBERTa-MNLI SNLI pool
    (results/accord/public_controls/sizectl_c108/roberta_scores.json),
  - accord.public_controls.evaluation.label_mix_lambda and
    draw_label_mix_replicate, whose population AUROC is exact and equal to the
    target by construction,
  - 12 positive slots and 241 negative slots, the G_253 geometry.

Two interval methods are measured:
  declared_unstratified -- the method the thesis's headline decontaminated
    interval (0.3210 to 0.7034) was produced under and re-derived under in
    accord/stats/recompute_decontam_aggregate.py: resample all n rows with
    replacement, drop single-class resamples, 2.5/97.5 percentiles.
  stratified -- accord.public_controls.evaluation.stratified_bootstrap_auroc:
    resample positives and negatives separately, same percentiles. This is the
    method C108's own bootstrap decision rule used.

The bootstrap AUROC here is computed with average (mid-)rank ties handling,
which is what sklearn.roc_auc_score does; a pinning check against
sklearn.roc_auc_score on random tie-heavy inputs is run and recorded.

No MIMIC data, no model load, no network. Public SNLI scores only.

    python3 accord/c134_reanalysis/c134a_bootstrap_coverage.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accord.public_controls import evaluation as ev  # noqa: E402

SCORES = ROOT / "results" / "accord" / "public_controls" / "sizectl_c108" / "roberta_scores.json"
META = ROOT / "results" / "accord" / "public_controls" / "sizectl_c108" / "roberta_scores_meta.json"
OUT = ROOT / "results" / "accord" / "c134_reanalysis" / "c134a_bootstrap_coverage.json"

N_POS, N_NEG = 12, 241
N = N_POS + N_NEG
TARGETS = [0.50, 0.60, 0.7355, 0.7466, 0.80]
N_REPS = 2000
B = 10000
MASTER_SEED = 134
FLOOR = 0.7466
NOMINAL = 0.95


def auroc_rows(ys: np.ndarray, ss: np.ndarray) -> np.ndarray:
    """AUROC per row of (B, n) label/score matrices, average-rank ties.
    NaN where a row is single class."""
    r = rankdata(ss, axis=1)
    m = ys.sum(axis=1).astype(float)
    n_neg = ss.shape[1] - m
    rs_pos = (r * ys).sum(axis=1)
    den = m * n_neg
    out = np.full(ss.shape[0], np.nan)
    ok = den > 0
    out[ok] = (rs_pos[ok] - m[ok] * (m[ok] + 1.0) / 2.0) / den[ok]
    return out


def pin_check(rng) -> dict:
    """auroc_rows must match sklearn.roc_auc_score, including on ties."""
    worst = 0.0
    for _ in range(200):
        y = np.zeros(N, int)
        y[rng.choice(N, rng.integers(1, N), replace=False)] = 1
        s = rng.integers(0, 12, N).astype(float)  # deliberately tie-heavy
        got = float(auroc_rows(y[None, :], s[None, :])[0])
        worst = max(worst, abs(got - float(roc_auc_score(y, s))))
    return {"max_abs_diff_vs_sklearn_on_tie_heavy_inputs": worst, "passes": bool(worst < 1e-12)}


def ci_unstratified(y, s, rng, B=B):
    idx = rng.integers(0, len(y), size=(B, len(y)))
    a = auroc_rows(y[idx], s[idx])
    a = a[~np.isnan(a)]
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), int(a.size)


def ci_stratified(y, s, rng, B=B):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    idx = np.concatenate([rng.choice(pos, (B, len(pos)), True),
                          rng.choice(neg, (B, len(neg)), True)], axis=1)
    a = auroc_rows(y[idx], s[idx])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), int(a.size)


def main():
    records = json.loads(SCORES.read_text())
    meta = json.loads(META.read_text())
    is_contra = np.array([r["is_contradiction"] for r in records.values()])
    scores = np.array([r["score"] for r in records.values()], dtype=float)
    a0 = float(roc_auc_score(is_contra, scores))
    assert abs(a0 - meta["A0_auroc"]) < 1e-3, "recomputed A0 does not match meta"
    pos_pool, neg_pool = scores[is_contra == 1], scores[is_contra == 0]

    root = np.random.SeedSequence(MASTER_SEED)
    pin_seq, draw_seq = root.spawn(2)
    pin = pin_check(np.random.default_rng(pin_seq))
    assert pin["passes"], "auroc_rows does not match sklearn"

    target_seqs = draw_seq.spawn(len(TARGETS))
    results = {}
    t0 = time.time()
    for target, tseq in zip(TARGETS, target_seqs):
        lam = ev.label_mix_lambda(a0, target)
        draw_rng = np.random.default_rng(tseq.spawn(1)[0])
        boot_rngs = {m: np.random.default_rng(s)
                     for m, s in zip(("declared_unstratified", "stratified"), tseq.spawn(2))}
        acc = {m: {"cover": 0, "miss_low": 0, "miss_high": 0, "upper_gt_floor": 0,
                   "lower_gt_half": 0, "contains_half": 0, "width": [], "upper": [], "lower": []}
               for m in boot_rngs}
        point = []
        for _ in range(N_REPS):
            y, s = ev.draw_label_mix_replicate(pos_pool, neg_pool, N_POS, N_NEG, lam, draw_rng)
            point.append(float(auroc_rows(y[None, :], s[None, :])[0]))
            for m, fn in (("declared_unstratified", ci_unstratified), ("stratified", ci_stratified)):
                lo, hi, _ = fn(y, s, boot_rngs[m])
                a = acc[m]
                if lo <= target <= hi:
                    a["cover"] += 1
                elif hi < target:
                    a["miss_low"] += 1
                else:
                    a["miss_high"] += 1
                a["upper_gt_floor"] += int(hi > FLOOR)
                a["lower_gt_half"] += int(lo > 0.5)
                a["contains_half"] += int(lo <= 0.5 <= hi)
                a["width"].append(hi - lo)
                a["upper"].append(hi)
                a["lower"].append(lo)
        row = {"target_auroc": target, "label_mix_lambda": lam,
               "point_auroc_mean": float(np.mean(point)),
               "point_auroc_sd": float(np.std(point, ddof=1)), "methods": {}}
        for m, a in acc.items():
            cov = a["cover"] / N_REPS
            row["methods"][m] = {
                "n_reps": N_REPS, "B": B,
                "coverage": cov,
                "coverage_mc_se": float(np.sqrt(cov * (1 - cov) / N_REPS)),
                "noncoverage": 1 - cov,
                "miss_rate_interval_entirely_below_truth": a["miss_low"] / N_REPS,
                "miss_rate_interval_entirely_above_truth": a["miss_high"] / N_REPS,
                "p_upper_end_exceeds_0_7466": a["upper_gt_floor"] / N_REPS,
                "p_lower_end_exceeds_0_5": a["lower_gt_half"] / N_REPS,
                "p_interval_contains_0_5": a["contains_half"] / N_REPS,
                "mean_width": float(np.mean(a["width"])),
                "mean_upper_end": float(np.mean(a["upper"])),
                "mean_lower_end": float(np.mean(a["lower"])),
                "verdict": ("anti-conservative" if cov < NOMINAL - 2 * np.sqrt(NOMINAL * 0.05 / N_REPS)
                            else "conservative" if cov > NOMINAL + 2 * np.sqrt(NOMINAL * 0.05 / N_REPS)
                            else "at nominal"),
            }
        results[f"{target}"] = row
        print(f"target {target}: "
              + ", ".join(f"{m} cov={row['methods'][m]['coverage']:.4f} "
                          f"p(hi>{FLOOR})={row['methods'][m]['p_upper_end_exceeds_0_7466']:.4f}"
                          for m in row["methods"]), flush=True)

    out = {
        "what": "measured coverage of the nominal 95 percent percentile bootstrap AUROC "
                "interval at 12 positives and 241 negatives",
        "addresses": "accord/thesis/msu_ch3_methodology.tex: 'nothing here measures how often the "
                     "percentile bootstrap's upper end covers the true value at 12 positives'",
        "machinery": "accord/public_controls/evaluation.py label-mix construction, C108 pool",
        "pool": {"file": str(SCORES.relative_to(ROOT)), "n": int(len(scores)),
                 "A0_auroc": a0, "n_pos_pool": int(pos_pool.size), "n_neg_pool": int(neg_pool.size)},
        "design": {"n_positives": N_POS, "n_negatives": N_NEG, "n_reps_per_target": N_REPS,
                   "bootstrap_B": B, "nominal_coverage": NOMINAL, "floor": FLOOR,
                   "master_seed": MASTER_SEED, "targets": TARGETS},
        "pin_check": pin,
        "per_target": results,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
