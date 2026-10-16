"""C143: does the ceiling-normalised coefficient carry decision value?

Runs the analysis fixed in C143_PREREGISTRATION.md. Reads only committed
artifacts, runs no inference, and touches no clinical data.

    python3 accord/c143_ceiling_sweep/c143.py
"""
import json, math, pathlib
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
c136 = json.loads((ROOT / "results/accord/c136_gate_validation/c136.json").read_text())

N = c136["frames"]["pilot"][0]
K = c136["frames"]["pilot"][1]
BAND_LO, BAND_HI = 0.1225296442687747, 0.5533596837944664   # from make_fig2.py


def mcc(tp, fp, fn, tn):
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d > 0 else 0.0


def ceiling(n):
    tp = min(n, K); fp = n - tp; fn = K - tp; tn = N - n - fn
    return mcc(tp, fp, fn, tn)


rows = []
for r in c136["labellers"]:
    n = r["flag_count"]
    tp = r["pilot_true_positives_flagged"]
    fp = n - tp; fn = K - tp; tn = N - n - fn
    obs = mcc(tp, fp, fn, tn)
    cap = ceiling(n)
    rows.append({
        "labeller": r["labeller"], "family": r["family"],
        "n_flagged": n, "flag_rate": n / N, "true_positives_flagged": tp,
        "observed_mcc": obs, "ceiling_mcc": cap,
        "ratio": (obs / cap) if cap > 0 else None,
        "student_auroc_minilm": r["student_auroc_minilm"],
    })

usable = [r for r in rows if r["ratio"] is not None
          and r["student_auroc_minilm"] is not None]
raw = stats.spearmanr([r["observed_mcc"] for r in usable],
                      [r["student_auroc_minilm"] for r in usable])
rat = stats.spearmanr([r["ratio"] for r in usable],
                      [r["student_auroc_minilm"] for r in usable])

delta = rat.statistic - raw.statistic
if delta > 0.05:
    outcome, verdict = "A", "the ratio ranks better than the raw coefficient"
elif delta < -0.05:
    outcome, verdict = "C", "the ratio ranks worse than the raw coefficient"
else:
    outcome, verdict = "B", "the two are indistinguishable at the registered margin"

qualifying = [
    r for r in usable
    if BAND_LO <= r["flag_rate"] <= BAND_HI
    and ((r["ratio"] > 0.90 and r["observed_mcc"] < 0.50)
         or (r["ratio"] < 0.40 and r["observed_mcc"] > 0.20))
]

out = {
    "check": "C143",
    "what_this_closes": "whether the ceiling-normalised coefficient carries "
                        "decision value, and whether the check binds in the "
                        "non-degenerate regime",
    "prereg": "accord/c143_ceiling_sweep/C143_PREREGISTRATION.md",
    "frame": {"n": N, "n_positives": K, "prevalence": K / N},
    "n_labellers": len(rows), "n_usable": len(usable),
    "q1_ranking": {
        "spearman_raw_mcc_vs_student_auroc": raw.statistic,
        "spearman_raw_p": raw.pvalue,
        "spearman_ratio_vs_student_auroc": rat.statistic,
        "spearman_ratio_p": rat.pvalue,
        "ratio_minus_raw": delta,
        "registered_margin": 0.05,
        "outcome": outcome,
        "verdict": verdict,
    },
    "q2_non_degenerate": {
        "band_flag_rate_lo": BAND_LO, "band_flag_rate_hi": BAND_HI,
        "n_qualifying": len(qualifying),
        "qualifying": qualifying,
    },
    "q3_table": sorted(rows, key=lambda r: r["flag_rate"]),
}
dest = ROOT / "results/accord/c143_ceiling_sweep"
dest.mkdir(parents=True, exist_ok=True)
(dest / "c143.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("n_labellers", "n_usable")}, indent=1))
print(json.dumps(out["q1_ranking"], indent=1))
print("qualifying in band:", len(qualifying))
for r in qualifying:
    print("  %-28s n=%3d rate=%.4f obs=%.4f ceil=%.4f ratio=%.4f auroc=%.4f"
          % (r["labeller"], r["n_flagged"], r["flag_rate"], r["observed_mcc"],
             r["ceiling_mcc"], r["ratio"], r["student_auroc_minilm"]))
