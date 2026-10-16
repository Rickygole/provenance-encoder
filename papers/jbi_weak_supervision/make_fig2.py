#!/usr/bin/env python3
"""Figure 2: the attainable ceiling as a function of the operating point.

Pure arithmetic from Equation 1; the single plotted observation is read from
the committed artifact. No new inference.

    python3 make_fig2.py
"""
import json, math, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
c120 = json.loads((ROOT / "results/accord/c120_exact_join/c120.json").read_text())
g = c120["g253"]
N, K = 253, 12
n_obs = int(round(g["flag_rate"] * N))
obs_ceiling = g["ceiling_mcc"]
obs_mcc = g["mcc"]

def ceiling(N, K, n):
    tp = min(n, K); fp = n - tp; fn = K - tp; tn = N - n - fn
    d = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn - fp*fn)/d if d else float("nan")

plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 300,
                     "savefig.bbox": "tight", "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica"],
                     "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold",
                     "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(4.6, 2.95), constrained_layout=True)

styles = [(0.0474, "-", "#2563EB"), (0.10, "--", "#F59E0B"), (0.25, "-.", "#0D9488"), (0.50, ":", "#E11D48")]
for prev, ls, col in styles:
    KK = max(1, int(round(prev * N)))
    ns = np.arange(KK, N - 1)
    ys = [ceiling(N, KK, int(n)) for n in ns]
    ax.plot(ns / N, ys, ls, color=col, linewidth=1.6,
            label="prevalence %.4f" % (KK / N))

# The region is computed, not drawn: on the study prevalence curve, the flag
# rates whose ceiling lies in [LO, HI]. Inside it a coefficient of LO is
# between LO/HI and all of what the margins allow, so the same reported value
# carries a different meaning at each end.
LO, HI = 0.20, 0.60
grid = np.arange(K, N - 1)
cs = np.array([ceiling(N, K, int(n)) for n in grid])
inband = grid[(cs >= LO) & (cs <= HI)] / N
band_lo, band_hi = float(inband.min()), float(inband.max())
ax.axvspan(band_lo, band_hi, ymin=0.0, ymax=0.60/1.03,
           color="#FEF3C7", zorder=0)
ax.text((band_lo + band_hi) / 2, 0.635,
        "at prevalence %.4f the ceiling\nhere lies between %.2f and %.2f" % (K / N, LO, HI),
        ha="center", va="bottom", fontsize=9, color="#92400E",
        bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=1.5))

ax.plot([n_obs / N], [obs_ceiling], "o", color="#7C3AED", markersize=8,
        markeredgecolor="white", markeredgewidth=1.2, zorder=5)
ax.annotate("this stage: flag rate %.4f,\nceiling %.4f, observed %.4f"
            % (n_obs / N, obs_ceiling, obs_mcc),
            xy=(n_obs / N, obs_ceiling), xytext=(0.50, 0.17), fontsize=9,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1.5),
            arrowprops=dict(arrowstyle="-", color="#7C3AED", linewidth=1.0,
                            shrinkA=2, shrinkB=2))

ax.set_xlabel("flag rate $n/N$")
ax.set_ylabel("attainable maximum of the Matthews correlation")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.03)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title("The ceiling is fixed by the gap between flag rate and prevalence, "
             "not by prevalence alone", fontsize=9, loc="left")
out = HERE / "fig2.pdf"
fig.savefig(out); print(out)
# values the caption quotes, generated not typed
print(json.dumps({
    "ceiling_at_flagrate_equals_prevalence": ceiling(N, K, K),
    "ceiling_at_n_24": ceiling(N, K, 24),
    "ceiling_at_n_50": ceiling(N, K, 50),
    "ceiling_at_n_100": ceiling(N, K, 100),
    "ceiling_observed": ceiling(N, K, n_obs),
    "band_lo_flag_rate": band_lo,
    "band_hi_flag_rate": band_hi,
    "band_ceiling_range": [LO, HI],
}, indent=1))
