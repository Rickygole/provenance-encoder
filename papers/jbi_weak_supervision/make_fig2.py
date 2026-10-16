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

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 300,
                     "savefig.bbox": "tight", "font.family": "sans-serif"})
fig, ax = plt.subplots(figsize=(6.4, 3.6))

styles = [(0.0474, "-", "#2563EB"), (0.10, "--", "#F59E0B"), (0.25, "-.", "#0D9488"), (0.50, ":", "#E11D48")]
for prev, ls, col in styles:
    KK = max(1, int(round(prev * N)))
    ns = np.arange(KK, N - 1)
    ys = [ceiling(N, KK, int(n)) for n in ns]
    ax.plot(ns / N, ys, ls, color=col, linewidth=1.6,
            label="prevalence %.2f" % (KK / N))

# the band where a developer could mistake a low coefficient for a real one
ax.axvspan(0.15, 0.50, color="#FEF3C7", zorder=0)
ax.text(0.325, 0.93, "coefficient looks small\nbut is near its maximum",
        ha="center", va="top", fontsize=8, color="#92400E")

ax.plot([n_obs / N], [obs_ceiling], "o", color="#7C3AED", markersize=8,
        markeredgecolor="white", markeredgewidth=1.2, zorder=5)
ax.annotate("this stage: flag rate %.4f,\nceiling %.4f, observed %.4f"
            % (n_obs / N, obs_ceiling, obs_mcc),
            xy=(n_obs / N, obs_ceiling), xytext=(0.60, 0.33), fontsize=8,
            arrowprops=dict(arrowstyle="-", color="#7C3AED", linewidth=1.0))

ax.set_xlabel("flag rate $n/N$")
ax.set_ylabel("attainable maximum of the Matthews correlation")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.03)
ax.legend(frameon=False, fontsize=8, loc="upper right")
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
}, indent=1))
