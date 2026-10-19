#!/usr/bin/env python3
"""Figure 3: the verdict as a distribution over fold assignments.

Reads the per-assignment counts the cross-fit emits and draws them. No new
inference, and every summary the figure marks is asserted against the artifact
rather than recomputed here, so the figure cannot disagree with the text.

Colour carries the x variable redundantly, using a perceptually uniform map
that is monotonic in lightness. The bars therefore stay separable when the page
is printed in greyscale, which the journal figure standard requires, and the
colour only makes the spread easier to see.

    python3 make_fig3.py
"""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
q = json.loads(
    (ROOT / "results/accord/c141_crossfit_repair/c142_delivery.json").read_text()
)["q3_stability_over_fold_assignments"]

counts = np.array(q["clearing_0_05_per_assignment"], int)
cells = q["n_distinct_cells"]
med = q["clearing_0_05_median"]
lo, hi = q["clearing_0_05_pct2_5"], q["clearing_0_05_pct97_5"]
mean = q["clearing_0_05_mean"]
assert len(counts) == q["repeats"], "array length does not match the repeat count"
assert float(np.median(counts)) == med, "median does not match the artifact"
assert counts.min() == q["clearing_0_05_min"], "min does not match the artifact"
assert counts.max() == q["clearing_0_05_max"], "max does not match the artifact"
assert abs(counts.mean() - mean) < 5e-3, "mean does not match the artifact"

plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False,
                     "savefig.bbox": "tight", "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica"],
                     "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold",
                     "axes.labelsize": 9, "legend.fontsize": 8,
                     "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(5.0, 2.95), constrained_layout=True)

heights = np.bincount(counts, minlength=cells + 1)
centres = np.arange(cells + 1)
cmap = plt.get_cmap("viridis")
colours = cmap(np.linspace(0.08, 0.92, cells + 1))
ax.bar(centres, heights, width=0.86, color=colours,
       edgecolor="#1F2937", linewidth=0.5, zorder=3)

# The interval is outlined rather than filled: a fill behind the bars competes
# with them for tone and the bars inside it stop being readable in greyscale.
ax.axvspan(lo, hi, facecolor="none", edgecolor="#B91C1C", linewidth=1.0,
           linestyle=(0, (5, 2)), zorder=5,
           label="95%% of assignments: %g to %g" % (lo, round(hi)))
ax.axvline(med, color="#B91C1C", linewidth=2.2, zorder=6,
           label="median %g of %d" % (med, cells))
ax.axvline(mean, color="#1F2937", linewidth=1.2, linestyle=(0, (1, 1.6)),
           zorder=6, label="mean %.2f" % mean)

top = heights.max()
for x, note in ((0, "no cell\nclears"), (cells, "every cell\nclears")):
    ax.annotate(note, xy=(x, heights[x] + 1.5), xytext=(x, top * 0.56),
                ha="center", va="bottom", fontsize=7.5, color="#374151",
                arrowprops=dict(arrowstyle="-", color="#9CA3AF", linewidth=0.7,
                                shrinkA=2, shrinkB=2), zorder=7)

ax.set_xlabel("cells clearing $p<0.05$ two-sided, out of %d" % cells)
ax.set_ylabel("fold assignments of %d" % len(counts))
ax.set_xlim(-0.75, cells + 0.75)
ax.set_ylim(0, top * 1.14)
ax.set_xticks(range(0, cells + 1, 3))
ax.grid(axis="y", color="#E5E7EB", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=True, framealpha=0.95, edgecolor="none", fontsize=8,
          loc="upper right")
ax.set_title("The same 253 pairs give any verdict from %d to %d,\n"
             "depending only on the fold assignment"
             % (counts.min(), counts.max()), fontsize=9.5, loc="left",
             color="#111827")

out = HERE / "fig3.pdf"
fig.savefig(out)
print(out)
print("%d assignments, median %g, mean %.2f, 95%% %g to %g, range %d to %d"
      % (len(counts), med, mean, lo, hi, counts.min(), counts.max()))
