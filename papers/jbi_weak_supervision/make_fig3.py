#!/usr/bin/env python3
"""Figure 3: the verdict as a distribution over fold assignments.

Reads the per-assignment counts the cross-fit emits and draws them. No new
inference and no arithmetic beyond the quantiles the artifact already records,
which are asserted against the artifact rather than recomputed for the figure.

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
med, lo, hi = q["clearing_0_05_median"], q["clearing_0_05_pct2_5"], q["clearing_0_05_pct97_5"]
assert len(counts) == q["repeats"], "array length does not match the repeat count"
assert float(np.median(counts)) == med, "median does not match the artifact"
assert counts.min() == q["clearing_0_05_min"] and counts.max() == q["clearing_0_05_max"]

plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False,
                     "savefig.bbox": "tight", "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica"],
                     "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold",
                     "axes.labelsize": 9, "legend.fontsize": 8,
                     "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(4.6, 2.75), constrained_layout=True)

# The bars carry the data, so they take the only solid tone. The interval is
# drawn as an outlined region rather than a fill, because a second fill behind
# the bars becomes the same grey as the bars when the page is printed in
# greyscale and the bars inside it stop being readable.
bins = np.arange(-0.5, cells + 1.5, 1)
ax.hist(counts, bins=bins, color="#94A3B8", edgecolor="#0F172A", linewidth=0.7)
ax.axvspan(lo, hi, facecolor="none", edgecolor="#B45309", linewidth=0.9,
           linestyle=(0, (4, 2)), zorder=4,
           label="95%% of assignments: %g to %g" % (lo, round(hi)))
ax.axvline(med, color="#B45309", linewidth=2.0, zorder=5,
           label="median %g of %d" % (med, cells))

ax.set_xlabel("cells clearing $p<0.05$, out of %d" % cells)
ax.set_ylabel("fold assignments of %d" % len(counts))
ax.set_xlim(-0.5, cells + 0.5)
ax.set_xticks(range(0, cells + 1, 3))
ax.legend(frameon=True, framealpha=1.0, edgecolor="none",
          fontsize=8, loc="upper right")
ax.set_title("The same %d pairs give any verdict from %d to %d,\n"
             "depending only on the fold assignment"
             % (253, counts.min(), counts.max()), fontsize=9, loc="left")

out = HERE / "fig3.pdf"
fig.savefig(out)
print(out)
print("counts %d assignments, median %g, 95%% %g to %g, min %d, max %d"
      % (len(counts), med, lo, hi, counts.min(), counts.max()))
