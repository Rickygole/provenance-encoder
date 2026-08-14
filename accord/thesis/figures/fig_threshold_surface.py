"""F2 -- threshold robustness surface.

18 operating points x 15 selection strengths from C40. Colour is the
demographic-parity gap. The solid contour marks the 0.05 tolerance. The
underpowered region -- where the minimum detectable gap exceeds tolerance,
so the test cannot see rather than fails to find -- is washed out, hard
enough that it plainly reads as a different kind of surface: a figure that
declares two thirds of its own cells unable to answer must not draw those
cells at the same strength as the cells that can.

Every key is a direct label on the thing it names, including the power
boundary itself, which now carries a small rotated label riding the line
rather than only a block of text parked in the region beside it. The one
accented element is the operating point this study actually shipped.

Read from results/accord/selection/threshold_robustness_results.json, the
committed record, rather than from the working CSV outside the repository:
the operating point used to be typed here as 0.1994 and is now the value the
file carries, which is also what the notebook cell reads.
"""
import json, pathlib, numpy as np, pandas as pd
from style import (setup, FIGW, INK, BLUE, PALE, SCARLET, MUTE, GREY,
                   ACCENT, MUTED, LIGHT, DEAD, canvas)
setup()
import matplotlib.pyplot as plt

# Long provenance notes belong in the LaTeX caption, not inside the plot.
# Set True to draw them again.
SHOW_INFIGURE_NOTES = False
SHOW_INFIGURE_TITLES = False
from matplotlib.colors import LinearSegmentedColormap

ROOT = pathlib.Path("/Users/rickygole/provenance-encoder")
d_thresh = json.loads((ROOT / "results/accord/selection/"
                       "threshold_robustness_results.json").read_text())
df = pd.DataFrame(d_thresh["cells"])
piv = df.pivot_table(index="achieved_rate", columns="beta",
                     values="demographic_parity_gap", aggfunc="mean")
mde = df.pivot_table(index="achieved_rate", columns="beta",
                     values="mde_demographic_parity_gap", aggfunc="mean")
R, B, Z, M = piv.index.values, piv.columns.values, piv.values, mde.values
e9_rate = d_thresh["original_e9_population_pred_pos_rate"]

# Named setting, not read from this file: the audit's own demographic-parity tolerance.
TOLERANCE = 0.05

# One hue, five stops, chosen so greyscale lightness falls almost exactly
# linearly with the value: 255, 202, 149, 96, 43 out of 255. The surface
# therefore still reads as a magnitude in a black and white print, and because
# nothing in it depends on separating two hues it survives any dichromacy.
cmap = LinearSegmentedColormap.from_list(
    "acc", [(0.00, "#FFFFFF"), (0.25, "#C0CBE2"), (0.50, "#7E96C8"),
            (0.75, "#42639F"), (1.00, BLUE)])
fig, ax = plt.subplots(figsize=(FIGW, 2.55))
canvas(fig)
fig.subplots_adjust(left=0.100, right=0.855, top=0.975, bottom=0.165)
im = ax.pcolormesh(B, R, Z, cmap=cmap, shading="nearest", vmin=0, vmax=0.20)

# Wash the underpowered region. At the old strength the veil left a cell whose
# gap the sweep cannot measure looking very much like a cell whose gap it can,
# so a figure whose entire point is that most of this surface has no power was
# drawing announced noise at nearly the strength of signal. At this strength
# the two regions are different surfaces at a glance, and the values underneath
# are still legible against the colour bar for anyone reading them.
veil = np.where(M > TOLERANCE, 0.76, 0.0)
ax.pcolormesh(B, R, veil, cmap=LinearSegmentedColormap.from_list("v", ["#ffffff00", "#ffffff"]),
              shading="nearest", vmin=0, vmax=1, zorder=2)
ax.contour(B, R, (M > TOLERANCE).astype(float), levels=[0.5],
           colors=GREY, linewidths=0.6, linestyles=[(0, (4, 2))], zorder=3)
# The boundary is labelled at the boundary, riding it at its own angle, in the
# text tone that belongs to a grey reference line. The block of text in the
# region says what the region means; this says what the line is.
ax.text(2.46, 0.218, "power limit", fontsize=8.0, color=MUTE,
        ha="center", va="bottom", rotation=-20.4, rotation_mode="anchor",
        zorder=7)
ax.text(8.35, 0.55, "underpowered: the minimum\ndetectable gap exceeds tolerance",
        ha="right", va="top", fontsize=8.0, color=MUTE, linespacing=1.3, zorder=7)

# Every fragment of the tolerance contour is drawn whole, including the short
# one near the bottom left. The label is placed beside the longest fragment
# rather than inline in it, because an inline label is cut out of the line it
# names, and on a surface this broken a gap in a contour is indistinguishable
# from a fragment somebody removed.
cs = ax.contour(B, R, Z, levels=[TOLERANCE], colors=INK, linewidths=0.8, zorder=4)
ax.text(1.55, 0.385, f"{TOLERANCE} tolerance", fontsize=8.0, color=INK,
        ha="left", va="bottom", rotation=78.0, rotation_mode="anchor", zorder=7)

# The operating point this study shipped: the one accented element on the canvas.
ax.plot(B, [e9_rate]*len(B), color="w", lw=1.8, zorder=5)
ax.plot(B, [e9_rate]*len(B), color=SCARLET, lw=0.8, zorder=6)
# Label parked at the right-hand end of the line it names. Anchored at the left
# of the panel it lay across the 0.05 contour.
ax.text(B[-1], e9_rate + 0.008, "original E9 operating point", ha="right", va="bottom",
        fontsize=8.0, color=SCARLET, zorder=7)

ax.set_xlabel(r"Selection strength through documentation volume ($\beta$)")
ax.set_ylabel("Predicted-positive rate")
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
if SHOW_INFIGURE_TITLES: ax.set_title("The audit never becomes cleanly blind; it becomes unmeasurable",
             loc="left", fontsize=9.5, pad=16)

cb = fig.colorbar(im, ax=ax, pad=0.020, fraction=0.040)
cb.set_label("Demographic-parity gap", fontsize=9.0)
cb.set_ticks([0, 0.05, 0.10, 0.15, 0.20])
cb.ax.tick_params(labelsize=8, width=0.6, length=3)
cb.outline.set_visible(False)

out = pathlib.Path(__file__).parent / "fig-threshold-surface.pdf"
fig.savefig(out)
n_under = int((M > TOLERANCE).sum())
print(f"wrote {out.name}  ({Z.shape[0]}x{Z.shape[1]} cells, {n_under} underpowered)")
