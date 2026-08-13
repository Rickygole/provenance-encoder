"""F2 -- threshold robustness surface.

18 operating points x 15 selection strengths from C40. Colour is the
demographic-parity gap. The solid contour marks the 0.05 tolerance. The
underpowered region -- where the minimum detectable gap exceeds tolerance,
so the test cannot see rather than fails to find -- is veiled rather than
hatched, so it reads as "less certain" without obscuring the data beneath.
"""
import pathlib, numpy as np, pandas as pd
from style import setup, ACCENT, MUTED, LIGHT, DEAD
setup()
import matplotlib.pyplot as plt

# Long provenance notes belong in the LaTeX caption, not inside the plot.
# Set True to draw them again.
SHOW_INFIGURE_NOTES = False
SHOW_INFIGURE_TITLES = False
from matplotlib.colors import LinearSegmentedColormap

df = pd.read_csv("/Users/rickygole/mimic_local_only/accord_selection/threshold_robustness_cells.csv")
piv = df.pivot_table(index="achieved_rate", columns="beta",
                     values="demographic_parity_gap", aggfunc="mean")
mde = df.pivot_table(index="achieved_rate", columns="beta",
                     values="mde_demographic_parity_gap", aggfunc="mean")
R, B, Z, M = piv.index.values, piv.columns.values, piv.values, mde.values

# White to amber to teal to blue: monotonically darkening (greyscale luminance
# 255, 211, 90, 43), so the surface still reads as a magnitude in a black and
# white print, and the hue path stays inside the range dichromats separate.
cmap = LinearSegmentedColormap.from_list(
    "acc", [(0.0, "#FFFFFF"), (0.34, LIGHT), (0.70, MUTED), (1.0, ACCENT)])
fig, ax = plt.subplots(figsize=(6.9, 3.9))
fig.subplots_adjust(left=0.085, right=0.995, top=0.855, bottom=0.135)
im = ax.pcolormesh(B, R, Z, cmap=cmap, shading="nearest", vmin=0, vmax=0.20)

# veil the underpowered region instead of hatching it
veil = np.where(M > 0.05, 0.62, 0.0)
ax.pcolormesh(B, R, veil, cmap=LinearSegmentedColormap.from_list("v", ["#ffffff00", "#ffffff"]),
              shading="nearest", vmin=0, vmax=1, zorder=2)
ax.contour(B, R, (M > 0.05).astype(float), levels=[0.5],
           colors=DEAD, linewidths=1.0, linestyles="dashed", zorder=3)

cs = ax.contour(B, R, Z, levels=[0.05], colors="k", linewidths=1.1, zorder=4)
ax.clabel(cs, fmt={0.05: "0.05"}, fontsize=8.0, inline=True, inline_spacing=6)

ax.plot(B, [0.1994]*len(B), color="w", lw=1.6, zorder=5)
ax.plot(B, [0.1994]*len(B), color="k", lw=0.9, ls=(0, (5, 2)), zorder=6)
# Label parked at the right-hand end of the line it names. Anchored at the left
# of the panel it lay across the 0.05 contour.
ax.text(B[-1], 0.1994 + 0.006, "original E9 operating point", ha="right", va="bottom",
        fontsize=8.0, color="k", zorder=7,
        bbox=dict(boxstyle="square,pad=0.2", facecolor="white", alpha=0.85,
                  edgecolor="none"))

ax.set_xlabel(r"selection strength through documentation volume  ($\beta$)")
ax.set_ylabel("predicted-positive rate")
if SHOW_INFIGURE_TITLES: ax.set_title("The audit never becomes cleanly blind; it becomes unmeasurable",
             loc="left", fontsize=9.5, pad=16)
# Two lines above the axes: on one line this key ran the full width of the
# figure, over the colourbar and off the right-hand edge.
ax.text(0, 1.075, "solid line: 0.05 tolerance", transform=ax.transAxes,
        fontsize=8.0, color=MUTED, va="bottom")
ax.text(0, 1.015, "shaded pale: minimum detectable gap exceeds tolerance "
        "(underpowered, not passing)", transform=ax.transAxes,
        fontsize=8.0, color=MUTED, va="bottom")

cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.043)
cb.set_label("demographic-parity gap", fontsize=8.0)
cb.ax.tick_params(labelsize=8); cb.outline.set_linewidth(0.5)

out = pathlib.Path(__file__).parent / "fig-threshold-surface.pdf"
fig.savefig(out)
n_under = int((M > 0.05).sum())
print(f"wrote {out.name}  ({Z.shape[0]}x{Z.shape[1]} cells, {n_under} underpowered)")
