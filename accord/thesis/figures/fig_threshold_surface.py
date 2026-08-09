"""F2 -- threshold robustness surface.

18 operating points x 15 selection strengths from C40. Colour is the
demographic-parity gap. The solid contour marks the 0.05 tolerance. The
underpowered region -- where the minimum detectable gap exceeds tolerance,
so the test cannot see rather than fails to find -- is veiled rather than
hatched, so it reads as "less certain" without obscuring the data beneath.
"""
import pathlib, numpy as np, pandas as pd
from style import setup, ACCENT
setup()
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

df = pd.read_csv("/Users/rickygole/mimic_local_only/accord_selection/threshold_robustness_cells.csv")
piv = df.pivot_table(index="achieved_rate", columns="beta",
                     values="demographic_parity_gap", aggfunc="mean")
mde = df.pivot_table(index="achieved_rate", columns="beta",
                     values="mde_demographic_parity_gap", aggfunc="mean")
R, B, Z, M = piv.index.values, piv.columns.values, piv.values, mde.values

cmap = LinearSegmentedColormap.from_list("acc", ["#fbfcfd", "#cdd7e0", "#8ba0b6", ACCENT])
fig, ax = plt.subplots(figsize=(6.8, 3.8))
im = ax.pcolormesh(B, R, Z, cmap=cmap, shading="nearest", vmin=0, vmax=0.20)

# veil the underpowered region instead of hatching it
veil = np.where(M > 0.05, 0.62, 0.0)
ax.pcolormesh(B, R, veil, cmap=LinearSegmentedColormap.from_list("v", ["#ffffff00", "#ffffff"]),
              shading="nearest", vmin=0, vmax=1, zorder=2)
ax.contour(B, R, (M > 0.05).astype(float), levels=[0.5],
           colors="0.45", linewidths=0.8, linestyles="dashed", zorder=3)

cs = ax.contour(B, R, Z, levels=[0.05], colors="k", linewidths=1.1, zorder=4)
ax.clabel(cs, fmt={0.05: "0.05"}, fontsize=7, inline=True, inline_spacing=6)

ax.plot(B, [0.1994]*len(B), color="w", lw=1.6, zorder=5)
ax.plot(B, [0.1994]*len(B), color="k", lw=0.9, ls=(0, (5, 2)), zorder=6)
ax.annotate("original E9 operating point", xy=(B[1], 0.1994), xytext=(B[1], 0.132),
            fontsize=7, color="k", zorder=7,
            arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))

ax.set_xlabel(r"selection strength through documentation volume  ($\beta$)")
ax.set_ylabel("predicted-positive rate")
ax.set_title("The audit never becomes cleanly blind — it becomes unmeasurable",
             loc="left", fontsize=9.5, pad=16)
ax.text(0, 1.02, "solid line: 0.05 tolerance      shaded pale: minimum detectable gap "
        "exceeds tolerance (underpowered, not passing)",
        transform=ax.transAxes, fontsize=6.9, color="0.4", va="bottom")

cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.043)
cb.set_label("demographic-parity gap", fontsize=8)
cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(0.5)

out = pathlib.Path(__file__).parent / "fig-threshold-surface.pdf"
fig.savefig(out)
n_under = int((M > 0.05).sum())
print(f"wrote {out.name}  ({Z.shape[0]}x{Z.shape[1]} cells, {n_under} underpowered)")
