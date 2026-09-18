"""Fig 10 -- threshold robustness surface.

18 operating points x 15 selection strengths from C40. Colour is the
demographic-parity gap on viridis: perceptually uniform, monotone in lightness
so the greyscale reading survives, and safe under dichromacy. It is the only
heatmap in the set and it is retained.

THE ENCODING FIX. The v2 version washed the underpowered region with white,
which is the SAME CHANNEL as the data: a washed dark cell is indistinguishable
from an unwashed pale one, so a figure whose whole point is that two thirds of
its surface cannot answer was drawing announced noise at the strength of
signal. The region now moves to a different channel entirely -- a flat MIST
overlay plus 45 degree SLATE hatching behind a hard GRAPHITE boundary. No
alpha, no desaturation. The values under it are deliberately unreadable, which
is the honest statement: that region cannot answer.

Every on-field label has left the field. The two contour labels sit in the
right margin at the y where their contour exits the plot, horizontal; the
underpowered note is one line under the x-axis label. Nothing is rotated and
nothing sits over data, so there is no inline contour labelling left to leave a
stray glyph behind.

Read from results/accord/selection/threshold_robustness_results.json, the
committed record.
"""
import json, pathlib, numpy as np, pandas as pd
import style
from style import (FIGW, INK, PAPER, GRAPHITE, MUTE, SLATE, MIST, HAIR,
                   SCARLET_DEEP, SCARLET_CORE,
                   T_AXIS, T_TICK, T_SERIES, T_NOTE, LW_SPINE, LW_DATA,
                   canvas, frame, signature, case)
style.setup()
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[3]
d_thresh = json.loads((ROOT / "results/accord/selection/"
                       "threshold_robustness_results.json").read_text())
df = pd.DataFrame(d_thresh["cells"])
piv = df.pivot_table(index="achieved_rate", columns="beta",
                     values="demographic_parity_gap", aggfunc="mean")
mde = df.pivot_table(index="achieved_rate", columns="beta",
                     values="mde_demographic_parity_gap", aggfunc="mean")
R, B, Z, M = piv.index.values, piv.columns.values, piv.values, mde.values
e9_rate = d_thresh["original_e9_population_pred_pos_rate"]

# Named setting: the audit's own demographic-parity tolerance.
TOLERANCE = 0.05

plt.rcParams["hatch.color"] = SLATE
plt.rcParams["hatch.linewidth"] = 0.7

# 2.85in, not the 3.06 a proportional growth would give: at 0.98 of the measure
# 3.06 puts the block at 4.50in and this figure into class B, and its whole
# design brief was a single-page-share float. The width takes the growth
# instead, and the heatmap field goes from 3.28in wide to 3.77in, which is
# where the illegibility actually was.
fig = plt.figure(figsize=(FIGW, 2.85))
canvas(fig)
ax = fig.add_axes([0.08973, 0.20828, 0.64473, 0.60002])
# The colourbar sits 0.017 further left than an absolute-margin conversion puts
# it: its rotated label plus its widest tick ran 6.5pt past the right edge and
# the tight box grew to catch them, which costs the drawn-at-printed-size
# invariant the whole geometry rests on.
cax = fig.add_axes([0.89000, 0.20828, 0.02564, 0.60002])
im = ax.pcolormesh(B, R, Z, cmap=plt.get_cmap("viridis"), shading="nearest",
                   vmin=0, vmax=0.20)

under = (M > TOLERANCE).astype(float)
# A different channel, not a weaker version of the same one.
ax.contourf(B, R, under, levels=[0.5, 1.5], colors=[MIST], zorder=2)
ax.contourf(B, R, under, levels=[0.5, 1.5], colors=["none"], hatches=["////"],
            zorder=2.1)
ax.contour(B, R, under, levels=[0.5], colors=[GRAPHITE], linewidths=1.2,
           zorder=3)
# THE SMALL CAP ON THE BOTTOM EDGE NEAR beta = 0.5 IS DATA, NOT A LABELLING
# ARTIFACT. At the lowest predicted-positive rate the parity gap runs
# 0.0132, 0.0543, 0.0295 across beta = 0, 0.5, 1.0, so it crosses the 0.05
# tolerance up and back down inside one column. contour returns it as its own
# three-point segment, [(0.448, 0.090), (0.500, 0.106), (0.586, 0.090)], and it
# is drawn for the same reason the long branches are. Verified by rendering
# allsegs with no labelling of any kind in the figure. Do not delete it.
cs_tol = ax.contour(B, R, Z, levels=[TOLERANCE], colors=[GRAPHITE],
                    linewidths=1.2, zorder=4)

ax.set_xlabel(r"Selection strength through documentation volume ($\beta$)")
ax.set_ylabel("Predicted-positive rate")
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_xlim(B.min(), B.max())
ax.set_ylim(R.min(), R.max())
frame(ax)

# The E9 operating point: the one scarlet element, and it is the claim.
ax.plot(B, [e9_rate] * len(B), color=SCARLET_CORE, lw=LW_DATA, zorder=6)
case(ax.text(B[-1], e9_rate + 0.012, "the audit\u2019s operating point",
             ha="right", va="bottom", fontsize=T_SERIES, color=SCARLET_DEEP,
             zorder=7))


def exit_y(cset):
    """The y at which a contour leaves the plot on the right, so its label can
    sit in the margin beside it instead of riding the line."""
    best = None
    for seg in cset.allsegs[0]:
        if len(seg) == 0:
            continue
        i = int(np.argmax(seg[:, 0]))
        if best is None or seg[i, 0] > best[0]:
            best = (seg[i, 0], seg[i, 1])
    return None if best is None else best[1]


def exit_x_top(cset, y_top, tol=1e-9):
    """The x values at which a contour leaves the plot through the TOP edge.

    The tolerance contour never reaches the right-hand spine: both of its
    branches run off the top, at beta = 0.73 and beta = 2.59. Labelled at the
    right margin it pointed at a spine the line does not touch, so the label
    goes above the top edge at the branch's own exit x instead.
    """
    out = []
    for seg in cset.allsegs[0]:
        for x, y in seg:
            if abs(y - y_top) <= tol + 1e-12:
                out.append(float(x))
    return sorted(out)


AX_X0, AX_Y0, AX_W, AX_H = 0.08973, 0.20828, 0.64473, 0.60002
y_pow = exit_y(ax.contour(B, R, under, levels=[0.5], colors=["none"],
                          linewidths=0, zorder=0))
margin_x = 0.74170
if y_pow is not None:
    yf = AX_Y0 + AX_H * (y_pow - R.min()) / (R.max() - R.min())
    fig.text(margin_x, yf, "power limit", ha="left", va="center",
             fontsize=T_SERIES, color=GRAPHITE)
    fig.patches.append(plt.Rectangle(
        (0.73446, yf - 0.0022), 0.005467, 0.0044, transform=fig.transFigure,
        facecolor=GRAPHITE, edgecolor="none", zorder=8))

tol_x = exit_x_top(cs_tol, R.max())
if tol_x:
    # The rightmost of the two top exits: the left one at beta = 0.73 would
    # centre a 51pt label off the left spine.
    xt = tol_x[-1]
    xf = AX_X0 + AX_W * (xt - B.min()) / (B.max() - B.min())
    fig.patches.append(plt.Rectangle(
        (xf - 0.0010, AX_Y0 + AX_H), 0.0020, 3.0 / (72 * 2.85),
        transform=fig.transFigure, facecolor=GRAPHITE, edgecolor="none",
        zorder=8))
    fig.text(xf, AX_Y0 + AX_H + 5.0 / (72 * 2.85), f"{TOLERANCE} tolerance",
             ha="center", va="bottom", fontsize=T_SERIES, color=GRAPHITE)

# The inability statement is carried on canvas by the headline and by the
# hatching itself; the sentence naming the hatched region is the caption's job
# and the caption says it verbatim (msu_ch5_discussion.tex). It was set twice.

cb = fig.colorbar(im, cax=cax)
cb.set_label("Demographic-parity gap", fontsize=T_AXIS)
cb.set_ticks([0, 0.05, 0.10, 0.15, 0.20])
cb.ax.tick_params(labelsize=T_TICK, width=LW_SPINE, length=3.2)
cb.outline.set_visible(True)
cb.outline.set_linewidth(LW_SPINE)
cb.outline.set_edgecolor(INK)

signature(fig, SCARLET_DEEP,
          "The audit passed at an operating point where nothing was detectable.")
out = pathlib.Path(__file__).parent / "fig-threshold-surface.pdf"
fig.savefig(out)
n_under = int((M > TOLERANCE).sum())
print(f"wrote {out.name}  ({Z.shape[0]}x{Z.shape[1]} cells, {n_under} underpowered)")
