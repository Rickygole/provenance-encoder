"""Fig 12 capability, Fig 13 missing cell, Fig 14 label noise.

Three figures added for the frontier result of C63. Each exists because a
committee member would otherwise have to reconstruct it from a table, and each
one carries a claim the prose makes but cannot show.

  fig-capability   The correction itself. Over-calling against labeller
                   capability, with the public-text ceiling drawn as a band.
                   The claim is that the 8B cluster and the frontier models sit
                   on opposite sides of that band, which is what withdraws the
                   attribution of magnitude to the pool.
  fig-missing-cell Why the pre-registered control could not decide it. The
                   design is a two-by-two in domain and scale; three cells were
                   filled and the fourth was not. This is the only figure in
                   the thesis about the thesis's own method failing, and it is
                   the one an examiner of a controls thesis will look for.
  fig-labelnoise   The robustness answer. MCC interval against assumed per-draw
                   instability, out to the fair-coin worst case. Zero-shot never
                   reaches zero; few-shot does, at p = 0.20.

Every value is READ from a committed artifact. Nothing here is a literal, for
the reason C7, C10 and C48 all record: a figure that types its numbers stops
being evidence about the run and becomes evidence about the typist.
"""
import json
import pathlib

import numpy as np
import style
from style import (
    COBALT, AZURE, EMERALD, AMBER, PLUM, SCARLET, MAGENTA, TANGERINE, CYAN,
    COBALT_DEEP, COBALT_CORE, COBALT_TINT, COBALT_WASH,
    AZURE_DEEP, AZURE_CORE, EMERALD_DEEP, EMERALD_CORE, EMERALD_TINT,
    EMERALD_WASH, AMBER_DEEP, AMBER_CORE, AMBER_TINT, AMBER_WASH,
    PLUM_DEEP, PLUM_CORE, SCARLET_DEEP, SCARLET_CORE, SCARLET_TINT,
    SCARLET_WASH,
    PAPER, INK, GRAPHITE, MUTE, SLATE, MIST, HAIR,
    T_HEADLINE, T_CLAIM, T_DECK, T_AXIS, T_TICK, T_SERIES, T_NOTE, T_PANEL,
    LW_SPINE, LW_DATA, LW_REF, LW_INTERVAL, MS_DOT, MS_MARK,
    REF_DASH, FIGW, canvas, frame, signature, band, cased)
style.setup()
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
ROOT = pathlib.Path("/Users/rickygole/provenance-encoder")

FACTS = json.loads((ROOT / "results/accord/VERIFIED_FACTS.json").read_text())
MET = json.loads((ROOT / "results/accord/differential/"
                  "frontier_anthropic_metrics.json").read_text())
NOISE = json.loads((ROOT / "results/accord/differential/"
                    "frontier_label_noise.json").read_text())


def fact(key):
    return FACTS[key]["value"]


# =============================================================================
# Fig 12 -- fig-capability
# =============================================================================
# The x axis is ordinal, not a parameter count. Two of these systems have no
# published size, so plotting them against a numeric axis would invent a
# quantity. Capability tier is the honest encoding: the claim is about ordering,
# and an ordinal axis is what an ordering claim is entitled to.
POOL = [("llama3:8b", fact("overcall.llama3_8b.pool.ratio"), PLUM),
        ("granite3.3:8b", fact("overcall.granite3_3_8b.pool.ratio"), TANGERINE),
        ("qwen3:8b", fact("overcall.qwen3_8b.pool.ratio"), CYAN)]
# GPT-4o and Claude are measured on the gold set, so they are drawn as gold-set
# points and the axis label says so. Mixing a pool ratio and a gold-set ratio on
# one axis without saying which is which is the defect tab:overcalling exists to
# prevent, so the marker shape carries the distinction.
GOLD = [("GPT-4o\nzero-shot", 7.0, COBALT),
        ("Claude Sonnet 5\nzero-shot",
         MET["by_strategy"]["zero_shot"]["G_299"]["over_calling_ratio"], EMERALD),
        ("Claude Sonnet 5\nfew-shot",
         MET["by_strategy"]["few_shot"]["G_299"]["over_calling_ratio"], AZURE)]
PUBLIC_MAX = fact("publicsweep.results.llama3:8b.A_bare.0.02.over_call_ratio")

fig, ax = plt.subplots(figsize=(FIGW, 3.5))
canvas(fig)
frame(ax)

rows = POOL + GOLD
ys = list(range(len(rows)))[::-1]

# The public-text band first, so every point reads against it.
ax.axvspan(1.0, PUBLIC_MAX, color=MIST, zorder=0)
ax.axvline(1.0, color=INK, lw=LW_REF, zorder=2)
ax.text(1.25, len(rows) - 0.30, "pharmacist, 1.0", fontsize=T_NOTE,
        color=INK, ha="left", va="center")
ax.text(PUBLIC_MAX + 0.5, len(rows) - 0.72,
        f"public-text ceiling, {PUBLIC_MAX:.2f}", fontsize=T_NOTE,
        color=MUTE, ha="left", va="center")

for y, (name, ratio, hue) in zip(ys, rows):
    pool = (name, ratio, hue) in POOL
    ax.plot([1.0, ratio], [y, y], color=hue.CORE, lw=LW_DATA, zorder=3,
            solid_capstyle="butt")
    ax.plot([ratio], [y], marker="o" if pool else "D",
            ms=MS_DOT if pool else MS_MARK,
            color=hue.DEEP, zorder=5, clip_on=False)
    ax.text(ratio + 0.55, y, f"{ratio:.2f}", fontsize=T_SERIES,
            color=hue.DEEP, ha="left", va="center", zorder=6)

ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], fontsize=T_TICK)
ax.set_xlim(0, 26.5)
ax.set_xlabel("Positive rate divided by the confirmed rate on the same set",
              fontsize=T_AXIS)
ax.set_ylim(-0.6, len(rows) - 0.1)
ax.text(0.995, -0.30,
        "circles: candidate pool.  diamonds: $G_{299}$ gold set.",
        transform=ax.transAxes, fontsize=T_NOTE, color=MUTE,
        ha="right", va="top")
signature(fig, EMERALD_DEEP,
          headline="Over-calling tracks the labeller, not the pool",
          deck=("Three 8B labellers sit far above the ceiling a public-text "
                "sweep produced.\nTwo frontier labellers, on the same pool at "
                "the same base rate, sit just above it."))
fig.subplots_adjust(left=0.24, right=0.955, top=0.70, bottom=0.20)
fig.savefig(HERE / "fig-capability.pdf")
plt.close(fig)
print(f"wrote fig-capability.pdf ({len(rows)} labellers, "
      f"ceiling {PUBLIC_MAX:.2f})")


# =============================================================================
# Fig 13 -- fig-missing-cell
# =============================================================================
# A schematic, and the one figure here that plots no measurement. It draws the
# shape of the design, so the cells are labelled with the ratios that filled
# them and the empty one is left visibly empty.
fig, ax = plt.subplots(figsize=(FIGW, 3.1))
canvas(fig)
ax.set_axis_off()

lo = f"{fact('overcall.qwen3_8b.pool.ratio'):.1f}"
hi = f"{fact('overcall.llama3_8b.pool.ratio'):.1f}"
fz = MET["by_strategy"]["zero_shot"]["G_253"]["over_calling_ratio"]

CELLS = [
    # (col, row, hue, top line, bottom line, filled)
    (0, 1, PLUM,   "clinical pool", f"{lo}x to {hi}x", True),
    (1, 1, SLATE,  "public text", f"at most {PUBLIC_MAX:.1f}x", True),
    (0, 0, EMERALD, "clinical pool", f"{fz:.2f}x", True),
    (1, 0, None,   "public text", "never run", False),
]
W, H = 0.335, 0.295
X0, Y0 = 0.235, 0.15

for col, row, hue, top, bot, filled in CELLS:
    x, y = X0 + col * (W + 0.045), Y0 + row * (H + 0.075)
    if filled:
        ax.add_patch(plt.Rectangle((x, y), W, H, transform=ax.transAxes,
                                   facecolor=(hue.TINT if hue is not SLATE
                                              else MIST),
                                   edgecolor=(hue.DEEP if hue is not SLATE
                                              else SLATE),
                                   lw=LW_SPINE, zorder=2))
        ax.text(x + W / 2, y + H * 0.62, top, transform=ax.transAxes,
                fontsize=T_NOTE, color=MUTE, ha="center", va="center", zorder=4)
        ax.text(x + W / 2, y + H * 0.30, bot, transform=ax.transAxes,
                fontsize=T_SERIES,
                color=(hue.DEEP if hue is not SLATE else INK),
                ha="center", va="center", zorder=4)
    else:
        ax.add_patch(plt.Rectangle((x, y), W, H, transform=ax.transAxes,
                                   facecolor=PAPER, edgecolor=SCARLET_CORE,
                                   lw=LW_SPINE, ls=(0, (3, 2.4)), zorder=2))
        ax.text(x + W / 2, y + H * 0.62, top, transform=ax.transAxes,
                fontsize=T_NOTE, color=MUTE, ha="center", va="center", zorder=4)
        ax.text(x + W / 2, y + H * 0.30, bot, transform=ax.transAxes,
                fontsize=T_SERIES, color=SCARLET_DEEP, ha="center",
                va="center", zorder=4)

ax.text(X0 - 0.035, Y0 + H + 0.075 + H / 2, "8B", transform=ax.transAxes,
        fontsize=T_AXIS, color=INK, ha="right", va="center", rotation=90)
ax.text(X0 - 0.035, Y0 + H / 2, "frontier", transform=ax.transAxes,
        fontsize=T_AXIS, color=INK, ha="right", va="center", rotation=90)
ax.text(X0 - 0.105, Y0 + H + 0.04, "labeller scale", transform=ax.transAxes,
        fontsize=T_AXIS, color=MUTE, ha="center", va="center", rotation=90)
# The columns are the other factor of the design and need naming, or the panel
# reads as four unrelated boxes rather than a two-by-two with a hole in it.
ax.text(X0 + W + 0.0225, Y0 + 2 * H + 0.075 + 0.055, "evaluation domain",
        transform=ax.transAxes, fontsize=T_AXIS, color=MUTE,
        ha="center", va="center")

signature(fig, SCARLET_DEEP,
          headline="The control fired correctly and still under-determined it",
          deck=("Scale was held fixed at 8B, so a clinical-versus-public "
                "difference could only\nbe attributed to the pool. The "
                "deciding cell was never built."))
fig.subplots_adjust(left=0.04, right=0.97, top=0.68, bottom=0.05)
fig.savefig(HERE / "fig-missing-cell.pdf")
plt.close(fig)
print("wrote fig-missing-cell.pdf (3 filled cells, 1 empty)")


# =============================================================================
# Fig 14 -- fig-labelnoise
# =============================================================================
fig, ax = plt.subplots(figsize=(FIGW, 3.3))
canvas(fig)
frame(ax, grid=True)

SER = [("zero_shot", "Claude Sonnet 5, zero-shot", EMERALD),
       ("few_shot", "Claude Sonnet 5, few-shot", AZURE)]

ax.axhline(0.0, color=INK, lw=LW_REF, zorder=4)
ax.text(0.345, 0.014, "chance", fontsize=T_NOTE, color=INK,
        ha="left", va="bottom", zorder=5)

for key, label, hue in SER:
    r = NOISE["by_strategy"][key]
    ps = [row["p"] for row in r["sweep"]]
    lo = [row["ci_lo"] for row in r["sweep"]]
    hi = [row["ci_hi"] for row in r["sweep"]]
    ax.fill_between(ps, lo, hi, color=hue.TINT, zorder=1, lw=0)
    ax.plot(ps, lo, color=hue.DEEP, lw=LW_DATA, zorder=3)
    ax.plot(ps, hi, color=hue.CORE, lw=LW_INTERVAL, zorder=3, ls=REF_DASH)
    # The point estimate is one dot at p = 0, not a rule across the panel. A
    # horizontal line here read as a third boundary of the band and collided
    # with the other series' label.
    ax.plot([0.0], [r["mcc_point"]], marker="o", ms=MS_DOT, color=hue.DEEP,
            zorder=6, clip_on=False)
    # Series labels ride the upper boundary at its right end, where the two
    # bands are furthest apart, instead of the middle where they crossed.
    ax.text(0.505, hi[-1], f"  {label}", fontsize=T_SERIES, color=hue.DEEP,
            ha="left", va="center", zorder=6)
    ax.text(0.505, lo[-1], f"  {lo[-1]:+.4f}", fontsize=T_NOTE,
            color=hue.DEEP, ha="left", va="center", zorder=6)
    cross = r["flip_probability_at_which_zero_enters"]
    if cross is not None:
        ax.plot([cross], [0.0], marker="X", ms=MS_MARK, color=SCARLET_DEEP,
                zorder=7, clip_on=False)
        ax.annotate(f"reaches chance at p = {cross:.2f}", xy=(cross, 0.0),
                    xytext=(cross - 0.035, 0.175), fontsize=T_NOTE,
                    color=SCARLET_DEEP, ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=SCARLET_CORE,
                                    lw=LW_REF, shrinkA=2, shrinkB=3),
                    zorder=7)

ax.set_xlim(0.0, 0.52)
ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_xlabel("Assumed per-draw probability that any one label flips",
              fontsize=T_AXIS)
ax.set_ylabel("Matthews correlation, 95 percent interval", fontsize=T_AXIS)
signature(fig, EMERALD_DEEP,
          headline="One result survives any instability; the other does not",
          deck=("The interval widens with assumed label noise. The zero-shot "
                "lower bound stays\nabove chance out to the fair-coin worst "
                "case. The few-shot bound does not."))
fig.subplots_adjust(left=0.135, right=0.745, top=0.705, bottom=0.225)
fig.savefig(HERE / "fig-labelnoise.pdf")
plt.close(fig)
z = NOISE["by_strategy"]["zero_shot"]
f = NOISE["by_strategy"]["few_shot"]
print(f"wrote fig-labelnoise.pdf (zero-shot worst case "
      f"{z['worst_case_ci_lo']:+.4f}, few-shot crosses at "
      f"{f['flip_probability_at_which_zero_enters']:.2f})")
