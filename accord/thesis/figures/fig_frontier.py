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
                   reaches zero; few-shot does, at p = 0.1159.

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
    REF_DASH, FIGW, canvas, frame, signature, band, cased, wilson_ci)
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
# Every point is on G_299, the gold set, and the register carries a gold-set
# ratio for all five labellers. An earlier version mixed candidate-pool ratios
# for the 8B models with gold-set ratios for the frontier ones and carried a
# two-marker legend to excuse it; the mixing was never necessary. It also
# plotted GPT-4o at a literal 7.0 that traces to no artifact, in a module whose
# docstring says nothing here is a literal. GPT-4o's committed cells are on
# G_253 only, so it is named in the caption and not drawn on this axis.
GROUPS = [
    ("eight billion parameters", [
        ("llama3:8b", "overcall.llama3_8b.gold", MAGENTA),
        ("granite3.3:8b", "overcall.granite3_3_8b.gold", TANGERINE),
        ("qwen3:8b", "overcall.qwen3_8b.gold", CYAN)]),
    ("frontier", [
        ("Claude Sonnet 5, zero-shot", None, EMERALD),
        ("Claude Sonnet 5, few-shot", None, AZURE)]),
]
PUBLIC_MAX = fact("publicsweep.results.llama3:8b.A_bare.0.02.over_call_ratio")
BASE = fact("gold.base_rate_299")
N299 = 299

rows = []
for gname, members in GROUPS:
    block = []
    for label, key, hue in members:
        if key:
            rate = fact(f"{key}.positive_rate")
            ratio = fact(f"{key}.ratio")
        else:
            s = "zero_shot" if "zero" in label else "few_shot"
            c = MET["by_strategy"][s]["G_299"]
            rate, ratio = c["flag_rate"], c["over_calling_ratio"]
        k = int(round(rate * N299))
        lo, hi = wilson_ci(k, N299)
        block.append((label, ratio, lo / BASE, hi / BASE, hue))
    rows.append((gname, block))

fig, ax = plt.subplots(figsize=(FIGW, 3.6))
canvas(fig)
frame(ax)

# One y slot per labeller, plus a blank slot between the two groups so the
# separation is structural rather than something the reader has to infer from
# the values. Two blocks with a gap IS the two-cluster claim.
ypos, ylab, y = [], [], 0
for gi, (gname, block) in enumerate(rows):
    if gi:
        y -= 1
    for label, ratio, lo, hi, hue in block:
        ypos.append((y, label, ratio, lo, hi, hue, gname))
        ylab.append(label)
        y -= 1

# The ceiling is a measured maximum, not a tolerance region. Drawn as a rule.
ax.axvline(1.0, color=INK, lw=LW_REF, zorder=4)
ax.axvline(PUBLIC_MAX, color=SLATE, lw=LW_REF, ls=REF_DASH, zorder=4)

for yv, label, ratio, lo, hi, hue, gname in ypos:
    # Stem from the pharmacist's 1.0 to the point, then the Wilson interval on
    # top of it. The stem is the claim -- how far above calibrated the labeller
    # sits -- and dropping it for the interval alone left a dot plot in which
    # the distance being asserted was carried by the axis alone.
    ax.plot([1.0, ratio], [yv, yv], color=hue.TINT, lw=LW_DATA, zorder=2,
            solid_capstyle="butt")
    ax.plot([lo, hi], [yv, yv], color=hue.CORE, lw=LW_INTERVAL, zorder=3,
            solid_capstyle="butt")
    ax.plot([ratio], [yv], marker="o", ms=MS_DOT, color=hue.DEEP, zorder=5)
    ax.text(hi + 0.55, yv, f"{ratio:.2f}", fontsize=T_SERIES,
            color=hue.DEEP, ha="left", va="center", zorder=6)

ax.set_yticks([r[0] for r in ypos])
ax.set_yticklabels([r[1] for r in ypos], fontsize=T_TICK)
ax.set_xlim(0, 27.5)
ax.set_ylim(min(r[0] for r in ypos) - 1.0, 0.9)
ax.set_xlabel("Positive rate on $G_{299}$ divided by that set's confirmed rate",
              fontsize=T_AXIS)

# group rules, in the margin, so the two blocks are named
for gname, block in rows:
    ys = [r[0] for r in ypos if r[6] == gname]
    ax.text(-0.455, (max(ys) + min(ys)) / 2.0, gname,
            transform=ax.get_yaxis_transform(), fontsize=T_NOTE, color=MUTE,
            ha="center", va="center", rotation=90)

_ybot = min(r[0] for r in ypos)
ax.text(1.3, 0.5, "pharmacist, 1.0", color=INK,
        ha="left", va="center", **cased(T_NOTE))
ax.text(PUBLIC_MAX + 0.45, _ybot - 0.55,
        f"public-text maximum, {PUBLIC_MAX:.2f}",
        color=MUTE, ha="left", va="center", **cased(T_NOTE))
signature(fig, EMERALD_DEEP,
          headline="Over-calling tracks the labeller, not the pool",
          deck=("Same pool, same base rate, same prompt. Three 8B labellers "
                "run four to seven times\nthe public-text maximum; two frontier "
                "labellers run close to it. Bars are Wilson intervals."))
fig.subplots_adjust(left=0.395, right=0.955, top=0.705, bottom=0.175)
fig.savefig(HERE / "fig-capability.pdf")
plt.close(fig)
print(f"wrote fig-capability.pdf ({len(ypos)} labellers, one denominator, "
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

# Both clinical cells are on G_299. An earlier version put pool ratios in the
# 8B cell and a G_253 ratio in the frontier cell, so the two cells in a column
# headed "clinical" were computed on different denominators, and a caption was
# doing work a mislabelled cell had made impossible.
lo = f"{fact('overcall.qwen3_8b.gold.ratio'):.1f}"
hi = f"{fact('overcall.llama3_8b.gold.ratio'):.1f}"
fz = MET["by_strategy"]["zero_shot"]["G_299"]["over_calling_ratio"]

CELLS = [
    # (col, row, hue, top line, bottom line, filled)
    (0, 1, MAGENTA, "clinical, $G_{299}$", f"{lo}x to {hi}x", True),
    (1, 1, SLATE,  "public text", f"at most {PUBLIC_MAX:.1f}x", True),
    (0, 0, EMERALD, "clinical, $G_{299}$", f"{fz:.2f}x", True),
    (1, 0, None,   "public text", "never run", False),  # see below
]
W, H = 0.345, 0.305
X0, Y0 = 0.185, 0.14

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
        ax.text(x + W / 2, y + H * 0.42, bot, transform=ax.transAxes,
                fontsize=T_SERIES, color=SCARLET_DEEP, ha="center",
                va="center", zorder=4)
        # The empty cell is the figure's subject, so it carries what filling it
        # would have decided rather than only the fact that it was skipped.
        ax.text(x + W / 2, y + H * 0.17,
                "would separate the two readings", transform=ax.transAxes,
                fontsize=T_NOTE, color=MUTE, ha="center", va="center",
                zorder=4)

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
                "difference could only\nbe attributed to the domain. The "
                "deciding cell was never built."))
fig.subplots_adjust(left=0.04, right=0.97, top=0.68, bottom=0.05)
fig.savefig(HERE / "fig-missing-cell.pdf")
plt.close(fig)
print("wrote fig-missing-cell.pdf (3 filled cells, 1 empty)")


# =============================================================================
# Fig 14 -- fig-labelnoise
# =============================================================================
# Two stacked panels, not one. A single axes required two overlapping area
# fills, and with alpha banned by the style system the second fill painted the
# first out entirely: the zero-shot band, which is the subject of the headline,
# was invisible under the few-shot band. Separate panels also give each series
# its own zero rule, so "stays above chance" and "reaches chance" are each read
# against a line inside their own frame, and they end the label collisions at
# the right spine, where a clipped plus sign printed +0.0829 as -0.0829.
SER = [("zero_shot", "Claude Sonnet 5, zero-shot", EMERALD),
       ("few_shot", "Claude Sonnet 5, few-shot", AZURE)]

fig, axes = plt.subplots(2, 1, figsize=(FIGW, 4.05), sharex=True)
canvas(fig)

for ax, (key, label, hue) in zip(axes, SER):
    frame(ax, grid=True)
    r = NOISE["by_strategy"][key]
    ps = [row["p"] for row in r["sweep"]]
    lo = [row["ci_lo"] for row in r["sweep"]]
    hi = [row["ci_hi"] for row in r["sweep"]]
    ax.fill_between(ps, lo, hi, color=hue.TINT, zorder=1, lw=0)
    ax.plot(ps, lo, color=hue.DEEP, lw=LW_DATA, zorder=3)
    ax.plot(ps, hi, color=hue.DEEP, lw=LW_DATA, zorder=3)
    ax.axhline(0.0, color=INK, lw=LW_REF, zorder=4)
    ax.plot([0.0], [r["mcc_point"]], marker="o", ms=MS_DOT, color=hue.DEEP,
            zorder=6)
    ax.text(0.012, r["mcc_point"], f"  {r['mcc_point']:+.4f} at $p=0$",
            color=hue.DEEP, ha="left", va="center", zorder=6,
            **cased(T_NOTE))
    # Series name and worst case inside the panel, well clear of both spines.
    ax.text(0.485, 0.90, label, transform=ax.transAxes,
            color=hue.DEEP, ha="right", va="top", zorder=6, **cased())
    end = r["sweep"][-1]
    cross = r["flip_probability_at_which_zero_enters"]
    if cross is None:
        ax.text(0.485, 0.13, f"worst case {end['ci_lo']:+.4f}, still above chance",
                transform=ax.transAxes, color=hue.DEEP,
                ha="right", va="bottom", zorder=6, **cased(T_NOTE))
    else:
        ax.plot([cross], [0.0], marker="X", ms=MS_MARK, color=SCARLET_DEEP,
                zorder=7)
        ax.text(0.485, 0.13,
                f"reaches chance at $p = {cross:.4f}$", transform=ax.transAxes,
                color=SCARLET_DEEP, ha="right", va="bottom",
                zorder=7, **cased(T_NOTE))
    ax.set_ylabel(r"Matthews, 95\% interval", fontsize=T_AXIS)

axes[0].text(0.012, 0.012, "chance", fontsize=T_NOTE, color=INK,
             ha="left", va="bottom", zorder=6)
axes[-1].set_xlim(0.0, 0.52)
axes[-1].set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
axes[-1].set_xlabel("Assumed per-draw probability that any one label flips",
                    fontsize=T_AXIS)
signature(fig, EMERALD_DEEP,
          headline="One result survives any instability; the other does not",
          deck=("Each interval widens with assumed label noise while its point "
                "estimate is held fixed.\nThe upper panel never reaches chance. "
                "The lower one does."))
fig.subplots_adjust(left=0.155, right=0.965, top=0.755, bottom=0.115,
                    hspace=0.20)
fig.savefig(HERE / "fig-labelnoise.pdf")
plt.close(fig)
z = NOISE["by_strategy"]["zero_shot"]
f = NOISE["by_strategy"]["few_shot"]
print(f"wrote fig-labelnoise.pdf (zero-shot worst case "
      f"{z['worst_case_ci_lo']:+.4f}, few-shot crosses at "
      f"{f['flip_probability_at_which_zero_enters']:.4f})")
