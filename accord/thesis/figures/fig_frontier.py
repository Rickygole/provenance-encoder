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
# figures -> thesis -> accord -> repository root. Was the author's absolute path.
ROOT = pathlib.Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# ALL THREE FIGURES IN THIS MODULE ARE RETIRED, and the module stops here
# rather than at its first savefig, because two of the artifacts it would
# otherwise read are withdrawn.
#
#   fig-capability   drew four hosted rows: two Bedrock arms, Claude Haiku 4.5
#                    and Claude Sonnet 5. Every one of them was obtained by
#                    transmitting credentialed data to a hosted service, and
#                    every value behind them now sits under withdrawn.hosted.*
#                    in the register. Four of its seven rows cannot be drawn,
#                    its lower panel is entirely a local-versus-hosted
#                    decomposition, and its headline names the channel. There
#                    is no version of this figure that survives the withdrawal.
#   fig-labelnoise   drew two Claude Sonnet 5 series and nothing else.
#   fig-missing-cell was a two-by-two of four rectangles containing text, with
#                    no axis and no scale: a table drawn as a picture. It was
#                    considered for conversion to a 2x2 tabular here and is
#                    deleted instead, because the cell that made it a design
#                    rather than a list is the frontier-clinical cell, and that
#                    cell is Claude Sonnet 5's over-calling ratio. A tabular
#                    would have to print a withdrawn hosted value or print a
#                    hole where the figure's whole argument was.
#
# Retirement reasons of record live in tools/check_figure_invariants.py. The
# committed PDFs are left in place, as every other retired figure's is, so the
# withdrawal is auditable rather than silent; nothing in the thesis binds them.
# ---------------------------------------------------------------------------
print("fig_frontier.py: all three figures retired (two drew hosted rows, the "
      "third depended on one); see RETIRED in tools/check_figure_invariants.py")
raise SystemExit(0)

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
# Every point is on G_299 and on the zero-shot prompt, so the rows are
# comparable. Two earlier versions of this figure were wrong in ways worth
# recording. The first mixed candidate-pool ratios for the 8B models with
# gold-set ratios for the frontier ones and carried a two-marker legend to
# excuse it, and plotted GPT-4o at a literal 7.0 that traces to no artifact. The
# second was correct but stale: it drew three local 8B models and two Sonnet
# conditions and so showed neither Haiku nor either Bedrock arm, which meant the
# figure carrying the capability claim contradicted the table beside it.
#
# Upper panel is the ladder. Lower panel is the decomposition, which is the part
# a reader cannot get from the ladder: three factors move this quantity and the
# ordering among them is the finding. Two panels under one caption was a defect
# in another figure in this set, where the panels shared nothing but the caption.
# Here the lower panel is the upper one read in pairs, which is the same claim at
# two levels rather than two claims.
LADDER = [
    ("llama3:8b", "local", "overcall.llama3_8b.gold", MAGENTA),
    ("meta.llama3-8b-instruct", "Bedrock",
     "frontier_bedrock_metallama38binstructv1.zero_shot.G_299", TANGERINE),
    ("granite3.3:8b", "local", "overcall.granite3_3_8b.gold", CYAN),
    ("qwen3:8b", "local", "overcall.qwen3_8b.gold", CYAN),
    ("meta.llama3-70b-instruct", "Bedrock",
     "frontier_bedrock_metallama370binstructv1.zero_shot.G_299", TANGERINE),
    ("Claude Haiku 4.5", "Anthropic",
     "frontier_anthropic_claudehaiku4520251001.zero_shot.G_299", AZURE),
    ("Claude Sonnet 5", "Anthropic", "frontier.zero_shot.G_299", EMERALD),
]
PUBLIC_MAX = fact("publicsweep.results.llama3:8b.A_bare.0.02.over_call_ratio")
BASE = fact("gold.base_rate_299")
N299 = 299

rows = []
for label, chan, key, hue in LADDER:
    # The two key shapes differ: the local sweep registers positive_rate and
    # ratio, the frontier metrics register flag_rate and over_calling_ratio.
    if key.startswith("overcall."):
        rate, ratio = fact(f"{key}.positive_rate"), fact(f"{key}.ratio")
    else:
        rate, ratio = fact(f"{key}.flag_rate"), fact(f"{key}.over_calling_ratio")
    k = int(round(rate * N299))
    lo, hi = wilson_ci(k, N299)
    rows.append((label, chan, ratio, lo / BASE, hi / BASE, hue))
rows.sort(key=lambda r: -r[2])

fig, axes = plt.subplots(2, 1, figsize=(FIGW, 4.9),
                         gridspec_kw={"height_ratios": [3.0, 1.0]})
canvas(fig)
ax, axd = axes
frame(ax)

ax.axvline(1.0, color=INK, lw=LW_REF, zorder=4)
ax.axvline(PUBLIC_MAX, color=SLATE, lw=LW_REF, ls=REF_DASH, zorder=4)

for i, (label, chan, ratio, lo, hi, hue) in enumerate(rows):
    y = -i
    ax.plot([1.0, ratio], [y, y], color=hue.TINT, lw=LW_DATA, zorder=2,
            solid_capstyle="butt")
    ax.plot([lo, hi], [y, y], color=hue.CORE, lw=LW_INTERVAL, zorder=3,
            solid_capstyle="butt")
    ax.plot([ratio], [y], marker="o", ms=MS_DOT, color=hue.DEEP, zorder=5)
    ax.text(hi + 0.55, y, f"{ratio:.2f}", fontsize=T_SERIES,
            color=hue.DEEP, ha="left", va="center", zorder=6)

ax.set_yticks([-i for i in range(len(rows))])
ax.set_yticklabels([f"{r[0]}  ({r[1]})" for r in rows], fontsize=T_TICK)
ax.set_xlim(0, 27.5)
ax.set_ylim(-len(rows) + 0.45, 0.75)
ax.text(1.3, 0.35, "pharmacist, 1.0", color=INK,
        ha="left", va="center", **cased(T_NOTE))
ax.text(PUBLIC_MAX + 0.45, -len(rows) + 0.95,
        f"public-text maximum, {PUBLIC_MAX:.2f}",
        color=MUTE, ha="left", va="center", **cased(T_NOTE))
ax.set_xlabel("Positive rate on $G_{299}$ divided by that set's confirmed rate",
              fontsize=T_AXIS)

# Lower panel. Each bar is a ratio of two rows above, with everything else held
# fixed, so the bar lengths are directly comparable and the ordering is the
# claim. Channel last because it is smallest, which is the result.
FACTORS = [
    ("capability, 8B to 70B", "family, maker, channel held",
     "capability.within_channel_step_8b_to_70b", EMERALD),
    ("prompt, zero to few-shot", "one model, everything else held",
     "capability.prompt_effect_within_70b", AZURE),
    ("channel, local to hosted", "family, maker, scale held",
     "capability.channel_effect_at_8b_local_over_hosted", TANGERINE),
]
frame(axd)
axd.axvline(1.0, color=INK, lw=LW_REF, zorder=4)
for i, (label, held, key, hue) in enumerate(FACTORS):
    v = fact(key)
    y = -i
    axd.plot([1.0, v], [y, y], color=hue.CORE, lw=LW_DATA * 2.1, zorder=3,
             solid_capstyle="butt")
    axd.plot([v], [y], marker="o", ms=MS_DOT, color=hue.DEEP, zorder=5)
    axd.text(v + 0.045, y, f"{v:.4f}", fontsize=T_SERIES, color=hue.DEEP,
             ha="left", va="center", zorder=6)
    axd.text(2.42, y, held, fontsize=T_NOTE, color=MUTE,
             ha="right", va="center", zorder=6)
axd.set_yticks([-i for i in range(len(FACTORS))])
axd.set_yticklabels([f[0] for f in FACTORS], fontsize=T_TICK)
axd.set_xlim(0.94, 2.45)
axd.set_ylim(-len(FACTORS) + 0.5, 0.6)
axd.set_xlabel("Factor by which the ratio moves when one thing changes",
               fontsize=T_AXIS)

signature(fig, EMERALD_DEEP,
          headline="Capability orders the over-calling; the channel does not",
          deck=("One denominator, one prompt. Upper: seven labellers against the "
                "pharmacist\u2019s rate, bars are Wilson\nintervals. Lower: the same "
                "rows read in pairs. Serving a model remotely costs a tenth of "
                "what scaling it does."))
fig.subplots_adjust(left=0.395, right=0.955, top=0.755, bottom=0.105, hspace=0.62)
fig.savefig(HERE / "fig-capability.pdf")
plt.close(fig)
print(f"wrote fig-capability.pdf ({len(rows)} labellers on one denominator, "
      f"3 factors, ceiling {PUBLIC_MAX:.2f})")


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
# its own zero rule, so each crossing is read against a line inside its own
# frame, and they end the label collisions at the right spine, where a clipped
# plus sign once printed a positive bound as a negative one.
#
# The mean is drawn as well as the interval. The earlier version of this figure
# plotted a single dot at p = 0 and no centre line, because the analysis behind
# it held the point estimate fixed and moved only the bounds. That was the
# error: the centre is where most of the movement is. A figure that shows only
# widening bounds cannot be read against a fair coin, which is exactly the
# reading the withdrawn claim rested on.
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
    means = [row["mean"] for row in r["sweep"]]
    ax.plot(ps, means, color=hue.DEEP, lw=LW_DATA, ls="--", zorder=5)
    ax.plot([0.0], [r["mcc_point"]], marker="o", ms=MS_DOT, color=hue.DEEP,
            zorder=6)
    ax.text(0.012, r["mcc_point"], f"  {r['mcc_point']:+.4f} at $p=0$",
            color=hue.DEEP, ha="left", va="center", zorder=6,
            **cased(T_NOTE))
    # The end of the dashed line is the whole point of the correction: at a fair
    # coin the centre is on the chance rule, not above it.
    ax.text(0.5, means[-1], f"{means[-1]:+.4f}  ", color=hue.DEEP,
            ha="right", va="bottom", zorder=6, **cased(T_NOTE))
    # Series name and worst case inside the panel, well clear of both spines.
    ax.text(0.485, 0.90, label, transform=ax.transAxes,
            color=hue.DEEP, ha="right", va="top", zorder=6, **cased())
    cross = r["flip_probability_at_which_zero_enters"]
    if cross is None:
        end = r["sweep"][-1]
        ax.text(0.485, 0.13,
                f"admits chance nowhere below $p = {end['p']:.2f}$",
                transform=ax.transAxes, color=hue.DEEP,
                ha="right", va="bottom", zorder=6, **cased(T_NOTE))
    else:
        ax.plot([cross], [0.0], marker="X", ms=MS_MARK, color=SCARLET_DEEP,
                zorder=7)
        ax.text(0.485, 0.13,
                f"reaches chance at $p = {cross:.3f}$", transform=ax.transAxes,
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
          headline="Neither result survives arbitrary label instability",
          deck=("Solid bounds are the 95 percent interval, dashed is the mean; "
                "both move, because labels are flipped and pairs\nresampled "
                "jointly. The zero-shot interval admits chance at one label in "
                "four, the few-shot at one in eleven."))
fig.subplots_adjust(left=0.155, right=0.965, top=0.755, bottom=0.115,
                    hspace=0.20)
fig.savefig(HERE / "fig-labelnoise.pdf")
plt.close(fig)
z = NOISE["by_strategy"]["zero_shot"]
f = NOISE["by_strategy"]["few_shot"]
print(f"wrote fig-labelnoise.pdf (zero-shot crosses at "
      f"{z['flip_probability_at_which_zero_enters']:.3f}, few-shot at "
      f"{f['flip_probability_at_which_zero_enters']:.3f})")
