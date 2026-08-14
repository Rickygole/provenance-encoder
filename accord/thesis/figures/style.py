"""Shared figure conventions for the thesis.

One rule specific to this document: no figure may show a point estimate
without its uncertainty. A thesis about instruments that mislead cannot
itself present a bare number.

The visual system, applied to all fourteen bound figures:

  Typography  DejaVu Serif throughout (pinned for script/notebook parity).
              Axis labels 9pt sentence case, ticks and annotations 8pt,
              nothing below 8pt, and no bold anywhere. Multi-panel figures
              carry italic (a), (b) labels outside the axes.
  Colour      A five-hue categorical series palette for the things being
              compared, one reserved accent for the claim, and a quiet
              neutral structure behind both. See SERIES below.
  Estimates   A dot with a thin interval and small caps, never a bar.
  Bars        Counts and rates of a whole only. No outlines, 0.55 of the slot.
  Legends     None at all. Curves, bands and series are labelled at their own
              right edge in their own colour; reference lines at the line.
  Axes        Left and bottom spines only, 0.6pt, clipped to the range they
              measure, ticks outward 3pt and chosen rather than defaulted, no
              grid unless a panel is genuinely read for values.
  Measure     Every bound figure is drawn 5.40 inches wide, the width it is
              printed at, so 8 point text is 8 point on the page.
  Intervals   1pt line, 2.5pt caps, the colour of their own marker.

Three rules the second review settled, recorded here because each one governs
more than one figure:

  Spines      A spine is drawn only on an axis that measures something. A
              categorical axis gets none: a vertical rule running past four
              unrelated row labels is read as a bracket that groups them, and
              nothing in the data groups them. The exception is a bar chart,
              whose left spine is the baseline its bars are measured from.

  Intervals   An interval is drawn in the colour of the mark it belongs to,
              always, including the caps on a bar. An interval in a second
              colour asks the reader to decode a distinction that is not there.

  Scarlet     Scarlet marks the figure's CLAIM ELEMENT: the one mark that
              carries what this figure is asserting. It does not mark an
              entity. ACCORD is not scarlet because it is ours; it is scarlet
              on fig-met-exchange because the whole figure is about where
              ACCORD sits against the exchange rates, and it takes its series
              hue on fig-intro-headline (b) and on fig-forest because those
              two figures claim that everything is at chance, which no single
              mark carries. A figure whose claim is a shape rather than a
              point carries no scarlet at all, and that is the correct
              outcome, not a missing accent. Scarlet is never a series colour
              and no series colour is ever asked to be the claim.

  Intervals   On a bar, the interval is cased: a wider white stroke under the
              bar-coloured interval, so the half that falls inside the fill
              reads against the fill and the half outside reads against the
              paper. Without it a same-colour interval loses its lower half in
              the bar and the estimate reads one-sided, which is the one thing
              this set's first rule exists to prevent. This is the treatment
              for every bar that carries an interval.

  De-emphasis Two jobs that look alike and are not the same job. A
              de-emphasised COMPARATOR is a mark with an interval that the
              figure measures something else against (the pretrained arm on
              fig-scaling): it keeps a full series hue, because it is a
              measured arm, and it recedes by being the reference the other
              arm is read against. A BARE ESTIMATE has no committed interval
              data at all, is drawn without one, and is called underpowered in
              the caption (the transfer scores on fig-falsification): it takes
              its series hue LIGHTENED (see SERIES_BARE), so that the bareness
              is visible in the mark itself and not only in the caption. In
              both cases the label goes in MUTE.

  Intervals   Drawn only from committed data: a confidence interval already in
              a result file, or a closed-form Wilson interval computed from
              counts already in a result file. Per-pair scores are not
              committed anywhere in this repository (the data agreement
              releases aggregates only), so no new bootstrap is possible. An
              estimate with no committed interval is drawn in GREY, labelled
              in MUTE, and called underpowered in the caption. Spread is never
              invented to fill a gap in the record.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Every thesis figure is bound at width=0.9\linewidth in a 6 inch text block, so
# it prints 5.40 inches wide. Each one is therefore drawn 5.40 inches wide
# (FIGW plus the 0.02 inch save pad on each side), which makes the scale factor
# 1.0 and puts 8 point text on the page at 8 point.
FIGW = 5.36
PRINT_W = 5.40

# ---------------------------------------------------------------------------
# STRUCTURE. Unchanged. These five tones draw everything that is not a
# measurement: type, rules, reference lines, secondary bands, annotation.
#
#   Greyscale lightness, 0.299R + 0.587G + 0.114B out of 255:
#       INK      #1A1A1A  near-black       26   text, spines, ticks
#       MUTE     #5B5B5B  grey             91   secondary annotation text
#       GREY     #8A94A6  cool grey       147   reference lines, seed dots
#       PALE     #C9D2E3  pale blue-grey  209   secondary bands
#       GRID     #E8E8E8  hairline grey   232   optional value grid
#
# Text contrast on white: INK 17.4:1, MUTE 6.8:1. GREY is 3.1:1 and is
# therefore never used for text, only for reference lines and de-emphasised
# markers that are also labelled in MUTE.
INK = "#1A1A1A"
PALE = "#C9D2E3"
MUTE = "#5B5B5B"
GREY = "#8A94A6"
GRID = "#E8E8E8"

# ---------------------------------------------------------------------------
# CLAIM. Reserved, and reserved harder than before. SCARLET marks exactly one
# element per figure: the element that is this study, or the element that
# failed. It is never a series colour, and no series colour is ever asked to
# carry a claim. There is never a second red mark on a canvas.
SCARLET = "#D81E05"

# ---------------------------------------------------------------------------
# SERIES. Five hues, from the Okabe-Ito colour-blind-safe set, used wherever a
# figure compares distinct THINGS. Before this revision the whole set was drawn
# in a single royal blue, which is quiet but says nothing: four systems on one
# canvas looked like one system measured four times.
#
#   Greyscale lightness and contrast on white:
#       BLUE2    #0072B2  blue             87   5.19:1
#       GREEN    #009E73  bluish green    106   3.42:1
#       PURPLE   #CC79A7  reddish purple  151   3.06:1
#       SKY      #56B4E9  sky blue        158   2.31:1
#       ORANGE   #E69F00  orange          162   2.25:1
#
# Okabe-Ito's vermillion is not used: it would sit next to the reserved
# scarlet. Its yellow is not used: it is illegible on white at hairline weight.
# The five that remain are the five below.
BLUE2 = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
SERIES = (BLUE2, GREEN, ORANGE, PURPLE, SKY)

# The entity-to-hue map. An entity that appears on more than one figure is the
# same hue on every one of them, so that a reader who learns the map once on
# the graphical abstract carries it through the whole document.
#
#   BLUE2   GPT-4o zero-shot.        Also qwen3:8b in the prevalence sweep and
#                                    the from-scratch arm on fig-scaling; no
#                                    GPT-4o series appears on either canvas.
#   SKY     GPT-4o few-shot.         The same model as BLUE2 under a different
#                                    prompt, so it is deliberately the same hue
#                                    at a lighter value rather than a new hue.
#   ORANGE  ACCORD.                  Except on fig-met-exchange, where ACCORD
#                                    is the claim element and is therefore
#                                    scarlet; it takes ORANGE everywhere else.
#   PURPLE  Llama 3 8B.              Which is also the production weak
#                                    labeller (Chapter 3), so the weak-labeller
#                                    bars on fig-intro-headline (a) and the
#                                    llama3:8b column on fig-prevalence-response
#                                    are the same hue as the Llama 3 8B
#                                    evaluation baseline. Same model, three
#                                    roles; the row labels say which role.
#   GREEN   Pharmacist truth.        The human reference standard, and by
#                                    extension the arm a figure measures
#                                    against: the pretrained arm on fig-scaling
#                                    and the defect-free stratum on
#                                    fig-disc-strata.
ENTITY_HUE = {
    "gpt4o_zeroshot": BLUE2,
    "gpt4o_fewshot": SKY,
    "accord": ORANGE,
    "llama3_8b": PURPLE,
    "weak_labeller": PURPLE,        # the production weak labeller is Llama 3 8B
    "pharmacist_truth": GREEN,
    "pretrained_reference": GREEN,
}

# Text tones. This set labels a series in the series' own colour rather than in
# a legend, and 8 point type on white must clear 4.5:1. Three of the five hues
# do not, so each one has a text tone: the same hue, darkened until it clears.
# The MARK always keeps the full palette value; only type moves. BLUE2 clears
# at 5.19:1 and is its own text tone.
BLUE2_T = "#0072B2"     # 5.19:1
GREEN_T = "#008662"     # 4.58:1
ORANGE_T = "#9D6C00"    # 4.59:1
PURPLE_T = "#A36186"    # 4.55:1
SKY_T = "#3B7CA1"       # 4.58:1
TEXT_TONE = {BLUE2: BLUE2_T, GREEN: GREEN_T, ORANGE: ORANGE_T,
             PURPLE: PURPLE_T, SKY: SKY_T, SCARLET: SCARLET, GREY: MUTE,
             INK: INK, MUTE: MUTE}

# Bare estimates: a series hue lightened toward the paper until it prints at a
# greyscale weight of 160 of 255, whichever hue it started from. Used only for
# a point estimate with no committed interval. Two things follow from fixing
# the weight rather than the lightening fraction. The mark is lighter than the
# GREY this set draws reference lines in (147), so a bare estimate reads as
# weaker than anything on the canvas that carries an interval; and three bare
# estimates side by side carry equal weight, so hue distinguishes them without
# any one of them looking like the important one. See the de-emphasis note.
SERIES_BARE = {BLUE2: "#6FAFD3", GREEN: "#5DC1A6", ORANGE: "#E69F00",
               PURPLE: "#D085AF", SKY: "#59B6E9"}

# Greyscale survivability. In print, PURPLE 151, SKY 158 and ORANGE 162 are
# within 11 of one another and of GREY 147, so hue alone does not survive a
# black-and-white printer. It is not asked to: on every figure in this set that
# uses more than one series hue, the series are ALSO separated by a direct
# label on the mark (a row label, a column heading, a label at the line) and,
# where the grammar allows a marker other than a dot, by marker shape as well.
# Hue is the fast read; the label is the guarantee.

# Historical names, kept so that nothing importing this module breaks. BLUE was
# the single data colour of the previous system and now resolves to the first
# series hue, so any figure not yet ported still draws something sensible.
BLUE = BLUE2
TEAL = GREEN
ACCENT, MUTED, LIGHT, DEAD, ALERT = BLUE2, GREEN, PALE, GREY, SCARLET


def setup():
    # Identical to the rcParams block in make_figures.py and to the notebook's
    # setup cell, so the same figure drawn from any of the three is the same
    # figure. Nothing goes below 8 point.
    plt.rcParams.update({
        "font.family": "serif",
        # Pinned. The notebook is the path that ships these figures and it runs
        # in Colab, where Times New Roman is not installed, so it has always
        # drawn DejaVu Serif; these scripts were drawing Times locally, which
        # gave the same figure two different sets of text metrics. DejaVu Serif
        # ships with matplotlib, so this renders identically everywhere. It is
        # also the wider of the two, so the spacing checked here is the safe case.
        "font.serif": ["DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42,
        # A negative tick label must be a hyphen-minus, not U+2212: the thesis
        # bans the unicode minus on the page and on every canvas, and matplotlib
        # draws it by default.
        "axes.unicode_minus": False,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.grid": False, "grid.linewidth": 0.25, "grid.color": GRID,
        "lines.linewidth": 1.1,
        "legend.frameon": False,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


# One grammar for the whole set. Reference values (chance, a null, an
# initialisation) are grey hairlines with one dash pattern, labelled at the line
# itself. Point estimates are dots with a thin interval and small caps, in the
# dot's own colour, never bars. Bars are for a count or a rate of a whole only.
CHANCE_STYLE = dict(color=GREY, linestyle=(0, (4, 2)), linewidth=0.6, zorder=1)
ERRBAR = dict(elinewidth=1.0, capthick=1.0, capsize=2.5)
DOT = dict(markersize=4.6)


def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion. Closed-form and
    deterministic: it needs the two counts and nothing else, which is why it is
    the only interval this set is allowed to add to a figure that does not
    already carry one."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    adj = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    # Clamped to [0, 1] and to the estimate itself. At k = 0 the exact lower
    # bound is 0 and the arithmetic lands a few parts in 10^18 either side of
    # it, which is not a real interval but is enough to make a plotting library
    # refuse a negative error bar.
    lo = min(p, max(0.0, (centre - adj) / denom))
    hi = max(p, min(1.0, (centre + adj) / denom))
    return (lo, hi)


def canvas(fig):
    """Pin the saved bounding box to the whole figure, so the file is exactly
    FIGW wide and every figure in the thesis prints at one measure."""
    fig.patches.append(mpatches.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                          fill=False, linewidth=0, edgecolor="none"))


def lean(ax, grid=False, xbounds=None, ybounds=None):
    """Left and bottom spines only, optionally clipped to the range they
    actually measure, plus an optional single hairline value grid."""
    ax.spines[["top", "right"]].set_visible(False)
    if xbounds is not None:
        ax.spines["bottom"].set_bounds(*xbounds)
    if ybounds is not None:
        ax.spines["left"].set_bounds(*ybounds)
    if grid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.25)
