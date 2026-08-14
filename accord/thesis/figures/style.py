"""Shared figure conventions for the thesis.

One rule specific to this document: no figure may show a point estimate
without its uncertainty. A thesis about instruments that mislead cannot
itself present a bare number.

The visual system, applied to all fourteen bound figures:

  Typography  DejaVu Serif throughout (pinned for script/notebook parity).
              Axis labels 9pt sentence case, ticks and annotations 8pt,
              nothing below 8pt, and no bold anywhere. Multi-panel figures
              carry italic (a), (b) labels outside the axes.
  Colour      One primary data colour, one neutral fill, one accent.
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
              ACCORD sits against the exchange rates, and it is blue on
              fig-intro-headline (b) and on fig-forest because those two
              figures claim that everything is at chance, which no single mark
              carries. A figure whose claim is a shape rather than a point
              carries no scarlet at all, and that is the correct outcome, not
              a missing accent.

  Intervals   On a bar, the interval is cased: a wider white stroke under the
              bar-coloured interval, so the half that falls inside the fill
              reads against the fill and the half outside reads against the
              paper. Without it a same-colour interval loses its lower half in
              the bar and the estimate reads one-sided, which is the one thing
              this set's first rule exists to prevent. This is the treatment
              for every bar that carries an interval.

  Grey        Grey does two jobs of de-emphasis and they are not the same job.
              It is the de-emphasised comparator, a mark with an interval that
              the figure measures something else against (the pretrained arm on
              fig-scaling); and it is the bare estimate with no committed
              interval data at all, drawn without one and called underpowered
              in the caption (the transfer scores on fig-falsification). The
              first has an interval and recedes, the second has none and says
              so, and in both cases the label goes in MUTE.

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

# Palette. Six tones, each with one job, chosen against three hard constraints.
#
# 1. Greyscale, using 0.299R + 0.587G + 0.114B out of 255:
#       INK      #1A1A1A  near-black       26   text, spines, ticks
#       BLUE     #0A2795  royal blue       43   the primary data colour
#       SCARLET  #D81E05  scarlet          83   the one accent, see below
#       TEAL     #00817B  deep teal        90   retained, no longer drawn
#       MUTE     #5B5B5B  grey             91   secondary annotation text
#       GREY     #8A94A6  cool grey       147   de-emphasised data
#       PALE     #C9D2E3  pale blue-grey  209   secondary bars and bands
#       GRID     #E8E8E8  hairline grey   232   optional value grid
#    Where two tones encode a data distinction they are far apart in print:
#    BLUE against PALE is 166, BLUE against GREY is 104, BLUE against TEAL is
#    47, SCARLET against BLUE is 40. SCARLET, TEAL and MUTE sit within 9 of one
#    another, and never carry a distinction against each other: GREY draws every
#    reference line, MUTE only secondary text, and SCARLET appears once per
#    figure, on an element that is also distinguished by position or shape.
#
# 2. Colour-blind safety. Every data distinction is blue against a pale
#    blue-grey or against a neutral grey, which is a lightness step no form of
#    dichromacy removes. Red is never set against green, and never against
#    another chromatic tone that carries meaning on the same canvas.
#
# 3. Text contrast on white: INK 17.4:1, BLUE 12.1:1, MUTE 6.8:1,
#    SCARLET 5.1:1, TEAL 4.7:1. All clear 4.5:1 at 8 point. GREY is 3.1:1 and
#    is therefore never used for text, only for de-emphasised lines and markers
#    that are also labelled in MUTE.
#
# SCARLET is reserved. It marks exactly one thing per figure: the element that
# is this study, or the element that failed. There is never a second red mark
# on the same canvas.
INK = "#1A1A1A"
BLUE = "#0A2795"
PALE = "#C9D2E3"
TEAL = "#00817B"
SCARLET = "#D81E05"
MUTE = "#5B5B5B"
GREY = "#8A94A6"
GRID = "#E8E8E8"

# Historical names, kept so that nothing importing this module breaks. The
# vivid amber LIGHT used to be is retired: it is now the pale blue-grey.
ACCENT, MUTED, LIGHT, DEAD, ALERT = BLUE, TEAL, PALE, GREY, SCARLET


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
