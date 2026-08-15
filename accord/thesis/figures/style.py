"""The v3 visual system for the bound thesis figures.

This module is the ONLY file in the repository permitted to contain a colour
literal. Every drawing script imports from here; the notebook's setup and
palette cells are generated from here; tools/check_parity.py makes any drift a
build failure rather than a proofreading job.

WHAT v3 OVERTURNED. The v2 docstring that used to live here was a
specification, and a maintainer who reads it will restore v2 behaviour by
reflex. It is deleted. The rules it stated that are now DEAD:

    left and bottom spines only        -> all four spines, boxed
    a spine only on a measuring axis   -> closure replaces extent
    no bold anywhere                   -> bold in SOLID chips and claim numerals
    nothing above 9pt                  -> 13pt headline, 22pt claim numeral
    bars carry no outline              -> bars take a 1.0pt DEEP outline
    colour reserved for marks          -> colour is a carrying channel
    no figure carries a title          -> every figure carries a finding headline
    DejaVu Serif pinned                -> STIXGeneral pinned

WHAT SURVIVED, unchanged and absolute:

    No legends. Series are labelled at their own mark in their own DEEP tone.
    No point estimate without its uncertainty. A thesis about instruments that
      mislead cannot itself present a bare number.
    Intervals come from committed data only: a confidence interval already in a
      result file, or a closed-form Wilson interval on counts already in a
      result file. Spread is never invented to fill a gap in the record.
    A point estimate with no committed interval is drawn in its hue's BARE
      stop, so the bareness is in the mark and not only in the caption.
    No alpha. Anywhere. Every translucent effect is a precomputed tint hex;
      transparency is unpredictable across the PDF backend, the notebook's
      inline PNG and the printer.
    No leader lines. Labels sit adjacent to their marks; where adjacency
      collides the axis range expands or the mark moves.
    No en dash, em dash or U+2212 on any canvas. No file name, register key or
      repository path on any canvas.

THE COLOUR MODEL. Every hue is a four-stop ramp and the stop determines the
job. Identity lives in DEEP and CORE only; TINT and WASH are structural ground
and carry no entity meaning. That is what lets a schematic fill nine boxes with
COBALT_TINT without any reader thinking those boxes are GPT-4o.

    DEEP   text on paper; strokes on TINT fills; solid fills carrying white
           text. Relative luminance ~0.085, which buys >= 7.7:1 both for the
           hue as text on white AND for white text on the hue as a fill.
    CORE   the mark: dots, lines, bars, arrows, strokes on DEEP.
    TINT   area fills, ground bands, phase bands, region shading. INK on it.
    WASH   very light grounds, second-order regions, band interiors.
    BARE   a point estimate with no committed interval.

Three legal fill/stroke pairings and no others:

    F1 TINT PANEL   X_TINT fill, X_DEEP stroke 1.5pt, INK text
    F2 SOLID CHIP   X_DEEP fill, X_CORE stroke 1.4pt, PAPER text, bold allowed
    F3 WASH GROUND  X_WASH fill, no stroke, INK or X_DEEP text

A CORE stop is never an area fill. A TINT never carries white text.

SCARLET has one semantic: the failure this thesis is reporting, the exact mark
where the instrument or the pipeline is broken. Budget is at most one saturated
scarlet element (CORE or DEEP) per AXES, not per canvas. SCARLET_TINT and
SCARLET_WASH are grounds and sit outside the budget. On a schematic scarlet
encodes a STATE, and a state is data, so it repeats; on a plot it is a single
claim mark. That is the only bifurcation in the system.

COLOUR-BLIND SAFETY, measured. Machado (2009) at severity 1.0, minimum pairwise
dE76 across the six hues plus SLATE: normal 32.6, protan 21.2, deutan 14.7,
tritan 15.0. Greyscale ladder of the CORE stops, minimum adjacent gap 8.9:
COBALT 73.9, SCARLET 82.8, PLUM 96.3, EMERALD 113.1, SLATE 136.9, AZURE 146.0,
AMBER 168.9. The deutan EMERALD/SLATE pair is the weakest link and is handled
structurally, not chromatically: SLATE is only ever a dashed hairline or a
small open seed dot, and never takes the same mark type as an entity hue on the
same axes.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.patches import ArrowStyle


# ---------------------------------------------------------------------------
# 1. THE RAMPS
# ---------------------------------------------------------------------------
class Ramp:
    """One hue, four stops plus BARE. Immutable by convention."""

    __slots__ = ("name", "DEEP", "CORE", "TINT", "WASH", "BARE")

    def __init__(self, name, deep, core, tint, wash, bare):
        self.name = name
        self.DEEP, self.CORE, self.TINT, self.WASH, self.BARE = (
            deep, core, tint, wash, bare)

    def __repr__(self):
        return f"<Ramp {self.name}>"

    def asdict(self):
        return {"DEEP": self.DEEP, "CORE": self.CORE, "TINT": self.TINT,
                "WASH": self.WASH, "BARE": self.BARE}


#                name       DEEP       CORE       TINT       WASH       BARE
COBALT = Ramp("COBALT",  "#0A4DB1", "#0B53C0", "#BDD0EE", "#DEE8F7", "#98B6E4")
AZURE = Ramp("AZURE",    "#20577D", "#3FA9F5", "#A1D5FA", "#D1EBFD", "#78C2F8")
EMERALD = Ramp("EMERALD", "#005F44", "#00A97A", "#98DCC9", "#CEEFE5", "#75D0B7")
AMBER = Ramp("AMBER",    "#714B00", "#F5A300", "#F9C765", "#FCE4B6", "#F6AC1A")
PLUM = Ramp("PLUM",      "#971D6B", "#C3268A", "#EEC2DE", "#F7E1EF", "#E296C6")
SCARLET = Ramp("SCARLET", "#A41704", "#D81E05", "#F5C2BC", "#FAE2DF", "#EE9A8F")

RAMPS = {r.name: r for r in (COBALT, AZURE, EMERALD, AMBER, PLUM, SCARLET)}

# Flat names, because a drawing script reads better as SCARLET_DEEP than as
# SCARLET.DEEP inside an f-string-heavy layout block.
COBALT_DEEP, COBALT_CORE, COBALT_TINT, COBALT_WASH, COBALT_BARE = (
    COBALT.DEEP, COBALT.CORE, COBALT.TINT, COBALT.WASH, COBALT.BARE)
AZURE_DEEP, AZURE_CORE, AZURE_TINT, AZURE_WASH, AZURE_BARE = (
    AZURE.DEEP, AZURE.CORE, AZURE.TINT, AZURE.WASH, AZURE.BARE)
EMERALD_DEEP, EMERALD_CORE, EMERALD_TINT, EMERALD_WASH, EMERALD_BARE = (
    EMERALD.DEEP, EMERALD.CORE, EMERALD.TINT, EMERALD.WASH, EMERALD.BARE)
AMBER_DEEP, AMBER_CORE, AMBER_TINT, AMBER_WASH, AMBER_BARE = (
    AMBER.DEEP, AMBER.CORE, AMBER.TINT, AMBER.WASH, AMBER.BARE)
PLUM_DEEP, PLUM_CORE, PLUM_TINT, PLUM_WASH, PLUM_BARE = (
    PLUM.DEEP, PLUM.CORE, PLUM.TINT, PLUM.WASH, PLUM.BARE)
SCARLET_DEEP, SCARLET_CORE, SCARLET_TINT, SCARLET_WASH, SCARLET_BARE = (
    SCARLET.DEEP, SCARLET.CORE, SCARLET.TINT, SCARLET.WASH, SCARLET.BARE)

# ---------------------------------------------------------------------------
# 2. STRUCTURE TONES. No identity, ever.
# ---------------------------------------------------------------------------
PAPER = "#FFFFFF"      # 1.0000
INK = "#16181D"        # 0.0091  17.76:1   text, spines, ticks
GRAPHITE = "#2E3440"   # 0.0341  12.49:1   connectors; a MEASURED non-entity
MUTE = "#5A6072"       # 0.1175   6.27:1   secondary annotation text
SLATE = "#7C8AA5"      # 0.2518   3.48:1   reference lines, seed dots, stubs
MIST = "#DCE3EE"       # 0.7633            neutral structural ground
HAIR = "#EAEDF3"       # 0.8453            the one optional value grid
PLATE_EDGE = "#C4CCDA" # schematic enclosure stroke only

STRUCTURE = {"PAPER": PAPER, "INK": INK, "GRAPHITE": GRAPHITE, "MUTE": MUTE,
             "SLATE": SLATE, "MIST": MIST, "HAIR": HAIR,
             "PLATE_EDGE": PLATE_EDGE}

# The single machine-readable source of every hex in the document.
PALETTE = {name: r.asdict() for name, r in RAMPS.items()}
PALETTE["STRUCTURE"] = dict(STRUCTURE)

# ---------------------------------------------------------------------------
# 3. THE ENTITY MAP. An entity that appears on more than one figure is the same
# hue on every one of them.
# ---------------------------------------------------------------------------
ENTITY_HUE = {
    "gpt4o_zeroshot":       COBALT,
    "gpt4o_fewshot":        AZURE,    # same model, other prompt -> same family
    "accord":               AMBER,
    "llama3_8b":            PLUM,
    "weak_labeller":        PLUM,     # the production weak labeller IS Llama 3 8B
    "pharmacist_truth":     EMERALD,
    "pretrained_reference": EMERALD,
    "qwen3_8b":             COBALT,   # no GPT-4o series on that canvas
    "granite33_8b":         AMBER,    # no ACCORD series on that canvas
    "control_model":        EMERALD,
}
# Alpha parameters, the MDE curve, coded-corpus counts: measured, but not an
# entity, so they take the structural measurement tone rather than a hue.
NON_ENTITY_MEASUREMENT = GRAPHITE

# ---------------------------------------------------------------------------
# 4. TYPE SCALE. Every figure is drawn at its printed size (scale 1.0), so a
# drawn point is a page point.
# ---------------------------------------------------------------------------
T_HEADLINE = 13.0      # INK, regular. A finding, never a title.
T_CLAIM = 22.0         # claim DEEP, bold. Only where a directive says so.
T_DECK = 9.5           # MUTE, regular
T_AXIS = 9.5           # INK
T_TICK = 9.0           # INK
T_SERIES = 9.0         # that series' DEEP
T_NOTE = 8.5           # MUTE, italic
T_PANEL = 10.0         # INK, italic
T_CHIP = 9.0           # PAPER, bold
T_FLOOR = 8.5

# ---------------------------------------------------------------------------
# 5. LINE WEIGHTS. The whole set, no exceptions.
# ---------------------------------------------------------------------------
LW_SIGNATURE = 3.0     # bar height in points
LW_PLATE = 1.0
LW_SPINE = 0.9
LW_GRID = 0.6
LW_DATA = 1.8
LW_REF = 1.0
LW_INTERVAL = 1.5
LW_NODE = 1.5          # F1 schematic node stroke
LW_CHIP = 1.4          # F2 solid chip rim
LW_CONNECTOR = 1.4
LW_BAR = 1.0           # bar outline, NEW in v3

MS_DOT = 6.6
MS_MARK = 6.2          # marker, non-dot
MS_BARE = 6.6          # open, 1.2pt edge in X_BARE
MS_SEED = 3.2          # SLATE, solid

REF_DASH = (0, (4, 2.4))
ERRBAR = dict(elinewidth=LW_INTERVAL, capthick=LW_INTERVAL, capsize=3.2)
DOT = dict(markersize=MS_DOT)

# ---------------------------------------------------------------------------
# 6. rcParams
# ---------------------------------------------------------------------------
RCPARAMS = {
    "font.family": "serif",
    # STIXGeneral is Times-metric-compatible and ships inside matplotlib's own
    # mpl-data/fonts/ttf, so it renders identically here and in Colab, which
    # was the entire reason DejaVu was pinned. It is also narrower than DejaVu,
    # so every spacing decision already checked has more room, not less.
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": T_TICK,
    "axes.labelsize": T_AXIS,
    "axes.titlesize": T_AXIS,
    "xtick.labelsize": T_TICK,
    "ytick.labelsize": T_TICK,
    "legend.fontsize": T_TICK,
    "pdf.fonttype": 42,
    # A negative tick label must be a hyphen-minus, not U+2212.
    "axes.unicode_minus": False,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    # Boxed. All four spines.
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.linewidth": LW_SPINE,
    "xtick.major.width": LW_SPINE,
    "ytick.major.width": LW_SPINE,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": False,
    "grid.linewidth": LW_GRID,
    "grid.color": HAIR,
    "lines.linewidth": LW_DATA,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def setup():
    plt.rcParams.update(RCPARAMS)


# ---------------------------------------------------------------------------
# 7. MEASURE
# ---------------------------------------------------------------------------
# \textwidth is 433.62pt = 6.014in. Two width classes only, 0.90 and 0.80.
TEXTWIDTH_IN = 6.014
FIGW = 5.36            # the 0.90 class, drawn
PRINT_W = 5.40


def figw(frac):
    """(drawn_width, printed_width) for a given \\linewidth fraction.

    savefig pad is 0.02in per side and canvas() pins the bbox, so
    saved width = drawn + 0.04 = printed exactly.
    """
    printed = frac * TEXTWIDTH_IN
    return printed - 0.04, printed


W90, _ = figw(0.90)    # 5.3726 -> the roster's 5.36 rounds to this
W80, _ = figw(0.80)    # 4.7712


def canvas(fig):
    """Pin the saved bounding box to the whole figure.

    savefig crops to the artists, so the width of the file would otherwise
    depend on how far the longest annotation happened to reach and no two
    figures would come out the same measure.
    """
    fig.patches.append(Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                 fill=False, linewidth=0, edgecolor="none"))


# ---------------------------------------------------------------------------
# 8. THE FRAME SYSTEM. Two enclosures, exactly one per figure.
# ---------------------------------------------------------------------------
def frame(ax, grid=False):
    """Full boxed axes. Replaces v2's lean().

    All four spines at 0.9pt INK, ticks outward. set_bounds() clipping is
    removed everywhere: a closed box cannot be clipped and the two ideas fight.

    grid draws the one optional value grid: True for both axes, or "x" / "y"
    for the one axis the plot is read along. The rule the builder does NOT get
    to decide: a plot read for VALUES gets the grid; a plot read for position
    or shape does not. It is specified per figure.
    """
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(LW_SPINE)
        ax.spines[s].set_color(INK)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(True, axis="both" if grid is True else grid, color=HAIR,
                linewidth=LW_GRID, zorder=0)
    return ax


def plate(fig):
    """The schematic enclosure: a rounded rectangle covering the full canvas,
    inset 1.5pt from every edge, corner radius 4pt, PAPER fill, PLATE_EDGE
    stroke at 1.0pt. No inner axes box. A plot never gets one."""
    w_in, h_in = fig.get_size_inches()
    ix, iy = 1.5 / (72 * w_in), 1.5 / (72 * h_in)
    p = FancyBboxPatch(
        (ix, iy), 1 - 2 * ix, 1 - 2 * iy,
        boxstyle="round,pad=0,rounding_size=0.010",
        transform=fig.transFigure, facecolor=PAPER, edgecolor=PLATE_EDGE,
        linewidth=LW_PLATE, mutation_aspect=w_in / h_in, zorder=-10,
        clip_on=False)
    fig.patches.append(p)
    return p


def signature(fig, hue_deep, headline=None, deck=None, headline_x=None):
    """The rule that makes twelve figures one set.

    A 3.0pt-tall solid bar flush to the top inside edge of the canvas, running
    its full width, in this figure's dominant DEEP hue. The headline sits 7pt
    below it: one line stating what the figure FOUND, never what it plots, at
    13pt against a 12pt body, so the figure stops reading as an inclusion.
    """
    h_in = fig.get_size_inches()[1]
    bar_h = LW_SIGNATURE / (72 * h_in)
    fig.patches.append(Rectangle((0, 1 - bar_h), 1, bar_h,
                                 transform=fig.transFigure, facecolor=hue_deep,
                                 edgecolor="none", zorder=20, clip_on=False))
    y = 1 - (LW_SIGNATURE + 7.0) / (72 * h_in)
    x = 0.012 if headline_x is None else headline_x
    if headline:
        fig.text(x, y, headline, ha="left", va="top", fontsize=T_HEADLINE,
                 color=INK, zorder=21)
        # Advance by the headline's real height. A headline may be two lines,
        # and advancing one line height put the deck on top of the second.
        n_lines = headline.count("\n") + 1
        y -= (T_HEADLINE * 1.06 * n_lines) / (72 * h_in)
    if deck:
        fig.text(x, y, deck, ha="left", va="top", fontsize=T_DECK, color=MUTE,
                 zorder=21)
    return y


def panel_label(fig, ax, text):
    """Italic 10pt INK in FIGURE coordinates: x at the axes box's left edge,
    y 5pt above the axes box top edge. Always outside the frame, never inside,
    never in data coordinates. No figure computes this itself."""
    h_in = fig.get_size_inches()[1]
    p = ax.get_position()
    return fig.text(p.x0, p.y1 + 5.0 / (72 * h_in), text, ha="left",
                    va="bottom", fontsize=T_PANEL, style="italic", color=INK,
                    zorder=21)


# ---------------------------------------------------------------------------
# 9. MARKS AND GROUNDS
# ---------------------------------------------------------------------------
# matplotlib's head_width is a HALF-width, and the 1.4pt stroke adds about
# 1.9pt to the rendered width, so the spec's literal (0.62, 0.30) at
# mutation_scale 12 renders 7.2pt long by 9.4pt wide and misses the 6.0-8.5pt
# width gate. These two numbers are the ones that hit the spec's stated target
# of about 7.4pt long by 7.2pt wide: measured 7.2 x 7.6 at scale 12.
_ARROWSTYLE = ArrowStyle("-|>", head_length=0.62, head_width=0.24)


def arrow(ax, xy_from, xy_to, color=GRAPHITE, lw=LW_CONNECTOR, zorder=3,
          mutation_scale=12, **kw):
    """One arrowhead for the whole set. matplotlib's default micro-heads are
    replaced everywhere: this targets a head about 7.4pt long and 7.2pt wide,
    solid-filled in the stroke colour. No script builds one by hand."""
    a = FancyArrowPatch(xy_from, xy_to, arrowstyle=_ARROWSTYLE,
                        mutation_scale=mutation_scale, lw=lw, shrinkA=0,
                        shrinkB=0, joinstyle="miter", capstyle="butt",
                        color=color, zorder=zorder, **kw)
    ax.add_patch(a)
    return a


def chip(ax, x, y, w, h, hue, text, fontsize=T_CHIP, radius=3.0, zorder=6,
         ha="center", stroke=None):
    """F2 SOLID CHIP: X_DEEP fill, PAPER text, bold, no rim.

    hue may be a Ramp or a (fill, rim) pair, so the SLATE/MIST no-verdict state
    can use the same helper without inventing a seventh ramp. The rim is kept in
    the signature and ignored by default: a saturated fill on paper needs no
    outline to hold its shape, and the 1.4pt CORE rim it used to carry made a
    small chip read as a button. Pass stroke= to draw one deliberately.
    """
    if isinstance(hue, Ramp):
        fill, rim = hue.DEEP, hue.CORE
    else:
        fill, rim = hue
    rim = stroke                      # None unless a caller asks for one
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}", mutation_aspect=1.0,
        linewidth=(LW_CHIP if rim else 0), edgecolor=(rim or "none"),
        facecolor=fill, zorder=zorder,
        clip_on=False))
    tx = x + w / 2 if ha == "center" else x + 6.0
    t = ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fontsize,
                color=PAPER, fontweight="bold", zorder=zorder + 1,
                clip_on=False)
    return t


def node(ax, x, y, w, h, hue_tint, hue_deep, radius=3.0, zorder=4, lw=LW_NODE):
    """F1 TINT PANEL: X_TINT fill, X_DEEP stroke 1.5pt. Text is drawn by the
    caller in INK, because a node's text is sometimes two lines at two sizes."""
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}", mutation_aspect=1.0,
        linewidth=lw, edgecolor=hue_deep, facecolor=hue_tint, zorder=zorder,
        clip_on=False)
    ax.add_patch(p)
    return p


def band(ax, x0, x1, tint, zorder=0, axis="x"):
    """F3 WASH/TINT GROUND spanning the full extent of the other axis. No
    stroke, no alpha: the tint hex is precomputed."""
    if axis == "x":
        return ax.axvspan(x0, x1, facecolor=tint, edgecolor="none",
                          linewidth=0, zorder=zorder)
    return ax.axhspan(x0, x1, facecolor=tint, edgecolor="none", linewidth=0,
                      zorder=zorder)


def case(artist, linewidth=2.8):
    """Casing: any text or line sitting on a filled field takes a white stroke
    under it, so it reads against the fill and against the paper both. Applies
    over TINT, WASH and heatmap fields only. Text on a SOLID chip is white and
    is never cased."""
    fx = [mpe.withStroke(linewidth=linewidth, foreground=PAPER)]
    try:
        artist.set_path_effects(fx)
    except AttributeError:
        for a in artist:
            a.set_path_effects(fx)
    return artist


def cased(fontsize=T_SERIES, linewidth=2.8):
    """Kwargs form, for the common `ax.text(..., **style.cased())` call."""
    return dict(fontsize=fontsize,
                path_effects=[mpe.withStroke(linewidth=linewidth,
                                             foreground=PAPER)])


# ---------------------------------------------------------------------------
# 10. STATISTICS
# ---------------------------------------------------------------------------
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
