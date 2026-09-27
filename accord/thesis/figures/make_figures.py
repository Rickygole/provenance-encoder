#!/usr/bin/env python3
"""
Render the ACCORD thesis figures from committed result files.

Every number drawn on every axis in this script is read out of a JSON file
under results/accord/ at the repository root (or computed
analytically from numbers read out of those files -- e.g. the MCC
indifference curve in make_fig_met_exchange is a closed-form function of a
confusion matrix that IS in a result file). Nothing here is a literal typed
in because it "looked right." Where a spec number could not be found in any
committed file, the figure is either built with a substitute that IS
sourced (and the substitution is stated in the figure's caption strip and in
the manifest), or it is skipped entirely -- see SKIPPED at the bottom and
the printed report.

Run:
    python3 accord/thesis/figures/make_figures.py      (from the repository root)

Output:
    accord/thesis/figures/fig-*.pdf   (one per buildable spec)
    accord/thesis/figures/manifest.json

Rough / non-negotiable notes (read before trusting a number here):
  - No per-pair prediction scores are committed to this repo (MIMIC-III DUA:
    aggregates only). Any figure that would normally show an ROC curve or a
    per-rule bootstrap CI is replaced with a point-estimate + CI panel using
    only the aggregate statistics that ARE committed (n, pos, AUROC, CI,
    Mann-Whitney p). This is stated on each such figure.
  - Seeds: this script does not fit or resample anything -- every bootstrap
    CI plotted was already computed upstream (seed recorded in the source
    JSON, typically 42) and is simply read and plotted. Where this script
    computes its own interval (Wilson score intervals on raw counts) the
    formula is closed-form and deterministic; no RNG is used anywhere in
    this file, so there is no seed to fix, but np.random.seed(42) is set
    regardless for defensive reproducibility if that ever changes.
"""

import json
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Long provenance notes belong in the LaTeX caption, not inside the plot.
# Set True to draw them again.
SHOW_INFIGURE_NOTES = False
SHOW_INFIGURE_TITLES = True   # permanent in v3; the headline is a finding, not a flag
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import norm
from scipy.optimize import brentq

np.random.seed(42)

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
# Resolved from this file's own location rather than hardcoded. A literal path
# here put the author's home directory into manifest.json's source list, and
# from there into the published supplement, every time this script was run.
REPO = str(pathlib.Path(__file__).resolve().parents[3])
RESULTS = os.path.join(REPO, "results", "accord")
FIGDIR = os.path.join(REPO, "accord", "thesis", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def load(*parts):
    path = os.path.join(RESULTS, *parts)
    with open(path) as f:
        return json.load(f), path


# ----------------------------------------------------------------------------
# The v3 visual system. style.py is the only file in this repository allowed to
# contain a colour literal; everything below is a name imported from it, so the
# script path and the notebook path cannot drift. tools/check_parity.py makes
# that a build failure rather than a proofreading job.
# ----------------------------------------------------------------------------
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import style
from style import (
    COBALT, AZURE, EMERALD, AMBER, PLUM, SCARLET, CYAN,
    COBALT_DEEP, COBALT_CORE, COBALT_TINT, COBALT_WASH, COBALT_BARE,
    AZURE_DEEP, AZURE_CORE, AZURE_TINT, AZURE_WASH, AZURE_BARE,
    EMERALD_DEEP, EMERALD_CORE, EMERALD_TINT, EMERALD_WASH, EMERALD_BARE,
    AMBER_DEEP, AMBER_CORE, AMBER_TINT, AMBER_WASH, AMBER_BARE,
    PLUM_DEEP, PLUM_CORE, PLUM_TINT, PLUM_WASH, PLUM_BARE,
    SCARLET_DEEP, SCARLET_CORE, SCARLET_TINT, SCARLET_WASH, SCARLET_BARE,
    PAPER, INK, GRAPHITE, MUTE, SLATE, MIST, HAIR, PLATE_EDGE,
    ENTITY_HUE, NON_ENTITY_MEASUREMENT, NURSING_GROUND, DISCHARGE_GROUND,
    T_HEADLINE, T_CLAIM, T_DECK, T_AXIS, T_TICK, T_SERIES, T_NOTE, T_PANEL,
    T_CHIP, T_FLOOR,
    LW_PLATE, LW_SPINE, LW_GRID, LW_DATA, LW_REF, LW_INTERVAL,
    LW_NODE, LW_CHIP, LW_CONNECTOR, LW_BAR,
    MS_DOT, MS_MARK, MS_BARE, MS_SEED, REF_DASH, ERRBAR, DOT,
    FIGW, PRINT_W, W98, W88, figw, canvas,
    frame, plate, signature, panel_label, arrow, chip, node, band, case, cased,
    wilson_ci,
)

style.setup()

# Where the PDFs and manifest.json are written: FIGDIR, unless ACCORD_FIG_OUT
# redirects it (the deep gate's figure pass renders into a temporary directory
# so it never rewrites the committed PDFs it compares against).
OUTDIR = str(style.out_dir(FIGDIR))

# Legacy names, for the four figures in this file that the thesis does not bind
# (funnel, prevalence, tracking, met-sharpened) and for the two whose drawing
# functions are kept unreferenced (rule-precision, disc-strata, survey-audit).
# They are aliases onto v3 stops, not new colours.
BLUE = BLUE2 = COBALT_CORE
GREEN = EMERALD_CORE
ORANGE = AMBER_CORE
PURPLE = PLUM_CORE
SKY = AZURE_CORE
TEAL = EMERALD_CORE
PALE = MIST
GREY = SLATE
GRID = HAIR
ALERT = SCARLET_CORE
BLUE2_T, GREEN_T, ORANGE_T = COBALT_DEEP, EMERALD_DEEP, AMBER_DEEP
PURPLE_T, SKY_T = PLUM_DEEP, AZURE_DEEP
GREY_DARK, GREY_LIGHT, GREY_FILL, GREY_DEAD = COBALT_CORE, EMERALD_CORE, MIST, SLATE
GREY_MED = MUTE
BARE_BLUE2, BARE_GREEN, BARE_PURPLE = COBALT_BARE, EMERALD_BARE, PLUM_BARE
HATCH_A = "///"
HATCH_B = "\\\\\\"
CHANCE_STYLE = dict(color=SLATE, linestyle=REF_DASH, linewidth=LW_REF, zorder=1)

MANIFEST = []


def _relative_to_repo(p):
    """Source paths go into the published supplement, so they must not carry
    this machine's home directory. Anything under the repository is recorded
    repo-relative; anything outside it (credentialed local-only data, the Drive
    mount) is recorded as a bracketed token rather than a live path."""
    s = str(p)
    if s.startswith(REPO + os.sep):
        return s[len(REPO) + 1:]
    if "/mimic_local_only/" in s:
        return "[LOCAL]/" + s.split("/mimic_local_only/", 1)[1]
    if "/My Drive/" in s:
        return "[DRIVE]/" + s.split("/My Drive/", 1)[1]
    return s


def register(fname, label, sources, note=""):
    MANIFEST.append({
        "file": os.path.join("figures", fname),
        "label": label,
        "sources": [_relative_to_repo(s) for s in sources],
        "note": note,
    })


def mcc_from_counts(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if den == 0:
        return 0.0
    return num / den


# ============================================================================
# The thesis register. Every number the round-7 rebuilds draw is read here, by
# key, and a withdrawn.* key is refused outright: withdrawn values are not
# citable, and a figure that could draw one would be a way round that rule.
_FACTS_PATH = os.path.join(RESULTS, "VERIFIED_FACTS.json")
with open(_FACTS_PATH) as _f:
    _FACTS = json.load(_f)


def fact(key):
    assert not key.startswith("withdrawn."), f"{key} is withdrawn, not citable"
    return _FACTS[key]["value"]


def fact_source(key):
    """The artifact a register key points at, repo-relative, without the
    ::json-path suffix; 'derived:' entries resolve to the register itself."""
    a = _FACTS[key]["artifact"]
    if a.startswith("derived") or a.startswith("["):
        return _FACTS_PATH
    return os.path.join(REPO, a.split("::", 1)[0])


# ============================================================================
# Figure 1 -- fig:pipeline-status  (ch1_introduction.tex)
PIPELINE_KEYS = [
    "gold.base_rate_299",
    "gold.n_total",
    "exemplar.rule_1_2_pooled_exemplar_correct_rate_in_299.rate",
    "exemplardirect.defect_rate",
    "exemplar_recomputed.rule_1_2_pooled.wrong_rate",
    "c120.g299.n",
    "c120.g299.n_flagged",
    "c120.g299.mcc",
    "c120.g299.ceiling_mcc",
    "rulehead.test_rule_type_macro_f1",
    "decontam.aggregate.ensemble_auroc",
    "decontam.aggregate.ensemble_ci95_hi",
    "sizectl108.power_curves.primary_crossing_power_0_80",
]


def make_fig_pipeline_status():
    # ROUND 7 REBUILD (fixes_r7 N9). The previous canvas numbered six stages,
    # headlined "four of six stages hid a failure", drew evaluation as a
    # failure, credited "three ablation claims" that no ablation ever tested and
    # checked the architecture with an OOD transfer F1 the thesis does not
    # interpret. This one draws the thesis's own four stages, numbered 1 to 4
    # as Chapter 1 numbers them. Stages 1 to 3 each reported a number that its
    # control or code audit does not support the reading of that number;
    # stage 4 is a bounded null, and
    # the two states are told apart by fill (solid chip against outlined chip)
    # and by a word, so the distinction survives greyscale.
    #
    # The candidate chip is a RANGE, never a rate: no instrument establishes
    # how often the positional exemplar misses (C106). The low end is one
    # minus the step2plus join's pooled correct rate; the high end is the
    # unvalidated direct extractor.
    base = fact("gold.base_rate_299")
    n_gold = fact("gold.n_total")
    ex_lo = 1.0 - fact("exemplar.rule_1_2_pooled_exemplar_correct_rate_in_299.rate")
    ex_hi = fact("exemplardirect.defect_rate")
    ex_mid = fact("exemplar_recomputed.rule_1_2_pooled.wrong_rate")
    assert ex_lo < ex_mid < ex_hi
    # THE FOURTH INSTRUMENT. This chip used to print the three readings 0,
    # 0.4745 and 0.9315 and nothing else, which was the honest summary while no
    # instrument could measure the annotated rows without excluding some of
    # them. One now can: for Rules 1 and 2 the trigger fires only on an
    # opposite action pair, so a displayed pair that is not an opposite pair
    # provably could not have fired it, and that is decidable on every one of
    # the 165 annotated pairs with nothing excluded. It is a LOWER BOUND, so
    # the verdict on this stage does not move (the rate is still not pinned by
    # any instrument) but the chip must carry it, or this figure and
    # fig:exemplar, which face each other in Chapter 1, print two different
    # stories about the same stage. Read as two counts from the artifact, as
    # fig:exemplar reads it; there is no register key yet.
    lb_pooled = load("clinical_core",
                     "displayed_exemplar_lower_bound.json")[0][
                         "rules_1_and_2"]["pooled"]
    ex_bound = (lb_pooled["n_displayed_pair_is_not_opposite"]
                / lb_pooled["n_pairs"])
    assert ex_lo < ex_bound < ex_hi
    n_c120 = fact("c120.g299.n")
    n_flag = fact("c120.g299.n_flagged")
    mcc = fact("c120.g299.mcc")
    mcc_ceil = fact("c120.g299.ceiling_mcc")
    assert n_c120 == n_gold
    f1_rule = fact("rulehead.test_rule_type_macro_f1")
    auroc = fact("decontam.aggregate.ensemble_auroc")
    ci_hi = fact("decontam.aggregate.ensemble_ci95_hi")
    floor = fact("sizectl108.power_curves.primary_crossing_power_0_80")
    # The bounded-null reading only holds while the interval's upper end sits
    # under the floor; if either moves past the other the chip would lie.
    assert ci_hi < floor, "interval upper bound no longer under the C108 floor"

    # ROUND 8 (committee). Stage 1 is "not established", not "unsupported":
    # three instruments disagree on the exemplar defect (0, 0.4745, 0.9315), so
    # no check rejects the reported precision; none establishes it either. The
    # three states are told apart in greyscale by chip value alone: solid
    # scarlet (dark) for unsupported, a SCARLET_TINT fill under a dashed rim
    # (mid grey) for not established, paper under a solid rim for bounded null.
    NOTEST, UNSUP, BOUNDED = "not established", "unsupported", "bounded null"
    # The not-established rim, shared by the chip and its key swatch.
    NOTEST_RIM, NOTEST_DASH = 1.7, (0, (2.6, 1.5))
    # ROUND 9 (committee). The numbers largely stand; what failed is the
    # reading of each. The per-stage label for an unsupported stage says so.
    LABEL = {NOTEST: NOTEST, UNSUP: "reading unsupported", BOUNDED: BOUNDED}
    # ROUND 10 (audit item 9). One rounding rule for the whole canvas: two
    # decimals, full precision stays in the tables and in the manifest note
    # below.
    # FIGURE POLISH. One exception, row 1's three miss rates, which print at
    # the precision every caption and Figure 2's deck print them (0, 0.4745,
    # 0.9315): the same three numbers at two precisions on facing figures read
    # as two different sets of numbers.
    rows = [
        (1, "Candidate mining", NOTEST,
         f"precision {base:.2f}\non {n_gold} gold pairs",
         "Is the exemplar in the trigger set?",
         f"miss rate at least {ex_bound:.4f}, not pinned"),
        (2, "Weak supervision", UNSUP,
         f"{n_flag} of {n_c120} pairs\nflagged",
         "Marginal ceiling, then measured agreement",
         f"MCC {mcc:.2f}, ceiling {mcc_ceil:.2f}"),
        (3, "Architecture", UNSUP,
         # ROUND 10 (audit item 10): name the metric and the split, not just
         # "F1", so a reader does not have to guess which head or which set.
         # FIGURE POLISH. Broken after the metric name, not after the value:
         # the four-decimal chip in row 1 is 18pt wider than the two-decimal
         # one and this line set the width of the whole REPORTED column.
         f"rule-type test macro-F1\n{f1_rule:.2f}, two more components",
         "Code audit of the three credited components",
         "artifact, target in input, untrained"),
        (4, "Evaluation", BOUNDED,
         f"AUROC {auroc:.2f}",
         "Interval upper end against the\nsimulated 80 percent power floor",
         f"upper {ci_hi:.2f}, power floor {floor:.2f}"),
    ]

    # ROUND 10 (audit item 3). H grows by 0.24in (about 8 percent) over the
    # round-9 canvas, the only size change in this figure, so a three-item key
    # fits between the headline block and the header row without crowding the
    # four rows, which keep their original bottom-anchored positions below.
    # FIGURE POLISH. H grows again, by 0.15in, and all of it goes between the
    # headline block and the header row: the headline's third line ended 44pt
    # below the top edge and the header row was centred at 45.2pt, so the two
    # shared a band.
    W, H = FIGW, 3.49
    fig, ax = plt.subplots(figsize=(W, H))
    canvas(fig)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W * 72)
    ax.set_ylim(0, H * 72)
    ax.axis("off")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def width_pt(text, **kw):
        probe = ax.text(0, 0, text, **kw)
        w = probe.get_window_extent(renderer).width * 72.0 / fig.dpi
        probe.remove()
        return w

    row_mid = {1: 150.0, 2: 108.0, 3: 66.0, 4: 24.0}
    # A narrow spine: the rail, the stage numbers and the stage names take the
    # left fifth of the canvas, and the columns are placed from measured text
    # widths rather than fixed offsets.
    X_SPINE, X_NUM, X_NAME = 15.0, 32.0, 42.0
    name_w = max(max(width_pt(r[1], fontsize=10.5) for r in rows),
                 max(width_pt(LABEL[r[2]], fontsize=T_NOTE, style="italic")
                     for r in rows))
    X_REP = X_NAME + name_w + 10.0
    rep_w = max(width_pt(r[3], fontsize=T_SERIES, linespacing=1.22) for r in rows)
    # The gutter before the CHECK column was 11.0. The serial comma the
    # graduate school's checklist requires in the row 1 chip ("0, 0.4745, or
    # 0.9315") made that chip 2.2pt wider and put it 0.3pt over the 4pt right
    # margin asserted below. One point comes out of this gutter, which now
    # matches the 10.0 before the REPORTED column; no type size moves.
    X_CHK = X_REP + rep_w + 10.0

    # The carrier: stages 2 to 4 consumed the exemplar stage 1 selected. It
    # records shared input, not cause and not a defect rate, so it is a neutral
    # rule and not scarlet (scarlet marks a failed check only).
    ax.add_patch(mpatches.Circle((X_SPINE, row_mid[1] + 5.5), 3.0,
                                 facecolor=GRAPHITE, edgecolor="none",
                                 zorder=6))
    ax.plot([X_SPINE, X_SPINE], [row_mid[1] + 5.5, row_mid[4] + 5.5],
            color=GRAPHITE, lw=1.4, zorder=4, solid_capstyle="butt")
    for n in (2, 3, 4):
        arrow(ax, (X_SPINE, row_mid[n] + 5.5), (X_NUM - 7.0, row_mid[n] + 5.5),
              color=GRAPHITE, lw=1.2, mutation_scale=9)
    ax.text(X_SPINE - 6.5, (row_mid[2] + row_mid[4]) / 2 + 5.5,
            "the same selected exemplar", rotation=90, ha="center",
            va="center", fontsize=T_NOTE, style="italic", color=MUTE,
            zorder=5)

    for n, name, status, reported, check, finding in rows:
        mid = row_mid[n]
        ax.text(X_NUM, mid + 5.5, str(n), ha="center", va="center",
                fontsize=10.0, color=INK, zorder=5)
        ax.text(X_NAME, mid + 5.5, name, ha="left", va="center", fontsize=10.5,
                color=INK, zorder=5)
        ax.text(X_NAME, mid - 8.0, LABEL[status], ha="left", va="center",
                fontsize=T_NOTE, style="italic",
                color=SCARLET_DEEP if status == UNSUP else MUTE, zorder=5)
        # FIGURE POLISH. The first line of every reported block sits on the
        # stage name's line. Centring the block on mid did that for the
        # two-line rows only and left row 4's single line half a line low.
        n_rep = reported.count("\n") + 1
        ax.text(X_REP, mid + 5.5 - (n_rep - 1) * 0.5 * T_SERIES * 1.22,
                reported, ha="left", va="center",
                fontsize=T_SERIES, color=INK, linespacing=1.22, zorder=5)
        ax.text(X_CHK, mid + 9.5, check, ha="left", va="center",
                fontsize=T_NOTE, style="italic", color=MUTE, linespacing=1.15,
                zorder=5)
        # ROUND 10. This check text had no width assertion, unlike the finding
        # chip below it, and silently ran the canvas 49pt past its own right
        # edge once the row 3 relabel (audit item 10) pushed X_CHK rightward.
        # tools/check_figure_invariants.py's printed-measure gate is the only
        # thing that would have caught it; asserted here too so it fails loud
        # the moment any upstream column width changes again.
        check_w = max(width_pt(line, fontsize=T_NOTE, style="italic")
                      for line in check.split("\n"))
        assert X_CHK + check_w < W * 72 - 4.0, \
            f"check text overflows the canvas: {check!r}"
        w_i = width_pt(finding, fontsize=T_CHIP, fontweight="bold") + 12.0
        # Three chip encodings that stay distinct in greyscale: a solid dark
        # scarlet fill for unsupported, a light MIST fill under a dashed rim
        # for not established, bare paper under a solid rim for bounded null.
        if status == UNSUP:
            chip(ax, X_CHK, mid - 14.5, w_i, 13.0, SCARLET, finding,
                 fontsize=T_CHIP, radius=3.0, ha="left")
        elif status == NOTEST:
            # ROUND 11 (coordinator regression). A full-fill hatch ran behind
            # this chip's own text and made it the least legible chip on the
            # canvas, and was replaced with a solid GRAPHITE bar down the
            # chip's left edge.
            # VISUAL AUDIT. That bar, a dark slab cutting across a dashed rim
            # on one side only, read as a rendering glitch on the chip and on
            # its key swatch. It is gone. The verdict is carried by the rim
            # alone, made heavy enough to be read as a choice (NOTEST_RIM:
            # 1.7pt against the bounded-null chip's solid 1.4pt, long dashes
            # with butt ends), over the same MIST fill. The three verdicts
            # stay three things in greyscale: a dark solid fill with white
            # type, a grey fill under a heavy dashed rim, paper under a solid
            # rim. A leading glyph inside the chip was the other candidate and
            # does not fit: this chip ends 1.8pt inside the right margin
            # asserted below.
            p = node(ax, X_CHK, mid - 14.5, w_i, 13.0, MIST, GRAPHITE,
                     radius=3.0, zorder=6, lw=NOTEST_RIM)
            p.set_linestyle(NOTEST_DASH)
            p.set_capstyle("butt")
            ax.text(X_CHK + 6.0, mid - 8.0, finding, ha="left", va="center",
                    fontsize=T_CHIP, fontweight="bold", color=INK, zorder=7)
        else:
            node(ax, X_CHK, mid - 14.5, w_i, 13.0, PAPER, GRAPHITE,
                 radius=3.0, zorder=6, lw=1.4)
            ax.text(X_CHK + 6.0, mid - 8.0, finding, ha="left", va="center",
                    fontsize=T_CHIP, fontweight="bold", color=INK, zorder=7)
        assert X_CHK + w_i < W * 72 - 4.0, f"chip overflows the canvas: {finding}"

    # ROUND 10 (audit item 3). H grew so a key fits between the header and
    # the rows. FIGURE POLISH: the header is now 56.0pt below the top edge, not
    # 45.2pt, which clears the three-line headline by a full line of lead.
    HEADER_Y = H * 72 - 56.0
    LEGEND_Y = HEADER_Y - 16.0
    ax.text(X_REP, HEADER_Y, "REPORTED", ha="left", va="center",
            fontsize=T_FLOOR, color=MUTE, zorder=5)
    ax.text(X_CHK, HEADER_Y, "CHECK AND FINDING", ha="left", va="center",
            fontsize=T_FLOOR, color=MUTE, zorder=5)

    # ROUND 10 (audit item 3). A three-item key, one swatch per chip encoding,
    # naming the same words each row already carries in LABEL: a reader no
    # longer has to infer fill-and-rim from the rows alone.
    ax.text(X_REP, LEGEND_Y, "KEY", ha="left", va="center", fontsize=T_FLOOR,
            color=MUTE, zorder=5)
    sw_w, sw_h = 12.0, 8.5
    x_leg = X_REP + width_pt("KEY", fontsize=T_FLOOR) + 8.0
    for status, word in ((NOTEST, LABEL[NOTEST]), (UNSUP, LABEL[UNSUP]),
                         (BOUNDED, LABEL[BOUNDED])):
        y0 = LEGEND_Y - sw_h / 2
        if status == UNSUP:
            ax.add_patch(FancyBboxPatch(
                (x_leg, y0), sw_w, sw_h, boxstyle="round,pad=0,rounding_size=1.5",
                facecolor=SCARLET_DEEP, edgecolor="none", zorder=6,
                clip_on=False))
        elif status == NOTEST:
            p = FancyBboxPatch(
                (x_leg, y0), sw_w, sw_h, boxstyle="round,pad=0,rounding_size=1.5",
                facecolor=MIST, edgecolor=GRAPHITE, linewidth=NOTEST_RIM,
                zorder=6, clip_on=False)
            p.set_linestyle(NOTEST_DASH)
            p.set_capstyle("butt")
            ax.add_patch(p)
        else:
            ax.add_patch(FancyBboxPatch(
                (x_leg, y0), sw_w, sw_h, boxstyle="round,pad=0,rounding_size=1.5",
                facecolor=PAPER, edgecolor=GRAPHITE, linewidth=1.4, zorder=6,
                clip_on=False))
        ax.text(x_leg + sw_w + 4.0, LEGEND_Y, word, ha="left", va="center",
                fontsize=T_NOTE, color=INK, zorder=7)
        x_leg += sw_w + 4.0 + width_pt(word, fontsize=T_NOTE) + 9.0
        assert x_leg < W * 72 - 4.0, f"legend overflows the canvas at {word!r}"

    # THE FRAMING, changed with the thesis's. This headline read "Three of the
    # four stage readings did not survive their checks ... the fourth ...
    # survived as a bounded null", which is the language of an audit of an
    # earlier draft: it asks the reader to score stages as survivors and
    # casualties. The thesis is one first study, so the headline says what each
    # check FOUND instead. Every verdict is unchanged, and so is every number,
    # every chip encoding and the on-canvas key that names them.
    signature(fig, SCARLET_DEEP,
              headline="Four stages, four checks, four findings: one reading no "
              "instrument establishes,\ntwo readings the checks do not support, "
              "and one near-chance reading\nbounded under the power floor.")
    fname = "fig-pipeline-status.pdf"
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)
    srcs = [_FACTS_PATH] + sorted({fact_source(k) for k in PIPELINE_KEYS} - {_FACTS_PATH})
    register(fname, "fig:pipeline-status", srcs,
             "Four stages numbered 1 to 4 as Chapter 1 numbers them. Every number "
             "is read by key from results/accord/VERIFIED_FACTS.json (keys: "
             + ", ".join(PIPELINE_KEYS) + "); no withdrawn key is readable. Every "
             "number on the canvas prints to two decimals, except stage 1's three "
             "miss rates, which print to four as the captions do; full precision "
             "stays in this note and in the tables. Three "
             "chip encodings, distinct in greyscale and named in an on-canvas key: "
             f"stage 1 not established (MIST fill under a heavy dashed GRAPHITE "
             f"rim, 1.7pt, with no hatch and no edge bar; miss rate at least "
             f"{ex_bound:.4f}, the displayed-exemplar lower bound read as "
             f"{lb_pooled['n_displayed_pair_is_not_opposite']} of "
             f"{lb_pooled['n_pairs']} from results/accord/clinical_core/"
             f"displayed_exemplar_lower_bound.json, against readings of "
             f"{ex_lo:.0f} and {ex_hi:.4f} from the other two instruments; the "
             f"verdict does not move, because a lower bound does not pin the "
             f"rate); "
             f"stages 2 and 3 reading unsupported (solid scarlet, the only scarlet "
             f"on the canvas: MCC {mcc:.4f} against a ceiling of {mcc_ceil:.4f}; "
             "the three credited architectural components fail a code audit); "
             f"stage 4 bounded null (paper fill, solid rim: interval upper end "
             f"{ci_hi:.4f} under the 80 percent power floor {floor:.4f}). The "
             "carrier rail is GRAPHITE, not scarlet: it records that stages 2 to 4 "
             "consumed the exemplar stage 1 selected, which is shared input, not a "
             "failed check. No plate; the stage column is sized from measured text "
             "widths so the spine takes about the left fifth of the canvas. No "
             "internal identifiers on the canvas.")


# ============================================================================
# Figure 2 -- fig:intro-headline  (ch1_introduction.tex)
# ============================================================================
def make_fig_intro_headline():
    step01, p_step01 = load("step0_1_join_and_calibration.json")
    xtab = step01["weak_confidence_is_degenerate"]["crosstab_label_x_confidence"]
    weak_pos, weak_neg = xtab["0.95"]["contradiction"], xtab["1.0"]["not_contradiction"]
    weak_n = weak_pos + weak_neg
    weak_rate = weak_pos / weak_n

    exp1a, p_exp1a = load("differential", "exp1a_gold_error_rates_by_group.json")
    lang = exp1a["language_group3"]
    pred_pos = sum(g["TP"] + g["FP"] for g in lang)
    pred_n = sum(g["n"] for g in lang)
    pred_rate = pred_pos / pred_n

    step2, p_step2 = load("step2_signal_check.json")
    d = step2["contra_prob_vs_pharm_label__ambiguous_as_negative_n299"]
    pharm_pos, pharm_n = d["n_pos"], d["n"]
    pharm_rate = pharm_pos / pharm_n

    # Panel (b) is gone. Its four Matthews-correlation rows are Figure 8's
    # first family, drawn identically 85 pages later, and Figure 8 is now the
    # one place every headline estimate in this thesis lives. What remains is
    # the number the rest of the thesis follows: a rate of a whole, so bars.
    fig, ax = plt.subplots(figsize=(FIGW, 2.78))
    canvas(fig)
    # The margins are held at their old ABSOLUTE size and the whole of the
    # canvas's growth is given to the axes: the type in the margins is still
    # 9pt, so a margin that grew with the canvas would only have bought white
    # paper. The top margin is 3pt smaller than it was, which is the space the
    # signature bar used to occupy.
    fig.subplots_adjust(left=0.2747, right=0.9863, top=0.7306, bottom=0.1559)

    # "candidate pool", not "train pool": weak_n is the full weak-labelled
    # candidate pool, which is a different and larger denominator than the
    # training pool the detector was actually fit on. Chapter 1's caption
    # prose says exactly this, and the old label contradicted it.
    bars = [("Weak labeller, candidate pool", weak_pos, weak_n, wilson_ci(weak_pos, weak_n)),
            ("Weak labeller, gold set", pred_pos, pred_n, wilson_ci(pred_pos, pred_n)),
            ("Pharmacist truth, gold set", pharm_pos, pharm_n, wilson_ci(pharm_pos, pharm_n))]
    ypos = np.arange(len(bars))[::-1]
    rates = [b[1] / b[2] for b in bars]
    los = [b[3][0] for b in bars]
    his = [b[3][1] for b in bars]
    errs = [[r - lo for r, lo in zip(rates, los)],
            [hi - r for r, hi in zip(rates, his)]]
    # The weak labeller is Llama 3 8B (Chapter 3), so it carries Llama's hue on
    # both of its rows; the pharmacist carries the truth hue.
    hues = [PLUM, PLUM, EMERALD]

    # 1.56, not 1.86. The right third of the old range held nothing at all: the
    # bracket and the multiplier end at 1.42 and the ticks stop at 1.0. On a
    # canvas that is now 9 percent wider that dead band was the largest single
    # area of white on the sheet, and closing it makes the three measured bars
    # 19 percent longer without changing one number.
    ax.set_xlim(0, 1.56)
    ax.set_ylim(-0.66, 2.66)
    # The gap between the pharmacist's rate and the weak labeller's on the same
    # 299 pairs IS the finding, so it is drawn as a region rather than left for
    # the reader to measure off the axis.
    band(ax, pharm_pos / pharm_n, pred_pos / pred_n, SCARLET_WASH, zorder=0)
    # Flat fills. The 1.0pt DEEP outline fought the cased interval running
    # through the bar and made three bars read as three outlined boxes; the
    # CORE fill on paper already has all the edge it needs.
    ax.barh(ypos, rates, color=[h.CORE for h in hues], height=0.62,
            linewidth=0, edgecolor="none", zorder=3)
    # v3 (this round): drawing the interval through the bar is the defect,
    # not the stop it was drawn in. Measured WCAG contrast of the foreground
    # line against the CORE fill it crossed: hue.DEEP on the plum bar 2.00:1,
    # on the emerald bar 2.56:1; INK on the plum bar 4.02:1; PAPER on the plum
    # bar 4.42:1. None of the tones this system has clear the 4.5:1 a thin
    # line needs to read at this figure's print size (0.98 linewidth) -- the
    # bar fill itself is the problem, not the line colour drawn on it. The
    # interval is moved clear of the bar instead: a floating I-beam on bare
    # paper, directly above each bar, aligned to the same x-range as its own
    # estimate (same lo/hi, same rate) so it is read as that bar's
    # uncertainty and nothing else's. INK on PAPER is 17.76:1 and no hue can
    # swallow it, so this needs no casing -- there is no fill under it to
    # case against. _iv_dy clears the bar's own half-height (0.31 data units)
    # by about two points at this axis's scale, and stays well inside the
    # 0.38-unit gap to the row above, so the cap never touches a bar edge.
    _iv_dy = 0.46
    for y, r, e_lo, e_hi in zip(ypos, rates, errs[0], errs[1]):
        ax.errorbar([r], [y + _iv_dy], xerr=[[e_lo], [e_hi]], fmt="none",
                    ecolor=INK, zorder=5, **ERRBAR)
    # One format for every bar on this axis, not one per bar. The x axis
    # itself is ticked in decimals (0.00 .. 1.00, labelled "Positive rate"),
    # so a bar label in percent form disagrees with the axis it sits on -- a
    # reader reads "87.3%" against tick marks reading "0.75, 1.00" and has to
    # convert one of them. This used to print the pharmacist's row (index 2)
    # as a decimal, matching the 0.0401 fixed in the chapters' prose, and the
    # two weak-labeller rows as one-decimal percentages; that is two formats
    # on one decimal-ticked axis. All three are now the same four-decimal
    # form the pharmacist's rate already had, computed from the same counts
    # loaded above (1,425/1,679, 261/299, 12/299), so nothing is hardcoded and
    # nothing on this axis needs converting to read the other two.
    for i, (y, b, r, hi, hue) in enumerate(zip(ypos, bars, rates, his, hues)):
        label = f"{r:.4f}"
        ax.text(hi + 0.030, y, label, va="center", ha="left",
                fontsize=T_SERIES, color=hue.DEEP, zorder=6)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{b[0]}\n({b[1]:,} of {b[2]:,})" for b in bars],
                       fontsize=T_SERIES, linespacing=1.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Positive rate")
    frame(ax)

    # The claim numeral. The bracket spans only the two bars measured on the
    # same 299-pair gold set; the training-pool bar has a different denominator
    # and is not part of the comparison.
    gold_ratio = (pred_pos / pred_n) / (pharm_pos / pharm_n)
    x_br = 1.16
    ax.plot([x_br] * 2, [ypos[2], ypos[1]], color=SCARLET_CORE, lw=LW_DATA,
            solid_capstyle="butt", zorder=6)
    for y in (ypos[1], ypos[2]):
        ax.plot([x_br, x_br - 0.045], [y, y], color=SCARLET_CORE, lw=LW_DATA,
                zorder=6)
    # The multiplier sits beside its bracket; what it is a ratio OF is the
    # figure's deck, under the headline, where a subtitle belongs. Wedged into
    # the axes it had nowhere to go: beside the bracket it wrapped to four lines
    # and the last was struck through by the bottom spine, and moved below the
    # bracket it ran across the wash through empty plot space.
    ax.text(x_br + 0.06, (ypos[1] + ypos[2]) / 2, f"{gold_ratio:.0f}x",
            va="center", ha="left", fontsize=T_CLAIM, fontweight="bold",
            color=SCARLET_DEEP, zorder=6)

    # The bars above are now all four-decimal rates (fix for the mixed-format
    # defect: percent labels on a decimal-ticked axis). Restating the weak
    # labeller's clause here as "87.3%" would reintroduce exactly that
    # mismatch one level up, between the bar and the headline that names it,
    # so this line is stated in the raw counts instead -- the same counts
    # already printed on the y-axis tick labels below each bar ("261 of
    # 299"), which cannot conflict with a rate format because they are not a
    # rate. The pharmacist's clause keeps 0.0401, the same four-decimal
    # fraction its own bar now carries and the one already fixed by the
    # surrounding prose, so nothing here is in two formats.
    signature(fig, PLUM_DEEP,
              f"The weak labeller calls {pred_pos} of {pred_n} gold-set "
              f"pairs contradictions.\n"
              f"The pharmacist\u2019s base rate is {pharm_pos/pharm_n:.4f}.",
              deck=f"{gold_ratio:.0f}x: the weak labeller\u2019s marginal over the "
                   f"pharmacist\u2019s, on the same {pharm_n} pairs.")
    fname = "fig-intro-headline.pdf"
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)
    register(fname, "fig:intro-headline",
             [p_step01, p_exp1a, p_step2],
             "The left-panel middle bar is the weak labeller marginal, 261 of 299 "
             "(gold.weaklabel_contradiction); ACCORD predicts contradiction for 278 of 299 "
             "(gold.accord_pred_contradiction, reconciliation_c28.json). The Matthews "
             "correlation panel that used to sit under this one is deleted: those four "
             "estimates are fig:forest's first family.")


# ============================================================================
# Figure 3 -- fig:funnel  (ch3_data.tex)
# ============================================================================
def make_fig_funnel():
    icd, p_icd = load("stats", "cohort_icd_counts.json")
    hf = icd["heart_failure"]
    era, p_era = load("era", "era_confound_closed.json")
    n_final = era["n"]

    fig, ax = plt.subplots(figsize=(FIGW, 1.75))
    canvas(fig)
    fig.subplots_adjust(left=0.42, right=0.985, top=0.94, bottom=0.06)
    stages = [
        (f"ICD-9 428-prefix admissions", hf["pre_los"]),
        (f"After LOS filter (removed {hf['removed']})", hf["post_los"]),
        (f"Final note-pair cohort\n(both notes present and selectable)", n_final),
    ]
    ys = np.arange(len(stages))[::-1]
    maxw = stages[0][1]
    for y, (label, val) in zip(ys, stages):
        w = val / maxw
        ax.barh(y, w, height=0.55, color=BLUE, linewidth=0)
        ax.text(w + 0.02, y, f"{val:,}", va="center", fontsize=8.0, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in stages], fontsize=8.0, linespacing=1.25)
    ax.set_xlim(0, 1.30)
    ax.set_ylim(-0.62, len(stages) - 0.38)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_bounds(0, len(stages) - 1)
    ax.tick_params(axis="y", length=0)
    if SHOW_INFIGURE_TITLES: ax.set_title("Cohort funnel (heart failure), traceable stages only", fontsize=9.0)
    if SHOW_INFIGURE_NOTES: fig.text(0.5, 0.02,
            "The original spec's intermediate step (9,825 admissions with both note roles present, 72.2%) and the three\n"
            "alternative three-role configurations (16.7%, 16.8%, 12.4%) are not present in any committed result file\n"
            "(grep of results/accord/**/*.json found no match for those counts) and are omitted here rather than typed in.",
            ha="center", fontsize=8.0, color=MUTE)
    fname = "fig-funnel.pdf"
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)
    register(fname, "fig:funnel", [p_icd, p_era],
             "Simplified from the 5-stage spec to the 3 stages traceable to committed files; "
             "see in-figure footnote.")


# ============================================================================
# Figure 4 -- fig:exemplar  (ch1_introduction.tex)
#
# REBUILT when the fourth instrument arrived. The previous canvas was a
# schematic: five abstract mention slots, a brace, and a one line deck printing
# three miss rates (0, 0.4745, 0.9315) whose message was that the instruments
# disagree so the rate is unresolved. Two things make that canvas wrong now.
#
# The first is that a fourth instrument exists and it needs neither a
# reconstruction nor a key join. It asks one decidable question of the
# annotated rows themselves: for Rules 1 and 2 the trigger fires only on an
# opposite action pair, so a displayed pair that is not an opposite pair
# provably could not have fired the trigger. That is 72 of 165 annotated pairs,
# a measured LOWER BOUND with nothing excluded, and it lands where two other
# readings already were.
#
# The second is that the "unresolved" message was never the honest one. Ranking
# the instruments by how many of the 165 rows each one throws away turns three
# numbers that look like noise into a pattern: the readings that exclude little
# or nothing agree, and the two that wander are the two that exclude the most.
# So this figure now PLOTS the readings, ordered by exclusion, on one axis,
# with the intervals the counts support and the excluded count spelled out per
# row. The schematic's mechanism is one sentence of the caption; it drew no
# measured quantity and the generator said so itself.
#
# WHAT THE FIGURE MUST NOT SAY, and the encoding that keeps it from saying it:
#   - The 0.9315 reading is not drawn as an equal. Its mark is open, it carries
#     the largest excluded count of the annotated instruments after the key
#     join, and the canvas prints the count of its own defect rows that have no
#     classified action on at least one side.
#   - Rules 3 and 4 are not on the canvas at all, and a note says why. Their
#     triggers reference no action label, so a high non-opposite rate is the
#     rule behaving as designed and carries no defect meaning. The artifact
#     tags them defect_interpretation "none" and this figure honours that.
#   - The rebuilt-pool row is separated by a hairline and named as a different
#     substrate. It shares no rows with the four above it.
# ============================================================================
EXEMPLAR_KEYS = [
    "exemplar_recomputed.rule_1_2_pooled.n",
    "exemplar_recomputed.rule_1_2_pooled.n_wrong",
    "exemplar_recomputed.rule_1_2_pooled.n_resolved",
    "exemplar_recomputed.rule_1_2_pooled.n_unresolved",
    "exemplar.rule_1_2_pooled_exemplar_correct_rate_in_299.n_known",
    "exemplar.rule_1_2_pooled_exemplar_correct_rate_in_299.n_correct",
    "exemplardirect.n_wrong",
    "exemplardirect.n_decidable",
    "exemplardirect.n_undecidable",
    "exemplardirect.n_wrong_involves_unclassified",
    "c129.pooled.positional_outside_trigger_set.either_side.count",
    "c129.pooled.positional_outside_trigger_set.either_side.n",
]


def make_fig_exemplar():
    # The fourth instrument has no register key yet, so it is read from its own
    # committed artifact by the same rule every other number here follows: a
    # count and a denominator, never a rate typed in. The Wilson interval is
    # recomputed from the two counts and asserted against the one the artifact
    # stores, so a drifted artifact fails the build rather than the reader.
    lb, p_lb = load("clinical_core", "displayed_exemplar_lower_bound.json")
    pooled12 = lb["rules_1_and_2"]["pooled"]
    assert pooled12["trigger_requires_opposite_action_pair"] is True
    n_lb = pooled12["n_pairs"]
    k_lb = pooled12["n_displayed_pair_is_not_opposite"]
    # Rules 3 and 4 carry no defect interpretation. Asserted, not assumed: if a
    # future artifact ever claimed one, this figure would have to be rethought
    # rather than quietly gain two rows.
    for r in ("3", "4"):
        assert lb["rules_3_and_4_not_a_defect_measurement"][r][
            "defect_interpretation"] == "none"

    n_pairs = fact("exemplar_recomputed.rule_1_2_pooled.n")
    assert n_lb == n_pairs

    # Instrument 1, the displayed-exemplar lower bound: nothing excluded.
    # Instrument 2, the reconstruction: 28 of 165 unresolved.
    # Instrument 3, the direct extractor: 19 of 165 undecidable.
    # Instrument 4, the key join: resolves 23 of 165 and reports no defect.
    k_rec = fact("exemplar_recomputed.rule_1_2_pooled.n_wrong")
    n_rec = fact("exemplar_recomputed.rule_1_2_pooled.n_resolved")
    x_rec = fact("exemplar_recomputed.rule_1_2_pooled.n_unresolved")
    assert n_rec + x_rec == n_pairs
    k_dir = fact("exemplardirect.n_wrong")
    n_dir = fact("exemplardirect.n_decidable")
    x_dir = fact("exemplardirect.n_undecidable")
    assert n_dir + x_dir == n_pairs
    k_unc = fact("exemplardirect.n_wrong_involves_unclassified")
    n_join = fact("exemplar.rule_1_2_pooled_exemplar_correct_rate_in_299.n_known")
    k_join = n_join - fact(
        "exemplar.rule_1_2_pooled_exemplar_correct_rate_in_299.n_correct")
    assert k_join == 0
    k_des = fact("c129.pooled.positional_outside_trigger_set.either_side.count")
    n_des = fact("c129.pooled.positional_outside_trigger_set.either_side.n")

    # (id, label, k, n, n_excluded, credible, note)
    # credible False means the mark is drawn open: this row's own analysis
    # limits what it can support, and the reason is printed beside it. Rows are
    # sorted below by how many of the 165 pairs each instrument throws away,
    # which is the one ordering on this canvas that is not a judgement. It is
    # NOT claimed to be monotone in the reading: the direct extractor excludes
    # fewer rows than the reconstruction and lands twice as high, and that is
    # the point of printing the reason beside it rather than ranking on it.
    rows = [
        ("lb", "Displayed exemplar could not\nhave fired the trigger",
         k_lb, n_lb, 0, True, "a measured lower bound on these rows"),
        ("rec", "Reconstruction of the\nfiring mention",
         k_rec, n_rec, x_rec, True, None),
        ("dir", "Direct extractor", k_dir, n_dir, x_dir, False,
         f"{k_unc} of its {k_dir} defect rows have no\nclassified action on at "
         f"least one side"),
        ("join", "Key join to the rebuilt\ncandidates table", k_join, n_join,
         n_pairs - n_join, False,
         f"resolves {n_join} of {n_pairs} rows and finds\nno defect in any of "
         f"them"),
    ]
    rows.sort(key=lambda r: r[4])
    design = ("des", "Positional selector on the\nrebuilt candidate pool",
              k_des, n_des)

    ci = {r[0]: wilson_ci(r[2], r[3]) for r in rows}
    ci_des = wilson_ci(k_des, n_des)
    # The stored intervals are the check on the recomputation, not the source.
    assert abs(ci["lb"][0] - pooled12["wilson_ci95"][0]) < 5e-4
    assert abs(ci["lb"][1] - pooled12["wilson_ci95"][1]) < 5e-4
    assert abs(ci_des[0] - fact(
        "c129.pooled.positional_outside_trigger_set.either_side.wilson_lo")) < 5e-4
    assert abs(ci_des[1] - fact(
        "c129.pooled.positional_outside_trigger_set.either_side.wilson_hi")) < 5e-4

    # THE CLUSTER, computed and not asserted by eye: the two annotated-pair
    # readings that resolve every row or nearly every row, plus the rebuilt
    # pool, and the union of their three intervals. The band is that union. If
    # the readings ever stop agreeing the band widens and the headline, which
    # prints its own ends, says so.
    cl_lo = min(ci["lb"][0], ci["rec"][0], ci_des[0])
    cl_hi = max(ci["lb"][1], ci["rec"][1], ci_des[1])
    outlier_lo = ci["dir"][0]
    assert outlier_lo > cl_hi, \
        "the high reading now overlaps the cluster; the headline would be false"
    assert ci["join"][1] < cl_lo, \
        "the key join reading now reaches the cluster; the canvas would mislead"

    W, H = W88, 3.58
    fig = plt.figure(figsize=(W, H))
    canvas(fig)
    # The label column is measured, not guessed: the longest instrument name
    # sets the axes' left edge, so no name is clipped and no space is wasted.
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()

    def _wfrac(text, **kw):
        t = fig.text(0, 0, text, **kw)
        w = t.get_window_extent(_r).width / fig.bbox.width
        t.remove()
        return w

    names = [r[1] for r in rows] + [design[1]]
    lab_w = max(max(_wfrac(line, fontsize=T_SERIES) for line in n_.split("\n"))
                for n_ in names)
    X_LEFT = 0.012 + lab_w + 10.0 / (72 * W)
    X_RIGHT = 0.722
    A0, AH = 1.12, 1.72
    ax = fig.add_axes([X_LEFT, A0 / H, X_RIGHT - X_LEFT, AH / H])
    # Four annotated-pair rows at y 3 to 0, the rebuilt pool at y -1.35 under a
    # hairline, and headroom at the top for the band's own label.
    Y_LO, Y_HI = -1.95, 3.75
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    frame(ax, grid="x")

    # One scarlet element on the axes, and it is the finding: the interval band
    # the three low-exclusion readings agree inside. SCARLET means a failed
    # check in this set and an exemplar defect rate is the measurement of one.
    band(ax, cl_lo, cl_hi, SCARLET_TINT, zorder=0)
    ax.text((cl_lo + cl_hi) / 2, Y_HI - 0.10,
            f"{cl_lo:.2f} to {cl_hi:.2f}", ha="center", va="top",
            color=INK, zorder=6, **cased(fontsize=T_NOTE))

    # The rebuilt pool is a different substrate and sits under a hairline.
    Y_SPLIT, Y_DES = -0.62, -1.35
    ax.axhline(Y_SPLIT, color=PLATE_EDGE, lw=0.8, zorder=1)

    plotted = [(r[1], r[2], r[3], r[4], r[5], 3 - i)
               for i, r in enumerate(rows)]
    plotted.append((design[1], design[2], design[3], 0, True, Y_DES))
    for lab, k, n, n_ex, credible, y in plotted:
        e = k / n
        lo, hi = wilson_ci(k, n)
        ax.errorbar([e], [y], xerr=[[e - lo], [hi - e]], fmt="o",
                    color=GRAPHITE,
                    markerfacecolor=GRAPHITE if credible else PAPER,
                    markeredgecolor=GRAPHITE,
                    markeredgewidth=1.0 if credible else 1.3,
                    zorder=5, **dict(ERRBAR, elinewidth=1.6, capthick=1.6),
                    **DOT)

    # The two substrate headers sit INSIDE the axes, cased, each just above the
    # rows it governs. In figure coordinates they collided with the value
    # column and with the axis label; in data coordinates they cannot.
    t = ax.text(0.018, Y_SPLIT - 0.30,
                f"On the {n_des:,} rebuilt Rules 1 and 2 candidates",
                ha="left", va="top", color=MUTE, style="italic", zorder=6,
                **cased(fontsize=T_NOTE))
    case(t, linewidth=3.0)

    # Row labels and the value column in figure coordinates, on one grid.
    def y_fig(y):
        return (A0 + AH * (y - Y_LO) / (Y_HI - Y_LO)) / H

    X_VAL = 0.9960
    quals = [f"{k / n:.4f}  [{wilson_ci(k, n)[0]:.2f}, {wilson_ci(k, n)[1]:.2f}]"
             for _, k, n, *_ in plotted]
    excls = ["nothing excluded" if r[3] == 0
             else f"{r[3]} of {n_pairs} pairs excluded" for r in plotted]
    # BOTH sub-columns set this width, not just the estimates. Sizing on the
    # estimates alone put three "n of 165 pairs excluded" lines 15pt past the
    # right edge, savefig expanded the file and the measure gate failed the
    # build, which is that gate doing its job.
    X_EST = X_VAL - max([_wfrac(q, fontsize=T_SERIES) for q in quals]
                        + [_wfrac(e, fontsize=T_NOTE, style="italic")
                           for e in excls])
    assert X_EST > X_RIGHT + 7.0 / (72 * W), "value column reaches the axes"
    for (lab, k, n, n_ex, credible, y), qual, excl in zip(plotted, quals,
                                                          excls):
        yf = y_fig(y)
        nl = lab.count("\n") + 1
        fig.text(0.012, yf + (nl - 1) * 0.5 * T_SERIES * 1.18 / (72 * H), lab,
                 ha="left", va="center", fontsize=T_SERIES, color=INK,
                 linespacing=1.18)
        fig.text(X_EST, yf + 4.6 / (72 * H), qual, ha="left", va="center",
                 fontsize=T_SERIES, color=INK)
        fig.text(X_EST, yf - 5.6 / (72 * H), excl, ha="left", va="center",
                 fontsize=T_NOTE, color=MUTE, style="italic")

    fig.text(0.012, (A0 + AH + 0.060) / H,
             f"On the {n_pairs} annotated Rules 1 and 2 pairs, ordered by how "
             f"much each instrument excludes",
             ha="left", va="bottom", fontsize=T_NOTE, color=MUTE,
             style="italic")

    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=T_TICK)
    ax.set_xlabel("Share of Rules 1 and 2 candidates whose displayed exemplar "
                  "is wrong", fontsize=T_AXIS, color=INK, labelpad=3.0)
    # The two reasons the open marks are open, and the one thing a reader could
    # take from this canvas that the artifact forbids, in one block under the
    # axis label. They were per-row remarks in the value column and printed
    # through the row below.
    # THE MEASURE GATE CAUGHT THIS BLOCK. Set as three long lines it ran about
    # 60pt past the canvas, savefig(bbox="tight") expanded the file to 443pt,
    # and every type size on the page would have printed at 0.957 of its stated
    # size. It is four shorter lines now and each one is measured against the
    # measure and asserted, so a longer string fails the build rather than
    # shrinking the type.
    foot = [
        f"Open marks: the direct extractor, {k_unc} of whose {k_dir} defect "
        f"rows have no classified",
        f"action on at least one side; the key join, which resolves {n_join} of "
        f"{n_pairs} rows.",
        "Rules 3 and 4 are not drawn: their triggers reference no action label, "
        "so a high",
        "non-opposite rate there is the rule working as designed, not a defect.",
    ]
    for line in foot:
        w = _wfrac(line, fontsize=T_NOTE, style="italic")
        assert 0.012 + w < 0.996, f"footnote overflows the canvas: {line!r}"
    fig.text(0.012, 0.026, "\n".join(foot), ha="left", va="bottom",
             fontsize=T_NOTE, color=MUTE, style="italic", linespacing=1.18)

    e_dir = k_dir / n_dir
    # Three lines at the 0.88 measure, not two. Two lines put 83 characters of
    # 11pt type on a 378pt canvas, savefig expanded the file to 397pt and the
    # whole figure would have printed at 0.96 scale. Asserted, so the next edit
    # to this sentence fails the build instead of shrinking the type.
    head = ["Three readings on two substrates put the exemplar defect",
            f"rate between {cl_lo:.2f} and {cl_hi:.2f}; the fourth, "
            f"{e_dir:.2f}, fails its own check."]
    for line in head:
        assert 0.012 + _wfrac(line, fontsize=T_HEADLINE) < 0.996, \
            f"headline overflows the canvas: {line!r}"
    signature(fig, SCARLET_DEEP, "\n".join(head))
    fname = "fig-exemplar.pdf"
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)
    register(fname, "fig:exemplar",
             [_FACTS_PATH, p_lb]
             + sorted({fact_source(k) for k in EXEMPLAR_KEYS} - {_FACTS_PATH}),
             "Five readings of one quantity, the share of Rules 1 and 2 "
             "candidates whose displayed exemplar is wrong, on one axis. Four "
             f"are on the {n_pairs} annotated pairs and are ordered by how many "
             "of those pairs each instrument excludes; the fifth, under a "
             f"hairline, is on the {n_des:,} rebuilt candidates and shares no "
             "row with them. Every rate is a count over a denominator read from "
             "results/accord/VERIFIED_FACTS.json by key ("
             + ", ".join(EXEMPLAR_KEYS) + "), except the displayed-exemplar "
             "lower bound, which has no register key yet and is read as two "
             "counts from results/accord/clinical_core/"
             f"displayed_exemplar_lower_bound.json ({k_lb} of {n_lb}). No rate "
             "is typed. Every interval is a Wilson interval recomputed here "
             "from the row's own two counts and asserted against the interval "
             "the artifact stores. The SCARLET_TINT band is the union of the "
             "three low-exclusion intervals, computed on the canvas, and the "
             f"headline prints its ends ({cl_lo:.4f} to {cl_hi:.4f}); the "
             "assertion that the high reading's lower end "
             f"({outlier_lo:.4f}) clears that union fails the build if the "
             "cluster ever dissolves. Open marks, not a second hue, mark the "
             "two readings their own analysis limits: the direct extractor "
             f"({k_unc} of its {k_dir} defect rows have no classified action on "
             "at least one side) and the key join (23 rows resolved of 165). "
             "Rules 3 and 4 are named on the canvas and drawn nowhere: the "
             "artifact tags both defect_interpretation none, which this "
             "generator asserts. No clinical text, no identifiers, no pair "
             "identifiers on the canvas.")


# ============================================================================
# fig:rule-precision -- NO LONGER BOUND INTO THE THESIS.
#   Removed at the committee review: the per-rule precision table in the results
#   chapter carries this content, and a per-rule ranking statistic on two to four
#   confirmed positives was not worth a figure. The function is kept so the
#   figure can be rebuilt if it is ever wanted again; main() does not call it and
#   the verification harness does not check it.
#
#   MODIFIED: the spec's precision-by-rule numbers (pharmacist 2.2/9.7/5.1/3.6
#   and ACCORD 2.3/10.0/5.2/2.4) are not both in any committed file -- only
#   the pharmacist side is (it equals the per-rule positive rate). ACCORD's
#   actual per-rule *precision at its operating threshold* is not released
#   per-rule anywhere in results/accord. What IS released per rule is
#   ACCORD's own AUROC per rule (defect_free_strata.json), which answers the
#   same underlying question ("does the model add anything the rule alone
#   doesn't give you") without inventing a number. Substituted accordingly.
# ============================================================================
def make_fig_rule_precision():
    strata, p_strata = load("defect_free_strata.json")
    intervals, p_intervals = load("stats", "intervals.json")
    e3 = intervals["E3_defect_free_stratum"]

    per_rule = [s for s in strata["strata"] if s["name"].startswith("rule ")]
    per_rule = sorted(per_rule, key=lambda s: s["name"])

    # An AUROC on three positives is an estimate, not a quantity, and a filled
    # bar asserts a solidity these counts cannot support. Drawn as dots, with an
    # interval on the one stratum for which an interval exists, in the same
    # grammar as every other estimate comparison in the thesis. The counts that
    # used to float above the bars are now part of the row label.
    fig, ax = plt.subplots(figsize=(FIGW, 2.65))
    canvas(fig)
    fig.subplots_adjust(left=0.325, right=0.985, top=0.910, bottom=0.160)
    ys = np.arange(len(per_rule))[::-1] + 1.0
    aurocs = [s["auroc"] for s in per_rule]
    ax.axvline(0.5, **CHANCE_STYLE)
    # No interval exists for any single rule. defect_free_strata.json records
    # only a point AUROC per rule, stats/intervals.json bootstraps the pooled
    # stratum and nothing finer, and the per-pair scores a rule-level bootstrap
    # would need are not committed anywhere in this repository. Rather than
    # print four bare estimates in the same colour as the one row that does
    # carry an interval, the four drop to grey and are labelled in grey's own
    # text tone: they are what they look like, four underpowered numbers on two
    # to four positives.
    ax.plot(aurocs, ys, "o", color=GREY, linestyle="none", zorder=3, **DOT)
    # The pooled defect-free stratum is the only row with a committed interval,
    # so it is the only row drawn as an estimate; it sits below a hairline rule
    # because it pools two of the rows above it rather than joining them.
    ax.errorbar([e3["auroc"]], [0.0],
                xerr=[[e3["auroc"] - e3["unstratified_ci95"][0]],
                      [e3["unstratified_ci95"][1] - e3["auroc"]]],
                fmt="o", color=BLUE, zorder=3, **ERRBAR, **DOT)
    ax.axhline(0.62, color=GRID, linewidth=0.6, zorder=0)
    labels = [f"{s['name'].replace('rule ', 'Rule ')}\n(n={s['n']}, {s['pos']} positives)"
              for s in per_rule]
    labels.append(f"Rules 3 and 4 pooled\n(n={e3['n']}, {e3['n_pos']} positives, 95% CI)")
    ax.set_yticks(list(ys) + [0.0])
    ax.set_yticklabels(labels, fontsize=8.0, linespacing=1.25)
    for lab, y in zip(ax.get_yticklabels(), list(ys) + [0.0]):
        lab.set_color(MUTE if y > 0 else INK)
    ax.set_xlabel("ACCORD AUROC within the rule")
    ax.set_xlim(0.14, 0.84)
    ax.set_ylim(-0.60, 4.60)
    ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.text(0.5, 4.78, "chance", fontsize=8.0, color=MUTE, ha="center", va="bottom")
    # The measured axis keeps its spine; the categorical one loses it, because a
    # rule running the height of five row labels reads as a bracket joining all
    # five, and the hairline below rule 4 says the opposite.
    frame(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if SHOW_INFIGURE_TITLES: ax.set_title("ACCORD\u2019s own ranking performance does not exceed chance\nwithin any single rule", fontsize=8.3)
    if SHOW_INFIGURE_NOTES: fig.text(0.02, 0.02,
             "Deviation from spec: per-rule ACCORD *precision* (as in the written spec) is not in any\n"
             "committed result file; per-rule AUROC is (defect_free_strata.json) and is shown instead.\n"
             "Individual-rule CIs are not computable here (per-pair scores not committed); only the\n"
             "pooled rules-3+4 CI (stats/intervals.json E3) is available and is shown as a reference band.",
             fontsize=8.0, color=MUTE)
    fname = "fig-rule-precision.pdf"
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)
    register(fname, "fig:rule-precision", [p_strata, p_intervals],
             "MODIFIED from spec: shows per-rule AUROC (sourced) rather than per-rule precision "
             "(pharmacist side sourced, ACCORD side not committed per-rule anywhere in results/accord). "
             "No per-rule interval exists in defect_free_strata.json or in stats/intervals.json, and "
             "the per-pair scores one would be computed from are not committed, so the four "
             "single-rule rows are drawn in grey as underpowered point estimates and only the "
             "pooled rules-3+4 row, which has a committed CI, is drawn as an estimate.")


# ============================================================================
# Figure 6 -- fig:prevalence  (ch5_weak_supervision.tex)
# ============================================================================
def make_fig_prevalence():
    step01, p_step01 = load("step0_1_join_and_calibration.json")
    xtab = step01["weak_confidence_is_degenerate"]["crosstab_label_x_confidence"]
    weak_pos, weak_neg = xtab["0.95"]["contradiction"], xtab["1.0"]["not_contradiction"]
    weak_n = weak_pos + weak_neg

    step2, p_step2 = load("step2_signal_check.json")
    d = step2["contra_prob_vs_pharm_label__ambiguous_as_negative_n299"]
    gold_pos, gold_n = d["n_pos"], d["n"]
    d_all_incl = step2["contra_prob_vs_pharm_label__ambiguous_as_positive_n299"]
    gold_ambig = d_all_incl["n_pos"] - gold_pos  # 58 (pos incl. ambiguous) - 12 (pos excl.) = 46
    gold_neg = gold_n - gold_pos - gold_ambig

    fig, ax = plt.subplots(figsize=(FIGW, 2.3))
    canvas(fig)
    fig.subplots_adjust(left=0.22, right=0.90, top=0.86, bottom=0.20)
    cats_weak = [("contradiction", weak_pos), ("not contradiction", weak_neg)]
    cats_gold = [("contradiction", gold_pos), ("not contradiction", gold_neg), ("ambiguous", gold_ambig)]

    def stacked(y, cats, n, hatches, small_dir):
        left = 0
        for (lab, val), h in zip(cats, hatches):
            frac = val / n
            ax.barh(y, frac, left=left, height=0.55, linewidth=0,
                    color=PALE if h else BLUE)
            centre = left + frac / 2
            if frac > 0.40:
                ax.text(centre, y, f"{lab}\n{val}/{n} ({frac*100:.1f}%)",
                        ha="center", va="center", fontsize=8.0, linespacing=1.25,
                        color="white" if not h else INK)
            elif frac > 0.20:
                ax.text(centre, y, f"{lab}\n{val}/{n}\n({frac*100:.1f}%)",
                        ha="center", va="center", fontsize=8.0, linespacing=1.25,
                        color="white" if not h else INK)
            else:
                y_lab = y + small_dir * 0.55
                ax.annotate(f"{lab}\n{val}/{n} ({frac*100:.1f}%)", xy=(centre, y + small_dir * 0.25),
                            xytext=(centre, y_lab), ha="center",
                            va="bottom" if small_dir > 0 else "top", fontsize=8.0,
                            arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.6))
            left += frac

    stacked(1, cats_weak, weak_n, [None, HATCH_A], small_dir=1)
    stacked(0, cats_gold, gold_n, [None, HATCH_A, HATCH_B], small_dir=-1)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([f"Weak labels\n(n={weak_n})", f"Pharmacist gold\n(n={gold_n})"], fontsize=8.0)
    ax.set_ylim(-1.0, 1.9)
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("Share of labels")
    frame(ax)
    ratio = (weak_pos / weak_n) / (gold_pos / gold_n)
    ax.plot([1.05, 1.05], [0, 1], color=MUTE, lw=0.6, clip_on=False)
    ax.text(1.09, 0.5, f"~{ratio:.0f}x", va="center", fontsize=8.0, color=MUTE,
            clip_on=False)
    implied_wrong = int(np.floor(weak_pos * (1 - gold_pos / gold_n)))
    if SHOW_INFIGURE_TITLES: ax.set_title(f"Marginals are close to complementary, not noisy copies\n"
                 f"implied floor: ≥{implied_wrong:,} of {weak_pos:,} positive weak labels wrong "
                 f"at the measured {gold_pos/gold_n*100:.1f}% base rate", fontsize=8.0)
    fname = "fig-prevalence.pdf"
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)
    register(fname, "fig:prevalence", [p_step01, p_step2],
             "Weak-label counts from step0_1 weak_confidence crosstab; gold counts "
             "(12 contradiction / 241 not / 46 ambiguous) reconstructed from step2_signal_check "
             "ambiguous-included vs ambiguous-excluded base rates.")


# ============================================================================
# Figure 7 -- fig:tracking  (ch5_weak_supervision.tex)
# ============================================================================
def make_fig_tracking():
    step01, p_step01 = load("step0_1_join_and_calibration.json")
    cal = step01["rule_stratified_calibration"]
    rules = ["1", "2", "3", "4"]
    ns = {r: cal[r]["n_gold"] for r in rules}
    llama = [cal[r]["llama_weak_label_pos_rate_on_these_299_pairs"] for r in rules]
    accord = [cal[r]["accord_predicted_pos_rate_gold"] for r in rules]
    truth = [cal[r]["pharmacist_true_pos_rate_gold"] for r in rules]

    fig, ax = plt.subplots(figsize=(FIGW, 2.5))
    canvas(fig)
    fig.subplots_adjust(left=0.115, right=0.760, top=0.965, bottom=0.175)
    xs = np.arange(4)

    def plot_series(vals, marker, ls, colour, label, y_label):
        cis = [wilson_ci(round(v * ns[r]), ns[r]) for v, r in zip(vals, rules)]
        lo = [v - c[0] for v, c in zip(vals, cis)]
        hi = [c[1] - v for v, c in zip(vals, cis)]
        ax.errorbar(xs, vals, yerr=[lo, hi], marker=marker, linestyle=ls, color=colour,
                    markersize=4.0, linewidth=1.1, **ERRBAR)
        # Direct label past the right end of the series, in the series' own
        # colour, on its own baseline so no two labels can stack.
        ax.text(3.14, y_label, label, fontsize=8.0, color=colour, va="center",
                ha="left", clip_on=False, linespacing=1.25)

    plot_series(llama, "o", "-", BLUE, "Llama 3 8B\n(teacher weak label)", 0.50)
    plot_series(accord, "s", "--", GREY, "ACCORD\n(student)", 0.90)
    plot_series(truth, "^", ":", SCARLET_CORE, "pharmacist truth", 0.10)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"Rule {r}\n(n={ns[r]})" for r in rules], fontsize=8.0,
                       linespacing=1.25)
    ax.set_ylabel("Positive rate")
    ax.set_ylim(-0.06, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.35, 3.05)
    frame(ax)
    if SHOW_INFIGURE_TITLES: ax.set_title("The student tracks the teacher\u2019s marginal, not the truth\u2019s",
                 fontsize=8.5)
    fname = "fig-tracking.pdf"
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)
    register(fname, "fig:tracking", [p_step01],
             "Wilson score 95% CIs added per point (n from n_gold per rule: 134/31/79/55); "
             "not in the original spec, added per the CI non-negotiable.")


# ============================================================================
# Figure 8 -- fig:met-exchange  (ch8_metrics.tex)
# ============================================================================
def make_fig_met_exchange():
    baselines, p_baselines = load("step3_baselines.json")
    # The two GPT-4o systems were withdrawn: they came from a hosted API that
    # the methodology chapter records as a departure from the data use
    # agreement, and no reported number in this thesis derives from them any
    # more. The frontier rows replace them, read from the compliant run's own
    # metrics artifact rather than from step3_baselines.json.
    fm = json.loads(open(os.path.join(
        REPO, "results/accord/differential/"
              "frontier_anthropic_metrics.json")).read())
    def _cells(strategy):
        c = fm["by_strategy"][strategy]["G_253"]
        return {"tp": c["tp"], "tn": c["tn"], "fp": c["fp"], "fn": c["fn"]}
    systems = {
        "ACCORD": baselines["accord_binary_at_0.5_253core"],
        "Claude Sonnet 5 zero-shot": _cells("zero_shot"),
        "Claude Sonnet 5 few-shot": _cells("few_shot"),
        "Llama 3 8B": baselines["llama_baseline"],
    }
    ref = systems["ACCORD"]
    TP0, TN0, FP0, FN0 = ref["tp"], ref["tn"], ref["fp"], ref["fn"]
    P = TP0 + FN0
    N = TN0 + FP0
    mcc0 = mcc_from_counts(TP0, TN0, FP0, FN0)

    fig, ax = plt.subplots(figsize=(FIGW, 3.39))
    canvas(fig)
    # The right margin is wide because the three slope labels live in it, and
    # it is a fixed number of 9pt lines wide however big the canvas gets.
    # right=0.69626, not the 0.7802 that holds the old margin absolute. The
    # three slope labels sit in this margin and the longest of them measures
    # 118.4pt; at 0.7802 it ran 35pt off the right edge of the canvas, savefig's
    # tight box grew to catch it, and \includegraphics then scaled the whole
    # figure DOWN by 8.3 percent to fit the measure -- which set this figure's
    # "9pt" type at 8.3pt on the page, under the set's own floor. The margin now
    # holds the longest label exactly, the canvas comes out at its nominal
    # width, and the plot prints 9 percent larger than it did even though its
    # drawn width is unchanged.
    fig.subplots_adjust(left=0.1080, right=0.69626, top=0.8294, bottom=0.1372)
    # Zoomed to the occupied wedge. The old range ran to x = 12 only to give the
    # two long leaders somewhere to go, and the leaders are gone. Starting at a
    # negative x is what pulls the boxed left spine off the origin cluster.
    X_LO, X_HI = -1.2, 7.0
    # Y_HI is the FRAME TOP, not the data maximum. set_yticks() below unions the
    # view interval with its own ticks, so a 200 tick pulls the top of the box to
    # 200 whatever set_ylim said; building the region polygons to a smaller
    # number left a white strip across the plot under the frame top. One
    # constant, used by the limits, the ticks and the fills alike.
    Y_LO, Y_HI = -30.0, 200.0
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(Y_LO, Y_HI)

    d = np.linspace(0, X_HI, 200)
    slope_ba = N / P
    # The indifference lines are not measurements: they are what each statistic
    # would accept, computed from the class counts. What a reader needs is the
    # SPAN of what different statistics will trade, so the span is filled.
    deltas = np.linspace(0, min(TP0, X_HI), 60)
    dtn_curve = []
    for delta in deltas:
        tp = TP0 - delta
        fn = P - tp

        def f(tn):
            fp = N - tn
            return mcc_from_counts(tp, tn, fp, fn) - mcc0

        try:
            dtn_curve.append(brentq(f, 0, N) - TN0)
        except ValueError:
            dtn_curve.append(np.nan)
    dtn_curve = np.array(dtn_curve)

    # Three regions, and between them they are the whole content of this
    # figure. Above the balanced-accuracy slope every statistic on this canvas
    # counts the trade as a gain; below the PABAK slope none of them does;
    # between the two it depends entirely on which statistic was chosen, and
    # that span is the thing a reader has to see. Wider region first, narrower
    # over it. HAIR under MIST rather than MIST under something darker: there
    # is no structural stop between MIST and SLATE, and SLATE at area scale is
    # a mark colour, not a ground.
    dfull = np.linspace(X_LO, X_HI, 200)
    ax.fill_between(dfull, np.clip(dfull * slope_ba, Y_LO, Y_HI), Y_HI,
                    facecolor=HAIR, edgecolor="none", linewidth=0, zorder=1)
    ax.fill_between(dfull, Y_LO, np.clip(dfull, Y_LO, Y_HI), facecolor=HAIR,
                    edgecolor="none", linewidth=0, zorder=1)
    ax.fill_between(dfull, np.clip(dfull, Y_LO, Y_HI),
                    np.clip(dfull * slope_ba, Y_LO, Y_HI), facecolor=MIST,
                    edgecolor="none", linewidth=0, zorder=2)
    case(ax.text(6.7, 26.0, "only some of these statistics\n"
                            "accept a trade in this wedge",
                 ha="right", va="center", fontsize=T_NOTE, color=MUTE,
                 linespacing=1.24, zorder=4))
    ax.plot(d, d * slope_ba, color=GRAPHITE, lw=LW_REF, zorder=3)
    ax.plot(d, d, color=GRAPHITE, lw=LW_REF, zorder=3)
    ax.plot(deltas, dtn_curve, color=GRAPHITE, lw=LW_REF, zorder=3)
    # The x = 0 reference, now clearly separate from the boxed left spine.
    ax.axvline(0, color=SLATE, linestyle=REF_DASH, lw=LW_REF, zorder=2)

    # Each slope is labelled at the right edge of the plot at its own exit y,
    # horizontal. Where two exit y values fall within 11pt the lower one drops
    # to the next slot and takes a 3pt tick at the frame; a tick at the frame
    # is not a leader.
    exits = [
        (min(X_HI * slope_ba, Y_HI), f"balanced accuracy (slope {slope_ba:.1f})"),
        (X_HI, "PABAK (slope 1)"),
        (float(dtn_curve[-1]), f"MCC through ACCORD ({mcc0:.3f})"),
    ]
    exits.sort(key=lambda e: -e[0])
    pt_per_y = (0.8294 - 0.1372) * 3.39 * 72 / (Y_HI - Y_LO)
    placed_y = []
    for y, text in exits:
        yy = y
        for prev in placed_y:
            if abs(yy - prev) * pt_per_y < 11.0:
                yy = prev - 11.0 / pt_per_y
        placed_y.append(yy)
        ax.plot([X_HI - 0.10, X_HI], [yy, yy], color=GRAPHITE, lw=1.0,
                clip_on=False, zorder=5)
        ax.text(X_HI + 0.18, yy, text, ha="left", va="center",
                fontsize=T_SERIES, color=GRAPHITE, clip_on=False, zorder=5)

    # The four systems. ACCORD is the claim: this whole figure is about where
    # ACCORD sits against exchange rates it never chose, so it is the one solid
    # scarlet mark and it gives up its amber on this canvas.
    spec = {
        "ACCORD": ("D", SCARLET_DEEP, SCARLET_CORE, SCARLET_DEEP, 7.0,
                   (-6, 7), "right", "bottom"),
        "Llama 3 8B": ("D", PAPER, PLUM_CORE, PLUM_DEEP, MS_MARK,
                       (8, -6), "left", "top"),
        # EMERALD and CYAN, matching the hues fig-forest gives these two rows.
        # COBALT and AZURE belonged to the withdrawn GPT-4o systems and are not
        # reused, so a reader who learned the palette earlier is not misled.
        "Claude Sonnet 5 zero-shot": ("s", PAPER, EMERALD.CORE, EMERALD.DEEP,
                                      MS_MARK, (7, 4), "left", "bottom"),
        "Claude Sonnet 5 few-shot": ("^", PAPER, CYAN.CORE, CYAN.DEEP,
                                     MS_MARK, (7, -5), "left", "top"),
    }
    for name, s_ in systems.items():
        delta = TP0 - s_["tp"]
        dtn = s_["tn"] - TN0
        mk, face, edge, tone, ms, off, ha, va = spec[name]
        ax.plot([delta], [dtn], mk, markerfacecolor=face, markeredgecolor=edge,
                markeredgewidth=LW_CHIP if name == "ACCORD" else 1.2, ms=ms,
                zorder=6)
        case(ax.annotate(name, (delta, dtn), textcoords="offset points",
                         xytext=off, ha=ha, va=va, fontsize=T_SERIES,
                         color=tone, zorder=7))

    ax.set_xlabel(r"$\delta$ = forgone true positives, relative to ACCORD")
    ax.set_ylabel(r"$\Delta$TN, relative to ACCORD")
    ax.set_xticks([0, 2, 4, 6])
    ax.set_yticks([0, 50, 100, 150, Y_HI])
    frame(ax)
    signature(fig, SCARLET_DEEP,
              "Two of three systems clear every exchange rate ACCORD\u2019s "
              "counts define.")
    fname = "fig-met-exchange.pdf"
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)
    register(fname, "fig:met-exchange", [p_baselines],
             "Indifference lines are closed-form (PABAK, balanced accuracy) or numerically "
             "root-solved (MCC, scipy.optimize.brentq) from the confusion-matrix class counts "
             "in step3_baselines.json; not fabricated curve shapes.")


# ============================================================================
# Figure 9 -- fig:met-sharpened  (ch8_metrics.tex)
# ============================================================================
def make_fig_met_sharpened():
    baselines, p_baselines = load("step3_baselines.json")
    intervals, p_intervals = load("stats", "intervals.json")
    order = ["gpt4o_zeroshot", "gpt4o_fewshot", "accord_binary_at_0.5_253core", "llama_baseline"]
    labels = ["GPT-4o zero-shot", "GPT-4o few-shot", "ACCORD", "Llama 3 8B"]
    key_map = {"gpt4o_zeroshot": "gpt4o_zeroshot", "gpt4o_fewshot": "gpt4o_fewshot",
               "accord_binary_at_0.5_253core": "accord_binary_at_0.5_253core",
               "llama_baseline": "llama_baseline"}

    fig, (axL, axR) = plt.subplots(2, 1, figsize=(FIGW, 3.60), sharey=False)
    canvas(fig)
    fig.subplots_adjust(left=0.300, right=0.985, top=0.945, bottom=0.100, hspace=0.60)
    xs = np.arange(4)

    P = 12
    recalls, cis = [], []
    for k in order:
        s = baselines[k]
        recalls.append(s["tp"])
        cis.append(wilson_ci(s["tp"], P))
    err = [[r - c[0] * P for r, c in zip(recalls, cis)], [c[1] * P - r for r, c in zip(recalls, cis)]]
    ys = xs[::-1]
    # Same set-wide rule as every other interval here: the caps are the bar's
    # own colour, because they belong to the bar and not to a second series.
    axL.barh(ys, recalls, xerr=err, color=BLUE, linewidth=0, height=0.55,
             ecolor=BLUE, error_kw=ERRBAR)
    axL.set_xlim(0, 13.4)
    axL.set_xticks([0, 3, 6, 9, 12])
    axL.set_xlabel(f"True positives (of {P})")
    axL.set_yticks(ys)
    axL.set_yticklabels(labels, fontsize=8.0)
    axL.set_ylim(-0.62, 3.62)
    frame(axL)
    accord_pp = baselines["accord_binary_at_0.5_253core"]["tp"] + baselines["accord_binary_at_0.5_253core"]["fp"]
    llama_pp = baselines["llama_baseline"]["tp"] + baselines["llama_baseline"]["fp"]
    n253 = baselines["accord_binary_at_0.5_253core"]["tp"] + baselines["accord_binary_at_0.5_253core"]["fp"] + \
           baselines["accord_binary_at_0.5_253core"]["tn"] + baselines["accord_binary_at_0.5_253core"]["fn"]
    # Inside the bar, right-aligned: at this measure the same note set outside
    # the bar ran off the canvas.
    axL.text(recalls[2] - 0.4, ys[2], f"{accord_pp}/{n253} predicted positive (contaminated)",
             va="center", ha="right", fontsize=8.0, color="white")
    axL.text(recalls[3] - 0.4, ys[3], f"{llama_pp}/{n253} predicted positive",
             va="center", ha="right", fontsize=8.0, color="white")
    if SHOW_INFIGURE_TITLES: axL.set_title("Recall", fontsize=8.5)

    mcc = intervals["E2_mcc_intervals"]
    vals = [mcc[key_map[k]]["mcc"] for k in order]
    lo = [mcc[key_map[k]]["ci95"][0] for k in order]
    hi = [mcc[key_map[k]]["ci95"][1] for k in order]
    err2 = [[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]]
    axR.axvline(0, **CHANCE_STYLE)
    axR.errorbar(vals, ys, xerr=err2, fmt="o", color=BLUE, linestyle="none",
                 **ERRBAR, **DOT)
    axR.set_yticks(ys)
    axR.set_yticklabels(labels, fontsize=8.0)
    axR.set_xlabel("Matthews correlation, 95% CI")
    axR.set_ylim(-0.62, 3.62)
    axR.set_xlim(-0.28, 0.28)
    axR.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    axR.text(0.0, 3.80, "chance", fontsize=8.0, color=MUTE, va="bottom", ha="center")
    frame(axR)

    for ax, tag in ((axL, "(a)"), (axR, "(b)")):
        pos = ax.get_position()
        fig.text(0.004, pos.y1 + 0.010, tag, ha="left", va="bottom",
                 fontsize=8.0, style="italic", color=INK)
    if SHOW_INFIGURE_TITLES: axR.set_title("Matthews correlation, 95% CI", fontsize=8.5)

    if SHOW_INFIGURE_TITLES: fig.suptitle("The high-recall systems predict positive almost everywhere;\n"
                 "no MCC interval excludes chance", fontsize=8.0, y=0.99)
    fname = "fig-met-sharpened.pdf"
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)
    register(fname, "fig:met-sharpened", [p_baselines, p_intervals],
             "Recall CIs are Wilson score on (tp, 12) -- not in the original spec, added per "
             "the CI non-negotiable. MCC CIs from stats/intervals.json E2 (all four contain "
             "zero, per CORRECTIONS.md C36).")


# ============================================================================
# Figure 10 -- fig:disc-strata  (ch9_discussion.tex)
#   MODIFIED: no per-pair prediction scores are committed to this repo, so a
#   real ROC curve cannot be drawn (only the aggregate AUROC + CI + n + p
#   are). Shown as point-estimate-with-CI panels instead of a fabricated ROC
#   curve shape.
# ============================================================================
def make_fig_disc_strata():
    decontam, p_decontam = load("decontam", "corrected_gold_metrics.json")
    # Both rows now come from the decontaminated run. The defect-free stratum used
    # to be read from stats/intervals.json E3, which is computed on the
    # contaminated released scores, so the two rows were not comparable.
    control, p_control = load("control", "final_report.json")
    e3 = control["ensemble_mean_of_seed_scores"]["gold_defect_free_rule34"]

    agg = decontam["aggregate"]
    # Sample sizes, positives and the Mann-Whitney p live in the row label, out
    # of the plot area, so the data region holds nothing but the two estimates
    # and their intervals. The estimate itself is read off the axis.
    rows = [
        ("All non-ambiguous pairs\n(decontaminated 3-seed ensemble)\n"
         f"n={decontam['n']}, {decontam['n_positives']} positives, "
         f"MWU p={agg['ensemble_mannwhitney_p']:.3f}",
         agg["ensemble_auroc"], agg["ensemble_ci95"]),
        ("Rules 3 and 4 only\n(exemplar-defect-free stratum)\n"
         f"n={e3['n']}, {e3['pos']} positives",
         e3["auroc"], e3["bootstrap"]["ci95"]),
    ]

    fig, ax = plt.subplots(figsize=(FIGW, 2.20))
    canvas(fig)
    fig.subplots_adjust(left=0.400, right=0.985, top=0.900, bottom=0.300)
    ys = np.arange(len(rows))[::-1]
    ax.axvline(0.5, **CHANCE_STYLE)
    # Two rows, two hues, because they are two different populations scored by
    # two different models and the caption has to say so in words when they
    # share one colour. The defect-free stratum takes the truth hue: it is the
    # subset whose evidence is correct by construction, which is the closest
    # this evaluation gets to clean ground.
    row_hues = [BLUE2, GREEN]
    for y, (label, auroc, ci), hue in zip(ys, rows, row_hues):
        ax.errorbar([auroc], [y], xerr=[[auroc - ci[0]], [ci[1] - auroc]], fmt="o",
                    color=hue, zorder=3, **ERRBAR, **DOT)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.0, linespacing=1.3)
    ax.set_xlim(0.16, 0.94)
    ax.set_ylim(-0.62, 1.62)
    ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("AUROC, 95% bootstrap interval")
    ax.text(0.5, 1.74, "chance", fontsize=8.0, color=MUTE, ha="center", va="bottom")
    # The spine now runs past the stratum's upper bound at 0.828, which it used
    # to stop short of, so neither interval ends beyond the scale. The
    # categorical axis has no spine: the vertical rule that used to join these
    # two rows read as a bracket, and a bracket asserts that the lower row is a
    # subset of the upper one. It is not. The caption says in as many words
    # that the second row is a different model on a different stratum, and the
    # figure was contradicting its own caption.
    frame(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # Note sits below the axes, clear of both intervals.
    fig.text(0.985, 0.025,
             f"bootstrap: {control['bootstrap_n_resamples']:,} resamples, "
             f"{e3['bootstrap']['n_boot_valid']:,} valid on the defect-free stratum",
             ha="right", fontsize=8.0, color=MUTE)
    if SHOW_INFIGURE_TITLES: fig.suptitle("On evidence correct by construction, ranking is at chance,\nnot merely attenuated toward it",
                 fontsize=8.3, y=0.99)
    if SHOW_INFIGURE_NOTES: fig.text(0.5, 0.03,
             "No ROC curve is drawn: per-pair prediction scores are not committed to this repository (MIMIC-III DUA\n"
             "restricts released files to aggregates). Panels show the committed point estimate and its bootstrap CI instead.",
             ha="center", fontsize=8.0, color=MUTE)
    fname = "fig-disc-strata.pdf"
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)
    register(fname, "fig:disc-strata", [p_decontam, p_control],
             "MODIFIED from spec (ROC panels) to point-estimate+CI panels because per-pair scores "
             "are not committed. Full-set row uses the decontaminated ensemble AUROC 0.5145 / CI "
             "[0.321, 0.703] / p=0.867 per CORRECTIONS.md C16, replacing the withdrawn 0.460/p=0.64. "
             "Defect-free row now reads control/final_report.json "
             "ensemble_mean_of_seed_scores.gold_defect_free_rule34 (0.5469, CI [0.242, 0.828]), the "
             "decontaminated counterpart, replacing stats/intervals.json E3 (0.5078) which is "
             "computed on the contaminated released scores; its stratified CI and its "
             "dropped-resample count belong to that superseded bootstrap and are no longer drawn.")


# ============================================================================
# Figure 11 -- fig:disc-actionability  (ch9_discussion.tex)
#   Structural/schema diagram: the MIMIC-III field-availability fact that
#   DISCHTIME precedes the discharge summary's CHARTDATE-only timestamp, and
#   nursing notes carry CHARTTIME but discharge summaries do not. This is a
#   documented data-model property (CORRECTIONS.md C6), not a statistic
#   drawn from a results JSON, so no CI applies -- there is no estimate on
#   this figure.
# ============================================================================
def make_fig_disc_actionability():
    """Fig 11. A fat timeline, not a hairline axis.

    The three phases ARE the content, so they are drawn as regions: a WASH
    ground per phase running the height of the plate, and a 16pt TINT band on
    the timeline itself carrying the phase name inside it, cased. The first
    phase runs back to the left edge and the last runs on to the right edge,
    because both are open-ended in time; only the middle phase has two ends.

    One grammar, so nothing collides: event names sit ABOVE the band on two
    staggered levels, marks sit BELOW it with their labels beneath, and the
    inside of the band carries the phase names and nothing else. The field
    detail that used to ride these labels in parentheses is in the caption.
    """
    W, H = FIGW, 2.13
    fig, ax = plt.subplots(figsize=(W, H))
    canvas(fig)
    plate(fig)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W * 72)
    ax.set_ylim(0, H * 72)
    ax.axis("off")

    # The point grid moves with the plate, which is 9 percent larger in both
    # directions than it was; the type on it does not move, so the timeline
    # gains air rather than size.
    x_nurse, x_order, x_disch, x_summary = 52.0, 164.0, 293.0, 367.0
    x0, x1 = 1.5, W * 72 - 1.5
    g_lo, g_hi = 11.0, 122.0
    band_lo, band_hi = 57.0, 74.0          # the phase band
    mid = (band_lo + band_hi) / 2

    phases = [
        (x0, x_order, EMERALD_WASH, EMERALD_TINT, EMERALD_DEEP,
         "intervention possible"),
        (x_order, x_disch, AMBER_WASH, AMBER_TINT, AMBER_DEEP,
         "window closing"),
        (x_disch, x1, SCARLET_WASH, SCARLET_TINT, SCARLET_DEEP,
         "no intervention possible"),
    ]
    for a, b, wash, tint, deep, name in phases:
        ax.add_patch(mpatches.Rectangle((a, g_lo), b - a, g_hi - g_lo,
                                        facecolor=wash, edgecolor="none",
                                        linewidth=0, zorder=0))
    for a, b, wash, tint, deep, name in phases:
        ax.add_patch(mpatches.Rectangle((a, band_lo), b - a, band_hi - band_lo,
                                        facecolor=tint, edgecolor=deep,
                                        linewidth=LW_NODE, zorder=2))
    for a, b, wash, tint, deep, name in phases:
        case(ax.text((a + b) / 2, mid, name, ha="center", va="center",
                     fontsize=T_NOTE, color=INK, zorder=6))

    # Event ticks pierce the band; names above it on two staggered levels, so
    # no two names share a line and none needs a leader.
    # The tick meets the band at both edges rather than running through it: the
    # inside of the band belongs to the phase name, and a tick crossing that
    # name is the one collision this layout can produce.
    events = [
        (x_nurse, "last nursing note", GRAPHITE, 7.0, "center"),
        (x_order, "discharge order", GRAPHITE, 22.0, "center"),
        (x_disch, "the patient leaves", GRAPHITE, 7.0, "center"),
        (x_summary, "discharge summary finalised", SCARLET_DEEP, 22.0, "right"),
    ]
    for x, label, tone, dy, ha in events:
        for lo, hi in ((band_lo - 6.0, band_lo), (band_hi, band_hi + 6.0)):
            ax.plot([x, x], [lo, hi], color=tone, lw=LW_CONNECTOR,
                    solid_capstyle="butt", zorder=5)
        ax.text(x1 - 4.0 if ha == "right" else x, band_hi + dy, label, ha=ha,
                va="bottom", fontsize=T_SERIES, color=tone, zorder=6)

    y_mark = band_lo - 10.0
    y_note = band_lo - 22.0
    # The point a detector could have fired: a proposal that was measured, so a
    # CORE mark in the entity blue rather than an accent.
    ax.plot([x_order], [y_mark], "o", color=COBALT_CORE, markersize=MS_MARK,
            markeredgecolor=COBALT_DEEP, markeredgewidth=1.0, zorder=7)
    ax.text(x_order, y_note, "proposed detection point", ha="center", va="top",
            fontsize=T_SERIES, color=COBALT_DEEP, zorder=7)

    # The system's trigger point: the one saturated scarlet mark on the canvas.
    ax.plot([x_summary], [y_mark], "D", color=SCARLET_DEEP, markersize=9.4,
            markeredgecolor=SCARLET_CORE, markeredgewidth=LW_CHIP, zorder=7)
    ax.text(x_summary, y_mark, "!", ha="center", va="center", fontsize=T_NOTE,
            color=PAPER, fontweight="bold", zorder=8)
    ax.text(x1 - 4.0, y_note, "the system triggers here", ha="right", va="top",
            fontsize=T_SERIES, color=SCARLET_DEEP, zorder=7)

    signature(fig, SCARLET_DEEP,
              "The document the system triggers on is finalised at or "
              "after the patient leaves.")
    fname = "fig-disc-actionability.pdf"
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)
    register(fname, "fig:disc-actionability", ["accord/CORRECTIONS.md (C6)"],
             "Structural/schema diagram, not derived from a numeric result file; documents a "
             "MIMIC-III field-availability fact. No CI applies (no estimate is plotted).")


# ============================================================================
# Figure 12 -- fig:cohort-flow  (ch3_methodology.tex)
#   The CONSORT-style path from the source database to the two evaluated sets.
#   Nine stages, four exclusions. Every count is read from a committed file:
#   the ICD and length-of-stay counts from stats/cohort_icd_counts.json, the
#   candidate count from generality/transfer_results_correct.json, the gold
#   counts from step2_signal_check.json, and the four stages whose own
#   upstream artifact is quarantined off this repository (note-role coverage,
#   pair selection, mention extraction, the drawn annotation sample) from the
#   thesis register, which is itself committed under results/accord. The
#   both-roles-to-pairs exclusion is also read from the register: it is the
#   entire shortfall between has_both_roles and valid_admissions_both_selected,
#   and it is entirely on the nursing side, since a nursing-role admission is
#   selectable only when some nursing note carries a chart time strictly
#   before discharge, and the discharge side carries no matching condition.
#
#   One stage the funnel would normally carry is still skipped for want of a
#   committed count and is named in the register note: the exclusion between
#   the length-of-stay filter and the both-roles stage. That difference is
#   arithmetic on two committed counts, but it is not itself a recorded
#   exclusion count, and a funnel that invents its own subtractions is the
#   thing this set exists to avoid.
# ============================================================================
def make_fig_cohort_flow():
    # Every count is read from the thesis register by key. The register is
    # imported above as _FACTS through fact(), which refuses withdrawn keys.
    KEYS = [
        "cohort.heart_failure.pre_los", "cohort.heart_failure.post_los",
        "cohort.heart_failure.removed", "cohort_chain.has_both_roles",
        "cohort_chain.valid_admissions_both_selected",
        "cohort_chain.both_roles_not_selectable",
        "extraction.medication_extraction.total_mentions",
        "extraction.medication_extraction.classified_mentions",
        "extraction.medication_extraction.unclassified_derived",
        "extraction.temporal_prefilter.final_candidates",
        "extraction.annotation_sample.total_sampled",
        "c120.g299.n", "c120.g299.n_positives", "c120.g253.n",
        "c120.g253.n_positives", "overcallamb.n_ambiguous",
    ]
    n_pre = fact("cohort.heart_failure.pre_los")
    n_post = fact("cohort.heart_failure.post_los")
    n_los_removed = fact("cohort.heart_failure.removed")
    assert n_pre - n_post == n_los_removed
    n_both_roles = fact("cohort_chain.has_both_roles")
    n_pairs = fact("cohort_chain.valid_admissions_both_selected")
    n_not_selectable = fact("cohort_chain.both_roles_not_selectable")
    assert n_both_roles - n_pairs == n_not_selectable
    n_mentions = fact("extraction.medication_extraction.total_mentions")
    n_classified = fact("extraction.medication_extraction.classified_mentions")
    n_unclassified = fact("extraction.medication_extraction.unclassified_derived")
    assert n_mentions - n_classified == n_unclassified
    n_candidates = fact("extraction.temporal_prefilter.final_candidates")
    n_sampled = fact("extraction.annotation_sample.total_sampled")
    n_annot = fact("c120.g299.n")
    n_pos = fact("c120.g299.n_positives")
    n_core = fact("c120.g253.n")
    assert fact("c120.g253.n_positives") == n_pos
    # Differences computed here, not typed: drops no file records as its own
    # exclusion are drawn as dashed stubs.
    n_no_roles = n_post - n_both_roles
    # AUDIT ITEM 5 gave the step from classified mentions to prefilter
    # candidates a dashed box holding n_classified - n_candidates, labelled as
    # mentions dropped.
    # VISUAL AUDIT. That subtraction mixes units. A candidate is a pair of
    # mentions matched under a rule, not a mention, so 88,336 mentions minus
    # 1,978 candidates is not a count of dropped mentions, and no file records
    # one. The difference is no longer computed or drawn. The box stays,
    # dashed, and says what happens at the step instead: the unit changes. The
    # two registered counts stand in their own stage boxes, each with its unit.
    n_not_sampled = n_candidates - n_sampled
    n_not_annotated = n_sampled - n_annot
    # FIGURE POLISH. This stub is drawn solid, which the caption defines as a
    # recorded exclusion, and it used to be computed here as n_annot - n_core,
    # which the same caption defines as dashed. The count IS recorded (the
    # pharmacist coded these pairs ambiguous, and the register carries it), so
    # the stub stays solid and the number is now read, not subtracted; the
    # subtraction is kept as the check.
    n_ambiguous = fact("overcallamb.n_ambiguous")
    assert n_ambiguous == n_annot - n_core
    # Cross-check the ambiguous count against the signal-check artifact, which
    # records the positives on the same 299 rows both ways.
    step2, p_step2 = load("step2_signal_check.json")
    d_neg = step2["contra_prob_vs_pharm_label__ambiguous_as_negative_n299"]
    d_amb = step2["contra_prob_vs_pharm_label__ambiguous_as_positive_n299"]
    assert d_amb["n_pos"] - d_neg["n_pos"] == n_ambiguous

    W, H = W88, 4.34
    fig, ax = plt.subplots(figsize=(W, H))
    canvas(fig)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W * 72)
    ax.set_ylim(0, H * 72)
    ax.axis("off")
    x_right = W * 72 - 1.5

    # The one stub that is not an exclusion: no count, by design (see above).
    UNIT_NOTE = "unit changes: mentions are\npaired into candidates"
    # (stage name, count with unit, stub below this stage or None)
    # a stub is (text, recorded): recorded -> solid, a computed difference -> dashed
    stages = [
        ("Heart failure, ICD-9 prefix", f"{n_pre:,} admissions",
         (f"{n_los_removed:,} admissions with no\npositive length of stay", True)),
        ("Length-of-stay filter", f"{n_post:,} admissions",
         (f"{n_no_roles:,} admissions without\nboth note types", False)),
        ("Both note types present", f"{n_both_roles:,} admissions",
         (f"{n_not_selectable:,} admissions, no nursing\nnote before discharge", True)),
        ("One note pair per admission", f"{n_pairs:,} pairs", None),
        ("Medication mentions extracted", f"{n_mentions:,} mentions",
         (f"{n_unclassified:,} mentions with\nno action class", True)),
        # AUDIT ITEM 5. All three stubs below now state what left the path at
        # that step (an exclusion), matching the "left the path" heading; the
        # previous two stated what was kept instead, which contradicted it.
        ("Mentions with an action class", f"{n_classified:,} mentions",
         (UNIT_NOTE, False)),
        ("After the temporal prefilter", f"{n_candidates:,} candidates",
         (f"{n_not_sampled:,} candidates not\nsampled for annotation", False)),
        ("Sampled for annotation", f"{n_sampled:,} pairs",
         (f"{n_not_annotated:,} sampled pair\nnot annotated", False)),
        ("Annotated by the pharmacist", f"{n_annot:,} pairs",
         (f"{n_ambiguous:,} pairs coded ambiguous", True)),
        ("Non-ambiguous evaluation core",
         f"{n_core:,} pairs, {n_pos:,} confirmed contradictions", None),
    ]
    assert n_not_sampled > 0 and n_not_annotated >= 0

    box_x, box_w = 26.0, 222.0
    x_gut = 262.0
    box_h, last_h, gap = 17.0, 28.0, 10.0
    y_top = H * 72 - 34.0
    tops, y = [], y_top
    for i in range(len(stages)):
        h = last_h if i == len(stages) - 1 else box_h
        tops.append((y, h))
        y -= h + gap

    # Phase groups share one ground tone; the phase is carried by its name,
    # the box hue (cobalt extraction, emerald the pharmacist) and the rail.
    phases = [
        (range(0, 4), HAIR, MIST, SLATE, SLATE, "cohort construction"),
        (range(4, 7), HAIR, COBALT_TINT, COBALT_DEEP, COBALT_DEEP,
         "extraction and mining"),
        (range(7, 10), HAIR, EMERALD_TINT, EMERALD_DEEP, EMERALD_DEEP,
         "annotation"),
    ]
    phase_of = {}
    for rows, wash, tint, deep, rail, name in phases:
        idx = list(rows)
        for i in idx:
            phase_of[i] = (tint, deep)
        b_hi = tops[idx[0]][0] + 5.0
        b_lo = tops[idx[-1]][0] - tops[idx[-1]][1] - 5.0
        ax.add_patch(mpatches.Rectangle((0, b_lo), W * 72, b_hi - b_lo,
                                        facecolor=wash, edgecolor="none",
                                        linewidth=0, zorder=0))
        ax.add_patch(mpatches.Rectangle((7.0, b_lo + 2.0), 3.0,
                                        b_hi - b_lo - 4.0, facecolor=rail,
                                        edgecolor="none", linewidth=0, zorder=1))
        ax.text(17.0, (b_lo + b_hi) / 2, name, ha="center", va="center",
                rotation=90, fontsize=T_NOTE, color=MUTE, zorder=2)

    x_stem = box_x + box_w / 2
    for i, ((t, h), (name, count, stub)) in enumerate(zip(tops, stages)):
        last = i == len(stages) - 1
        mid = t - h / 2
        if last:
            # The evaluated core: the pharmacist's set, a solid emerald chip.
            chip(ax, box_x, t - h, box_w, h, EMERALD,
                 f"{name}\n{count}", fontsize=T_SERIES, radius=3.0, zorder=4)
        else:
            tint, deep = phase_of[i]
            node(ax, box_x, t - h, box_w, h, tint, deep, radius=2.5, zorder=4)
            ax.text(box_x + 7.0, mid, name, ha="left", va="center",
                    fontsize=T_SERIES, color=INK, zorder=5)
            # SLATE is too light for type (3.48:1), so the cohort phase's
            # counts are set in GRAPHITE.
            ax.text(box_x + box_w - 7.0, mid, count, ha="right", va="center",
                    fontsize=T_SERIES, color=GRAPHITE if deep == SLATE else deep,
                    zorder=5)
            arrow(ax, (x_stem, t - h), (x_stem, t - h - gap), color=GRAPHITE,
                  lw=LW_CONNECTOR, zorder=6)
        if stub is not None:
            text, recorded = stub
            ls = "solid" if recorded else (0, (3.0, 2.0))
            x_elbow = box_x + box_w + 5.0
            y_gap = t - h - gap / 2
            ax.plot([box_x + box_w, x_elbow], [mid, mid], color=SLATE,
                    lw=1.0, zorder=6, solid_capstyle="butt", linestyle=ls)
            ax.plot([x_elbow, x_elbow], [mid, y_gap], color=SLATE,
                    lw=1.0, zorder=6, solid_capstyle="butt", linestyle=ls)
            ax.plot([x_elbow, x_gut], [y_gap, y_gap], color=SLATE,
                    lw=1.0, zorder=6, solid_capstyle="butt", linestyle=ls)
            lines = text.count("\n") + 1
            ch = 10.0 * lines + 5.0
            p = node(ax, x_gut, y_gap - ch / 2, x_right - 2.0 - x_gut, ch,
                     PAPER if not recorded else MIST, SLATE, radius=2.0,
                     zorder=4, lw=1.0)
            if not recorded:
                p.set_linestyle((0, (3.0, 2.0)))
            # The unit note is a remark, not a count, and is set in italic so
            # it does not read as one more exclusion in the column.
            ax.text(x_gut + 5.0, y_gap, text, ha="left", va="center",
                    fontsize=T_NOTE, color=MUTE if not recorded else GRAPHITE,
                    style="italic" if text == UNIT_NOTE else "normal",
                    linespacing=1.15, zorder=5)

    ax.text(x_gut, y_top + 6.0, "left the path", ha="left", va="bottom",
            fontsize=T_NOTE, color=MUTE, style="italic")

    signature(fig, EMERALD_DEEP,
              f"Of {n_candidates:,} candidates, {n_pos:,} contradictions survive "
              f"sampling and annotation.")
    fname = "fig-cohort-flow.pdf"
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)
    srcs = [_FACTS_PATH] + sorted({fact_source(k) for k in KEYS} - {_FACTS_PATH}) + [p_step2]
    register(fname, "fig:cohort-flow", srcs,
             "Path from the source database to the evaluated core, ten stages. Every "
             "count is read from results/accord/VERIFIED_FACTS.json by key ("
             + ", ".join(KEYS) + "), and every right-hand count carries its unit. "
             "Every stub under the 'left the path' heading states an exclusion, "
             "never a kept count (audit item 5), except the one at the temporal "
             "prefilter, which carries no count: solid stubs are exclusions the "
             f"register records as counts ({n_los_removed} with no positive length "
             f"of stay, {n_not_selectable} with no nursing note before discharge, "
             f"{n_unclassified:,} unclassified mentions, {n_ambiguous} coded "
             "ambiguous). Dashed stubs are drops no file records as an exclusion, "
             "drawn as differences computed in code from two register counts: "
             f"admissions after the length-of-stay filter minus admissions with both "
             f"note types ({n_no_roles}); candidates minus the sample "
             f"({n_not_sampled:,}); the sample minus the annotated pairs "
             f"({n_not_annotated}). The dashed stub at the temporal prefilter "
             "used to print mentions with an action class minus candidates as a "
             "count of dropped mentions. A candidate is a pair of mentions "
             "matched under a rule, so that difference mixed units and is no "
             "longer computed or drawn; the stub now says in italic that the "
             "unit changes from mentions to candidates, and the two registered "
             f"counts ({n_classified:,} mentions, {n_candidates:,} candidates) stand "
             "in their stage boxes. The ambiguous count is read from "
             f"overcallamb.n_ambiguous, asserted equal to {n_annot} minus {n_core} and "
             "cross-checked against the signal-check file. The core is drawn in the "
             "pharmacist's emerald rather than scarlet, which the set reserves for a "
             f"failed check. Headline built from the register: of {n_candidates:,} "
             f"candidates, {n_pos} contradictions survive sampling and annotation.")


# ============================================================================
# Figure 13 -- fig:survey-audit  (ch4_results.tex)
#   One rate of a whole, so one bar. Three segments over the 75 coded papers:
#   the strict reading, the papers that clear only the permissive reading, and
#   the papers that clear neither.
# ============================================================================
def make_fig_survey_audit():
    coding, p_coding = load("survey", "coding_summary.json")
    n_total = coding["n_included_papers"]
    dist = coding["distributions"]
    n_strict = dist["audits_selection_strict"]["yes"]
    n_literal = dist["audits_selection_literal"]["yes"]
    n_neither = dist["audits_selection_literal"]["no"]
    # The middle segment is the population the two readings disagree about. It
    # is not stored on its own line and is not invented: it is the permissive
    # count less the strict count, both read above, and the three segments are
    # checked against the denominator before anything is drawn.
    n_permissive_only = n_literal - n_strict
    assert n_strict + n_permissive_only + n_neither == n_total

    fig, ax = plt.subplots(figsize=(FIGW, 1.66))
    canvas(fig)
    # The right margin holds the last tick label, not the bar: ticked to the
    # denominator, the "75" is centred on the axis end, so half of it sits
    # outside the axes and the canvas has to be wide enough to contain it.
    fig.subplots_adjust(left=0.015, right=0.982, top=0.985, bottom=0.235)

    # Counts, not percentages, on the axis: the denominator is 75 and the top
    # tick is 75, so the whole is on the canvas and every segment is read
    # against it directly.
    ax.set_xlim(0, n_total)
    ax.set_ylim(0, 1)

    y_bar_lo, y_bar_hi = 0.845, 1.0
    segments = [
        (n_strict, SCARLET_CORE, SCARLET_CORE,
         f"{n_strict} papers ({n_strict / n_total * 100:.1f}%) report the check "
         "under the strict reading"),
        # The middle segment is a real, named population, not a leftover: the
        # papers the two readings disagree about. It used to be drawn in the
        # pale band tone, which is the tone this set uses for structure, so the
        # one thing on the canvas that is a finding looked like backing. It now
        # carries a series hue, light enough to stay behind the claim segment
        # and dark enough to be a segment rather than a shadow.
        (n_permissive_only, SKY, MUTE,
         f"{n_permissive_only} papers ({n_permissive_only / n_total * 100:.1f}%) "
         "report it only under the permissive reading"),
        (n_neither, GREY, MUTE,
         f"{n_neither} papers ({n_neither / n_total * 100:.1f}%) do not report "
         "it under either reading"),
    ]

    left = 0.0
    placed = []
    for n, fill, text_colour, label in segments:
        ax.add_patch(mpatches.Rectangle((left, y_bar_lo), n, y_bar_hi - y_bar_lo,
                                        facecolor=fill, edgecolor="none",
                                        zorder=2))
        placed.append((left, n, fill, text_colour, label))
        left += n

    # A staircase of direct labels: each one begins under the left edge of its
    # own segment and is tied to it by a leader dropped from that edge, so no
    # key is needed to say which label belongs to which colour. The four-paper
    # segment is a quarter of an inch on the page and will not hold a label
    # inside it, and neither pale nor grey may carry text, so all three labels
    # sit below the bar.
    #
    # The rows run widest segment first, which is what keeps the leaders off
    # the type: a leader dropped from a segment further left passes only rows
    # whose text starts further right, so it never strikes through a label. In
    # the other order, every leader crossed the two labels above it.
    y_rows = [0.615, 0.395, 0.175]
    for (left, n, fill, text_colour, label), y_row in zip(
            sorted(placed, key=lambda p: -p[0]), y_rows):
        leader = SCARLET_CORE if fill is SCARLET_CORE else GREY
        # Leaders anchor at the segment's left edge, with one stated exception:
        # a segment starting at x = 0 anchors at its midpoint, because a
        # full-height hairline directly above the 0 tick, on an axes whose left
        # spine is deliberately absent, reads as a spine in claim colour.
        anchor = left + n / 2 if left == 0 else left
        ax.plot([anchor, anchor], [y_row + 0.055, y_bar_lo], color=leader,
                lw=0.5, zorder=1)
        ax.text(anchor + 0.7, y_row, label, ha="left", va="center",
                fontsize=8.0, color=text_colour)

    ax.set_xlabel("Papers coded")
    ax.set_xticks([0, 25, 50, n_total])
    ax.set_yticks([])
    # The count axis measures something and keeps its spine; the bar has one row
    # and no categorical axis, so nothing runs up the left edge.
    frame(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fname = "fig-survey-audit.pdf"
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)
    register(fname, "fig:survey-audit", [p_coding],
             "One bar, three segments, a rate of a whole. The strict count, the permissive "
             "count and the count clearing neither reading are read straight from the coding "
             "summary; the middle segment is the permissive count less the strict count, and "
             "the three are asserted to sum to the committed denominator before the figure is "
             "drawn. Scarlet marks the four-paper segment, which is what this figure claims; "
             "the other two segments carry no accent.")


# ============================================================================
# Run all buildable figures + report skipped ones
# ============================================================================
SKIPPED = [
    {
        "label": "fig:met-sharpened",
        "chapter": "retired",
        "reason": (
            "Retired rather than repaired. The figure's headline was a hardcoded "
            "string, \"no MCC interval excludes chance\", and its system list was a "
            "hardcoded four, so it could not track intervals.json. When the two "
            "frontier rows were added it went on asserting a claim Chapter 4 had "
            "withdrawn, in a committed PDF that no chapter bound. fig:forest shows "
            "the same Matthews family from the artifact, with every system and a "
            "computed headline, so this one is deleted rather than kept in step."
        ),
    },
    {
        "label": "fig:asymmetry",
        "chapter": "ch3_data.tex",
        "reason": (
            "Spec requires classified/unclassified mention counts (219,683 / 88,336 / "
            "131,347) and a discharge-vs-nursing note-role split (79,385 / 8,951) for the "
            "heart-failure cohort. No committed file under results/accord contains this "
            "breakdown for the heart-failure cohort: aki/sepsis have phase2 summaries with "
            "total_mentions/classified_mentions, but the primary heart-failure cohort does "
            "not, and no file anywhere gives a discharge-vs-nursing split for any cohort. "
            "mention_count_crosscheck.json gives only a single crosscheck total (218,544 "
            "official vs 227,363 simplified), which is insufficient to reconstruct either "
            "panel. Left as \\figspec."
        ),
    },
    {
        "label": "fig:arch",
        "chapter": "ch6_architecture.tex",
        "reason": (
            "Spec requires the model's parameter budget and per-module status (PPC deleted, "
            "GGHE randn(20,256) bank, 1.54M vs ~7.5M total params, which heads receive "
            "gradient). These facts are documented only in accord/CORRECTIONS.md (C4) as "
            "prose describing source files (accord_phase3_v2_ablations.py) that are not "
            "committed anywhere in this repository -- not in results/accord, not "
            "elsewhere. There is no JSON to point \\includegraphics-adjacent code at. "
            "Left as \\figspec."
        ),
    },
]

# fig:falsification used to be listed above as unbuildable, on the grounds that
# transfer_results_correct.json did not exist here. It does exist, at
# results/accord/generality/. The figure was retired (two numbers on an axis;
# reason of record in tools/check_figure_invariants.py) and its generator,
# make_figures_extra.py, which had been switched off, was removed from the
# repository; its numbers are in a table in the results chapter. The one number
# the original entry was right about is the held-out 45.2%: 0.4515 appears in
# no result file anywhere, so that reference line was removed rather than
# sourced.


def main():
    make_fig_pipeline_status()
    # make_fig_intro_headline() is retired. It drew the first rows of the
    # capability figure sixty pages before that figure drew them, and the
    # capability figure is itself retired for drawing hosted rows, so the
    # duplicate outlived the thing it duplicated. What it argued -- that the
    # labeller flags an order of magnitude more pairs than the pharmacist
    # confirms -- is now argued by fig:ceiling from the same three integers,
    # and argued further: fig:ceiling shows what that ratio costs in
    # attainable agreement rather than only that the ratio is large.
    make_fig_funnel()
    make_fig_exemplar()
    # make_fig_rule_precision() removed from the thesis per committee review;
    # the per-rule table carries this. The drawing function is left in place
    # above so the figure can be rebuilt if it is ever wanted again, but
    # nothing binds it and nothing calls it.
    make_fig_prevalence()
    make_fig_tracking()
    # make_fig_met_exchange() is retired. Four markers, two of them stacked at
    # the origin, on a y axis running to 200 with nothing drawn between 0 and
    # 185, under a 203-word caption that had to carry a worked example because
    # the panel could not. A figure whose caption has to do the arithmetic for
    # the reader is a paragraph, and the exchange rate it computes is stated in
    # the discussion text.
    # make_fig_met_sharpened() is retired, not merely unbound. Its headline
    # was the literal string "no MCC interval excludes chance" and its system
    # list was a hardcoded four, so it could not follow the data: after the
    # frontier rows were added to intervals.json it kept asserting a claim the
    # results chapter had withdrawn, and it shipped that claim in a committed
    # PDF. fig:forest draws the same Matthews family from the artifact, with
    # all six systems and a headline it computes, so nothing is lost by
    # deleting this one and a stale duplicate is removed.
    # make_fig_disc_strata() and make_fig_survey_audit() are no longer bound.
    # disc-strata redrew fig:forest's second family identically, sixty pages
    # apart; survey-audit was cut by the author and its content is one sentence
    # of body text. Both drawing functions are left in place above, exactly as
    # make_fig_rule_precision was handled at the committee review, so either
    # figure can be rebuilt if it is ever wanted again.
    # make_fig_disc_actionability() is retired. A timeline with three coloured
    # zones, no measurement of any kind, and colour used purely evaluatively:
    # green for the window it approves of, scarlet for the one it does not.
    # This set's own rule is that colour is a carrying channel for identity and
    # state, not a verdict, and a figure that plots nothing cannot be evidence
    # for the verdict it colours. The field-availability fact it documents is
    # one sentence of the discussion chapter.
    make_fig_cohort_flow()

    # Read from the committed manifest, written next to the PDFs just built;
    # the two are the same file unless ACCORD_FIG_OUT redirects the output.
    manifest_src = os.path.join(FIGDIR, "manifest.json")
    manifest_path = os.path.join(OUTDIR, "manifest.json")
    # Entries for figures built by fig_batch2.py and fig_ceiling.py are kept:
    # this script does not build them and must not erase their notes.
    ours = {m["label"] for m in MANIFEST}
    try:
        with open(manifest_src) as f:
            kept = [m for m in json.load(f).get("built", []) if m["label"] not in ours]
    except (OSError, ValueError):
        kept = []
    with open(manifest_path, "w") as f:
        json.dump({"built": MANIFEST + kept, "skipped": SKIPPED}, f, indent=2)

    print(f"Built {len(MANIFEST)} figures, skipped {len(SKIPPED)}.")
    for m in MANIFEST:
        print(f"  BUILT  {m['label']:28s} -> {m['file']}")
    for s in SKIPPED:
        print(f"  SKIP   {s['label']:28s} ({s['chapter']})")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
