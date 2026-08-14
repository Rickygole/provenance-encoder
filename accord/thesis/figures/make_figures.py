#!/usr/bin/env python3
"""
Render the ACCORD thesis figures from committed result files.

Every number drawn on every axis in this script is read out of a JSON file
under /Users/rickygole/provenance-encoder/results/accord/ (or computed
analytically from numbers read out of those files -- e.g. the MCC
indifference curve in make_fig_met_exchange is a closed-form function of a
confusion matrix that IS in a result file). Nothing here is a literal typed
in because it "looked right." Where a spec number could not be found in any
committed file, the figure is either built with a substitute that IS
sourced (and the substitution is stated in the figure's caption strip and in
the manifest), or it is skipped entirely -- see SKIPPED at the bottom and
the printed report.

Run:
    python3 /Users/rickygole/provenance-encoder/accord/thesis/figures/make_figures.py

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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Long provenance notes belong in the LaTeX caption, not inside the plot.
# Set True to draw them again.
SHOW_INFIGURE_NOTES = False
SHOW_INFIGURE_TITLES = False
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import norm
from scipy.optimize import brentq

np.random.seed(42)

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO = "/Users/rickygole/provenance-encoder"
RESULTS = os.path.join(REPO, "results", "accord")
FIGDIR = os.path.join(REPO, "accord", "thesis", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def load(*parts):
    path = os.path.join(RESULTS, *parts)
    with open(path) as f:
        return json.load(f), path


# ----------------------------------------------------------------------------
# matplotlib style: serif, print-legible, greyscale-safe
#
# Identical to style.setup() and to the notebook's setup cell, so the same
# figure drawn from any of the three is the same figure.
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    # Pinned so that this script, style.py and the notebook all draw the same
    # text metrics. See the note in style.py.
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    # A negative tick label must be a hyphen-minus, not U+2212: the thesis bans
    # the unicode minus on the page and on every canvas, and matplotlib draws it
    # by default.
    "axes.unicode_minus": False,
    "text.color": "#1A1A1A",
    "axes.labelcolor": "#1A1A1A",
    "axes.edgecolor": "#1A1A1A",
    "xtick.color": "#1A1A1A",
    "ytick.color": "#1A1A1A",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": False,
    "grid.linewidth": 0.25,
    "grid.color": "#E8E8E8",
    "lines.linewidth": 1.1,
    "legend.frameon": False,
    "figure.dpi": 150,
    # Matches the notebook, so a figure built here and the same figure built
    # there have the same margins and neither clips an annotation.
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Palette. Shared verbatim with style.py and with the notebook's palette cell,
# so the whole set looks like one document. Full rationale in style.py.
#
# Greyscale luminance, 0.299R + 0.587G + 0.114B, out of 255:
#     INK      #1A1A1A  near-black       26   text, spines, ticks
#     BLUE     #0A2795  royal blue       43   the primary data colour
#     SCARLET  #D81E05  scarlet          83   one accented element per figure
#     TEAL     #00817B  deep teal        90   reference lines only
#     MUTE     #5B5B5B  grey             91   secondary annotation text
#     GREY     #8A94A6  cool grey       147   de-emphasised data
#     PALE     #C9D2E3  pale blue-grey  209   secondary bars and bands
# Every distinction that carries data is a large lightness step within one hue
# family: BLUE against PALE is 166 of 255, BLUE against GREY 104, BLUE against
# TEAL 47. Nothing depends on telling red from green.
INK = "#1A1A1A"
BLUE = "#0A2795"
PALE = "#C9D2E3"
TEAL = "#00817B"
SCARLET = "#D81E05"
MUTE = "#5B5B5B"
GREY = "#8A94A6"
GRID = "#E8E8E8"

# Historical names, still used by the figures in this file that the thesis does
# not bind (funnel, prevalence, tracking, met-sharpened).
GREY_DARK, GREY_LIGHT, GREY_FILL, GREY_DEAD = BLUE, TEAL, PALE, GREY
ALERT, GREY_MED = SCARLET, MUTE
HATCH_A = "///"
HATCH_B = "\\\\\\"
# One grammar for the whole set, applied identically in all fourteen figures.
#
#   CHANCE_STYLE  every reference value (chance, a null, an initialisation):
#                 grey, hairline, one dash pattern, labelled at the line itself.
#   ERRBAR/DOT    every point estimate: a dot with a thin interval and small
#                 caps, in the dot's own colour. Estimates are never bars.
#   Bars          only for a count or a rate of a whole, no outline, 0.55 wide.
#   SCARLET       exactly one element per canvas: this study, or the failure.
CHANCE_STYLE = dict(color=GREY, linestyle=(0, (4, 2)), linewidth=0.6, zorder=1)
ERRBAR = dict(elinewidth=1.0, capthick=1.0, capsize=2.5)
DOT = dict(markersize=4.6)


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

# Every figure in this file is bound at width=0.9\linewidth in a 6 inch text
# block, so it is printed 5.40 inches wide. Each one is therefore DRAWN 5.40
# inches wide (5.36 plus the 0.02 inch save pad on each side), which makes the
# scale factor 1.0 and puts 8 point text on the page at 8 point. Before this,
# figures were drawn 5.7 to 7.2 inches wide and shrunk to fit, so the same
# 8 point label printed anywhere between 6.0 and 7.7 point depending on which
# figure it was in.
FIGW = 5.36
PRINT_W = 5.40

MANIFEST = []


def canvas(fig):
    """Pin the saved bounding box to the whole figure.

    savefig crops to the artists, so the width of the file depended on how far
    the longest annotation happened to reach and no two figures came out the
    same measure. One invisible full-figure patch fixes the crop to the canvas,
    so FIGW plus the 0.02 inch pad on each side is exactly the 5.40 inches the
    document prints. Margins are then a layout decision, made in
    subplots_adjust, rather than an accident of the text.
    """
    fig.patches.append(mpatches.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                          fill=False, linewidth=0, edgecolor="none"))


def register(fname, label, sources, note=""):
    MANIFEST.append({
        "file": os.path.join("figures", fname),
        "label": label,
        "sources": sources,
        "note": note,
    })


def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion. Closed-form,
    deterministic, no simulation."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    adj = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    # Clamped to [0, 1] and to the estimate itself, identically to the helper in
    # style.py: at k = 0 the exact lower bound is 0 and the arithmetic lands a
    # few parts in 10^18 either side of it, which is not a real interval but is
    # enough to make a plotting library refuse a negative error bar.
    lo = min(p, max(0.0, (centre - adj) / denom))
    hi = max(p, min(1.0, (centre + adj) / denom))
    return (lo, hi)


def mcc_from_counts(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if den == 0:
        return 0.0
    return num / den


# ============================================================================
# Figure 1 -- fig:pipeline-status  (ch1_introduction.tex)
# ============================================================================
def make_fig_pipeline_status():
    exemplar, p_exemplar = load("exemplar", "exemplar_correctness_recomputed.json")
    # The former reference rate was supplied by a task description, never measured
    # (REVISION_LOG R8). Use the measured wrong rate on the resolved candidates.
    _ex = json.load(open(os.path.join(RESULTS, "exemplar", "exemplar_correctness_recomputed.json")))["rule_1_2_pooled"]
    defect_rate = _ex["n_wrong"] / _ex["n_resolved"]

    step01, p_step01 = load("diag", "step0_1_join_and_calibration.json")
    xtab = step01["weak_confidence_is_degenerate"]["crosstab_label_x_confidence"]
    weak_pos = xtab["0.95"]["contradiction"]
    weak_neg = xtab["1.0"]["not_contradiction"]
    weak_pos_rate = weak_pos / (weak_pos + weak_neg)

    step2, p_step2 = load("diag", "step2_signal_check.json")
    pharm_rate = step2["contra_prob_vs_pharm_label__ambiguous_excluded_n253"]["base_rate"]
    # One multiplier for the whole thesis, and it is the gold-set one: the weak
    # labeller's marginal on the 299 annotated pairs over the pharmacist's base
    # rate on those same 299 pairs. Both terms are measured on the same rows,
    # which is the only version of this ratio that compares like with like.
    # This figure used to divide the training-pool marginal by the gold base
    # rate, which is a different denominator on each side of the division and
    # gave a second, slightly different headline number that the introduction
    # then had to spend a sentence in two captions explaining away.
    exp1a, p_exp1a = load("differential", "exp1a_gold_error_rates_by_group.json")
    _lang = exp1a["language_group3"]
    gold_weak_pos = sum(g["TP"] + g["FP"] for g in _lang)
    gold_n = sum(g["n"] for g in _lang)
    _d299 = step2["contra_prob_vs_pharm_label__ambiguous_as_negative_n299"]
    pharm_rate_all = _d299["base_rate"]
    ratio = (gold_weak_pos / gold_n) / pharm_rate_all

    decontam, p_decontam = load("decontam", "corrected_gold_metrics.json")
    auroc = decontam["aggregate"]["ensemble_auroc"]
    ci = decontam["aggregate"]["ensemble_ci95"]

    # The pipeline runs down the page, not across it. Six stages in one column
    # of identical boxes, the verdict and the diagnostic that produced it in two
    # aligned text columns beside them. Drawn on a strict point grid: the axes
    # span exactly the figure and one data unit is one typographic point, so the
    # boxes are identical and the rows are on one baseline by construction.
    W, H = FIGW, 3.66
    fig, ax = plt.subplots(figsize=(W, H))
    canvas(fig)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W * 72)
    ax.set_ylim(0, H * 72)
    ax.axis("off")

    stages = ["Cohort construction", "Medication extraction",
              "Rule-based candidate\ngeneration", "LLM weak supervision",
              "Detector training", "Evaluation"]
    # Sentence case, not the shouted capitals of the old filled badges: these
    # are typographic labels now, and the rule beside them carries the verdict.
    # No asterisks. They pointed at a note that had already been moved into the
    # chapter caption, so on the page they were two marks referring to nothing.
    verdicts = ["Works", "Unvalidated",
                f"Enriches but\nmis-attributes evidence\n({defect_rate*100:.1f}%)",
                f"Fails\n({ratio:.0f}x prevalence mismatch,\ngold set)",
                "Three claims falsified",
                f"Fails\n(AUROC {auroc:.3f},\n95% CI [{ci[0]:.2f}, {ci[1]:.2f}])"]
    # Written out rather than left as a dash: the thesis bans en and em dashes,
    # and a double hyphen renders on canvas as something a reader reads as one.
    # The parameter-displacement diagnostic is set as mathematics because that
    # is what it is; spelled out in underscored identifiers it read as a
    # variable name someone had pasted in.
    diagnostics = ["no diagnostic run", "no diagnostic run",
                   "action-set match vs\n.iloc[0] index",
                   "weak-label marginal\nvs gold marginal",
                   r"$\|\Delta\theta\|$ from init," + "\n" + r"OOD transfer $F_1$",
                   "bootstrap CI on AUROC\nbefore write-up"]

    n = len(stages)
    box_x, box_w, box_h, pitch = 3.0, 139.0, 30.0, 42.0
    x_rule, x_verdict, x_diag = 150.0, 160.0, 284.0
    y_top = 246.0
    tops = [y_top - i * pitch for i in range(n)]
    for i, (t, s, v, d) in enumerate(zip(tops, stages, verdicts, diagnostics)):
        ax.add_patch(FancyBboxPatch(
            (box_x, t - box_h), box_w, box_h,
            boxstyle="round,pad=0,rounding_size=2.0", mutation_aspect=1.0,
            linewidth=0.6, edgecolor=INK, facecolor="white"))
        ax.text(box_x + box_w / 2, t - box_h / 2, s, ha="center", va="center",
                fontsize=8.0, color=INK, linespacing=1.25)
        if i < n - 1:
            ax.annotate("", xy=(box_x + box_w / 2, t - box_h - pitch + box_h),
                        xytext=(box_x + box_w / 2, t - box_h),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.6,
                                        mutation_scale=7, shrinkA=0, shrinkB=0))
        ax.text(x_verdict, t - box_h / 2, v, ha="left", va="center",
                fontsize=8.0, color=INK, linespacing=1.25)
        ax.text(x_diag, t - box_h / 2, d, ha="left", va="center", fontsize=8.0,
                style="italic", color=MUTE, linespacing=1.25)

    ax.text(x_verdict, y_top + 7, "verdict", ha="left", va="bottom",
            fontsize=8.0, color=MUTE)
    ax.text(x_diag, y_top + 7, "diagnostic", ha="left", va="bottom",
            fontsize=8.0, color=MUTE)

    # The verdict rule, running beside the stages it judges. Three verdicts,
    # three weights of rail, broken between them: grey against the one stage a
    # diagnostic passed, pale against the one stage no diagnostic ever ran on,
    # and scarlet against the four a diagnostic faulted. The first two used to
    # share a single unbroken grey segment, which gave a stage that was never
    # tested exactly the same mark as a stage that was tested and passed.
    ax.plot([x_rule, x_rule], [tops[0], tops[0] - box_h],
            color=GREY, lw=0.8, solid_capstyle="butt")
    ax.plot([x_rule, x_rule], [tops[1], tops[1] - box_h],
            color=PALE, lw=0.8, solid_capstyle="butt")
    ax.plot([x_rule, x_rule], [tops[2], tops[5] - box_h],
            color=SCARLET, lw=0.8, solid_capstyle="butt")

    if SHOW_INFIGURE_TITLES: ax.text(W * 36, H * 72 - 4, "The pipeline as run, stage by stage",
            ha="center", va="top", fontsize=9.0)
    if SHOW_INFIGURE_NOTES: fig.text(0.5, 0.02,
            "*Categorical only (not re-plotted with a number): CORRECTIONS.md C1/C2/C4 document that the\n"
            "authority-weight and rule-F1 claims did not survive re-audit; no committed result file in this\n"
            "repo carries the underlying numbers, so they are not re-typed here.",
            ha="center", fontsize=8.0, color=MUTE)
    fname = "fig-pipeline-status.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    register(fname, "fig:pipeline-status",
             [p_exemplar, p_step01, p_step2, p_decontam],
             "Two stage verdicts are categorical narrative with no committed numeric source "
             "(CORRECTIONS.md); the asterisks that used to flag them are gone, because the "
             "chapter caption already carries that note. The three parenthetical numbers "
             "(47.45% wrong of resolved, ~22x, AUROC 0.5145 with CI) are computed from the cited JSON files. "
             "The multiplier is now the gold-set one, 261/299 over 12/299, the same ratio "
             "fig:intro-headline brackets, replacing the earlier train-pool-over-gold-set ratio "
             "whose two terms had different denominators. "
             "AUROC corrected per CORRECTIONS.md C16 (was 0.460, now the decontaminated "
             "3-seed ensemble 0.5145).")


# ============================================================================
# Figure 2 -- fig:intro-headline  (ch1_introduction.tex)
# ============================================================================
def make_fig_intro_headline():
    step01, p_step01 = load("diag", "step0_1_join_and_calibration.json")
    xtab = step01["weak_confidence_is_degenerate"]["crosstab_label_x_confidence"]
    weak_pos, weak_neg = xtab["0.95"]["contradiction"], xtab["1.0"]["not_contradiction"]
    weak_n = weak_pos + weak_neg
    weak_rate = weak_pos / weak_n

    exp1a, p_exp1a = load("differential", "exp1a_gold_error_rates_by_group.json")
    lang = exp1a["language_group3"]
    pred_pos = sum(g["TP"] + g["FP"] for g in lang)
    pred_n = sum(g["n"] for g in lang)
    pred_rate = pred_pos / pred_n

    step2, p_step2 = load("diag", "step2_signal_check.json")
    d = step2["contra_prob_vs_pharm_label__ambiguous_as_negative_n299"]
    pharm_pos, pharm_n = d["n_pos"], d["n"]
    pharm_rate = pharm_pos / pharm_n

    intervals, p_intervals = load("stats", "intervals.json")
    mcc = intervals["E2_mcc_intervals"]
    order = ["gpt4o_zeroshot", "gpt4o_fewshot", "accord_binary_at_0.5_253core", "llama_baseline"]
    labels = ["GPT-4o zero-shot", "GPT-4o few-shot", "ACCORD (contam.)", "Llama 3 8B"]

    # The graphical abstract of the thesis: two stacked panels sharing one label
    # gutter and one measure, both read left to right along a common grammar.
    # (a) is a rate of a whole, so bars; (b) is an estimate, so dots with
    # intervals. Stacked rather than side by side because at this measure two
    # label gutters side by side leave no room for either panel's data.
    fig, (axL, axR) = plt.subplots(2, 1, figsize=(FIGW, 3.42),
                                   gridspec_kw={"height_ratios": [74, 84]})
    canvas(fig)
    fig.subplots_adjust(left=0.330, right=0.985, top=0.938, bottom=0.122, hspace=0.640)

    # Middle bar: 261/299 is the weak labeller's marginal on the gold set, not
    # ACCORD's own predicted-positive count (that is 278/299). reconciliation_c28.json
    # exists to settle exactly this, and the bar was previously mislabelled ACCORD.
    # The denominators live in the tick labels now, out of the plot area.
    bars = [("Weak labeller, train pool", weak_pos, weak_n, wilson_ci(weak_pos, weak_n)),
            ("Weak labeller, gold set", pred_pos, pred_n, wilson_ci(pred_pos, pred_n)),
            ("Pharmacist truth, gold set", pharm_pos, pharm_n, wilson_ci(pharm_pos, pharm_n))]
    ypos = np.arange(len(bars))[::-1]
    rates = [b[1] / b[2] for b in bars]
    los = [b[3][0] for b in bars]
    his = [b[3][1] for b in bars]
    errs = [[r - lo for r, lo in zip(rates, los)], [hi - r for r, hi in zip(rates, his)]]
    # The interval is the bar's own, so it is the bar's own colour. Black caps
    # on a blue bar read as a second series laid over the first. But a blue
    # interval on a blue bar loses its lower half inside the fill, so the
    # interval reads one-sided and the bar reads as if it were exact: the
    # interval is therefore cased in white, a slightly wider white stroke drawn
    # under it, so both halves read against the fill and against the paper.
    axL.barh(ypos, rates, color=BLUE, linewidth=0, height=0.55, zorder=3)
    axL.errorbar(rates, ypos, xerr=errs, fmt="none", ecolor="white",
                 elinewidth=1.8, capthick=1.8, capsize=ERRBAR["capsize"],
                 zorder=4)
    axL.errorbar(rates, ypos, xerr=errs, fmt="none", ecolor=BLUE, zorder=5,
                 **ERRBAR)
    # One value label per bar, all in the same place: clear of the upper interval
    # cap, rate above and count below, so no label can reach the bracket.
    for y, b, r, hi in zip(ypos, bars, rates, his):
        axL.text(hi + 0.05, y, f"{r*100:.1f}%\n{b[1]:,} of {b[2]:,}", va="center",
                 ha="left", fontsize=8.0, color=INK, linespacing=1.25)
    axL.set_yticks(ypos)
    axL.set_yticklabels([b[0] for b in bars], fontsize=8.0, linespacing=1.3)
    axL.set_xlim(0, 1.75)
    axL.set_ylim(-0.62, 2.62)
    axL.set_xticks([0, 0.5, 1.0])
    axL.set_xlabel("Positive rate")
    # The spines are cut back to the range they measure, so neither axis line
    # runs on into the annotation gutter.
    lean(axL, xbounds=(0, 1.0), ybounds=(0, 2))
    if SHOW_INFIGURE_TITLES: axL.set_title(f"Marginal, weak label against truth ({weak_rate/pharm_rate:.0f}x)", fontsize=8.5)
    # The bracket compares the two bars measured on the same 299-pair gold set.
    # It used to span the training-pool bar and the gold-set bar, which have
    # different denominators.
    gold_ratio = (pred_pos / pred_n) / (pharm_pos / pharm_n)
    x_bracket = 1.42
    axL.plot([x_bracket] * 2, [ypos[2], ypos[1]], color=MUTE, lw=0.6,
             solid_capstyle="butt", zorder=4)
    for y in (ypos[1], ypos[2]):
        axL.plot([x_bracket, x_bracket - 0.045], [y, y], color=MUTE, lw=0.6, zorder=4)
    axL.text(x_bracket + 0.05, (ypos[1] + ypos[2]) / 2, f"~{gold_ratio:.0f}x",
             va="center", ha="left", fontsize=8.0, color=MUTE)

    # An MCC is an estimate, not a quantity of anything, so it is drawn as a dot
    # with its interval, in the same grammar as every other estimate in this
    # thesis. Rows, not columns, so that the eye reads the four intervals
    # against one vertical chance line.
    ys = np.arange(len(order))[::-1]
    vals = [mcc[k]["mcc"] for k in order]
    ci_lo = [mcc[k]["ci95"][0] for k in order]
    ci_hi = [mcc[k]["ci95"][1] for k in order]
    err = [[v - lo for v, lo in zip(vals, ci_lo)], [hi - v for v, hi in zip(vals, ci_hi)]]
    axR.axvline(0, **CHANCE_STYLE)
    axR.errorbar(vals, ys, xerr=err, fmt="o", color=BLUE, zorder=3, **ERRBAR, **DOT)
    axR.set_yticks(ys)
    axR.set_yticklabels(labels, fontsize=8.0)
    axR.set_ylim(-0.62, 3.62)
    # Ticks and spine both run past the widest cap (the zero-shot interval
    # reaches 0.2484), so no interval ends beyond the scale it is read on.
    axR.set_xlim(-0.34, 0.34)
    axR.set_xticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    axR.set_xlabel("Matthews correlation, 95% CI")
    if SHOW_INFIGURE_TITLES: axR.set_title("MCC, 95% CI, n=253", fontsize=8.5)
    # Sits at the top of the reference line it names, clear of every interval.
    axR.text(0.0, 3.72, "chance", fontsize=8.0, color=MUTE, va="bottom",
             ha="center", clip_on=False)
    # ACCORD is not accented here. This panel's claim is that all four
    # intervals contain zero, which is a property of the column and not of any
    # one row, so nothing in it takes the accent.
    lean(axR, xbounds=(-0.3, 0.3))
    axR.spines["left"].set_visible(False)
    axR.tick_params(axis="y", length=0)

    for ax, tag in ((axL, "(a)"), (axR, "(b)")):
        p = ax.get_position()
        fig.text(0.004, p.y1 + 0.010, tag, ha="left", va="bottom",
                 fontsize=8.0, style="italic", color=INK)

    if SHOW_INFIGURE_TITLES: fig.suptitle("The number that determines the rest of the thesis", fontsize=9.0, y=0.97)
    if SHOW_INFIGURE_NOTES: fig.text(0.5, 0.02,
            "Left-panel middle bar (weak labeller gold-set marginal): 261/299, summed from per-group counts in\n"
            "exp1a_gold_error_rates_by_group.json. CORRECTIONS.md C28 flags an unresolved 17-row discrepancy against a\n"
            "raw-file count of 278/299 in a file not committed to this repo; the value shown is the one reconstructable here.",
            ha="center", fontsize=8.0, color=MUTE)
    fname = "fig-intro-headline.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    register(fname, "fig:intro-headline",
             [p_step01, p_exp1a, p_step2, p_intervals],
             "MCC panel uses stats/intervals.json E2 (all four CIs contain zero, per CORRECTIONS.md "
             "C36). The left-panel middle bar is the weak labeller marginal, 261 of 299 (gold.weaklabel_contradiction); ACCORD predicts contradiction for 278 of 299 (gold.accord_pred_contradiction, reconciliation_c28.json). Earlier notes conflated the two reading with "
             "the discrepancy noted on the figure.")


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
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    register(fname, "fig:funnel", [p_icd, p_era],
             "Simplified from the 5-stage spec to the 3 stages traceable to committed files; "
             "see in-figure footnote.")


# ============================================================================
# Figure 4 -- fig:exemplar  (ch4_candidates.tex)
# ============================================================================
def make_fig_exemplar():
    exemplar, p_exemplar = load("exemplar", "exemplar_correctness_recomputed.json")

    # Laid out on the same point grid as the other two schematics: one data unit
    # is one typographic point, so the two columns are identical by construction
    # and every row sits on one baseline. The canvas is shorter than it was by
    # exactly the two lines of summary sentence that have moved to the caption.
    W, H = FIGW, 2.58
    fig, ax = plt.subplots(figsize=(W, H))
    canvas(fig)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W * 72)
    ax.set_ylim(0, H * 72)
    ax.axis("off")

    # Two columns of abstract mention slots (no clinical text).
    col_x = {"Nursing note": 60.0, "Discharge summary": 190.0}
    box_w, box_h, pitch = 54.0, 20.0, 26.0
    rows_y = [124.0 - i * pitch for i in range(5)]
    for label, x in col_x.items():
        ax.text(x, 177.0, label, ha="center", va="center", fontsize=8.0,
                color=INK)

    nurse_tags = ["M1", "M2", "M3", "M4", "M5"]
    disch_tags = ["CONT", "CONT", "CONT", "CONT", "CESS"]
    for i, y in enumerate(rows_y):
        for col, tags in ((col_x["Nursing note"], nurse_tags),
                          (col_x["Discharge summary"], disch_tags)):
            # The one highlighted slot is a scarlet stroke and scarlet text, not
            # a filled block: the accent marks it without shouting.
            hit = (col == col_x["Discharge summary"] and i == 4)
            edge = SCARLET if hit else INK
            ax.add_patch(FancyBboxPatch(
                (col - box_w / 2, y - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0,rounding_size=2.0", mutation_aspect=1.0,
                linewidth=1.0 if hit else 0.6, edgecolor=edge, facecolor="white"))
            ax.text(col, y, tags[i], ha="center", va="center", fontsize=8.0,
                    color=edge)

    # The trigger arrow spans the two columns it relates, from the centre of
    # the M column to the centre of the CONT column, dropping a short stem at
    # each end onto the column it names. It used to float in the gutter between
    # them, touching neither, which left the reader to guess what was being
    # related to what.
    x_l, x_r = col_x["Nursing note"], col_x["Discharge summary"]
    y_arrow, y_stem = 146.0, 136.0
    for x in (x_l, x_r):
        ax.plot([x, x], [y_stem, y_arrow], color=INK, lw=0.6,
                solid_capstyle="butt", zorder=1)
    ax.annotate("", xy=(x_r, y_arrow), xytext=(x_l, y_arrow),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.6,
                                mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.text((x_l + x_r) / 2, y_arrow + 4.0,
            "trigger: sets conflict\n(quantifies over the set)", ha="center",
            va="bottom", fontsize=8.0, color=MUTE, linespacing=1.25)

    # Leaders: one weight, one length, attached to the right edge of the slot
    # they name, so neither crosses a box. Both notes are annotation, so both
    # are drawn in the secondary text tone with their arrows: the scarlet CESS
    # slot is the one accent on this canvas and carries the finding by itself.
    # A scarlet note beside a scarlet box put the accent on the canvas twice
    # and made the sentence compete with the thing it points at.
    x_tip = col_x["Discharge summary"] + box_w / 2 + 2
    x_note = x_tip + 26
    for y, note, colour in ((rows_y[0], ".iloc[0]\n(what the model,\nannotator and LLM saw)", MUTE),
                            (rows_y[4], "fix: action-matched\nand dose-adjacent\n(the CESS mention)", MUTE)):
        ax.annotate("", xy=(x_tip, y), xytext=(x_note - 3, y),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=0.6,
                                    mutation_scale=7, shrinkA=0, shrinkB=0))
        ax.text(x_note, y, note, va="center", ha="left", fontsize=8.0,
                color=colour, linespacing=1.25)

    # The measured rate is gone from the canvas and lives in the caption. This
    # panel is a schematic of one candidate: five slots, one of them wrong.
    # Welding a population rate to a picture of a single case invited the
    # reader to take the picture as the estimate of the rate, which it is not,
    # and the rate itself is reported two ways in the chapter.
    fname = "fig-exemplar.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    register(fname, "fig:exemplar", [p_exemplar],
             "Schematic; slots are abstract labels (M1..M5, CONT/CESS action-tag codes), no "
             "clinical text and, since round two, no number: the wrong-exemplar rate it used "
             "to carry is a population rate and this panel draws a single candidate, so the "
             "rate is stated in the chapter caption instead.")


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
    lean(ax, xbounds=(0.2, 0.8))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if SHOW_INFIGURE_TITLES: ax.set_title("ACCORD's own ranking performance does not exceed chance\nwithin any single rule", fontsize=8.3)
    if SHOW_INFIGURE_NOTES: fig.text(0.02, 0.02,
             "Deviation from spec: per-rule ACCORD *precision* (as in the written spec) is not in any\n"
             "committed result file; per-rule AUROC is (defect_free_strata.json) and is shown instead.\n"
             "Individual-rule CIs are not computable here (per-pair scores not committed); only the\n"
             "pooled rules-3+4 CI (stats/intervals.json E3) is available and is shown as a reference band.",
             fontsize=8.0, color=MUTE)
    fname = "fig-rule-precision.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
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
    step01, p_step01 = load("diag", "step0_1_join_and_calibration.json")
    xtab = step01["weak_confidence_is_degenerate"]["crosstab_label_x_confidence"]
    weak_pos, weak_neg = xtab["0.95"]["contradiction"], xtab["1.0"]["not_contradiction"]
    weak_n = weak_pos + weak_neg

    step2, p_step2 = load("diag", "step2_signal_check.json")
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
    lean(ax, xbounds=(0, 1.0), ybounds=(0, 1))
    ratio = (weak_pos / weak_n) / (gold_pos / gold_n)
    ax.plot([1.05, 1.05], [0, 1], color=MUTE, lw=0.6, clip_on=False)
    ax.text(1.09, 0.5, f"~{ratio:.0f}x", va="center", fontsize=8.0, color=MUTE,
            clip_on=False)
    implied_wrong = int(np.floor(weak_pos * (1 - gold_pos / gold_n)))
    if SHOW_INFIGURE_TITLES: ax.set_title(f"Marginals are close to complementary, not noisy copies\n"
                 f"implied floor: ≥{implied_wrong:,} of {weak_pos:,} positive weak labels wrong "
                 f"at the measured {gold_pos/gold_n*100:.1f}% base rate", fontsize=8.0)
    fname = "fig-prevalence.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    register(fname, "fig:prevalence", [p_step01, p_step2],
             "Weak-label counts from diag/step0_1 weak_confidence crosstab; gold counts "
             "(12 contradiction / 241 not / 46 ambiguous) reconstructed from diag/step2_signal_check "
             "ambiguous-included vs ambiguous-excluded base rates.")


# ============================================================================
# Figure 7 -- fig:tracking  (ch5_weak_supervision.tex)
# ============================================================================
def make_fig_tracking():
    step01, p_step01 = load("diag", "step0_1_join_and_calibration.json")
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
    plot_series(truth, "^", ":", SCARLET, "pharmacist truth", 0.10)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"Rule {r}\n(n={ns[r]})" for r in rules], fontsize=8.0,
                       linespacing=1.25)
    ax.set_ylabel("Positive rate")
    ax.set_ylim(-0.06, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.35, 3.05)
    lean(ax, xbounds=(0, 3), ybounds=(0, 1))
    if SHOW_INFIGURE_TITLES: ax.set_title("The student tracks the teacher's marginal, not the truth's",
                 fontsize=8.5)
    fname = "fig-tracking.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    register(fname, "fig:tracking", [p_step01],
             "Wilson score 95% CIs added per point (n from n_gold per rule: 134/31/79/55); "
             "not in the original spec, added per the CI non-negotiable.")


# ============================================================================
# Figure 8 -- fig:met-exchange  (ch8_metrics.tex)
# ============================================================================
def make_fig_met_exchange():
    baselines, p_baselines = load("diag", "step3_baselines.json")
    systems = {
        "ACCORD": baselines["accord_binary_at_0.5_253core"],
        "GPT-4o zero-shot": baselines["gpt4o_zeroshot"],
        "GPT-4o few-shot": baselines["gpt4o_fewshot"],
        "Llama 3 8B": baselines["llama_baseline"],
    }
    ref = systems["ACCORD"]
    TP0, TN0, FP0, FN0 = ref["tp"], ref["tn"], ref["fp"], ref["fn"]
    P = TP0 + FN0
    N = TN0 + FP0
    mcc0 = mcc_from_counts(TP0, TN0, FP0, FN0)

    fig, ax = plt.subplots(figsize=(FIGW, 3.10))
    canvas(fig)
    fig.subplots_adjust(left=0.125, right=0.985, top=0.955, bottom=0.155)
    # The three indifference lines are not measurements. They are what each
    # statistic would accept, computed from the class counts, and they are the
    # grid this figure reads the four systems against. So they are drawn in the
    # reference grammar the rest of the set uses for a null or a chance line:
    # grey hairlines, labelled at the line in the secondary text tone. Blue is
    # returned to the thing that was actually measured, which is the systems.
    d = np.linspace(0, P, 200)
    ax.plot(d, d, color=GREY, linewidth=0.8, zorder=3)
    slope_ba = N / P
    ax.plot(d, d * slope_ba, color=GREY, linewidth=0.8, linestyle=(0, (5, 2)), zorder=3)

    # MCC indifference curve through ACCORD's own operating point, traced
    # numerically: for each delta (forgone TPs), solve for the TN that holds
    # MCC constant at mcc0.
    # Sweep to TP0, not to P. A system cannot forgo more true positives than it
    # found, and ACCORD found 11 of the 12. Running to P made tp negative at the
    # last step, which put a negative product under the sqrt in mcc_from_counts,
    # returned NaN, and silently truncated the curve while printing a warning.
    deltas = np.linspace(0, TP0, 60)
    dtn_curve = []
    for delta in deltas:
        tp = TP0 - delta
        fn = P - tp

        def f(tn):
            fp = N - tn
            return mcc_from_counts(tp, tn, fp, fn) - mcc0

        try:
            tn_sol = brentq(f, 0, N)
            dtn_curve.append(tn_sol - TN0)
        except ValueError:
            dtn_curve.append(np.nan)
    # The MCC line runs under the balanced-accuracy line for nearly its whole
    # length, so it is drawn as a band wide enough to show on both sides of it.
    ax.plot(deltas, dtn_curve, color=PALE, linewidth=3.0, zorder=2)

    # Direct labels at the right end of each line, in the tone of the line they
    # name. No legend: three keys in a box is three facts stated twice.
    ax.text(P - 0.2, N + 8, f"balanced accuracy\n(slope {slope_ba:.2f})", ha="right",
            va="bottom", fontsize=8.0, color=MUTE, linespacing=1.25, zorder=6)
    # The MCC band is labelled at its own right edge, where the band stops and
    # the balanced-accuracy line carries on: the one place on the canvas the
    # two are visibly different things. The label itself sits in the empty
    # wedge below both diagonals, because there is no room beside the band that
    # the balanced-accuracy line does not already cross, and it is reached by a
    # tail in the band's own tone.
    ax.plot([deltas[-1], deltas[-1] + 0.9], [dtn_curve[-1] - 4, 134],
            color=PALE, lw=1.2, solid_capstyle="butt", zorder=2)
    ax.text(P + 0.5, 130, f"MCC indifference through\nACCORD ({mcc0:.3f})",
            ha="right", va="top", fontsize=8.0, color=MUTE, linespacing=1.25, zorder=6)
    ax.text(P - 0.2, P + 6, "PABAK (slope 1)", ha="right", va="bottom", fontsize=8.0,
            color=MUTE, zorder=6)

    markers = {"ACCORD": "o", "GPT-4o zero-shot": "s", "GPT-4o few-shot": "^", "Llama 3 8B": "D"}
    # Every label is placed away from the two near-coincident diagonals: the
    # GPT points are labelled up and to the left of the lines, ACCORD to the left
    # of the origin, Llama directly below its own glyph, which is the only
    # placement that clears the crowd of marks and axis lines around the origin.
    label_offset = {"ACCORD": (-8, 8), "GPT-4o zero-shot": (-8, 5), "GPT-4o few-shot": (-8, 5),
                     "Llama 3 8B": (0, -9)}
    label_align = {"ACCORD": "right", "GPT-4o zero-shot": "right", "GPT-4o few-shot": "right",
                   "Llama 3 8B": "center"}
    label_valign = {"ACCORD": "center", "GPT-4o zero-shot": "center",
                    "GPT-4o few-shot": "center", "Llama 3 8B": "top"}
    for name, s in systems.items():
        delta = TP0 - s["tp"]
        dtn = s["tn"] - TN0
        # ACCORD is the claim element here: this whole figure is about where
        # ACCORD sits against exchange rates it never chose, so it takes the
        # accent. The other three are the measured comparison, so they carry
        # the primary data colour.
        this_study = name == "ACCORD"
        ax.scatter([delta], [dtn], marker=markers[name], s=34,
                   edgecolor=SCARLET if this_study else BLUE,
                   facecolor=SCARLET if this_study else "white",
                   linewidth=0.8, zorder=5)
        ax.annotate(name, (delta, dtn), textcoords="offset points", xytext=label_offset[name],
                    ha=label_align[name], va=label_valign[name], fontsize=8.0, zorder=6,
                    color=SCARLET if this_study else BLUE)

    # Origin lines bounded to the range they frame, so neither runs on into the
    # margin the Llama label now sits in.
    ax.plot([-2.3, P + 0.6], [0, 0], color=GREY, linewidth=0.6, zorder=0)
    ax.plot([0, 0], [0, N], color=GREY, linewidth=0.6, zorder=0)
    ax.set_xlabel(r"$\delta$ = forgone true positives, relative to ACCORD")
    ax.set_ylabel(r"$\Delta$TN, relative to ACCORD")
    ax.set_xlim(-2.3, P + 0.6)
    ax.set_ylim(-46, N + 40)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_yticks([0, 50, 100, 150, 200])
    lean(ax, xbounds=(0, 12), ybounds=(0, 200))
    if SHOW_INFIGURE_TITLES: ax.set_title(f"What each statistic trades (P={P}, N={N}, n={P+N})", fontsize=8.5)
    fname = "fig-met-exchange.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    register(fname, "fig:met-exchange", [p_baselines],
             "Indifference lines are closed-form (PABAK, balanced accuracy) or numerically "
             "root-solved (MCC, scipy.optimize.brentq) from the confusion-matrix class counts "
             "in diag/step3_baselines.json; not fabricated curve shapes.")


# ============================================================================
# Figure 9 -- fig:met-sharpened  (ch8_metrics.tex)
# ============================================================================
def make_fig_met_sharpened():
    baselines, p_baselines = load("diag", "step3_baselines.json")
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
    lean(axL, xbounds=(0, 12), ybounds=(0, 3))
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
    lean(axR, xbounds=(-0.2, 0.2), ybounds=(0, 3))

    for ax, tag in ((axL, "(a)"), (axR, "(b)")):
        pos = ax.get_position()
        fig.text(0.004, pos.y1 + 0.010, tag, ha="left", va="bottom",
                 fontsize=8.0, style="italic", color=INK)
    if SHOW_INFIGURE_TITLES: axR.set_title("Matthews correlation, 95% CI", fontsize=8.5)

    if SHOW_INFIGURE_TITLES: fig.suptitle("The high-recall systems predict positive almost everywhere;\n"
                 "no MCC interval excludes chance", fontsize=8.0, y=0.99)
    fname = "fig-met-sharpened.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
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
    for y, (label, auroc, ci) in zip(ys, rows):
        ax.errorbar([auroc], [y], xerr=[[auroc - ci[0]], [ci[1] - auroc]], fmt="o",
                    color=BLUE, zorder=3, **ERRBAR, **DOT)
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
    lean(ax, xbounds=(0.2, 0.9))
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
    fig.savefig(os.path.join(FIGDIR, fname))
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
    # One clean timeline. Events are tick-marked on a single axis, their labels
    # alternate above and below on thin leaders, and the two moments the
    # argument turns on carry the only accents: the point a detector could have
    # fired, and the point this one actually does. Drawn on the same point grid
    # as the other two schematics, one data unit to one typographic point.
    W, H = FIGW, 1.84
    fig, ax = plt.subplots(figsize=(W, H))
    canvas(fig)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W * 72)
    ax.set_ylim(0, H * 72)
    ax.axis("off")

    line_y = 72.0
    band_lo, band_hi = line_y - 11.0, line_y + 11.0
    # (x, label, above?, accent colour or None)
    events = [
        (48.0, "last nursing note\n(CHARTTIME present)", True, None),
        (150.0, "discharge order\nproposed detection point", False, BLUE),
        (196.0, "DISCHTIME\n(patient leaves)", True, None),
        (308.0, "discharge summary finalised\n(CHARTDATE only, no CHARTTIME)\n"
                "the system's trigger point", False, SCARLET),
    ]

    # The span in which nothing can be done: a light band, labelled directly
    # above itself, sitting behind the axis rather than on top of it.
    ax.add_patch(mpatches.Rectangle((196.0, band_lo), 308.0 - 196.0, band_hi - band_lo,
                                    facecolor=PALE, edgecolor="none", alpha=0.55,
                                    zorder=0))
    ax.text((196.0 + 308.0) / 2, band_hi + 3, "no intervention possible",
            ha="center", va="bottom", fontsize=8.0, style="italic", color=MUTE)

    ax.annotate("", xy=(368.0, line_y), xytext=(22.0, line_y),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.6,
                                mutation_scale=7, shrinkA=0, shrinkB=0))
    for x, label, above, accent in events:
        colour = accent or INK
        ax.plot([x, x], [line_y - 3.5, line_y + 3.5], color=colour, lw=0.9,
                solid_capstyle="butt", zorder=4)
        ax.plot([x], [line_y], "o", color=colour, markersize=3.2,
                markeredgewidth=0, zorder=5)
        if above:
            ax.plot([x, x], [line_y + 5, 100.0], color=GREY, lw=0.5, zorder=2)
            ax.text(x, 103.0, label, ha="center", va="bottom", fontsize=8.0,
                    color=colour, linespacing=1.3)
        else:
            ax.plot([x, x], [line_y - 5, 49.0], color=GREY, lw=0.5, zorder=2)
            ax.text(x, 46.0, label, ha="center", va="top", fontsize=8.0,
                    color=colour, linespacing=1.3)

    if SHOW_INFIGURE_TITLES: ax.text(W * 36, H * 72 - 4, "The trigger fires after the patient has left",
                 ha="center", va="top", fontsize=9.0)
    if SHOW_INFIGURE_NOTES: fig.text(0.5, 0.02,
             "Schematic of MIMIC-III field timing (CORRECTIONS.md C6: CHARTTIME is null for every\n"
             "discharge summary). No numeric estimate is shown; not sourced from a results JSON.",
             ha="center", fontsize=8.0, color=MUTE)
    fname = "fig-disc-actionability.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    register(fname, "fig:disc-actionability", ["accord/CORRECTIONS.md (C6)"],
             "Structural/schema diagram, not derived from a numeric result file; documents a "
             "MIMIC-III field-availability fact. No CI applies (no estimate is plotted).")


# ============================================================================
# Figure 12 -- fig:cohort-flow  (ch3_methodology.tex)
#   The CONSORT-style path from the source database to the two evaluated sets.
#   Nine stages, three exclusions. Every count is read from a committed file:
#   the ICD and length-of-stay counts from stats/cohort_icd_counts.json, the
#   candidate count from generality/transfer_results_correct.json, the gold
#   counts from diag/step2_signal_check.json, and the four stages whose own
#   upstream artifact is quarantined off this repository (note-role coverage,
#   pair selection, mention extraction, the drawn annotation sample) from the
#   thesis register, which is itself committed under results/accord.
#
#   Two stages the funnel would normally carry are skipped for want of a
#   committed count and are named in the register note: the exclusion between
#   the length-of-stay filter and the both-roles stage, and the exclusion
#   between the both-roles stage and the selected pairs. Both differences are
#   arithmetic on two committed counts, but neither is itself a recorded
#   exclusion count, and a funnel that invents its own subtractions is the
#   thing this set exists to avoid.
# ============================================================================
def make_fig_cohort_flow():
    icd, p_icd = load("stats", "cohort_icd_counts.json")
    hf = icd["heart_failure"]

    transfer, p_transfer = load("generality", "transfer_results_correct.json")
    n_candidates = transfer["hf_reference"]["n_candidates"]

    step2, p_step2 = load("diag", "step2_signal_check.json")
    d299 = step2["contra_prob_vs_pharm_label__ambiguous_as_negative_n299"]
    d253 = step2["contra_prob_vs_pharm_label__ambiguous_excluded_n253"]
    d_amb = step2["contra_prob_vs_pharm_label__ambiguous_as_positive_n299"]
    # The ambiguous count is not stored on its own line. It is the difference
    # between the two positive counts recorded in the same file on the same 299
    # rows, ambiguous counted positive against ambiguous counted negative, which
    # is the same subtraction make_fig_prevalence already performs.
    n_ambiguous = d_amb["n_pos"] - d299["n_pos"]

    facts, p_facts = load("VERIFIED_FACTS.json")
    def fact(key):
        return facts[key]["value"]

    n_both_roles = fact("cohort_chain.has_both_roles")
    n_pairs = fact("cohort_chain.valid_admissions_both_selected")
    n_note_rows = fact("cohort_chain.total_note_pair_rows")
    n_mentions = fact("extraction.medication_extraction.total_mentions")
    n_classified = fact("extraction.medication_extraction.classified_mentions")
    n_unclassified = fact("extraction.medication_extraction.unclassified_derived")
    n_drawn = fact("extraction.annotation_sample.total_sampled")

    # Same point grid as the other schematics in this set: the axes span exactly
    # the figure and one data unit is one typographic point, so the nine boxes
    # are identical by construction and every row sits on one baseline.
    W, H = FIGW, 4.62
    fig, ax = plt.subplots(figsize=(W, H))
    canvas(fig)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W * 72)
    ax.set_ylim(0, H * 72)
    ax.axis("off")

    # (stage name, count line, exclusion below this stage or None)
    stages = [
        ("Heart failure admissions, ICD-9 prefix",
         f"{hf['pre_los']:,} admissions",
         f"{hf['removed']:,} admissions with no\npositive length of stay"),
        ("After the length of stay filter",
         f"{hf['post_los']:,} admissions", None),
        ("Admissions carrying both note roles",
         f"{n_both_roles:,} admissions", None),
        ("Note pair selected, one note per role",
         f"{n_pairs:,} pairs, {n_note_rows:,} note rows", None),
        ("Medication mentions extracted",
         f"{n_mentions:,} mentions",
         f"{n_unclassified:,} mentions with no\naction class assigned"),
        ("Mentions carrying an action class",
         f"{n_classified:,} mentions", None),
        ("Candidates after the temporal prefilter",
         f"{n_candidates:,} candidates", None),
        ("Pairs annotated by the pharmacist",
         f"{d299['n']} of {n_drawn} drawn",
         f"{n_ambiguous} pairs coded ambiguous"),
        ("Non-ambiguous evaluation core",
         f"{d253['n']} pairs, {d253['n_pos']} confirmed contradictions", None),
    ]

    box_x, box_w, box_h, pitch = 6.0, 194.0, 26.0, 36.0
    x_excl = 212.0
    y_top = H * 72 - 9.0
    tops = [y_top - i * pitch for i in range(len(stages))]
    x_stem = box_x + box_w / 2

    for i, (t, (name, count, excl)) in enumerate(zip(tops, stages)):
        # The last box is the one mark that carries what this figure asserts:
        # the two evaluated sets the whole path ends at, and the twelve
        # confirmed contradictions inside them. Scarlet stroke and scarlet
        # text, the same treatment the exemplar schematic gives its one wrong
        # slot, rather than a filled block.
        last = i == len(stages) - 1
        edge = SCARLET if last else INK
        ax.add_patch(FancyBboxPatch(
            (box_x, t - box_h), box_w, box_h,
            boxstyle="round,pad=0,rounding_size=2.0", mutation_aspect=1.0,
            linewidth=1.0 if last else 0.6, edgecolor=edge, facecolor="white"))
        ax.text(box_x + box_w / 2, t - box_h / 2, f"{name}\n{count}",
                ha="center", va="center", fontsize=8.0, color=edge,
                linespacing=1.35)
        if not last:
            ax.annotate("", xy=(x_stem, t - pitch),
                        xytext=(x_stem, t - box_h),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.6,
                                        mutation_scale=7, shrinkA=0, shrinkB=0))
        if excl is not None:
            # A CONSORT exclusion leaves the trunk in the gap between two
            # stages and runs right, so it reads as taken out between them
            # rather than as a second thing the stage produced.
            y_gap = t - box_h - (pitch - box_h) / 2
            ax.annotate("", xy=(x_excl - 4.0, y_gap), xytext=(x_stem, y_gap),
                        arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=0.6,
                                        mutation_scale=7, shrinkA=0, shrinkB=0))
            ax.text(x_excl, y_gap, excl, ha="left", va="center", fontsize=8.0,
                    color=MUTE, linespacing=1.25)

    ax.text(x_excl, y_top + 1.0, "excluded", ha="left", va="top", fontsize=8.0,
            color=MUTE)

    fname = "fig-cohort-flow.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    register(fname, "fig:cohort-flow",
             [p_icd, p_transfer, p_step2, p_facts],
             "CONSORT-style path from the source database to the two evaluated sets. Nine "
             "stages, three exclusions, every count read from a committed file. Two exclusion "
             "branches the diagram would otherwise carry are omitted because no committed file "
             "records them as counts: the admissions dropped between the length-of-stay filter "
             "and the both-roles requirement, and those dropped between the both-roles "
             "requirement and the selected pairs. Both are differences of two committed counts "
             "and neither is a recorded exclusion, so neither is drawn. The ambiguous count is "
             "the difference between the two positive counts recorded on the same 299 rows in "
             "the signal-check file, the same subtraction the prevalence figure performs. Four "
             "stages read the thesis register rather than a primary artifact, because their own "
             "upstream summaries live outside this repository under the data agreement.")


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
        (n_strict, SCARLET, SCARLET,
         f"{n_strict} papers ({n_strict / n_total * 100:.1f}%) report the check "
         "under the strict reading"),
        (n_permissive_only, PALE, MUTE,
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
        leader = SCARLET if fill is SCARLET else GREY
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
    lean(ax, xbounds=(0, n_total))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fname = "fig-survey-audit.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
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
# results/accord/generality/, and make_figures_extra.py builds the figure from
# it. The one number that entry was right about is the held-out 45.2%: 0.4515
# appears in no result file anywhere, so that reference line was removed from
# the figure rather than sourced.


def main():
    make_fig_pipeline_status()
    make_fig_intro_headline()
    make_fig_funnel()
    make_fig_exemplar()
    # make_fig_rule_precision() removed from the thesis per committee review;
    # the per-rule table carries this. The drawing function is left in place
    # above so the figure can be rebuilt if it is ever wanted again, but
    # nothing binds it and nothing calls it.
    make_fig_prevalence()
    make_fig_tracking()
    make_fig_met_exchange()
    make_fig_met_sharpened()
    make_fig_disc_strata()
    make_fig_disc_actionability()
    make_fig_cohort_flow()
    make_fig_survey_audit()

    manifest_path = os.path.join(FIGDIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"built": MANIFEST, "skipped": SKIPPED}, f, indent=2)

    print(f"Built {len(MANIFEST)} figures, skipped {len(SKIPPED)}.")
    for m in MANIFEST:
        print(f"  BUILT  {m['label']:28s} -> {m['file']}")
    for s in SKIPPED:
        print(f"  SKIP   {s['label']:28s} ({s['chapter']})")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
