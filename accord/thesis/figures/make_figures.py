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
    COBALT, AZURE, EMERALD, AMBER, PLUM, SCARLET,
    COBALT_DEEP, COBALT_CORE, COBALT_TINT, COBALT_WASH, COBALT_BARE,
    AZURE_DEEP, AZURE_CORE, AZURE_TINT, AZURE_WASH, AZURE_BARE,
    EMERALD_DEEP, EMERALD_CORE, EMERALD_TINT, EMERALD_WASH, EMERALD_BARE,
    AMBER_DEEP, AMBER_CORE, AMBER_TINT, AMBER_WASH, AMBER_BARE,
    PLUM_DEEP, PLUM_CORE, PLUM_TINT, PLUM_WASH, PLUM_BARE,
    SCARLET_DEEP, SCARLET_CORE, SCARLET_TINT, SCARLET_WASH, SCARLET_BARE,
    PAPER, INK, GRAPHITE, MUTE, SLATE, MIST, HAIR, PLATE_EDGE,
    ENTITY_HUE, NON_ENTITY_MEASUREMENT,
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
# Figure 1 -- fig:pipeline-status  (ch1_introduction.tex)
# ============================================================================
# v4 REFRAME (second-round review). The v3 figure was a verdict table: a
# coloured Works/Unvalidated/Fails/Falsified chip per stage, with the number
# and the diagnostic both reduced to small italic text beside it. Chapter 1
# Section 1.2 no longer argues "most stages failed"; it argues that four of
# those failures were METRIC-BLIND -- each produced a number that would have
# passed peer review, because no standard reported metric was sensitive to
# the defect behind it, and each has a one-line check that is. The figure has
# to carry that argument, not the verdict it used to carry. Three changes:
#   (1) the verdict chip is gone. A stage's status is no longer a coloured
#       box; it is the content of the row. The row's pale tint (a WASH, not a
#       SOLID chip) is the only colour-coded state left, and it is reserved
#       for a glance, not a claim.
#   (2) the "reported number" -- what looked fine -- is now set in plain INK
#       beside the stage, each one ending in the word "unflagged", stated
#       once per row rather than as a separate sentence, because the four
#       rows read as a set and the repetition is the point.
#   (3) the check is the chip now: a solid INK box naming the one-line test,
#       with the number it actually catches given underneath in SCARLET_DEEP
#       -- the one semantic that colour carries in this set, the exact mark
#       where the pipeline is broken, now attached to the check's finding
#       rather than to a verdict word.
#
# v5 (third-round review, six verified defects). v4's INK-box-on-the-check
# design got four things wrong at once:
#   (1) CONTRAST. The finding line sat in SCARLET_DEEP directly on the row's
#       SCARLET_TINT wash. Measured under WCAG's own formula that is 4.93:1,
#       which clears the 4.5:1 AA floor for this pairing on paper, but the
#       reviewer's tooling read it closer to 4:1 and flagged it as the
#       figure's most serious defect twice running -- on a schematic where
#       small italic type sits directly on a tinted fill, a nominal pass with
#       no margin is not a fix. The finding no longer sits as coloured text
#       on a tint at all: see (3).
#   (2) PALETTE. Three washes (EMERALD_TINT row 1, MIST row 2, SCARLET_TINT
#       rows 3-6) was the original traffic-light problem back, and worse in
#       grayscale: TINT stops are pale by construction, so any two of them
#       convert to nearly the same light grey (measured below) regardless of
#       hue. A background wash cannot carry this distinction at the TINT
#       stop, in any hue. It is now binary and it is not carried by the wash
#       alone: FAULT rows keep a SCARLET_TINT ground (state is data on a
#       schematic, and SCARLET's one semantic is exactly "the failure this
#       thesis is reporting"); PASS/NONE rows carry no wash at all, which is
#       maximally separable from a tint in luminance terms rather than
#       marginally so. The real, greyscale-and-colour-blind-safe separator is
#       (3): a solid chip appears on a fault row and nowhere else.
#   (3) WEIGHT. The solid chip -- the highest-weight mark on the canvas --
#       used to wrap the check (the diagnostic procedure, the least
#       interesting column) while the finding it catches sat underneath as
#       small plain text. That is backwards: the finding is the figure's
#       point. The chip now wraps the finding, in SCARLET (the fault
#       semantic, and F2 SOLID CHIP already buys >=7.7:1 for PAPER text on
#       SCARLET_DEEP with no new colour needed); the check is demoted to
#       plain italic text, the same voice every non-fault row's check column
#       already used, because it is genuinely the supporting column now. It
#       is GRAPHITE rather than MUTE on a FAULT row specifically, because it
#       now sits directly on the row's own SCARLET_TINT wash rather than
#       inside an opaque chip, and MUTE-on-SCARLET_TINT measures 3.98:1,
#       under the 4.5:1 AA floor for 8.5pt text; GRAPHITE-on-SCARLET_TINT
#       measures 7.93:1, and GRAPHITE is already this system's structural
#       measurement tone, not a new colour.
#   (4) Three correctness/consistency fixes that do not touch layout: row 2's
#       reported and check columns said the same thing twice (dropped the
#       check column's copy, since the reported column already carries "why
#       there is nothing to report" for every PASS/NONE row); row 1's
#       reported column no longer claims "no defect found" (Sec.3.2.2 records
#       that the LOS filter was computed and never applied, so the note joins
#       ran unfiltered -- "counts reconcile" is what the section supports,
#       and the check column is reworded to name the reconciliation check
#       rather than restate that same conclusion); and the rule-generation
#       finding is now printed as the decimal, "0 to 0.4745", in the same
#       plain digits as the pharmacist base rate two lines above it in the
#       same row, rather than as a percentage that truncates instead of
#       rounds.
# Every number below is read from the same four files the v3 figure read;
# nothing new is loaded and nothing is typed in.
def make_fig_pipeline_status():
    exemplar, p_exemplar = load("exemplar", "exemplar_correctness_recomputed.json")
    # The former reference rate was supplied by a task description, never measured
    # (REVISION_LOG R8). Use the measured wrong rate on the resolved candidates.
    _ex = exemplar["rule_1_2_pooled"]
    defect_rate = _ex["n_wrong"] / _ex["n_resolved"]

    # v7 (fourth-round review, external reviewer). step0_1_join_and_calibration.json
    # used to be loaded here for weak_pos_rate, and step2_signal_check.json's
    # ambiguous-excluded base rate for pharm_rate -- neither is read anywhere in
    # this function, only assigned once and never referenced again. That dead
    # pair put a source into this figure's own manifest entry that contributes
    # nothing to the canvas, which is exactly the class of provenance rot the
    # thesis is about; it should not be in the thesis's own manifest. Both loads
    # and both dead variables are gone. step2 itself stays: pharm_rate_all below
    # IS read, from the same file.
    step2, p_step2 = load("diag", "step2_signal_check.json")
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
    gold_weak_rate = gold_weak_pos / gold_n

    decontam, p_decontam = load("decontam", "corrected_gold_metrics.json")
    auroc = decontam["aggregate"]["ensemble_auroc"]
    ci = decontam["aggregate"]["ensemble_ci95"]

    # v7 REDESIGN (fourth-round review, external reviewer, rejected on all three
    # prior rounds). v6 read as "a table with a headline": six uniform row
    # bands, three of them a wash whose only job was fault/non-fault, and
    # nothing on the canvas said that the stage 3 exemplar defect is the shared
    # corrupted input the three stages after it then each independently
    # mishandled -- the propagation claim the chapter actually rests on. Three
    # changes:
    #   (1) rows 1 and 2 (Cohort construction, Medication extraction) carried no
    #       number and no finding; they are dropped from the row list entirely
    #       and replaced by one grey stub line under the headline.
    #   (2) a spine at x=22 runs the row column: a clean grey segment for
    #       stages 1-2, a scarlet origin dot on row 3 (the defect enters here),
    #       a heavier scarlet carrier down to row 6, and a delivery arrow into
    #       rows 4/5/6 only -- row 3 gets no arrow, because it is the source,
    #       not a recipient. Vertical position now reads as causal order,
    #       which it did not before. This is the whole redesign.
    #   (3) the row washes are gone. With rows 1-2 gone, all four remaining
    #       rows are faults, so a wash on all four encoded nothing, added ink,
    #       and reintroduced the MUTE-on-SCARLET_TINT contrast problem already
    #       fixed once (v5). The finding chip is the only colour-coded mark
    #       left, and it is sized to its own text rather than to the column.
    W, H = FIGW, 3.39
    fig, ax = plt.subplots(figsize=(W, H))
    canvas(fig)
    plate(fig)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W * 72)
    ax.set_ylim(0, H * 72)
    ax.axis("off")
    # A renderer has to exist before any text's get_window_extent() call below
    # can measure anything; draw() once, on the still-empty axis, to get one.
    fig.canvas.draw()

    # Four rows only -- stages 3 through 6, the four that hid a failure behind
    # an unflagged number. The stage numeral keeps the six-stage pipeline's own
    # numbering (3, 4, 5, 6), not a renumbering to 1-4, so it still names which
    # stage this is in the same pipeline the stub line below summarizes stages
    # 1-2 of.
    rows = [
        (3, "Rule-based candidate\ngeneration",
         f"{pharm_rate_all:.4f} reported\nprecision",
         "Action-set match vs position index",
         # An upper bound, not a point estimate: only 137 of 165 Rule 1/2
         # candidates were resolved. Four chapters read it this way; the
         # figure must not contradict them. Printed as the decimal (0.4745),
         # not the percentage: 0.4744525... is 47.45% truncated to two
         # places, which rounds to 47.5%, not the 47.4% a direct percentage
         # format would print. The decimal removes the rounding question and
         # matches the base rate two lines above it in this same row.
         f"0 to {defect_rate:.4f} wrong"),
        (4, "LLM weak supervision",
         # Rewrapped (v7): the old "0.8729 positive on\ngold set" put 76.5pt
         # of "0.8729 positive on" against an 80pt column, one of the
         # tightest margins on the canvas. "0.8729 positive\non gold set"
         # measures 55.6pt on its first line instead.
         f"{gold_weak_rate:.4f} positive\non gold set",
         "Weak-label marginal vs gold marginal",
         # Also kept as printed: 21.768... rounded with the same .0f used at
         # fig:intro-headline for the identical ratio, so the two figures
         # agree; no chapter states this multiplier to four decimals for
         # that rounding to conflict with.
         f"{ratio:.0f}x mismatch"),
        (5, "Detector training",
         "Three ablation\nclaims held",
         r"$\|\Delta\theta\|$ from init, OOD transfer $F_1$",
         # No committed numeric value exists for this stage -- the finding
         # is categorical, not a number, so it is reported in the thesis's
         # own words for it: a component "whose credited mechanism is
         # reproduced by a deliberately meaningless substitute has been
         # falsified" (Ch.5, "Architecture: reversed, and positively";
         # Ch.4 Sec."Architecture" audits the same two components).
         "Credited mechanisms falsified"),
        (6, "Evaluation",
         f"AUROC {auroc:.4f}\nalone",
         "Bootstrap CI on AUROC before write-up",
         # Not a percentage and not borderline (0.3210 and 0.7034 each round
         # to their printed 2dp without ambiguity).
         f"[{ci[0]:.2f}, {ci[1]:.2f}]"),
    ]
    # Vertical grid, points from the bottom. y_top=172.0, band_h=38.5,
    # gutter=4.0, floor=6.0 solve to these four mids; written as a literal
    # dict rather than re-derived, because the numeral (3, 4, 5, 6) is the row
    # key everywhere below, not a 0-based loop index.
    row_mid = {3: 152.75, 4: 110.25, 5: 67.75, 6: 25.25}

    # --- The spine: the load-bearing element, and the whole of the redesign.
    # Vertical position now reads as causal order because the rail changes
    # tone AND weight at the exact point the defect enters, and only rows 4-6
    # -- the stages that consumed it -- get an arrowhead; row 3 alone has
    # none, and that asymmetry IS the claim. GRAPHITE/thin above the origin is
    # stages 1-2 (clean, unrelated to the defect); SCARLET_CORE/thick from the
    # origin down is "this carries the stage 3 defect".
    #
    # OVERCLAIM GUARD: this rail asserts SHARED CORRUPTED INPUT into rows
    # 4-6, not that rows 4-6's numbers follow causally from row 3's. The 22x
    # mismatch at row 4 is a prevalence and specificity effect, not an effect
    # of the exemplar defect, and nothing drawn here says or implies
    # otherwise -- the rail only ever claims the same wrong exemplar reached
    # all three, which is the carrier label's own words.
    ax.plot([22, 22], [202.0, row_mid[3]], color=GRAPHITE, lw=1.2,
            solid_capstyle="butt", zorder=3)
    ax.add_patch(mpatches.Circle((22, 202.0), 2.5, facecolor=GRAPHITE,
                                  edgecolor="none", zorder=3))
    ax.add_patch(mpatches.Circle((22, row_mid[3]), 4.0, facecolor=SCARLET_DEEP,
                                  edgecolor="none", zorder=6))
    ax.plot([22, 22], [row_mid[3], row_mid[6]], color=SCARLET_CORE, lw=2.6,
            zorder=4)
    for n in (4, 5, 6):
        arrow(ax, (22.0, row_mid[n]), (46.0, row_mid[n]), color=SCARLET_CORE,
              lw=2.0, mutation_scale=11)
    ax.text(11.0, 89.0, "the same wrong exemplar", rotation=90, ha="center",
            va="center", fontsize=T_NOTE, style="italic", color=SCARLET_DEEP,
            zorder=5)

    # --- The four rows.
    for n, name, reported, check, finding in rows:
        mid = row_mid[n]
        ax.text(54.0, mid, str(n), ha="center", va="center", fontsize=10.0,
                color=INK, zorder=5)
        ax.text(66.0, mid, name, ha="left", va="center", fontsize=10.5,
                color=INK, linespacing=1.18, zorder=5)
        ax.text(176.0, mid, reported, ha="left", va="center",
                fontsize=T_SERIES, color=INK, linespacing=1.22, zorder=5)
        ax.text(256.0, mid + 10.5, check, ha="left", va="center",
                fontsize=T_NOTE, style="italic", color=MUTE, zorder=5)
        # Chips are sized to their own text, not to the column: a uniform
        # chip wide enough for "Credited mechanisms falsified" left
        # "[0.32, 0.70]" sitting in mostly-empty scarlet and reading as a
        # table cell again. Measured once with a throwaway text artist at the
        # chip's own font and weight, then discarded; style.chip stays the
        # only call site that actually draws a chip, so tools/check_parity.py
        # still sees one drawer.
        probe = ax.text(0, 0, finding, fontsize=T_CHIP, fontweight="bold")
        bbox = probe.get_window_extent(fig.canvas.get_renderer())
        w_i = bbox.width * 72.0 / fig.dpi + 12.0
        probe.remove()
        chip(ax, 256.0, mid - 14.0, w_i, 12.5, SCARLET, finding,
             fontsize=T_CHIP, radius=3.0, ha="left")

    # Stages 1 and 2 carried no number and no finding; two full row bands to
    # say "counts reconcile" and "never diagnosed" cost a third of the canvas
    # for two facts that are not this figure's argument. One grey stub line,
    # at the spine's clean-segment cap, replaces both rows.
    ax.text(32.0, 202.0,
            "1, 2   Cohort construction and medication extraction: counts "
            "reconcile, never diagnosed.",
            ha="left", va="center", fontsize=T_NOTE, style="italic",
            color=MUTE, zorder=5)

    # Column headers. This pipeline's font (STIXGeneral, Agg/pdf backend) has
    # no reliable small-caps rendering, so terse upper-case at a reduced size
    # stands in for the "small caps" header treatment.
    ax.text(176.0, 183.0, "REPORTED", ha="left", va="center",
            fontsize=T_FLOOR, color=MUTE, zorder=5)
    ax.text(256.0, 183.0, "CHECK", ha="left", va="center",
            fontsize=T_FLOOR, color=MUTE, zorder=5)

    # Headline now states the propagation claim as well as the metric-blind
    # one: four of six stages hid a failure behind an unflagged number, and
    # the one at stage 3 is the input every stage after it consumed.
    signature(fig, SCARLET_DEEP,
              headline="Four of six stages hid a failure behind an "
              "unflagged number.\nThe stage 3 exemplar defect was consumed "
              "by every stage after it.")
    fname = "fig-pipeline-status.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    register(fname, "fig:pipeline-status",
             [p_exemplar, p_step2, p_exp1a, p_decontam],
             "Stages 1 and 2 (cohort construction, medication extraction) carry no "
             "committed numeric source and are summarized in one note line rather "
             "than drawn as rows. The detector-training row's finding is categorical "
             "narrative with no committed numeric source (CORRECTIONS.md). The "
             "reported-number column otherwise reads the pharmacist base rate on "
             "G299 (0.0401, step2_signal_check.json), the weak labeller's positive "
             "rate on the same 299 gold pairs (0.8729, "
             "exp1a_gold_error_rates_by_group.json), and the decontaminated AUROC "
             "point estimate alone (0.5145, corrected_gold_metrics.json). The "
             "check-finding column reads the wrong-rate upper bound (0 to 0.4745 of "
             "resolved candidates, exemplar_correctness_recomputed.json), the "
             "prevalence multiplier (~22x, same gold-set ratio fig:intro-headline "
             "brackets), and the bootstrap 95% CI on AUROC ([0.32, 0.70], "
             "corrected_gold_metrics.json). The spine asserts shared corrupted "
             "input from the stage 3 exemplar defect into rows 4-6, not that rows "
             "4-6's numbers follow causally from it: the 22x mismatch is a "
             "prevalence and specificity effect. AUROC corrected per "
             "CORRECTIONS.md C16 (was 0.460, now the decontaminated 3-seed ensemble "
             "0.5145).")


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
              f"The pharmacist's base rate is {pharm_pos/pharm_n:.4f}.",
              deck=f"{gold_ratio:.0f}x: the weak labeller's marginal over the "
                   f"pharmacist's, on the same {pharm_n} pairs.")
    fname = "fig-intro-headline.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
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

    # Abstract slots, coloured loudly. No synthetic clinical sentences: a thesis
    # about instruments that mislead, built on a MIMIC-III data agreement, does
    # not put invented clinical text on a canvas where a reader could take it
    # for a record. One data unit is one typographic point.
    W, H = W88, 2.70
    fig, ax = plt.subplots(figsize=(W, H))
    canvas(fig)
    plate(fig)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W * 72)
    ax.set_ylim(0, H * 72)
    ax.axis("off")

    # Every constant below is a typographic point on the plate, and the plate
    # is 10 percent wider and taller than it was. The slot grid grows with it
    # rather than sitting at its old size in a bigger frame: the type inside a
    # slot is still 9pt, so a bigger slot is a slot with more air in it, which
    # is what "make it bigger" has to mean when the type cannot move.
    x_nurse, x_disch, box_w = 64.0, 174.0, 92.0
    x_note = 244.0
    box_h, pitch = 21.0, 24.0
    y_row0 = 116.0
    rows_y = [y_row0 - i * pitch for i in range(5)]
    x_right = W * 72 - 1.5

    # The mismatch is a ROW, so it is drawn as one: a wash ground across the
    # whole canvas at the CESS row, under both columns and the annotation.
    ax.add_patch(mpatches.Rectangle(
        (1.5, rows_y[4] - box_h / 2 - 4.0), x_right - 1.5, box_h + 8.0,
        facecolor=SCARLET_WASH, edgecolor="none", linewidth=0, zorder=0))

    # The trigger bracket over the two column heads: the rule quantifies over
    # the whole set, which is the fact the positional selector ignores.
    y_br = 148.0
    ax.plot([x_nurse, x_disch], [y_br, y_br], color=COBALT_CORE, lw=LW_DATA,
            solid_capstyle="butt", zorder=4)
    for x in (x_nurse, x_disch):
        ax.plot([x, x], [y_br - 5.5, y_br], color=COBALT_CORE, lw=LW_DATA,
                solid_capstyle="butt", zorder=4)
    ax.text(x_disch + 14.0, y_br, "the rule quantifies over the whole set",
            ha="left", va="center", fontsize=T_SERIES, color=COBALT_DEEP,
            zorder=5)

    for x, head in ((x_nurse, "Nursing note"), (x_disch, "Discharge summary")):
        ax.text(x, 133.0, head, ha="center", va="center", fontsize=10.0,
                color=INK, zorder=5)

    # M1..M5 are mention identifiers (which slot); CONT/CESS are action
    # classes (what the rule reads off that slot) -- a different kind of
    # thing, and the two used to be set in identical plain weight, so a
    # reader's eye read the two columns as one series rather than two kinds
    # of label. No new colour: the fix is weight. The action-class column now
    # carries the same bold that its one CESS cell already got for free from
    # chip()'s F2 SOLID CHIP text, so all five rows of that column read as one
    # typographic family (bold, the finding's own weight) and the identifier
    # column stays plain, which also keeps M1..M5 from reading louder than
    # the action classes that carry the finding.
    nurse_tags = ["M1", "M2", "M3", "M4", "M5"]
    disch_tags = ["CONT", "CONT", "CONT", "CONT", "CESS"]
    for i, y in enumerate(rows_y):
        node(ax, x_nurse - box_w / 2, y - box_h / 2, box_w, box_h,
             COBALT_TINT, COBALT_DEEP, radius=2.5, zorder=3)
        ax.text(x_nurse, y, nurse_tags[i], ha="center", va="center",
                fontsize=T_SERIES, color=INK, zorder=5)
        if i < 4:
            node(ax, x_disch - box_w / 2, y - box_h / 2, box_w, box_h,
                 MIST, SLATE, radius=2.5, zorder=3)
            ax.text(x_disch, y, disch_tags[i], ha="center", va="center",
                    fontsize=T_SERIES, color=INK, fontweight="bold", zorder=5)
        else:
            # The one loud element on the canvas.
            chip(ax, x_disch - box_w / 2, y - box_h / 2, box_w, box_h,
                 SCARLET, disch_tags[i], fontsize=T_SERIES, radius=2.5,
                 zorder=3)

    # Annotations align to their own row and are joined to it by a rule on the
    # row's own baseline. No diagonals, no leaders.
    #
    # Row 4's note used to be set in SCARLET_DEEP. Its ground is NOT
    # SCARLET_TINT (the row band drawn above is SCARLET_WASH, a paler stop):
    # measured under WCAG, SCARLET_DEEP on SCARLET_WASH is 6.29:1, not the
    # 4.93:1 that pairing measures on SCARLET_TINT. But the note still sits on
    # a saturated ground rather than on paper, it is the same category of
    # defect the pipeline figure already carried and fixed (coloured text
    # directly on a coloured fill, at a margin a reviewer's tooling read as
    # thinner than it is), and the row's one loud element -- the CESS chip
    # this note explains -- already carries SCARLET at F2's >=7.7:1. The note
    # itself does not need to repeat the hue to make the point, so it is
    # retoned to GRAPHITE, this system's structural measurement tone rather
    # than a new colour, matching the pipeline figure's own precedent for
    # exactly this pairing (GRAPHITE on a SCARLET ground, not MUTE, because
    # MUTE-on-SCARLET_TINT measures 3.98:1 and fails 4.5:1 outright).
    # Measured: GRAPHITE on SCARLET_WASH = 10.12:1 (up from 6.29:1).
    notes = [
        (0, "selected by position (row 1):\nwhat the annotator, the weak\n"
            "labeller and the model saw", MUTE),
        (4, "the mention the rule\nactually fired on:\naction matched, dose adjacent",
         GRAPHITE),
    ]
    for i, text, tone in notes:
        y = rows_y[i]
        ax.plot([x_disch + box_w / 2 + 4.0, x_note - 5.0], [y, y], color=SLATE,
                lw=1.1, solid_capstyle="butt", zorder=4)
        ax.text(x_note, y, text, ha="left", va="center", fontsize=T_NOTE,
                color=tone, linespacing=1.24, zorder=5)

    signature(fig, SCARLET_DEEP,
              "The rule fires on the set.\nThe exemplar shown to everyone is row 1.")
    fname = "fig-exemplar.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    register(fname, "fig:exemplar", [p_exemplar],
             "Schematic; slots are abstract labels (M1..M5, CONT/CESS action-tag codes), no "
             "clinical text and no number: the wrong-exemplar rate it used to carry is a "
             "population rate and this panel draws a single candidate, so the rate is stated "
             "in the chapter body instead.")


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
    plot_series(truth, "^", ":", SCARLET_CORE, "pharmacist truth", 0.10)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"Rule {r}\n(n={ns[r]})" for r in rules], fontsize=8.0,
                       linespacing=1.25)
    ax.set_ylabel("Positive rate")
    ax.set_ylim(-0.06, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.35, 3.05)
    frame(ax)
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
        "GPT-4o zero-shot": ("s", PAPER, COBALT_CORE, COBALT_DEEP, MS_MARK,
                             (7, 4), "left", "bottom"),
        "GPT-4o few-shot": ("^", PAPER, AZURE_CORE, AZURE_DEEP, MS_MARK,
                            (7, -5), "left", "top"),
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
              "Two of three systems clear every exchange rate ACCORD's "
              "counts define.")
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
    fig.savefig(os.path.join(FIGDIR, fname))
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
#   counts from diag/step2_signal_check.json, and the four stages whose own
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
    # Registered directly rather than re-derived: has_both_roles holds the
    # nursing role on any nursing-labelled note, but a nursing note is
    # selectable only if its chart time is recorded and falls strictly before
    # discharge. The discharge side applies no such condition, so the whole
    # shortfall from has_both_roles to valid_admissions_both_selected is on
    # the nursing side.
    n_not_selectable = fact("cohort_chain.both_roles_not_selectable")
    n_note_rows = fact("cohort_chain.total_note_pair_rows")
    n_mentions = fact("extraction.medication_extraction.total_mentions")
    n_classified = fact("extraction.medication_extraction.classified_mentions")
    n_unclassified = fact("extraction.medication_extraction.unclassified_derived")
    n_drawn = fact("extraction.annotation_sample.total_sampled")

    # A true branching flow in three phase groups. The phase band is a WASH
    # ground and the stage boxes inside it are the TINT of the same hue, so the
    # group reads as a group and the boxes still read as boxes; a band and its
    # boxes in one stop would differ only by their strokes. One data unit is
    # one typographic point.
    # 3.86in, not the 4.18 a proportional growth would give: at 0.88 of the
    # measure the block height is the printed graphic plus 1.40in of caption
    # allowance, and 4.18 puts this figure at 5.62in, inside the 5.34 to 6.89
    # band that produces bottom-gapped pages. The height is what gives; the
    # type does not.
    W, H = W88, 3.86
    fig, ax = plt.subplots(figsize=(W, H))
    canvas(fig)
    plate(fig)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W * 72)
    ax.set_ylim(0, H * 72)
    ax.axis("off")
    x_right = W * 72 - 1.5

    # (stage name, count, exclusion below this stage or None)
    stages = [
        ("Heart failure admissions, ICD-9 prefix", f"{hf['pre_los']:,}",
         f"{hf['removed']:,} admissions with no\npositive length of stay"),
        ("After the length-of-stay filter", f"{hf['post_los']:,}", None),
        ("Admissions carrying both note roles", f"{n_both_roles:,}",
         f"{n_not_selectable:,} admissions, no nursing\nnote before discharge"),
        ("Note pair selected, one per role", f"{n_pairs:,} pairs", None),
        ("Medication mentions extracted", f"{n_mentions:,}",
         f"{n_unclassified:,} mentions with no\naction class assigned"),
        ("Mentions carrying an action class", f"{n_classified:,}", None),
        ("Candidates after the temporal prefilter", f"{n_candidates:,}", None),
        ("Pairs annotated by the pharmacist", f"{d299['n']} of {n_drawn}",
         f"{n_ambiguous} pairs coded ambiguous"),
        ("Non-ambiguous evaluation core",
         f"{d253['n']} pairs, {d253['n_pos']} confirmed contradictions", None),
    ]

    # The exclusion stub's arrowhead is 7.4pt long and 7.2pt wide, and it used
    # to land with its back edge at x = 225.6 against stage boxes that ended at
    # 228: the flare fused with the rounded bottom-right and top-right corners
    # of the boxes above and below, and both strokes were interrupted. The head
    # moves clear of the boxes instead of the boxes shrinking, because the two
    # chromatic boxes on this canvas carry its whole chroma budget. The stage
    # column shifts 2pt left, the gutter 4pt right, and the head now stops 3pt
    # short of the gutter: back edge 231.6, box stroke outer edge 226.6, 5.0pt
    # of clear paper between them. The gutter node loses 2pt of the 8pt it had
    # spare on its longest line.
    box_x, box_w = 26.0, 222.0
    x_gut = 268.0
    stub_gap = 3.0
    box_h, last_h, gap = 17.0, 28.0, 8.5
    y_top = H * 72 - 44.0
    tops, y = [], y_top
    for i in range(len(stages)):
        h = last_h if i == len(stages) - 1 else box_h
        tops.append((y, h))
        y -= h + gap

    # Phase groups: rows, WASH band, box TINT, DEEP stroke, rail, name.
    phases = [
        (range(0, 4), HAIR, MIST, SLATE, SLATE, "cohort construction"),
        (range(4, 7), COBALT_WASH, COBALT_TINT, COBALT_DEEP, COBALT_DEEP,
         "extraction and mining"),
        (range(7, 9), EMERALD_WASH, EMERALD_TINT, EMERALD_DEEP, EMERALD_DEEP,
         "annotation"),
    ]
    phase_of = {}
    for rows, wash, tint, deep, rail, name in phases:
        idx = list(rows)
        for i in idx:
            phase_of[i] = (tint, deep)
        b_hi = tops[idx[0]][0] + 5.0
        b_lo = tops[idx[-1]][0] - tops[idx[-1]][1] - 5.0
        ax.add_patch(mpatches.Rectangle((1.5, b_lo), x_right - 1.5, b_hi - b_lo,
                                        facecolor=wash, edgecolor="none",
                                        linewidth=0, zorder=0))
        ax.add_patch(mpatches.Rectangle((7.0, b_lo + 2.0), 3.0,
                                        b_hi - b_lo - 4.0, facecolor=rail,
                                        edgecolor="none", linewidth=0, zorder=1))
        ax.text(17.0, (b_lo + b_hi) / 2, name, ha="center", va="center",
                rotation=90, fontsize=T_NOTE, color=MUTE, zorder=2)

    x_stem = box_x + box_w / 2
    for i, ((t, h), (name, count, excl)) in enumerate(zip(tops, stages)):
        last = i == len(stages) - 1
        mid = t - h / 2
        if last:
            # The evaluated core: the one solid element, and the end of the path.
            chip(ax, box_x, t - h, box_w, h, SCARLET,
                 f"{name}\n{count}", fontsize=T_SERIES, radius=3.0, zorder=4)
        else:
            tint, deep = phase_of[i]
            node(ax, box_x, t - h, box_w, h, tint, deep, radius=2.5, zorder=4)
            ax.text(box_x + 7.0, mid, name, ha="left", va="center",
                    fontsize=T_SERIES, color=INK, zorder=5)
            ax.text(box_x + box_w - 7.0, mid, count, ha="right", va="center",
                    fontsize=T_AXIS, color=deep, zorder=5)
        if not last:
            arrow(ax, (x_stem, t - h), (x_stem, t - h - gap), color=GRAPHITE,
                  lw=LW_CONNECTOR, zorder=6)
        if excl is not None:
            # An exclusion is accounting, not failure: SLATE and MIST, never
            # scarlet. It leaves the trunk in the gap between two stages, so it
            # reads as taken out between them.
            #
            # It used to leave the trunk arrow itself, at the vertical middle
            # of the gap. That point sits inside the trunk arrowhead: the head
            # is 7.4pt of an 8.5pt gap, so a stub drawn there fuses with the
            # black head instead of reading as its own branch, and it reads as
            # feeding the stage below rather than being taken from the stage
            # above. A later pass moved the stub off the trunk (x_stem + 20)
            # and dropped it from the box's own bottom edge to the gap's
            # mid-line before turning right. That cleared the arrowhead, but
            # the turn itself still landed on the seam between two rows: the
            # stub above it was only gap/2 = 4.25pt tall, so the joint read as
            # sitting in the gap, equidistant from the row above and the row
            # below, rather than belonging to either.
            #
            # The connector now leaves the row's own right edge -- a physical
            # join to the box, not a line dropped from underneath it -- at
            # that row's own vertical centre, `mid`, the same height its name
            # and count already sit at. It runs clear of the box before it
            # ever turns, so the first thing a reader sees is a tap off THIS
            # row's own side, and only then does it drop to the gutter's
            # height and turn again for the excluded node. The whole path sits
            # at x >= box_x + box_w = 248, more than 100pt clear of the trunk
            # arrowhead centred on x_stem (137, +/-3.6 for its 7.2pt width):
            # it cannot fuse with that head at any y, not because the two are
            # offset by a margin but because they never share an x at all.
            x_elbow = box_x + box_w + 6.0
            y_gap = t - h - gap / 2
            ax.plot([box_x + box_w, x_elbow], [mid, mid], color=SLATE,
                    lw=LW_CONNECTOR, zorder=6, solid_capstyle="butt")
            ax.plot([x_elbow, x_elbow], [mid, y_gap], color=SLATE,
                    lw=LW_CONNECTOR, zorder=6, solid_capstyle="butt")
            arrow(ax, (x_elbow, y_gap), (x_gut - stub_gap, y_gap),
                  color=SLATE, lw=LW_CONNECTOR, zorder=6)
            lines = excl.count("\n") + 1
            ch = 11.0 * lines + 7.0
            node(ax, x_gut, y_gap - ch / 2, x_right - 4.0 - x_gut, ch, MIST,
                 SLATE, radius=2.5, zorder=4)
            ax.text(x_gut + 6.0, y_gap, excl, ha="left", va="center",
                    fontsize=T_NOTE, color=GRAPHITE, linespacing=1.24, zorder=5)

    ax.text(x_gut, y_top + 8.0, "excluded", ha="left", va="bottom",
            fontsize=T_NOTE, color=MUTE, style="italic")

    signature(fig, SCARLET_DEEP,
              "13,608 admissions in; 253 pairs and\n"
              "12 confirmed contradictions out.")
    fname = "fig-cohort-flow.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    register(fname, "fig:cohort-flow",
             [p_icd, p_transfer, p_step2, p_facts],
             "CONSORT-style path from the source database to the two evaluated sets. Nine "
             "stages, four exclusions, every count read from a committed file. One exclusion "
             "branch the diagram would otherwise carry is omitted because no committed file "
             "records it as a count: the admissions dropped between the length-of-stay filter "
             "and the both-roles requirement. That is a difference of two committed counts and "
             "not itself a recorded exclusion, so it is not drawn. The admissions dropped "
             "between the both-roles requirement and the selected pairs are drawn because that "
             "count is itself registered: the shortfall is entirely on the nursing side, since a "
             "both-roles admission is selectable only when some nursing note carries a chart "
             "time strictly before discharge, and the discharge side carries no matching "
             "condition. The ambiguous count is the difference between the two positive counts "
             "recorded on the same 299 rows in the signal-check file, the same subtraction the "
             "prevalence figure performs. Four stages read the thesis register rather than a "
             "primary artifact, because their own upstream summaries live outside this "
             "repository under the data agreement.")


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
# results/accord/generality/, but make_figures_extra.py no longer reads it:
# the rule-head transfer panel that once drew from that file was dropped from
# fig:falsification and its numbers moved to a table in the results chapter.
# The figure make_figures_extra.py builds now loads only the authority-scale
# aggregate (accord_v2_decontam_gold_aggregate.json). The one number the
# original entry was right about is the held-out 45.2%: 0.4515 appears in no
# result file anywhere, so that reference line was removed rather than
# sourced.


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
    make_fig_disc_actionability()
    make_fig_cohort_flow()

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
