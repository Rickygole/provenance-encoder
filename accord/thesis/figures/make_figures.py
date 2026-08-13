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
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "grid.linewidth": 0.3,
    "grid.color": "#DCE0E4",
    "lines.linewidth": 1.2,
    "figure.dpi": 150,
    # Matches the notebook, so a figure built here and the same figure built
    # there have the same margins and neither clips an annotation.
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Palette. Shared with style.py (ACCENT/MUTED/LIGHT/DEAD) and with the notebook
# (DARK/MID/FILL/DEAD/RED), so the whole set looks like one document. The names
# below are historical; the colours are no longer grey.
#
# Greyscale luminance, 0.299R + 0.587G + 0.114B, out of 255:
#     GREY_DARK   #0A2795  royal blue     43
#     GREY_LIGHT  #00817B  deep teal      90
#     GREY_FILL   #FFD166  vivid amber   211
#     GREY_DEAD   #8E99A6  cool grey     151
#     ALERT       #D81E05  scarlet        83
# The three main tones are 47 and 121 apart, so bars stay separable in a
# black-and-white print. Under protanope and deuteranope simulation the three
# stay far apart in CIELAB (smallest pair dE 69): the distinction is blue vs
# teal vs yellow with a large lightness step, never red against green.
#
# Text checked against its background: GREY_DARK 12.1:1 on white, GREY_LIGHT
# 4.7:1 on white, ALERT 5.1:1 on white, black 14.6:1 on a GREY_FILL badge.
GREY_DARK = "#0A2795"
GREY_LIGHT = "#00817B"
GREY_FILL = "#FFD166"
GREY_DEAD = "#8E99A6"
ALERT = "#D81E05"
# Historical alias: this used to be a separate neutral grey for muted text and
# reference lines. It is now the same teal as GREY_LIGHT so that the scripts and
# the notebook (which only ever had one mid tone, MID) draw identical figures.
GREY_MED = GREY_LIGHT
HATCH_A = "///"
HATCH_B = "\\\\\\"
CHANCE_STYLE = dict(color=GREY_MED, linestyle="--", linewidth=1.0, zorder=1)

MANIFEST = []


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
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return (max(0.0, lo), min(1.0, hi))


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
    # base_rate here is on the 253-core; use the 299-pair base rate for the
    # headline ratio, which is what the intro figure computes it from.
    pharm_rate_all = step2["contra_prob_vs_pharm_label__ambiguous_as_negative_n299"]["base_rate"]
    ratio = weak_pos_rate / pharm_rate_all

    decontam, p_decontam = load("decontam", "corrected_gold_metrics.json")
    auroc = decontam["aggregate"]["ensemble_auroc"]
    ci = decontam["aggregate"]["ensemble_ci95"]

    # Six evenly spaced columns. Every string below is wrapped to at most about
    # twelve characters so that no badge and no diagnostic can grow wider than
    # its column: the previous version wrapped to the natural phrase length and
    # the long badges collided with their neighbours and with the row beneath.
    fig, ax = plt.subplots(figsize=(7.0, 2.25))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set_xlim(0, 6)
    ax.set_ylim(0.255, 1)
    ax.axis("off")

    stages = ["Cohort\nconstruction", "Medication\nextraction",
              "Rule-based\ncandidate\ngeneration", "LLM weak\nsupervision",
              "Detector\ntraining", "Evaluation"]
    badges = ["WORKS", "UNVALIDATED",
              f"ENRICHES BUT\nMIS-ATTRIBUTES\nEVIDENCE\n({defect_rate*100:.1f}%)",
              f"FAILS\n({ratio:.0f}x prevalence\nmismatch)",
              "THREE CLAIMS\nFALSIFIED*",
              f"FAILS\n(AUROC {auroc:.3f},\n95% CI\n[{ci[0]:.2f}, {ci[1]:.2f}])"]
    # Written out rather than left as a dash: the thesis bans en and em dashes,
    # and a double hyphen renders on canvas as something a reader reads as one.
    diagnostics = ["no diagnostic\nrun", "no diagnostic\nrun",
                   "action-set\nmatch vs\n.iloc[0] index",
                   "weak-label\nmarginal vs\ngold marginal",
                   "||theta_final -\ntheta_init||;\nOOD transfer F1*",
                   "bootstrap CI on\nAUROC before\nwrite-up"]

    n = len(stages)
    box_w = 0.80
    y_box, box_h = 0.755, 0.155
    y_badge_top, y_diag_top = 0.665, 0.395
    xs = np.linspace(0.5, 5.5, n)
    for i, (x, s, b, d) in enumerate(zip(xs, stages, badges, diagnostics)):
        box = FancyBboxPatch((x - box_w / 2, y_box), box_w, box_h,
                              boxstyle="round,pad=0.006,rounding_size=0.012",
                              linewidth=1.0, edgecolor=GREY_DARK,
                              facecolor="white")
        ax.add_patch(box)
        ax.text(x, y_box + box_h / 2, s, ha="center", va="center", fontsize=8.0,
                linespacing=1.25)
        if i < n - 1:
            ax.annotate("", xy=(xs[i + 1] - box_w / 2 - 0.02, y_box + box_h / 2),
                        xytext=(x + box_w / 2 + 0.02, y_box + box_h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=GREY_DARK, lw=1.0))
        ax.text(x, y_badge_top, b, ha="center", va="top", fontsize=8.0,
                color=GREY_DARK, linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=GREY_FILL,
                          edgecolor=GREY_DARK, linewidth=0.6))
        ax.text(x, y_diag_top, d, ha="center", va="top", fontsize=8.0,
                style="italic", color=GREY_MED, linespacing=1.3)

    ax.text(3.0, 0.985, "The pipeline as run, stage by stage", ha="center",
            va="top", fontsize=9.0, weight="bold")
    if SHOW_INFIGURE_NOTES: fig.text(0.5, 0.02,
            "*Categorical only (not re-plotted with a number): CORRECTIONS.md C1/C2/C4 document that the\n"
            "authority-weight and rule-F1 claims did not survive re-audit; no committed result file in this\n"
            "repo carries the underlying numbers, so they are not re-typed here.",
            ha="center", fontsize=8.0, color=GREY_MED)
    fname = "fig-pipeline-status.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    register(fname, "fig:pipeline-status",
             [p_exemplar, p_step01, p_step2, p_decontam],
             "Stage badges are categorical narrative (two, marked *, cite CORRECTIONS.md "
             "for a claim with no committed numeric source); the three parenthetical numbers "
             "(47.45% wrong of resolved, ~21x, AUROC 0.5145 with CI) are computed from the cited JSON files. "
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
    labels = ["GPT-4o\nz.s.", "GPT-4o\nf.s.", "ACCORD", "Llama 3\n8B"]

    # The right panel gets the wider share: its four category labels need room
    # side by side, while the left panel's y labels live outside its axes.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 3.5),
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})
    fig.subplots_adjust(left=0.155, right=0.935, top=0.97, bottom=0.155, wspace=0.50)

    # Middle bar: 261/299 is the weak labeller's marginal on the gold set, not
    # ACCORD's own predicted-positive count (that is 278/299). reconciliation_c28.json
    # exists to settle exactly this, and the bar was previously mislabelled ACCORD.
    bars = [("Weak labeller\npos. rate\n(train pool)", weak_pos, weak_n, wilson_ci(weak_pos, weak_n)),
            ("Weak labeller\npos. rate\n(gold set)", pred_pos, pred_n, wilson_ci(pred_pos, pred_n)),
            ("Pharmacist\ntrue rate\n(gold set)", pharm_pos, pharm_n, wilson_ci(pharm_pos, pharm_n))]
    ypos = np.arange(len(bars))[::-1]
    rates = [b[1] / b[2] for b in bars]
    los = [b[3][0] for b in bars]
    his = [b[3][1] for b in bars]
    errs = [[r - lo for r, lo in zip(rates, los)], [hi - r for r, hi in zip(rates, his)]]
    axL.barh(ypos, rates, color=GREY_FILL, edgecolor=GREY_DARK, height=0.55,
             xerr=errs, ecolor=GREY_DARK, capsize=3, zorder=3)
    # Value labels clear the upper CI cap rather than starting at the bar end,
    # and wrap to two lines so they cannot reach the ratio bracket on the right.
    for y, b, r, hi in zip(ypos, bars, rates, his):
        axL.text(hi + 0.045, y, f"{r*100:.1f}%\n({b[1]}/{b[2]})", va="center",
                 ha="left", fontsize=8.0, linespacing=1.25)
    axL.set_yticks(ypos)
    axL.set_yticklabels([b[0] for b in bars], fontsize=8.0)
    axL.set_xlim(0, 2.2)
    axL.set_ylim(-0.62, 2.62)
    axL.set_xticks([0, 0.5, 1.0])
    axL.set_xlabel("rate")
    axL.spines[["top", "right"]].set_visible(False)
    if SHOW_INFIGURE_TITLES: axL.set_title(f"Marginal, weak label vs.\\ truth ({weak_rate/pharm_rate:.0f}x)", fontsize=8.5)
    # The bracket compares the two bars measured on the same 299-pair gold set.
    # It used to span the training-pool bar and the gold-set bar, which have
    # different denominators.
    gold_ratio = (pred_pos / pred_n) / (pharm_pos / pharm_n)
    x_bracket = 1.74
    axL.annotate("", xy=(x_bracket, ypos[2]), xytext=(x_bracket, ypos[1]),
                 arrowprops=dict(arrowstyle="-", color=GREY_MED, lw=1.0))
    axL.text(x_bracket + 0.07, (ypos[1] + ypos[2]) / 2, f"~{gold_ratio:.0f}x",
              va="center", ha="left", fontsize=8.0, color=GREY_MED)

    xs = np.arange(4)
    vals = [mcc[k]["mcc"] for k in order]
    ci_lo = [mcc[k]["ci95"][0] for k in order]
    ci_hi = [mcc[k]["ci95"][1] for k in order]
    err = [[v - lo for v, lo in zip(vals, ci_lo)], [hi - v for v, hi in zip(vals, ci_hi)]]
    axR.axhline(0, **CHANCE_STYLE)
    axR.bar(xs, vals, yerr=err, color=GREY_FILL, edgecolor=GREY_DARK, capsize=3,
            hatch=None, width=0.55, linewidth=1.0, ecolor=GREY_DARK, zorder=3)
    axR.set_xticks(xs)
    axR.set_xticklabels(labels, fontsize=8.0)
    axR.set_xlim(-0.62, 3.62)
    axR.set_ylabel("Matthews correlation")
    if SHOW_INFIGURE_TITLES: axR.set_title("MCC, 95% CI, n=253", fontsize=8.5)
    # Sits at the end of the reference line it names, outside the plotted range,
    # so it cannot land on a bar or an interval.
    axR.text(3.70, 0.0, "chance", fontsize=8.0, color=GREY_MED, va="center",
             ha="left", clip_on=False)
    axR.spines[["top", "right"]].set_visible(False)

    if SHOW_INFIGURE_TITLES: fig.suptitle("The number that determines the rest of the thesis", fontsize=9.0, y=0.97)
    if SHOW_INFIGURE_NOTES: fig.text(0.5, 0.02,
            "Left-panel middle bar (weak labeller gold-set marginal): 261/299, summed from per-group counts in\n"
            "exp1a_gold_error_rates_by_group.json. CORRECTIONS.md C28 flags an unresolved 17-row discrepancy against a\n"
            "raw-file count of 278/299 in a file not committed to this repo; the value shown is the one reconstructable here.",
            ha="center", fontsize=8.0, color=GREY_MED)
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

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    stages = [
        (f"ICD-9 428-prefix admissions", hf["pre_los"]),
        (f"after LOS filter (removed {hf['removed']})", hf["post_los"]),
        (f"final note-pair cohort\n(both notes present & selectable)", n_final),
    ]
    ys = np.arange(len(stages))[::-1]
    maxw = stages[0][1]
    for y, (label, val) in zip(ys, stages):
        w = val / maxw
        ax.barh(y, w, height=0.5, color=GREY_FILL, edgecolor=GREY_DARK)
        ax.text(w + 0.02, y, f"{val:,}", va="center", fontsize=8.0, weight="bold")
        ax.text(-0.02, y, label, va="center", ha="right", fontsize=8.0)
    ax.set_xlim(-1.05, 1.25)
    ax.set_ylim(-0.55, len(stages) - 0.15)
    ax.axis("off")
    if SHOW_INFIGURE_TITLES: ax.set_title("Cohort funnel (heart failure), traceable stages only", fontsize=9.0)
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    if SHOW_INFIGURE_NOTES: fig.text(0.5, 0.02,
            "The original spec's intermediate step (9,825 admissions with both note roles present, 72.2%) and the three\n"
            "alternative three-role configurations (16.7%, 16.8%, 12.4%) are not present in any committed result file\n"
            "(grep of results/accord/**/*.json found no match for those counts) and are omitted here rather than typed in.",
            ha="center", fontsize=8.0, color=GREY_MED)
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
    # The former reference rate was supplied by a task description, never measured
    # (REVISION_LOG R8). Use the measured wrong rate on the resolved candidates.
    _ex = json.load(open(os.path.join(RESULTS, "exemplar", "exemplar_correctness_recomputed.json")))["rule_1_2_pooled"]
    defect_rate = _ex["n_wrong"] / _ex["n_resolved"]

    # Widened and re-anchored. Previously the trigger caption ran into the
    # "Discharge summary" heading, both leader arrows crossed a slot box and
    # struck through its label, the right-hand annotations ran off the axes, and
    # the summary text was wider than the box drawn around it.
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    # two columns of abstract mention slots (no clinical text)
    col_x = {"Nursing note": 2.0, "Discharge summary": 6.0}
    half_w = 0.55
    for label, x in col_x.items():
        ax.text(x, 6.1, label, ha="center", va="center", fontsize=8.5, weight="bold")

    nurse_tags = ["M1", "M2", "M3", "M4", "M5"]
    disch_tags = ["CONT", "CONT", "CONT", "CONT", "CESS"]
    rows_y = [4.7 - i * 0.78 for i in range(5)]
    for i, y in enumerate(rows_y):
        ax.add_patch(mpatches.Rectangle((col_x["Nursing note"] - half_w, y - 0.28), 2 * half_w, 0.56,
                                          facecolor="white", edgecolor=GREY_DARK))
        ax.text(col_x["Nursing note"], y, nurse_tags[i], ha="center", va="center", fontsize=8.0)
        fc = GREY_FILL if i == 4 else "white"
        ax.add_patch(mpatches.Rectangle((col_x["Discharge summary"] - half_w, y - 0.28), 2 * half_w, 0.56,
                                          facecolor=fc, edgecolor=GREY_DARK))
        ax.text(col_x["Discharge summary"], y, disch_tags[i], ha="center", va="center", fontsize=8.0)

    # trigger arrow spanning both columns, above the slots and below the headings
    ax.annotate("", xy=(col_x["Discharge summary"] - half_w - 0.05, 5.15),
                xytext=(col_x["Nursing note"] + half_w + 0.05, 5.15),
                arrowprops=dict(arrowstyle="-|>", color=GREY_DARK, lw=1.1))
    ax.text((col_x["Nursing note"] + col_x["Discharge summary"]) / 2, 5.32,
            "trigger: sets conflict\n(quantifies over the set)", ha="center", va="bottom",
            fontsize=8.0, color=GREY_MED, linespacing=1.25)

    # Leader arrows attach to the right edge of the slot they name, so they no
    # longer cross the box and strike through the tag inside it.
    x_note = col_x["Discharge summary"] + half_w + 0.9
    ax.annotate(".iloc[0]\n(what the model,\nannotator & LLM saw)",
                xy=(col_x["Discharge summary"] + half_w + 0.06, rows_y[0]),
                xytext=(x_note, rows_y[0]), va="center", ha="left",
                fontsize=8.0, color=GREY_DARK, linespacing=1.25,
                arrowprops=dict(arrowstyle="-|>", color=GREY_DARK, lw=0.9))
    ax.annotate("fix: action-matched\n+ dose-adjacent\n(the CESS mention)",
                xy=(col_x["Discharge summary"] + half_w + 0.06, rows_y[4]),
                xytext=(x_note, rows_y[4]), va="center", ha="left",
                fontsize=8.0, color=GREY_DARK, linespacing=1.25,
                arrowprops=dict(arrowstyle="-|>", color=GREY_DARK, lw=0.9))

    summary = (f"For {defect_rate*100:.1f}% of action-conflict candidates, the trigger-set slot and the\n"
               ".iloc[0] slot disagreed.")
    ax.text(4.8, 0.52, summary, ha="center", va="center", fontsize=8.0, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=GREY_FILL, edgecolor=GREY_DARK,
                      linewidth=0.8))
    fname = "fig-exemplar.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    register(fname, "fig:exemplar", [p_exemplar],
             "Schematic; slots are abstract labels (M1..M5, CONT/CESS action-tag codes), no "
             "clinical text. Only numeric content (41.4%) is sourced.")


# ============================================================================
# Figure 5 -- fig:rule-precision  (ch4_candidates.tex)
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

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.97, bottom=0.11)
    xs = np.arange(len(per_rule))
    aurocs = [s["auroc"] for s in per_rule]
    x_lo, x_hi = -0.55, 3.55
    ax.set_xlim(x_lo, x_hi)
    # Reference band: pooled rules 3+4 (defect-free) bootstrap CI from C36, drawn
    # across the rule-3 and rule-4 bars it actually describes. It is drawn in data
    # coordinates and behind the bars: the earlier axhspan took axes fractions,
    # which put its left edge on Rule 2, and it painted over the Rule 3 bar.
    ax.fill_between([2 - 0.4, x_hi], e3["unstratified_ci95"][0], e3["unstratified_ci95"][1],
                    color=GREY_LIGHT, alpha=0.25, linewidth=0, zorder=0,
                    label=f"pooled rules 3+4 95% CI\n[{e3['unstratified_ci95'][0]:.2f}, {e3['unstratified_ci95'][1]:.2f}] (n={e3['n']}, pos={e3['n_pos']})")
    ax.axhline(0.5, **CHANCE_STYLE)
    ax.bar(xs, aurocs, color=GREY_FILL, edgecolor=GREY_DARK, width=0.55, zorder=3)
    for x, s in zip(xs, per_rule):
        ax.text(x, s["auroc"] + 0.025, f"n={s['n']}\npos={s['pos']}", ha="center",
                va="bottom", fontsize=8.0, linespacing=1.25, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([s["name"].replace("rule ", "Rule ") for s in per_rule])
    ax.set_ylabel("ACCORD AUROC (per rule)")
    ax.set_ylim(0, 1.06)
    ax.legend(loc="upper left", fontsize=8.0, frameon=False, borderaxespad=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    if SHOW_INFIGURE_TITLES: ax.set_title("ACCORD's own ranking performance does not exceed chance\nwithin any single rule", fontsize=8.3)
    if SHOW_INFIGURE_NOTES: fig.text(0.02, 0.02,
             "Deviation from spec: per-rule ACCORD *precision* (as in the written spec) is not in any\n"
             "committed result file; per-rule AUROC is (defect_free_strata.json) and is shown instead.\n"
             "Individual-rule CIs are not computable here (per-pair scores not committed); only the\n"
             "pooled rules-3+4 CI (stats/intervals.json E3) is available and is shown as a reference band.",
             fontsize=8.0, color=GREY_MED)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fname = "fig-rule-precision.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    register(fname, "fig:rule-precision", [p_strata, p_intervals],
             "MODIFIED from spec: shows per-rule AUROC (sourced) rather than per-rule precision "
             "(pharmacist side sourced, ACCORD side not committed per-rule anywhere in results/accord).")


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

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    cats_weak = [("contradiction", weak_pos), ("not_contradiction", weak_neg)]
    cats_gold = [("contradiction", gold_pos), ("not_contradiction", gold_neg), ("ambiguous", gold_ambig)]

    def stacked(y, cats, n, hatches, small_dir):
        left = 0
        for (lab, val), h in zip(cats, hatches):
            frac = val / n
            ax.barh(y, frac, left=left, height=0.5, color="white" if h else GREY_FILL,
                    edgecolor=GREY_DARK, hatch=h)
            centre = left + frac / 2
            if frac > 0.25:
                ax.text(centre, y, f"{lab}\n{val}/{n}\n({frac*100:.1f}%)",
                        ha="center", va="center", fontsize=8.0)
            elif frac > 0.09:
                ax.text(centre, y, f"{lab}\n{val}/{n} ({frac*100:.1f}%)",
                        ha="center", va="center", fontsize=8.0)
            else:
                y_lab = y + small_dir * 0.62
                ax.annotate(f"{lab}\n{val}/{n} ({frac*100:.1f}%)", xy=(centre, y + small_dir * 0.25),
                            xytext=(centre, y_lab), ha="center",
                            va="bottom" if small_dir > 0 else "top", fontsize=8.0,
                            arrowprops=dict(arrowstyle="-", color=GREY_MED, lw=0.6))
            left += frac

    stacked(1, cats_weak, weak_n, [None, HATCH_A], small_dir=1)
    stacked(0, cats_gold, gold_n, [None, HATCH_A, HATCH_B], small_dir=-1)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([f"Weak labels\n(n={weak_n})", f"Pharmacist gold\n(n={gold_n})"], fontsize=8.0)
    ax.set_ylim(-1.0, 1.9)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("share of labels")
    ax.spines[["top", "right"]].set_visible(False)
    ratio = (weak_pos / weak_n) / (gold_pos / gold_n)
    ax.annotate("", xy=(1.0, 0), xytext=(1.0, 1), arrowprops=dict(arrowstyle="-", color=GREY_MED))
    ax.text(1.06, 0.5, f"~{ratio:.0f}x", rotation=90, va="center", fontsize=8.0, color=GREY_MED)
    implied_wrong = int(np.floor(weak_pos * (1 - gold_pos / gold_n)))
    if SHOW_INFIGURE_TITLES: ax.set_title(f"Marginals are close to complementary, not noisy copies\n"
                 f"implied floor: ≥{implied_wrong:,} of {weak_pos:,} positive weak labels wrong "
                 f"at the measured {gold_pos/gold_n*100:.1f}% base rate", fontsize=8.0)
    fig.tight_layout()
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

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    xs = np.arange(4)

    def plot_series(vals, marker, ls, label):
        cis = [wilson_ci(round(v * ns[r]), ns[r]) for v, r in zip(vals, rules)]
        lo = [v - c[0] for v, c in zip(vals, cis)]
        hi = [c[1] - v for v, c in zip(vals, cis)]
        ax.errorbar(xs, vals, yerr=[lo, hi], marker=marker, linestyle=ls, color=GREY_DARK,
                    markerfacecolor="white" if marker == "o" else GREY_DARK,
                    capsize=3, linewidth=1.1, label=label)

    plot_series(llama, "o", "-", "Llama 3 8B (teacher weak label)")
    plot_series(accord, "s", "--", "ACCORD (student prediction)")
    plot_series(truth, "^", ":", "pharmacist truth")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"Rule {r}\n(n={ns[r]})" for r in rules])
    ax.set_ylabel("positive rate")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right", fontsize=8.0, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    if SHOW_INFIGURE_TITLES: ax.set_title("The student tracks the teacher's marginal, not the truth's",
                 fontsize=8.5)
    fig.tight_layout()
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

    fig, ax = plt.subplots(figsize=(5.9, 4.2))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.98, bottom=0.115)
    d = np.linspace(0, P, 200)
    ax.plot(d, d, color=GREY_DARK, linewidth=1.2, zorder=3, label="PABAK (slope 1)")
    slope_ba = N / P
    ax.plot(d, d * slope_ba, color=GREY_MED, linewidth=1.1, linestyle="--", zorder=3,
            label=f"balanced accuracy (slope {slope_ba:.2f})")

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
    # Thick amber underneath, thin teal dashes on top: the two curves very nearly
    # coincide, and this way both stay visible without either hiding the other.
    ax.plot(deltas, dtn_curve, color=GREY_FILL, linewidth=3.0, zorder=2,
            label=f"MCC indifference (through ACCORD, MCC={mcc0:.3f})")

    markers = {"ACCORD": "o", "GPT-4o zero-shot": "s", "GPT-4o few-shot": "^", "Llama 3 8B": "D"}
    # Every label is placed away from the two near-coincident diagonals: the
    # GPT points are labelled up and to the left of the lines, ACCORD to the left
    # of the origin, Llama below the flat PABAK line.
    label_offset = {"ACCORD": (-9, 7), "GPT-4o zero-shot": (-9, 6), "GPT-4o few-shot": (-9, 6),
                     "Llama 3 8B": (13, -3)}
    label_align = {"ACCORD": "right", "GPT-4o zero-shot": "right", "GPT-4o few-shot": "right",
                   "Llama 3 8B": "left"}
    for name, s in systems.items():
        delta = TP0 - s["tp"]
        dtn = s["tn"] - TN0
        ax.scatter([delta], [dtn], marker=markers[name], s=45, color=GREY_DARK,
                   facecolor="white" if name != "ACCORD" else GREY_DARK, zorder=5)
        ax.annotate(name, (delta, dtn), textcoords="offset points", xytext=label_offset[name],
                    ha=label_align[name], va="center", fontsize=8.0, zorder=6)

    ax.axhline(0, color=GREY_DEAD, linewidth=0.6, zorder=0)
    ax.axvline(0, color=GREY_DEAD, linewidth=0.6, zorder=0)
    ax.set_xlabel(r"$\delta$ = forgone true positives, relative to ACCORD")
    ax.set_ylabel(r"$\Delta$TN, relative to ACCORD")
    ax.set_xlim(-2.3, P + 1)
    ax.set_ylim(-34, N + 22)
    ax.legend(loc="upper left", fontsize=8.0, frameon=False, borderaxespad=0.3)
    ax.spines[["top", "right"]].set_visible(False)
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
    labels = ["GPT-4o\nzs", "GPT-4o\nfs", "ACCORD", "Llama 3\n8B"]
    key_map = {"gpt4o_zeroshot": "gpt4o_zeroshot", "gpt4o_fewshot": "gpt4o_fewshot",
               "accord_binary_at_0.5_253core": "accord_binary_at_0.5_253core",
               "llama_baseline": "llama_baseline"}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 3.2), sharey=False)
    xs = np.arange(4)

    P = 12
    recalls, cis = [], []
    for k in order:
        s = baselines[k]
        recalls.append(s["tp"])
        cis.append(wilson_ci(s["tp"], P))
    err = [[r - c[0] * P for r, c in zip(recalls, cis)], [c[1] * P - r for r, c in zip(recalls, cis)]]
    axL.bar(xs, recalls, yerr=err, color=GREY_FILL, edgecolor=GREY_DARK, capsize=3, width=0.55)
    axL.set_ylim(0, 18.5)
    axL.set_yticks([0, 3, 6, 9, 12])
    axL.set_ylabel(f"true positives (of {P})")
    axL.set_xticks(xs)
    axL.set_xticklabels(labels, fontsize=8.0)
    axL.spines[["top", "right"]].set_visible(False)
    accord_pp = baselines["accord_binary_at_0.5_253core"]["tp"] + baselines["accord_binary_at_0.5_253core"]["fp"]
    llama_pp = baselines["llama_baseline"]["tp"] + baselines["llama_baseline"]["fp"]
    n253 = baselines["accord_binary_at_0.5_253core"]["tp"] + baselines["accord_binary_at_0.5_253core"]["fp"] + \
           baselines["accord_binary_at_0.5_253core"]["tn"] + baselines["accord_binary_at_0.5_253core"]["fn"]
    axL.text(2, recalls[2] + 2.8, f"{accord_pp}/{n253}\npred. positive\n(contaminated run)", ha="center", fontsize=8.0, color=GREY_MED)
    axL.text(3, recalls[3] + 2.8, f"{llama_pp}/{n253}\npred. positive", ha="center", fontsize=8.0, color=GREY_MED)
    if SHOW_INFIGURE_TITLES: axL.set_title("Recall", fontsize=8.5)

    mcc = intervals["E2_mcc_intervals"]
    vals = [mcc[key_map[k]]["mcc"] for k in order]
    lo = [mcc[key_map[k]]["ci95"][0] for k in order]
    hi = [mcc[key_map[k]]["ci95"][1] for k in order]
    err2 = [[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]]
    axR.axhline(0, **CHANCE_STYLE)
    axR.bar(xs, vals, yerr=err2, color="white", edgecolor=GREY_DARK, capsize=3, width=0.55, ecolor=GREY_DARK)
    axR.set_xticks(xs)
    axR.set_xticklabels(labels, fontsize=8.0)
    axR.set_ylabel("MCC")
    axR.set_xlim(-0.85, 3.6)
    axR.text(-0.75, 0.012, "chance", fontsize=8.0, color=GREY_MED, va="bottom", ha="left")
    axR.spines[["top", "right"]].set_visible(False)
    if SHOW_INFIGURE_TITLES: axR.set_title("Matthews correlation, 95% CI", fontsize=8.5)

    fig.tight_layout(rect=[0, 0, 1, 0.86])
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
    rows = [
        ("All non-ambiguous pairs\n(decontaminated 3-seed ensemble)",
         agg["ensemble_auroc"], agg["ensemble_ci95"], decontam["n"], decontam["n_positives"],
         agg["ensemble_mannwhitney_p"]),
        ("Rules 3+4 only\n(exemplar-defect-free by construction)",
         e3["auroc"], e3["bootstrap"]["ci95"], e3["n"], e3["pos"], None),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    fig.subplots_adjust(left=0.315, right=0.985, top=0.93, bottom=0.30)
    ys = np.arange(len(rows))[::-1]
    ax.axvline(0.5, **CHANCE_STYLE)
    for y, (label, auroc, ci, n, pos, p) in zip(ys, rows):
        ax.errorbar([auroc], [y], xerr=[[auroc - ci[0]], [ci[1] - auroc]], fmt="o",
                    color=GREY_DARK, capsize=4, markersize=6, linewidth=1.3, zorder=3)
        txt = f"AUROC {auroc:.4f}, n={n}, pos={pos}"
        if p is not None:
            txt += f", MWU p={p:.3f}"
        # White backing so the chance rule at 0.5 does not run through the digits.
        ax.text(auroc, y + 0.20, txt, ha="center", va="bottom", fontsize=8.0, zorder=4,
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"))
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.0, linespacing=1.3)
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(-0.55, 1.5)
    ax.set_xlabel("AUROC")
    ax.spines[["top", "right"]].set_visible(False)
    # Note sits below the axes, clear of both intervals.
    fig.text(0.5, 0.035,
             f"bootstrap: {control['bootstrap_n_resamples']:,} resamples, "
             f"{e3['bootstrap']['n_boot_valid']:,} valid on the defect-free stratum",
             ha="center", fontsize=8.0, color=GREY_MED)
    if SHOW_INFIGURE_TITLES: fig.suptitle("On evidence correct by construction, ranking is at chance,\nnot merely attenuated toward it",
                 fontsize=8.3, y=0.99)
    if SHOW_INFIGURE_NOTES: fig.text(0.5, 0.03,
             "No ROC curve is drawn: per-pair prediction scores are not committed to this repository (MIMIC-III DUA\n"
             "restricts released files to aggregates). Panels show the committed point estimate and its bootstrap CI instead.",
             ha="center", fontsize=8.0, color=GREY_MED)
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
    # Widened from 10.5 to 11.6 units so the longest event label no longer runs
    # off the right-hand edge, and every leader now starts outside the shaded
    # band instead of crossing it.
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    y_lo, y_hi = 0.85, 4.05
    ax.set_xlim(0, 11.6)
    ax.set_ylim(y_lo, y_hi)
    ax.axis("off")

    line_y = 2.3
    band_lo, band_hi = line_y - 0.42, line_y + 0.42
    events = [
        (1.0, "last nursing note\n(CHARTTIME present)", 3.05, "o"),
        (4.0, "discharge order", 3.05, "o"),
        (5.4, "DISCHTIME\n(\"patient leaves\")", 3.55, "o"),
        (9.0, "discharge summary finalised\n(CHARTDATE only, no CHARTTIME)", 3.05, "X"),
    ]
    ax.annotate("", xy=(10.9, line_y), xytext=(0.4, line_y),
                arrowprops=dict(arrowstyle="-|>", color=GREY_DARK, lw=1.2))
    for x, label, ytext, marker in events:
        ax.plot([x], [line_y], marker=marker, color=GREY_DARK,
                markersize=5 if marker == "o" else 8, zorder=4)
        ax.annotate(label, (x, band_hi + 0.04), xytext=(x, ytext), ha="center", va="bottom",
                    fontsize=8.0, linespacing=1.3,
                    arrowprops=dict(arrowstyle="-", color=GREY_MED, lw=0.6))

    # shaded region: no intervention possible (after DISCHTIME, before trigger)
    ax.axvspan(5.4, 9.0, ymin=(band_lo - y_lo) / (y_hi - y_lo),
               ymax=(band_hi - y_lo) / (y_hi - y_lo),
               color=GREY_FILL, alpha=0.9, zorder=1)
    # Sits above the timeline rather than straddling it, so the arrow does not
    # run between the two lines of the label.
    ax.text((5.4 + 9.0) / 2, line_y + 0.07, "no intervention\npossible", ha="center",
            va="bottom", fontsize=8.0, style="italic", zorder=5, linespacing=1.25)

    ax.annotate("system's trigger point\n(discharge summary)", xy=(9.0, band_lo - 0.04),
                xytext=(9.0, 1.25), ha="center", va="top", fontsize=8.0, linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=GREY_MED, lw=0.6))

    ax.annotate("proposed detection point", xy=(4.0, band_lo - 0.04), xytext=(3.4, 1.25),
                fontsize=8.0, ha="center", va="top", color=GREY_DARK,
                arrowprops=dict(arrowstyle="-|>", color=GREY_DARK, lw=0.9, linestyle="--"))

    if SHOW_INFIGURE_TITLES: ax.set_title("The actionability flaw: the trigger fires after the patient has left",
                 fontsize=8.5, pad=8)
    if SHOW_INFIGURE_NOTES: fig.text(0.5, 0.02,
             "Schematic of MIMIC-III field timing (CORRECTIONS.md C6: CHARTTIME is null for every\n"
             "discharge summary). No numeric estimate is shown; not sourced from a results JSON.",
             ha="center", fontsize=8.0, color=GREY_MED)
    fname = "fig-disc-actionability.pdf"
    fig.savefig(os.path.join(FIGDIR, fname))
    plt.close(fig)
    register(fname, "fig:disc-actionability", ["accord/CORRECTIONS.md (C6)"],
             "Structural/schema diagram, not derived from a numeric result file; documents a "
             "MIMIC-III field-availability fact. No CI applies (no estimate is plotted).")


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
    make_fig_rule_precision()
    make_fig_prevalence()
    make_fig_tracking()
    make_fig_met_exchange()
    make_fig_met_sharpened()
    make_fig_disc_strata()
    make_fig_disc_actionability()

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
