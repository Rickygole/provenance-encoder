"""Fig 5 prevalence response, Fig 7 power, Fig 8 forest, Fig 9 scaling.

Three of the four are bound by the thesis. Fig 9, fig-scaling, is retired and
still built here; the reason of record is in RETIRED in
tools/check_figure_invariants.py and is repeated at its own block below.

Every value is READ from a committed result file. An earlier version typed the
C37 and C44 numbers as literals and drew the power curve from an ad-hoc formula
that did not reproduce the one real MDE it plotted -- the exact defect
corrections C7 and C10 retract elsewhere in this project. See C48.

All four figures follow the v3 system in style.py: boxed axes, direct labels
only, precomputed tints instead of alpha, and one saturated scarlet element per
axes at most. Figures 8 and 9 carry no scarlet at all, because Figure 8's claim
is a shape and Figure 9's claim is a success.

Figures 5 and 8 were rebuilt in the round that followed the compliance
withdrawal, each around a defect a reader could see and no tool could: Figure 5
annotated its most important datum with an arrow leaving the top of a panel
instead of plotting it, and Figure 8 put three incommensurable units on one
shared numeric axis. Both rebuilds are argued at their own blocks.
"""
import json, pathlib, glob, math, statistics as st
import numpy as np
import style
from style import (
    COBALT, AZURE, EMERALD, AMBER, PLUM, SCARLET, MAGENTA, TANGERINE, CYAN,
    COBALT_DEEP, COBALT_CORE, COBALT_TINT, COBALT_WASH, COBALT_BARE,
    AZURE_DEEP, AZURE_CORE, EMERALD_DEEP, EMERALD_CORE, EMERALD_TINT,
    EMERALD_WASH, AMBER_DEEP, AMBER_CORE, AMBER_TINT, AMBER_WASH,
    PLUM_DEEP, PLUM_CORE, SCARLET_DEEP, SCARLET_CORE, SCARLET_TINT,
    SCARLET_WASH,
    PAPER, INK, GRAPHITE, MUTE, SLATE, MIST, HAIR, PLATE_EDGE,
    T_HEADLINE, T_CLAIM, T_DECK, T_AXIS, T_TICK, T_SERIES, T_NOTE, T_PANEL,
    LW_SPINE, LW_DATA, LW_REF, LW_INTERVAL, MS_DOT, MS_MARK, MS_SEED,
    REF_DASH, ERRBAR, DOT, FIGW, canvas, frame, signature, panel_label, arrow,
    band, case, cased, wilson_ci)
style.setup()
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
ROOT = pathlib.Path("/Users/rickygole/provenance-encoder")

# =============================================================================
# Fig 5 -- fig-prevalence-response
# =============================================================================
# THIS FIGURE WAS REBUILT AROUND ITS OWN MOST IMPORTANT DATUM. It used to be a
# two-by-three grid of small multiples, one panel per model per prompt, on a y
# axis that stopped at 0.38 -- and the clinical operating point, the one place
# on the canvas where the deployed labeller meets the real base rate, did not
# fit on that axis. It was drawn as an arrow leaving the top of the first panel
# with its coordinates written beside it. An annotation that names a value is
# not a plot of it: the reader could not see how far above every swept point
# the real one sits, which is the whole reason the sweep understates the
# problem.
#
# The axis now reaches it, so it is a mark on the canvas like any other, and
# the cost is stated rather than hidden: the swept points occupy the bottom
# quarter of each panel instead of most of it, and the calibration diagonal is
# shallow rather than at forty-five degrees. That trade is the correct one.
# The claim is not how far apart 0.07 and 0.17 are, which is a table; the claim
# is that every response is flat against prevalence and that all of them sit an
# order of magnitude below where the deployed labeller actually operates.
#
# Six panels became two, one per prompt, three models each, so a reader
# compares models by looking at one panel rather than by saccading between
# three.
# The copy under results/, not the one under accord/prevalence_replication/.
# The two files are byte-identical; only this one is inside the tree that gets
# copied to a runtime, so reading it is what lets this figure build in Colab.
d = json.loads((ROOT / "results/accord/prevalence_replication/sweep_results.json").read_text())
models, prompts = d["models"], ["A_bare", "B_baserate_abstain"]
# The clinical comparison point is recomputed from the same two diagnostic
# files figures 1 and 2 draw, never typed.
_x01 = json.loads((ROOT / "results/accord/diag/step0_1_join_and_calibration.json").read_text())
_xtab = _x01["weak_confidence_is_degenerate"]["crosstab_label_x_confidence"]
clinical_y = _xtab["0.95"]["contradiction"] / (_xtab["0.95"]["contradiction"]
                                               + _xtab["1.0"]["not_contradiction"])
clinical_x = json.loads((ROOT / "results/accord/diag/step2_signal_check.json").read_text())[
    "contra_prob_vs_pharm_label__ambiguous_as_negative_n299"]["base_rate"]

PL = {"A_bare": "bare prompt",
      "B_baserate_abstain": "base rate stated, abstention allowed"}
# One hue per labeller, held across both panels. granite3.3:8b takes TANGERINE:
# it used to draw in the green this set reserves for pharmacist truth, and a
# weak labeller wearing the truth colour is the kind of thing this thesis is
# about.
MODEL_HUE = {"llama3:8b": MAGENTA, "granite3.3:8b": TANGERINE,
             "qwen3:8b": CYAN}
# Where each series is named, as (x of the swept point, vertical side). Chosen
# per panel so that three labels in a band a quarter of a panel tall never
# collide: the check that holds this is the text-collision gate in
# tools/check_figure_invariants.py, not an eye.
# The series are named in the LEFT panel only, and the right panel carries no
# label at all. That is not a legend by another route: both panels draw the
# same three labellers in the same hues under two prompts, which is what a
# small multiple is, and the deck says so. The alternative was measured and
# rejected. On a y axis that reaches the deployed operating point every swept
# series lives in the bottom quarter of the panel, and six labels in that band
# either print over one another or print over somebody else's interval; the
# only way to seat all six was a white casing laid over a data line, which
# erases the line and is a thing this set allows over a fill and not over a
# mark.
LABEL_AT = {
    "A_bare": {"llama3:8b": ("0.05", +1), "granite3.3:8b": ("0.25", +1),
               "qwen3:8b": ("0.25", -1)},
    "B_baserate_abstain": {},
}
Y_HI, X_HI = 0.90, 0.32
SWEEP_P = ["0.02", "0.05", "0.1", "0.25"]

fig, axes = plt.subplots(1, 2, figsize=(FIGW, 3.94), sharey=True)
canvas(fig)
fig.subplots_adjust(left=0.0930, right=0.9830, top=0.7810, bottom=0.1130,
                    wspace=0.075)
for c, pr in enumerate(prompts):
    ax = axes[c]
    # The wedge above the diagonal is over-prediction, and it is the whole
    # story of this figure, so it is drawn rather than left as white space.
    gx = np.linspace(0, X_HI, 2)
    ax.fill_between(gx, gx, Y_HI, facecolor=MIST, edgecolor="none",
                    linewidth=0, zorder=0)
    ax.plot([0, X_HI], [0, X_HI], color=SLATE, linestyle=REF_DASH,
            linewidth=LW_REF, zorder=2)
    for m in models:
        xs, ys, lo, hi = [], [], [], []
        for p in SWEEP_P:
            e = d["results"][m][pr].get(p)
            if not e:
                continue
            xs.append(float(p)); ys.append(e["positive_rate_of_decided"])
            # The sweep commits both counts behind every rate, so the interval
            # is a closed-form Wilson score interval on those counts.
            a, b = wilson_ci(e["n_contradiction"], e["n_decided"])
            lo.append(a); hi.append(b)
        hue = MODEL_HUE[m]
        ax.errorbar(xs, ys, yerr=[[y - a for y, a in zip(ys, lo)],
                                  [b - y for y, b in zip(ys, hi)]],
                    fmt="-o", color=hue.CORE, ms=6.4, lw=2.1, zorder=4,
                    markerfacecolor=hue.CORE, markeredgecolor=hue.DEEP,
                    markeredgewidth=1.0, ecolor=hue.CORE,
                    elinewidth=1.4, capthick=1.4, capsize=3.0)
        # Named at its own mark, in its own DEEP tone. No legend anywhere.
        if m in LABEL_AT[pr]:
            at, side = LABEL_AT[pr][m]
            i = SWEEP_P.index(at)
            y_anchor = (hi[i] + 0.022) if side > 0 else (lo[i] - 0.008)
            case(ax.text(xs[i], y_anchor, m, ha="center",
                         va="bottom" if side > 0 else "top",
                         fontsize=T_SERIES, color=hue.DEEP, zorder=6))
    if c == 0:
        # THE CLINICAL OPERATING POINT, PLOTTED. Its x is the pharmacist's
        # confirmed rate on the gold pairs and its y is the deployed weak
        # labeller's positive rate among the pairs it decided, both read from
        # the diagnostic files above. It is drawn on the bare-prompt panel
        # because the deployed labeller is not given a base rate and is not
        # allowed to abstain, which is that panel's condition.
        ax.plot([clinical_x], [clinical_y], marker="o", ms=MS_DOT + 1.6,
                color=SCARLET_CORE, markeredgecolor=SCARLET_DEEP,
                markeredgewidth=1.0, zorder=7)
        case(ax.text(clinical_x + 0.016, clinical_y,
                     f"deployed weak labeller:\n{clinical_x:.4f} read as "
                     f"{clinical_y:.4f}", ha="left", va="center",
                     fontsize=T_SERIES, color=SCARLET_DEEP, zorder=7,
                     linespacing=1.3))
        case(ax.text(X_HI - 0.008, Y_HI - 0.020, "over-prediction",
                     ha="right", va="top", fontsize=T_NOTE, color=MUTE,
                     zorder=5))
    if c == 1:
        # The diagonal's angle on the page, not 45 degrees: the panel is not
        # square and the two axes do not share a range.
        bb = ax.get_window_extent()
        ang = math.degrees(math.atan2(X_HI / Y_HI * bb.height, bb.width))
        # MUTE, not the SLATE of the line it names: SLATE is 3.48:1 on white
        # and this set does not set type in it.
        case(ax.text(0.205, 0.212, "perfect calibration", fontsize=T_NOTE,
                     color=MUTE, ha="center", va="bottom", rotation=ang,
                     rotation_mode="anchor", zorder=5))
    ax.set_xlim(0, X_HI); ax.set_ylim(0, Y_HI)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_title(PL[pr], fontsize=T_SERIES, pad=5, color=INK)
    ax.tick_params(labelsize=T_TICK)
    frame(ax, grid="y")
fig.supxlabel("True prevalence", fontsize=T_AXIS, y=0.016)
axes[0].set_ylabel("Labeller predicted-positive rate", fontsize=T_AXIS)
signature(fig, SCARLET_DEEP,
          "No labeller tracks prevalence, and none comes near the deployed "
          "rate.",
          # Two lines, each held under about 105 characters: a longer deck line
          # overflows the pinned bounding box and costs the whole canvas a
          # silent scale-down on the page. Invariant 1 is the check.
          deck="Bars are Wilson intervals on the counts behind each swept rate. "
               "The dashed line is perfect calibration;\neverything above it is "
               "over-prediction. Both panels draw the same three labellers, "
               "named in the left panel.")
fig.savefig(HERE / "fig-prevalence-response.pdf"); plt.close(fig)
print("wrote fig-prevalence-response.pdf")

# =============================================================================
# Fig 9 -- fig-scaling. RETIRED, and still built here.
#
# The figure is three points and a line, every number of which is already in a
# table, and its terminal claim -- the pretrained arm beats the 100M scratch
# arm -- is a row of fig:forest's third family, drawn there with an interval
# this panel does not carry. Nothing in the thesis binds it any more and the
# reason of record is in tools/check_figure_invariants.py.
#
# It goes on being built, rather than being switched off the way fig_frontier.py
# and make_figures_extra.py were, because this module still builds three live
# figures and because that is what this directory already does with a retired
# figure whose generator shares a file: make_figures.py has kept building
# fig-funnel, fig-prevalence and fig-tracking since each of them was retired.
# The retirement record is what keeps a retired PDF out of the thesis; a
# generator that stops running only makes the PDF stale.
#
# Panel (b) was deleted earlier: its content is Figure 8's last row.
# =============================================================================
sc = json.loads((ROOT / "results/accord/scaling/scaling_results.json").read_text())
by, pr_ = {}, {}
for x in sc:
    by.setdefault(x["tag"], []).append(x["accuracy"]); pr_[x["tag"]] = x["non_emb_params"]
tags = ["scratch_10M", "scratch_30M", "scratch_100M"]
xs = [pr_[t] for t in tags]; mu = [st.mean(by[t]) for t in tags]
sd = [st.stdev(by[t]) for t in tags]
pm, ps, pp = st.mean(by["pretrained_bert"]), st.stdev(by["pretrained_bert"]), pr_["pretrained_bert"]

fig, ax = plt.subplots(figsize=(FIGW, 2.84))
canvas(fig)
# Absolute margins held; the whole of the canvas's growth goes to the plot.
fig.subplots_adjust(left=0.1209, right=0.9890, top=0.7904, bottom=0.1648)
# The pretrained arm is a measured arm with three seeds and a spread of its
# own, not a chance line, so its band, its centre line and its label are all
# one hue and that hue is the reference-standard green.
ax.axhspan(pm - ps, pm + ps, facecolor=EMERALD_WASH, edgecolor="none",
           linewidth=0, zorder=1)
ax.axhline(pm, color=EMERALD_CORE, lw=LW_DATA, zorder=2)
# The gap between the two arms IS this figure's subject, and it was white
# space. It is the region between two measured series, so it is drawn as one.
ax.fill_between(xs, mu, [pm] * len(xs), facecolor=MIST, edgecolor="none",
                linewidth=0, zorder=1)
ax.errorbar(xs, mu, yerr=sd, fmt="none", ecolor=COBALT_CORE, zorder=3, **ERRBAR)
# SEED DOTS ARE DRAWN ON THE FROM-SCRATCH ARM ONLY, and the pretrained arm is
# the reason. Size is the variable on the scratch arm, so each of its points has
# an x and its three seeds stack on it. The pretrained arm is not a point on
# that sweep: it is a reference level, which is why it is drawn as a band across
# the WHOLE width rather than a mark at 8.6e7. Its three seeds are already on
# the canvas twice over -- they are what the band's half-height and the label's
# "plus or minus 0.0010" are computed from. Drawn a third time as dots they
# landed on the band's own centre line: two vanished under the 1.8pt stroke and
# the third cleared it by 2pt, leaving one isolated slate fleck grazing the
# label. Z-order cannot fix that (the third seed lies outside mean +/- sd, so it
# clears the band as well as the line) and jitter would put a false parameter
# count under a log axis. The band is the honest drawing of these three numbers.
for x, t in zip(xs, tags):
    ax.plot([x] * len(by[t]), by[t], "o", color=SLATE, ms=MS_SEED,
            markeredgewidth=0, zorder=4)
ax.plot(xs, mu, "-o", color=COBALT_CORE, ms=MS_DOT, lw=LW_DATA, zorder=5)
for x, m_, s_ in zip(xs, mu, sd):
    ax.annotate(f"{pm - m_:+.4f}", (x, m_ - s_), textcoords="offset points",
                xytext=(0, -11), ha="center", va="top", fontsize=T_NOTE,
                color=MUTE)
ax.set_xscale("log")
ax.set_xlabel("Non-embedding parameters (log scale)")
ax.set_ylabel("Accuracy, AG News")
ax.set_ylim(0.851, 0.950)
ax.set_yticks([0.86, 0.88, 0.90, 0.92, 0.94])
# The xlim runs 8% further right than the pretrained point, which is the room
# the label needs to sit inside the box at the line's own right end.
ax.set_xlim(7.0e6, 2.1e8)
case(ax.text(2.0e8, pm - 0.0034,
             f"pretrained BERT {pm:.4f} plus or minus {ps:.4f}",
             fontsize=T_SERIES, color=EMERALD_DEEP, va="top", ha="right",
             zorder=6))
# The subject of the figure, named inside itself. MUTE on MIST is 4.9:1.
ax.text(5.0e7, 0.9150, "the gap", fontsize=T_NOTE, color=MUTE,
        ha="center", va="center", zorder=6)
frame(ax, grid=True)
ax.plot(xs, mu, "-o", color=COBALT_CORE, ms=MS_DOT, lw=LW_DATA, zorder=5)
case(ax.text(xs[1], mu[1] + 0.0060, "from scratch", fontsize=T_SERIES,
             color=COBALT_DEEP, va="bottom", ha="center", zorder=6))
signature(fig, EMERALD_DEEP,
          "The from-scratch arm rises with size\n"
          "and never reaches the pretrained band.")
fig.savefig(HERE / "fig-scaling.pdf"); plt.close(fig)
print("wrote fig-scaling.pdf")

# =============================================================================
# Fig 8 -- fig-forest. The master summary: every headline estimate in the
# thesis, and the only place they appear.
#
# THIS FIGURE WAS REBUILT, and the defect it was rebuilt for is the one a
# reader cannot see: it put three incommensurable units on one shared numeric
# axis. Its own axis label admitted it, "unit named per family", which is an
# apology and not a fix -- a reader who compares a Matthews correlation with an
# AUROC by horizontal distance on a shared axis is comparing nothing, and the
# figure invited exactly that by drawing them on one ruler. Each unit family
# now has its own sub-axes with its own x range, its own ticks and its own zero
# rule, stacked and separated by whitespace and a hairline.
#
# Two more defects went with it. The three family GROUNDS were MIST,
# EMERALD_WASH and AMBER_WASH, which are 0.7633, 0.8051 and 0.7956 in relative
# luminance: three fills a greyscale reader cannot tell apart, doing the work
# of separating the families. The separation is now structural (a gap, a
# hairline, and a heading that names the unit) and no ground is drawn at all,
# which is also what takes this figure's ink coverage down; see
# tools/measure_ink.py. And the marker hue varied row to row with no key on the
# canvas, so six hues encoded nothing a reader could decode. Every mark is now
# the structural measurement tone, and the row's identity is where it always
# actually was, in the label beside it.
#
# THE HOSTED ROWS ARE GONE and are not filtered by name. intervals.json still
# carries claude_sonnet5_zeroshot and claude_sonnet5_fewshot in its reported
# block; the thesis register does not, because the compliance withdrawal moved
# every value obtained by transmitting this data to a hosted service under
# withdrawn.hosted.*. This figure therefore draws the rows the REGISTER carries
# (mccint.E2_mcc_intervals.*) and drops any row in the artifact that the
# register does not mirror. A row withdrawn later disappears from this canvas
# without anybody editing this file, which is the opposite of the failure mode
# CORRECTIONS records for the hand-maintained key set.
# =============================================================================
_FACTS = json.loads((ROOT / "results/accord/VERIFIED_FACTS.json").read_text())
iv = json.loads((ROOT / "results/accord/stats/intervals.json").read_text())["E2_mcc_intervals"]
# The register key is mccint.E2_mcc_intervals.<row>.<field> and <row> itself
# contains dots (accord_binary_at_0.5_253core), so the row name is everything
# between the prefix and the LAST dot. Splitting on the third dot instead
# silently dropped the ACCORD row, which is the kind of near miss this project
# has shipped before.
_PREFIX = "mccint.E2_mcc_intervals."
_live = {k[len(_PREFIX):].rsplit(".", 1)[0] for k in _FACTS
         if k.startswith(_PREFIX)}
_dropped = sorted(set(iv) - _live)
iv = {k: v for k, v in iv.items() if k in _live}
assert iv, "no Matthews row survives the register filter"
NICE = {"accord_binary_at_0.5_253core": "ACCORD at 0.5",
        "llama_baseline": "Llama 3 8B"}
fam_mcc = [((NICE.get(k, k) + f" ({v['n']} pairs)") if k.startswith("accord")
            else NICE.get(k, k), v["mcc"], v["ci95"][0], v["ci95"][1])
           for k, v in iv.items()]
_dec = json.loads((ROOT / "results/accord/decontam/corrected_gold_metrics.json").read_text())["aggregate"]
_ctrl = json.loads((ROOT / "results/accord/control/final_report.json").read_text())
_df34 = _ctrl["ensemble_mean_of_seed_scores"]["gold_defect_free_rule34"]
fam_auroc = [
    ("Decontaminated ensemble", _dec["ensemble_auroc"] - 0.5,
     _dec["ensemble_ci95"][0] - 0.5, _dec["ensemble_ci95"][1] - 0.5),
    ("Defect-free stratum", _df34["auroc"] - 0.5,
     _df34["bootstrap"]["ci95"][0] - 0.5, _df34["bootstrap"]["ci95"][1] - 0.5),
]
# The per-seed mortality-task scores read from the committed aggregate rather
# than from the quarantined per-run files under local_only/. The values are
# identical (verified seed by seed); the committed file is the one that travels,
# so the figure builds anywhere the repository does and nothing MIMIC-derived
# has to leave its quarantine for a figure to be drawn.
_task_runs = json.loads((ROOT / "results/task_runs.json").read_text())
_pb2 = [_task_runs[f"concat_s{s}"]["auroc"] for s in (1, 2, 3)]
_ou2 = [_task_runs[k]["auroc"] for k in sorted(_task_runs)
        if k.startswith("ours_current_s")]
_g = st.mean(_pb2) - st.mean(_ou2)
_se = math.sqrt(st.stdev(_pb2) ** 2 / 3 + st.stdev(_ou2) ** 2 / 3)
_sc = json.loads((ROOT / "results/accord/scaling/scaling_results.json").read_text())
_p100 = [x["accuracy"] for x in _sc if x["tag"] == "scratch_100M"]
_pbert = [x["accuracy"] for x in _sc if x["tag"] == "pretrained_bert"]
_g44 = st.mean(_pbert) - st.mean(_p100)
_se44 = math.sqrt(st.stdev(_pbert) ** 2 / 3 + st.stdev(_p100) ** 2 / 3)
# "In-hospital mortality, AUROC" not "... AUROC (3 seeds each)": the longer
# label overflowed the label column. It is still true and still in the data
# (task_runs.json carries exactly 3 seeds per arm); the caption owner needs to
# say it.
fam_gap = [
    ("In-hospital mortality, AUROC", _g, _g - 1.96 * _se, _g + 1.96 * _se),
    ("AG News at 100M, single budget", _g44, _g44 - 1.96 * _se44,
     _g44 + 1.96 * _se44),
]
# Each family's heading names its own unit, because each family now has its own
# ruler and nothing on this canvas is comparable across one.
FAMILIES = [("Matthews correlation", fam_mcc),
            ("AUROC above chance", fam_auroc),
            ("Pretraining gap, pretrained minus from scratch", fam_gap)]
for _, rs in FAMILIES:
    rs.sort(key=lambda r: r[1])

# ---------------------------------------------------------------------------
# Layout, in inches, resolved to figure fractions. Three stacked sub-axes share
# one left label column and one right value column; the plot column is the only
# thing that differs between them, which is the point.
# ---------------------------------------------------------------------------
# PAD_TOP holds the headline, the deck and the first family heading; the three
# collided at 0.40 and the heading printed through the deck. X_LEFT is set by
# the longest row label ("AG News at 100M, single budget"), which overflowed
# the pinned bounding box to the LEFT at 0.256 and took the saved measure to
# 440.95pt against this class's 424.348.
H = 3.38
PAD_TOP, PAD_BOT = 0.72, 0.28
AX_H, GAP = 0.50, 0.44
X_LEFT, X_RIGHT = 0.3250, 0.6950          # the plot column
X_VALUE = 0.9950                          # right-aligned value text
fig = plt.figure(figsize=(FIGW, H))
canvas(fig)

rows_all, sliver, axes_ = [], [], []
top = H - PAD_TOP
plot_w_pt = (X_RIGHT - X_LEFT) * FIGW * 72
for fam, rs in FAMILIES:
    ax = fig.add_axes([X_LEFT, (top - AX_H) / H, X_RIGHT - X_LEFT, AX_H / H])
    axes_.append(ax)
    # Its own range, from its own numbers, with symmetric padding so the zero
    # rule is never pushed to a spine and the reader cannot read the frame as a
    # bound.
    lo = min(r[2] for r in rs)
    hi = max(r[3] for r in rs)
    span = max(hi - lo, 1e-6)
    x0, x1 = min(lo, 0.0) - 0.16 * span, max(hi, 0.0) + 0.16 * span
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.72, len(rs) - 0.28)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5, prune=None))
    frame(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=T_TICK)
    ax.axvline(0, color=GRAPHITE, lw=1.2, zorder=2)
    for i, (lab, e, clo, chi) in enumerate(rs):
        half_pt = (chi - clo) / 2 / (x1 - x0) * plot_w_pt
        # Every mark is the structural measurement tone. A hue per row was six
        # hues with no key; the row is named in the column to its left.
        ax.errorbar([e], [i], xerr=[[e - clo], [chi - e]], fmt="o",
                    color=GRAPHITE, zorder=5, **ERRBAR, **DOT)
        mark = ""
        if half_pt < 1.5:
            for b in (clo, chi):
                ax.plot([b, b], [i - 0.16, i + 0.16], color=INK, lw=1.5,
                        solid_capstyle="butt", zorder=6)
            mark = "‡ "
            sliver.append(lab)
        fig.text(X_VALUE,
                 ((top - AX_H) + AX_H * (i + 0.5) / len(rs)) / H,
                 f"{mark}{e:+.4f}  [{clo:+.4f}, {chi:+.4f}]",
                 ha="right", va="center", fontsize=T_SERIES, color=GRAPHITE)
        rows_all.append((lab, e, clo, chi))
    ax.set_yticks(range(len(rs)))
    ax.set_yticklabels([r[0] for r in rs], fontsize=T_SERIES)
    # The family heading sits in the left column, above its own sub-axes, and
    # names the unit that sub-axes is drawn in.
    fig.text(0.012, (top + 0.085) / H, fam, ha="left", va="bottom",
             fontsize=T_SERIES, color=MUTE, style="italic")
    top -= AX_H + GAP
    if top > PAD_BOT:
        # The one separator: a hairline across the full measure in the enclosure
        # tone, which is what replaces three coloured grounds.
        # 0.285in above the next sub-axes: clear of the previous sub-axes'
        # tick labels above it and of the next family heading below it. At
        # GAP * 0.30 the rule printed straight through the heading.
        fig.add_artist(plt.Line2D([0.012, X_VALUE], [(top + 0.285) / H] * 2,
                                  color=PLATE_EDGE, lw=0.6))

if sliver:
    fig.text(0.012, 0.026, "‡ interval narrower than the marker",
             ha="left", va="bottom", fontsize=T_NOTE, color=MUTE)
n_cross = sum(1 for r in rows_all if r[2] <= 0 <= r[3])
signature(fig, GRAPHITE,
          f"{n_cross} of the {len(rows_all)} headline estimates in this thesis "
          f"cross zero.",
          deck="Three units, three rulers. Nothing on this canvas is "
               "comparable across a family boundary.")
fig.savefig(HERE / "fig-forest.pdf"); plt.close(fig)
print(f"wrote fig-forest.pdf ({n_cross}/{len(rows_all)} cross zero; "
      f"sliver rows: {sliver}; dropped as not in the register: {_dropped})")

# =============================================================================
# Fig 7 -- fig-power
# =============================================================================
pw = json.loads((ROOT / "results/accord/section6_auroc_power.json").read_text())
n_pos_study, n_neg_study = pw["n_positives"], pw["n_negatives"]
alpha_study, power_study = pw["alpha"], pw["power"]
_mort = json.loads((ROOT / "results/task_runs.json").read_text())["concat_s1"]["data_lineage"]
mort_n_pos = _mort["test_n_pos"]
mort_n_neg = _mort["test_n"] - mort_n_pos


# The significance level is named alpha_level, not alpha: this set bans the
# matplotlib alpha= keyword outright and verification greps for it, so a
# statistical alpha must not wear the same name as a transparency.
def hanley_mde(n_pos, n_neg, alpha_level=0.05, power=0.80):
    """Smallest AUROC distinguishable from 0.5 at the given power, Hanley and
    McNeil variance. Uses BOTH class counts."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha_level / 2) + norm.ppf(power)
    out = []
    for m_, n_ in zip(np.atleast_1d(n_pos), np.atleast_1d(n_neg)):
        lo, hi = 0.5, 0.9999
        for _ in range(80):
            a = (lo + hi) / 2
            q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
            v = (a * (1 - a) + (m_ - 1) * (q1 - a * a) + (n_ - 1) * (q2 - a * a)) / (m_ * n_)
            if (a - 0.5) / math.sqrt(max(v, 1e-12)) < z:
                lo = a
            else:
                hi = a
        out.append((lo + hi) / 2)
    return np.array(out)


n_pos = np.logspace(np.log10(5), np.log10(3000), 60)
n_neg = n_pos * (n_neg_study / n_pos_study)
mde = hanley_mde(n_pos, n_neg, alpha_level=alpha_study, power=power_study)
mde_study = hanley_mde([n_pos_study], [n_neg_study], alpha_level=alpha_study,
                       power=power_study)[0]
mde_mort = hanley_mde([mort_n_pos], [mort_n_neg], alpha_level=alpha_study,
                      power=power_study)[0]

fig, ax = plt.subplots(figsize=(FIGW, 2.85))
canvas(fig)
fig.subplots_adjust(left=0.1172, right=0.9890, top=0.8168, bottom=0.1737)
curve = np.clip(mde, 0.5, 1.0)
# Two grounds, hard boundary at the curve. The region above the curve is what
# this evaluation could have resolved; the region under it is what it could
# not, which is where this thesis's own evaluation sits.
#
# THE LOWER GROUND IS A TINT AND NOT A WASH, and the luminance gate in
# tools/check_figure_invariants.py is why. Every WASH stop in the palette sits
# within 0.010 relative luminance of every other (EMERALD_WASH 0.8051,
# SCARLET_WASH 0.8004), so a figure that separates two categories by two washes
# separates them by nothing at all in black and white: the detectable and the
# undetectable halves of this canvas printed as the same grey. SCARLET_TINT is
# 0.6163, which is 0.189 clear of EMERALD_WASH, and the fill is the same area,
# so the greyscale reader gets the partition back at no cost in coverage. This
# is the same defect, and the same remedy, that the pipeline figure's row
# grounds were reduced for.
ax.fill_between(n_pos, curve, 1.0, facecolor=EMERALD_WASH, edgecolor="none",
                linewidth=0, zorder=1)
ax.fill_between(n_pos, 0.5, curve, facecolor=SCARLET_TINT, edgecolor="none",
                linewidth=0, zorder=1)
# The MDE curve is measured, but it is not an entity, so it takes the
# structural measurement tone.
ax.plot(n_pos, curve, color=GRAPHITE, lw=LW_DATA, zorder=4)
ax.set_xscale("log")
ax.set_xlim(5, 3000)
ax.set_ylim(0.5, 1.0)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("Positive examples in the evaluation set (log scale)")
ax.set_ylabel(f"Minimum detectable AUROC\n({power_study * 100:.0f}% power)",
              linespacing=1.3)
case(ax.text(2700, 0.965, "detectable", fontsize=T_SERIES, color=EMERALD_DEEP,
             ha="right", va="top", zorder=5))
# INK, not SCARLET_DEEP: on the new SCARLET_TINT ground scarlet type measures
# 4.93:1, which the set's own advisory calls thin, and INK measures 11.3:1.
case(ax.text(6.0, 0.515, "not detectable at this sample size",
             fontsize=T_SERIES, color=INK, ha="left", va="bottom",
             zorder=5))
# Both marks are labelled adjacent to themselves. No leaders anywhere.
# GRAPHITE, not the SLATE this used to carry: SLATE is 3.48:1 on paper and this
# set does not set type in it, and the checker that enforces that ban counts a
# marker inside a text call's own window as type. The mortality control point
# is a second measured quantity on this GRAPHITE curve, not an entity, so it
# takes the same structural measurement tone as the curve it sits on rather
# than SLATE's reference-line-and-seed-dot role. GRAPHITE on paper is 12.49:1.
ax.plot([mort_n_pos], [mde_mort], "s", color=GRAPHITE, ms=MS_MARK,
        markeredgewidth=0, zorder=6)
case(ax.text(mort_n_pos, mde_mort + 0.022,
             f"mortality control, {mort_n_pos} positives", fontsize=T_SERIES,
             color=MUTE, ha="center", va="bottom", zorder=6))
ax.plot([n_pos_study], [mde_study], "o", color=SCARLET_CORE, ms=MS_DOT + 1.4,
        markeredgewidth=0, zorder=6)
case(ax.text(n_pos_study * 1.5, mde_study + 0.030,
             f"this study, {n_pos_study} positives", fontsize=T_SERIES,
             color=SCARLET_DEEP, ha="left", va="bottom", zorder=6))
frame(ax)
signature(fig, SCARLET_DEEP,
          f"At {n_pos_study} positives, nothing below AUROC {mde_study:.4f} "
          f"was ever detectable.")
fig.savefig(HERE / "fig-power.pdf"); plt.close(fig)
print("wrote fig-power.pdf")
