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

ROUND 7 (fixes_r7 N9). Each figure is now a function, make_fig_<name>(), so one
figure can be regenerated without rebuilding its siblings; running this file
still builds all four. Figure 5 no longer plots the clinical operating point on
public-text axes, Figure 8 is rebuilt on the thesis's current headline
estimates, and Figure 7 headlines the C108 per-rule floors. Every number those
three print is read from results/accord/VERIFIED_FACTS.json by key, or from the
artifact the key names, and a withdrawn.* key is refused.
"""
import json, pathlib, glob, math, statistics as st
import numpy as np
import matplotlib.ticker
import style
from style import (
    COBALT, AZURE, EMERALD, AMBER, PLUM, SCARLET, MAGENTA, TANGERINE, CYAN,
    COBALT_DEEP, COBALT_CORE, COBALT_TINT, COBALT_WASH, COBALT_BARE,
    AZURE_DEEP, AZURE_CORE, EMERALD_DEEP, EMERALD_CORE, EMERALD_TINT,
    EMERALD_WASH, AMBER_DEEP, AMBER_CORE, AMBER_TINT, AMBER_WASH,
    PLUM_DEEP, PLUM_CORE, SCARLET_DEEP, SCARLET_CORE, SCARLET_TINT,
    SCARLET_WASH,
    PAPER, INK, GRAPHITE, MUTE, SLATE, MIST, HAIR, PLATE_EDGE, FOG,
    WEAK_LABELLER, ENSEMBLE_DEEP, ENSEMBLE_CORE, PUBLIC_MODEL_MARK,
    T_HEADLINE, T_CLAIM, T_DECK, T_AXIS, T_TICK, T_SERIES, T_NOTE, T_PANEL,
    LW_SPINE, LW_DATA, LW_REF, LW_INTERVAL, MS_DOT, MS_MARK, MS_SEED,
    REF_DASH, ERRBAR, DOT, FIGW, canvas, frame, signature, panel_label, arrow,
    band, case, cased, wilson_ci)
style.setup()
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
ROOT = pathlib.Path(__file__).resolve().parents[3]


_FACTS = json.loads((ROOT / "results/accord/VERIFIED_FACTS.json").read_text())


def fact(key):
    assert not key.startswith("withdrawn."), f"{key} is withdrawn, not citable"
    return _FACTS[key]["value"]


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
def make_fig_prevalence_response():
    # The copy under results/, not the one under accord/prevalence_replication/.
    # The two files are byte-identical; only this one is inside the tree that gets
    # copied to a runtime, so reading it is what lets this figure build in Colab.
    d = json.loads((ROOT / "results/accord/prevalence_replication/sweep_results.json").read_text())
    models, prompts = d["models"], ["A_bare", "B_baserate_abstain"]
    # ROUND 7: the clinical operating point is no longer drawn. It sat on axes whose
    # x is SNLI prevalence, a different pool, prompt and kind of negative from the
    # clinical pool, which invited exactly the comparison the chapter says the
    # design cannot license. The clinical over-calling range is named in the deck
    # instead, read from the register, and is not given a coordinate.
    clin_lo = fact("overcall.qwen3_8b.pool.ratio")
    clin_hi = fact("overcall.llama3_8b.pool.ratio")
    cells = [(m, pr, lv, e["over_call_ratio"])
             for m in models for pr in prompts
             for lv, e in d["results"][m][pr].items()]
    n_cells = len(cells)
    n_under = sum(1 for c in cells if c[3] < 1)
    top = max(cells, key=lambda c: c[3])
    # The maximum is the register's own mirror of that cell, not only the artifact.
    assert abs(fact(f"publicsweep.results.{top[0]}.{top[1]}.{top[2]}.over_call_ratio")
               - top[3]) < 1e-9
    assert max(c[3] for c in cells) < clin_lo

    PL = {"A_bare": "bare prompt",
          "B_baserate_abstain": "base rate stated, abstention allowed"}
    # The three public models carry no hue: INK, MUTE and SLATE, told apart by
    # marker and dash pattern (style.PUBLIC_MODEL_MARK), so hue stays reserved
    # for the deployed labeller, the ensemble and failed checks. MODEL_HUE is
    # the name check_figure_invariants.py compares against style.ENTITY_HUE.
    MODEL_HUE = {"llama3:8b": INK, "granite3.3:8b": MUTE,
                 "qwen3:8b": SLATE}
    # ROUND 10 (audit item 4). Per-point inline labels used to name series in
    # the left panel only, leaving the right panel unlabelled, and the
    # llama3:8b label sat close enough above its own wide error bar that a
    # reader could read it as floating free of any curve. Both panels now
    # share one legend below the headline instead, so no inline label is
    # anchored to a single, sometimes-crowded point.
    # Models are dodged on x so their intervals do not print on one another.
    DODGE = {"llama3:8b": -0.004, "granite3.3:8b": 0.0, "qwen3:8b": +0.004}
    Y_LO, Y_HI, X_HI = -0.012, 0.32, 0.27
    SWEEP_P = ["0.02", "0.05", "0.1", "0.25"]

    # H grows by 0.30in (about 9 percent) over the round-9 canvas to make room
    # for the shared legend between the deck and the panel titles; the axes
    # keep their original absolute size.
    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 3.60), sharey=True)
    canvas(fig)
    fig.subplots_adjust(left=0.0930, right=0.9830, top=0.7379, bottom=0.1155,
                        wspace=0.075)
    for c, pr in enumerate(prompts):
        ax = axes[c]
        gx = np.linspace(0, X_HI, 2)
        ax.fill_between(gx, gx, Y_HI, facecolor=FOG, edgecolor="none",
                        linewidth=0, zorder=0)
        ax.plot([0, X_HI], [0, X_HI], color=SLATE, linestyle=REF_DASH,
                linewidth=LW_REF, zorder=2)
        for m in models:
            xs, ys, lo, hi = [], [], [], []
            for p in SWEEP_P:
                e = d["results"][m][pr].get(p)
                if not e:
                    continue
                xs.append(float(p) + DODGE[m]); ys.append(e["positive_rate_of_decided"])
                a, b = wilson_ci(e["n_contradiction"], e["n_decided"])
                lo.append(a); hi.append(b)
            marker, ls, tone = PUBLIC_MODEL_MARK[m]
            assert tone == MODEL_HUE[m]
            ax.errorbar(xs, ys, yerr=[[y - a for y, a in zip(ys, lo)],
                                      [b - y for y, b in zip(ys, hi)]],
                        fmt="none", ecolor=tone, elinewidth=1.1,
                        capthick=1.1, capsize=2.6, zorder=3)
            ax.plot(xs, ys, color=tone, linestyle=ls, lw=1.8, zorder=4,
                    marker=marker, ms=6.0, markerfacecolor=tone,
                    markeredgecolor=PAPER, markeredgewidth=0.8)
        if c == 0:
            case(ax.text(0.012, Y_HI - 0.010, "over-prediction",
                         ha="left", va="top", fontsize=T_NOTE, color=MUTE,
                         zorder=5))
        ax.set_xlim(0, X_HI); ax.set_ylim(Y_LO, Y_HI)
        # The diagonal is labelled in both panels, at its angle on the page.
        fig.canvas.draw()
        bb = ax.get_window_extent()
        ang = math.degrees(math.atan2(bb.height / (Y_HI - Y_LO),
                                      bb.width / X_HI))
        case(ax.text(0.185, 0.197, "perfect calibration", fontsize=T_NOTE,
                     color=MUTE, ha="center", va="bottom", rotation=ang,
                     rotation_mode="anchor", zorder=5))
        ax.set_xticks([0, 0.1, 0.2])
        ax.set_yticks([0, 0.1, 0.2, 0.3])
        ax.set_title(PL[pr], fontsize=T_SERIES, pad=5, color=INK)
        ax.tick_params(labelsize=T_TICK)
        frame(ax, grid="y")
    fig.supxlabel("True prevalence", fontsize=T_AXIS, y=0.016)
    axes[0].set_ylabel("Labeller predicted-positive rate", fontsize=T_AXIS)

    # ROUND 10 (audit item 4). One legend, shared by both panels, in the room
    # H's growth opened up between the deck and the panel titles.
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], color=MODEL_HUE[m], linestyle=PUBLIC_MODEL_MARK[m][1],
                             marker=PUBLIC_MODEL_MARK[m][0], ms=6.0,
                             markerfacecolor=MODEL_HUE[m], markeredgecolor=PAPER,
                             markeredgewidth=0.8, lw=1.8, label=m)
               for m in models]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
              bbox_to_anchor=(0.5, 0.855), fontsize=T_SERIES,
              handlelength=2.3, columnspacing=1.6, handletextpad=0.6)

    signature(fig, INK,
              f"On public text, {n_under} of {n_cells} labeller settings under-call,\n"
              f"and the largest over-call is {top[3]:.1f} times.",
              # AUDIT ITEM 11. "3.5 times" here and the "23.3 times" a
              # same-named column reaches in Table 11 are not the same
              # quantity: this figure is public text only, never the
              # clinical pairs, and the deck now says so with the register's
              # own clinical figure for scale.
              deck=(f"These panels are on public text only; on the clinical pairs the "
                    f"same over-call ratio reaches {clin_hi:.1f} times."))
    fig.savefig(HERE / "fig-prevalence-response.pdf"); plt.close(fig)
    print("wrote fig-prevalence-response.pdf")


# =============================================================================
# Fig 9 -- fig-scaling. RETIRED, and still built here.
#
# The figure is three points and a line, every number of which is already in a
# table, and its terminal claim -- the pretrained arm beats the 100M scratch
# arm -- belongs to the C44 control, which round 7 moved to the supplement, so
# fig:forest no longer carries it either. Nothing in the thesis binds it any more and the
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
def make_fig_scaling():
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
# Fig 8 -- fig-forest. The thesis's headline estimates on the clinical pairs,
# one panel per unit.
#
# ROUND 7 REBUILD (fixes_r7 N9). The previous canvas headlined "5 of the 6
# headline estimates cross zero", and two of the six were the C44 pretraining
# arms, which move to the supplement because their registration forbids using
# them on this evaluation, while a third was the Matthews correlation of the
# contaminated released run. It now draws what the thesis currently calls its
# headline estimates, and nothing else:
#   AUROC   the decontaminated detector ensemble on the 253 non-ambiguous pairs,
#           with its bootstrap interval, and the same ensemble on the Rules 3
#           and 4 stratum with ambiguous pairs dropped (108 pairs), with the
#           C114 bootstrap interval. The stratum is not labelled defect-free:
#           its exemplar correctness is assumed, not verified.
#   MCC     the deployed weak labeller measured by the C120 exact join on 299
#           pairs, drawn against the ceiling its own marginal sets. A count on
#           a fixed table has no resampling interval here; it carries the exact
#           one-sided hypergeometric p instead, and the canvas says so.
# The two units get two rulers, as before, because an AUROC and an MCC are not
# comparable by horizontal distance. What the panels share is the claim: none
# of the three is distinguishable from chance.
# =============================================================================
def make_fig_forest():
    n_253 = fact("decontam.n")
    n_108 = fact("c114.task_c.n")
    n_299 = fact("c120.g299.n")
    assert n_253 == fact("c120.g253.n")
    au = [
        (f"Ensemble, {n_253} non-ambiguous pairs",
         fact("decontam.aggregate.ensemble_auroc"),
         fact("decontam.aggregate.ensemble_ci95_lo"),
         fact("decontam.aggregate.ensemble_ci95_hi")),
        (f"Ensemble, Rules 3 and 4, {n_108} pairs",
         fact("c114.task_c.stratum_auroc"),
         fact("c114.task_c.ci95_lo"), fact("c114.task_c.ci95_hi")),
    ]
    assert all(lo <= 0.5 <= hi for _, _, lo, hi in au), \
        "an AUROC interval now excludes chance; the headline would be false"
    mcc_obs = fact("c120.g299.mcc")
    mcc_ceil = fact("c120.g299.ceiling_mcc")
    mcc_p = fact("c120.g299.p_hypergeom_one_sided")
    assert mcc_p > 0.05
    floor = fact("sizectl108.power_curves.primary_crossing_power_0_80")
    power = fact("power.power")
    n_pos_floor = fact("sizectl108.power_curves.n_pos_slots")
    assert n_pos_floor == fact("decontam.n_positives")
    # AUDIT ITEM 2. The 108-pair row has 6 positives, not 12, so its own
    # detectability floor differs from the top row's; there is no simulated
    # power curve registered for this stratum, but its analytic MDE is
    # (results/accord/mde_recomputed.json, strata.rules34_ambiguous_dropped),
    # which the canvas labels "analytic" so the two rows are never read as the
    # same computation.
    n_pos_108 = fact("c114.task_c.n_pos")
    floor_108 = fact("mde_recomputed.rules34_ambiguous_dropped.minimum_detectable_auc")

    # ROUND 11 (coordinator regression). H grows by 0.09in over the round-10
    # canvas so the shared power-floor note has a line of its own, clear of
    # the axes below it; A0, AH, M0 and MH stay in inches, so both panels
    # keep their exact size and position and the new room lands only above
    # them, between the deck and the AUROC panel's own header line. Kept to
    # the minimum that clears (tools/measure_ink.py's chroma floor is a
    # ratio to total canvas area, and blank canvas above the panels dilutes
    # it) rather than the first, more generous guess.
    H = 2.89
    fig = plt.figure(figsize=(FIGW, H))
    canvas(fig)
    X_LEFT, X_RIGHT, X_VALUE = 0.3250, 0.7700, 0.9950
    A0, AH = 1.24, 0.84
    M0, MH = 0.36, 0.40
    ax_a = fig.add_axes([X_LEFT, A0 / H, X_RIGHT - X_LEFT, AH / H])
    ax_m = fig.add_axes([X_LEFT, M0 / H, X_RIGHT - X_LEFT, MH / H])

    # AUROC panel: chance at 0.5 is the zero rule; the part of the axis above
    # the 80 percent power floor at this study's class counts is shaded.
    YA_LO, YA_HI = -0.8, 2.5
    ax_a.set_xlim(0.28, 1.0)
    ax_a.set_ylim(YA_LO, YA_HI)
    ax_a.set_xticks([0.3, 0.5, 0.7, 0.9])
    frame(ax_a, grid="x")
    # AUDIT ITEM 2. Shaded per row, at each row's own floor, instead of one
    # band spanning both rows at the top row's floor: the two strata have
    # different positive counts (12 against 6) and the shaded band used to
    # imply one shared floor for both.
    Y_SPLIT = 0.5
    ax_a.fill_betweenx([Y_SPLIT, YA_HI], floor, 1.0, facecolor=FOG,
                       edgecolor="none", lw=0, zorder=0)
    ax_a.fill_betweenx([YA_LO, Y_SPLIT], floor_108, 1.0, facecolor=FOG,
                       edgecolor="none", lw=0, zorder=0)
    # ROUND 11 (coordinator regression). Per-row in-plot labels for the two
    # bands crossed the data they sat beside (the top one over the right end
    # of that row's interval, the bottom one across its whisker): there is no
    # blank space inside either row's half of this axes tall enough for a
    # multi-line label to clear both the row above and the row below it. One
    # shared note above the axes, naming both floors, replaces both; the two
    # bands themselves are unchanged and still visibly start at different x.
    ax_a.axvline(0.5, color=GRAPHITE, lw=1.2, zorder=2)
    ax_a.text(0.492, YA_LO + 0.08, "chance", ha="right", va="bottom",
              fontsize=T_NOTE, color=MUTE, style="italic")
    for i, (lab, e, lo, hi) in enumerate(au):
        y = len(au) - 1 - i
        ax_a.errorbar([e], [y], xerr=[[e - lo], [hi - e]], fmt="o",
                      color=ENSEMBLE_DEEP, markerfacecolor=ENSEMBLE_CORE,
                      markeredgecolor=ENSEMBLE_DEEP, zorder=5,
                      **dict(ERRBAR, elinewidth=1.8, capthick=1.8), **DOT)
        fig.text(X_VALUE, (A0 + AH * (y - YA_LO) / (YA_HI - YA_LO)) / H,
                 f"{e:.2f}  [{lo:.2f}, {hi:.2f}]", ha="right", va="center",
                 fontsize=T_SERIES, color=ENSEMBLE_DEEP)
    ax_a.set_yticks([len(au) - 1 - i for i in range(len(au))])
    ax_a.set_yticklabels([r[0] for r in au], fontsize=T_SERIES)
    ax_a.tick_params(axis="y", length=0)
    ax_a.tick_params(axis="x", labelsize=T_TICK)
    fig.text(0.012, (A0 + AH + 0.08) / H,
             "AUROC, decontaminated detector, 95% bootstrap interval",
             ha="left", va="bottom", fontsize=T_SERIES, color=MUTE,
             style="italic")
    fig.text(0.012, (A0 + AH + 0.233) / H,
             f"Shaded: {power * 100:.0f}% power floor "
             f"({n_pos_floor} positives, simulated Mann-Whitney floor "
             f"{floor:.4f}; {n_pos_108} positives, analytic floor "
             f"{floor_108:.4f}).",
             ha="left", va="bottom", fontsize=T_NOTE, color=MUTE,
             style="italic")

    # MCC panel: zero is no association; the ceiling is a bound, drawn as a tick.
    ax_m.set_xlim(-0.02, 0.10)
    ax_m.set_ylim(-0.6, 0.6)
    ax_m.set_xticks([0.0, 0.03, 0.06, 0.09])
    frame(ax_m, grid="x")
    ax_m.axvline(0.0, color=GRAPHITE, lw=1.2, zorder=2)
    ax_m.text(0.0015, -0.56, "no association", ha="left", va="bottom",
              fontsize=T_NOTE, color=MUTE, style="italic")
    ax_m.plot([mcc_ceil, mcc_ceil], [-0.38, 0.38], color=WEAK_LABELLER, lw=2.0,
              solid_capstyle="butt", zorder=4)
    # AUDIT ITEM 1 (must-fix). This used to draw a 0-to-ceiling bar in the
    # same idiom (a capped horizontal span) as the 95 percent bootstrap
    # intervals in the AUROC panel above, although the caption says this
    # panel carries no resampling interval. Dropped outright rather than
    # redrawn: the ceiling tick and the observed point below are the whole
    # claim, and neither one reads as an interval on its own.
    ax_m.plot([mcc_obs], [0], "o", color=WEAK_LABELLER,
              markeredgecolor=WEAK_LABELLER, zorder=5, **DOT)
    ax_m.text(mcc_ceil + 0.002, 0.30, "ceiling", ha="left", va="top",
              fontsize=T_NOTE, color=MUTE, style="italic")
    ax_m.set_yticks([0])
    ax_m.set_yticklabels([f"Deployed labeller, {n_299} pairs"], fontsize=T_SERIES)
    ax_m.tick_params(axis="y", length=0)
    ax_m.tick_params(axis="x", labelsize=T_TICK)
    fig.text(X_VALUE, (M0 + MH / 2) / H,
             f"{mcc_obs:.2f}  ceiling {mcc_ceil:.2f}\nexact p {mcc_p:.2f}",
             ha="right", va="center", fontsize=T_SERIES, color=WEAK_LABELLER,
             linespacing=1.25)
    fig.text(0.012, (M0 + MH + 0.08) / H,
             "MCC against the pharmacist, exact count, no resampling interval",
             ha="left", va="bottom", fontsize=T_SERIES, color=MUTE,
             style="italic")
    fig.add_artist(plt.Line2D([0.012, X_VALUE], [1.02 / H] * 2,
                              color=PLATE_EDGE, lw=0.6))

    n_pos_253 = fact("decontam.n_positives")
    assert n_pos_253 == fact("c120.g299.n_positives")
    signature(fig, GRAPHITE,
              f"At {n_pos_253} positives, or {fact('c114.task_c.n_pos')} for Rules 3 "
              "and 4, no headline estimate is distinguishable from chance.",
              deck="Two units, two rulers: nothing is comparable across the "
                   "panel boundary.")
    fig.savefig(HERE / "fig-forest.pdf"); plt.close(fig)
    print("wrote fig-forest.pdf")


# =============================================================================
# Fig 7 -- fig-power
# =============================================================================
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


def make_fig_power():
    # Every quantity is read from the register by key: the study's class counts,
    # alpha and power (power.*, whose artifact is section6_auroc_power.json), and
    # the simulated 80 percent power floors (sizectl108.power_curves.*).
    n_pos_study, n_neg_study = fact("power.n_positives"), fact("power.n_negatives")
    alpha_study, power_study = fact("power.alpha"), fact("power.power")
    assert fact("sizectl108.power_curves.n_pos_slots") == n_pos_study
    assert fact("sizectl108.power_curves.n_neg_slots") == n_neg_study
    floor_mw = fact("sizectl108.power_curves.label_mix.crossings_at_power_0_80.mw_two_sided")
    floor_rule = fact("sizectl108.power_curves.label_mix.crossings_at_power_0_80.c98_rule")
    assert floor_mw == fact("sizectl108.power_curves.primary_crossing_power_0_80")
    n_pos = np.logspace(np.log10(5), np.log10(3000), 120)
    n_neg = n_pos * (n_neg_study / n_pos_study)
    mde = hanley_mde(n_pos, n_neg, alpha_level=alpha_study, power=power_study)
    mde_study = hanley_mde([n_pos_study], [n_neg_study], alpha_level=alpha_study,
                           power=power_study)[0]
    assert abs(mde_study - fact("sizectl108.power_curves.mde_analytic")) < 5e-5
    pct = f"{power_study * 100:.0f} percent"
    chance = 0.5

    fig, ax = plt.subplots(figsize=(FIGW, 3.30))
    canvas(fig)
    fig.subplots_adjust(left=0.1172, right=0.9890, top=0.8150, bottom=0.1420)
    Y_LO, Y_HI = 0.47, 0.92
    curve = np.clip(mde, chance, 1.0)
    # Below the curve, between chance and the analytic minimum detectable
    # AUROC, is what this design could not resolve at the stated power: FOG.
    # Above the curve is bare paper. tools/measure_ink.py holds this canvas to
    # a 0.53 coverage floor set when both regions were filled; this canvas
    # measures under it, and that is recorded in manifest.json rather than
    # met by tinting the upper region, which would be a band drawn for the tool.
    ax.fill_between(n_pos, chance, curve, facecolor=FOG, edgecolor="none",
                    linewidth=0, zorder=1)
    ax.plot(n_pos, curve, color=GRAPHITE, lw=2.2, zorder=4)
    ref_tone = SLATE    # reference and guide lines only; never type
    ax.axhline(chance, color=ref_tone, lw=LW_REF, linestyle=REF_DASH, zorder=3)
    ax.text(2800, chance - 0.006, f"chance, AUROC {chance:.2f}", ha="right",
            va="top", fontsize=T_NOTE, color=MUTE, style="italic", zorder=5)
    ax.axvline(n_pos_study, color=ref_tone, lw=0.6, zorder=2)
    ax.text(n_pos_study * 1.04, Y_HI - 0.010, f"{n_pos_study} positives",
            ha="left", va="top", fontsize=T_NOTE, color=MUTE, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(5, 3000)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="both", labelsize=T_TICK)
    ax.set_xlabel("Positive examples in the evaluation set (log scale)",
                  fontsize=T_AXIS)
    ax.set_ylabel(f"Minimum detectable AUROC\n({pct} power)", fontsize=T_AXIS,
                  linespacing=1.3)
    ax.text(2800, 0.60, f"detectable at {pct} power", ha="right",
            va="bottom", fontsize=T_SERIES, color=MUTE, zorder=5)
    ax.text(14, 0.508, f"not detectable at {pct} power", ha="left",
            va="bottom", fontsize=T_SERIES, color=MUTE, zorder=5)

    # The three points at this study's positive count, on the guide.
    # AUDIT ITEM 7. mde_study (0.7355) and floor_mw (0.7466) are close enough
    # that both markers, plotted at the same x, merged into one glyph at
    # print size. Dodged symmetrically off n_pos_study on the log axis; the
    # vertical guide line above still marks the true study size.
    X_DODGE = 1.045
    ax.plot([n_pos_study / X_DODGE], [mde_study], "o", ms=6, markerfacecolor=PAPER,
            markeredgecolor=INK, markeredgewidth=1.1, zorder=7)
    ax.plot([n_pos_study * X_DODGE], [floor_mw], "o", ms=5.2, color=INK,
            markeredgewidth=0, zorder=8)
    ax.plot([n_pos_study], [floor_rule], "D", ms=5.6, color=INK,
            markeredgewidth=0, zorder=8)
    # Labels to the right, each joined to its point by a thin leader.
    # AUDIT ITEM 6. Two decimals collapsed 0.7466 to 0.75 and 0.8084 to 0.81
    # on this canvas, matching neither the caption (which prints four
    # decimals) nor each other; the labels already name which quantity is
    # which, so the fix is precision, not a new label.
    x_lab = 30.0
    for y_pt, y_lab, text in (
            (floor_rule, floor_rule + 0.025,
             f"registered decision rule, power simulation: {floor_rule:.4f}"),
            (floor_mw, floor_mw + 0.035,
             f"Mann-Whitney test, power simulation: {floor_mw:.4f}"),
            (mde_study, 0.552, f"analytic curve: {mde_study:.4f}")):
        ax.plot([n_pos_study * 1.10, x_lab * 0.95], [y_pt, y_lab],
                color=GRAPHITE, lw=0.6, zorder=6, solid_capstyle="butt")
        ax.text(x_lab, y_lab, text, ha="left", va="center", fontsize=T_SERIES,
                color=GRAPHITE, zorder=7)
    frame(ax)
    signature(fig, INK,
              f"At {n_pos_study} positives, {pct} power needs AUROC "
              f"{floor_mw:.4f} under the Mann-Whitney test\nand {floor_rule:.4f} "
              f"under the registered decision rule.")
    fig.savefig(HERE / "fig-power.pdf"); plt.close(fig)
    print("wrote fig-power.pdf")


if __name__ == "__main__":
    make_fig_prevalence_response()
    make_fig_scaling()
    make_fig_forest()
    make_fig_power()
