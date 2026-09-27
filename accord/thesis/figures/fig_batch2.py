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
    VIOLET,
    COBALT_DEEP, COBALT_CORE, COBALT_TINT, COBALT_WASH, COBALT_BARE,
    AZURE_DEEP, AZURE_CORE, EMERALD_DEEP, EMERALD_CORE, EMERALD_TINT,
    EMERALD_WASH, AMBER_DEEP, AMBER_CORE, AMBER_TINT, AMBER_WASH,
    PLUM_DEEP, PLUM_CORE, SCARLET_DEEP, SCARLET_CORE, SCARLET_TINT,
    SCARLET_WASH,
    PAPER, INK, GRAPHITE, MUTE, SLATE, MIST, HAIR, PLATE_EDGE, FOG,
    WEAK_LABELLER, ENSEMBLE_DEEP, ENSEMBLE_CORE, PUBLIC_MODEL_MARK,
    PUBLIC_MODEL_NAME,
    BOUND_DEEP, BOUND_WASH,
    T_HEADLINE, T_CLAIM, T_DECK, T_AXIS, T_TICK, T_SERIES, T_NOTE, T_PANEL,
    LW_SPINE, LW_DATA, LW_REF, LW_INTERVAL, MS_DOT, MS_MARK, MS_SEED,
    REF_DASH, ERRBAR, DOT, FIGW, canvas, frame, signature, panel_label, arrow,
    band, case, cased, wilson_ci)
style.setup()
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
OUTDIR = style.out_dir(HERE)  # HERE unless ACCORD_FIG_OUT redirects it
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
    # COLOUR ROUND. The three public models take one hue each, the three they
    # wore in v3, and keep the marker and dash pattern they were given as
    # neutrals (style.PUBLIC_MODEL_MARK): every DEEP stop sits at one
    # lightness, so in greyscale marker and dash still do all the separating.
    # llama3:8b is MAGENTA, PLUM's neighbour, because it is the deployed
    # labeller's own model under another prompt. The line and the interval are
    # the ramp's DEEP, the marker face its CORE under a DEEP edge. The wedge
    # stays MIST: three hues is this canvas's whole budget. MODEL_HUE is the
    # name check_figure_invariants.py compares against style.ENTITY_HUE and
    # style.PUBLIC_MODEL_MARK.
    # HUE COLLISION ROUND. granite3.3:8b was TANGERINE, which at print size
    # was the ensemble's AMBER two figures on (9.7 dE76 apart at DEEP). It is
    # VIOLET now, 104.5 from AMBER at DEEP and used for nothing else; the
    # measurements are in style.py. Marker and dash are unchanged.
    MODEL_HUE = {"llama3:8b": MAGENTA, "granite3.3:8b": VIOLET,
                 "qwen3:8b": CYAN}
    # ROUND 10 (audit item 4). Per-point inline labels used to name series in
    # the left panel only, leaving the right panel unlabelled, and the
    # llama3:8b label sat close enough above its own wide error bar that a
    # reader could read it as floating free of any curve. Both panels now
    # share one legend below the headline instead, so no inline label is
    # anchored to a single, sometimes-crowded point.
    # Models are dodged on x so their intervals do not print on one another.
    # The dodge is a drawing offset, not data: all three models were run at
    # the same four prevalence levels, and the figure's note says so.
    DODGE = {"llama3:8b": -0.004, "granite3.3:8b": 0.0, "qwen3:8b": +0.004}
    # FIGURE POLISH. X_HI was 0.27 under a last tick of 0.2, so the 0.25 sweep
    # level sat past the end of the labelled axis. The axis now runs to a tick.
    Y_LO, Y_HI, X_HI = -0.012, 0.32, 0.30
    SWEEP_P = ["0.02", "0.05", "0.1", "0.25"]

    # H grows by 0.30in (about 9 percent) over the round-9 canvas to make room
    # for the shared legend between the deck and the panel titles; the axes
    # keep their original absolute size.
    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 3.60), sharey=True)
    canvas(fig)
    # VISUAL AUDIT. wspace was 0.075, a 14pt gutter, and the left panel's last
    # tick label "0.3" and the right panel's first, "0", sat 5pt apart and read
    # as one number. The gutter is now 22pt. Both end ticks stay: each panel
    # is read on its own axis, and the axis still runs from a tick to a tick.
    fig.subplots_adjust(left=0.0930, right=0.9830, top=0.7379, bottom=0.1155,
                        wspace=0.120)
    for c, pr in enumerate(prompts):
        ax = axes[c]
        gx = np.linspace(0, X_HI, 2)
        # FIGURE POLISH. The wedge was always drawn above the identity line
        # only, but in FOG, 0.89 against paper's 1.00, under a dashed boundary:
        # on the page the edge did not read and the panel looked filled. MIST
        # (0.76) is the set's neutral structural ground and makes the wedge a
        # wedge, in greyscale as well.
        ax.fill_between(gx, np.minimum(gx, Y_HI), Y_HI, facecolor=MIST,
                        edgecolor="none", linewidth=0, zorder=0)
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
            marker, ls, hue = PUBLIC_MODEL_MARK[m]
            assert hue is MODEL_HUE[m]
            ax.errorbar(xs, ys, yerr=[[y - a for y, a in zip(ys, lo)],
                                      [b - y for y, b in zip(ys, hi)]],
                        fmt="none", ecolor=hue.DEEP, elinewidth=1.1,
                        capthick=1.1, capsize=2.6, zorder=3)
            ax.plot(xs, ys, color=hue.DEEP, linestyle=ls, lw=1.8, zorder=4,
                    marker=marker, ms=6.0, markerfacecolor=hue.CORE,
                    markeredgecolor=hue.DEEP, markeredgewidth=1.0)
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
        # FIGURE POLISH. The label sat with its baseline on the line, so its
        # casing erased the dashes under it, and its upper end ran into the
        # 0.25 level's intervals. It is now lifted 5pt off the line along the
        # line's normal and centred in the gap between the 0.1 and 0.25 levels.
        rad = math.radians(ang)
        ax.annotate("perfect calibration", xy=(0.172, 0.172),
                    xytext=(-5.0 * math.sin(rad), 5.0 * math.cos(rad)),
                    textcoords="offset points", fontsize=T_NOTE, color=MUTE,
                    ha="center", va="bottom", rotation=ang,
                    rotation_mode="anchor", zorder=5)
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        ax.set_xticklabels(["0", "0.1", "0.2", "0.3"])
        ax.set_yticks([0, 0.1, 0.2, 0.3])
        ax.set_title(PL[pr], fontsize=T_SERIES, pad=5, color=INK)
        ax.tick_params(labelsize=T_TICK)
        frame(ax, grid="y")
    fig.supxlabel("True prevalence", fontsize=T_AXIS, y=0.016)
    axes[0].set_ylabel("Labeller positive rate", fontsize=T_AXIS)

    # ROUND 10 (audit item 4). One legend, shared by both panels, in the room
    # H's growth opened up between the deck and the panel titles.
    # VISUAL AUDIT. The legend printed the registry tags (llama3:8b) where
    # every table in the thesis prints the display name (Llama 3 8B); it now
    # prints style.PUBLIC_MODEL_NAME, so figure and table name one model one
    # way.
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], color=MODEL_HUE[m].DEEP,
                             linestyle=PUBLIC_MODEL_MARK[m][1],
                             marker=PUBLIC_MODEL_MARK[m][0], ms=6.0,
                             markerfacecolor=MODEL_HUE[m].CORE,
                             markeredgecolor=MODEL_HUE[m].DEEP,
                             markeredgewidth=1.0, lw=1.8,
                             label=PUBLIC_MODEL_NAME[m])
               for m in models]
    # Each name in its own series' DEEP tone, the set's rule for a series label.
    leg = fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
                     bbox_to_anchor=(0.5, 0.855), fontsize=T_SERIES,
                     handlelength=2.3, columnspacing=1.6, handletextpad=0.6)
    for txt, m in zip(leg.get_texts(), models):
        txt.set_color(MODEL_HUE[m].DEEP)

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
    fig.savefig(OUTDIR / "fig-prevalence-response.pdf"); plt.close(fig)
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
# was, because this module still builds three live
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
    fig.savefig(OUTDIR / "fig-scaling.pdf"); plt.close(fig)
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
    # (results/accord/mde_standard_condition.json, strata.g108), which the
    # canvas labels "analytic" so the two rows are never read as the same
    # computation.
    n_pos_108 = fact("c114.task_c.n_pos")
    # CORRECTED, AND THE DEFERRAL THAT USED TO STAND HERE IS GONE. This band
    # was drawn for months from mde_recomputed, which solves the defective one
    # standard error condition documented above mde_as_implemented in this same
    # file. At this stratum's counts (6 positives, 102 negatives) that route
    # returns 0.8064 and the standard two standard error condition returns
    # 0.8274, so the band used to start about 0.021 too far left, on the
    # optimistic side, in a figure whose subject is what the design could not
    # detect. It now draws the corrected floor, which is the same condition
    # fig-power draws and the same one Chapter 3's table and Chapter 4's prose
    # now print, so the reader meets one number rather than two.
    floor_108 = fact("mdestandard.g108.minimum_detectable_auc")
    # The pin. Both routes are recomputed here from the class counts: the drawn
    # floor must be the corrected one, and it must be the HIGHER of the two, so
    # a future edit cannot quietly walk this band back to the optimistic value.
    assert abs(mde_two_se([6], [102])[0] - floor_108) < 5e-5, \
        "the drawn 108-pair floor no longer reproduces the standard Hanley " \
        "and McNeil condition; see mdestandard in accord/stats/mde_standard_condition.py"
    assert floor_108 > mde_as_implemented([6], [102])[0], \
        "the drawn 108-pair floor is no longer above the as-implemented one; " \
        "the defective one standard error route has come back"

    # ROUND 11 (coordinator regression). H grows by 0.09in over the round-10
    # canvas so the shared power-floor note has a line of its own, clear of
    # the axes below it; A0, AH, M0 and MH stay in inches, so both panels
    # keep their exact size and position and the new room lands only above
    # them, between the deck and the AUROC panel's own header line. Kept to
    # the minimum that clears (tools/measure_ink.py's chroma floor is a
    # ratio to total canvas area, and blank canvas above the panels dilutes
    # it) rather than the first, more generous guess.
    # VISUAL AUDIT. The shaded-band note now says which side is shaded and
    # spells "percent", which made it 24pt wider than the canvas on one line,
    # so it is set on two and needs 0.15in more above the AUROC panel. Only
    # 0.07in of that is new canvas: the other 0.08in is taken from the 17pt of
    # bare paper that sat under the MCC panel's tick labels, by moving both
    # panels and the hairline between them down together (A0, M0 and Y_RULE
    # below; AH and MH are untouched, so both panels keep their size). The
    # chroma floor in tools/measure_ink.py is a ratio to canvas area, and this
    # canvas sits 4 percent over it rather than 1.
    H = 2.96
    fig = plt.figure(figsize=(FIGW, H))
    canvas(fig)
    X_LEFT, X_RIGHT, X_VALUE = 0.3250, 0.7700, 0.9950
    # FIGURE POLISH. H is unchanged (the chroma floor is a ratio to canvas
    # area and this canvas sits on it). 0.13in moves from the AUROC panel's
    # empty headroom to the MCC panel, at the same points-per-row scale in
    # both, so each panel has one clear line at its top for the names of its
    # reference marks. Those names used to sit at the bottom of each panel,
    # where "chance" touched the 108-pair interval and "no association"
    # crossed the frame.
    A0, AH = 1.29, 0.71
    M0, MH = 0.28, 0.53
    Y_RULE = 1.07          # the hairline between the two panels, in inches
    ax_a = fig.add_axes([X_LEFT, A0 / H, X_RIGHT - X_LEFT, AH / H])
    ax_m = fig.add_axes([X_LEFT, M0 / H, X_RIGHT - X_LEFT, MH / H])

    # AUROC panel: chance at 0.5 is the zero rule; the part of the axis above
    # the 80 percent power floor at this study's class counts is shaded.
    YA_LO, YA_HI = -0.8, 2.0
    ax_a.set_xlim(0.28, 1.0)
    ax_a.set_ylim(YA_LO, YA_HI)
    ax_a.set_xticks([0.3, 0.5, 0.7, 0.9])
    frame(ax_a, grid="x")
    # AUDIT ITEM 2. Shaded per row, at each row's own floor, instead of one
    # band spanning both rows at the top row's floor: the two strata have
    # different positive counts (12 against 6) and the shaded band used to
    # imply one shared floor for both.
    Y_SPLIT = 0.5
    ax_a.fill_betweenx([Y_SPLIT, YA_HI], floor, 1.0, facecolor=MIST,
                       edgecolor="none", lw=0, zorder=0)
    ax_a.fill_betweenx([YA_LO, Y_SPLIT], floor_108, 1.0, facecolor=MIST,
                       edgecolor="none", lw=0, zorder=0)
    # ROUND 11 (coordinator regression). Per-row in-plot labels for the two
    # bands crossed the data they sat beside (the top one over the right end
    # of that row's interval, the bottom one across its whisker): there is no
    # blank space inside either row's half of this axes tall enough for a
    # multi-line label to clear both the row above and the row below it. One
    # shared note above the axes, naming both floors, replaces both; the two
    # bands themselves are unchanged and still visibly start at different x.
    ax_a.axvline(0.5, color=GRAPHITE, lw=1.2, zorder=2)
    ax_a.text(0.492, YA_HI - 0.10, "chance", ha="right", va="top",
              fontsize=T_NOTE, color=MUTE, style="italic")
    # FIGURE POLISH. The value column is two sub-columns, the same in both
    # panels: the estimate right-aligned at X_EST, its qualifier (an interval
    # here, the ceiling below) right-aligned at X_VALUE, one line per row.
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()

    def _w(text):
        t = fig.text(0, 0, text, fontsize=T_SERIES)
        w = t.get_window_extent(_r).width / fig.bbox.width
        t.remove()
        return w
    quals = [f"[{lo:.2f}, {hi:.2f}]" for _, _, lo, hi in au] \
        + [f"ceiling {mcc_ceil:.2f}"]
    X_EST = X_VALUE - max(_w(q) for q in quals) - 8.0 / (72 * FIGW)
    assert X_EST - _w("0.00") > X_RIGHT + 6.0 / (72 * FIGW), \
        "value column reaches the axes"
    for i, (lab, e, lo, hi) in enumerate(au):
        y = len(au) - 1 - i
        ax_a.errorbar([e], [y], xerr=[[e - lo], [hi - e]], fmt="o",
                      color=ENSEMBLE_DEEP, markerfacecolor=ENSEMBLE_CORE,
                      markeredgecolor=ENSEMBLE_DEEP, zorder=5,
                      **dict(ERRBAR, elinewidth=1.8, capthick=1.8), **DOT)
        y_fig = (A0 + AH * (y - YA_LO) / (YA_HI - YA_LO)) / H
        fig.text(X_EST, y_fig, f"{e:.2f}", ha="right", va="center",
                 fontsize=T_SERIES, color=ENSEMBLE_DEEP)
        fig.text(X_VALUE, y_fig, quals[i], ha="right", va="center",
                 fontsize=T_SERIES, color=ENSEMBLE_DEEP)
    ax_a.set_yticks([len(au) - 1 - i for i in range(len(au))])
    ax_a.set_yticklabels([r[0] for r in au], fontsize=T_SERIES)
    ax_a.tick_params(axis="y", length=0)
    ax_a.tick_params(axis="x", labelsize=T_TICK)
    fig.text(0.012, (A0 + AH + 0.08) / H,
             "AUROC, decontaminated detector, nominal 95 percent bootstrap "
             "interval",
             ha="left", va="bottom", fontsize=T_SERIES, color=MUTE,
             style="italic")
    # VISUAL AUDIT. This line said "Shaded: 80% power floor" and did not say
    # which side of the floor the band is on, while the power figure, one
    # figure earlier, shaded the side that is NOT detectable. Both figures now
    # shade what IS detectable, in MIST, and both say so in the same words,
    # with "percent" spelled as the power figure spells it. The two named
    # floors are kept.
    fig.text(0.012, (A0 + AH + 0.233) / H,
             f"Shaded: detectable at {power * 100:.0f} percent power "
             f"({n_pos_floor} positives, simulated Mann-Whitney floor "
             f"{floor:.4f};\n{n_pos_108} positives, analytic floor "
             f"{floor_108:.4f}).",
             ha="left", va="bottom", fontsize=T_NOTE, color=MUTE,
             style="italic", linespacing=1.15)

    # MCC panel: zero is no association; the ceiling is a bound, drawn as a tick.
    ax_m.set_xlim(-0.02, 0.10)
    YM_LO, YM_HI = -0.6, 1.0
    ax_m.set_ylim(YM_LO, YM_HI)
    ax_m.set_xticks([0.0, 0.03, 0.06, 0.09])
    frame(ax_m, grid="x")
    ax_m.axvline(0.0, color=GRAPHITE, lw=1.2, zorder=2)
    ax_m.text(0.0015, YM_HI - 0.06, "no association", ha="left", va="top",
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
    # The ceiling tick is the labeller's own mark, so its name takes the
    # labeller's tone, as the same label does in fig-ceiling; "chance" and
    # "no association" name reference lines and stay MUTE.
    ax_m.text(mcc_ceil + 0.002, YM_HI - 0.06, "ceiling", ha="left", va="top",
              fontsize=T_NOTE, color=WEAK_LABELLER, style="italic")
    # FIGURE POLISH. The exact p was the value column's second line, which no
    # other row had. It is a statement about this estimate, so it sits at the
    # estimate, in its tone, and the value column keeps one shape.
    ax_m.text(mcc_obs + 0.0045, 0, f"exact p {mcc_p:.2f}", ha="left",
              va="center", fontsize=T_NOTE, color=WEAK_LABELLER, zorder=6)
    ax_m.set_yticks([0])
    ax_m.set_yticklabels([f"Deployed labeller, {n_299} pairs"], fontsize=T_SERIES)
    ax_m.tick_params(axis="y", length=0)
    ax_m.tick_params(axis="x", labelsize=T_TICK)
    # FIGURE POLISH. One line, the same two sub-columns as the AUROC rows.
    y_fig = (M0 + MH * (0 - YM_LO) / (YM_HI - YM_LO)) / H
    fig.text(X_EST, y_fig, f"{mcc_obs:.2f}", ha="right", va="center",
             fontsize=T_SERIES, color=WEAK_LABELLER)
    fig.text(X_VALUE, y_fig, quals[-1], ha="right", va="center",
             fontsize=T_SERIES, color=WEAK_LABELLER)
    fig.text(0.012, (M0 + MH + 0.08) / H,
             "MCC against the pharmacist, exact count, no resampling interval",
             ha="left", va="bottom", fontsize=T_SERIES, color=MUTE,
             style="italic")
    fig.add_artist(plt.Line2D([0.012, X_VALUE], [Y_RULE / H] * 2,
                              color=PLATE_EDGE, lw=0.6))

    n_pos_253 = fact("decontam.n_positives")
    assert n_pos_253 == fact("c120.g299.n_positives")
    # THE INTERVAL IS NOMINAL AND THE DECK NOW SAYS SO. "95 percent bootstrap
    # interval" was printed on this canvas as a property of the interval. It is
    # a property of the recipe: C134a resamples the C108 pool at this study's
    # own class counts, 12 positives and 241 negatives, 2,000 replications at
    # each of five true AUROC values, and measures how often the declared
    # percentile interval actually covers the truth. It never reaches 0.95 at
    # any of the five, which is not a defect this figure can fix and is a fact
    # every reading of its intervals depends on. The deck it replaces said
    # "Two units, two rulers: nothing is comparable across the panel boundary",
    # which the two panel header lines already say in more detail.
    #
    # The DECLARED method is what the thesis ran, so the declared method is
    # what is reported; the stratified variant covers worse still and naming
    # both here would cost the deck its one line to gain a comparison the
    # chapter makes.
    cov = json.loads((ROOT / "results/accord/c134_reanalysis"
                      / "c134a_bootstrap_coverage.json").read_text())
    assert cov["design"]["n_positives"] == n_pos_253
    assert cov["design"]["nominal_coverage"] == 0.95
    covs = [t["methods"]["declared_unstratified"]["coverage"]
            for t in cov["per_target"].values()]
    assert len(covs) == len(cov["design"]["targets"])
    cov_lo, cov_hi = min(covs), max(covs)
    assert cov_hi < cov["design"]["nominal_coverage"], \
        "the measured coverage now reaches nominal; this deck would mislead"
    signature(fig, GRAPHITE,
              f"At {n_pos_253} positives, or {fact('c114.task_c.n_pos')} for Rules 3 "
              "and 4, no headline estimate is distinguishable from chance.",
              deck=f"Nominal 95 percent bootstrap intervals; measured "
                   f"coverage at {n_pos_253} positives is {cov_lo:.3f} to "
                   f"{cov_hi:.3f}.")
    fig.savefig(OUTDIR / "fig-forest.pdf"); plt.close(fig)
    print("wrote fig-forest.pdf")


# =============================================================================
# Fig 7 -- fig-power
# =============================================================================
# The significance level is named alpha_level, not alpha: this set bans the
# matplotlib alpha= keyword outright and verification greps for it, so a
# statistical alpha must not wear the same name as a transparency.
# -----------------------------------------------------------------------------
# THE MINIMUM DETECTABLE AUROC, AND THE CONDITION THAT WAS ACTUALLY SOLVED.
#
# This curve was drawn for months from a condition that is not a power
# calculation. The route below named mde_as_implemented divides the WHOLE
# effect by the standard error at the ALTERNATIVE:
#
#     (a - 0.5) / SE(a)  >=  z_{1-alpha/2} + z_{power}
#
# The standard Hanley and McNeil / Obuchowski condition splits the two terms
# and uses the NULL standard error for the significance term, because the
# critical value of the test statistic is set under the null, and the
# alternative only for the power term:
#
#     a - 0.5  >=  z_{1-alpha/2} * SE(0.5)  +  z_{power} * SE(a)
#
# SE(a) < SE(0.5) for every a > 0.5, so the implemented route asks the effect
# to clear a smaller quantity and every floor it returns is TOO LOW. At this
# study's class counts it returns 0.7355 where the standard condition returns
# 0.7382, and at the 6-positive stratum 0.8064 where the standard condition
# returns 0.8274. A figure whose subject is what the design could not detect
# cannot draw the optimistic one of the two.
#
# ONE ARITHMETIC FACT WORTH RECORDING, because it is why SE(0.5) is not a
# second approximation smuggled in beside the Hanley and McNeil variance. At
# a = 0.5 that variance has Q1 = Q2 = 1/3, and it collapses exactly to
# (n_pos + n_neg + 1) / (12 n_pos n_neg), which is the simple normal
# approximation the methodology chapter writes out. The two are the same
# number, not two conventions, so _se_null below is the Hanley and McNeil
# variance evaluated at the null and nothing else.
#
# All three routes are kept. The corrected one draws the figure; the other two
# exist so make_fig_power can ASSERT that they sit either side of it, which is
# the check that would have caught this and is now impossible to revert past.
# -----------------------------------------------------------------------------
def _auroc_var_hanley(a, n_pos, n_neg):
    """Hanley and McNeil (1982) sampling variance of an AUROC estimate a."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return (a * (1 - a) + (n_pos - 1) * (q1 - a * a)
            + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)


def _se_null(n_pos, n_neg):
    """Standard error under the null, a = 0.5. See the note above: this is the
    Hanley and McNeil variance at 0.5 and equals (m + n + 1) / (12 m n)."""
    return math.sqrt(_auroc_var_hanley(0.5, n_pos, n_neg))


def _bisect_mde(holds, iters=200):
    """Smallest a in (0.5, 1) for which the condition holds. Monotone in a."""
    lo, hi = 0.5, 0.9999
    for _ in range(iters):
        a = (lo + hi) / 2
        if holds(a):
            hi = a
        else:
            lo = a
    return (lo + hi) / 2


def mde_two_se(n_pos, n_neg, alpha_level=0.05, power=0.80):
    """CORRECTED, and the only one this figure draws.

        a - 0.5 >= z_{1-alpha/2} * SE(0.5) + z_{power} * SE(a)

    The significance term takes the null standard error and the power term the
    alternative's, written out as two terms so the next reader can see which
    standard error belongs to which."""
    from scipy.stats import norm
    z_sig, z_pow = norm.ppf(1 - alpha_level / 2), norm.ppf(power)
    out = []
    for m_, n_ in zip(np.atleast_1d(n_pos), np.atleast_1d(n_neg)):
        se_null = _se_null(m_, n_)
        out.append(_bisect_mde(
            lambda a, m_=m_, n_=n_, se_null=se_null:
            (a - 0.5) >= z_sig * se_null
            + z_pow * math.sqrt(max(_auroc_var_hanley(a, m_, n_), 1e-12))))
    return np.array(out)


def mde_as_implemented(n_pos, n_neg, alpha_level=0.05, power=0.80):
    """THE DEFECTIVE ROUTE, kept only so the corrected figure can assert that
    it is the optimistic one. This is what accord/stats/recompute_mde.py
    solves and what every mde_recomputed.* register key still holds:

        (a - 0.5) / SE(a) >= z_{1-alpha/2} + z_{power}

    Nothing may draw from it."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha_level / 2) + norm.ppf(power)
    out = []
    for m_, n_ in zip(np.atleast_1d(n_pos), np.atleast_1d(n_neg)):
        out.append(_bisect_mde(
            lambda a, m_=m_, n_=n_:
            (a - 0.5) / math.sqrt(max(_auroc_var_hanley(a, m_, n_), 1e-12))
            >= z))
    return np.array(out)


def mde_null_only(n_pos, n_neg, alpha_level=0.05, power=0.80):
    """Both terms at the null: 0.5 + (z_sig + z_pow) * SE(0.5). The
    conservative end of the three, registered under
    mde_recomputed.*.minimum_detectable_auc_null_variance, and kept here for
    the same reason the defective route is."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha_level / 2) + norm.ppf(power)
    return np.array([0.5 + z * _se_null(m_, n_)
                     for m_, n_ in zip(np.atleast_1d(n_pos),
                                       np.atleast_1d(n_neg))])


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
    mde = mde_two_se(n_pos, n_neg, alpha_level=alpha_study, power=power_study)
    mde_study = mde_two_se([n_pos_study], [n_neg_study],
                           alpha_level=alpha_study, power=power_study)[0]

    # ---------------------------------------------------------------------
    # THE ASSERT THAT SHOULD HAVE CAUGHT THIS. It locked the drawn marker to
    # sizectl108.power_curves.mde_analytic and it passed for months, because
    # the figure and the register were computing the same wrong thing. It is
    # not removed. It is split into four, and each one names what it is for.
    #
    # The corrected value is NOT typed here. The register already holds it,
    # under a key the current recompute script overwrote believing it did not
    # reproduce: control.power_ceiling.referenced_gold_all_ambiguous_excluded,
    # 12 positives, 241 negatives, alpha 0.05, power 0.80, minimum detectable
    # AUROC 0.7382. It reproduces exactly under the two standard error
    # condition, to four decimals, which is the precision the register stores.
    # ---------------------------------------------------------------------
    REG = "control.power_ceiling.referenced_gold_all_ambiguous_excluded"
    assert fact(f"{REG}.n_positives") == n_pos_study
    assert fact(f"{REG}.n_negatives") == n_neg_study
    assert fact(f"{REG}.alpha") == alpha_study
    assert fact(f"{REG}.power") == power_study
    assert abs(_se_null(n_pos_study, n_neg_study)
               - fact(f"{REG}.se_auc_under_null")) < 5e-5, \
        "the null standard error no longer matches the registered one"
    # 1. The curve's own value at this study's size is the registered one.
    mde_registered_correct = fact(f"{REG}.minimum_detectable_auc")
    assert abs(mde_study - mde_registered_correct) < 5e-5, \
        "the drawn analytic floor no longer reproduces the registered 0.7382"
    # 2. The defective route is the OPTIMISTIC one, and it is exactly what
    #    mde_recomputed and sizectl108 still hold. If either key is ever
    #    corrected upstream, this assert fails and this figure gets revisited
    #    rather than quietly disagreeing with the chapter.
    mde_impl = mde_as_implemented([n_pos_study], [n_neg_study],
                                  alpha_level=alpha_study,
                                  power=power_study)[0]
    assert abs(mde_impl - fact("sizectl108.power_curves.mde_analytic")) < 5e-5, \
        "sizectl108.power_curves.mde_analytic no longer holds the as-implemented value"
    assert abs(mde_impl
               - fact("mde_recomputed.gold_ambiguous_excluded"
                      ".minimum_detectable_auc")) < 5e-5
    assert mde_impl < mde_study, \
        "the as-implemented route is no longer the optimistic one; the " \
        "direction of the defect has changed and this figure must be reread"
    # 3. Holding BOTH terms at the null is the conservative end, and it too is
    #    registered. The corrected condition must sit strictly between them.
    mde_null = mde_null_only([n_pos_study], [n_neg_study],
                             alpha_level=alpha_study, power=power_study)[0]
    assert abs(mde_null
               - fact("mde_recomputed.gold_ambiguous_excluded"
                      ".minimum_detectable_auc_null_variance")) < 5e-5
    assert mde_impl < mde_study < mde_null, \
        "the corrected floor no longer lies between the two one-sided routes"
    # 4. The pre-registered threshold is a HISTORICAL FACT, not a sensitivity
    #    claim: it is the number several checks in this thesis were judged
    #    against, and it happens to equal the as-implemented route because
    #    that route is what computed it. It is drawn as a fixed constant and
    #    is never moved here; moving it is the author's decision.
    mde_registered_rule = fact("sizectl108.power_curves.mde_analytic")

    pct = f"{power_study * 100:.0f} percent"
    chance = 0.5

    # H 3.30 to 3.42 and top 0.8150 to 0.7900, both for the same reason: the
    # headline is three lines rather than two now that the corrected analytic
    # floor and the pre-registered threshold are two separate numbers, and at
    # the old geometry the third line landed on the axes box. The axes gives
    # up 12pt of height; no type size and no tuned margin else moves.
    fig, ax = plt.subplots(figsize=(FIGW, 3.42))
    canvas(fig)
    fig.subplots_adjust(left=0.1172, right=0.9890, top=0.7900, bottom=0.1420)
    # FIGURE POLISH. Y_LO was 0.47, which left 10pt under the chance line for
    # an 8.5pt label: it touched the dashes above it and the frame below it.
    # At 0.44 the band is 20pt and the label sits in the middle of it.
    Y_LO, Y_HI = 0.44, 0.92
    curve = np.clip(mde, chance, 1.0)
    # Below the curve, between chance and the analytic minimum detectable
    # AUROC, is what this design could not resolve at the stated power.
    # COLOUR ROUND: the curve is an analytic bound, so it is BOUND_DEEP. One
    # hue on this canvas: the chance line, the guide and the two simulated
    # floors stay neutral, because none of them is the analytic bound.
    # VISUAL AUDIT. The shaded region used to be the one UNDER the curve, what
    # is not detectable, in BOUND_WASH, while the forest shades what IS
    # detectable: two figures a page apart whose shading meant opposite
    # things. This canvas now shades the region ABOVE the curve, in the MIST
    # the forest's bands use, so shading means "detectable at the stated
    # power" in both, and both regions keep the names they had. Under the
    # curve is bare paper. The change was made for that reading and not for
    # tools/measure_ink.py, whose coverage it happens to raise.
    ax.fill_between(n_pos, curve, Y_HI, facecolor=MIST,
                    edgecolor="none", linewidth=0, zorder=1)
    ax.plot(n_pos, curve, color=BOUND_DEEP, lw=2.2, zorder=4)
    ref_tone = SLATE    # reference and guide lines only; never type
    ax.axhline(chance, color=ref_tone, lw=LW_REF, linestyle=REF_DASH, zorder=3)
    ax.text(2400, (Y_LO + chance) / 2, f"chance, AUROC {chance:.2f}",
            ha="right", va="center", fontsize=T_NOTE, color=MUTE,
            style="italic", zorder=5)
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
    # The shaded region's name sits on the shading and says it is the shaded
    # one. GRAPHITE, 9.5 to 1 on MIST where MUTE was 4.85, and uncased like
    # the four other labels that now sit on the same field.
    ax.text(2800, 0.60, f"shaded: detectable at {pct} power", ha="right",
            va="bottom", fontsize=T_SERIES, color=GRAPHITE, zorder=5)
    # FIGURE POLISH. Was bottom-aligned 3pt over the chance line, so its
    # descenders straddled the dashes, and it could not move up because the
    # analytic-curve label sat in the same region. That label now joins the
    # other two above the curve (below), and this one is centred in the
    # region it names: 8pt over the chance line, 5pt under the curve.
    # It names the side the bound closes off, so it keeps the bound's DEEP,
    # now on bare paper.
    ax.text(14, 0.538, f"not detectable at {pct} power", ha="left",
            va="center", fontsize=T_SERIES, color=BOUND_DEEP, zorder=5)

    # The three points at this study's positive count, on the guide.
    # AUDIT ITEM 7. mde_study (0.7355) and floor_mw (0.7466) are close enough
    # that both markers, plotted at the same x, merged into one glyph at
    # print size. Dodged symmetrically off n_pos_study on the log axis; the
    # vertical guide line above still marks the true study size.
    # COLOUR ROUND. The open circle is a point ON the analytic curve, so its
    # ring and its label take the curve's tone. The two simulated floors are
    # not the curve and stay INK and GRAPHITE. Open ring, filled disc and
    # diamond are the three shapes they always were, so nothing here is told
    # apart by hue alone.
    # VISUAL AUDIT. The symmetric 1.045 dodge was 2.5pt each way under 6pt
    # markers, so the 0.7466 disc still sat on the 0.7355 ring, and it moved
    # the ring off the curve it is a point of. The ring and the diamond are
    # now ON the guide, at the study's own positive count, the ring on the
    # curve; only the Mann-Whitney disc is pulled aside, by X_PULL on the log
    # axis (about 8pt), and a hairline ties it back to the guide at its own
    # height so it cannot be read as a different sample size. Markers are a
    # point smaller, which leaves about 2pt of paper between ring and disc.
    # FOUR MARKS NOW, NOT THREE, because the corrected analytic floor and the
    # pre-registered threshold are two different kinds of thing and used to be
    # one mark. 0.7382 is a statement about this design's sensitivity and it
    # moves whenever the class counts do; 0.7355 is a historical fact about
    # what was registered and several checks were judged against, and it does
    # not move at all. They are 0.0027 apart, which is under a point of paper
    # at this scale, so they cannot share an x.
    #
    # The registered threshold is drawn as a short FIXED rule left of the
    # guide rather than as a point on the curve, because it is a constant and
    # not a function of the positive count. Its tone is GRAPHITE, the set's
    # non-entity measurement tone, not the curve's BOUND_DEEP: it is not a
    # point of the curve and must not read as one.
    X_PULL, X_PUSH = 1.15, 0.80
    ax.plot([n_pos_study], [mde_study], "o", ms=5.0, markerfacecolor=PAPER,
            markeredgecolor=BOUND_DEEP, markeredgewidth=1.3, zorder=7)
    ax.plot([n_pos_study * X_PUSH * 0.88, n_pos_study * X_PUSH * 1.14],
            [mde_registered_rule, mde_registered_rule], color=GRAPHITE,
            lw=1.8, zorder=8, solid_capstyle="butt")
    ax.plot([n_pos_study, n_pos_study * X_PULL], [floor_mw, floor_mw],
            color=INK, lw=0.6, zorder=6, solid_capstyle="butt")
    ax.plot([n_pos_study * X_PULL], [floor_mw], "o", ms=4.4, color=INK,
            markeredgewidth=0, zorder=8)
    ax.plot([n_pos_study], [floor_rule], "D", ms=5.0, color=INK,
            markeredgewidth=0, zorder=8)
    # Labels to the right, each joined to its point by a thin leader.
    # AUDIT ITEM 6. Two decimals collapsed 0.7466 to 0.75 and 0.8084 to 0.81
    # on this canvas, matching neither the caption (which prints four
    # decimals) nor each other; the labels already name which quantity is
    # which, so the fix is precision, not a new label.
    x_lab = 30.0
    for x_pt, y_pt, y_lab, text, tone in (
            (n_pos_study, floor_rule, floor_rule + 0.022,
             f"registered decision rule, power simulation: {floor_rule:.4f}",
             GRAPHITE),
            (n_pos_study * X_PULL, floor_mw, floor_mw + 0.030,
             f"Mann-Whitney test, power simulation: {floor_mw:.4f}", GRAPHITE),
            (n_pos_study, mde_study, mde_study - 0.022,
             f"analytic curve, two standard errors: {mde_study:.4f}",
             BOUND_DEEP),
            (n_pos_study * X_PUSH, mde_registered_rule,
             mde_registered_rule - 0.068,
             f"pre-registered threshold, fixed at {mde_registered_rule:.4f}",
             GRAPHITE)):
        ax.plot([x_pt * 1.10, x_lab * 0.95], [y_pt, y_lab],
                color=tone, lw=0.6, zorder=6, solid_capstyle="butt")
        ax.text(x_lab, y_lab, text, ha="left", va="center", fontsize=T_SERIES,
                color=tone, zorder=7)
    frame(ax)
    # Three lines, each measured against the canvas. Two lines put 96
    # characters of 11pt type on a 421pt sheet, savefig expanded the file to
    # 454pt and the whole figure would have printed at 0.935 scale.
    fig.canvas.draw()
    _rp = fig.canvas.get_renderer()

    def _hw(line):
        t = fig.text(0, 0, line, fontsize=T_HEADLINE)
        w = t.get_window_extent(_rp).width / fig.bbox.width
        t.remove()
        return w

    head = [f"At {n_pos_study} positives, {pct} power needs AUROC "
            f"{mde_study:.4f} analytically",
            f"and {floor_mw:.4f} under the simulated Mann-Whitney test; the "
            f"threshold",
            f"registered in advance, {mde_registered_rule:.4f}, is below both."]
    for line in head:
        assert 0.012 + _hw(line) < 0.996, \
            f"headline overflows the canvas: {line!r}"
    signature(fig, BOUND_DEEP, "\n".join(head))
    fig.savefig(OUTDIR / "fig-power.pdf"); plt.close(fig)
    print("wrote fig-power.pdf")


if __name__ == "__main__":
    make_fig_prevalence_response()
    make_fig_scaling()
    make_fig_forest()
    make_fig_power()
