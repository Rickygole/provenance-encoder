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
    Y_HI, X_HI = 0.32, 0.32
    SWEEP_P = ["0.02", "0.05", "0.1", "0.25"]

    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 3.46), sharey=True)
    canvas(fig)
    fig.subplots_adjust(left=0.0930, right=0.9830, top=0.7450, bottom=0.1200,
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
            case(ax.text(0.100, Y_HI - 0.010, "over-prediction",
                         ha="left", va="top", fontsize=T_NOTE, color=MUTE,
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
        ax.set_yticks([0, 0.1, 0.2, 0.3])
        ax.set_title(PL[pr], fontsize=T_SERIES, pad=5, color=INK)
        ax.tick_params(labelsize=T_TICK)
        frame(ax, grid="y")
    fig.supxlabel("True prevalence", fontsize=T_AXIS, y=0.016)
    axes[0].set_ylabel("Labeller predicted-positive rate", fontsize=T_AXIS)
    signature(fig, SCARLET_DEEP,
              f"On public text, {n_under} of {n_cells} labeller settings under-call,\n"
              f"and the largest over-call is {top[3]:.1f} times.",
              deck=f"Not drawn: the clinical pool's {clin_lo:.4f} to {clin_hi:.4f} "
                   "times, from a different pool, prompt and negative.")
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
    au = [
        ("Ensemble, 253 non-ambiguous pairs",
         fact("decontam.aggregate.ensemble_auroc"),
         fact("decontam.aggregate.ensemble_ci95_lo"),
         fact("decontam.aggregate.ensemble_ci95_hi")),
        ("Ensemble, Rules 3 and 4, 108 pairs",
         fact("c114.task_c.stratum_auroc"),
         fact("c114.task_c.ci95_lo"), fact("c114.task_c.ci95_hi")),
    ]
    assert fact("c114.task_c.n") == 108 and fact("c114.task_c.n_pos") == 6
    assert all(lo <= 0.5 <= hi for _, _, lo, hi in au), \
        "an AUROC interval now excludes chance; the headline would be false"
    mcc_obs = fact("c120.g299.mcc")
    mcc_ceil = fact("c120.g299.ceiling_mcc")
    mcc_p = fact("c120.g299.p_hypergeom_one_sided")
    assert mcc_p > 0.05

    H = 2.80
    fig = plt.figure(figsize=(FIGW, H))
    canvas(fig)
    X_LEFT, X_RIGHT, X_VALUE = 0.3350, 0.7050, 0.9950
    A0 = 1.22
    ax_a = fig.add_axes([X_LEFT, A0 / H, X_RIGHT - X_LEFT, 0.62 / H])
    ax_m = fig.add_axes([X_LEFT, 0.36 / H, X_RIGHT - X_LEFT, 0.36 / H])

    # AUROC panel: chance at 0.5 is the zero rule.
    ax_a.set_xlim(0.25, 1.0)
    ax_a.set_ylim(-0.8, 1.6)
    ax_a.set_xticks([0.3, 0.5, 0.7, 0.9])
    frame(ax_a, grid="x")
    ax_a.axvline(0.5, color=GRAPHITE, lw=1.2, zorder=2)
    ax_a.text(0.492, -0.74, "chance", ha="right", va="bottom", fontsize=T_NOTE,
              color=MUTE, style="italic")
    for i, (lab, e, lo, hi) in enumerate(au):
        y = len(au) - 1 - i
        # AMBER is the entity hue style.py gives ACCORD, and this detector is
        # ACCORD retrained subject-disjoint; the MCC row below takes PLUM, the
        # production weak labeller's hue. The hues repeat the set's entity map
        # and carry nothing a greyscale reader needs: each row is named.
        ax_a.errorbar([e], [y], xerr=[[e - lo], [hi - e]], fmt="o",
                      color=AMBER_DEEP, markerfacecolor=AMBER_CORE,
                      markeredgecolor=AMBER_DEEP, zorder=5,
                      **ERRBAR, **DOT)
        fig.text(X_VALUE, (A0 + 0.62 * (y + 0.8) / 2.4) / H,
                 f"{e:.4f}  [{lo:.4f}, {hi:.4f}]", ha="right", va="center",
                 fontsize=T_SERIES, color=AMBER_DEEP)
    ax_a.set_yticks([len(au) - 1 - i for i in range(len(au))])
    ax_a.set_yticklabels([r[0] for r in au], fontsize=T_SERIES)
    ax_a.tick_params(axis="y", length=0)
    ax_a.tick_params(axis="x", labelsize=T_TICK)
    fig.text(0.012, (A0 + 0.62 + 0.10) / H,
             "AUROC, decontaminated detector, 95% bootstrap interval",
             ha="left", va="bottom", fontsize=T_SERIES, color=MUTE,
             style="italic")

    # MCC panel: zero is chance; the ceiling is a bound, drawn as a tick.
    ax_m.set_xlim(-0.02, 0.10)
    ax_m.set_ylim(-0.6, 0.6)
    ax_m.set_xticks([0.0, 0.03, 0.06, 0.09])
    frame(ax_m, grid="x")
    ax_m.axvline(0.0, color=GRAPHITE, lw=1.2, zorder=2)
    ax_m.plot([mcc_ceil, mcc_ceil], [-0.38, 0.38], color=PLUM_DEEP, lw=2.0,
              solid_capstyle="butt", zorder=4)
    ax_m.plot([0.0, mcc_ceil], [0, 0], color=HAIR, lw=3.0, zorder=3,
              solid_capstyle="butt")
    ax_m.plot([mcc_obs], [0], "o", color=PLUM_CORE, markeredgecolor=PLUM_DEEP,
              zorder=5, **DOT)
    ax_m.text(mcc_ceil + 0.002, 0.30, "ceiling", ha="left", va="top",
              fontsize=T_NOTE, color=MUTE, style="italic")
    ax_m.set_yticks([0])
    ax_m.set_yticklabels(["Deployed labeller, 299 pairs"], fontsize=T_SERIES)
    ax_m.tick_params(axis="y", length=0)
    ax_m.tick_params(axis="x", labelsize=T_TICK)
    fig.text(X_VALUE, (0.36 + 0.18) / H,
             f"{mcc_obs:.4f}  ceiling {mcc_ceil:.4f}\nexact p {mcc_p:.4f}",
             ha="right", va="center", fontsize=T_SERIES, color=PLUM_DEEP,
             linespacing=1.25)
    fig.text(0.012, (0.36 + 0.36 + 0.10) / H,
             "MCC against the pharmacist, exact count, no resampling interval",
             ha="left", va="bottom", fontsize=T_SERIES, color=MUTE,
             style="italic")
    # ROUND 8: was 1.06 / H, which ran through the AUROC tick labels.
    fig.add_artist(plt.Line2D([0.012, X_VALUE], [1.00 / H] * 2,
                              color=PLATE_EDGE, lw=0.6))

    # ROUND 8 (committee). "At 12 positives" was false for the Rules 3 and 4
    # row, which rests on 6. Both counts are read and named.
    n_pos_253 = fact("decontam.n_positives")
    assert n_pos_253 == fact("c120.g299.n_positives")
    signature(fig, GRAPHITE,
              f"At {n_pos_253} positives, or {fact('c114.task_c.n_pos')} for Rules 3 "
              "and 4, no headline estimate\nis distinguishable from chance.",
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
    pw = json.loads((ROOT / "results/accord/section6_auroc_power.json").read_text())
    n_pos_study, n_neg_study = pw["n_positives"], pw["n_negatives"]
    alpha_study, power_study = pw["alpha"], pw["power"]


    # The C108 floors, read by key, and the analytic point checked against the
    # register rather than trusted from this file's own bisection.
    assert fact("sizectl108.power_curves.n_pos_slots") == n_pos_study
    assert fact("sizectl108.power_curves.n_neg_slots") == n_neg_study
    floor_mw = fact("sizectl108.power_curves.label_mix.crossings_at_power_0_80.mw_two_sided")
    floor_rule = fact("sizectl108.power_curves.label_mix.crossings_at_power_0_80.c98_rule")
    assert floor_mw == fact("sizectl108.power_curves.primary_crossing_power_0_80")
    n_pos = np.logspace(np.log10(5), np.log10(3000), 60)
    n_neg = n_pos * (n_neg_study / n_pos_study)
    mde = hanley_mde(n_pos, n_neg, alpha_level=alpha_study, power=power_study)
    mde_study = hanley_mde([n_pos_study], [n_neg_study], alpha_level=alpha_study,
                           power=power_study)[0]
    assert abs(mde_study - fact("sizectl108.power_curves.mde_analytic")) < 5e-5

    fig, ax = plt.subplots(figsize=(FIGW, 2.98))
    canvas(fig)
    fig.subplots_adjust(left=0.1172, right=0.9890, top=0.7760, bottom=0.1560)
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
    case(ax.text(2700, 0.965, "80% power or more (analytic)", fontsize=T_SERIES,
                 color=EMERALD_DEEP, ha="right", va="top", zorder=5))
    # INK, not SCARLET_DEEP: on the new SCARLET_TINT ground scarlet type measures
    # 4.93:1, which the set's own advisory calls thin, and INK measures 11.3:1.
    case(ax.text(6.0, 0.515, "under 80% power (analytic)",
                 fontsize=T_SERIES, color=INK, ha="left", va="bottom",
                 zorder=5))
    # ROUND 8 (committee). The C44 mortality control point and its label are
    # gone. C44 moved to the supplement because its registration forbids
    # reading it against this evaluation, and a square on the same curve beside
    # the evaluation marker was exactly that reading. results/task_runs.json is
    # no longer read by this figure.
    # ROUND 7 (fixes_r7 N9). The old headline, "nothing below AUROC 0.7355 was ever
    # detectable", read an 80 percent power point as a detection threshold. The
    # claim the thesis makes is the C108 per-rule floor: on a real score
    # distribution at these class counts, 80 percent power arrives at 0.7466 for a
    # two-sided Mann-Whitney test and at 0.8084 for the registered decision rule.
    # Both are drawn at this study's positive count, on the analytic curve's own
    # axes, and the analytic value is asserted to be the register's.
    ax.plot([n_pos_study], [mde_study], "o", color=GRAPHITE, ms=MS_SEED + 1.4,
            markeredgewidth=0, zorder=6)
    ax.plot([n_pos_study], [floor_mw], "o", color=SCARLET_CORE, ms=MS_DOT + 0.6,
            markeredgecolor=SCARLET_DEEP, markeredgewidth=1.0, zorder=7)
    ax.plot([n_pos_study], [floor_rule], "D", color=SCARLET_DEEP, ms=MS_MARK,
            markeredgewidth=0, zorder=7)
    case(ax.text(n_pos_study * 1.35, floor_rule, f"registered rule, {floor_rule:.4f}",
                 fontsize=T_SERIES, color=SCARLET_DEEP, ha="left", va="center",
                 zorder=7))
    case(ax.text(n_pos_study * 1.35, floor_mw + 0.004,
                 f"Mann-Whitney, {floor_mw:.4f}", fontsize=T_SERIES,
                 color=SCARLET_DEEP, ha="left", va="bottom", zorder=7))
    case(ax.text(n_pos_study * 0.86, mde_study - 0.022,
                 f"analytic\n{mde_study:.4f}", fontsize=T_NOTE, color=INK,
                 ha="right", va="top", zorder=7, linespacing=1.15))
    frame(ax)
    signature(fig, SCARLET_DEEP,
              f"At {n_pos_study} positives, 80 percent power needs AUROC "
              f"{floor_mw:.4f} under Mann-Whitney\nand {floor_rule:.4f} under the "
              f"registered rule.")
    fig.savefig(HERE / "fig-power.pdf"); plt.close(fig)
    print("wrote fig-power.pdf")


if __name__ == "__main__":
    make_fig_prevalence_response()
    make_fig_scaling()
    make_fig_forest()
    make_fig_power()
