"""Fig 5 prevalence response, Fig 7 power, Fig 8 forest, Fig 9 scaling.

Every value is READ from a committed result file. An earlier version typed the
C37 and C44 numbers as literals and drew the power curve from an ad-hoc formula
that did not reproduce the one real MDE it plotted -- the exact defect
corrections C7 and C10 retract elsewhere in this project. See C48.

All four figures follow the v3 system in style.py: boxed axes, direct labels
only, precomputed tints instead of alpha, and one saturated scarlet element per
axes at most. Figures 8 and 9 carry no scarlet at all, because Figure 8's claim
is a shape and Figure 9's claim is a success.
"""
import json, pathlib, glob, math, statistics as st
import numpy as np
import style
from style import (
    COBALT, AZURE, EMERALD, AMBER, PLUM, SCARLET,
    COBALT_DEEP, COBALT_CORE, COBALT_TINT, COBALT_WASH, COBALT_BARE,
    AZURE_DEEP, AZURE_CORE, EMERALD_DEEP, EMERALD_CORE, EMERALD_TINT,
    EMERALD_WASH, AMBER_DEEP, AMBER_CORE, AMBER_TINT, AMBER_WASH,
    PLUM_DEEP, PLUM_CORE, SCARLET_DEEP, SCARLET_CORE, SCARLET_TINT,
    SCARLET_WASH,
    PAPER, INK, GRAPHITE, MUTE, SLATE, MIST, HAIR,
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

PL = {"A_bare": "bare prompt", "B_baserate_abstain": "base rate stated,\nabstention allowed"}
# One hue per labeller, held down both rows of its column. granite3.3:8b takes
# AMBER: it used to draw in the green this set reserves for pharmacist truth,
# and a weak labeller wearing the truth colour is the kind of thing this thesis
# is about.
MODEL_HUE = {"llama3:8b": PLUM, "granite3.3:8b": AMBER, "qwen3:8b": COBALT}

# THE Y AXIS. It ran 0 to 1.0 and every measured point sat in the bottom
# quarter of it, which is why the author could not read this grid. The tallest
# point on the sweep is 0.17 and the highest Wilson bound anywhere on the canvas
# is 0.256, so nine tenths of the height was holding one off-sweep annotation.
#
# It now runs 0 to 0.38 against an x that runs 0 to 0.32. That single change is
# what makes this figure legible: the two axes are on very nearly the same
# scale, so PERFECT CALIBRATION IS A DIAGONAL AT ROUGHLY 45 DEGREES and a
# labeller that does not track prevalence is a line visibly flat against it. On
# the old axis the diagonal was a shallow ramp and flatness had no reference to
# be flat against. No point was dropped and no interval was clipped: the whole
# sweep, bounds included, clears the frame top by a third of the plot.
#
# What did not fit is the clinical operating point at 0.85, and it is drawn
# where it belongs -- at its own x, running off the top of the panel it belongs
# to, with both of its coordinates written beside it. An off-scale mark that
# names its own value states the reader's position honestly; squashing six
# panels to make room for one point does not.
Y_HI, X_HI = 0.38, 0.32
fig, axes = plt.subplots(2, 3, figsize=(FIGW, 3.78), sharex=True, sharey=True)
canvas(fig)
# 3.78in against the 3.60 a proportional growth gives: the extra 0.18 is what
# squares the panels up under a 0 to 0.38 y axis, and the block still lands at
# 5.22in, inside class B and clear of the forbidden 5.34.
fig.subplots_adjust(left=0.13735, right=0.98900, top=0.83642, bottom=0.11786,
                    wspace=0.14, hspace=0.20)
for r, pr in enumerate(prompts):
    for c, m in enumerate(models):
        ax = axes[r, c]
        xs, ys, lo, hi = [], [], [], []
        for p in ["0.02", "0.05", "0.1", "0.25"]:
            e = d["results"][m][pr].get(p)
            if not e:
                continue
            xs.append(float(p)); ys.append(e["positive_rate_of_decided"])
            # The sweep commits both counts behind every rate, so the interval
            # is a closed-form Wilson score interval on those counts.
            a, b = wilson_ci(e["n_contradiction"], e["n_decided"])
            lo.append(a); hi.append(b)
        hue = MODEL_HUE[m]
        # The wedge above the diagonal is over-prediction, and it is the whole
        # story of this grid, so it is drawn rather than left as white space.
        gx = np.linspace(0, X_HI, 2)
        ax.fill_between(gx, gx, Y_HI, facecolor=MIST, edgecolor="none",
                        linewidth=0, zorder=0)
        ax.plot([0, X_HI], [0, X_HI], color=SLATE, linestyle=REF_DASH,
                linewidth=LW_REF, zorder=2)
        # Heavier than the set's 1.8pt data line and 6.6pt dot, and the only
        # place in the set that departs from that table. The author asked this
        # figure for vibrancy; no hue on it could be brightened without costing
        # the greyscale separation the whole palette is verified on (style.py
        # records the arithmetic), so the vibrancy is bought in WEIGHT instead.
        # A 2.4pt line and a 7.4pt marker faced in CORE and rimmed in DEEP put
        # two saturated stops on every point of every series.
        ax.errorbar(xs, ys, yerr=[[y - a for y, a in zip(ys, lo)],
                                  [b - y for y, b in zip(ys, hi)]],
                    fmt="-o", color=hue.CORE, ms=7.4, lw=2.4, zorder=4,
                    markerfacecolor=hue.CORE, markeredgecolor=hue.DEEP,
                    markeredgewidth=1.0, ecolor=hue.CORE,
                    elinewidth=1.6, capthick=1.6, capsize=3.6)
        if r == 0 and c == 0:
            # The clinical operating point. Its true position is (0.04, 0.85),
            # which is off the top of a panel whose highest measured bound is
            # 0.256, so it is drawn as what it is: a mark at its own x, leaving
            # the panel upward, carrying both of its coordinates in its label.
            # No marker sits under the arrow. A square drawn at a y the point
            # does not have would be a false position however it is labelled,
            # and this figure is in a thesis about instruments that mislead.
            # The arrow starts above the highest bound in the panel and leaves
            # through the top.
            arrow(ax, (clinical_x, 0.272), (clinical_x, Y_HI - 0.010),
                  color=SCARLET_CORE, lw=2.4, zorder=6)
            case(ax.text(X_HI - 0.014, 0.348,
                         f"clinical: {clinical_x * 100:.0f}% read as "
                         f"{clinical_y * 100:.0f}%",
                         ha="right", va="center", fontsize=T_SERIES,
                         color=SCARLET_DEEP, zorder=7))
        if r == 0 and c == 2:
            case(ax.text(X_HI - 0.005, Y_HI - 0.012, "over-prediction",
                         ha="right", va="top", fontsize=T_NOTE, color=MUTE,
                         zorder=5))
        if r == 1 and c == 2:
            # The diagonal's angle on the page, not 45 degrees: the panel is
            # not square and the two axes do not share a range. It is close to
            # 45 now, which is the point of the new axis.
            bb = ax.get_window_extent()
            ang = math.degrees(math.atan2(X_HI / Y_HI * bb.height,
                                          bb.width))
            # MUTE, not the SLATE of the line it names: SLATE is 3.48:1 on
            # white and this set does not set type in it.
            case(ax.text(0.150, 0.156, "perfect calibration", fontsize=T_NOTE,
                         color=MUTE, ha="center", va="bottom", rotation=ang,
                         rotation_mode="anchor", zorder=5))
        ax.set_xlim(0, X_HI); ax.set_ylim(0, Y_HI)
        ax.set_xticks([0, 0.1, 0.2, 0.3]); ax.set_yticks([0, 0.1, 0.2, 0.3])
        if r == 0:
            ax.set_title(m, fontsize=T_SERIES, pad=5, color=MODEL_HUE[m].DEEP)
        if c == 0:
            ax.set_ylabel(PL[pr], fontsize=T_TICK, linespacing=1.3)
        ax.tick_params(labelsize=T_TICK)
        frame(ax)
fig.supxlabel("True prevalence", fontsize=T_AXIS, y=0.018)
fig.supylabel("Labeller predicted-positive rate", fontsize=T_AXIS, x=0.006)
signature(fig, SCARLET_DEEP,
          "No labeller tracks prevalence. Stating the base rate does not fix it.")
fig.savefig(HERE / "fig-prevalence-response.pdf"); plt.close(fig)
print("wrote fig-prevalence-response.pdf")

# =============================================================================
# Fig 9 -- fig-scaling. Panel (b) deleted: its content is Figure 8's last row.
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
# =============================================================================
iv = json.loads((ROOT / "results/accord/stats/intervals.json").read_text())["E2_mcc_intervals"]
NICE = {"gpt4o_zeroshot": "GPT-4o zero-shot", "gpt4o_fewshot": "GPT-4o few-shot",
        "accord_binary_at_0.5_253core": "ACCORD at 0.5",
        "llama_baseline": "Llama 3 8B"}
# Rows take their ENTITY hue, which they did not before: family 1's four rows
# all drew one blue, contradicting the graphical abstract 85 pages earlier.
ROW_HUE = {"gpt4o_zeroshot": COBALT, "gpt4o_fewshot": AZURE,
           "accord_binary_at_0.5_253core": AMBER, "llama_baseline": PLUM}
fam_mcc = [((NICE.get(k, k) + f" ({v['n']} pairs)") if k.startswith("accord")
            else NICE.get(k, k), v["mcc"], v["ci95"][0], v["ci95"][1],
            ROW_HUE[k]) for k, v in iv.items()]
_dec = json.loads((ROOT / "results/accord/decontam/corrected_gold_metrics.json").read_text())["aggregate"]
_ctrl = json.loads((ROOT / "results/accord/control/final_report.json").read_text())
_df34 = _ctrl["ensemble_mean_of_seed_scores"]["gold_defect_free_rule34"]
fam_auroc = [
    ("Decontaminated ensemble", _dec["ensemble_auroc"] - 0.5,
     _dec["ensemble_ci95"][0] - 0.5, _dec["ensemble_ci95"][1] - 0.5, AMBER),
    ("Defect-free stratum", _df34["auroc"] - 0.5,
     _df34["bootstrap"]["ci95"][0] - 0.5, _df34["bootstrap"]["ci95"][1] - 0.5,
     EMERALD),
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
fam_gap = [
    ("In-hospital mortality, AUROC", _g, _g - 1.96 * _se, _g + 1.96 * _se, EMERALD),
    ("AG News at 100M, accuracy", _g44, _g44 - 1.96 * _se44, _g44 + 1.96 * _se44,
     EMERALD),
]
# A family is not an entity, so the family band is a structural ground and the
# family heading is set in the annotation grey, not in a fake hue.
FAMILIES = [("Matthews correlation", fam_mcc, MIST),
            ("AUROC above chance", fam_auroc, EMERALD_WASH),
            ("Pretraining gap, pretrained minus from scratch", fam_gap, AMBER_WASH)]
for _, rs, _b in FAMILIES:
    rs.sort(key=lambda r: r[1])

placed, bands_, heads, y = [], [], [], 0.0
for fi, (fam, rs, bg) in enumerate(reversed(FAMILIES)):
    y0 = y - 0.55
    for r in rs:
        placed.append((y, r)); y += 1.0
    y -= 1.0
    heads.append((y + 0.52, fam))
    bands_.append((y0, y + 1.05, bg))
    y += 1.55
rows = [r for _, r in placed]
y_ceiling = placed[-1][0] + 1.10

fig, ax = plt.subplots(figsize=(FIGW, 3.71))
canvas(fig)
fig.subplots_adjust(left=0.2976, right=0.9890, top=0.8783, bottom=0.1173)
X_LO, X_HI = -0.40, 1.06
ax.set_xlim(X_LO, X_HI)
ax.set_ylim(-0.80, y_ceiling)
ax.set_xticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
frame(ax, grid="x")
for y0, y1, bg in bands_:
    ax.axhspan(y0, y1, facecolor=bg, edgecolor="none", linewidth=0, zorder=0)
# The zero line is the spine of this figure, not a reference to be looked past.
ax.axvline(0, color=GRAPHITE, lw=1.2, zorder=2)

# The drawn half-width of an interval, in points, decides whether the sliver
# rule fires. Anything under 1.5pt is invisible beside its own marker, and an
# invisible interval with no cue is the failure this set's first rule exists to
# prevent.
plot_w_pt = (0.9890 - 0.2976) * FIGW * 72
sliver = []
for yy, (lab, e, lo, hi, hue) in placed:
    half_pt = (hi - lo) / 2 / (X_HI - X_LO) * plot_w_pt
    ax.errorbar([e], [yy], xerr=[[e - lo], [hi - e]], fmt="o", color=hue.CORE,
                zorder=5, **ERRBAR, **DOT)
    mark = ""
    if half_pt < 1.5:
        for b in (lo, hi):
            ax.plot([b, b], [yy - 0.19, yy + 0.19], color=hue.DEEP, lw=1.5,
                    solid_capstyle="butt", zorder=6)
        mark = "‡ "
        sliver.append(lab)
    ax.text(X_HI - 0.02, yy, f"{mark}{e:+.4f}  [{lo:+.3f}, {hi:+.3f}]",
            ha="right", va="center", fontsize=T_SERIES, color=hue.DEEP,
            zorder=6)

gt = ax.get_yaxis_transform()
ax.set_yticks([yy for yy, _ in placed])
ax.set_yticklabels([r[0] for r in rows], fontsize=T_SERIES, linespacing=1.25)
for yh, fam in heads:
    ax.text(-0.385, yh, fam, ha="left", va="bottom", fontsize=T_SERIES,
            color=MUTE, style="italic", zorder=4)
ax.set_xlabel("Effect, 95% interval (unit named per family)")
ax.tick_params(axis="y", length=0)
if sliver:
    ax.text(-0.385, -0.44, "‡ interval narrower than the marker",
            ha="left", va="center", fontsize=T_NOTE, color=MUTE, zorder=6)
n_cross = sum(1 for r in rows if r[2] <= 0 <= r[3])
signature(fig, GRAPHITE,
          f"{n_cross} of the {len(rows)} headline estimates in this thesis "
          f"cross zero.")
fig.savefig(HERE / "fig-forest.pdf"); plt.close(fig)
print(f"wrote fig-forest.pdf ({n_cross}/{len(rows)} cross zero; "
      f"sliver rows: {sliver})")

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
ax.fill_between(n_pos, curve, 1.0, facecolor=EMERALD_WASH, edgecolor="none",
                linewidth=0, zorder=1)
ax.fill_between(n_pos, 0.5, curve, facecolor=SCARLET_WASH, edgecolor="none",
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
case(ax.text(6.0, 0.515, "not detectable at this sample size",
             fontsize=T_SERIES, color=SCARLET_DEEP, ha="left", va="bottom",
             zorder=5))
# Both marks are labelled adjacent to themselves. No leaders anywhere.
ax.plot([mort_n_pos], [mde_mort], "s", color=SLATE, ms=MS_MARK,
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
          f"At {n_pos_study} positives, nothing below AUROC {mde_study:.3f} "
          f"was ever detectable.")
fig.savefig(HERE / "fig-power.pdf"); plt.close(fig)
print("wrote fig-power.pdf")
