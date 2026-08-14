"""F5 prevalence response, F6 scaling, F7 forest, F10 power landscape.

Every value is READ from a committed result file. An earlier version typed the
C37 and C44 numbers as literals and drew the power curve from an ad-hoc formula
that did not reproduce the one real MDE it plotted -- the exact defect
corrections C7 and C10 retract elsewhere in this project. See C48.

All four figures follow the shared grammar in style.py: one measure, dots with
intervals for estimates, bars only for counts, grey hairline reference lines
labelled at the line, and one scarlet element per canvas."""
import json, pathlib, glob, math, statistics as st
import numpy as np
from style import (setup, FIGW, INK, BLUE, PALE, TEAL, SCARLET, MUTE, GREY, GRID,
                   BLUE2, GREEN, ORANGE, PURPLE, SKY,
                   BLUE2_T, GREEN_T, ORANGE_T, PURPLE_T, SKY_T,
                   ACCENT, MUTED, LIGHT, DEAD, ALERT,
                   CHANCE_STYLE, ERRBAR, DOT, canvas, lean, wilson_ci)
setup()
import matplotlib.pyplot as plt

# Long provenance notes belong in the LaTeX caption, not inside the plot.
# Set True to draw them again.
SHOW_INFIGURE_NOTES = False
SHOW_INFIGURE_TITLES = False

HERE = pathlib.Path(__file__).parent
ROOT = pathlib.Path("/Users/rickygole/provenance-encoder")

# ---------- F5: prevalence response -----------------------------------------
d = json.loads((ROOT/"accord/prevalence_replication/results/sweep_results.json").read_text())
models, prompts = d["models"], ["A_bare", "B_baserate_abstain"]
# The clinical comparison point was typed here as (0.04, 0.849). It is the same
# weak-label-versus-pharmacist shift figures 1 and 2 draw, so it is recomputed
# from the same two diagnostic files instead, which is also what the notebook does.
_x01 = json.loads((ROOT/"results/accord/diag/step0_1_join_and_calibration.json").read_text())
_xtab = _x01["weak_confidence_is_degenerate"]["crosstab_label_x_confidence"]
clinical_y = _xtab["0.95"]["contradiction"] / (_xtab["0.95"]["contradiction"]
                                               + _xtab["1.0"]["not_contradiction"])
clinical_x = json.loads((ROOT/"results/accord/diag/step2_signal_check.json").read_text())[
    "contra_prob_vs_pharm_label__ambiguous_as_negative_n299"]["base_rate"]
# A small-multiple grid: the row and column headings identify every panel, so
# panel letters would be a third naming of the same six cells.
PL = {"A_bare": "bare prompt", "B_baserate_abstain": "base rate stated,\nabstention allowed"}
# One hue per labeller, held down both rows of its column, so a column reads as
# one model under two prompts rather than as six unrelated panels. llama3:8b is
# the thesis's own production weak labeller, so it carries the same hue here
# that the weak-labeller bars and the Llama 3 8B baseline carry elsewhere.
MODEL_HUE = {"llama3:8b": PURPLE, "granite3.3:8b": GREEN, "qwen3:8b": BLUE2}
MODEL_HUE_T = {"llama3:8b": PURPLE_T, "granite3.3:8b": GREEN_T, "qwen3:8b": BLUE2_T}
fig, axes = plt.subplots(2, 3, figsize=(FIGW, 2.60), sharex=True, sharey=True)
canvas(fig)
fig.subplots_adjust(left=0.185, right=0.995, top=0.885, bottom=0.175,
                    wspace=0.16, hspace=0.22)
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
            # is a closed-form Wilson score interval on those counts: no
            # resampling, no simulation, nothing invented. A rate printed on
            # 100 decisions without one invites the reader to read a two-point
            # wobble as a response to prevalence.
            a, b = wilson_ci(e["n_contradiction"], e["n_decided"])
            lo.append(a); hi.append(b)
        hue = MODEL_HUE[m]
        ax.plot([0, .30], [0, .30], **CHANCE_STYLE)
        ax.errorbar(xs, ys, yerr=[[y - a for y, a in zip(ys, lo)],
                                  [b - y for y, b in zip(ys, hi)]],
                    fmt="-o", color=hue, ms=3.0, lw=1.1, zorder=3,
                    markeredgewidth=0, ecolor=hue, elinewidth=0.8,
                    capthick=0.8, capsize=2.0)
        # The clinical operating point is the claim this whole grid is drawn
        # against, so it is the one accented mark, and it appears once. Six
        # copies of one anchor read as six separate findings and put a scarlet
        # square in every panel.
        if r == 0 and c == 0:
            ax.plot([clinical_x], [clinical_y], "s", color=SCARLET, ms=3.6, zorder=4)
            ax.annotate(f"clinical\n({clinical_x * 100:.0f}% read as {clinical_y * 100:.0f}%)",
                        (clinical_x, clinical_y), xytext=(0.082, 0.52),
                        fontsize=8.0, color=SCARLET, linespacing=1.25,
                        arrowprops=dict(arrowstyle="-", lw=.5, color=SCARLET))
        if r == 0 and c == 2:
            ax.text(0.295, 0.34, "perfect\ncalibration", fontsize=8.0, color=MUTE,
                    ha="right", va="bottom", linespacing=1.25)
        # Bounds and ticks run past every mark and every cap, so nothing on the
        # grid hangs off the end of the scale it is measured on.
        ax.set_xlim(0, .32); ax.set_ylim(0, .99)
        ax.set_xticks([0, 0.1, 0.2, 0.3]); ax.set_yticks([0, 0.3, 0.6, 0.9])
        # Panel heading, not a figure title: it names which labeller this column
        # is, and it is set in that labeller's own hue, darkened to the value 8
        # point type needs on white. That heading is also what carries the
        # column apart in a greyscale print, where three of these hues are the
        # same grey.
        if r == 0:
            ax.set_title(m, fontsize=8.0, pad=4, color=MODEL_HUE_T[m])
        if c == 0: ax.set_ylabel(PL[pr], fontsize=8.0, linespacing=1.3)
        ax.tick_params(labelsize=8)
        lean(ax, xbounds=(0, 0.30), ybounds=(0, 0.9))
fig.supxlabel("True prevalence", fontsize=9.0, y=0.020)
fig.supylabel("Labeller predicted-positive rate", fontsize=9.0, x=0.008)
if SHOW_INFIGURE_TITLES: fig.suptitle("The labeller under-responds to prevalence, but nowhere near the clinical shift",
             fontsize=9.5, x=0.02, ha="left", y=0.99)
fig.savefig(HERE/"fig-prevalence-response.pdf"); plt.close(fig); print("wrote fig-prevalence-response.pdf")

# ---------- F6: scaling with clinical overlay --------------------------------
sc = json.loads((ROOT/"results/accord/scaling/scaling_results.json").read_text())
by, pr_ = {}, {}
for x in sc:
    by.setdefault(x["tag"], []).append(x["accuracy"]); pr_[x["tag"]] = x["non_emb_params"]
tags = ["scratch_10M", "scratch_30M", "scratch_100M"]
xs = [pr_[t] for t in tags]; mu = [st.mean(by[t]) for t in tags]; sd = [st.stdev(by[t]) for t in tags]
pm, ps, pp = st.mean(by["pretrained_bert"]), st.stdev(by["pretrained_bert"]), pr_["pretrained_bert"]

# C37 clinical comparison, read from the run records -- never typed
_pb = [json.loads((ROOT/f"local_only/task_runs/concat_s{s}.json").read_text())["auroc"] for s in (1,2,3)]
_ou = [json.loads(pathlib.Path(f).read_text())["auroc"]
       for f in sorted(glob.glob(str(ROOT/"local_only/task_runs/ours_current_s*.json")))]
_pbm, _pbs = st.mean(_pb), st.stdev(_pb)
_oum, _ous = st.mean(_ou), st.stdev(_ou)
_gap = _pbm - _oum

# Two panels, not a panel with a floating inset. The clinical comparison is a
# second experiment on a second task with a second metric; an inset says
# "detail of the panel behind me", which this is not, and it borrowed the host
# panel's frame so it had no way to name its own unit. As a panel it has its
# own frame, its own axis label naming the metric and the task, and its own
# letter.
fig, (ax, axb) = plt.subplots(2, 1, figsize=(FIGW, 3.34),
                              gridspec_kw={"height_ratios": [2.00, 1.0]})
canvas(fig)
fig.subplots_adjust(left=0.175, right=0.985, top=0.965, bottom=0.115, hspace=0.66)
# Two arms, two hues. The pretrained arm is the reference this figure measures
# against, but it is a measured arm with three seeds and a spread of its own,
# not a chance line, so it is no longer drawn in the grey this set reserves for
# structure: it takes the hue every other reference-standard arm in the set
# takes, and its band, its line and its label are all that one hue, because an
# interval belongs to the mark it came from.
ax.axhspan(pm-ps, pm+ps, color=GREEN, alpha=0.22, linewidth=0, zorder=1)
ax.axhline(pm, color=GREEN, lw=0.8, zorder=2)
ax.text(1.62e8, pm+0.0030, f"pretrained BERT {pm:.4f} ± {ps:.4f} ({pp/1e6:.0f}M)",
        fontsize=8.0, color=GREEN_T, va="bottom", ha="right")
# Every seed that went into each mean is drawn behind it, on both panels, in
# the same small grey dot: the caps here are the seed standard deviation of
# three committed runs, not an analytic bound, so the runs themselves are on
# the canvas. The seeds share the mean's x (they are the same model size), so
# the interval is drawn first and the seeds over it: drawn under the interval
# they were hidden by its own rule and the panel read as caps with no seeds.
# The mean is drawn last and still sits in front, so where the three seeds
# agree to within the width of a marker they are simply the mark itself.
ax.errorbar(xs, mu, yerr=sd, fmt="none", ecolor=BLUE2, zorder=3, **ERRBAR)
for x, t in zip(xs + [pp], tags + ["pretrained_bert"]):
    ax.plot([x] * len(by[t]), by[t], "o", color=GREY, ms=2.6, markeredgewidth=0,
            alpha=0.65, zorder=4)
ax.plot(xs, mu, "-o", color=BLUE2, ms=4.0, lw=1.1, zorder=5)
# The gap labels sit below their points; the x limits are set wide enough that the
# leftmost one is not cut off by the axis, as it was before.
for x, m, s, t in zip(xs, mu, sd, tags):
    ax.annotate(f"{pm-m:+.4f}", (x, m - s), textcoords="offset points", xytext=(0, -10),
                ha="center", va="top", fontsize=8.0, color=MUTE)
ax.set_xscale("log"); ax.set_xlabel("Non-embedding parameters (log scale)")
ax.set_ylabel("Accuracy, AG News"); ax.set_ylim(0.851, 0.950)
# The top tick has to reach past the pretrained reference. Ticked to 0.93 the
# scale stopped below the band this panel is measured against, so the reference
# line and its own seeds sat above the last value the axis named. Even steps,
# so the ticks still read as one scale rather than a regular ladder with one
# odd rung added at the top to reach the reference.
ax.set_yticks([0.86, 0.88, 0.90, 0.92, 0.94])
ax.set_xlim(7.0e6, 1.7e8)
lean(ax, ybounds=(0.86, 0.941))
if SHOW_INFIGURE_TITLES: ax.set_title("Pretraining is worth five points here, and more parameters do not close it",
             loc="left", fontsize=9.5, pad=8)

# (b) The clinical arms. Same grammar and the same two hues as panel (a): the
# from-scratch arm blue, the pretrained arm the reference hue, each mean with
# its three seeds drawn behind it in grey. The two panels are two tasks, so
# holding the arm colours across them is the only thing on the canvas saying
# that the same comparison is being made twice. No accent: the claim here is
# the size of the gap, which is a relation between two marks rather than either
# of them.
for i, (vals, mean, sdv, colour) in enumerate(
        ((_ou, _oum, _ous, BLUE2), (_pb, _pbm, _pbs, GREEN))):
    axb.plot(vals, [i] * len(vals), "o", color=GREY, ms=2.6, markeredgewidth=0,
             alpha=0.65, zorder=2)
    axb.errorbar([mean], [i], xerr=[sdv], fmt="o", color=colour, ms=4.0,
                 ecolor=colour, zorder=3, **ERRBAR)
axb.set_yticks([0, 1])
axb.set_yticklabels(["from scratch", "PubMedBERT"], fontsize=8.0)
# Each row label is the direct label for its own arm, so it is set in that
# arm's colour at the value type needs. That label is also what separates the
# two arms in a greyscale print.
for lab, c in zip(axb.get_yticklabels(), (BLUE2_T, GREEN_T)):
    lab.set_color(c)
# Bounds and ticks reach past the outermost seed (0.7578) and the widest
# cap, so nothing in the panel sits off the end of its own scale.
axb.set_ylim(-0.75, 1.75); axb.set_xlim(0.7455, 0.8045)
axb.set_xticks([0.75, 0.76, 0.77, 0.78, 0.79, 0.80])
axb.set_xlabel("AUROC, in-hospital mortality task")
axb.text(0.8040, 1.66, f"gap {_gap:+.4f}", fontsize=8.0, color=MUTE,
         ha="right", va="top")
axb.tick_params(labelsize=8)
lean(axb, xbounds=(0.75, 0.80))
axb.spines["left"].set_visible(False)
axb.tick_params(axis="y", length=0)

for a_, tag in ((ax, "(a)"), (axb, "(b)")):
    p = a_.get_position()
    fig.text(0.004, p.y1 + 0.008, tag, ha="left", va="bottom",
             fontsize=8.0, style="italic", color=INK)
fig.savefig(HERE/"fig-scaling.pdf"); plt.close(fig); print("wrote fig-scaling.pdf")

# ---------- F7: forest of every headline claim -------------------------------
iv = json.loads((ROOT/"results/accord/stats/intervals.json").read_text())["E2_mcc_intervals"]
# The register keys are turned into readable names here. Presentation only: the
# same four systems, spelled the way the rest of the thesis spells them.
NICE = {"gpt4o_zeroshot": "GPT-4o zero-shot", "gpt4o_fewshot": "GPT-4o few-shot",
        "accord_binary_at_0.5_253core": "ACCORD at 0.5",
        "llama_baseline": "Llama 3 8B"}
# Eight estimates in three families. They were previously eight rows on one
# sorted position axis, which put a Matthews correlation, an AUROC and an
# accuracy gap in a single ranking and invited the reader to compare the length
# of one against the length of another. They do not share a unit and cannot be
# compared that way. Grouping them, naming the unit once per group and ruling a
# hairline between the groups says what the axis actually supports: within a
# family the rows are comparable, across families only the position against
# zero is.
fam_mcc = [((NICE.get(k, k) + f" ({v['n']} pairs)") if k.startswith("accord")
            else NICE.get(k, k), v["mcc"], v["ci95"][0], v["ci95"][1])
           for k, v in iv.items()]
# Read, not typed: this row used to carry 0.5145 / [0.321, 0.703] as literals,
# which are the same numbers rounded.
_dec = json.loads((ROOT/"results/accord/decontam/corrected_gold_metrics.json").read_text())["aggregate"]
# Defect-free stratum, decontaminated. This row used to carry the E3 numbers from
# stats/intervals.json (0.5078, CI [0.2217, 0.7615]), which are computed on the
# contaminated released scores, so it was not comparable with the row above it.
_ctrl = json.loads((ROOT/"results/accord/control/final_report.json").read_text())
_df34 = _ctrl["ensemble_mean_of_seed_scores"]["gold_defect_free_rule34"]
fam_auroc = [
    ("Decontaminated ensemble", _dec["ensemble_auroc"]-0.5,
     _dec["ensemble_ci95"][0]-0.5, _dec["ensemble_ci95"][1]-0.5),
    ("Defect-free stratum", _df34["auroc"]-0.5,
     _df34["bootstrap"]["ci95"][0]-0.5, _df34["bootstrap"]["ci95"][1]-0.5),
]
_pb2 = [json.loads((ROOT/f"local_only/task_runs/concat_s{s}.json").read_text())["auroc"] for s in (1,2,3)]
_ou2 = [json.loads(pathlib.Path(f).read_text())["auroc"]
        for f in sorted(glob.glob(str(ROOT/"local_only/task_runs/ours_current_s*.json")))]
_g = st.mean(_pb2) - st.mean(_ou2)
_se = math.sqrt(st.stdev(_pb2)**2/3 + st.stdev(_ou2)**2/3)
_sc = json.loads((ROOT/"results/accord/scaling/scaling_results.json").read_text())
_p100 = [x["accuracy"] for x in _sc if x["tag"]=="scratch_100M"]
_pbert = [x["accuracy"] for x in _sc if x["tag"]=="pretrained_bert"]
_g44 = st.mean(_pbert) - st.mean(_p100)
_se44 = math.sqrt(st.stdev(_pbert)**2/3 + st.stdev(_p100)**2/3)
fam_gap = [
    ("In-hospital mortality, AUROC", _g, _g-1.96*_se, _g+1.96*_se),
    ("AG News at 100M, accuracy", _g44, _g44-1.96*_se44, _g44+1.96*_se44),
]
# Families run down the page in the order the thesis argues them; within a
# family the rows are sorted, because there the position axis does rank.
# One hue per family, not per row. On this canvas the distinction that has to
# survive is which unit a row is measured in, because that is the only thing
# the position axis does not support comparing across; the four systems in the
# first family are separated by their own row labels, as they are everywhere
# else in the set, and take the family's colour here rather than their own.
FAMILIES = [("Matthews correlation", fam_mcc, BLUE2),
            ("AUROC above chance", fam_auroc, GREEN),
            ("Pretraining gap, pretrained minus from scratch", fam_gap, ORANGE)]
for _, rs, _hue in FAMILIES:
    rs.sort(key=lambda r: r[1])

# Lay the rows out bottom up: one unit per row, the family heading half a unit
# above the family's top row, and the hairline in the band above that.
placed, seps, heads, y = [], [], [], 0.0
for fi, (fam, rs, hue) in enumerate(reversed(FAMILIES)):
    for r in rs:
        placed.append((y, r, hue)); y += 1.0
    y -= 1.0
    heads.append((y + 0.52, fam, {BLUE2: BLUE2_T, GREEN: GREEN_T,
                                  ORANGE: ORANGE_T}[hue]))
    y += 1.30
    if fi < len(FAMILIES) - 1:
        seps.append(y)
        y += 0.72
rows = [r for _, r, _h in placed]
y_top_row = placed[-1][0]
y_ceiling = y_top_row + 1.00

fig, ax = plt.subplots(figsize=(FIGW, 3.00))
canvas(fig)
fig.subplots_adjust(left=0.400, right=0.985, top=0.985, bottom=0.140)
# Colour by family, never by whether an interval contains zero. That fact is
# already drawn, by whether the interval crosses the line; saying it again in a
# second colour states one fact twice and needs a key to decode. Nothing here
# takes the accent either: this figure's claim is that the whole column sits on
# the line, which no single mark carries.
for yy, (lab, e, lo, hi), hue in placed:
    ax.errorbar([e], [yy], xerr=[[e-lo], [hi-e]], fmt="o", color=hue, **ERRBAR, **DOT)
ax.axvline(0, **CHANCE_STYLE)
gt = ax.get_yaxis_transform()
for ys in seps:
    ax.plot([-0.66, 1.0], [ys, ys], transform=gt, color=GRID, lw=0.6,
            clip_on=False, zorder=0)
# The family heading is this figure's direct label for the family, so it is set
# in the family's own colour rather than in the annotation grey. It is also the
# redundancy: in a greyscale print the heading, the hairline and the position
# separate the three families, and the hue is only the fast read.
for yh, fam, fam_t in heads:
    ax.text(-0.65, yh, fam, transform=gt, ha="left", va="bottom", fontsize=8.0,
            color=fam_t, style="italic", clip_on=False)
ax.set_yticks([yy for yy, _, _h in placed])
ax.set_yticklabels([r[0] for r in rows], fontsize=8.0, linespacing=1.25)
ax.set_xlabel("Effect, 95% interval (unit named per family)")
ax.set_xlim(-0.36, 0.46)
ax.set_xticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
ax.set_ylim(-0.60, y_ceiling)
n_cross = sum(1 for r in rows if r[2] <= 0 <= r[3])
if SHOW_INFIGURE_TITLES: ax.set_title(f"{n_cross} of {len(rows)} headline estimates cross zero",
             loc="left", fontsize=9.5, pad=8)
ax.text(0, y_top_row + 0.46, "no effect", fontsize=8.0, color=MUTE, ha="center",
        va="bottom")
# The measured axis keeps its spine, extended past the widest cap so no
# interval runs off the end of the scale it is read on. The categorical axis
# has none: a rule beside eight row labels reads as a bracket over all eight.
lean(ax, xbounds=(-0.3, 0.4))
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.savefig(HERE/"fig-forest.pdf"); plt.close(fig)
print(f"wrote fig-forest.pdf ({n_cross}/{len(rows)} cross zero)")

# ---------- F10: power landscape ---------------------------------------------
pw = json.loads((ROOT/"results/accord/section6_auroc_power.json").read_text())
# No fallback literal: if the key is ever missing this must fail, not quietly draw
# a number with no source behind it.
n_pos_study, n_neg_study = pw["n_positives"], pw["n_negatives"]
alpha_study, power_study = pw["alpha"], pw["power"]
# The mortality comparison is read from the run record rather than typed, so this
# script and the notebook cannot drift apart.
_mort = json.loads((ROOT/"results/task_runs.json").read_text())["concat_s1"]["data_lineage"]
mort_n_pos = _mort["test_n_pos"]
mort_n_neg = _mort["test_n"] - mort_n_pos
def hanley_mde(n_pos, n_neg, alpha=0.05, power=0.80):
    """Smallest AUROC distinguishable from 0.5 at the given power, Hanley-McNeil
    variance. Uses BOTH class counts -- the earlier version ignored n_neg and did
    not reproduce the one properly computed value it plotted."""
    from scipy.stats import norm
    z = norm.ppf(1-alpha/2) + norm.ppf(power)
    out = []
    for m, n in zip(np.atleast_1d(n_pos), np.atleast_1d(n_neg)):
        lo, hi = 0.5, 0.9999
        for _ in range(80):
            a = (lo+hi)/2
            q1, q2 = a/(2-a), 2*a*a/(1+a)
            v = (a*(1-a) + (m-1)*(q1-a*a) + (n-1)*(q2-a*a)) / (m*n)
            if (a-0.5)/math.sqrt(max(v,1e-12)) < z: lo = a
            else: hi = a
        out.append((lo+hi)/2)
    return np.array(out)

n_pos = np.logspace(np.log10(5), np.log10(3000), 60)
n_neg = n_pos * (n_neg_study/n_pos_study)   # this evaluation set's own positive:negative ratio
mde = hanley_mde(n_pos, n_neg, alpha=alpha_study, power=power_study)
# Both markers are placed on the curve the axes actually draws. The study marker
# used to be plotted at the stored 0.7382, which this curve does not pass through.
mde_study = hanley_mde([n_pos_study], [n_neg_study], alpha=alpha_study, power=power_study)[0]
mde_mort = hanley_mde([mort_n_pos], [mort_n_neg], alpha=alpha_study, power=power_study)[0]
fig, ax = plt.subplots(figsize=(FIGW, 2.75))
canvas(fig)
fig.subplots_adjust(left=0.135, right=0.985, top=0.965, bottom=0.165)
ax.fill_between(n_pos, np.clip(mde, .5, 1), 1.0, color=PALE, alpha=.55, lw=0, zorder=1)
ax.plot(n_pos, np.clip(mde, 0.5, 1.0), color=BLUE2, lw=1.3, zorder=3)
ax.text(2600, 0.955, "detectable", fontsize=8.0, color=MUTE, ha="right")
ax.text(6.4, 0.545, "not detectable at this sample size", fontsize=8.0, color=MUTE)
# The study is the one accented mark. The comparator recedes to grey, and the
# two also differ in shape, so a greyscale print keeps them apart.
ax.plot([mort_n_pos], [mde_mort], "s", color=GREY, ms=4.2, zorder=5)
ax.annotate(f"mortality task\n{mort_n_pos} test positives", (mort_n_pos, mde_mort),
            xytext=(105, 0.615), fontsize=8.0, color=MUTE, linespacing=1.3,
            arrowprops=dict(arrowstyle="-", lw=.6, color=GREY))
ax.plot([n_pos_study], [mde_study], "o", color=SCARLET, ms=5.0, zorder=5)
ax.annotate(f"this study\n{n_pos_study} positives, MDE {mde_study:.3f}",
            (n_pos_study, mde_study), xytext=(22, 0.855),
            fontsize=8.0, color=SCARLET, linespacing=1.3,
            arrowprops=dict(arrowstyle="-", lw=.6, color=SCARLET))
ax.set_xscale("log"); ax.set_xlabel("Positive examples in the evaluation set (log scale)")
ax.set_ylabel(f"Minimum detectable AUROC\n({power_study*100:.0f}% power)", linespacing=1.3)
ax.set_ylim(0.5, 1.0); ax.set_xlim(5, 3000)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
lean(ax, ybounds=(0.5, 1.0))
if SHOW_INFIGURE_TITLES: ax.set_title("What each evaluation in this thesis could have seen",
             loc="left", fontsize=9.5, pad=8)
fig.savefig(HERE/"fig-power.pdf"); plt.close(fig); print("wrote fig-power.pdf")
