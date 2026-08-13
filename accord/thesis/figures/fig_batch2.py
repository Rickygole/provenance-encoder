"""F5 prevalence, F6 scaling, F7 forest, F10 power landscape.

Every value is READ from a committed result file. An earlier version typed the
C37 and C44 numbers as literals and drew the power curve from an ad-hoc formula
that did not reproduce the one real MDE it plotted -- the exact defect
corrections C7 and C10 retract elsewhere in this project. See C48."""
import json, pathlib, glob, math, statistics as st
import numpy as np
from style import setup, ACCENT, MUTED, LIGHT, DEAD, ALERT
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
# The second row label is wrapped: on one line it ran the full height of its
# panel and collided with the row above it.
PL = {"A_bare": "bare prompt", "B_baserate_abstain": "base rate stated,\nabstention allowed"}
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0), sharex=True, sharey=True)
for r, pr in enumerate(prompts):
    for c, m in enumerate(models):
        ax = axes[r, c]
        xs, ys = [], []
        for p in ["0.02", "0.05", "0.1", "0.25"]:
            e = d["results"][m][pr].get(p)
            if e: xs.append(float(p)); ys.append(e["positive_rate_of_decided"])
        ax.plot([0, .28], [0, .28], ls="--", lw=.7, color=DEAD, zorder=1)
        ax.plot(xs, ys, "-o", color=ACCENT, ms=3.6, lw=1.2, zorder=3)
        ax.plot([clinical_x], [clinical_y], "s", color=ALERT, ms=5, zorder=4)
        if r == 0 and c == 0:
            ax.annotate(f"clinical\n({clinical_x * 100:.0f}% to {clinical_y * 100:.0f}%)",
                        (clinical_x, clinical_y), xytext=(0.075, 0.60),
                        fontsize=8.0, color=ALERT, linespacing=1.25,
                        arrowprops=dict(arrowstyle="-", lw=.5, color=ALERT))
        ax.set_xlim(0, .285); ax.set_ylim(0, .92)
        # Panel heading, not a figure title: it names which labeller this column is.
        if r == 0: ax.set_title(m, fontsize=8.5, pad=4)
        if c == 0: ax.set_ylabel(PL[pr], fontsize=8.0, linespacing=1.3)
        ax.tick_params(labelsize=8)
fig.supxlabel("true prevalence", fontsize=9.0, y=0.028)
fig.supylabel("labeller predicted-positive rate", fontsize=9.0, x=0.005)
if SHOW_INFIGURE_TITLES: fig.suptitle("The labeller under-responds to prevalence, but nowhere near the clinical shift",
             fontsize=9.5, x=0.02, ha="left", y=0.99)
if SHOW_INFIGURE_NOTES: fig.text(0.02, 0.925, "dashed: perfect calibration", fontsize=8.0, color="0.4")
fig.tight_layout(rect=[0.022, 0.058, 1, 1])
fig.savefig(HERE/"fig-prevalence-response.pdf"); plt.close(fig); print("wrote fig-prevalence-response.pdf")

# ---------- F6: scaling with clinical overlay --------------------------------
sc = json.loads((ROOT/"results/accord/scaling/scaling_results.json").read_text())
by, pr_ = {}, {}
for x in sc:
    by.setdefault(x["tag"], []).append(x["accuracy"]); pr_[x["tag"]] = x["non_emb_params"]
tags = ["scratch_10M", "scratch_30M", "scratch_100M"]
xs = [pr_[t] for t in tags]; mu = [st.mean(by[t]) for t in tags]; sd = [st.stdev(by[t]) for t in tags]
pm, ps, pp = st.mean(by["pretrained_bert"]), st.stdev(by["pretrained_bert"]), pr_["pretrained_bert"]

fig, ax = plt.subplots(figsize=(6.6, 3.9))
fig.subplots_adjust(left=0.105, right=0.985, top=0.97, bottom=0.135)
ax.axhspan(pm-ps, pm+ps, color=LIGHT, zorder=1)
ax.axhline(pm, color=MUTED, lw=1.1, zorder=2)
ax.text(8.6e6, pm+0.0035, f"pretrained BERT   {pm:.4f} +/- {ps:.4f}   ({pp/1e6:.0f}M)",
        fontsize=8.0, color=MUTED, va="bottom")
ax.errorbar(xs, mu, yerr=sd, fmt="-o", color=ACCENT, ms=5, lw=1.3, capsize=3, zorder=4)
# The gap labels sit below their points; the x limits are set wide enough that the
# leftmost one is not cut off by the axis, as it was before.
for x, m, s, t in zip(xs, mu, sd, tags):
    ax.annotate(f"{pm-m:+.4f}", (x, m - s), textcoords="offset points", xytext=(0, -12),
                ha="center", va="top", fontsize=8.0, color=ACCENT)
ax.set_xscale("log"); ax.set_xlabel("non-embedding parameters (log scale)")
ax.set_ylabel("accuracy, AG News"); ax.set_ylim(0.865, 0.952)
ax.set_xlim(7.0e6, 1.7e8)
if SHOW_INFIGURE_TITLES: ax.set_title("Pretraining is worth five points here, and more parameters do not close it",
             loc="left", fontsize=9.5, pad=8)
# C37 clinical comparison, read from the run records -- never typed
_pb = [json.loads((ROOT/f"local_only/task_runs/concat_s{s}.json").read_text())["auroc"] for s in (1,2,3)]
_ou = [json.loads(pathlib.Path(f).read_text())["auroc"]
       for f in sorted(glob.glob(str(ROOT/"local_only/task_runs/ours_current_s*.json")))]
_pbm, _pbs = st.mean(_pb), st.stdev(_pb)
_oum, _ous = st.mean(_ou), st.stdev(_ou)
_gap = _pbm - _oum
_pooled = math.sqrt((_pbs**2 + _ous**2)/2)
# The inset used to sit at the lower right, on top of the 100M point and its gap
# label. It now occupies the empty band between the scratch curve and the
# pretrained reference.
ins = fig.add_axes([0.505, 0.455, 0.30, 0.235])
ins.axhspan(_pbm-_pbs, _pbm+_pbs, color=LIGHT)
ins.axhline(_pbm, color=MUTED, lw=1.0)
ins.errorbar([1.06e7], [_oum], yerr=[_ous], fmt="o", color=ACCENT, ms=4, capsize=2.5)
ins.set_xscale("log"); ins.set_xlim(5e6, 2e8); ins.set_ylim(_oum-0.025, _pbm+0.025)
ins.set_title(f"clinical task: gap {_gap:+.4f}", fontsize=8.0, pad=3)
ins.tick_params(labelsize=8); ins.set_yticks([0.75, 0.79]); ins.set_xticks([1e7, 1e8])
fig.savefig(HERE/"fig-scaling.pdf"); plt.close(fig); print("wrote fig-scaling.pdf")

# ---------- F7: forest of every headline claim -------------------------------
iv = json.loads((ROOT/"results/accord/stats/intervals.json").read_text())["E2_mcc_intervals"]
rows = []
for k, v in iv.items():
    rows.append((f"MCC, {k.replace('_',' ')}", v["mcc"], v["ci95"][0], v["ci95"][1]))
# Read, not typed: this row used to carry 0.5145 / [0.321, 0.703] as literals,
# which are the same numbers rounded.
_dec = json.loads((ROOT/"results/accord/decontam/corrected_gold_metrics.json").read_text())["aggregate"]
rows.append(("AUROC minus 0.5 (decontaminated)", _dec["ensemble_auroc"]-0.5,
             _dec["ensemble_ci95"][0]-0.5, _dec["ensemble_ci95"][1]-0.5))
# Defect-free stratum, decontaminated. This row used to carry the E3 numbers from
# stats/intervals.json (0.5078, CI [0.2217, 0.7615]), which are computed on the
# contaminated released scores, so it was not comparable with the row above it.
_ctrl = json.loads((ROOT/"results/accord/control/final_report.json").read_text())
_df34 = _ctrl["ensemble_mean_of_seed_scores"]["gold_defect_free_rule34"]
rows.append(("AUROC minus 0.5 (defect-free stratum)", _df34["auroc"]-0.5,
             _df34["bootstrap"]["ci95"][0]-0.5, _df34["bootstrap"]["ci95"][1]-0.5))
_pb2 = [json.loads((ROOT/f"local_only/task_runs/concat_s{s}.json").read_text())["auroc"] for s in (1,2,3)]
_ou2 = [json.loads(pathlib.Path(f).read_text())["auroc"]
        for f in sorted(glob.glob(str(ROOT/"local_only/task_runs/ours_current_s*.json")))]
_g = st.mean(_pb2) - st.mean(_ou2)
_se = math.sqrt(st.stdev(_pb2)**2/3 + st.stdev(_ou2)**2/3)
rows.append(("C37 gap, PubMedBERT minus scratch", _g, _g-1.96*_se, _g+1.96*_se))
_sc = json.loads((ROOT/"results/accord/scaling/scaling_results.json").read_text())
_p100 = [x["accuracy"] for x in _sc if x["tag"]=="scratch_100M"]
_pbert = [x["accuracy"] for x in _sc if x["tag"]=="pretrained_bert"]
_g44 = st.mean(_pbert) - st.mean(_p100)
_se44 = math.sqrt(st.stdev(_pbert)**2/3 + st.stdev(_p100)**2/3)
rows.append(("C44 gap at 100M (AG News)", _g44, _g44-1.96*_se44, _g44+1.96*_se44))
rows.sort(key=lambda r: r[1])

fig, ax = plt.subplots(figsize=(6.8, 3.7))
fig.subplots_adjust(left=0.315, right=0.985, top=0.985, bottom=0.135)
for i, (lab, e, lo, hi) in enumerate(rows):
    crosses = lo <= 0 <= hi
    c = DEAD if crosses else ACCENT
    ax.hlines(i, lo, hi, color=c, lw=1.6)
    ax.plot(e, i, "o", color=c, ms=4.8, markeredgecolor="w", markeredgewidth=0.6)
ax.axvline(0, color=MUTED, lw=0.9, ls="--", zorder=0)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8.0)
ax.set_xlabel("effect (units differ by row; zero = no effect)")
# One extra row of headroom at the bottom holds the key, so it no longer sits on
# the lowest interval.
ax.set_ylim(-1.15, len(rows) - 0.5)
n_cross = sum(1 for r in rows if r[2] <= 0 <= r[3])
if SHOW_INFIGURE_TITLES: ax.set_title(f"{n_cross} of {len(rows)} headline estimates cross zero",
             loc="left", fontsize=9.5, pad=8)
ax.text(0.995, 0.015, "grey: interval contains zero", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8.0, color=DEAD)
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
fig, ax = plt.subplots(figsize=(6.4, 3.7))
fig.subplots_adjust(left=0.115, right=0.985, top=0.975, bottom=0.145)
ax.plot(n_pos, np.clip(mde, 0.5, 1.0), color=ACCENT, lw=1.6, zorder=3)
ax.fill_between(n_pos, np.clip(mde, .5, 1), 1.0, color=LIGHT, alpha=.45, lw=0, zorder=1)
ax.text(760, 0.93, "detectable", fontsize=8.0, color=MUTED)
ax.text(9, 0.56, "not detectable\nat this sample size", fontsize=8.0, color=MUTED,
        linespacing=1.3)
# Round marker for the study, square for the comparator: in a greyscale print the
# scarlet and the teal are close in luminance, so shape carries the distinction.
ax.plot([n_pos_study], [mde_study], "o", color=ALERT, ms=6.5, zorder=5)
ax.annotate(f"this study\n{n_pos_study} positives, MDE {mde_study:.3f}",
            (n_pos_study, mde_study), xytext=(26, 0.865),
            fontsize=8.0, color=ALERT, linespacing=1.3,
            arrowprops=dict(arrowstyle="-", lw=.6, color=ALERT))
ax.plot([mort_n_pos], [mde_mort], "s", color=MUTED, ms=5.5, zorder=5)
ax.annotate(f"mortality task\n{mort_n_pos} test positives", (mort_n_pos, mde_mort),
            xytext=(150, 0.605), fontsize=8.0, color=MUTED, linespacing=1.3,
            arrowprops=dict(arrowstyle="-", lw=.6, color=MUTED))
ax.set_xscale("log"); ax.set_xlabel("positive examples in the evaluation set (log scale)")
ax.set_ylabel(f"minimum detectable AUROC ({power_study*100:.0f}% power)")
ax.set_ylim(0.5, 1.0); ax.set_xlim(5, 3000)
if SHOW_INFIGURE_TITLES: ax.set_title("What each evaluation in this thesis could have seen",
             loc="left", fontsize=9.5, pad=8)
fig.savefig(HERE/"fig-power.pdf"); plt.close(fig); print("wrote fig-power.pdf")
