"""F5 prevalence response, F6 scaling with clinical overlay, F7 forest of every
claim, F10 power landscape. All from committed aggregates."""
import json, pathlib, glob, math, statistics as st
import numpy as np
from style import setup, ACCENT, MUTED, LIGHT, DEAD
setup()
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
ROOT = pathlib.Path("/Users/rickygole/provenance-encoder")

# ---------- F5: prevalence response -----------------------------------------
d = json.loads((ROOT/"accord/prevalence_replication/results/sweep_results.json").read_text())
models, prompts = d["models"], ["A_bare", "B_baserate_abstain"]
PL = {"A_bare": "bare prompt", "B_baserate_abstain": "base rate stated, abstention allowed"}
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharex=True, sharey=True)
for r, pr in enumerate(prompts):
    for c, m in enumerate(models):
        ax = axes[r, c]
        xs, ys = [], []
        for p in ["0.02", "0.05", "0.1", "0.25"]:
            e = d["results"][m][pr].get(p)
            if e: xs.append(float(p)); ys.append(e["positive_rate_of_decided"])
        ax.plot([0, .28], [0, .28], ls="--", lw=.7, color="0.55", zorder=1)
        ax.plot(xs, ys, "-o", color=ACCENT, ms=3.6, lw=1.2, zorder=3)
        ax.plot([0.04], [0.849], "s", color="#b03030", ms=5, zorder=4)
        if r == 0 and c == 0:
            ax.annotate("clinical\n(4% → 85%)", (0.04, 0.849), xytext=(0.085, 0.72),
                        fontsize=6.5, color="#b03030",
                        arrowprops=dict(arrowstyle="-", lw=.5, color="#b03030"))
        ax.set_xlim(0, .285); ax.set_ylim(0, .92)
        if r == 0: ax.set_title(m, fontsize=8.5)
        if c == 0: ax.set_ylabel(PL[pr], fontsize=7)
        ax.tick_params(labelsize=7)
fig.supxlabel("true prevalence", fontsize=9, y=0.015)
fig.supylabel("labeller predicted-positive rate", fontsize=9, x=0.005)
fig.suptitle("The labeller under-responds to prevalence — but nowhere near the clinical shift",
             fontsize=9.5, x=0.02, ha="left", y=0.99)
fig.text(0.02, 0.925, "dashed: perfect calibration", fontsize=6.8, color="0.4")
fig.tight_layout(rect=[0.02, 0.03, 1, 0.93])
fig.savefig(HERE/"fig-prevalence-response.pdf"); plt.close(fig); print("wrote fig-prevalence-response.pdf")

# ---------- F6: scaling with clinical overlay --------------------------------
sc = json.loads((ROOT/"results/accord/scaling/scaling_results.json").read_text())
by, pr_ = {}, {}
for x in sc:
    by.setdefault(x["tag"], []).append(x["accuracy"]); pr_[x["tag"]] = x["non_emb_params"]
tags = ["scratch_10M", "scratch_30M", "scratch_100M"]
xs = [pr_[t] for t in tags]; mu = [st.mean(by[t]) for t in tags]; sd = [st.stdev(by[t]) for t in tags]
pm, ps, pp = st.mean(by["pretrained_bert"]), st.stdev(by["pretrained_bert"]), pr_["pretrained_bert"]

fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.axhspan(pm-ps, pm+ps, color=LIGHT, zorder=1)
ax.axhline(pm, color=MUTED, lw=1.1, zorder=2)
ax.text(1.05e7, pm+0.0035, f"pretrained BERT   {pm:.4f} ± {ps:.4f}   ({pp/1e6:.0f}M)",
        fontsize=7.5, color="0.25", va="bottom")
ax.errorbar(xs, mu, yerr=sd, fmt="-o", color=ACCENT, ms=5, lw=1.3, capsize=3, zorder=4)
for x, m, s, t in zip(xs, mu, sd, tags):
    ax.annotate(f"{pm-m:+.4f}", (x, m), textcoords="offset points", xytext=(0, -17),
                ha="center", fontsize=7.5, color=ACCENT)
ax.set_xscale("log"); ax.set_xlabel("non-embedding parameters (log scale)")
ax.set_ylabel("accuracy, AG News"); ax.set_ylim(0.865, 0.952)
ax.set_title("Pretraining is worth five points here — and more parameters do not close it",
             loc="left", fontsize=9.5, pad=8)
ins = fig.add_axes([0.60, 0.24, 0.28, 0.24])
ins.axhspan(0.7693-0.0102, 0.7693+0.0102, color=LIGHT)
ins.axhline(0.7693, color=MUTED, lw=1.0)
ins.errorbar([1.06e7], [0.7662], yerr=[0.0073], fmt="o", color=ACCENT, ms=4, capsize=2.5)
ins.set_xscale("log"); ins.set_xlim(5e6, 2e8); ins.set_ylim(0.745, 0.792)
ins.set_title("clinical task: gap $+0.0031$, $p=0.69$", fontsize=6.5, pad=3)
ins.tick_params(labelsize=5.5); ins.set_yticks([0.75, 0.77, 0.79])
fig.savefig(HERE/"fig-scaling.pdf"); plt.close(fig); print("wrote fig-scaling.pdf")

# ---------- F7: forest of every headline claim -------------------------------
iv = json.loads((ROOT/"results/accord/stats/intervals.json").read_text())["E2_mcc_intervals"]
rows = []
for k, v in iv.items():
    rows.append((f"MCC — {k.replace('_',' ')}", v["mcc"], v["ci95"][0], v["ci95"][1]))
rows.append(("AUROC $-$ 0.5 (decontaminated)", 0.5145-0.5, 0.321-0.5, 0.703-0.5))
rows.append(("AUROC $-$ 0.5 (defect-free stratum)", 0.5078-0.5, 0.2217-0.5, 0.7615-0.5))
rows.append(("C37 gap: PubMedBERT $-$ scratch", 0.0031, -0.0128, 0.0190))
rows.append(("C44 gap at 100M (AG News)", 0.0518, 0.0478, 0.0558))
rows.sort(key=lambda r: r[1])

fig, ax = plt.subplots(figsize=(6.6, 3.6))
for i, (lab, e, lo, hi) in enumerate(rows):
    crosses = lo <= 0 <= hi
    c = DEAD if crosses else ACCENT
    ax.hlines(i, lo, hi, color=c, lw=1.4)
    ax.plot(e, i, "o", color=c, ms=4.4, markeredgecolor="w", markeredgewidth=0.6)
ax.axvline(0, color="0.25", lw=0.8, ls="--")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.set_xlabel("effect (units differ by row; zero = no effect)")
n_cross = sum(1 for r in rows if r[2] <= 0 <= r[3])
ax.set_title(f"{n_cross} of {len(rows)} headline estimates cross zero",
             loc="left", fontsize=9.5, pad=8)
ax.text(0.99, 0.03, "grey: interval contains zero", transform=ax.transAxes,
        ha="right", fontsize=7, color="0.45")
fig.savefig(HERE/"fig-forest.pdf"); plt.close(fig)
print(f"wrote fig-forest.pdf ({n_cross}/{len(rows)} cross zero)")

# ---------- F10: power landscape ---------------------------------------------
pw = json.loads((ROOT/"results/accord/section6_auroc_power.json").read_text())
mda = pw.get("minimum_detectable_auc", 0.7382)
n_pos = np.logspace(np.log10(5), np.log10(3000), 200)
mde = 0.5 + 0.5*np.sqrt(1/np.maximum(n_pos, 1))*2.8/np.sqrt(2)
fig, ax = plt.subplots(figsize=(6.2, 3.7))
ax.plot(n_pos, np.clip(mde, 0.5, 1.0), color=ACCENT, lw=1.4)
ax.fill_between(n_pos, np.clip(mde, .5, 1), 1.0, color=LIGHT, alpha=.55, lw=0)
ax.text(700, 0.93, "detectable", fontsize=8, color="0.35")
ax.text(9, 0.56, "not detectable\nat this sample size", fontsize=8, color="0.35")
ax.plot([12], [mda], "o", color="#b03030", ms=6, zorder=5)
ax.annotate(f"this study\n12 positives, MDE {mda:.3f}", (12, mda), xytext=(28, 0.86),
            fontsize=7.5, color="#b03030",
            arrowprops=dict(arrowstyle="-", lw=.6, color="#b03030"))
ax.plot([570], [0.5+0.5*np.sqrt(1/570)*2.8/np.sqrt(2)], "o", color=MUTED, ms=5)
ax.annotate("mortality task\n570 test positives", (570, 0.5+0.5*np.sqrt(1/570)*2.8/np.sqrt(2)),
            xytext=(230, 0.60), fontsize=7.5, color="0.3",
            arrowprops=dict(arrowstyle="-", lw=.6, color="0.5"))
ax.set_xscale("log"); ax.set_xlabel("positive examples in the evaluation set (log scale)")
ax.set_ylabel("minimum detectable AUROC (80% power)")
ax.set_ylim(0.5, 1.0); ax.set_xlim(5, 3000)
ax.set_title("What each evaluation in this thesis could have seen",
             loc="left", fontsize=9.5, pad=8)
fig.savefig(HERE/"fig-power.pdf"); plt.close(fig); print("wrote fig-power.pdf")
