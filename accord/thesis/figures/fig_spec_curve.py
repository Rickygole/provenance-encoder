"""F1 -- specification curve (Simonsohn style).

The missing-language odds ratio across five volume specifications x three
cohorts, sorted by effect size, with a covariate matrix below. Shows the
effect dissolving as the correct covariate enters, and which choice did it.
"""
import json, pathlib, numpy as np
from style import setup, ACCENT, MUTED, LIGHT, DEAD
setup()
import matplotlib.pyplot as plt

D = json.loads(pathlib.Path(
    "/Users/rickygole/mimic_local_only/accord_generality/experimentA_corrected_replication.json").read_text())

SPECS = [("none", [], "none"), ("total", ["total"], "log(total)"),
         ("nursing_only", ["nurs"], "log(nursing)"),
         ("discharge_only", ["disch"], "log(discharge)"),
         ("nursing_plus_discharge", ["nurs", "disch"], "log(nurs) + log(disch)")]
KEY = "C(language_group3, Treatment('ENGLISH'))[T.MISSING]"
COVS = ["total", "nurs", "disch"]
COV_LABEL = {"total": "total note chars", "nurs": "nursing chars", "disch": "discharge chars"}
COHORTS = [("hf", "heart failure"), ("sepsis", "sepsis"), ("aki", "AKI")]

pts = []
for ck, cname in COHORTS:
    curve = D[ck]["specification_curve"]
    for skey, covs, slabel in SPECS:
        e = curve.get(skey)
        if not e: continue
        v = e["robust"][KEY]
        pts.append(dict(cohort=cname, spec=slabel, covs=covs,
                        OR=v["OR"], lo=v["CI_low"], hi=v["CI_high"], p=v["p"]))
pts.sort(key=lambda r: r["OR"])

fig = plt.figure(figsize=(7.0, 5.2))
gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.06)
ax, mx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

x = np.arange(len(pts))
for i, r in enumerate(pts):
    sig = r["hi"] < 1.0
    c = ACCENT if sig else DEAD
    ax.vlines(i, r["lo"], r["hi"], color=c, lw=1.1, zorder=2)
    ax.plot(i, r["OR"], "o", ms=3.4, color=c, zorder=3)

ax.axhline(1.0, color="0.25", lw=0.7, ls="--", zorder=1)
ax.text(len(pts)-0.4, 1.005, "no effect", fontsize=7, color="0.35", va="bottom", ha="right")
ax.set_ylabel("odds ratio, unrecorded LANGUAGE\n(reference: English)")
ax.set_xlim(-0.8, len(pts)-0.2); ax.set_xticks([])
ax.set_ylim(0.28, 1.35)
ax.set_title("Every specification, every cohort, sorted by effect size",
             loc="left", fontsize=9.5, pad=8)

n_sig = sum(1 for r in pts if r["hi"] < 1.0)
ax.text(0.4, 1.28, f"{n_sig} of {len(pts)} specifications exclude 1.0",
        fontsize=7.5, color="0.3", style="italic")

# covariate matrix
rows = COVS + ["__coh__"]
for i, r in enumerate(pts):
    for j, cv in enumerate(COVS):
        on = cv in r["covs"]
        mx.plot(i, len(rows)-1-j, "o", ms=3.2,
                color=ACCENT if on else LIGHT, zorder=3)
    ci = [c[1] for c in COHORTS].index(r["cohort"])
    mx.plot(i, 0, "s", ms=3.2, color=[ACCENT, MUTED, DEAD][ci], zorder=3)

for j, cv in enumerate(COVS):
    mx.axhline(len(rows)-1-j, color=LIGHT, lw=0.4, zorder=1)
mx.axhline(0, color=LIGHT, lw=0.4, zorder=1)
mx.set_yticks(list(range(len(rows))))
mx.set_yticklabels(["cohort"] + [COV_LABEL[c] for c in reversed(COVS)])
mx.set_xlim(-0.8, len(pts)-0.2); mx.set_xticks([])
mx.set_ylim(-0.7, len(rows)-0.3)
mx.set_xlabel("specification", labelpad=4)
for s in ("top", "right", "bottom", "left"): mx.spines[s].set_visible(False)
mx.tick_params(length=0)

hs = [plt.Line2D([], [], marker="s", ls="", ms=3.2, color=c, label=l)
      for c, (_, l) in zip([ACCENT, MUTED, DEAD], COHORTS)]
mx.legend(handles=hs, ncol=3, frameon=False, loc="lower center",
          bbox_to_anchor=(0.5, -0.62), handletextpad=0.3, columnspacing=1.4)

out = pathlib.Path(__file__).parent / "fig-spec-curve.pdf"
fig.savefig(out); print("wrote", out.name, f"({len(pts)} specifications, {n_sig} exclude 1.0)")
