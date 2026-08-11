"""F3 -- the gap trajectory.

Three measurements of the same quantity as the measurement got more honest.
Each point carries the seed variance available to it at the time; the final
point is the only one with variance on both arms.
"""
import json, pathlib, statistics as st, glob
from style import setup, ACCENT, MUTED, LIGHT, DEAD
setup()
import matplotlib.pyplot as plt

ours_cur = [json.load(open(f))["auroc"] for f in sorted(glob.glob(
    "/Users/rickygole/provenance-encoder/local_only/task_runs/ours_current_s*.json"))]
pb_cur = [json.load(open(f"/Users/rickygole/provenance-encoder/local_only/task_runs/concat_s{s}.json"))["auroc"]
          for s in (1, 2, 3)]
old = json.load(open("/Users/rickygole/provenance-encoder/results/task_runs.json"))
ours_old = [old[f"ours_randinit_s{i}"]["auroc"] for i in range(3)]

STAGES = [
    ("C32\nold data\n1 comparator seed", 0.0220, st.stdev(ours_old), None,
     "not comparable,\ndifferent datasets"),
    ("C34\ncurrent data\n1 comparator seed", 0.0149, st.stdev(ours_cur), None,
     "one arm's variance only"),
    ("C37\ncurrent data\n3 seeds each", st.mean(pb_cur)-st.mean(ours_cur),
     st.stdev(ours_cur), st.stdev(pb_cur), "pooled variance,\nboth arms"),
]

fig, ax = plt.subplots(figsize=(6.2, 3.9))
x = [0, 1, 2]
gaps = [s[1] for s in STAGES]

for i, (lab, g, sd_a, sd_b, note) in enumerate(STAGES):
    band = sd_a if sd_b is None else (sd_a**2/2 + sd_b**2/2)**0.5
    ax.fill_between([i-0.26, i+0.26], [-band]*2, [band]*2,
                    color=LIGHT, zorder=1, lw=0)
    ax.plot([i-0.26, i+0.26], [band]*2, color=DEAD, lw=0.6, zorder=2)
    ax.plot([i-0.26, i+0.26], [-band]*2, color=DEAD, lw=0.6, zorder=2)

ax.plot(x, gaps, "-", color=ACCENT, lw=1.4, zorder=4)
ax.plot(x, gaps, "o", color=ACCENT, ms=6, zorder=5,
        markeredgecolor="w", markeredgewidth=0.8)
for i, g in enumerate(gaps):
    ax.annotate(f"{g:+.4f}", (i, g), textcoords="offset points",
                xytext=(0, 11), ha="center", fontsize=8.5, color=ACCENT)
for i, s in enumerate(STAGES):
    ax.text(i, -0.0088, s[4], ha="center", va="top", fontsize=6.8,
            color="0.4", style="italic")

ax.axhline(0, color="0.25", lw=0.7, ls="--", zorder=3)
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in STAGES], fontsize=8)
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.010, 0.028)
ax.set_ylabel("PubMedBERT $-$ from-scratch encoder\n(AUROC)")
ax.set_title("The gap shrank every time the comparison got more honest",
             loc="left", fontsize=9.5, pad=8)
ax.text(0.985, 0.05, "shaded band: seed standard deviation available at that stage",
        transform=ax.transAxes, ha="right", fontsize=6.8, color="0.4")

out = pathlib.Path(__file__).parent / "fig-gap-trajectory.pdf"
fig.savefig(out); print(f"wrote {out.name}   gaps: {[round(g,4) for g in gaps]}")
