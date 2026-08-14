"""Two figures the first pass declined: the data for both is on disk, in the
Drive quarantine (from_drive_2026-08-08/) rather than results/accord/, which is
why the earlier sweep missed it.

fig-asymmetry     -- fully sourced from hf_medication_mentions.csv
fig-falsification -- left panel sourced from the decontaminated 3-seed
                     aggregate; right panel's transfer F1 now read from
                     results/accord/generality/transfer_results_correct.json
                     (it was typed as three literals here), so this script and
                     the notebook cell cannot drift apart.

The 45.2% held-out rule-accuracy reference line that used to sit on the right
panel has been removed: 0.4515 appears in no result file anywhere on disk, only
in figure code and in the audit records that flag it as untraceable.

fig-falsification now draws both panels as dot-and-interval, in the shared
grammar of style.py. Neither quantity is a count: one is a learned parameter and
one is an F1, and drawing either as a filled bar asserted a solidity three seeds
do not carry. The three seeds behind each authority-scale mean are drawn.

Panel (a) plots the DISPLACEMENT, final minus the value the parameter was
initialised to, against a line at zero. That subtraction is arithmetic on two
committed numbers, and it makes the near-zero geometry itself the claim: the
reader sees that the parameter did not move rather than being asked to compare
a blue dot with a grey tick a pixel away. Panel (b) carries no intervals
because none exist to carry, and says so by being grey.
"""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style import (setup, FIGW, INK, BLUE, PALE, SCARLET, MUTE, GREY,
                   ACCENT, MUTED, LIGHT, DEAD, ALERT,
                   CHANCE_STYLE, ERRBAR, DOT, canvas, lean)

# Long provenance notes belong in the LaTeX caption, not inside the plot.
# Set True to draw them again.
SHOW_INFIGURE_NOTES = False
SHOW_INFIGURE_TITLES = False
import pandas as pd

setup()
HERE = pathlib.Path(__file__).parent
REPO = pathlib.Path("/Users/rickygole/provenance-encoder")
Q = pathlib.Path("/Users/rickygole/mimic_local_only/from_drive_2026-08-08/outputs")
S = pathlib.Path("/Users/rickygole/mimic_local_only/accord_supp/results")
man = {}

# ---- fig:asymmetry ---------------------------------------------------------
df = pd.read_csv(Q / "hf_medication_mentions.csv", usecols=["action", "note_role"])
clf = df[df.action != "unclassified"]
n_all, n_clf = len(df), len(clf)
roles = clf.note_role.value_counts()
disch, nurs = int(roles["discharge"]), int(roles["nursing"])

fig, (a, b) = plt.subplots(1, 2, figsize=(FIGW, 2.1))
canvas(fig)
fig.subplots_adjust(left=0.135, right=0.985, top=0.885, bottom=0.150, wspace=0.42)
a.bar([0], [n_clf], color=BLUE, linewidth=0, width=0.45)
a.bar([0], [n_all - n_clf], bottom=[n_clf], color=PALE, linewidth=0, width=0.45)
a.text(0.28, n_clf / 2, f"action-classified\n{n_clf:,}", fontsize=8.0, color=INK,
       va="center", ha="left", linespacing=1.25)
a.text(0.28, n_clf + (n_all - n_clf) / 2, f"unclassified\n{n_all - n_clf:,}",
       fontsize=8.0, color=MUTE, va="center", ha="left", linespacing=1.25)
a.set_xticks([]); a.set_xlim(-0.4, 1.5)
a.set_ylabel("Medication mentions")
a.set_title(f"all mentions, n = {n_all:,}", fontsize=8.0, color=MUTE, pad=4)
lean(a)
a.spines["bottom"].set_visible(False)

b.bar([0, 1], [disch, nurs], color=BLUE, linewidth=0, width=0.55)
b.set_xticks([0, 1]); b.set_xticklabels(["discharge", "nursing"], fontsize=8.0)
for x, v in enumerate([disch, nurs]):
    b.text(x, v + n_clf * .02, f"{v:,}", ha="center", fontsize=8.0, color=INK)
b.set_ylabel("Action-classified mentions")
b.set_title(f"by note role, {disch/nurs:.1f} to 1", fontsize=8.0, color=MUTE, pad=4)
b.set_ylim(0, disch * 1.20)
lean(b)
fig.savefig(HERE / "fig-asymmetry.pdf"); plt.close(fig)
man["fig:asymmetry"] = {"file": "fig-asymmetry.pdf",
    "source": "from_drive_2026-08-08/outputs/hf_medication_mentions.csv",
    "values": {"total": n_all, "classified": n_clf, "discharge": disch,
               "nursing": nurs, "ratio": round(disch/nurs, 3)}}

# ---- fig:falsification -----------------------------------------------------
agg_all = json.loads((S / "accord_v2_decontam_gold_aggregate.json").read_text())
agg = agg_all["aggregate"]
aN, aD = agg["alpha_n"]["mean"], agg["alpha_d"]["mean"]
sN, sD = agg["alpha_n"]["std"], agg["alpha_d"]["std"]
# The three seeds behind each mean, read from the same file.
seeds = sorted(agg_all["per_seed"])
seed_N = [agg_all["per_seed"][s]["authority_scale"]["alpha_nursing"] for s in seeds]
seed_D = [agg_all["per_seed"][s]["authority_scale"]["alpha_discharge"] for s in seeds]

transfer = json.loads((REPO / "results/accord/generality/transfer_results_correct.json").read_text())
f1 = [transfer["hf_reference"]["rule_type_f1"], transfer["sepsis"]["rule_type_f1"],
      transfer["aki"]["rule_type_f1"]]
n_cand = [transfer["hf_reference"]["n_candidates"], transfer["sepsis"]["n_candidates"],
          transfer["aki"]["n_candidates"]]

fig, (a, b) = plt.subplots(1, 2, figsize=(FIGW, 1.95),
                           gridspec_kw={"width_ratios": [1.12, 1.0]})
canvas(fig)
fig.subplots_adjust(left=0.150, right=0.985, top=0.815, bottom=0.230, wspace=0.46)

# (a) Drawn as a displacement, so the geometry is the claim. The quantity on
# the axis is how far each parameter moved from the value it was set to, and
# the reference is a line at no movement at all. A parameter reported as
# learned that lands on that line has not been learned. The earlier version
# plotted the parameter itself against a grey tick for the initialisation,
# which put the whole claim in a two-pixel gap the reader had to hunt for and
# needed a prose key to decode; the initialisation is now in the row label,
# where every other constant in this set lives.
for i, (init, mean, sdv, sds) in enumerate(((0.5, aN, sN, seed_N), (1.0, aD, sD, seed_D))):
    a.plot([s - init for s in sds], [i] * len(sds), "o", color=GREY, ms=2.6,
           markeredgewidth=0, alpha=0.65, zorder=2)
    a.errorbar([mean - init], [i], xerr=[sdv], fmt="o", color=BLUE, zorder=3,
               **ERRBAR, **DOT)
    # Annotated in the unit the axis is ticked in. It used to read as a
    # percentage of the initialisation while the ticks read as absolute
    # movement, so the label and the position of the dot it labelled were two
    # different quantities, and the two rows did not even share a denominator
    # (one parameter was set to half of what the other was).
    a.text(mean - init + 0.008, i, f"{mean - init:+.3f}", fontsize=8.0,
           color=MUTE, va="center", ha="left")
a.axvline(0, **CHANCE_STYLE)
a.text(0.0, 1.66, "stated initialisation", fontsize=8.0, color=MUTE,
       ha="center", va="bottom")
a.set_yticks([1, 0])
a.set_yticklabels([r"$\alpha_D$" + "\n(set to 1.00)", r"$\alpha_N$" + "\n(set to 0.50)"],
                  fontsize=8.0, linespacing=1.25)
a.set_ylim(-0.62, 1.62); a.set_xlim(-0.080, 0.080)
a.set_xticks([-0.05, 0.0, 0.05])
a.set_xlabel("Movement from initialisation")
# No left spine: the axis is categorical, and a rule running past both rows is
# read as a bracket joining two parameters that are not a group.
lean(a, xbounds=(-0.05, 0.05))
a.spines["left"].set_visible(False)
a.tick_params(axis="y", length=0)

# (b) Three F1 values. No interval for any of them is committed anywhere: the
# per-pair scores these would be resampled from are not released, so there is
# no honest interval to draw and none is invented. They are therefore drawn as
# what they are, bare underpowered point estimates, in grey and labelled in
# grey's own text tone.
ys = [2, 1, 0]
b.plot(f1, ys, "o", color=GREY, linestyle="none", **DOT)
for y, v in zip(ys, f1):
    b.text(v + 0.055, y, f"{v:.2f}", fontsize=8.0, color=MUTE, va="center", ha="left")
b.set_yticks(ys)
b.set_yticklabels([f"HF, in domain\n(n={n_cand[0]:,})", f"sepsis\n(n={n_cand[1]:,})",
                   f"AKI\n(n={n_cand[2]:,})"], fontsize=8.0, linespacing=1.25)
b.set_ylim(-0.62, 2.62); b.set_xlim(-0.03, 1.30)
b.set_xticks([0, 0.5, 1.0])
b.set_xlabel(r"Rule-type $F_1$")
lean(b, xbounds=(0, 1.0))
b.spines["left"].set_visible(False)
b.tick_params(axis="y", length=0)

for ax, tag in ((a, "(a)"), (b, "(b)")):
    p = ax.get_position()
    fig.text(p.x0 - 0.055, p.y1 + 0.055, tag, ha="left", va="bottom",
             fontsize=8.0, style="italic", color=INK)
if SHOW_INFIGURE_NOTES:
    b.text(0.5, 2.9, "transfer values recovered; see Appendix C", fontsize=8.0,
           style="italic", ha="center", va="bottom", color=MUTE)
fig.savefig(HERE / "fig-falsification.pdf"); plt.close(fig)
man["fig:falsification"] = {"file": "fig-falsification.pdf",
    "source": "accord_supp/results/accord_v2_decontam_gold_aggregate.json (left); "
              "results/accord/generality/transfer_results_correct.json (right)",
    "values": {"alpha_n": aN, "alpha_d": aD, "alpha_n_per_seed": seed_N,
               "alpha_d_per_seed": seed_D, "rule_type_f1": f1,
               "n_candidates": n_cand,
               "alpha_init_stated_design_constant": [0.5, 1.0],
               "alpha_n_movement": aN - 0.5, "alpha_d_movement": aD - 1.0},
    "removed": "45.2% held-out rule-accuracy reference line (0.4515 is in no result file)"}

p = HERE / "manifest_extra.json"; p.write_text(json.dumps(man, indent=2))
print(json.dumps(man, indent=2))
