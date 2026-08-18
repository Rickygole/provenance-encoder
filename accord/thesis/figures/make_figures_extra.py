"""Two figures the first pass declined: the data for both is on disk, in the
Drive quarantine (from_drive_2026-08-08/) rather than results/accord/.

fig-asymmetry     -- fully sourced from hf_medication_mentions.csv. Not bound
                     into the thesis; kept buildable.
fig-falsification -- panel (a) sourced from the decontaminated 3-seed
                     aggregate; panel (b)'s transfer F1 read from
                     results/accord/generality/transfer_results_correct.json.

The 45.2% held-out rule-accuracy reference line that used to sit on the right
panel is gone: 0.4515 appears in no result file anywhere on disk.

Panel (a) is a dumbbell on a RELATIVE axis: each parameter's final value as a
multiple of the value it was initialised to. Both parameters were set to
different constants (0.5 and 1.0), so an absolute axis puts the two rows two
units apart and compresses the movement -- which is the claim -- to under a
point. On the relative axis both initialisations sit on 1.0, the tolerance band
is literally a percentage, and the two dumbbells are read against each other.

The two initialisations themselves are the one pair of values on any canvas in
this set with no committed numeric source. They are design constants quoted
from source code in CORRECTIONS.md C1, not measurements, and they are labelled
on the canvas as values the parameters were SET to.
"""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style
from style import (FIGW, INK, PAPER, GRAPHITE, MUTE, SLATE, MIST, HAIR,
                   COBALT, COBALT_DEEP, COBALT_CORE, COBALT_TINT, COBALT_BARE,
                   EMERALD_CORE, PLUM_CORE,
                   SCARLET_DEEP, SCARLET_CORE, SCARLET_TINT, SCARLET_WASH,
                   T_CLAIM, T_AXIS, T_TICK, T_SERIES, T_NOTE,
                   LW_DATA, LW_BAR, LW_INTERVAL, MS_DOT, MS_MARK, MS_SEED,
                   ERRBAR, DOT, canvas, frame, signature, panel_label, arrow,
                   case, cased)
setup = style.setup
BLUE = BLUE2 = COBALT_CORE
GREEN = EMERALD_CORE
PURPLE = PLUM_CORE
PALE = MIST
GREY = SLATE

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
a.set_title(f"all mentions, n = {n_all:,}", fontsize=T_SERIES, color=MUTE, pad=4)
frame(a)
a.spines["bottom"].set_visible(False)

b.bar([0, 1], [disch, nurs], color=BLUE, linewidth=0, width=0.55)
b.set_xticks([0, 1]); b.set_xticklabels(["discharge", "nursing"], fontsize=8.0)
for x, v in enumerate([disch, nurs]):
    b.text(x, v + n_clf * .02, f"{v:,}", ha="center", fontsize=8.0, color=INK)
b.set_ylabel("Action-classified mentions")
b.set_title(f"by note role, {disch/nurs:.1f} to 1", fontsize=8.0, color=MUTE, pad=4)
b.set_ylim(0, disch * 1.20)
frame(b)
fig.savefig(HERE / "fig-asymmetry.pdf"); plt.close(fig)
man["fig:asymmetry"] = {"file": "fig-asymmetry.pdf",
    "source": "from_drive_2026-08-08/outputs/hf_medication_mentions.csv",
    "values": {"total": n_all, "classified": n_clf, "discharge": disch,
               "nursing": nurs, "ratio": round(disch/nurs, 3)}}

# ---- fig:falsification -----------------------------------------------------
agg_all = json.loads((S / "accord_v2_decontam_gold_aggregate.json").read_text())
agg = agg_all["aggregate"]
aN, aD = agg["alpha_n"]["mean"], agg["alpha_d"]["mean"]
seeds = sorted(agg_all["per_seed"])
seed_N = [agg_all["per_seed"][s_]["authority_scale"]["alpha_nursing"] for s_ in seeds]
seed_D = [agg_all["per_seed"][s_]["authority_scale"]["alpha_discharge"] for s_ in seeds]
INIT_N, INIT_D = 0.5, 1.0          # design constants, CORRECTIONS.md C1

transfer = json.loads((REPO / "results/accord/generality/transfer_results_correct.json").read_text())
f1 = [transfer["hf_reference"]["rule_type_f1"], transfer["sepsis"]["rule_type_f1"],
      transfer["aki"]["rule_type_f1"]]
n_cand = [transfer["hf_reference"]["n_candidates"], transfer["sepsis"]["n_candidates"],
          transfer["aki"]["n_candidates"]]

fig, (a, b) = plt.subplots(1, 2, figsize=(FIGW, 2.68),
                           gridspec_kw={"width_ratios": [1.06, 1.0]})
canvas(fig)
fig.subplots_adjust(left=0.1172, right=0.9863, top=0.7413, bottom=0.2148,
                    wspace=0.34)

# ---- (a) the dumbbell ------------------------------------------------------
# WHAT CAME OFF THIS PANEL, and why. The author called this figure busy. Panel
# (a) carried a tinted band, a two-line band label, two dumbbells, three seed
# dots per row, a percentage at each dumbbell's head and a two-line row label:
# six kinds of thing on two rows of data. Two of them said the same sentence
# twice. The band label read "neither left a 2% band" and the row labels read
# "+0.4%" and "+1.5%", which is the same claim in prose and in figures, and the
# figures are the measurement. THE PROSE WENT AND THE NUMBERS STAYED. The band
# stays as a ground and is named on the axis instead, where a tolerance belongs;
# the seed dots stay, because three seeds landing on top of each other is
# evidence and not decoration.
TOL = 0.02
# WASH, not TINT. The band covers three quarters of this panel, and at TINT
# that is a saturated field with two grey dumbbells on it: the ground was
# louder than the data it exists to locate. At WASH it is the same ground
# panel (b) uses, so the two panels now say "this is where it fails" in one
# tone, and it costs nothing measurable -- SCARLET_TINT sits at HSV saturation
# 0.233, under the 0.25 gate the ink tool counts at, so it registered as
# chroma 0 either way.
a.axvspan(1 - TOL, 1 + TOL, facecolor=SCARLET_WASH, edgecolor="none",
          linewidth=0, zorder=0)
moved = []
for i, (init, mean, sds) in enumerate(((INIT_N, aN, seed_N),
                                       (INIT_D, aD, seed_D))):
    ratio = mean / init
    moved.append(abs(ratio - 1))
    a.plot([1.0, ratio], [i, i], color=GRAPHITE, lw=2.4,
           solid_capstyle="butt", zorder=3)
    # The three seeds spread 10.8pt about a 6.6pt marker, so on the dumbbell's
    # own line they were neither hidden nor visible: a grey crescent behind the
    # final dot that reads as a print smudge. They drop to their own sub-row,
    # true x, where three dots read as three seeds.
    a.plot([s_ / init for s_ in sds], [i - 0.30] * len(sds), "o", color=SLATE,
           ms=MS_SEED, markeredgewidth=0, zorder=5)
    a.plot([1.0], [i], "o", markerfacecolor=PAPER, markeredgecolor=GRAPHITE,
           markeredgewidth=1.2, ms=MS_DOT, zorder=4)
    a.plot([ratio], [i], "o", color=GRAPHITE, ms=MS_DOT, zorder=6)
    # GRAPHITE at 9pt: the movement is a measured quantity that is not an
    # entity, the dumbbell it labels is drawn in GRAPHITE, and a series is
    # labelled in its own mark's tone. It was 8.5pt MUTE, which is the tone for
    # secondary annotation, and this is the panel's primary measurement.
    a.text(ratio + 0.0030, i, f"{(ratio - 1) * 100:+.1f}%",
           fontsize=T_SERIES, color=GRAPHITE, va="center", ha="left",
           zorder=6)
a.set_yticks([1, 0])
a.set_yticklabels([r"$\alpha_D$" + "\n(set to 1.00)",
                   r"$\alpha_N$" + "\n(set to 0.50)"],
                  fontsize=T_SERIES, linespacing=1.25)
a.set_ylim(-0.60, 1.45)
a.set_xlim(0.9755, 1.0295)
a.set_xticks([0.98, 0.99, 1.00, 1.01, 1.02])
a.set_xticklabels(["0.98", "0.99", "1.00", "1.01", "1.02"])
# The band is named on the axis, not on the field. A tolerance is a property
# of the scale, and the two tick labels the band runs between already read
# 0.98 and 1.02, so this line confirms what the axis shows instead of asserting
# a second time what the two percentages beside the dumbbells already measure.
a.set_xlabel("Final value, as a multiple of the initialisation\n"
             f"(shaded band: within {TOL * 100:.0f}% of it)",
             linespacing=1.25)
a.tick_params(axis="y", length=0)
frame(a)

# ---- (b) the cliff ---------------------------------------------------------
# Cohorts are not entities, so all three columns take one hue. The two the head
# was never fitted on are BARE: no interval for either is committed anywhere,
# because the per-pair scores they would be resampled from are not released.
xs = [0, 1, 2]
fills = [COBALT_TINT, COBALT_BARE, COBALT_BARE]
b.axvspan(0.5, 2.6, facecolor=SCARLET_WASH, edgecolor="none", linewidth=0,
          zorder=0)
b.bar(xs, f1, width=0.62, color=fills, edgecolor=COBALT_DEEP,
      linewidth=LW_BAR, zorder=3)
# MUTE, not SCARLET_DEEP. It names the ground it sits on, the ground is a
# WASH and outside the scarlet budget, and the drop arrow below is this panel's
# one saturated scarlet element.
case(b.text(1.5, 0.985, "not fitted here", ha="center", va="top",
            fontsize=T_NOTE, color=MUTE, zorder=5))
# THE TWO 22pt NUMERALS CAME OFF. They read 0.96 and 0.07, set one above the
# other beside a y axis that already runs 0 to 1.00 in quarters with the two
# columns sitting on those exact values: the same two numbers, twice, a
# centimetre apart. The author called the panel busy and this is what made it
# so. The drop stays, because the drop is the geometry that carries the finding
# and it reads at thumbnail size: a column at the top of the frame, a scarlet
# line falling almost to the floor, and two columns that are barely there.
# It starts at the fitted column's top right corner rather than at its centre:
# from the centre the shaft ran down through the column it is measuring from.
arrow(b, (0.34, f1[0] - 0.015), (0.97, f1[1] + 0.055), color=SCARLET_CORE,
      lw=2.2, mutation_scale=15, zorder=6)
b.set_xticks(xs)
b.set_xticklabels([f"HF,\nin domain\nn = {n_cand[0]:,}",
                   f"sepsis\nn = {n_cand[1]:,}",
                   f"AKI\nn = {n_cand[2]:,}"], fontsize=T_NOTE,
                  linespacing=1.28)
b.set_xlim(-0.62, 2.6)
b.set_ylim(0, 1.02)
b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
b.set_ylabel(r"Rule-type $F_1$")
frame(b)

for ax_, tag in ((a, "(a)"), (b, "(b)")):
    panel_label(fig, ax_, tag)
signature(fig, SCARLET_DEEP,
          "Two parameters reported as learned did not move;\n"
          "the rule head does not transfer.")
fig.savefig(HERE / "fig-falsification.pdf"); plt.close(fig)
man["fig:falsification"] = {"file": "fig-falsification.pdf",
    "source": "accord_supp/results/accord_v2_decontam_gold_aggregate.json (a); "
              "results/accord/generality/transfer_results_correct.json (b)",
    "values": {"alpha_n": aN, "alpha_d": aD, "alpha_n_per_seed": seed_N,
               "alpha_d_per_seed": seed_D, "rule_type_f1": f1,
               "n_candidates": n_cand,
               "alpha_init_stated_design_constant": [INIT_N, INIT_D],
               "alpha_n_ratio": aN / INIT_N, "alpha_d_ratio": aD / INIT_D},
    "removed": "45.2% held-out rule-accuracy reference line (0.4515 is in no result file)"}

p = HERE / "manifest_extra.json"; p.write_text(json.dumps(man, indent=2))
print(json.dumps(man, indent=2))
