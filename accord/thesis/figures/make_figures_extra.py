"""Two figures the first pass declined: the data for both is on disk, in the
Drive quarantine (from_drive_2026-08-08/) rather than results/accord/.

fig-asymmetry     -- fully sourced from hf_medication_mentions.csv. Not bound
                     into the thesis; kept buildable.
fig-falsification -- single panel, sourced from the decontaminated 3-seed
                     aggregate. It used to carry a second panel drawing the
                     rule head's transfer F1 from three cohorts as bars from
                     zero with no interval; a reviewer flagged that as the one
                     place in the set where a quantity with real sampling
                     spread was drawn with none, and separately rejected the
                     two panels as sharing a caption without sharing a claim.
                     That data moved to a table the author builds directly in
                     the .tex; this script no longer reads
                     transfer_results_correct.json and the figure is now a
                     single coherent claim: two parameters reported as
                     learned did not move.

The dumbbell is drawn on a RELATIVE axis: each parameter's final value as a
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
                   COBALT, COBALT_CORE,
                   EMERALD_CORE, PLUM_CORE,
                   SCARLET_DEEP, SCARLET_TINT, SCARLET_WASH,
                   T_CLAIM, T_AXIS, T_TICK, T_SERIES,
                   LW_DATA, LW_INTERVAL, MS_DOT, MS_MARK, MS_SEED,
                   ERRBAR, DOT, canvas, frame, signature,
                   cased)
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

# ---------------------------------------------------------------------------
# BOTH FIGURES IN THIS MODULE ARE RETIRED and it stops here.
#
#   fig-asymmetry     was already retired: the asymmetry it drew is two numbers
#                     in the text, and a two-point figure was never worth a
#                     float.
#   fig-falsification is retired on that same recorded precedent, which it
#                     matches exactly. It is two numbers on an axis running
#                     0.98 to 1.02, so a move of 1.5 percent is drawn as a
#                     distance the width of the panel. The axis does the
#                     arguing, not the data, and the two ratios are printed in
#                     the results chapter.
#
# Reasons of record live in tools/check_figure_invariants.py; the committed
# PDFs stay in place, as every other retired figure's does.
# ---------------------------------------------------------------------------
print("make_figures_extra.py: both figures retired; see RETIRED in "
      "tools/check_figure_invariants.py")
raise SystemExit(0)
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

# The transfer F1 panel that used to sit here (a bar chart of rule-head F1
# across three cohorts, drawn from zero with no interval) moved to a table the
# author builds in the .tex, so this script no longer reads
# transfer_results_correct.json. This is now a single-panel figure: one
# canvas, one claim, no gutter to widen or narrow.
fig, a = plt.subplots(figsize=(FIGW, 2.35))
canvas(fig)
# Height dropped from 2.68in to 2.35in: the removed panel's own width, not its
# height, funded the freed space, so the cut here is the vertical budget a
# two-line "(a)...(b)..." headline and a panel-letter label no longer need
# (one declarative headline sits at 0.19in of text against the old 0.4in of
# two lines, and there is no second panel_label() call to clear at all). The
# axes keep their former height (top-bottom fraction * 2.35in = 1.35in,
# against 1.41in before): the dumbbell's own row spacing did not shrink. What
# widened is the axes: with no second panel and no wspace gutter to share the
# canvas with, this panel now runs left=0.1172 (unchanged absolute margin,
# same width class) to right=0.9850, versus roughly half that width before.
fig.subplots_adjust(left=0.1172, right=0.9850, top=0.8181, bottom=0.2450)

# ---- the dumbbell -----------------------------------------------------------
# WHAT CAME OFF THIS PANEL, and why. The author called this figure busy. This
# panel carried a tinted band, a two-line band label, two dumbbells, three seed
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
# louder than the data it exists to locate. It costs nothing measurable --
# SCARLET_TINT sits at HSV saturation 0.233, under the 0.25 gate the ink tool
# counts at, so it registered as chroma 0 either way.
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

signature(fig, SCARLET_DEEP,
          "Two parameters reported as learned did not move.")
fig.savefig(HERE / "fig-falsification.pdf"); plt.close(fig)
man["fig:falsification"] = {"file": "fig-falsification.pdf",
    "source": "accord_supp/results/accord_v2_decontam_gold_aggregate.json",
    "values": {"alpha_n": aN, "alpha_d": aD, "alpha_n_per_seed": seed_N,
               "alpha_d_per_seed": seed_D,
               "alpha_init_stated_design_constant": [INIT_N, INIT_D],
               "alpha_n_ratio": aN / INIT_N, "alpha_d_ratio": aD / INIT_D},
    "removed": "the rule-head transfer F1 panel (moved to a table in the .tex; "
               "see results/accord/generality/transfer_results_correct.json) "
               "and the 45.2% held-out rule-accuracy reference line dropped in "
               "an earlier pass (0.4515 is in no result file)"}

p = HERE / "manifest_extra.json"; p.write_text(json.dumps(man, indent=2))
print(json.dumps(man, indent=2))
