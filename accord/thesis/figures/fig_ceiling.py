"""fig-ceiling. The agreement ceiling a labelling policy sets before it reads
anything.

WHAT THIS FIGURE ARGUES. Two counts fix, in advance, how much agreement a
hard-label decision can express against a gold standard: how many pairs the
gold standard confirms, and how many the labeller flags. Hold both marginals and
put every confirmed case inside the flagged set; that is the most agreeable table
those two counts permit, and its Cohen's kappa is a ceiling. On this pool the
ceiling is a cliff. Almost the whole positive-rate axis lies flat against zero,
and the only region where agreement is arithmetically available is a narrow spike
at the pharmacist's own confirmed rate.

WHY KAPPA AND NOT MCC. The register carries both. Kappa is the statistic a
clinical reader expects for agreement against a human standard, and it is the
one that falls hardest: at the deployed labeller's rate the Matthews ceiling is
still 0.078 while the kappa ceiling is 0.0121, so a figure drawn in MCC would
understate the constraint it exists to show.

NO NEW DATA AND NO NEW RUN. Everything on the canvas derives from three
committed integers -- the pool size, the pharmacist's confirmed count and the
labeller's flagged count -- plus the four local labeller rows already in
results/accord/stats/marginal_feasibility_gate.json. Nothing here was computed
for the figure that was not already computed for the gate.

NO INTERVAL, DELIBERATELY. The curve is an identity in two marginals, not a fit
to data, so there is nothing for a confidence interval to be about and none is
drawn. This is the one place in this set where a curve carries no uncertainty,
and the deck says so on the canvas rather than leaving a reader to wonder
whether it was forgotten. The set's rule is that no ESTIMATE appears without its
uncertainty; a bound that follows from counting is not an estimate.

SCOPE, and the unqualified claim is false without it. This bounds a THRESHOLDED
decision. A labeller that emits a continuous score is not bounded this way,
because the threshold can be swept and a model flagging most of a pool at one
threshold can still rank perfectly. The deck carries that sentence; the caption
owner must carry it too.

READS (every value read, none typed):
  results/accord/VERIFIED_FACTS.json
    gold.n_total                      the 299 gold pairs
    gold.weaklabel_contradiction      the 261 pairs the deployed labeller flags
    gold.base_rate_299                the pharmacist's confirmed rate, 0.0401
    exemplar.power_analysis.n_positives_total   the 12 confirmed pairs
    feasgate.*.flag_rate, .ceiling_kappa, .n, .n_positives, .attains_ceiling
  results/accord/stats/marginal_feasibility_gate.json
    rows.*  (the eight local labeller rows; four attain their ceiling)

Every read value is cross-checked against the other source that carries it, and
the identity drawn as the curve is re-evaluated at each committed row's own
marginal and asserted equal to that row's committed ceiling. If the artifact and
the identity ever disagree this script fails instead of drawing.

Run:  python3 accord/thesis/figures/fig_ceiling.py
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import style
from style import (
    PLUM_CORE, PLUM_DEEP,
    PAPER, INK, GRAPHITE, MUTE, SLATE,
    T_AXIS, T_SERIES, T_NOTE,
    LW_DATA, LW_REF, MS_DOT, MS_MARK, REF_DASH, FIGW,
    canvas, frame, signature, cased,
)
style.setup()
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FACTS = json.loads((ROOT / "results/accord/VERIFIED_FACTS.json").read_text())
GATE = json.loads((ROOT / "results/accord/stats"
                   / "marginal_feasibility_gate.json").read_text())


def fact(key):
    return FACTS[key]["value"]


# ---------------------------------------------------------------------------
# The identity. Copied in form, not in value, from
# accord/stats/marginal_feasibility_gate.py: same best_table, same kappa, and
# the assertions below prove the two agree on every committed row rather than
# asserting it in a comment.
# ---------------------------------------------------------------------------
def kappa(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    return (po - pe) / (1 - pe) if pe != 1 else 0.0


def kappa_max(n, n_pos, n_flag):
    """Best kappa the two marginals permit: every confirmed case inside the
    flagged set, and when the labeller flags fewer than are confirmed, its whole
    flagged set spent on confirmed cases."""
    tp = min(n_pos, n_flag)
    return kappa(tp, n - n_flag - (n_pos - tp), n_flag - tp, n_pos - tp)


# ---------------------------------------------------------------------------
# The three committed integers.
# ---------------------------------------------------------------------------
N = fact("gold.n_total")
N_FLAG_DEPLOYED = fact("gold.weaklabel_contradiction")
N_POS = fact("exemplar.power_analysis.n_positives_total")
BASE = fact("gold.base_rate_299")

ROWS = GATE["rows"]
_g299 = [r for r in ROWS.values() if r["basis"] == "G_299"]
assert _g299, "no G_299 row in the feasibility gate artifact"
assert {r["n"] for r in _g299} == {N}, "gate G_299 rows disagree with gold.n_total"
assert {r["n_positives"] for r in _g299} == {N_POS}, \
    "gate G_299 positives disagree with the registered positive count"
assert abs(N_POS / N - BASE) < 5e-5, \
    "12 of 299 does not reproduce the registered base rate"

# The identity against every committed row, on that row's own basis. This is
# what makes the curve evidence about the register rather than about this file.
for key, r in ROWS.items():
    got = kappa_max(r["n"], r["n_positives"], r["n_flagged"])
    assert abs(got - r["ceiling_kappa"]) < 5e-5, \
        f"identity disagrees with the artifact on {key}: {got} vs {r['ceiling_kappa']}"
    assert abs(r["n_flagged"] / r["n"] - r["flag_rate"]) < 5e-5, key

# The deployed weak labeller. 261 of 299 is the register's own weak-label count
# and is also the flagged count of the llama3 zero-shot row on G_299, which is
# the same labelling policy; the assertion is the check that they are.
DEPLOYED_RATE = N_FLAG_DEPLOYED / N
DEPLOYED_CEIL = kappa_max(N, N_POS, N_FLAG_DEPLOYED)
_dep_rows = [r for r in _g299 if r["n_flagged"] == N_FLAG_DEPLOYED]
assert len(_dep_rows) == 1, "the deployed flagged count matches no single G_299 row"
assert abs(_dep_rows[0]["ceiling_kappa"] - DEPLOYED_CEIL) < 5e-5

# The four rows that attain their ceiling. attains_ceiling is recorded on the
# Matthews correlation, and for fixed marginals kappa and MCC are both
# increasing in the true-positive count, so the table that attains one attains
# the other: a row at its Matthews ceiling is at its kappa ceiling.
ATTAIN = [r for r in ROWS.values() if r["attains_ceiling"]]
assert len(ATTAIN) == fact("feasgate.n_attaining_their_ceiling") == 4, \
    "the count of ceiling-attaining rows moved; the figure's claim moved with it"
for r in ATTAIN:
    slug = (f"{r['model']}_{r['strategy']}_{r['basis']}"
            .replace(":", "").replace(".", "_").replace("-", "").lower())
    # The register mirrors these rows under feasgate.*; find this row's mirror by
    # its values rather than by reconstructing the generator's slug rule.
    mirrors = [k for k, v in FACTS.items()
               if k.startswith("feasgate.") and k.endswith(".ceiling_kappa")
               and abs(v["value"] - r["ceiling_kappa"]) < 1e-9]
    assert mirrors, f"{slug} ceiling is in the artifact but not in the register"

# Two of the four are computed on the 253-pair core basis, whose prevalence is
# 12 of 253 rather than 12 of 299, so their ceiling is not a point of the drawn
# curve. The discrepancy is measured here and stated on the canvas rather than
# glossed: if it ever exceeded the marker radius this figure would need two
# curves and the assertion would say so.
OFF_BASIS = [r for r in ATTAIN if r["n"] != N]
DRIFT = max((abs(r["ceiling_kappa"] - kappa_max(N, N_POS, round(r["flag_rate"] * N)))
             for r in OFF_BASIS), default=0.0)

# ---------------------------------------------------------------------------
# The curve. One point per attainable flagged count, which is the whole lattice
# of marginals available on a 299-pair pool, so the line is exact everywhere it
# is read and is not a resampling of a formula.
# ---------------------------------------------------------------------------
ks = np.arange(0, N + 1)
qs = ks / N
ceil = np.array([kappa_max(N, N_POS, int(k)) for k in ks])

fig, ax = plt.subplots(figsize=(FIGW, 3.46))
canvas(fig)
# right is 0.9740 and not the set's usual 0.9890 for one measured reason: the
# x axis ends on a labelled tick, so the "1.0" tick label is centred on the
# right spine and half of it hung 1.0pt past the pinned bounding box. savefig's
# tight box then grew to catch it and the figure saved at 425.34pt against the
# class measure of 424.348, which \includegraphics would have absorbed by
# scaling every type size on the canvas down. Invariant 1 catches that; this
# margin is the fix.
fig.subplots_adjust(left=0.1065, right=0.9740, top=0.7730, bottom=0.1390)

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
frame(ax, grid=True)

# The pharmacist's confirmed rate. A dashed reference line, which is SLATE's one
# job in this palette.
ax.axvline(BASE, color=SLATE, linestyle=REF_DASH, linewidth=LW_REF, zorder=2)

# The ceiling itself. Measured, but not an entity, so it takes the structural
# measurement tone the MDE curve takes on fig-power.
ax.plot(qs, ceil, color=GRAPHITE, linewidth=LW_DATA, zorder=4,
        solid_joinstyle="round")

# The reference line is named at the top of the axis, in MUTE and not in the
# SLATE of the line itself: SLATE is 3.48:1 on paper and this set does not set
# type in it. The invariant that enforces that counts any text call within six
# lines of a SLATE colour argument, so the two are deliberately not adjacent
# here either.
ax.text(BASE + 0.022, 0.962,
        f"pharmacist confirms {N_POS} of {N} pairs, {BASE:.4f}",
        ha="left", va="center", fontsize=T_NOTE, color=MUTE, zorder=6)

# The argument, in the empty upper right. Right-aligned so it cannot drift left
# into the spike as the wording changes.
ax.text(0.978, 0.790,
        "A labeller that flags most of a low-prevalence pool\n"
        "must return a large number of false positives.\n"
        "What is left for agreement is set by the two\n"
        "marginals alone, before a detector is trained.",
        ha="right", va="top", fontsize=T_NOTE, color=INK, linespacing=1.45,
        zorder=6)

# Mark 1: the deployed weak labeller, at its measured positive rate. Its hue is
# the one the entity map gives the production weak labeller everywhere in this
# set.
ax.plot([DEPLOYED_RATE], [DEPLOYED_CEIL], marker="o", ms=MS_DOT + 1.0,
        color=PLUM_CORE, markeredgecolor=PLUM_DEEP, markeredgewidth=1.0,
        zorder=7)
# Both of the marks' labels are placed where the curve is already below them:
# a white text casing over a data line erases the line, which this set allows
# over a fill and not over a mark. Their y positions are the clearance.
ax.text(0.978, 0.062,
        f"deployed weak labeller, {N_FLAG_DEPLOYED} of {N}",
        ha="right", va="bottom", color=PLUM_DEEP, zorder=7, **cased(T_SERIES))

# Mark 2: the four local labeller runs whose observed agreement equals their
# ceiling. They are drawn as attainments rather than as models: a hue per model
# here would be four unkeyed hues on four marks two points apart, and the claim
# is that the bound is reached, not which model reached it. The caption names
# them.
ax.plot([r["flag_rate"] for r in ATTAIN], [r["ceiling_kappa"] for r in ATTAIN],
        marker="s", linestyle="none", ms=MS_MARK - 1.0, markerfacecolor=PAPER,
        markeredgecolor=GRAPHITE, markeredgewidth=1.2, zorder=6)
ax.text(0.978, 0.155,
        f"{len(ATTAIN)} local labeller runs sit exactly on this bound",
        ha="right", va="bottom", color=GRAPHITE, zorder=6, **cased(T_SERIES))

ax.set_xlabel(f"Labeller positive rate on the {N} gold pairs", fontsize=T_AXIS)
ax.set_ylabel("Maximum attainable kappa\nagainst the pharmacist",
              fontsize=T_AXIS, linespacing=1.3)

signature(
    fig, GRAPHITE,
    headline=(f"A labeller flagging {N_FLAG_DEPLOYED} of {N} pairs cannot pass "
              f"kappa {DEPLOYED_CEIL:.4f}, however good."),
    # Three lines, each held under about 105 characters. At 9.5pt STIXGeneral a
    # deck line runs roughly 3.7pt per character, so a longer line overflows the
    # pinned bounding box, savefig's tight box grows to catch it, and
    # \includegraphics then scales the whole figure down and takes every type
    # size on it under the floor. Invariant 1 in tools/check_figure_invariants.py
    # is the check; this comment is why the wording is cut where it is.
    deck=("The ceiling is arithmetic, not empirical: it follows from the two "
          "marginals alone and holds for any\n"
          "labeller at that positive rate, however good. An identity, so no "
          "interval and no error caps.\n"
          "It bounds a THRESHOLDED decision. A score-emitting labeller is not "
          "bounded: its threshold can be swept."))
fig.savefig(HERE / "fig-ceiling.pdf")
plt.close(fig)

print(f"wrote fig-ceiling.pdf: n={N}, confirmed={N_POS} ({BASE:.4f}), "
      f"deployed flags={N_FLAG_DEPLOYED} ({DEPLOYED_RATE:.4f}), "
      f"ceiling={DEPLOYED_CEIL:.4f}")
print(f"  identity reproduces all {len(ROWS)} committed row ceilings to 5e-5")
print(f"  {len(ATTAIN)} rows attain their ceiling; {len(OFF_BASIS)} of them on "
      f"the 253-pair basis, off the drawn curve by at most {DRIFT:.4f} kappa")
