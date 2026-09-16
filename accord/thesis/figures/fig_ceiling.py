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

WHY MCC AND NOT KAPPA (round 7, fixes_r7 N9). The thesis's contribution is
stated in the Matthews correlation: 0.0780 at 261 flagged of 299. The figure used
to headline the kappa ceiling, 0.0121, which is a different number from the one
every chapter cites, so a reader checking the figure against the text found two
ceilings. It is now drawn in MCC throughout. Each labeller row is drawn against
the ceiling curve of its OWN basis (G_299, 12 of 299; G_253, 12 of 253), where it
used to sit on the 299 curve whatever its basis, and the deployed labeller's
measured agreement from the C120 exact join (MCC 0.0269) is drawn under its own
ceiling, because a ceiling with nothing measured against it is half the check.

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
    feasgate.*.flag_rate, .ceiling_mcc, .n, .n_positives, .attains_ceiling
    c120.g299.n, .n_flagged, .n_positives, .mcc, .ceiling_mcc   the measured point
  results/accord/stats/marginal_feasibility_gate.json
    rows.*  (the eight local labeller rows; four attain their ceiling)

Every read value is cross-checked against the other source that carries it, and
the identity drawn as the curve is re-evaluated at each committed row's own
marginal and asserted equal to that row's committed ceiling. If the artifact and
the identity ever disagree this script fails instead of drawing.

Run:  python3 accord/thesis/figures/fig_ceiling.py   (calls make_fig_ceiling())
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import style
from style import (
    WEAK_LABELLER,
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
    assert not key.startswith("withdrawn."), f"{key} is withdrawn, not citable"
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


def mcc(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return num / den if den else 0.0


def best_table(n, n_pos, n_flag):
    tp = min(n_pos, n_flag)
    return tp, n - n_flag - (n_pos - tp), n_flag - tp, n_pos - tp


def mcc_max(n, n_pos, n_flag):
    """Best Matthews correlation the two marginals permit, on the same best
    table kappa_max uses."""
    return mcc(*best_table(n, n_pos, n_flag))


def kappa_max(n, n_pos, n_flag):
    """Best kappa the two marginals permit: every confirmed case inside the
    flagged set, and when the labeller flags fewer than are confirmed, its whole
    flagged set spent on confirmed cases."""
    tp = min(n_pos, n_flag)
    return kappa(tp, n - n_flag - (n_pos - tp), n_flag - tp, n_pos - tp)



def make_fig_ceiling():
    # -----------------------------------------------------------------------
    # The committed integers, read, never typed.
    # -----------------------------------------------------------------------
    N = fact("gold.n_total")
    N_POS = fact("exemplar.power_analysis.n_positives_total")
    BASE = fact("gold.base_rate_299")
    N_FLAG = fact("c120.g299.n_flagged")
    assert fact("c120.g299.n") == N and fact("c120.g299.n_positives") == N_POS
    assert N_FLAG == fact("gold.weaklabel_contradiction")
    MCC_OBS = fact("c120.g299.mcc")
    MCC_CEIL = fact("c120.g299.ceiling_mcc")
    assert abs(mcc_max(N, N_POS, N_FLAG) - MCC_CEIL) < 5e-9, \
        "the identity disagrees with C120's own ceiling"
    assert abs(N_POS / N - BASE) < 5e-5

    ROWS = GATE["rows"]
    bases = {}
    for key, r in ROWS.items():
        assert r["n_positives"] == N_POS, key
        assert abs(mcc_max(r["n"], r["n_positives"], r["n_flagged"])
                   - r["ceiling_mcc"]) < 5e-5, f"identity disagrees on {key}"
        assert abs(kappa_max(r["n"], r["n_positives"], r["n_flagged"])
                   - r["ceiling_kappa"]) < 5e-5, key
        bases.setdefault(r["basis"], r["n"])
        # Every row on a basis must share its n, not only the first one seen.
        assert r["n"] == bases[r["basis"]], f"{key}: n {r['n']} disagrees with basis {r['basis']}"
    assert bases == {"G_299": N, "G_253": fact("c120.g253.n")}, bases

    # Rows drawn: the ones that attain their ceiling. Each must be mirrored in
    # the register under feasgate.* and must not be withdrawn; a row the
    # register stops carrying drops off this canvas without an edit here.
    ATTAIN = []
    for r in ROWS.values():
        if not r["attains_ceiling"]:
            continue
        mirrors = [k for k, v in FACTS.items()
                   if k.startswith("feasgate.") and k.endswith(".ceiling_mcc")
                   and abs(v["value"] - r["ceiling_mcc"]) < 1e-9]
        if mirrors:
            ATTAIN.append(r)
    assert len(ATTAIN) == fact("feasgate.n_attaining_their_ceiling") == 4
    for r in ATTAIN:
        assert abs(r["observed_mcc"] - r["ceiling_mcc"]) < 5e-5

    def curve(n):
        ks = np.arange(0, n + 1)
        return ks / n, np.array([mcc_max(n, N_POS, int(k)) for k in ks])

    q299, c299 = curve(N)
    q253, c253 = curve(bases["G_253"])
    DASH_253 = (0, (4.0, 2.2))

    fig = plt.figure(figsize=(FIGW, 3.30))
    canvas(fig)
    axL = fig.add_axes([0.098, 0.150, 0.335, 0.690])
    axR = fig.add_axes([0.580, 0.150, 0.394, 0.690])

    # Left: the whole cliff. Right: the enlarged region under the zoom box.
    ZX0, ZX1, ZY1 = 0.66, 1.0, 0.18
    # Above both ceilings no table with these marginals exists, so that region
    # is shaded: it is what the identity says, not a band added for ink.
    qq = np.linspace(0, 1, 2001)
    upper = np.maximum(np.interp(qq, q299, c299), np.interp(qq, q253, c253))
    for ax, top in ((axL, 1.0), (axR, ZY1)):
        ax.fill_between(qq, upper, top, facecolor=style.FOG, edgecolor="none",
                        linewidth=0, zorder=0)
    for ax in (axL, axR):
        ax.plot(q299, c299, color=GRAPHITE, lw=LW_DATA, zorder=4,
                solid_joinstyle="round")
        ax.plot(q253, c253, color=GRAPHITE, lw=LW_REF + 0.2, zorder=3,
                linestyle=DASH_253)
    # A small margin on every side, so the zoom box's bottom and right edges
    # do not sit on the spines.
    axL.set_xlim(-0.02, 1.02); axL.set_ylim(-0.02, 1.0)
    axL.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axL.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    axL.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    frame(axL, grid=True)
    axL.add_patch(plt.Rectangle((ZX0, 0), ZX1 - ZX0, ZY1, facecolor="none",
                                edgecolor=SLATE, lw=0.6, zorder=6))
    axL.text((ZX0 + ZX1) / 2, ZY1 + 0.03, "enlarged at right", ha="center",
             va="bottom", fontsize=T_NOTE, color=MUTE, style="italic")
    axL.text(BASE + 0.05, 0.93, f"peak at the pharmacist's\nrate, {N_POS} of {N}",
             ha="left", va="top", fontsize=T_NOTE, color=MUTE, linespacing=1.3)
    axL.text(0.60, 0.62, "not attainable\non either basis", ha="center",
             va="center", fontsize=T_NOTE, color=MUTE, linespacing=1.2)
    axL.set_ylabel("Largest attainable MCC\nagainst the pharmacist",
                   fontsize=T_AXIS, linespacing=1.3)

    # AUDIT ITEM 8. The two curves were named only in the enlarged right
    # panel; a reader looking at the left panel alone had no way to tell the
    # solid curve from the dashed one. The cliff is too steep for a label
    # sitting directly beside a curve point to clear the line over the
    # label's own width, and the ground above the cliff is already busy with
    # the peak and "not attainable" annotations, so each name drops straight
    # down instead, on a short leader, into the empty paper below the tail
    # of its own curve, each at its own x so the two leaders do not cross.
    N_253_L = bases["G_253"]
    for x_pt, y_lab, txt in ((0.20, 0.16, f"{N_POS} of {N_253_L}"),
                             (0.32, 0.06, f"{N_POS} of {N}")):
        c_arr = c253 if txt.endswith(str(N_253_L)) else c299
        q_arr = q253 if txt.endswith(str(N_253_L)) else q299
        y_pt = float(np.interp(x_pt, q_arr, c_arr))
        axL.plot([x_pt, x_pt], [y_pt - 0.012, y_lab + 0.014], color=GRAPHITE,
                 lw=0.6, zorder=6, solid_capstyle="butt")
        axL.text(x_pt + 0.012, y_lab, txt, ha="left", va="center",
                 fontsize=T_NOTE, color=INK, zorder=8)

    axR.set_xlim(ZX0, ZX1); axR.set_ylim(0, ZY1)
    axR.set_xticks([0.7, 0.8, 0.9, 1.0])
    axR.set_yticks([0, 0.04, 0.08, 0.12, 0.16])
    frame(axR, grid=True)
    axR.set_ylabel("Largest attainable MCC (enlarged)", fontsize=9.5)

    # The two curves are named at their left ends in the enlarged panel, where
    # they are furthest apart; each name is built from its basis's counts.
    N_253 = bases["G_253"]
    x_lab = ZX0 + 0.010
    # Above the dashed curve at the label's left end; below the solid curve at
    # its right end (the curves fall to the right).
    y253 = float(np.interp(x_lab, q253, c253))
    y299 = float(np.interp(x_lab + 0.075, q299, c299))
    axR.text(x_lab, y253 + 0.005, f"{N_POS} of {N_253}",
             ha="left", va="bottom", fontsize=T_SERIES, color=INK, zorder=8)
    axR.text(x_lab, y299 - 0.008, f"{N_POS} of {N}",
             ha="left", va="top", fontsize=T_SERIES, color=INK, zorder=8)

    # The deployed labeller: its ceiling on the 299 pairs, and what it measured.
    q_dep = N_FLAG / N
    axR.plot([q_dep, q_dep], [MCC_OBS, MCC_CEIL], color=WEAK_LABELLER, lw=1.4,
             zorder=5, solid_capstyle="butt")
    axR.plot([q_dep], [MCC_CEIL], marker="o", ms=MS_DOT, markerfacecolor=PAPER,
             markeredgecolor=WEAK_LABELLER, markeredgewidth=1.4, zorder=7,
             linestyle="none")
    axR.plot([q_dep], [MCC_OBS], marker="o", ms=MS_DOT + 0.6,
             color=WEAK_LABELLER, markeredgecolor=WEAK_LABELLER,
             markeredgewidth=1.0, zorder=7, linestyle="none")
    axR.text(q_dep - 0.012, MCC_CEIL - 0.008, f"ceiling {MCC_CEIL:.2f}",
             ha="right", va="top", color=WEAK_LABELLER, zorder=8,
             **cased(T_SERIES))
    axR.text(q_dep - 0.012, MCC_OBS - 0.002,
             f"deployed labeller,\nmeasured {MCC_OBS:.2f}",
             ha="right", va="top", color=WEAK_LABELLER, zorder=8,
             linespacing=1.2, **cased(T_SERIES))

    fig.supxlabel("Labeller positive rate on the gold pairs", fontsize=T_AXIS,
                  y=0.020)

    signature(
        fig, GRAPHITE,
        headline=(f"Flagging {N_FLAG} of {N} pairs caps MCC at {MCC_CEIL:.2f}; "
                  f"the deployed labeller measured {MCC_OBS:.2f}."),
        deck="The ceiling follows from the two counts alone, so it carries no interval.")
    fig.savefig(HERE / "fig-ceiling.pdf")
    plt.close(fig)
    print(f"wrote fig-ceiling.pdf: {N_FLAG} of {N}, ceiling MCC {MCC_CEIL:.4f}, "
          f"measured {MCC_OBS:.4f}; {len(ATTAIN)} attaining rows checked, not drawn")


if __name__ == "__main__":
    make_fig_ceiling()
