"""Fig 14 fig-scale-ladder, Fig 15 fig-answer-polarity.

Two figures built on results/accord/framing_mechanism/answer_polarity.json, one
post hoc descriptive run in which the same 299 annotated pairs were put to
open-weights labellers under prompts that vary in exactly one way each.

Neither figure is pre-registered and the artifact says so in its own first key;
both generators assert that tag, so a reader of the code cannot mistake either
for a registered test, and the captions say it too.

THE TWO FIGURES ARGUE TWO DIFFERENT THINGS AND MUST NOT BE MERGED. The ladder
varies SIZE at a fixed prompt and finds that over-calling falls with parameter
count and never reaches the base rate. The polarity figure varies the QUESTION
at a fixed size and finds that two of three models answer the polarity of the
question rather than the pair. Drawing both on one canvas would put a size axis
and a question axis on one page and invite the reader to trade one against the
other, which neither run supports.

Every number is read from that artifact by key, or from
results/accord/VERIFIED_FACTS.json by key, and no rate is typed. Both figures
follow the v3 system in style.py: boxed axes, direct labels, precomputed tints,
no alpha.
"""
import json
import pathlib

import style
from style import (
    CYAN, EMERALD, MAGENTA, VIOLET,
    PAPER, INK, GRAPHITE, MUTE, MIST, PLATE_EDGE,
    EMERALD_DEEP, PUBLIC_MODEL_MARK, PUBLIC_MODEL_NAME,
    T_AXIS, T_TICK, T_SERIES, T_NOTE,
    LW_DATA, LW_REF, MS_DOT, REF_DASH, ERRBAR, DOT, FIGW,
    canvas, frame, signature, band, case, cased, wilson_ci,
)
style.setup()
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
OUTDIR = style.out_dir(HERE)
ROOT = pathlib.Path(__file__).resolve().parents[3]

POLARITY_PATH = ROOT / "results/accord/framing_mechanism/answer_polarity.json"
_POLARITY = json.loads(POLARITY_PATH.read_text())
assert "not pre-registered" in _POLARITY["note"], \
    "the artifact no longer declares itself post hoc; both captions say it does"

_FACTS = json.loads((ROOT / "results/accord/VERIFIED_FACTS.json").read_text())


def fact(key):
    assert not key.startswith("withdrawn."), f"{key} is withdrawn, not citable"
    return _FACTS[key]["value"]


def cell(model, prompt):
    """The one cell for a (model, prompt), or a failure. Never a default: a
    missing cell used to be the kind of thing that silently drew a zero."""
    hits = [c for c in _POLARITY["cells"]
            if c["model"] == model and c["prompt"] == prompt]
    assert len(hits) == 1, f"{model} under {prompt}: {len(hits)} cells"
    return hits[0]


# =============================================================================
# Fig 14 -- fig-scale-ladder
# =============================================================================
# ONE CLAIM: over-calling declines with parameter count and does not approach
# calibration at any size that can be served here.
#
# The ladder is one model family, one identical prompt, and the same 299 pairs
# at every rung, which is what makes the comparison a comparison. Four rungs
# carry a rate. The fifth, 4B, returned 299 unparseable replies out of 299 and
# is EXCLUDED, and the honest way to draw an excluded cell is to draw the cell
# and say it is empty: a gap in a line that is not marked reads as a line.
#
# WHAT THIS FIGURE MAY NOT DO. It may not extrapolate. The fitted slope and
# correlation are printed as a note because the artifact computes them, and the
# line on the canvas joins measured points and stops at the last one; nothing
# is drawn to the right of 14B, and the axis does not run there. Four points
# are four points, and a trend drawn past its data is how a reader is told that
# a bigger model would be calibrated when nothing here says so.
def make_fig_scale_ladder():
    ladder = _POLARITY["ladder"]
    assert ladder["family"] == "Qwen 3"
    assert ladder["monotone_decreasing"] is True
    prompt = ladder["prompt"]

    pts, excluded = [], []
    for p in ladder["points"]:
        if p.get("status") == "all unparsed":
            assert p["n_parsed"] == 0 and p["n_unparsed"] == p["n_total"]
            excluded.append(p)
            continue
        assert p["prompt"] == prompt and p["asks"] == "contradiction"
        # The rate is recomputed from the two counts, never read as a rate, and
        # the stored interval is the check on the recomputation.
        e = p["raw_yes_count"] / p["n_parsed"]
        lo, hi = wilson_ci(p["raw_yes_count"], p["n_parsed"])
        assert abs(e - p["raw_yes_rate"]) < 5e-5
        assert abs(lo - p["raw_yes_wilson95"][0]) < 5e-4
        assert abs(hi - p["raw_yes_wilson95"][1]) < 5e-4
        pts.append((p["params_b"], e, lo, hi, p["n_parsed"], p["n_unparsed"],
                    p["model"]))
    pts.sort()
    assert len(excluded) == 1 and len(pts) == 4
    assert [p[1] for p in pts] == sorted((p[1] for p in pts), reverse=True), \
        "the ladder is no longer ordered by size; the headline would be false"
    x_excl = float(excluded[0]["model"].split()[-1].rstrip("B"))
    assert pts[0][0] < x_excl < pts[-1][0]

    base = fact("gold.base_rate_299")
    n_gold = fact("gold.n_total")
    assert all(p[4] + p[5] == n_gold for p in pts)
    ratio = pts[-1][1] / base

    W, H = FIGW, 2.98
    fig = plt.figure(figsize=(W, H))
    canvas(fig)
    # THE MEASURE GATE CAUGHT THIS ONE. The right column started at 0.744 and
    # its header ran about 50pt past the canvas, so savefig(bbox="tight")
    # expanded the file to 459pt and \includegraphics would have scaled the
    # whole figure to 0.92 on the page, printing 8.5pt type at 7.85pt. The
    # column starts at 0.615 now and the axes gives up the width; every line
    # it sets is measured against the remaining measure and asserted below, so
    # a longer string fails the build instead of shrinking the type.
    A0, AH = 0.56, 1.78
    # X_LEFT is 0.135, not 0.082. The y axis label is two rotated lines, so it
    # is about 22pt wide, and behind it sit the tick labels: at 0.082 the whole
    # stack reached past the left edge of the canvas and savefig expanded the
    # file. The axes gives up the width rather than the label losing a line.
    X_LEFT, X_RIGHT = 0.135, 0.585
    ax = fig.add_axes([X_LEFT, A0 / H, X_RIGHT - X_LEFT, AH / H])
    ax.set_xscale("log")
    ax.set_xlim(0.48, 17.5)
    ax.set_ylim(-0.03, 1.06)
    ax.set_xticks([p[0] for p in pts] + [x_excl])
    ax.set_xticklabels([f"{p[0]:g}B" for p in pts] + [f"{x_excl:g}B"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    frame(ax, grid="y")

    # The ground: everything above the pharmacist's base rate is over-calling.
    # MIST carries that meaning on fig-prevalence-response already, so it is
    # the same tone here and means the same thing.
    band(ax, base, 1.06, MIST, zorder=0, axis="y")
    ax.axhline(base, color=EMERALD_DEEP, lw=LW_REF, linestyle=REF_DASH,
               zorder=3)
    t = ax.text(0.50, base + 0.035,
                f"pharmacist base rate {base:.4f} on the same {n_gold} pairs",
                ha="left", va="bottom", color=EMERALD_DEEP, style="italic",
                zorder=6, **cased(fontsize=T_NOTE))
    case(t, linewidth=3.0)

    # The excluded rung: drawn, named, and carrying no value.
    ax.plot([x_excl, x_excl], [-0.03, 1.06], color=GRAPHITE, lw=1.0,
            linestyle=(0, (2.4, 1.8)), zorder=2, solid_capstyle="butt")
    t = ax.text(x_excl * 0.93, 0.33,
                f"{excluded[0]['model']}\n{excluded[0]['n_unparsed']} of "
                f"{excluded[0]['n_total']} replies\nunparseable, excluded",
                ha="right", va="center", color=GRAPHITE, style="italic",
                zorder=6, linespacing=1.18, **cased(fontsize=T_NOTE))
    case(t, linewidth=3.0)

    # The series. The segment that spans the excluded rung is dashed, because a
    # solid segment there would assert a value the run does not have.
    mark, _dash, ramp = PUBLIC_MODEL_MARK["qwen3:8b"]
    for a, b in zip(pts, pts[1:]):
        spans = a[0] < x_excl < b[0]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=ramp.DEEP, lw=LW_DATA,
                linestyle=((0, (2.4, 1.8)) if spans else "-"), zorder=4,
                solid_capstyle="butt")
    for x, e, lo, hi, *_ in pts:
        ax.errorbar([x], [e], yerr=[[e - lo], [hi - e]], fmt=mark,
                    color=ramp.DEEP, markerfacecolor=ramp.CORE,
                    markeredgecolor=ramp.DEEP, zorder=6, **ERRBAR, **DOT)
    # Each value label is placed off its own marker on the side with room:
    # the first sits under its point because the axis ends just above it, the
    # last to its left because the axis ends just to its right.
    for i, (x, e, lo, hi, *_) in enumerate(pts):
        first, last = i == 0, i == len(pts) - 1
        below = first or last          # the two rungs whose label would leave
                                       # the axes, or print through a sibling
        ax.text(x * (0.90 if last else 1.10),
                e + (-0.050 if below else 0.040),
                f"{e:.4f}", ha="right" if last else "left",
                va="top" if below else "bottom", color=ramp.DEEP, zorder=7,
                **cased(fontsize=T_SERIES))

    ax.set_xlabel("Parameters, log scale", fontsize=T_AXIS, color=INK,
                  labelpad=3.0)
    ax.set_ylabel("Share of the 299 pairs flagged\nas a contradiction",
                  fontsize=T_AXIS, color=INK, labelpad=3.0, linespacing=1.18)
    ax.tick_params(axis="both", labelsize=T_TICK)

    # The value column: one row per rung, the family's own tone, plus the fit
    # the artifact computed. The fit is a note and not a line on the canvas.
    X_TXT, X_EDGE = 0.615, 0.9960
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()

    def _fits(text, **kw):
        for line in text.split("\n"):
            t = fig.text(0, 0, line, **kw)
            w = t.get_window_extent(_r).width / fig.bbox.width
            t.remove()
            assert X_TXT + w <= X_EDGE, \
                f"right column overflows the canvas: {line!r}"
        return text

    y0, pitch = (A0 + AH - 0.10) / H, 0.128
    fig.text(X_TXT, y0 + 0.055,
             _fits(f"{ladder['family']}, one prompt, the same {n_gold} pairs",
                   fontsize=T_NOTE, style="italic"), ha="left", va="center", fontsize=T_NOTE,
             color=MUTE, style="italic")
    for i, (x, e, lo, hi, n_par, n_unp, model) in enumerate(pts):
        y = y0 - i * pitch
        fig.text(X_TXT, y, _fits(model, fontsize=T_SERIES), ha="left",
                 va="center", fontsize=T_SERIES, color=INK)
        fig.text(X_TXT, y - 0.048,
                 _fits(f"{e:.4f}  [{lo:.2f}, {hi:.2f}]", fontsize=T_NOTE),
                 ha="left", va="center", fontsize=T_NOTE, color=MUTE)
    # The fitted slope and correlation are NOT drawn. Four points, one of them a
    # 0.6B model that flags every pair, do not support a trend, and the chapters
    # say nothing is concluded about scale in either direction. Printing a slope
    # beside that sentence contradicted it.
    fig.text(X_TXT, y0 - len(pts) * pitch,
             _fits("Four rungs. No trend is estimated.",
                   fontsize=T_NOTE, style="italic"),
             ha="left", va="top", fontsize=T_NOTE, color=MUTE, style="italic",
             linespacing=1.18)
    fig.text(X_TXT, 0.030, "Post hoc descriptive.", ha="left", va="bottom",
             fontsize=T_NOTE, color=MUTE, style="italic")

    signature(fig, ramp.DEEP,
              f"Flag rate is lower at larger sizes in this family, and is still\n"
              f"{ratio:.0f} times the pharmacist's base rate at the largest size "
              f"served here.")
    fig.savefig(OUTDIR / "fig-scale-ladder.pdf")
    plt.close(fig)
    print("wrote fig-scale-ladder.pdf")


# =============================================================================
# Fig 15 -- fig-answer-polarity
# =============================================================================
# ONE CLAIM: asking the complementary question moves two of three models' answer
# rate by more than the pharmacist's own ambiguity allows, so those two are
# answering the polarity of the question rather than the pair.
#
# THE GEOMETRY IS THE ARGUMENT. Put the yes rate to "is there a contradiction?"
# on x and the yes rate to "are these consistent?" on y. A labeller that reads
# the pair and answers both questions from what it read lands near the line
# x + y = 1: whatever share it calls contradictory it does not also call
# consistent. Landing BELOW that line means saying no to both, and a pair CAN
# be neither, so the line is not a law and the figure does not draw it as one.
# The pharmacist coded 46 of the 299 pairs ambiguous, so a band of that width
# under the line is defensible and is drawn as ground. What is not defensible
# is landing far below the band, and two of the three models do.
#
# WHAT THIS FIGURE MAY NOT SAY. It may not call any point logically impossible,
# because none of them is. It may not read complementary marginals as per-pair
# coherence either: the kappa between the two framings is between 0.22 and 0.38
# for all three models, so the model that sits ON the line still does not agree
# with itself pair by pair. That is printed on the canvas as a caveat rather
# than argued as a second claim, because the chapter's claim is about the
# marginal and the caveat stops the reader from over-reading it.
def make_fig_answer_polarity():
    FWD, REV = "P_neutral", "P_reversed"
    coh = [c for c in _POLARITY["coherence"]
           if c["forward_prompt"] == FWD and c["reverse_prompt"] == REV]
    assert len(coh) == 3
    kappa = {c["model"]: c["kappa_between_framings"] for c in coh}
    n_pairs = {c["n"] for c in coh}
    assert len(n_pairs) == 1
    n = n_pairs.pop()
    assert n == fact("gold.n_total")

    tags = ["llama3:8b", "granite3.3:8b", "qwen3:8b"]
    rows = []
    for tag in tags:
        name = PUBLIC_MODEL_NAME[tag]
        f, r = cell(name, FWD), cell(name, REV)
        assert f["asks"] == "contradiction" and r["asks"] == "consistency"
        assert f["n_parsed"] == n and r["n_parsed"] == n
        x = f["raw_yes_count"] / f["n_parsed"]
        y = r["raw_yes_count"] / r["n_parsed"]
        assert abs(x - f["raw_yes_rate"]) < 5e-5
        assert abs(y - r["raw_yes_rate"]) < 5e-5
        rows.append((tag, name, x, y, kappa[name]))

    # The allowance: the share of pairs the pharmacist coded ambiguous. A pair
    # she could not call is a pair a labeller may decline both ways, so this is
    # the width of the band under x + y = 1 that needs no explanation.
    n_amb = fact("overcallamb.n_ambiguous")
    assert n_amb == n - fact("c120.g253.n")
    amb = n_amb / n
    inside = [r for r in rows if r[2] + r[3] >= 1.0 - amb]
    outside = [r for r in rows if r[2] + r[3] < 1.0 - amb]
    assert len(inside) == 1 and len(outside) == 2, \
        "the split this figure headlines has changed; rewrite the headline"

    W, H = FIGW, 3.56
    fig = plt.figure(figsize=(W, H))
    canvas(fig)
    # A square axes box in INCHES, so the x + y = 1 reference really is a
    # 45 degree line on the page and the distance below it can be read as a
    # distance. A rectangular box would make that line lie.
    A0, AS = 0.70, 2.30
    X_LEFT = 0.088
    ax = fig.add_axes([X_LEFT, A0 / H, AS / W, AS / H])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    frame(ax, grid=True)

    # The defensible band, between x + y = 1 and x + y = 1 minus the ambiguity
    # allowance. Drawn as ground, with no stroke, like every other ground here.
    ax.fill_between([0.0, 1.0], [1.0 - amb, -amb], [1.0, 0.0],
                    facecolor=MIST, edgecolor="none", lw=0, zorder=0)
    ax.plot([0.0, 1.0], [1.0, 0.0], color=GRAPHITE, lw=1.2, zorder=2)
    # TWO SHORT TAGS INSIDE THE AXES AND NOTHING LONGER. A four line gloss on
    # the band was set here and was wider than the 2.30in axes box, so it ran
    # across the field it was explaining and through a model's own label. The
    # gloss is in the right column now, where there is measure for it; what
    # stays on the field is the name of the line and the name of the corridor.
    for xt, yt, txt in ((0.145, 0.872, "complementary"),
                        (0.415, 0.520, "ambiguity allowance")):
        t = ax.text(xt, yt, txt, ha="left", va="bottom", color=MUTE,
                    style="italic", zorder=6, **cased(fontsize=T_NOTE))
        case(t, linewidth=3.0)

    for tag, name, x, y, k in rows:
        mark, _dash, ramp = PUBLIC_MODEL_MARK[tag]
        ax.plot([x], [y], mark, color=ramp.DEEP, markerfacecolor=ramp.CORE,
                markeredgecolor=ramp.DEEP, markeredgewidth=1.2, zorder=7,
                **DOT)
        # A label to the right of a point near the right edge leaves the axes,
        # so points past 0.6 take their name on the left, on the same line.
        if x > 0.6:
            t = ax.text(x - 0.035, y, name, ha="right", va="center",
                        color=ramp.DEEP, zorder=8, **cased(fontsize=T_SERIES))
        else:
            t = ax.text(x, y - 0.035, name, ha="center", va="top",
                        color=ramp.DEEP, zorder=8, **cased(fontsize=T_SERIES))
        case(t, linewidth=3.0)

    ax.set_xlabel('Yes rate to "is there a contradiction?"', fontsize=T_AXIS,
                  color=INK, labelpad=3.0)
    ax.set_ylabel('Yes rate to "are these consistent?"', fontsize=T_AXIS,
                  color=INK, labelpad=3.0)
    ax.tick_params(axis="both", labelsize=T_TICK)

    X_TXT = X_LEFT + AS / W + 0.052
    y0, pitch = (A0 + AS - 0.12) / H, 0.115
    # ONE LINE. Two lines here printed through the first model row, and there
    # is no room above to move them into: the headline block ends 30pt from the
    # top of a 256pt canvas and the axes starts 56pt from it.
    fig.text(X_TXT, y0 + 0.052,
             f"Three models, both questions, {n} pairs",
             ha="left", va="center", fontsize=T_NOTE, color=MUTE,
             style="italic")
    for i, (tag, name, x, y, k) in enumerate(rows):
        _m, _d, ramp = PUBLIC_MODEL_MARK[tag]
        yy = y0 - i * pitch
        fig.text(X_TXT, yy, name, ha="left", va="center", fontsize=T_SERIES,
                 color=ramp.DEEP)
        fig.text(X_TXT, yy - 0.044,
                 f"{x:.4f} and {y:.4f}, sum {x + y:.4f}", ha="left",
                 va="center", fontsize=T_NOTE, color=MUTE)
    ks = [r[4] for r in rows]
    y_note = y0 - len(rows) * pitch + 0.030
    for blk in (f"Shaded: the {amb * 100:.0f} percent of pairs the pharmacist\n"
                f"coded ambiguous ({n_amb} of {n}). A pair can be neither\n"
                f"contradictory nor consistent, so the band needs no\n"
                f"explanation. Below it does.",
                f"Agreement between the two framings is low for all\n"
                f"three, kappa {min(ks):.2f} to {max(ks):.2f}. Complementary "
                f"marginals are\nnot per-pair coherence, and this figure claims "
                f"only\nthe marginal.",
                "Post hoc descriptive. The two prompts ask\ncomplementary "
                "questions of the same pair and\ndiffer in nothing else."):
        fig.text(X_TXT, y_note, blk, ha="left", va="top", fontsize=T_NOTE,
                 color=MUTE, style="italic", linespacing=1.18)
        y_note -= (blk.count("\n") + 1) * T_NOTE * 1.18 / (72 * H) + 0.028

    signature(fig, GRAPHITE,
              f"Reversing the question leaves one model's two answer rates "
              f"complementary and moves\nthe other two below anything the "
              f"pharmacist's own ambiguity explains.")
    fig.savefig(OUTDIR / "fig-answer-polarity.pdf")
    plt.close(fig)
    print("wrote fig-answer-polarity.pdf")


# =============================================================================
# The manifest. fig_batch2.py does not write one and its four figures reached
# manifest.json by hand, which is the failure mode CLAUDE.md records about the
# register: an entry maintained by hand outlives the thing it describes. These
# two entries are merged in from here, on the same rule make_figures.py uses
# (read what is there, replace only our own labels, keep every other note).
# =============================================================================
MANIFEST = [
    {
        "file": "figures/fig-scale-ladder.pdf",
        "label": "fig:scale-ladder",
        "sources": ["results/accord/framing_mechanism/answer_polarity.json",
                    "results/accord/VERIFIED_FACTS.json"],
        "note": "One model family, one identical prompt, the same 299 "
                "annotated pairs, four sizes with a rate and one excluded. "
                "Every rate is recomputed here from the cell's own raw_yes_count "
                "over n_parsed and asserted against the rate and the Wilson "
                "interval the artifact stores; nothing is typed. The excluded "
                "rung is Qwen 3 4B, 299 of 299 replies unparseable: it is drawn "
                "as a dashed rule with its count and carries no value, and the "
                "series segment that spans it is dashed for the same reason. "
                "The reference line is the pharmacist base rate, "
                "gold.base_rate_299, in the pharmacist's EMERALD; MIST fills "
                "everything above it, which is over-calling, as MIST fills the "
                "over-prediction wedge on fig-prevalence-response. Neither a fitted "
                "slope nor a correlation is printed and NO fitted line is "
                "drawn: four rungs, one of them a 0.6B model that flags every "
                "pair, do not support a trend, and the chapters conclude "
                "nothing about scale in either direction. The axis stops at "
                "the largest size that was run and the figure does not "
                "extrapolate. Post hoc descriptive, asserted from the "
                "artifact's own note key. No hosted model appears; the family "
                "is open-weights.",
    },
    {
        "file": "figures/fig-answer-polarity.pdf",
        "label": "fig:answer-polarity",
        "sources": ["results/accord/framing_mechanism/answer_polarity.json",
                    "results/accord/VERIFIED_FACTS.json"],
        "note": "Three open-weights labellers at one size, the same 299 pairs, "
                "two prompts asking complementary questions and differing in "
                "nothing else. x is the yes rate to the contradiction question "
                "and y the yes rate to the consistency question, both "
                "recomputed from raw_yes_count over n_parsed and asserted "
                "against the stored rates. The axes box is square in INCHES, "
                "so the x plus y equals 1 reference really is a 45 degree line "
                "and the distance below it can be read as a distance. The MIST "
                "corridor under that line is the share of pairs the pharmacist "
                "coded ambiguous, overcallamb.n_ambiguous over gold.n_total, "
                "asserted equal to 299 minus 253: a pair can be neither "
                "contradictory nor consistent, so that corridor needs no "
                "explanation and nothing on this canvas is called logically "
                "impossible. Kappa between the two framings is printed as a "
                "caveat, not as a claim, because complementary marginals are "
                "not per-pair coherence. Post hoc descriptive, asserted from "
                "the artifact's own note key.",
    },
]


def write_manifest():
    import json as _json
    path = OUTDIR / "manifest.json"
    src = HERE / "manifest.json"
    ours = {m["label"] for m in MANIFEST}
    try:
        doc = _json.loads(src.read_text())
    except (OSError, ValueError):
        doc = {"built": [], "skipped": []}
    doc["built"] = [m for m in doc.get("built", []) if m["label"] not in ours] \
        + MANIFEST
    path.write_text(_json.dumps(doc, indent=2))
    print(f"manifest updated with {len(MANIFEST)} entries -> {path}")


if __name__ == "__main__":
    make_fig_scale_ladder()
    make_fig_answer_polarity()
    write_manifest()
