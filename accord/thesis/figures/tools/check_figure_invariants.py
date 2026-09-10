"""Assert the invariants the figure set states about itself.

Every defect a hostile reviewer found in this set had the same shape: a rule the
author wrote down, proved, and applied to exactly one figure, which then went on
being violated by that figure's siblings. The measure invariant was diagnosed
and fixed for fig-met-exchange and left standing in fig-forest. The ban on
setting type in SLATE was written into fig_batch2 and broken in
make_fig_cohort_flow. The single source of every hex drifted out of sync with
the file that draws the entities it names.

That is metric blindness applied to a figure set: a defect that no reported
number was sensitive to. The answer is the one this thesis argues for
everywhere else, which is a cheap check that IS sensitive, run every build.

Run:  python3 accord/thesis/figures/tools/check_figure_invariants.py
Exit 0 if every invariant holds, 1 otherwise.
"""

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve()
FIGDIR = HERE.parents[1]
ROOT = HERE.parents[3]

# ---------------------------------------------------------------------------
# Invariant 1: drawn width must equal printed width.
#
# canvas() pins the bbox to the figure rectangle, but savefig(bbox="tight")
# still EXPANDS for artists that overflow it. When it does, \includegraphics
# scales the whole figure down to fit the measure, and every type size on it
# shrinks by the same factor, silently, below the floor style.py sets. This is
# invisible in the PDF at magnification and only appears on the page.
# ---------------------------------------------------------------------------
CLASS_WIDTH = {0.98: 424.348, 0.88: 381.047}
TOL = 0.5

# Figures that are legitimately drawn at a non-standard measure, with the
# reason. Anything not listed must land on one of the class widths above.
WIDTH_EXEMPT = {
    "fig-gap-trajectory": "not bound into the thesis",
    "fig-rule-precision": "not bound into the thesis",
    "fig-spec-curve": "not bound into the thesis",
}


def media_box_width(pdf):
    d = pdf.read_bytes()
    m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", d)
    if not m:
        return None
    return float(m.group(3)) - float(m.group(1))


def check_measure():
    bad = []
    for pdf in sorted(FIGDIR.glob("fig-*.pdf")):
        stem = pdf.stem
        if stem in WIDTH_EXEMPT:
            continue
        w = media_box_width(pdf)
        if w is None:
            bad.append((stem, "no MediaBox"))
            continue
        if not any(abs(w - t) <= TOL for t in CLASS_WIDTH.values()):
            nearest = min(CLASS_WIDTH.values(), key=lambda t: abs(w - t))
            scale = nearest / w
            bad.append((stem,
                        f"{w:.3f} pt, not a class measure. Scaled to "
                        f"{scale:.4f} on the page, so 8.5 pt type prints at "
                        f"{8.5 * scale:.2f} pt"))
    return bad


# ---------------------------------------------------------------------------
# Invariant 2: style.py is the single source of every hex, so the entity map it
# publishes must match the hues the drawing code actually uses. When it drifts,
# the next edit made against style.py silently recolours an entity.
# ---------------------------------------------------------------------------
def check_entity_hue():
    style = (FIGDIR / "style.py").read_text()
    batch = (FIGDIR / "fig_batch2.py").read_text()

    m = re.search(r"ENTITY_HUE\s*=\s*\{(.*?)\n\}", style, re.S)
    if not m:
        return [("style.py", "ENTITY_HUE not found")]
    declared = dict(re.findall(r'"([\w:.]+)"\s*:\s*([A-Z_]+)', m.group(1)))

    m2 = re.search(r"MODEL_HUE\s*=\s*\{(.*?)\}", batch, re.S)
    if not m2:
        return []
    used = dict(re.findall(r'"([\w:.]+)"\s*:\s*([A-Z_]+)', m2.group(1)))

    bad = []
    for key, hue in used.items():
        norm = key.replace(":", "_").replace(".", "")
        if norm in declared and declared[norm] != hue:
            bad.append(("ENTITY_HUE",
                        f"{key} drawn {hue}, style.py declares "
                        f"{declared[norm]} for {norm}"))
    return bad


# ---------------------------------------------------------------------------
# Invariant 3: the set does not set type in SLATE.
#
# fig_batch2 states the rule and the measurement behind it: SLATE is 3.48:1 on
# white, under the 4.5:1 the standard asks for. A figure that sets a DATUM in
# it is worse than the case that produced the ban.
# ---------------------------------------------------------------------------
TEXT_CALLS = ("ax.text(", "a.text(", "b.text(", ".set_title(", ".annotate(")


def check_no_slate_type():
    bad = []
    for src in sorted(FIGDIR.glob("*.py")):
        if src.name in ("style.py",):
            continue
        lines = src.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            if "color=SLATE" not in line and "color=deep" not in line:
                continue
            window = "\n".join(lines[max(0, i - 6):i + 1])
            if any(c in window for c in TEXT_CALLS) and "color=SLATE" in line:
                bad.append((src.name, f"line {i}: type in SLATE, 3.48:1 on paper"))
    return bad


# ---------------------------------------------------------------------------
# Invariant 4: no text tone may sit on a ground it does not clear.
#
# Added after a reviewer found the exemplar figure still drawing SCARLET_DEEP
# on SCARLET_TINT at 4.93:1, and wrote: "It survived because you fixed the
# other figure, not this one." That is the whole reason this file exists. A
# contrast defect was found, measured and fixed on one canvas, and the same
# pairing went on standing on another because nothing checked the set.
#
# 4.93:1 clears the 4.5:1 minimum, so this is reported at the margin the set
# actually adopted when it fixed the pipeline figure, not at the bare minimum.
#
# IMPORTANT, and the reason this is ADVISORY rather than gating: it compares
# palette constants pairwise. It does not know which pairings are actually
# drawn. A pairing listed here may appear on no canvas at all, and cannot then
# be fixed from any figure function, only by changing style.py. A gate that
# emits failures nobody can act on gets ignored, which would cost more than it
# buys. So this prints what the palette PERMITS and does not fail the build;
# the three checks above are tied to real artifacts and do fail it.
# ---------------------------------------------------------------------------
MARGIN = 6.0


def _lum(h):
    h = h.lstrip("#")
    ch = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    f = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in ch]
    return 0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2]


def _ratio(a, b):
    la, lb = _lum(a), _lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def check_text_on_ground():
    """Every text tone the set uses, against every ground it draws text on."""
    sys.path.insert(0, str(FIGDIR))
    try:
        import style
    except Exception as exc:                       # pragma: no cover
        return [("style.py", f"not importable: {exc}")]

    text_tones = {"INK": style.INK, "MUTE": style.MUTE,
                  "GRAPHITE": style.GRAPHITE,
                  "SCARLET_DEEP": style.SCARLET.DEEP}
    grounds = {"PAPER": style.PAPER,
               "SCARLET_TINT": style.SCARLET.TINT,
               "SCARLET_WASH": style.SCARLET.WASH,
               "MIST": style.MIST}

    bad = []
    for tn, tv in text_tones.items():
        for gn, gv in grounds.items():
            r = _ratio(tv, gv)
            if r < MARGIN:
                verdict = "FAILS AA" if r < 4.5 else "thin, clears AA only"
                bad.append((f"{tn} on {gn}", f"{r:.2f}:1, {verdict}"))
    return bad


def main():
    failures = []
    for name, fn in (("printed measure", check_measure),
                     ("entity hue map", check_entity_hue),
                     ("no type in SLATE", check_no_slate_type)):
        found = fn()
        status = "FAIL" if found else "pass"
        print(f"{status}: {name}")
        for what, why in found:
            print(f"    {what}: {why}")
        failures.extend(found)

    advisory = check_text_on_ground()
    print(f"{'note' if advisory else 'pass'}: text on ground (advisory, palette "
          f"pairs, not tied to draw calls)")
    for what, why in advisory:
        print(f"    {what}: {why}")

    print()
    if failures:
        print(f"{len(failures)} invariant violation(s).")
        if advisory:
            print(f"{len(advisory)} advisory pairing(s), not counted.")
        return 1
    print("all gating figure invariants hold.")
    if advisory:
        print(f"{len(advisory)} advisory pairing(s), not counted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
