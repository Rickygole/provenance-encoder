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


# =============================================================================
# Orphan check. Every built figure must either be included by a .tex in this
# thesis or be named below with the reason it was retired.
#
# Seven PDFs sat in this directory referenced by nothing. Two of them,
# fig-rule-precision and fig-prevalence, had been in a chapter and were removed
# by earlier reviews; the other five were built and never used at all. None of
# that was detectable, because nothing checked. A stale PDF beside fourteen live
# ones is a thing a reader or a co-author can pick up by mistake, and a figure
# that carries no claim the text depends on has no reason to exist.
#
# They are kept rather than deleted because their generators are committed and
# the figures are evidence of work that was done and reconsidered. What was
# missing was the record of WHY each one is not in the thesis. That record is
# here, and a new orphan now fails this gate instead of accumulating quietly.
# Three of the seven cannot be regenerated. fig_gap_trajectory.py and
# fig_spec_curve.py still import palette names the style module no longer defines
# (ACCENT, MUTED, LIGHT, DEAD), so they raise ImportError; fig-rule-precision is
# named in make_figures.py but no longer emitted by it. Their PDFs are therefore
# artifacts of a palette and a figure set that no longer exist, which is a reason
# to keep them out of the thesis rather than a reason to fix the generators. The
# four other retired figures do still regenerate.
UNREPRODUCIBLE = {"fig-gap-trajectory", "fig-spec-curve", "fig-rule-precision"}

RETIRED = {
    "fig-threshold-surface":
        "Was the one figure in the appendix giving the full treatment of the "
        "selection-channel anomaly. That appendix was moved out of the bound "
        "thesis to the released analysis code when the thesis was brought under "
        "its page limit, because the anomaly is reported in the discussion "
        "chapter as not a finding and its sweep carried no claim the text "
        "depends on. The figure travels with the appendix source.",
    "fig-rule-precision":
        "Was in the results chapter. Removed when the figure set was restructured "
        "on committee review: it plotted per-rule precision as bars with no "
        "sampling spread on counts small enough that the spread dominates, and "
        "the table it sat beside already gave the cells.",
    "fig-prevalence":
        "Was in an early draft of the results chapter. Superseded by "
        "fig-prevalence-response, which plots the same quantity as a response "
        "curve over the threshold sweep instead of at one operating point.",
    "fig-asymmetry":
        "Never reached a chapter. The asymmetry it drew, nursing-side against "
        "discharge-side flag rates, is reported as two numbers in the text; a "
        "two-point figure was never worth a float.",
    "fig-funnel":
        "Never reached a chapter. Superseded by fig-cohort-flow, which carries "
        "the same attrition with the stage labels the text refers to.",
    "fig-gap-trajectory":
        "Never reached a chapter. Superseded by fig-scaling and fig-power, which "
        "split the trajectory claim from the resolution claim; drawing both on "
        "one axis conflated a measured gap with a detectability threshold.",
    "fig-spec-curve":
        "Never reached a chapter. The specificity curve is a monotone "
        "re-expression of fig-threshold-surface at a fixed prevalence and adds "
        "no claim the surface does not already carry.",
    "fig-tracking":
        "Never reached a chapter. It tracked a quantity across correction "
        "entries, which is the correction record's job and reads as a progress "
        "chart rather than as evidence.",

    # ---------------------------------------------------------------------
    # Retired in the round that followed the compliance withdrawal. The
    # withdrawal moved every value obtained by transmitting credentialed data
    # to a hosted service under withdrawn.hosted.* in the register, 718 of
    # them, so any figure that drew a Claude or a Bedrock row was drawing a
    # value the thesis no longer reports. The rest of this block is the
    # weak-figure cull that ran with it. Each PDF stays on disk, as every
    # retired figure's does; what changed is that nothing binds them.
    # ---------------------------------------------------------------------
    "fig-capability":
        "Drew four hosted rows (two Bedrock arms, Claude Haiku 4.5, Claude "
        "Sonnet 5) whose values are now withdrawn.hosted.*, and its lower "
        "panel was entirely a local-versus-hosted decomposition, so there is "
        "no version of it that survives the withdrawal. It was independently "
        "defective and would have been rebuilt anyway: four text runs printed "
        "on top of each other in the PDF text layer, the public-text maximum "
        "label printed through a value label, and the canvas was 355pt tall to "
        "carry ten numbers with about 110pt of dead white. The deletion covers "
        "both the compliance fault and the drawing faults; the text-collision "
        "gate added in the same round is what would have caught the second.",
    "fig-labelnoise":
        "Drew two Claude Sonnet 5 series and nothing else. Both are hosted "
        "results under withdrawn.hosted.*, so the figure has no content left.",
    "fig-missing-cell":
        "Four rectangles containing text, with no axis and no scale: a table "
        "drawn as a picture. Conversion to a 2x2 tabular was considered and "
        "rejected, because the cell that made it a design rather than a list "
        "is the frontier-clinical cell and that cell is Claude Sonnet 5's "
        "over-calling ratio, now withdrawn. A tabular would print a withdrawn "
        "hosted value or print a hole where the argument was.",
    "fig-intro-headline":
        "Duplicated the first rows of fig-capability sixty pages before that "
        "figure drew them, and fig-capability is itself retired, so the "
        "duplicate outlived the thing it duplicated. Superseded by "
        "fig-ceiling, which argues the same ratio from the same three "
        "committed integers and argues further: it shows what the ratio costs "
        "in attainable agreement rather than only that the ratio is large.",
    "fig-disc-actionability":
        "A timeline with three coloured zones, no measurement of any kind, and "
        "colour used purely evaluatively. This set's rule is that colour "
        "carries identity and state, not a verdict, and a figure that plots "
        "nothing cannot be evidence for the verdict it colours. The "
        "field-availability fact it documented is one sentence of the "
        "discussion chapter.",
    "fig-falsification":
        "Two numbers on an axis running 0.98 to 1.02, so a move of 1.5 percent "
        "is drawn as a distance the width of the panel: the axis does the "
        "arguing, not the data. Retired on the precedent recorded for "
        "fig-asymmetry in this same table, that a two-point figure was never "
        "worth a float. Both ratios are printed in the results chapter.",
    "fig-scaling":
        "Three points and a line, every number of which is already in a table, "
        "and its terminal claim is a row of fig-forest's third family, drawn "
        "there with an interval this panel did not carry.",
    "fig-met-exchange":
        "Four markers, two of them stacked at the origin, on a y axis running "
        "to 200 with nothing drawn between 0 and 185, under a 203-word caption "
        "that had to carry a worked example because the panel could not. A "
        "figure whose caption does the arithmetic for the reader is a "
        "paragraph. The exchange rate it computed is stated in the discussion "
        "text.",
}

# A figure that is built and correct but that no .tex binds YET. This is not a
# second retirement list and it is not a place to park an orphan: an entry must
# name the chapter that is to include it and the label that chapter must use,
# and the gate prints it on every run so it cannot sit here unnoticed. It exists
# because the figures directory and the chapters are edited by different hands,
# and a figure cannot appear in a chapter and in this directory in the same
# breath.
PENDING_INCLUSION = {
}


def _bound_tex(thesis):
    """Every .tex file main_msu.tex actually inputs, transitively, skipping comments."""
    import re
    seen, stack = [], [thesis / "main_msu.tex"]
    while stack:
        f = stack.pop()
        if f in seen or not f.exists():
            continue
        seen.append(f)
        for line in f.read_text().splitlines():
            code = re.split(r"(?<!\\)%", line, 1)[0]
            for m in re.finditer(r"\\(?:input|include)\{([^}]+)\}", code):
                name = m.group(1)
                stack.append(thesis / (name if name.endswith(".tex") else name + ".tex"))
    if len(seen) < 2:
        raise SystemExit("orphan check: main_msu.tex inputs nothing; refusing to "
                         "report zero orphans by scanning nothing")
    return seen


def check_orphans():
    """Every built PDF is either included by a .tex or named in RETIRED."""
    import re
    thesis = pathlib.Path(__file__).resolve().parents[2]
    used = set()
    # "Used" means bound into the thesis, not present in the directory. The glob
    # over *.tex counted an appendix that main_msu.tex no longer inputs, so a
    # figure that could not reach the PDF read as live. The bound set is read
    # from the \input lines of main_msu.tex, following nested inputs.
    bound = _bound_tex(thesis)
    for tex in bound:
        for m in re.finditer(
                r"includegraphics\[[^\]]*\]\{figures/([^}]+?)(?:\.pdf)?\}",
                tex.read_text()):
            used.add(m.group(1))
    built = {f.stem for f in (thesis / "figures").glob("fig-*.pdf")}
    problems = []
    for name in sorted(built - used - set(RETIRED) - set(PENDING_INCLUSION)):
        problems.append((name, "built, included by no .tex, and not in RETIRED"))
    for name in sorted(set(RETIRED) & used):
        # Reported with the file and line, because closing this one is an edit
        # to a chapter and the person holding the chapter needs the address.
        where = []
        for tex in sorted(bound):
            for i, line in enumerate(tex.read_text().splitlines(), 1):
                if f"figures/{name}" in line:
                    where.append(f"{tex.name}:{i}")
        problems.append((name, "listed as RETIRED but still included by "
                               + ", ".join(where)))
    for name in sorted(set(PENDING_INCLUSION) & used):
        problems.append((name, "is now included by a .tex; delete its "
                               "PENDING_INCLUSION entry"))
    for name in sorted(set(PENDING_INCLUSION) - built):
        problems.append((name, "pending inclusion but its PDF is absent"))
    for name in sorted(set(RETIRED) - built):
        problems.append((name, "listed as RETIRED but its PDF is absent"))
    # A live figure that cannot be regenerated would be a real defect; a retired
    # one that cannot is a recorded fact. Fail only if the two sets overlap.
    for name in sorted(UNREPRODUCIBLE & used):
        problems.append((name, "is in the thesis but its generator no longer runs"))
    return problems, len(built), len(used), len(RETIRED)


# =============================================================================
# The three set-wide gates below, and the caption gate under them, exist for the
# reason stated at the top of this file: a rule proved on one canvas and left
# standing on a sibling is how every defect in this set has shipped. Each of
# them was written against a defect that a reviewer found by reading a figure,
# and each of them now reads every live figure instead.
#
# LIVE means built, not retired, and either bound by a chapter or waiting in
# PENDING_INCLUSION. Retired PDFs are deliberately not scanned: they are kept as
# a record of work that was reconsidered, and failing the build over a figure
# nothing binds would train a reader to ignore this tool.
# =============================================================================
def live_figures():
    thesis = pathlib.Path(__file__).resolve().parents[2]
    built = {f.stem for f in (thesis / "figures").glob("fig-*.pdf")}
    return sorted(built - set(RETIRED))


# -----------------------------------------------------------------------------
# Invariant 5: no two words on a canvas may print on top of each other.
#
# fig-capability carried four text runs printing through one another and a
# public-text maximum label printing through a value label. All of it was in the
# PDF text layer and none of it was visible in a thumbnail, which is exactly the
# class of defect this file exists for.
#
# HOW THE THRESHOLD WAS SET, because a naive box intersection fails every
# figure in this set. pdftotext reports a word's FONT box, ascender to
# descender, not the box its glyphs occupy: a 13pt headline reports 19.6pt tall,
# so a headline and the deck under it overlap by about 4.5pt on every canvas
# here while printing perfectly. Each box is therefore reduced to its central
# 55 percent vertically and shaved 0.6pt at each side, and a pair is a failure
# only when the smaller of the two reduced boxes is at least a quarter covered.
# Measured across all 22 PDFs in this directory at the time it was written, that
# rule fires on fig-capability and on nothing else.
#
# THE ROTATION ALLOWLIST. pdftotext reports one axis-aligned box per word
# whatever the word's own rotation, so a two-line rotated axis label can report
# two boxes that overlap in page coordinates while being two clean lines on the
# page. fig-ceiling draws exactly such a label. None of them currently trips the
# rule, so the allowlist is empty, and it is kept because the next rotated
# two-line label may. An entry is (figure stem, word) and it must name a word
# the maintainer has confirmed is rotated; it is not a place to silence a real
# collision on horizontal type.
# -----------------------------------------------------------------------------
WORD_RE = re.compile(
    r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">'
    r'(.*?)</word>')
COLLISION_KEEP = 0.55        # of each box's height
COLLISION_SHAVE = 0.6        # pt off each side
COLLISION_FRACTION = 0.25    # of the smaller box's area
ROTATION_ALLOWLIST = set()   # (stem, word) pairs; see above


def _words(pdf):
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "w.html"
        subprocess.run(["pdftotext", "-bbox", str(pdf), str(out)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        text = out.read_text()
    boxes = []
    for x0, y0, x1, y1, word in WORD_RE.findall(text):
        x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
        cy, half = (y0 + y1) / 2, (y1 - y0) * COLLISION_KEEP / 2
        boxes.append((x0 + COLLISION_SHAVE, cy - half,
                      x1 - COLLISION_SHAVE, cy + half, word))
    return boxes


def check_text_collisions():
    figs = live_figures()
    bad = []
    if not figs:
        return [("text collisions", "no live figure was scanned; this gate "
                                    "cannot pass by matching nothing")]
    for stem in figs:
        pdf = FIGDIR / f"{stem}.pdf"
        if not pdf.exists():
            bad.append((stem, "PDF missing"))
            continue
        boxes = _words(pdf)
        if not boxes:
            bad.append((stem, "no text layer; a figure in this set always "
                              "carries at least a headline"))
            continue
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ix = min(a[2], b[2]) - max(a[0], b[0])
                iy = min(a[3], b[3]) - max(a[1], b[1])
                if ix <= 0 or iy <= 0:
                    continue
                area = min((a[2] - a[0]) * (a[3] - a[1]),
                           (b[2] - b[0]) * (b[3] - b[1]))
                if area <= 0 or ix * iy / area < COLLISION_FRACTION:
                    continue
                if ((stem, a[4]) in ROTATION_ALLOWLIST
                        or (stem, b[4]) in ROTATION_ALLOWLIST):
                    continue
                bad.append((stem, f"{a[4]!r} prints through {b[4]!r} "
                                  f"({ix * iy / area:.0%} of the smaller box)"))
    return bad


# -----------------------------------------------------------------------------
# Invariant 6: two fills used as CATEGORY GROUNDS must be at least 0.10 apart in
# relative luminance, or the categories are one category in black and white.
#
# This is not a hypothetical. fig-forest separated its three unit families with
# MIST, EMERALD_WASH and AMBER_WASH, which are 0.7633, 0.8051 and 0.7956;
# fig-cohort-flow separated its three phases with HAIR, COBALT_WASH and
# EMERALD_WASH, 0.8453, 0.7996 and 0.8051; fig-power separated detectable from
# undetectable with EMERALD_WASH and SCARLET_WASH, 0.8051 and 0.8004. Every WASH
# stop in this palette is within 0.010 of every other and every TINT stop is
# within 0.005 of every other, by construction, because the ramps are built to
# sit at one luminance per stop. That is a virtue when the stop is structural
# ground and a defect the moment the stop is asked to encode a category, and
# nothing measured the difference until this gate.
#
# The registry is declared rather than parsed out of the drawing code, because
# no regex can tell a category ground from a region fill from a heatmap. It is
# kept honest by a coverage assertion instead: EVERY live figure must appear as
# a key, so a new figure cannot arrive without somebody deciding what its
# grounds are, and a figure with none declares the empty list.
# -----------------------------------------------------------------------------
GROUND_MIN_LUMINANCE_GAP = 0.10

CATEGORY_GROUNDS = {
    # fault rows against bare paper, a binary since the reviewer showed the
    # three-hue version failed in greyscale
    "fig-pipeline-status":     ["SCARLET_TINT", "PAPER"],
    "fig-cohort-flow":         ["HAIR"],        # one tone for all three phases
    "fig-exemplar":            ["COBALT_TINT", "MIST"],
    "fig-prevalence-response": ["MIST", "PAPER"],   # over-prediction wedge
    "fig-power":               ["EMERALD_WASH", "SCARLET_TINT"],
    "fig-forest":              [],   # families separated by gap and hairline
    "fig-ceiling":             [],   # a line on paper
    # a sequential heatmap field is not a category ground: its fills encode a
    # continuum and are read against a scale, not against each other
}


def check_category_grounds():
    sys.path.insert(0, str(FIGDIR))
    try:
        import style
    except Exception as exc:                       # pragma: no cover
        return [("style.py", f"not importable: {exc}")]
    tones = dict(style.STRUCTURE)
    for name, ramp in style.RAMPS.items():
        for stop, value in ramp.asdict().items():
            tones[f"{name}_{stop}"] = value

    figs = live_figures()
    bad = []
    if not figs:
        return [("category grounds", "no live figure was scanned")]
    for stem in figs:
        if stem not in CATEGORY_GROUNDS:
            bad.append((stem, "live, but declares no category grounds. Add it "
                              "to CATEGORY_GROUNDS, with [] if it has none."))
    for stem in sorted(set(CATEGORY_GROUNDS) - set(figs)):
        bad.append((stem, "declares category grounds but is not a live figure"))
    for stem, names in sorted(CATEGORY_GROUNDS.items()):
        for name in names:
            if name not in tones:
                bad.append((stem, f"{name} is not a tone style.py publishes"))
        pairs = [n for n in names if n in tones]
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                gap = abs(_lum(tones[pairs[i]]) - _lum(tones[pairs[j]]))
                if gap < GROUND_MIN_LUMINANCE_GAP:
                    bad.append((stem, f"{pairs[i]} and {pairs[j]} are "
                                      f"{gap:.3f} apart in luminance; two "
                                      f"category grounds need "
                                      f"{GROUND_MIN_LUMINANCE_GAP:.2f}"))
    return bad


# -----------------------------------------------------------------------------
# Invariant 7: no caption over 90 words.
#
# The set's captions ran to about 1235 words, one of them 205, and a caption
# that long is a paragraph that has been moved under a figure to avoid writing
# it into the chapter. The figure it belonged to is retired; the rule stays,
# because the caption grew to that length one clause at a time.
#
# THE TRAP THIS GATE IS WRITTEN AGAINST is recorded in CLAUDE.md: build.sh
# counted TODOs in ch*.tex when the files are msu_ch*.tex, and reported zero by
# scanning nothing. Two things here answer it. The file list is asserted
# non-empty, and so is the caption list: a thesis with no caption at all is a
# broken scan, not a clean pass. The second one is not theoretical -- the first
# draft of this gate matched \caption{ and found ZERO captions in 40, because
# every caption in this thesis carries a short-form optional argument and is
# written \caption[...]{...}.
# -----------------------------------------------------------------------------
CAPTION_MAX_WORDS = 90
CAPTION_RE = re.compile(r"\\caption\s*(?:\[[^\]]*\])?\s*\{")


def _caption_bodies(text):
    out = []
    for m in CAPTION_RE.finditer(text):
        i, depth = m.end(), 1
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        out.append((text[:m.start()].count("\n") + 1, text[m.end():i - 1]))
    return out


def _caption_words(body):
    body = re.sub(r"\\[a-zA-Z]+\*?\s*", " ", body)
    body = re.sub(r"[{}$\\~]", " ", body)
    return len(body.split())


def check_caption_length():
    thesis = FIGDIR.parent
    tex = sorted(thesis.glob("*.tex"))
    if not tex:
        return [("captions", "no .tex file was scanned; this gate cannot pass "
                             "by matching nothing")]
    bad, n_caps = [], 0
    for f in tex:
        for line, body in _caption_bodies(f.read_text()):
            n_caps += 1
            words = _caption_words(body)
            if words > CAPTION_MAX_WORDS:
                bad.append((f"{f.name}:{line}",
                            f"{words} words, over {CAPTION_MAX_WORDS}"))
    if not n_caps:
        return [("captions", f"{len(tex)} .tex files scanned and not one "
                             f"caption found; the pattern is wrong")]
    return bad


def main():
    failures = []
    for name, fn in (("printed measure", check_measure),
                     ("entity hue map", check_entity_hue),
                     ("no type in SLATE", check_no_slate_type),
                     ("no text collisions", check_text_collisions),
                     ("category grounds separable in greyscale",
                      check_category_grounds),
                     (f"no caption over {CAPTION_MAX_WORDS} words",
                      check_caption_length)):
        found = fn()
        status = "FAIL" if found else "pass"
        print(f"{status}: {name}")
        for what, why in found:
            print(f"    {what}: {why}")
        failures.extend(found)

    orphans, n_built, n_used, n_retired = check_orphans()
    if orphans:
        print(f"FAIL: figure orphans ({n_built} built, {n_used} used, "
              f"{n_retired} retired)")
        for name, why in orphans:
            print(f"    {name}: {why}")
        failures.extend(orphans)
    else:
        print(f"pass: no figure orphans ({n_used} live, {n_retired} retired "
              f"with a recorded reason)")
    for name, (label, chapter, why) in sorted(PENDING_INCLUSION.items()):
        print(f"note: {name} is built and waiting for {chapter} to include it "
              f"as {label}. {why}")

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
