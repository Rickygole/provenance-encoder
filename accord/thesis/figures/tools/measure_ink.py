#!/usr/bin/env python3
"""Measure ink coverage and chroma on every bound figure, against the v3 floors.

Two measured quantities, defined once so the builder and the critic measure the
same thing:

  coverage = fraction of pixels in the saved bounding box differing from
             #FFFFFF by more than 2/255 in any channel, at 300 dpi.
  chroma   = fraction of pixels with HSV saturation >= 0.25 and value >= 0.20.
  deadrun  = the tallest contiguous run of blank scanlines, as a fraction of
             canvas height. Blank means the whole row of pixels is paper.

Set-wide gating floors are coverage >= 0.28 and chroma >= 0.10 on every figure
and set-mean chroma >= 0.18; the per-figure floors below are the directives in
SPEC_V3 section 5 and are tighter.

Where a floor was found unreachable without inventing content that is not in
the data, the floor lost and was lowered here, with the reason recorded in
LOWERED. Spec risk 5 authorises exactly that: do not invent a band to pass this
tool.

DEAD SPACE IS THE COMPLEMENT COVERAGE CANNOT SEE. Coverage is a fraction of the
whole sheet, so it cannot tell a figure whose ink is spread thin from a figure
that is two thirds empty in one band: fig-capability measured 0.056 coverage
with about 110pt of unbroken white in a 355pt canvas, and nothing in this file
could say so. The tallest contiguous run of blank scanlines can, it is cheap,
and it fails the one thing a reader actually sees, which is a hole. Note that a
boxed plot cannot trip it inside its own frame, because every scanline that
crosses the axes box meets two spines; what it measures is the space BETWEEN
things, which is where dead space actually accumulates.

Exit status is non-zero if any figure is under its floor or over the dead-space
ceiling.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

FIGDIR = Path(__file__).resolve().parent.parent

# label -> (coverage floor, chroma floor)
FLOORS = {
    "fig-pipeline-status":    (0.72, 0.40),
    "fig-cohort-flow":        (0.60, 0.32),
    "fig-exemplar":           (0.48, 0.26),
    "fig-prevalence-response": (0.40, 0.14),
    "fig-power":              (0.62, 0.18),
    "fig-forest":             (0.46, 0.15),
    "fig-threshold-surface":  (0.90, 0.55),
    # fig-ceiling has no spec entry, because it was built after the spec. Its
    # floors are what the finished figure measures, and the reason they sit this
    # low is the reason fig-power's do; it is recorded here rather than in
    # LOWERED, which is for floors that MOVED. The figure is one line, two marks
    # and four labels on bare paper. It draws an identity in two marginals, so
    # there is no band, no interval and no region to fill, and the only way to
    # raise either number is to shade something the arithmetic does not say.
    # Spec risk 5 forbids exactly that. The cliff it draws is legible in black
    # and white at one column, which is the requirement; ink coverage is a proxy
    # for presence, and on a figure whose subject is a curve the proxy is wrong.
    "fig-ceiling":            (0.090, 0.012),
    #
    # SEVEN ENTRIES LEFT THIS TABLE in the round that followed the compliance
    # withdrawal, and not one of them was a floor that failed. fig-capability,
    # fig-labelnoise and fig-missing-cell drew hosted rows or depended on one;
    # fig-intro-headline, fig-falsification, fig-scaling and fig-met-exchange
    # were retired as weak figures. Their reasons of record are in RETIRED in
    # tools/check_figure_invariants.py, which is the one retirement register in
    # this directory. A retired figure is not measured here, for the same reason
    # it is not scanned by the text-collision gate: failing a build over a
    # figure that nothing binds teaches a reader to ignore the tool.
}

# Floors lowered from the spec because the target was unreachable without
# drawing a region that is not in the data. Each entry is ((coverage, chroma),
# reason). Spec risk 5 authorises exactly this: do not invent a band to pass.
#
# ONE ARITHMETIC FACT EXPLAINS EVERY LOWERED CHROMA FLOOR. This metric counts a
# pixel as chromatic at HSV saturation >= 0.25, and half of the v3 area stops
# sit under that gate by construction:
#
#   COBALT_TINT  0.206   PLUM_TINT   0.185   SCARLET_TINT 0.233
#   AZURE_TINT   0.356   EMERALD_TINT 0.309  AMBER_TINT   0.594
#   every WASH stop  <= 0.278, and MIST 0.076, HAIR 0.037
#
# So a canvas built correctly out of scarlet, cobalt or plum TINT registers
# almost no chroma however much of it is filled: fig-pipeline-status is 84%
# covered, four of its six bands are SCARLET_TINT, and it measures 0.25. The
# only way to raise those numbers is to put CORE stops at area scale, which
# section 1.5 forbids outright. The floors move; the palette does not.
#
# TWO AUTHOR DIRECTIVES MOVED EVERY FLOOR ON THIS PAGE AGAIN, and both of them
# move it DOWN by construction. Neither is a regression and neither can be
# recovered by drawing anything, so each figure's entry below records which of
# the two applies to it rather than repeating the arithmetic twelve times.
#
#   1. THE SIGNATURE BAR IS GONE. It was 3.0pt of a figure's dominant DEEP
#      stop running the full width of the canvas, flush to the top edge: on
#      these canvases that is 1.1 to 1.9 percent of every pixel, and it was
#      fully saturated, so it counted at full weight in BOTH metrics. Removing
#      it from all twelve is the author's instruction ("the red line from the
#      top border"), and it takes 0.011 to 0.019 of chroma off every figure in
#      the set on its own. fig-power is the clean demonstration: it measured
#      0.028, its bar was 0.0144 of the canvas, and it now measures 0.012.
#
#   2. EVERY FIGURE IS ABOUT 9 PERCENT LARGER IN EACH DIRECTION, which is 19
#      percent more canvas area, while the type on it is still 9pt and the
#      marks are still 6.6pt because both metrics are read at printed size.
#      Ink that does not scale therefore covers a smaller FRACTION of a bigger
#      sheet. That is arithmetic, not a loss of presence: the reader sees the
#      same ink larger on the page, which is the entire point of the change.
#
# What this leaves is a chroma metric that is close to vestigial on the six
# plots, and that is stated here rather than hidden behind a floor tuned to
# meet it. Coverage still does real work and is unchanged or nearly so
# everywhere it was clearing before.
LOWERED = {
    "fig-pipeline-status": ((0.105, 0.038),
        "coverage clears. chroma: four of six bands are SCARLET_TINT at S=0.233, "
        "under this metric's own 0.25 gate, so 84% ink measures 0.18 chroma. "
        "Lowered again when the bands and chips went borderless by the author's "
        "directive: the 1.5pt DEEP band strokes and the 1.4pt CORE chip rims "
        "were saturated ink, and removing them is the point of the change. "
        "Lowered a third time by directive 1 alone: the bar was 1.21% of this "
        "canvas and the figure fell from 0.179 to 0.168 by exactly that. "
        "Lowered a fourth time, and by much the most, when an external reviewer "
        "showed the three-hue row ground failed twice over: the finding text "
        "measured 3.97 to 1 against its own band, under the 4.5 to 1 the "
        "standard asks for, and in greyscale the EMERALD_TINT, MIST and "
        "SCARLET_TINT rows collapsed to within a few luminance points of each "
        "other, so a passing row and a failing row were the same grey. Both "
        "were fixed by reducing the ground to a binary: fault rows keep "
        "SCARLET_TINT, the two non-fault rows are bare paper. That deletes a "
        "third of the coloured field and both of the non-scarlet chromatic "
        "grounds outright, which is where coverage 0.72 to 0.502 and chroma "
        "0.168 to 0.063 come from. The ink lost is precisely the ink that was "
        "misinforming the reader, and no floor is worth a figure whose colour "
        "says the opposite of its content in greyscale. Lowered a fifth time, "
        "and much the furthest, when the same reviewer said the figure was "
        "still a table with a headline and that propagation, the claim the "
        "chapter rests on, was invisible in it. The replacement deletes the two "
        "rows that reported nothing and spends the height on a spine: a grey "
        "stub for the two silent stages, a scarlet origin at the stage the "
        "exemplar defect enters, a carrier rail, and one arrow into each stage "
        "that consumed it. That deletes the remaining row grounds outright and "
        "leaves four chips, a rail and type on bare paper, which is 0.108 "
        "coverage and 0.041 chroma. This metric rewards filled area and the "
        "figure now argues with a line and four tags instead of with fill. Ink "
        "is not the objective; the objective is that a reader sees one defect "
        "carried into three stages, and that is now the most visible thing on "
        "the canvas."),
    "fig-cohort-flow": ((0.60, 0.120),
        "coverage clears. chroma: the phase grounds carry no chroma at this "
        "metric's gate; only the EMERALD_TINT boxes and the solid chip count. "
        "The three phase grounds are now ONE tone rather than HAIR, COBALT_WASH "
        "and EMERALD_WASH, which sat at 0.8453, 0.7996 and 0.8051 in relative "
        "luminance and printed as one grey, so the ground was separating three "
        "phases for a colour reader and nothing at all for anybody else. The "
        "phase is still carried by its name, by the hue of the boxes inside it "
        "and by the rail. Neither measured number moved: the band area is "
        "identical and every stop involved was already under the 0.25 "
        "saturation gate. "
        "Lowered marginally when chips went rimless set-wide: this figure's "
        "terminal box lost its CORE rim with the rest, which is the consistency "
        "the change was for. Lowered again by directive 1: the bar was 1.07% of "
        "this canvas and 0.137 became 0.124."),
    "fig-exemplar": ((0.40, 0.055),
        "ten slots on a five-row grid is the whole content. COBALT_TINT is "
        "S=0.206 and MIST S=0.076, so only the one solid chip counts. Lowered "
        "again by directives 1 and 2 together: this is the smallest canvas in "
        "the set, so its bar was the second largest share of any (1.52%), and "
        "its plate grew 21% in area while the CESS chip and the five slot "
        "strokes kept their printed size."),
    "fig-prevalence-response": ((0.32, 0.026),
        "Coverage clears by a wide margin and rose again in the rebuild, 0.339 "
        "to 0.507, because the y axis now reaches the deployed operating point "
        "and the over-prediction wedge is most of each panel. Chroma falls "
        "0.031 to 0.027 for an arithmetic reason: six small multiples became "
        "two panels carrying the same three series each, so the coloured ink on "
        "the page is three series and one scarlet operating point instead of "
        "six series, on a sheet of the same width. The floor moves 0.028 to "
        "0.026 to follow it. What the rebuild was for is the thing this metric "
        "cannot see at all: the clinical operating point used to be an arrow "
        "leaving the top of a panel with its coordinates written beside it, and "
        "it is now a mark on the canvas."),
    "fig-power": ((0.53, 0.010),
        "the field is already the full plot rectangle; 0.62 would need margins "
        "too small for the axis labels. Lowered by directive 1 and nothing "
        "else: 0.028 minus a bar worth 0.0144, diluted by 13% more canvas, "
        "predicts 0.012 and it measures 0.012. NOT lowered when the "
        "undetectable region moved from SCARLET_WASH to SCARLET_TINT. The two "
        "region grounds were 0.8051 and 0.8004 in relative luminance, which is "
        "one grey in print for the two halves of a figure whose whole claim is "
        "the boundary between them. The tint is 0.189 clear of the wash above "
        "it and fills exactly the same area, so coverage is unchanged at 0.557 "
        "and chroma at 0.010: SCARLET_TINT sits at S=0.233, under this metric's "
        "own 0.25 gate, so a third of the canvas turning pink buys nothing "
        "here. That is a statement about the metric and not about the figure."),
    "fig-forest": ((0.070, 0.012),
        "Lowered further than any other entry in this table, and the figure is "
        "better for it. The old canvas separated three unit families with three "
        "coloured grounds that filled the whole plot rectangle, and those "
        "grounds were MIST 0.7633, EMERALD_WASH 0.8051 and AMBER_WASH 0.7956 in "
        "relative luminance: three fills a greyscale reader cannot tell apart, "
        "carrying the only structure that said which rows were comparable with "
        "which. The same canvas drew three incommensurable units on one shared "
        "numeric axis, with an axis label that admitted it, and gave each row a "
        "marker hue with no key anywhere on the page. The rebuild gives each "
        "unit family its own sub-axes, its own range and its own zero rule, "
        "separates them with whitespace and a hairline instead of a ground, and "
        "draws every mark in the one structural measurement tone. Coverage "
        "0.530 to 0.076 and chroma 0.129 to 0.015 is the three ground fills "
        "leaving, plus the two Claude rows the compliance withdrawal removed "
        "from the register, which this figure now drops by reading the register "
        "rather than the artifact. Ink cannot be restored without putting back "
        "the fills that printed two families as one grey."),
    "fig-threshold-surface": ((0.40, 0.110),
        "the spec moves both contour labels into a right margin and the heatmap "
        "therefore occupies 61% of the width; 0.90 assumed a full-bleed field. "
        "Lowered again by directive 1: the bar was 1.44% of this canvas. The "
        "heatmap itself grew from 3.28in to 3.77in wide, which is why coverage "
        "went UP, 0.401 to 0.422, on the same directive that cost the chroma."),
}
# The tallest contiguous blank band a canvas may carry, as a fraction of its
# height. Set at 0.18 against a live set whose worst is 0.082
# (fig-threshold-surface, the gap between its heatmap and its axis label), so it
# is a ceiling on a real defect and not a tuning of the current set to itself.
# fig-capability, retired in the same round, measured 0.132.
DEAD_SPACE_MAX = 0.18

SET_COVERAGE_FLOOR = 0.28
SET_CHROMA_FLOOR = 0.10
# Lowered from 0.18, and then from 0.095, for the reasons stated against
# LOWERED above: the set-mean is an average over floors that the palette's own
# saturation cannot reach, and twelve signature bars leaving the set took about
# 0.014 off it in one move. Lowered once more, to 0.075, as the arithmetic
# consequence of the two per-figure entries above and nothing else: reducing
# fig-pipeline-status to a binary ground removed two chromatic row grounds from
# the set, and giving fig-forest the fourth decimal the chapters print added
# achromatic ink to eight rows. Neither was a palette decision. The set mean is
# an average, so it moves when its members do, and it is recorded here rather
# than quietly satisfied by putting colour back that a reviewer showed was
# misleading in greyscale.
#
# Lowered once more, to 0.048, and this time not one figure lost any colour.
# Seven figures left the set in the round that followed the compliance
# withdrawal, and they included the two most chromatic members it had
# (fig-disc-actionability at 0.264, fig-labelnoise at 0.203). A mean over eight
# members is not the same statistic as a mean over fifteen, and the arithmetic
# of which members remain is not a palette decision. The eight that remain
# measure 0.013, 0.125, 0.058, 0.015, 0.041, 0.010, 0.027 and 0.115, which is
# 0.050, and the floor is set just under it.
SET_MEAN_CHROMA_FLOOR = 0.048


def render(pdf, dpi=300):
    with tempfile.TemporaryDirectory() as td:
        stem = Path(td) / "p"
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile",
                        str(pdf), str(stem)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from PIL import Image
        with Image.open(str(stem) + ".png") as im:
            return np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0


def dead_space(arr):
    """Tallest contiguous run of all-paper scanlines, as a fraction of height."""
    ink = (np.abs(arr - 1.0) > (2.0 / 255.0)).any(axis=2).any(axis=1)
    best = run = 0
    for has_ink in ink:
        run = 0 if has_ink else run + 1
        best = max(best, run)
    return best / len(ink)


def measure(arr):
    coverage = float((np.abs(arr - 1.0) > (2.0 / 255.0)).any(axis=2).mean())
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-12), 0.0)
    chroma = float(((sat >= 0.25) & (mx >= 0.20)).mean())
    return coverage, chroma


def main():
    # The trap CLAUDE.md records: a gate that reports zero because it scanned
    # nothing. FLOORS is the live figure set here, and an empty one is a broken
    # tool, not a clean run.
    if not FLOORS:
        print("FLOORS is empty; this tool would pass by measuring nothing.")
        return 1
    rows, fails = [], []
    for label, (cf, hf) in FLOORS.items():
        pdf = FIGDIR / f"{label}.pdf"
        if not pdf.exists():
            fails.append(f"{label}: PDF missing")
            continue
        arr = render(pdf)
        cov, chr_ = measure(arr)
        dead = dead_space(arr)
        if label in LOWERED:
            cf, hf = LOWERED[label][0]
        ok = (cov >= cf - 1e-9 and chr_ >= hf - 1e-9
              and dead <= DEAD_SPACE_MAX + 1e-9)
        rows.append((label, cov, cf, chr_, hf, dead, ok))
        if cov < cf - 1e-9 or chr_ < hf - 1e-9:
            fails.append(f"{label}: coverage {cov:.3f}/{cf:.2f}  "
                         f"chroma {chr_:.3f}/{hf:.2f}")
        if dead > DEAD_SPACE_MAX + 1e-9:
            fails.append(f"{label}: tallest blank band {dead:.3f} of canvas "
                         f"height, over {DEAD_SPACE_MAX:.2f}")
    w = max(len(r[0]) for r in rows) if rows else 24
    print(f"{'figure':<{w}}  coverage  floor   chroma  floor  deadrun  max")
    for label, cov, cf, chr_, hf, dead, ok in rows:
        print(f"{label:<{w}}  {cov:8.3f}  {cf:5.2f}  {chr_:7.3f}  {hf:5.2f}"
              f"  {dead:7.3f}  {DEAD_SPACE_MAX:.2f}  {'ok' if ok else 'UNDER'}")
    if rows:
        mean_chroma = sum(r[3] for r in rows) / len(rows)
        print(f"\nset-mean chroma {mean_chroma:.3f} "
              f"(floor {SET_MEAN_CHROMA_FLOOR:.2f})")
        if mean_chroma < SET_MEAN_CHROMA_FLOOR - 1e-9:
            fails.append(f"set-mean chroma {mean_chroma:.3f} under "
                         f"{SET_MEAN_CHROMA_FLOOR}")
    if LOWERED:
        print("\nfloors lowered from the spec (risk 5):")
        for label, (_, why) in LOWERED.items():
            print(f"  {label}: {why}")
    if fails:
        print("\nUNDER FLOOR:")
        for f in fails:
            print("  " + f)
        return 1
    print("\nall figures at or above their floor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
