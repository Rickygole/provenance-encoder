"""Shared figure conventions for the thesis.

One rule specific to this document: no figure may show a point estimate
without its uncertainty. A thesis about instruments that mislead cannot
itself present a bare number.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette. Vivid, but chosen against two hard constraints.
#
# 1. Greyscale separation, using 0.299R + 0.587G + 0.114B:
#       ACCENT  #0A2795  royal blue    43
#       MUTED   #00817B  deep teal     90
#       LIGHT   #FFD166  vivid amber  211
#    The smallest gap between the three main tones is 47 of 255, so a bar chart
#    still separates when the thesis is printed or photocopied in black and white.
#
# 2. Colour-blind safety. Under the Vienot 1999 protanope and deuteranope
#    simulations the three tones stay far apart in CIELAB (smallest pair dE 69),
#    because the distinction is blue vs teal vs yellow plus a large lightness
#    step, never red against green.
#
# ALERT is the failure/retraction accent. It stays red: against a blue-teal-amber
# palette red still reads as a warning, and at 5.1:1 on white it is legible as
# 8 point text. It is never the only cue on a figure that also uses MUTED,
# because in greyscale those two are close.
ACCENT, MUTED, LIGHT, DEAD = "#0A2795", "#00817B", "#FFD166", "#8E99A6"
ALERT = "#D81E05"

def setup():
    # Identical to the rcParams block in make_figures.py and to the notebook's
    # setup cell, so the same figure drawn from any of the three is the same
    # figure. legend.fontsize was 7.5; nothing in the thesis figures may go
    # below 8 point.
    plt.rcParams.update({
        "font.family": "serif",
        # Pinned. The notebook is the path that ships these figures and it runs
        # in Colab, where Times New Roman is not installed, so it has always
        # drawn DejaVu Serif; these scripts were drawing Times locally, which
        # gave the same figure two different sets of text metrics. DejaVu Serif
        # ships with matplotlib, so this renders identically everywhere. It is
        # also the wider of the two, so the spacing checked here is the safe case.
        "font.serif": ["DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "grid.linewidth": 0.3, "grid.color": "#DCE0E4", "lines.linewidth": 1.2,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
