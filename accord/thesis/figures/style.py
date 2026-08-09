"""Shared figure conventions for the thesis.

One rule specific to this document: no figure may show a point estimate
without its uncertainty. A thesis about instruments that mislead cannot
itself present a bare number.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCENT, MUTED, LIGHT, DEAD = "#1f3a5f", "#7f8c9b", "#d5dae0", "#b0b6bd"

def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "grid.linewidth": 0.3, "grid.color": LIGHT, "lines.linewidth": 1.2,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
