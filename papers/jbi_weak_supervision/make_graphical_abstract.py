#!/usr/bin/env python3
"""Graphical abstract for the JBI submission. Reads committed artifacts only.

    python3 make_graphical_abstract.py
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
c136 = json.loads((ROOT / "results/accord/c136_gate_validation/c136.json").read_text())
c120 = json.loads((ROOT / "results/accord/c120_exact_join/c120.json").read_text())

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "pdf.fonttype": 42,
})

pr = c136["primary_reading"]["gate"]
g253 = c120["g253"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.0))

# Left: the gate reduces to a flag count
labels = ["gate FAIL\nstudent failed", "gate PASS\nstudent failed", "gate PASS\nstudent useful"]
vals = [pr["cell_gatefail_studentfailed"], pr["cell_gatepass_studentfailed"],
        pr["cell_gatepass_studentuseful"]]
cols = ["#E11D48", "#F59E0B", "#0D9488"]
b = ax1.bar(labels, vals, color=cols, edgecolor="black", linewidth=0.8)
ax1.bar_label(b, padding=2, fontsize=10)
ax1.set_ylabel("labellers (of 24)")
ax1.set_ylim(0, max(vals) * 1.35)
ax1.set_title("A pre-registered gate is a threshold on its own flag count\n"
              "%d verdict disagreements against that one number"
              % pr_disagree if False else
              "A pre-registered gate is a threshold on its own flag count\n"
              "%d of %d labellers it passed produced a failing student"
              % (pr["cell_gatepass_studentfailed"], pr["n_gate_pass"]),
              fontsize=10, loc="left")

# Right: observed agreement against its attainable ceiling
obs, ceil = g253["mcc"], g253["ceiling_mcc"]
ax2.barh(["attainable\nceiling", "observed"], [ceil, obs], height=0.45,
         color=["#BFDBFE", "#2563EB"], edgecolor="#1E3A5F", linewidth=0.9)
ax2.text(ceil, 0, "  %.4f" % ceil, va="center", fontsize=10)
ax2.text(obs, 1, "  %.4f" % obs, va="center", fontsize=10)
ax2.set_xlim(0, ceil * 1.45)
ax2.set_ylim(-0.6, 1.6)
ax2.set_xlabel("Matthews correlation on 253 adjudicated pairs")
ax2.set_title("The ceiling costs three integers and is computable before scoring\n"
              "observed is %.2f%% of what the margins allow"
              % (g253["fraction_of_ceiling"] * 100), fontsize=10, loc="left")

fig.tight_layout()
out = HERE / "graphical_abstract.pdf"
fig.tight_layout(pad=0.6)
fig.savefig(out)
fig.savefig(HERE / "graphical_abstract.png", dpi=300)
print(out)
print(HERE / "graphical_abstract.png")
