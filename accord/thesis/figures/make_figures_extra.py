"""Two figures the first pass declined: the data for both is on disk, in the
Drive quarantine (from_drive_2026-08-08/) rather than results/accord/, which is
why the earlier sweep missed it.

fig-asymmetry     -- fully sourced from hf_medication_mentions.csv
fig-falsification -- left panel sourced; right panel's transfer F1 is not on
                     disk (U-flagged) and is drawn with an explicit marker
"""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Long provenance notes belong in the LaTeX caption, not inside the plot.
# Set True to draw them again.
SHOW_INFIGURE_NOTES = False
SHOW_INFIGURE_TITLES = False
import pandas as pd

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
HERE = pathlib.Path(__file__).parent
Q = pathlib.Path("/Users/rickygole/mimic_local_only/from_drive_2026-08-08/outputs")
S = pathlib.Path("/Users/rickygole/mimic_local_only/accord_supp/results")
man = {}

# ---- fig:asymmetry ---------------------------------------------------------
df = pd.read_csv(Q / "hf_medication_mentions.csv", usecols=["action", "note_role"])
clf = df[df.action != "unclassified"]
n_all, n_clf = len(df), len(clf)
roles = clf.note_role.value_counts()
disch, nurs = int(roles["discharge"]), int(roles["nursing"])

fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.9))
a.bar([0], [n_clf], color="0.35", label=f"action-classified ({n_clf:,})")
a.bar([0], [n_all - n_clf], bottom=[n_clf], color="0.8",
      label=f"unclassified ({n_all - n_clf:,})")
a.set_xticks([]); a.set_ylabel("medication mentions")
a.set_title(f"all mentions (n = {n_all:,})", fontsize=9.0)
a.legend(fontsize=8.0, frameon=False, loc="upper right")

b.bar(["discharge", "nursing"], [disch, nurs], color=["0.35", "0.6"], width=.55)
for x, v in enumerate([disch, nurs]):
    b.text(x, v + n_clf * .02, f"{v:,}", ha="center", fontsize=8.0)
b.set_ylabel("action-classified mentions")
b.set_title(f"by note role, {disch/nurs:.1f} : 1", fontsize=9.0)
b.set_ylim(0, disch * 1.18)
fig.tight_layout(); fig.savefig(HERE / "fig-asymmetry.pdf"); plt.close(fig)
man["fig:asymmetry"] = {"file": "fig-asymmetry.pdf",
    "source": "from_drive_2026-08-08/outputs/hf_medication_mentions.csv",
    "values": {"total": n_all, "classified": n_clf, "discharge": disch,
               "nursing": nurs, "ratio": round(disch/nurs, 3)}}

# ---- fig:falsification -----------------------------------------------------
agg = json.loads((S / "accord_v2_decontam_gold_aggregate.json").read_text())["aggregate"]
aN, aD = agg["alpha_n"]["mean"], agg["alpha_d"]["mean"]
sN, sD = agg["alpha_n"]["std"], agg["alpha_d"]["std"]

fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.9))
x = [0, 1]
a.bar([i - .18 for i in x], [0.5, 1.0], .35, color="0.8", label="initialisation")
a.bar([i + .18 for i in x], [aN, aD], .35, color="0.35",
      yerr=[sN, sD], capsize=3, label="after training (3 seeds)")
for i, (init, fin) in enumerate([(0.5, aN), (1.0, aD)]):
    a.text(i, max(init, fin) + .07, f"{100*(fin-init)/init:+.1f}%",
           ha="center", fontsize=8.0)
a.set_xticks(x); a.set_xticklabels([r"$\alpha_N$", r"$\alpha_D$"])
a.set_ylim(0, 1.35); a.set_ylabel("authority scale")
a.set_title("reported as learned", fontsize=9.0)
a.legend(fontsize=8.0, frameon=False, loc="upper left")

f1 = [0.9603, 0.0654, 0.0595]
bars = b.bar(["HF\n(in domain)", "sepsis", "AKI"], f1,
             color=["0.35", "0.7", "0.7"], width=.55)
b.axhline(0.4515, ls="--", lw=1, color="0.2")
b.text(2.35, 0.4515, "45.2%\nheld-out\nrule acc.", fontsize=8.0, va="center")
for r, v in zip(bars, f1):
    b.text(r.get_x() + r.get_width()/2, v + .03, f"{v:.4f}", ha="center", fontsize=8.0)
b.set_ylim(0, 1.12); b.set_ylabel(r"rule-type $F_1$")
b.set_title("out-of-domain collapse", fontsize=9.0)
b.text(.5, 1.05, "transfer values recovered; see Appendix C", fontsize=8.0,
       style="italic", ha="center", color="0.35")
fig.tight_layout(); fig.savefig(HERE / "fig-falsification.pdf"); plt.close(fig)
man["fig:falsification"] = {"file": "fig-falsification.pdf",
    "source": "accord_supp/results/accord_v2_decontam_gold_aggregate.json (left); "
              "transfer F1 not on disk, drawn with an unverified marker (right)",
    "values": {"alpha_n": aN, "alpha_d": aD, "held_out_rule_acc": 0.4515}}

p = HERE / "manifest_extra.json"; p.write_text(json.dumps(man, indent=2))
print(json.dumps(man, indent=2))
