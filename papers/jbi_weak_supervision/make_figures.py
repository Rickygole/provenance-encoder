import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

C136 = "results/accord/c136_gate_validation/c136.json"
C139 = "results/accord/c139_rule_prior/c139.json"

ZERO_RATIO_X = 0.26
USEFULNESS_THRESHOLD = 0.60

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    }
)


def load(root, relpath):
    with open(os.path.join(root, relpath), "r") as handle:
        return json.load(handle)


def figure_collapse(root, outpath):
    data = load(root, C136)
    primary = data["primary_reading"]
    gate_equivalent = primary["ratio_at_gate_equivalent_threshold"]["ratio_threshold"]
    tuned = primary["ratio_best_over_all_thresholds"]["ratio_threshold"]
    disagreements = primary["ratio_at_gate_equivalent_threshold"]["disagreements"]
    tuned_j = primary["ratio_best_over_all_thresholds"]["youden_j"]
    gate_j = primary["gate"]["youden_j"]

    fig, ax = plt.subplots(figsize=(7.1, 3.5))

    styles = {
        ("synthetic", False): dict(marker="o", facecolor="white", edgecolor="#2563EB"),
        ("synthetic", True): dict(marker="o", facecolor="#2563EB", edgecolor="#1E3A5F"),
        ("real_ollama", False): dict(marker="s", facecolor="white", edgecolor="#E11D48"),
        ("real_ollama", True): dict(marker="s", facecolor="#E11D48", edgecolor="#881337"),
    }

    n_zero_ratio = 0
    for row in data["labellers"]:
        ratio = row["ratio_flagrate_over_baserate"]
        auroc = row["student_auroc_minilm"]
        style = styles[(row["family"], bool(row["gate_fail"]))]
        if ratio <= 0:
            x = ZERO_RATIO_X
            n_zero_ratio += 1
        else:
            x = ratio
        ax.scatter(
            [x],
            [auroc],
            marker=style["marker"],
            facecolors=style["facecolor"],
            edgecolors=style["edgecolor"],
            s=34,
            linewidths=0.9,
            zorder=3,
        )

    ax.axhline(
        USEFULNESS_THRESHOLD, color="#0D9488", linestyle="-", linewidth=1.1, zorder=1
    )
    ax.axvline(gate_equivalent, color="#7C3AED", linestyle="-", linewidth=1.6, zorder=2)
    ax.axvline(tuned, color="#F59E0B", linestyle="--", linewidth=1.6, zorder=2)
    ax.axvspan(ZERO_RATIO_X * 0.75, ZERO_RATIO_X * 1.25, color="#FEE2E2", zorder=0)

    ax.set_xscale("log")
    ax.set_xlim(0.19, 45)
    ax.set_ylim(0.15, 0.88)
    ax.set_xlabel("pilot flag-rate ratio $n/K$ (log scale)")
    ax.set_ylabel("student AUROC on 10,000 item test frame")
    ax.set_xticks([0.26, 0.5, 1, 2, 5, 10, 20, 40])
    ax.set_xticklabels(["0", "0.5", "1", "2", "5", "10", "20", "40"])

    ax.annotate(
        "gate-equivalent\nthreshold {:.4f}".format(gate_equivalent),
        xy=(gate_equivalent, 0.845),
        xytext=(gate_equivalent * 0.92, 0.845),
        ha="right",
        va="top",
    )
    ax.annotate(
        "in-sample best\nthreshold {:.4f}".format(tuned),
        xy=(tuned, 0.30),
        xytext=(tuned * 1.1, 0.30),
        ha="left",
        va="center",
    )
    ax.annotate(
        "usefulness threshold {:.2f}".format(USEFULNESS_THRESHOLD),
        xy=(0.21, USEFULNESS_THRESHOLD + 0.012),
        ha="left",
        va="bottom",
    )
    ax.annotate(
        "flagged\nnothing\n({:d})".format(n_zero_ratio),
        xy=(ZERO_RATIO_X, 0.70),
        ha="center",
        va="bottom",
    )

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#2563EB",
            linestyle="none",
            label="synthetic, gate PASS",
        ),
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="#2563EB",
            markeredgecolor="#1E3A5F",
            linestyle="none",
            label="synthetic, gate FAIL",
        ),
        Line2D(
            [],
            [],
            marker="s",
            markerfacecolor="white",
            markeredgecolor="#E11D48",
            linestyle="none",
            label="open-weight model, gate PASS",
        ),
        Line2D(
            [],
            [],
            marker="s",
            markerfacecolor="#E11D48",
            markeredgecolor="#881337",
            linestyle="none",
            label="open-weight model, gate FAIL",
        ),
    ]
    ax.legend(handles=handles, loc="lower center", frameon=False, ncol=2,
              bbox_to_anchor=(0.5, -0.30))

    ax.set_title(
        "The gate's verdict is a function of the horizontal axis alone: "
        "{:d} disagreements on the 24 synthetic labellers, Youden J {:.4f} "
        "against {:.4f} in sample".format(disagreements, gate_j, tuned_j),
        loc="left",
    )

    fig.savefig(outpath)
    plt.close(fig)
    return outpath


def short_label(key):
    family, model, prompt = key.split("::")
    tag = "PS" if family == "promptsens" else "PC"
    return "{} {} {}".format(tag, model, prompt)


def figure_rival(root, outpath):
    data = load(root, C139)
    prior = data["rule_prior"]
    prior_auroc = prior["auroc"]
    prior_lo, prior_hi = prior["bootstrap"]["ci95"]
    corrected = data["bonferroni_alpha_over_n"]

    cells = sorted(
        data["cells"].items(), key=lambda item: item[1]["auroc_graded"]
    )
    labels = [short_label(k) for k, _ in cells]
    aurocs = [v["auroc_graded"] for _, v in cells]
    diffs = [v["auc_diff_graded_minus_prior"] for _, v in cells]
    lows = [v["ci95_diff"][0] for _, v in cells]
    highs = [v["ci95_diff"][1] for _, v in cells]
    positions = list(range(len(cells)))

    fig, axes = plt.subplots(
        1, 2, figsize=(7.1, 4.6), sharey=True, gridspec_kw={"width_ratios": [1, 1.15]}
    )

    left, right = axes

    left.axvspan(prior_lo, prior_hi, color="0.88", zorder=0)
    left.axvline(prior_auroc, color="black", linewidth=1.3, zorder=2)
    left.axvline(0.5, color="black", linewidth=0.8, linestyle=":", zorder=1)
    left.scatter(
        aurocs,
        positions,
        marker="o",
        facecolors="black",
        edgecolors="black",
        s=26,
        zorder=3,
    )
    left.set_yticks(positions)
    left.set_yticklabels(labels)
    left.set_xlim(0.44, 0.86)
    left.set_ylim(-0.8, len(cells) - 0.2)
    left.set_xlabel("AUROC on the 253 pairs")
    left.set_title(
        "(a) rival AUROC {:.4f}, interval {:.4f} to {:.4f}".format(
            prior_auroc, prior_lo, prior_hi
        ),
        loc="left",
    )

    right.axvline(0.0, color="black", linewidth=1.3, zorder=2)
    for pos, diff, lo, hi in zip(positions, diffs, lows, highs):
        right.plot([lo, hi], [pos, pos], color="0.35", linewidth=1.0, zorder=2)
        right.plot([lo, lo], [pos - 0.18, pos + 0.18], color="0.35", linewidth=1.0)
        right.plot([hi, hi], [pos - 0.18, pos + 0.18], color="0.35", linewidth=1.0)
    right.scatter(
        diffs,
        positions,
        marker="D",
        facecolors="black",
        edgecolors="black",
        s=20,
        zorder=3,
    )
    right.set_xlabel("paired DeLong difference, configuration minus rival")
    right.set_title(
        "(b) every interval covers zero; corrected threshold {:.4f}".format(corrected),
        loc="left",
    )

    legend_handles = [
        Patch(facecolor="0.88", edgecolor="none", label="rival bootstrap 95% interval"),
        Line2D([], [], color="black", linewidth=1.3, label="rival AUROC / zero"),
        Line2D([], [], color="black", linewidth=0.8, linestyle=":", label="chance"),
    ]
    left.legend(handles=legend_handles, loc="lower right", frameon=False)

    fig.savefig(outpath)
    plt.close(fig)
    return outpath


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    for relpath in (C136, C139):
        full = os.path.join(root, relpath)
        if not os.path.exists(full):
            raise SystemExit("missing artifact: {}".format(full))
    produced = [
        figure_collapse(root, os.path.join(HERE, "fig1.pdf")),
        figure_rival(root, os.path.join(HERE, "rival_forest_unused.pdf")),
    ]
    for path in produced:
        print(path)


if __name__ == "__main__":
    main()
