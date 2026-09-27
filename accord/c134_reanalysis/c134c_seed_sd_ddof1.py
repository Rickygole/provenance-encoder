"""C134c: every three-seed standard deviation the thesis reports, recomputed
under both conventions from the stored per-seed values.

Chapter 4 line 43 onward: "A spread reported as $\\pm$ or as SD in this chapter
is a standard deviation ... That standard deviation follows two conventions,
because the underlying [result files disagree]". Table tab:decontam and
tab:operating are printed under the population convention (ddof = 0);
tab:control299 under the sample convention (ddof = 1). This script recomputes
all of them with ddof = 1, records which convention each printed value came
from, and flags any printed value that matches neither.

All inputs are aggregates already in the repository. No MIMIC row-level data is
read, no identifiers are emitted.

    python3 accord/c134_reanalysis/c134c_seed_sd_ddof1.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = R / "accord" / "c134_reanalysis" / "c134c_seed_sd_ddof1.json"

DECONTAM_GOLD = R / "accord" / "decontam" / "accord_v2_decontam_gold_aggregate.json"
DECONTAM_AUROC = R / "accord" / "decontam" / "corrected_gold_metrics.json"
CONTROL = R / "accord" / "control" / "final_report.json"
SCALING = R / "accord" / "scaling" / "scaling_results.json"
CONVERGENCE = R / "accord" / "scaling" / "convergence_results.json"
TASK_RUNS = R / "task_runs.json"
ABLATION = R / "accord" / "artifact_recovery.json"

TEX = sorted((ROOT / "accord" / "thesis").glob("msu_*.tex"))


def row(quantity, values, printed, artifact, printed_where):
    v = np.asarray(values, dtype=float)
    sd0 = float(np.std(v, ddof=0))
    sd1 = float(np.std(v, ddof=1))
    dp = len(printed.split(".")[1]) if "." in printed else 0
    matches = []
    if f"{sd0:.{dp}f}" == printed:
        matches.append("ddof=0 (population)")
    if f"{sd1:.{dp}f}" == printed:
        matches.append("ddof=1 (sample)")
    return {
        "quantity": quantity,
        "n_seeds": int(v.size),
        "per_seed_values": [float(x) for x in v],
        "mean": float(v.mean()),
        "printed_in_thesis": printed,
        "printed_where": printed_where,
        "sd_ddof0": sd0,
        "sd_ddof1": sd1,
        "sd_ddof0_rounded": f"{sd0:.{dp}f}",
        "sd_ddof1_rounded": f"{sd1:.{dp}f}",
        "convention_the_printed_value_used": matches if matches else ["MATCHES NEITHER"],
        "ddof1_over_ddof0_ratio": (sd1 / sd0) if sd0 > 0 else None,
        "printed_value_changes_under_ddof1": f"{sd1:.{dp}f}" != printed,
    }


def main():
    dg = json.loads(DECONTAM_GOLD.read_text())
    da = json.loads(DECONTAM_AUROC.read_text())
    ctl = json.loads(CONTROL.read_text())
    rows = []

    # --- decontaminated detector, G_253, three seeds (population convention) ---
    auroc_seeds = [da["per_seed"][f"seed{i}"]["auroc"] for i in range(3)]
    rows.append(row("decontam detector AUROC on G_253, per-seed SD", auroc_seeds, "0.0201",
                    "results/accord/decontam/corrected_gold_metrics.json::aggregate.auroc_std",
                    "msu_ch4_results.tex tab:decontam 'Per-seed SD' and body text"))
    for metric, printed in (("precision", "0.0023"), ("recall", "0.0393"),
                            ("kappa", "0.0048"), ("ece", "0.0226")):
        vals = [dg["per_seed"][str(i)]["metrics"][metric] for i in range(3)]
        rows.append(row(f"decontam operating point, {metric}", vals, printed,
                        f"results/accord/decontam/accord_v2_decontam_gold_aggregate.json::per_seed.*.metrics.{metric}",
                        "msu_ch4_results.tex tab:operating 'SD' row"))
    for metric, printed in (("f1_pos", "0.0043"), ("f2_pos", "0.0092"),
                            ("f4_pos", "0.0199"), ("pr_auc", "0.0034")):
        vals = [dg["per_seed"][str(i)]["metrics"][metric] for i in range(3)]
        rows.append(row(f"decontam operating point, {metric} (in artifact aggregate, "
                        "not printed as an SD in a table)", vals, printed,
                        f"results/accord/decontam/accord_v2_decontam_gold_aggregate.json::per_seed.*.metrics.{metric}",
                        "artifact only"))
    for key, field, printed in (("alpha_n", "alpha_nursing", "0.0006"),
                                ("alpha_d", "alpha_discharge", "0.0010")):
        vals = [dg["per_seed"][str(i)]["authority_scale"][field] for i in range(3)]
        rows.append(row(f"decontam authority scale {key}", vals, printed,
                        f"results/accord/decontam/accord_v2_decontam_gold_aggregate.json::per_seed.*.authority_scale.{field}",
                        "msu_ch1_introduction.tex line 197"))

    # --- subject-disjoint control, three seeds (sample convention) ---
    for stratum, printed_auroc, printed_ap, label in (
            ("gold_all", "0.0350", "0.0024", "G_299"),
            ("gold_defect_free_rule34", "0.0158", "0.0108", "G_134")):
        rows.append(row(f"control AUROC on {label}, per-seed SD",
                        ctl["auroc_mean_std_across_seeds"][stratum]["values"], printed_auroc,
                        f"results/accord/control/final_report.json::auroc_mean_std_across_seeds.{stratum}",
                        "msu_ch4_results.tex tab:control299"))
        rows.append(row(f"control AP on {label}, per-seed SD",
                        [s[stratum]["ap"] for s in ctl["per_seed"]], printed_ap,
                        f"results/accord/control/final_report.json::per_seed[*].{stratum}.ap",
                        "msu_ch4_results.tex tab:control299"))

    # --- scaling ladder and convergence, three seeds each ---
    sc = json.loads(SCALING.read_text())
    printed_scaling = {"scratch_10M": "0.0040", "scratch_30M": "0.0009",
                       "scratch_100M": "0.0016", "pretrained_bert": "0.0010"}
    for tag, printed in printed_scaling.items():
        vals = [r["accuracy"] for r in sc if r["tag"] == tag]
        rows.append(row(f"scaling ladder accuracy SD, {tag}", vals, printed,
                        f"results/accord/scaling/scaling_results.json (tag={tag}, accuracy)",
                        "msu_ch4_results.tex scaling table"))
    cv = json.loads(CONVERGENCE.read_text())
    printed_conv = {("scratch_10M", 3): "0.0022", ("scratch_10M", 6): "0.0013",
                    ("scratch_10M", 12): "0.0011", ("scratch_10M", 24): "0.0036",
                    ("scratch_30M", 3): "0.0009", ("scratch_30M", 6): "0.0035",
                    ("scratch_30M", 12): "0.0022", ("scratch_30M", 24): "0.0015",
                    ("scratch_100M", 3): "0.0012", ("scratch_100M", 6): "0.0034",
                    ("scratch_100M", 12): "0.0011", ("scratch_100M", 24): "0.0004",
                    ("pretrained_bert", 3): "0.0012", ("pretrained_bert", 6): "0.0004",
                    ("pretrained_bert", 12): "0.0012", ("pretrained_bert", 24): "0.0029"}
    for (tag, budget), printed in printed_conv.items():
        vals = [r["accuracy"] for r in cv if r["tag"] == tag and r["budget"] == budget]
        if len(vals) != 3:
            rows.append({"quantity": f"convergence {tag} E{budget}", "error":
                         f"expected 3 seeds, found {len(vals)}"})
            continue
        rows.append(row(f"convergence test accuracy SD, {tag} E{budget}", vals, printed,
                        f"results/accord/scaling/convergence_results.json (tag={tag}, budget={budget}, accuracy)",
                        "msu_ch4_results.tex convergence table"))

    # --- C37 pretrained vs scratch, three runs each (different seed sets) ---
    tr = json.loads(TASK_RUNS.read_text())
    rows.append(row("C37 PubMedBERT concat AUROC SD (runs concat_s1..s3)",
                    [tr[f"concat_s{i}"]["auroc"] for i in (1, 2, 3)], "0.0102",
                    "results/task_runs.json::concat_s1/s2/s3.auroc",
                    "msu_ch4_results.tex C37 comparison"))
    rows.append(row("C37 from-scratch AUROC SD (runs ours_current_s0..s2)",
                    [tr[f"ours_current_s{i}"]["auroc"] for i in (0, 1, 2)], "0.0073",
                    "results/task_runs.json::ours_current_s0/s1/s2.auroc",
                    "msu_ch4_results.tex C37 comparison"))

    # --- architecture ablation, three seeds per arm ---
    ab = json.loads(ABLATION.read_text())["group_4_architecture_ablation"][0]
    for arm in ab["arms"]:
        printed = f"{arm['sample_std']:.4f}"
        rows.append(row(f"architecture ablation val loss SD, arm '{arm['arm']}'",
                        arm["per_seed_final_val_loss"], printed,
                        "results/accord/artifact_recovery.json::group_4_architecture_ablation[0].arms[*].sample_std",
                        "msu_ch4_results.tex ablation table / DESIGN.md"))

    tex_blob = "\n".join(p.read_text() for p in TEX)
    for r in rows:
        if "printed_in_thesis" in r:
            r["printed_string_found_in_thesis_tex"] = r["printed_in_thesis"] in tex_blob or \
                r["printed_in_thesis"].rstrip("0").rstrip(".") in tex_blob

    changed = [r for r in rows if r.get("printed_value_changes_under_ddof1")]
    out = {
        "what": "all three-seed standard deviations reported by the thesis, recomputed under "
                "ddof=0 and ddof=1 from the stored per-seed values",
        "n_quantities": len(rows),
        "n_printed_values_that_change_under_ddof1": len(changed),
        "quantities_that_change_under_ddof1": [
            {"quantity": r["quantity"], "printed": r["printed_in_thesis"],
             "ddof1": r["sd_ddof1_rounded"]} for r in changed],
        "conventions_found": sorted({c for r in rows for c in
                                     r.get("convention_the_printed_value_used", [])}),
        "note": "ddof=1 / ddof=0 ratio is sqrt(3/2) = 1.224744871... for every three-seed "
                "quantity by algebra; the per-row ratio is printed as a check.",
        "sqrt_3_over_2": float(np.sqrt(1.5)),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    for r in rows:
        if "printed_in_thesis" in r:
            print(f"{r['quantity'][:58]:58s} printed {r['printed_in_thesis']:>7s} "
                  f"ddof0 {r['sd_ddof0_rounded']:>7s} ddof1 {r['sd_ddof1_rounded']:>7s} "
                  f"{'/'.join(r['convention_the_printed_value_used'])}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
