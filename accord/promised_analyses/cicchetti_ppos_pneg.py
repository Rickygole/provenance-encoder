"""Cicchetti and Feinstein positive and negative agreement for every
scored labeller row in the feasibility-gate table.

Chapter 2 cites cicchetti1990resolving for the recommendation that kappa always
be accompanied by separate values of positive and negative agreement, and then
the thesis reports neither. This script supplies both from the confusion cells
already in results/accord/stats/marginal_feasibility_gate.json.

  p_pos = 2*TP / (2*TP + FP + FN)
  p_neg = 2*TN / (2*TN + FP + FN)

Cells are reconstructed from the margins each row already carries:
  TP = tp, FP = n_flagged - tp, FN = n_positives - tp,
  TN = n - n_flagged - FN.
Cohen's kappa and MCC are recomputed from the same cells so the three
coefficients in each row come from one 2x2.

Aggregates only. No note text, no HADM_ID, no SUBJECT_ID. No model inference.
"""
from __future__ import annotations

import collections
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
GATE = ROOT / "results/accord/stats/marginal_feasibility_gate.json"
OUT_DIR = ROOT / "results/accord/promised_analyses"
OUT = OUT_DIR / "cicchetti_ppos_pneg.json"


def cells_from_margins(n, n_positives, n_flagged, tp):
    fp = n_flagged - tp
    fn = n_positives - tp
    tn = n - n_flagged - fn
    return tp, fp, fn, tn


def kappa(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (n * n)
    return (po - pe) / (1.0 - pe) if pe < 1.0 else 0.0


def mcc(tp, fp, fn, tn):
    d = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d > 0 else 0.0


def p_pos(tp, fp, fn, tn):
    d = 2 * tp + fp + fn
    return (2 * tp) / d if d else None


def p_neg(tp, fp, fn, tn):
    d = 2 * tn + fp + fn
    return (2 * tn) / d if d else None


def row_record(rid, r, tp, source):
    n, npos, nflag = r["n"], r["n_positives"], r["n_flagged"]
    TP, FP, FN, TN = cells_from_margins(n, npos, nflag, tp)
    assert TP + FP + FN + TN == n, (rid, TP, FP, FN, TN, n)
    assert min(TP, FP, FN, TN) >= 0, (rid, TP, FP, FN, TN)
    pp, pn = p_pos(TP, FP, FN, TN), p_neg(TP, FP, FN, TN)
    rec = collections.OrderedDict()
    rec["basis"] = r["basis"]
    rec["n"] = n
    rec["tp"], rec["fp"], rec["fn"], rec["tn"] = TP, FP, FN, TN
    rec["flag_rate"] = round(nflag / n, 4)
    # Observed raw agreement, p_o. Chapter 5 line 84 asserts this labeller shows
    # Feinstein and Cicchetti's FIRST paradox, high observed agreement beside a
    # low kappa. p_o is printed here so that claim can be checked.
    rec["observed_agreement_p_o"] = round((TP + TN) / n, 4)
    rec["kappa"] = round(kappa(TP, FP, FN, TN), 4)
    rec["mcc"] = round(mcc(TP, FP, FN, TN), 4)
    rec["positive_agreement"] = round(pp, 4)
    rec["negative_agreement"] = round(pn, 4)
    rec["asymmetry_neg_minus_pos"] = round(pn - pp, 4)
    rec["ceiling_kappa"] = r.get("ceiling_kappa")
    rec["ceiling_mcc"] = r.get("ceiling_mcc")
    rec["observed_mcc_in_gate_artifact"] = r.get("observed_mcc")
    rec["tp_source"] = source
    return rec


def main():
    gate = json.loads(GATE.read_text())
    rows = collections.OrderedDict()
    for rid, r in gate["rows"].items():
        rows[rid] = row_record(rid, r, r["tp"], "gate_artifact_rows_tp")

    # Marginal-only entries that carry borrowed observed counts, plus the
    # deployed weak labeller, whose measured 2x2 is the same cell as the
    # llama3 zero-shot G_299 row. Recorded separately so no row is
    # double-counted in the summary.
    aliases = collections.OrderedDict()
    for rid, r in gate["marginal_only"].items():
        tp = r.get("observed_tp")
        src = "gate_artifact_marginal_only_observed_tp"
        if tp is None:
            same = r.get("same_measured_cell_as")
            if same is None and rid == "deployed_weak_labeller_g_299":
                # flagged_read_from gold.weaklabel_contradiction; identical
                # margins to the llama3 zero-shot G_299 scored row.
                same = "ollama_llama3::zero_shot::G_299"
                src = "identical_margins_to_scored_row"
            if same is None:
                continue
            sr = gate["rows"][same]
            if (sr["n"], sr["n_positives"], sr["n_flagged"]) != (r["n"], r["n_positives"], r["n_flagged"]):
                continue
            tp = sr["tp"]
            if src == "gate_artifact_marginal_only_observed_tp":
                src = "borrowed_from_" + same
        rec = row_record(rid, r, tp, src)
        rec["description"] = r.get("description")
        aliases[rid] = rec

    scored = list(rows.values())
    pos_vals = [x["positive_agreement"] for x in scored]
    neg_vals = [x["negative_agreement"] for x in scored]
    res = collections.OrderedDict()
    res["provenance"] = {
        "generated_by": "accord/promised_analyses/cicchetti_ppos_pneg.py",
        "reads": "results/accord/stats/marginal_feasibility_gate.json",
        "definitions": {
            "positive_agreement": "2*TP / (2*TP + FP + FN)",
            "negative_agreement": "2*TN / (2*TN + FP + FN)",
            "source": "cicchetti1990resolving, cited in msu_ch2_literature.tex line 189",
        },
        "cells_reconstructed_as": "TP=tp; FP=n_flagged-tp; FN=n_positives-tp; TN=n-n_flagged-FN",
        "no_model_inference": True,
    }
    res["scored_rows"] = rows
    res["marginal_only_rows_with_recoverable_cells"] = aliases
    res["summary"] = {
        "n_scored_rows": len(scored),
        "positive_agreement_min": min(pos_vals),
        "positive_agreement_max": max(pos_vals),
        "positive_agreement_median": sorted(pos_vals)[len(pos_vals) // 2],
        "negative_agreement_min": min(neg_vals),
        "negative_agreement_max": max(neg_vals),
        "negative_agreement_median": sorted(neg_vals)[len(neg_vals) // 2],
        "n_rows_negative_agreement_exceeds_positive": sum(1 for x in scored if x["negative_agreement"] > x["positive_agreement"]),
        "n_rows_positive_agreement_below_0_20": sum(1 for x in scored if x["positive_agreement"] < 0.20),
        "n_rows_observed_agreement_below_0_50": sum(1 for x in scored if x["observed_agreement_p_o"] < 0.50),
        "observed_agreement_min": min(x["observed_agreement_p_o"] for x in scored),
        "observed_agreement_max": max(x["observed_agreement_p_o"] for x in scored),
        "n_rows_kappa_below_0_10": sum(1 for x in scored if x["kappa"] < 0.10),
        "max_asymmetry_neg_minus_pos": max(x["asymmetry_neg_minus_pos"] for x in scored),
        "min_asymmetry_neg_minus_pos": min(x["asymmetry_neg_minus_pos"] for x in scored),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))

    w = max(len(k) for k in rows)
    print(f"{'row':<{w}}  {'n':>4} {'TP':>3} {'FP':>4} {'FN':>3} {'TN':>4} {'kappa':>7} {'MCC':>7} {'p_pos':>7} {'p_neg':>7}")
    for rid, r in rows.items():
        print(f"{rid:<{w}}  {r['n']:>4} {r['tp']:>3} {r['fp']:>4} {r['fn']:>3} {r['tn']:>4} "
              f"{r['kappa']:>7.4f} {r['mcc']:>7.4f} {r['positive_agreement']:>7.4f} {r['negative_agreement']:>7.4f}")
    print()
    for rid, r in aliases.items():
        print(f"[alias] {rid:<{w}} {r['n']:>4} {r['tp']:>3} {r['fp']:>4} {r['fn']:>3} {r['tn']:>4} "
              f"{r['kappa']:>7.4f} {r['mcc']:>7.4f} {r['positive_agreement']:>7.4f} {r['negative_agreement']:>7.4f}  {r['tp_source']}")
    print()
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
