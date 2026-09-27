"""C135 labeller arm, derived comparison: a fallible LLM judge against the error-free gold judge.

Reads only the two committed C135 artifacts and writes
`results/accord/c135_dissociation_cost/c135_labeller_vs_gold.json`. No model call, no network host,
no restricted clinical path. Every number in prose about this arm comes from here.
See C135_PREREGISTRATION.md section (2).

    python3 accord/c135_dissociation_cost/c135_labeller_vs_gold.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import c135_pool as P  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results/accord/c135_dissociation_cost"
SRC_LAB = "results/accord/c135_dissociation_cost/c135_labeller.json"
HEADLINE_RATE = 0.414


def newcombe(k1, n1, k2, n2):
    """Unpaired Newcombe 95 pct interval on p1 minus p2 from the two Wilson intervals.

    The two rates share their 400 candidate slots, so this is conservative for a paired contrast:
    it ignores the positive within-slot correlation and so is wider than the paired interval.
    """
    l1, u1 = P.wilson(k1, n1)
    l2, u2 = P.wilson(k2, n2)
    return l1 - u2, u1 - l2


def main():
    lab = json.loads((ROOT / SRC_LAB).read_text())
    n = lab["n_items_per_rate"]
    true_prec = lab["slot_true_positive_rate"]  # true precision on the 400 slots
    out = {
        "preregistration": "accord/c135_dissociation_cost/C135_PREREGISTRATION.md",
        "derived_from": SRC_LAB,
        "note": (
            "Measured precision under a judge is that judge's flag rate on the displayed evidence. "
            "The gold judge is the SNLI gold label of the displayed pair, the same error-free judge "
            "used in the bias arm, evaluated on the same 400 candidate slots. Interval on a "
            "difference of flag rates is the unpaired Newcombe interval, conservative here because "
            "the slots are shared across rates."
        ),
        "n_items_per_rate": n,
        "slot_true_precision": true_prec,
        "headline_rate": HEADLINE_RATE,
        "models": {},
    }
    for model, block in lab["models"].items():
        rows = []
        for b in block["by_rate"]:
            gold, llm = b["displayed_judge_flag_rate"], b["flag_rate"]
            lo, hi = P.wilson(round(llm * n), n)
            glo, ghi = P.wilson(round(gold * n), n)
            rows.append(
                {
                    "planted_rate": b["planted_rate"],
                    "llm_measured_precision": llm,
                    "llm_measured_precision_wilson_lo": lo,
                    "llm_measured_precision_wilson_hi": hi,
                    "llm_gap_measured_minus_true": llm - true_prec,
                    "gold_measured_precision": gold,
                    "gold_measured_precision_wilson_lo": glo,
                    "gold_measured_precision_wilson_hi": ghi,
                    "gold_gap_measured_minus_true": gold - true_prec,
                    "llm_gap_minus_gold_gap": llm - gold,
                    "accuracy_vs_true_label": b["accuracy_vs_true_label"],
                    "recall_on_true_positives": b["recall_on_true_positives"],
                    "specificity_on_true_negatives": b["specificity_on_true_negatives"],
                    "agreement_with_gold_judge": b["agreement_with_displayed_evidence_judge"],
                }
            )
        blk = {"skipped": block["skipped"], "by_rate": rows}
        base = next((r for r in rows if r["planted_rate"] == 0.0), None)
        head = next((r for r in rows if abs(r["planted_rate"] - HEADLINE_RATE) < 1e-9), None)
        if not block["skipped"] and base is not None and head is not None:
            dl_lo, dl_hi = newcombe(
                round(head["llm_measured_precision"] * n), n,
                round(base["llm_measured_precision"] * n), n,
            )
            dg_lo, dg_hi = newcombe(
                round(head["gold_measured_precision"] * n), n,
                round(base["gold_measured_precision"] * n), n,
            )
            dl = head["llm_measured_precision"] - base["llm_measured_precision"]
            dg = head["gold_measured_precision"] - base["gold_measured_precision"]
            blk["headline"] = {
                "planted_rate": HEADLINE_RATE,
                "llm_drop_from_rate0": dl,
                "llm_drop_newcombe_lo": dl_lo,
                "llm_drop_newcombe_hi": dl_hi,
                "llm_drop_excludes_zero": dl_hi < 0,
                "gold_drop_from_rate0": dg,
                "gold_drop_newcombe_lo": dg_lo,
                "gold_drop_newcombe_hi": dg_hi,
                "llm_drop_minus_gold_drop": dl - dg,
                "llm_drop_over_gold_drop": dl / dg if dg else None,
            }
        out["models"][model] = blk
    out["n_models_complete"] = sum(1 for b in out["models"].values() if not b["skipped"])
    (OUT / "c135_labeller_vs_gold.json").write_text(json.dumps(out, indent=1) + "\n")
    print("written", OUT / "c135_labeller_vs_gold.json",
          "complete models:", out["n_models_complete"])


if __name__ == "__main__":
    main()
