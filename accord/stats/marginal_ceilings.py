"""The largest Matthews correlation each labeller's marginal permits.

WHY THIS EXISTS
---------------
A labeller that emits hard labels at a fixed positive rate cannot reach an
arbitrary agreement with the gold standard. Holding its flagged count where it
is and placing every confirmed case inside the flagged set gives the best table
it could have produced, and the Matthews correlation of that table is a ceiling
implied by two marginals alone. It is a property of the labelling policy and is
independent of what the labeller knows.

C90 established this for the deployed weak labeller, where the ceiling is 0.0768
and the observed value is 15 percent of it. The same arithmetic applies to every
row this chapter compares, and without it a null is ambiguous between "this
labeller cannot discriminate" and "no thresholded label at this marginal could
have shown discrimination". That ambiguity matters most for the Bedrock rows,
which are reported as a second vendor failing to replicate: Llama 3 8B flags
roughly seven pairs in eight, and a ceiling-bound null is not evidence about a
vendor.

Reported alongside each observed value as a fraction of ceiling, so rows taken at
different marginals are compared on one normalised scale rather than against 1.0,
which none of them can reach.

Run:  python3 accord/stats/marginal_ceilings.py
Writes: results/accord/stats/marginal_ceilings.json
"""

import glob
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIFF = ROOT / "results" / "accord" / "differential"
OUT = ROOT / "results" / "accord" / "stats" / "marginal_ceilings.json"


def mcc(tp, tn, fp, fn):
    den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else (tp * tn - fp * fn) / den


def ceiling(n, n_pos, n_flag):
    """Best table at this marginal: every positive inside the flagged set.

    If the labeller flags fewer pairs than there are positives it cannot catch
    them all, so the best case is that every flagged pair is a positive.
    """
    tp = min(n_pos, n_flag)
    fp = n_flag - tp
    fn = n_pos - tp
    tn = n - tp - fp - fn
    return mcc(tp, tn, fp, fn), {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main():
    rows = {}
    for f in sorted(glob.glob(str(DIFF / "frontier_*_metrics*.json"))
                    + glob.glob(str(DIFF / "frontier_*metrics_*.json"))):
        d = json.loads(pathlib.Path(f).read_text())
        label = pathlib.Path(f).stem
        prov = d.get("provenance") or {}
        # The real model id lives per strategy, not at the top level; both
        # top-level fields are null in every metrics file, so an earlier version
        # of this fell through to the filename and recorded "anthropic" for a row
        # nothing in the artifact identifies as Sonnet 5.
        model = None
        for strat, node in (d.get("by_strategy") or {}).items():
            cell = node.get("G_253")
            if not cell or "tp" not in cell:
                continue
            model = (node.get("model") or prov.get("model") or d.get("model")
                     or label.replace("frontier_", "").replace("_metrics", ""))
            n = cell["n"]; n_pos = cell["n_positives"]; n_flag = cell["n_flagged"]
            cap, best = ceiling(n, n_pos, n_flag)
            obs = cell["mcc"]
            rows[f"{label}::{strat}"] = {
                "model": model,
                "strategy": strat,
                "n": n, "n_positives": n_pos, "n_flagged": n_flag,
                "flag_rate": round(n_flag / n, 4),
                "observed_mcc": round(obs, 4),
                "ceiling_mcc": round(cap, 4),
                "fraction_of_ceiling": round(obs / cap, 4) if cap else None,
                "best_possible_table_at_this_marginal": best,
                "discrimination_p": cell.get("hypergeom_p_one_sided"),
                "ci_excludes_zero": cell.get("mcc_ci_excludes_zero"),
            }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    for k, v in sorted(rows.items(), key=lambda kv: -kv[1]["observed_mcc"]):
        print(f"{v['model'][:34]:34s} {v['strategy']:10s} "
              f"flag={v['flag_rate']:.3f} mcc={v['observed_mcc']:+.4f} "
              f"ceil={v['ceiling_mcc']:.4f} frac={v['fraction_of_ceiling']}")


if __name__ == "__main__":
    main()
