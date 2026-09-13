"""The discrimination test for every frontier labeller row on G_253, generated.

WHY THIS EXISTS
---------------
Chapter 4's strongest claim rests on these p values, and the key the test
register read them from, `fisher_exact_p_two_sided`, was written by nothing in
this repository. `frontier_metrics.py` emits `hypergeom_p_one_sided` and only
that. The two were byte-identical in all four Anthropic rows, so no reported
number was wrong, but four headline p values had no generator and carried a name
the pipeline never computed. That is the defect this project's own rule exists to
prevent: every number is generated, never typed.

They are generated here, from the confusion cells, for every labeller row rather
than the four that happened to have the key. The one-sided and two-sided values
are BOTH computed and compared, because their agreement is an accident of these
margins and not a property of the test. With 12 positives among 253 and between
39 and 224 flagged, the most extreme left-tail table is more probable than the
observed table, so no left-tail table enters the two-sided sum and the two-sided
value collapses onto the right tail. A labeller with a different flag rate can
break that, and then a two-sided p compared against a threshold calibrated on
rows where the two coincided would be comparing unlike things. The script records
whether they agree per row rather than assuming it.

Run:  python3 accord/stats/frontier_discrimination_tests.py
Writes: results/accord/stats/frontier_discrimination_tests.json
"""

import glob
import json
import pathlib

from scipy.stats import fisher_exact, hypergeom

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIFF = ROOT / "results" / "accord" / "differential"
OUT = ROOT / "results" / "accord" / "stats" / "frontier_discrimination_tests.json"


def main():
    rows = {}
    for f in sorted(glob.glob(str(DIFF / "frontier_*metrics*.json"))):
        d = json.loads(pathlib.Path(f).read_text())
        stem = pathlib.Path(f).stem.replace("frontier_", "").replace("_metrics", "")
        for strat, node in (d.get("by_strategy") or {}).items():
            c = node.get("G_253")
            if not c or "tp" not in c:
                continue
            tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
            n, n_pos, n_flag = c["n"], c["n_positives"], c["n_flagged"]
            one = float(hypergeom.sf(tp - 1, n, n_pos, n_flag))
            # Power floor. The smallest p this row could have returned at ANY
            # outcome is the one where every confirmed case falls inside the
            # flagged set. A labeller that flags most of the pool cannot reach
            # significance however good it is, so a null from such a row is not
            # weak evidence against discrimination, it is no evidence: the test
            # was incapable of rejecting before the model saw a pair. This is a
            # sharper statement than the marginal ceiling on effect size, because
            # the conditional test is not biased by a high flag rate; what a high
            # flag rate destroys is power, not calibration.
            best_p = float(hypergeom.sf(min(n_pos, n_flag) - 1, n, n_pos, n_flag))
            odds, two = fisher_exact([[tp, fp], [fn, tn]], alternative="two-sided")
            rows[f"{stem}::{strat}"] = {
                "cells": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                "n": n, "n_positives": n_pos, "n_flagged": n_flag,
                "fisher_one_sided": one,
                "fisher_two_sided": float(two),
                "odds_ratio": (None if odds != odds or odds in (float("inf"),) else float(odds)),
                "best_attainable_p_one_sided": best_p,
                "can_reach_significance_at_0.05": bool(best_p < 0.05),
                "one_and_two_sided_agree": abs(one - float(two)) < 1e-12,
                "as_recorded_in_run_artifact": c.get("hypergeom_p_one_sided"),
                "matches_run_artifact": (c.get("hypergeom_p_one_sided") is not None
                                         and abs(one - c["hypergeom_p_one_sided"]) < 1e-12),
            }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    for k, v in sorted(rows.items(), key=lambda kv: kv[1]["fisher_two_sided"]):
        flag = "AGREE " if v["one_and_two_sided_agree"] else "DIFFER"
        pw = "" if v["can_reach_significance_at_0.05"] else "   <-- ZERO POWER at 0.05"
        print(f"{k:46s} 2s={v['fisher_two_sided']:.6g}  best_p={v['best_attainable_p_one_sided']:.3g}"
              f"  {flag}{pw}")


if __name__ == "__main__":
    main()
