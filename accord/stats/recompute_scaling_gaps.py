"""Single-budget accuracy gap between the pretrained AG News arm and each
from-scratch arm, from results/accord/scaling/scaling_results.json.

This is the single-budget sweep (no per-arm convergence tuning), distinct
from the pre-registered convergence sweep in results/accord/scaling/
convergence_results.json that licenses the 4.53-point headline figure used
elsewhere as the positive control's adjudicated result. Both describe the
same 100M-arm-versus-pretrained comparison; this script exists so the
single-budget table's own numbers are read from a file rather than typed.

Run:  python3 accord/stats/recompute_scaling_gaps.py
Writes: results/accord/scaling/scaling_gaps.json
"""

import json
import pathlib
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "accord" / "scaling" / "scaling_results.json"
OUT = ROOT / "results" / "accord" / "scaling" / "scaling_gaps.json"


def main():
    rows = json.loads(SRC.read_text())
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["tag"], []).append(r["accuracy"])

    means = {tag: round(st.mean(vals), 4) for tag, vals in by_tag.items()}
    pretrained = means["pretrained_bert"]

    out = {"pretrained_mean_accuracy": pretrained, "gaps_to_pretrained": {}}
    for tag in ("scratch_100M", "scratch_30M", "scratch_10M"):
        out["gaps_to_pretrained"][tag] = round(pretrained - means[tag], 4)
    out["arm_means"] = means
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
