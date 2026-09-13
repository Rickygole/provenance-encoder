"""Whether a labeller could have shown discrimination before it scored a pair.

WHY THIS EXISTS
---------------
Two counts decide, in advance, how much agreement a hard-label decision can
express against a gold standard: how many pairs the gold standard confirms, and
how many the labeller flags. Holding both marginals fixed and placing every
confirmed case inside the flagged set gives the best table the labeller could
have produced, and that table's agreement is a ceiling. It is a property of the
labelling policy, not of what the labeller knows.

The ceiling is arithmetic and old: it is the prevalence-and-bias problem that
Feinstein and Cicchetti (1990) and Byrt, Bishop and Carlin (1993) describe, and
this thesis cites them for it. What is not old, and what this script measures, is
that the ceiling can be reported as a PRECONDITION rather than discovered as an
explanation. For each labeller it emits the smallest p-value the labeller could
possibly have returned at its own marginal, and compares that floor against the
threshold the result is going to be read at. A labeller whose floor sits above
the threshold could not have rejected the null before it saw a single pair. That
is a screening test a reviewer can apply to somebody else's table, it costs no
compute, and it is reported nowhere in the literature this thesis surveys.

Scope, stated because the unqualified version of the claim is false: this bounds
a HARD label. A labeller that emits a score is not bounded this way, because the
threshold can be swept, and a model flagging most of a pool at one threshold can
rank perfectly. Everything here is therefore about thresholded decisions, which
is what the weak supervision literature this thesis surveys overwhelmingly uses.

Run:    python3 accord/stats/marginal_feasibility_gate.py
Reads:  results/accord/differential/frontier_ollama_metrics_*.json  (local runs)
        results/accord/stats/mde_recomputed.json                     (the MDE line)
Writes: results/accord/stats/marginal_feasibility_gate.json
        and registers every value it computes.
"""

import collections
import glob
import json
import math
import pathlib

from scipy.stats import hypergeom

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIFF = ROOT / "results" / "accord" / "differential"
OUT = ROOT / "results" / "accord" / "stats" / "marginal_feasibility_gate.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "feasgate"


def mcc(tp, tn, fp, fn):
    den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn - fp * fn) / den) if den else 0.0


def kappa(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    return (po - pe) / (1 - pe) if pe != 1 else 0.0


def best_table(n, n_pos, n_flag):
    """The most agreeable table these two marginals permit.

    Every confirmed case inside the flagged set, which is only possible when the
    labeller flags at least as many pairs as the gold standard confirms. When it
    flags fewer, the best it can do is fill its flagged set with confirmed cases
    and miss the rest.
    """
    tp = min(n_pos, n_flag)
    fp = n_flag - tp
    fn = n_pos - tp
    tn = n - tp - fp - fn
    return tp, tn, fp, fn


def attainable_floor_p(n, n_pos, n_flag):
    """Smallest one-sided hypergeometric p the labeller's marginal permits.

    The most extreme outcome available is the best table above, so the floor is
    the tail probability of that table. If the floor exceeds the threshold the
    result will be read at, the test could not have rejected at any outcome.
    """
    return float(hypergeom.sf(min(n_pos, n_flag) - 1, n, n_pos, n_flag))


def main():
    mde = json.loads((ROOT / "results" / "accord"
                      / "mde_recomputed.json").read_text())
    # The evaluation's own resolution limit, read rather than typed. Reported
    # beside the ceilings but deliberately NOT compared with them: this is an
    # AUROC and a ceiling here is a Matthews correlation, and there is no
    # conversion between them that holds without further assumptions. Putting
    # them on one scale would be the same defect as a forest plot that shares a
    # numeric axis across unit families. The two bound different things and the
    # chapter has to say which it is invoking.
    mde_auc = mde["strata"]["gold_ambiguous_excluded"]["minimum_detectable_auc"]

    rows = collections.OrderedDict()
    for f in sorted(glob.glob(str(DIFF / "frontier_*metrics*.json"))):
        d = json.loads(pathlib.Path(f).read_text())
        label = pathlib.Path(f).stem.replace("frontier_", "").replace("_metrics", "")
        for strat, node in (d.get("by_strategy") or {}).items():
            for basis, cell in node.items():
                if not isinstance(cell, dict) or "tp" not in cell:
                    continue
                n, n_pos = cell["n"], cell["n_positives"]
                n_flag = cell["n_flagged"]
                tp, tn, fp, fn = best_table(n, n_pos, n_flag)
                ceil_mcc = mcc(tp, tn, fp, fn)
                ceil_kap = kappa(tp, tn, fp, fn)
                floor = attainable_floor_p(n, n_pos, n_flag)
                obs = cell["mcc"]
                rows[f"{label}::{strat}::{basis}"] = collections.OrderedDict([
                    ("model", node.get("model") or label),
                    ("strategy", strat), ("basis", basis),
                    ("n", n), ("n_positives", n_pos), ("n_flagged", n_flag),
                    ("flag_rate", round(n_flag / n, 4)),
                    ("observed_mcc", round(obs, 4)),
                    ("ceiling_mcc", round(ceil_mcc, 4)),
                    ("ceiling_kappa", round(ceil_kap, 4)),
                    ("fraction_of_ceiling",
                     round(obs / ceil_mcc, 4) if ceil_mcc else None),
                    ("attains_ceiling", bool(ceil_mcc and abs(obs - ceil_mcc) < 5e-5)),
                    ("attainable_floor_p_one_sided", floor),
                    # The verdict. Read against the conventional threshold, which
                    # is the one this thesis applies; Section on multiplicity
                    # states that no correction is used, so this is the threshold
                    # every reported p is read at.
                    ("could_reject_at_0_05", bool(floor <= 0.05)),
                ])

    n_rows = len(rows)
    n_infeasible = sum(1 for r in rows.values() if not r["could_reject_at_0_05"])
    n_attaining = sum(1 for r in rows.values() if r["attains_ceiling"])

    out = collections.OrderedDict([
        ("provenance", {
            "generated_by": "accord/stats/marginal_feasibility_gate.py",
            "scope": ("Hard-label decisions only. A score-emitting labeller is "
                      "not bounded by its marginal, because the threshold can "
                      "be swept."),
            "mde_auc_read_from": "results/accord/mde_recomputed.json"
                                 "::strata.gold_ambiguous_excluded.minimum_detectable_auc",
            "threshold": 0.05,
        }),
        ("mde_auc", mde_auc),
        ("n_rows", n_rows),
        ("n_could_not_reject_at_0_05", n_infeasible),
        ("n_attaining_their_ceiling", n_attaining),
        ("rows", rows),
    ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    def put(key, value, node):
        facts[f"{PREFIX}.{key}"] = collections.OrderedDict(
            [("artifact", f"results/accord/stats/marginal_feasibility_gate.json::{node}"),
             ("value", value)])

    for key in ("n_rows", "n_could_not_reject_at_0_05",
                "n_attaining_their_ceiling"):
        put(key, out[key], key)
    for rk, r in rows.items():
        slug = (rk.replace("::", ".").replace(":", "").replace(".", "_")
                .replace("-", "").lower())
        for field in ("flag_rate", "observed_mcc", "ceiling_mcc", "ceiling_kappa",
                      "fraction_of_ceiling", "attainable_floor_p_one_sided",
                      "n", "n_positives", "n_flagged"):
            if r[field] is not None:
                put(f"{slug}.{field}", r[field], f"rows.{rk}.{field}")
    FACTS.write_text(json.dumps(facts, indent=1) + "\n")

    print(f"{n_rows} labeller rows; {n_infeasible} could not reject at 0.05 "
          f"before scoring a pair; {n_attaining} sit on their ceiling. "
          f"MDE AUROC for reference (not compared): {mde_auc}")
    print()
    hdr = f"{'labeller':34s} {'strat':10s} {'basis':7s} {'flag':>6s} {'obs':>8s} {'ceil':>8s} {'kap':>7s} {'frac':>6s} {'floor':>9s} {'gate':>6s}"
    print(hdr)
    for rk, r in sorted(rows.items(), key=lambda kv: -(kv[1]["fraction_of_ceiling"] or 0)):
        print(f"{str(r['model'])[:34]:34s} {r['strategy']:10s} {r['basis']:7s} "
              f"{r['flag_rate']:6.3f} {r['observed_mcc']:+8.4f} "
              f"{r['ceiling_mcc']:8.4f} {r['ceiling_kappa']:7.4f} "
              f"{str(r['fraction_of_ceiling']):>6s} "
              f"{r['attainable_floor_p_one_sided']:9.2e} "
              f"{'PASS' if r['could_reject_at_0_05'] else 'FAIL':>6s}")


if __name__ == "__main__":
    main()
