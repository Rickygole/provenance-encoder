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


# Hard-label marginals the thesis prints for labellers whose per-pair predictions
# are not in a frontier_*metrics artifact. The committee review found that the
# gate was applied only to rows that happened to have per-pair files, while the
# thesis printed other hard-label marginals it never gated, which is a selection
# the text did not state. The gate needs only two counts, so every printed
# hard-label marginal can be gated whether or not it was scored: these rows carry
# a ceiling and a floor and no observed agreement, which is exactly the point of
# a check computable before scoring. Each count is read from a registered value;
# rates are converted back to counts and the conversion must land on an integer.
# Each row names its CONFIGURATION (model and prompt), because a configuration
# measured on two nested bases is one configuration, not two. The first version of
# this list counted the evaluation baseline and the replication-prompt llama3:8b
# row separately; the methodology chapter states the replication reused the
# baseline's prompt unchanged, so they are one configuration on G_253 and G_299,
# and the committee re-review caught the double count this list was added to
# remove. The optional last field names a registered observed MCC where the row
# was in fact scored pair by pair, so a scored row is never reported as unscored.
MARGINAL_ONLY = [
    # (row id, configuration, description, basis, n, positives, flagged key,
    #  flagged is a rate?, observed MCC key or None)
    ("deployed_weak_labeller_g_299", "llama3:8b::deployed", "deployed weak labeller",
     "G_299", "gold.n_total", 12, "gold.weaklabel_contradiction", False, None),
    ("llama_baseline_g_253", "llama3:8b::baseline", "llama3:8b, baseline prompt",
     "G_253", 253, 12, "derived.llama_baseline.n_flagged", False,
     "baselines.llama_baseline.mcc"),
    ("replication_llama3_8b_g_299", "llama3:8b::baseline", "llama3:8b, baseline prompt",
     "G_299", "gold.n_total", 12, "overcall.llama3_8b.gold.positive_rate", True, None),
    ("replication_granite3_3_8b_g_299", "granite3.3:8b::baseline",
     "granite3.3:8b, baseline prompt", "G_299", "gold.n_total", 12,
     "overcall.granite3_3_8b.gold.positive_rate", True, None),
    ("replication_qwen3_8b_g_299", "qwen3:8b::baseline", "qwen3:8b, baseline prompt",
     "G_299", "gold.n_total", 12, "overcall.qwen3_8b.gold.positive_rate", True, None),
]


def _reg(facts, key_or_value):
    if isinstance(key_or_value, int):
        return key_or_value
    if key_or_value not in facts or key_or_value.startswith("withdrawn."):
        raise SystemExit(f"marginal-only row needs live register key {key_or_value}")
    return facts[key_or_value]["value"]


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

    facts_in = json.loads(FACTS.read_text())
    # The positive counts in MARGINAL_ONLY are typed as 12 because that is the
    # confirmed count on both bases; check it against the register rather than
    # trusting the literal.
    assert _reg(facts_in, "exemplar.power_analysis.n_positives_total") == 12
    marginal_only = collections.OrderedDict()
    for rid, config, desc, basis, n_k, pos_k, flag_k, is_rate, obs_k in MARGINAL_ONLY:
        n = int(_reg(facts_in, n_k)); n_pos = int(_reg(facts_in, pos_k))
        raw = _reg(facts_in, flag_k)
        n_flag = raw * n if is_rate else raw
        if abs(n_flag - round(n_flag)) > 1e-6:
            raise SystemExit(f"{rid}: {flag_k} x {n} is not a count ({n_flag})")
        n_flag = int(round(n_flag))
        tp, tn, fp, fn = best_table(n, n_pos, n_flag)
        floor = attainable_floor_p(n, n_pos, n_flag)
        marginal_only[rid] = collections.OrderedDict([
            ("configuration", config), ("description", desc), ("basis", basis),
            ("n", n),
            ("observed_mcc", (round(_reg(facts_in, obs_k), 4) if obs_k else None)),
            ("n_positives", n_pos), ("n_flagged", n_flag),
            ("flag_rate", round(n_flag / n, 4)),
            ("ceiling_mcc", round(mcc(tp, tn, fp, fn), 4)),
            ("ceiling_kappa", round(kappa(tp, tn, fp, fn), 4)),
            ("attainable_floor_p_one_sided", floor),
            ("could_reject_at_0_05", bool(floor <= 0.05)),
            ("flagged_read_from", flag_k),
        ])

    n_rows = len(rows)
    n_infeasible = sum(1 for r in rows.values() if not r["could_reject_at_0_05"])
    n_attaining = sum(1 for r in rows.values() if r["attains_ceiling"])

    # The eight scored rows are two labellers by two prompts by two bases, and
    # G_253 is a subset of G_299, so they are four configurations measured on two
    # nested bases, not eight independent labellers. Counted both ways so prose
    # cannot report the larger number without the smaller one beside it.
    configs = {(r["model"], r["strategy"]) for r in rows.values()}
    # A configuration is infeasible if it could not reject on any basis it was
    # measured on; with nested bases that is the conservative reading.
    all_configs = ({f"{m}::frontier_{st}" for m, st in configs}
                   | {r["configuration"] for r in marginal_only.values()})
    all_infeasible = ({f"{r['model']}::frontier_{r['strategy']}" for r in rows.values()
                       if not r["could_reject_at_0_05"]}
                      | {r["configuration"] for r in marginal_only.values()
                         if not r["could_reject_at_0_05"]})
    cfg_infeasible = {(r["model"], r["strategy"]) for r in rows.values()
                      if not r["could_reject_at_0_05"]}
    cfg_attaining = {(r["model"], r["strategy"]) for r in rows.values()
                     if r["attains_ceiling"]}
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
        ("n_configurations", len(configs)),
        ("n_configurations_could_not_reject", len(cfg_infeasible)),
        ("n_configurations_attaining", len(cfg_attaining)),
        ("n_configurations_both", len(cfg_infeasible & cfg_attaining)),
        ("n_marginal_only_rows", len(marginal_only)),
        ("n_marginal_only_could_not_reject",
         sum(1 for r in marginal_only.values() if not r["could_reject_at_0_05"])),
        ("n_all_configurations", len(all_configs)),
        ("n_all_could_not_reject", len(all_infeasible)),
        ("all_could_not_reject_are_llama3",
         all(c.startswith("llama3:8b") for c in all_infeasible)),
        ("n_marginal_only_rows_scored",
         sum(1 for r in marginal_only.values() if r["observed_mcc"] is not None)),
        ("marginal_only", marginal_only),
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
                "n_attaining_their_ceiling", "n_configurations",
                "n_configurations_could_not_reject",
                "n_configurations_attaining", "n_configurations_both",
                "n_marginal_only_rows", "n_marginal_only_could_not_reject",
                "n_all_configurations", "n_all_could_not_reject",
                "n_marginal_only_rows_scored"):
        put(key, out[key], key)
    for rk, r in rows.items():
        slug = (rk.replace("::", ".").replace(":", "").replace(".", "_")
                .replace("-", "").lower())
        for field in ("flag_rate", "observed_mcc", "ceiling_mcc", "ceiling_kappa",
                      "fraction_of_ceiling", "attainable_floor_p_one_sided",
                      "n", "n_positives", "n_flagged"):
            if r[field] is not None:
                put(f"{slug}.{field}", r[field], f"rows.{rk}.{field}")
    for rid, r in marginal_only.items():
        for field in ("n", "n_positives", "n_flagged", "flag_rate", "ceiling_mcc",
                      "ceiling_kappa", "attainable_floor_p_one_sided", "observed_mcc"):
            if r[field] is None:
                continue
            put(f"marginal_only.{rid}.{field}", r[field],
                f"marginal_only.{rid}.{field}")
    FACTS.write_text(json.dumps(facts, indent=1) + "\n")

    print(f"{n_rows} labeller rows; {n_infeasible} could not reject at 0.05 "
          f"before scoring a pair; {n_attaining} sit on their ceiling. "
          f"MDE AUROC for reference (not compared): {mde_auc}")
    print(f"as configurations: {len(configs)} configurations on two nested bases; "
          f"{len(cfg_infeasible)} could not reject, {len(cfg_attaining)} attain, "
          f"{len(cfg_infeasible & cfg_attaining)} both")
    for rid, r in marginal_only.items():
        print(f"  marginal only  {r['description'][:36]:36s} {r['basis']}  "
              f"{r['n_flagged']:3d}/{r['n']}  ceil_mcc={r['ceiling_mcc']:.4f} "
              f"kap={r['ceiling_kappa']:.4f}  floor={r['attainable_floor_p_one_sided']:.3e} "
              f"{'PASS' if r['could_reject_at_0_05'] else 'FAIL'}")
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
