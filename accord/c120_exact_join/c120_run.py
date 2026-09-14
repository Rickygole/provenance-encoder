"""C120: deployed weak labeller agreement on the exact five-key join.
See C120_PREREGISTRATION.md. Local only; aggregates only.

    python3 accord/c120_exact_join/c120_run.py
"""
import contextlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "accord" / "stats"))
from marginal_feasibility_gate import mcc, best_table  # noqa: E402

LOG = pathlib.Path.home() / "mimic_local_only" / "logs" / f"c120_run_{time.strftime('%Y%m%dT%H%M%S')}.log"
OUT = ROOT / "results" / "accord" / "c120_exact_join" / "c120.json"
B, SEED = 10000, 42


def cells(y, w):
    tp = int(((w == 1) & (y == 1)).sum()); fp = int(((w == 1) & (y == 0)).sum())
    fn = int(((w == 0) & (y == 1)).sum()); tn = int(((w == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def block(y, w, rng):
    n, n_pos, n_flag = len(y), int(y.sum()), int(w.sum())
    tp, fp, fn, tn = cells(y, w)
    obs = mcc(tp, tn, fp, fn)
    btp, btn, bfp, bfn = best_table(n, n_pos, n_flag)
    ceil = mcc(btp, btn, bfp, bfn)
    null = np.empty(B)
    for b in range(B):
        wp = rng.permutation(w)
        t, f, fn_, tn_ = cells(y, wp)
        null[b] = mcc(t, tn_, f, fn_)
    return {"n": n, "n_positives": n_pos, "n_flagged": n_flag, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "flag_rate": n_flag / n, "mcc": obs, "ceiling_mcc": ceil,
            "fraction_of_ceiling": obs / ceil if ceil else None,
            "p_hypergeom_one_sided": float(hypergeom.sf(tp - 1, n, n_pos, n_flag)),
            "random_join_fraction_ge_observed": float((null >= obs - 1e-12).mean()), "B": B}


def run():
    from reconcile_c28 import build_join
    gold = pd.read_csv("/Users/rickygole/mimic_local_only/derived/gold_standard_accord_results.csv")
    j, _ = build_join()  # asserts 299 rows and 1:1 internally
    n_rows = int(len(j)); one_to_one = bool(j["weak_label"].notna().sum() == 299)
    m = gold[["pair_id", "pharm_label"]].merge(j[["pair_id", "pharm_label"]], on="pair_id", suffixes=("_g", "_j"))
    label_agree = bool(len(m) == 299 and (m["pharm_label_g"] == m["pharm_label_j"]).all())
    out = {"preregistration": "accord/c120_exact_join/C120_PREREGISTRATION.md",
           "gates": {"n_rows": n_rows, "one_to_one": one_to_one, "pharm_label_agrees_all_299": label_agree}}
    if not (n_rows == 299 and one_to_one and label_agree):
        out["outcome"] = "C"
        return out
    rng = np.random.default_rng(SEED)
    y = j["pharm_pos"].to_numpy().astype(int); w = j["weak_pos"].to_numpy().astype(int)
    amb = (j["pharm_label"] == "ambiguous").to_numpy()
    out["g299"] = block(y, w, rng)
    out["g253"] = block(y[~amb], w[~amb], rng)
    out["outcome"] = "A"
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
        res = run()
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print("outcome", res["outcome"], res["gates"])
    for s in ("g299", "g253"):
        if s in res:
            r = res[s]
            print(s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
