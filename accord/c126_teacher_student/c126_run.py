"""C126: teacher (weak labeller) versus student (decontaminated ensemble) AUROC on the gold pairs.
See C126_PREREGISTRATION.md. Local only; aggregates only.

    python3 accord/c126_teacher_student/c126_run.py
"""
import collections
import contextlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accord" / "stats"))
sys.path.insert(0, str(ROOT / "accord" / "control"))
from accord.cloud_paths import assert_local_only  # noqa: E402

LOG = assert_local_only(pathlib.Path.home() / "mimic_local_only" / "logs" / f"c126_run_{time.strftime('%Y%m%dT%H%M%S')}.log")
DEC = pathlib.Path.home() / "mimic_local_only" / "from_drive_2026-09-13" / "outputs" / "gold_eval_decontam_predictions.csv"
OUT = ROOT / "results" / "accord" / "c126_teacher_student" / "c126.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "c126"


class AbortC(RuntimeError):
    pass


def paired_diff(y, a, b, B=10000, seed=43):
    rng = np.random.default_rng(seed); n = len(y); d = []
    while len(d) < B:
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n:
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def run():
    from reconcile_c28 import build_join
    from report import bootstrap_auroc_ci
    facts = json.loads(FACTS.read_text())
    j, _ = build_join()
    if len(j) != 299 or j["weak_label"].isna().any():
        raise AbortC("join not 1:1 on 299 rows")
    dec = pd.read_csv(DEC)
    m = j.merge(dec[["pair_id", "y_true", "prob_seed0", "prob_seed1", "prob_seed2"]], on="pair_id", how="inner")
    if len(m) != 253 or int((m["y_true"] != m["pharm_pos"]).sum()) != 0:
        raise AbortC("G_253 join or label agreement failed")
    y = m["pharm_pos"].to_numpy().astype(int)
    teacher = m["weak_pos"].to_numpy().astype(float)
    student = m[["prob_seed0", "prob_seed1", "prob_seed2"]].to_numpy().mean(axis=1)
    if round(float(roc_auc_score(y, student)), 4) != round(facts["decontam.aggregate.ensemble_auroc"]["value"], 4):
        raise AbortC("ensemble AUROC reproduction failed")

    def block(yy, s):
        ci = bootstrap_auroc_ci(yy, s, n_boot=10_000, seed=42)
        return {"n": int(len(yy)), "n_pos": int(yy.sum()), "auroc": float(roc_auc_score(yy, s)),
                "ci95_lo": ci["ci95"][0], "ci95_hi": ci["ci95"][1]}

    out = collections.OrderedDict()
    out["g253"] = {"teacher": block(y, teacher), "student": block(y, student)}
    diff = float(roc_auc_score(y, student) - roc_auc_score(y, teacher))
    lo, hi = paired_diff(y, student, teacher)
    out["g253"]["student_minus_teacher"] = {"diff": diff, "ci95_lo": lo, "ci95_hi": hi}
    y299 = j["pharm_pos"].to_numpy().astype(int)
    out["g299"] = {"teacher": block(y299, j["weak_pos"].to_numpy().astype(float))}
    rules = m["rule"].to_numpy()
    for name, rs in (("rules12", (1, 2)), ("rules34", (3, 4))):
        k = np.isin(rules, rs)
        out[name] = {"teacher": block(y[k], teacher[k]), "student": block(y[k], student[k])}
    out["outcome"] = "A" if (lo <= 0 <= hi and abs(diff) < 0.05) else "B"
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
        try:
            res = run()
        except AbortC as e:
            OUT.write_text(json.dumps({"outcome": "C", "abort_reason": str(e)}, indent=2))
            raise SystemExit("Outcome C; reason in output JSON")
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]
    src = "results/accord/c126_teacher_student/c126.json"

    def put(prefix, obj, path):
        for k, v in obj.items():
            if isinstance(v, dict):
                put(f"{prefix}.{k}", v, f"{path}.{k}")
            else:
                facts[f"{prefix}.{k}"] = {"artifact": f"{src}::{path}.{k}".replace("::.", "::"), "value": v}
    put(PREFIX, res, "")
    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("g253", "g299", "outcome")}, indent=1))
    print("rules12", res["rules12"]); print("rules34", res["rules34"])


if __name__ == "__main__":
    main()
