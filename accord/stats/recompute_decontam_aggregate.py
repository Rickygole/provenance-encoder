"""Recompute the decontaminated detector aggregate (results/accord/decontam/corrected_gold_metrics.json).

That file was committed by hand (407d2a8) with no generator. The per-seed and ensemble AUROCs are already
reproduced by c112_run.py; this script recomputes the rest from the per-pair predictions: per-seed AP,
top-15 true positives, per-seed and ensemble two-sided Mann-Whitney p, the seed mean and population SD,
and the ensemble percentile bootstrap 95 percent interval (B = 10000, seed 42 as recorded).

The bootstrap method is declared here before comparing: resample the 253 rows with replacement using
numpy.random.default_rng(42), skip resamples with one class (not counted toward B), percentile interval.
If that does not reproduce the recorded interval to 4 dp, one alternative is tried and labelled as such:
the legacy numpy.random.seed(42) with randint, same skipping. Whatever reproduces (or neither) is written.

Local only: reads credentialed per-pair scores; stdout/stderr go to ~/mimic_local_only/logs; the output
holds aggregates only.

    python3 accord/stats/recompute_decontam_aggregate.py
"""
import collections
import contextlib
import json
import os
import pathlib
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
MIMIC_LOCAL_ONLY_DIR = pathlib.Path(os.environ.get("MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
PRED = MIMIC_LOCAL_ONLY_DIR / "from_drive_2026-09-13" / "outputs" / "gold_eval_decontam_predictions.csv"
REC = ROOT / "results" / "accord" / "decontam" / "corrected_gold_metrics.json"
OUT = ROOT / "results" / "accord" / "decontam" / "decontam_aggregate_recomputed.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
LOG = MIMIC_LOCAL_ONLY_DIR / "logs" / f"decontam_recompute_{time.strftime('%Y%m%dT%H%M%S')}.log"
PREFIX = "decontamrepro"


def mw_two_sided(y, s):
    return float(mannwhitneyu(s[y == 1], s[y == 0], alternative="two-sided").pvalue)


def boot_default_rng(y, s, B=10000, seed=42):
    rng = np.random.default_rng(seed); n = len(y); out = []
    while len(out) < B:
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n:
            out.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def boot_legacy(y, s, B=10000, seed=42):
    np.random.seed(seed); n = len(y); out = []
    while len(out) < B:
        i = np.random.randint(0, n, n)
        if 0 < y[i].sum() < n:
            out.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    rec = json.loads(REC.read_text())
    with LOG.open("w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
        d = pd.read_csv(PRED)
        y = d["y_true"].to_numpy().astype(int)
        seeds = {s: d[f"prob_seed{s}"].to_numpy() for s in range(3)}
        ens = np.mean([seeds[s] for s in range(3)], axis=0)
        out = collections.OrderedDict([("n", int(len(y))), ("n_positives", int(y.sum()))])
        per = collections.OrderedDict()
        for s, sc in seeds.items():
            top15 = int(y[np.argsort(-sc, kind="stable")[:15]].sum())
            per[f"seed{s}"] = {"auroc": float(roc_auc_score(y, sc)), "ap": float(average_precision_score(y, sc)),
                               "top15_tp": top15, "mannwhitney_p": mw_two_sided(y, sc)}
        out["per_seed"] = per
        aucs = [per[f"seed{s}"]["auroc"] for s in range(3)]
        ci_a = boot_default_rng(y, ens)
        match_a = [round(v, 4) for v in ci_a] == [round(v, 4) for v in rec["aggregate"]["ensemble_ci95"]]
        agg = collections.OrderedDict([
            ("auroc_mean", float(np.mean(aucs))), ("auroc_std_population", float(np.std(aucs))),
            ("ensemble_auroc", float(roc_auc_score(y, ens))), ("ensemble_mannwhitney_p", mw_two_sided(y, ens)),
            ("ci95_declared_method", ci_a), ("ci95_declared_method_reproduces_recorded", bool(match_a))])
        if not match_a:
            ci_b = boot_legacy(y, ens)
            agg["ci95_alternative_legacy_seed"] = ci_b
            agg["ci95_alternative_reproduces_recorded"] = bool(
                [round(v, 4) for v in ci_b] == [round(v, 4) for v in rec["aggregate"]["ensemble_ci95"]])
        out["aggregate"] = agg
        checks = {
            "per_seed_auroc": all(round(per[k]["auroc"], 6) == round(rec["per_seed"][k]["auroc"], 6) for k in per),
            "per_seed_ap": all(round(per[k]["ap"], 6) == round(rec["per_seed"][k]["ap"], 6) for k in per),
            "per_seed_top15": all(per[k]["top15_tp"] == rec["per_seed"][k]["top15_tp"] for k in per),
            "per_seed_mw_p": all(round(per[k]["mannwhitney_p"], 6) == round(rec["per_seed"][k]["mannwhitney_p"], 6) for k in per),
            "mean": round(agg["auroc_mean"], 6) == round(rec["aggregate"]["auroc_mean"], 6),
            "std": round(agg["auroc_std_population"], 6) == round(rec["aggregate"]["auroc_std"], 6),
            "ensemble_auroc": round(agg["ensemble_auroc"], 6) == round(rec["aggregate"]["ensemble_auroc"], 6),
            "ensemble_mw_p": round(agg["ensemble_mannwhitney_p"], 4) == round(rec["aggregate"]["ensemble_mannwhitney_p"], 4),
        }
        out["reproduction_checks"] = checks
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]
    src = "results/accord/decontam/decontam_aggregate_recomputed.json"
    for k, v in agg.items():
        if isinstance(v, list):
            facts[f"{PREFIX}.{k}.lo"] = {"artifact": f"{src}::aggregate.{k}[0]", "value": v[0]}
            facts[f"{PREFIX}.{k}.hi"] = {"artifact": f"{src}::aggregate.{k}[1]", "value": v[1]}
        else:
            facts[f"{PREFIX}.{k}"] = {"artifact": f"{src}::aggregate.{k}", "value": v}
    for k, v in checks.items():
        facts[f"{PREFIX}.check.{k}"] = {"artifact": f"{src}::reproduction_checks.{k}", "value": v}
    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(json.dumps({"aggregate": agg, "checks": checks}))


if __name__ == "__main__":
    main()
