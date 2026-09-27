"""C134e: does any stored artifact carry DECONTAMINATED-ensemble per-pair scores
for the 46 ambiguous gold pairs?

Chapter 5: "The checks also cannot reach the $46$ ambiguous pairs, which
outnumber the $12$ positives". The decontaminated three-seed run is scored on
G_253 only, so no G_299 reading exists for it, in either direction of the
ambiguous convention. This script does not assume that; it searches every
per-pair score file on the machine and records, per file, how many of the 299
gold pairs it covers, whether the 46 ambiguous pairs are among them, and whether
the file's provenance is the decontaminated run or something else.

If and only if a file is found that covers all 299 pairs AND is provably the
decontaminated ensemble, the script computes AUROC and average precision under
both ambiguous conventions with percentile bootstrap intervals at B = 10000. It
never substitutes a different run's scores for the decontaminated ones.

Local only: reads credentialed per-pair files. The JSON written under results/
records file paths, row counts and label counts only. No pair identifiers of
any admission or subject kind, no note text. The PAIR_nnnn surrogate ids are not
written out either; only counts of them.

    python3 accord/c134_reanalysis/c134e_ambiguous46_score_audit.py
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accord.cloud_paths import assert_local_only  # noqa: E402

MIMIC_LOCAL_ONLY_DIR = pathlib.Path(os.environ.get(
    "MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
LOG_DIR = MIMIC_LOCAL_ONLY_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG = assert_local_only(LOG_DIR / f"c134e_{time.strftime('%Y%m%dT%H%M%S')}.log")
OUT = ROOT / "results" / "accord" / "c134_reanalysis" / "c134e_ambiguous46_score_audit.json"

GOLD = MIMIC_LOCAL_ONLY_DIR / "from_drive_2026-09-13" / "outputs" / "gold_standard_accord_results.csv"
DECONTAM_PRED = MIMIC_LOCAL_ONLY_DIR / "from_drive_2026-09-13" / "outputs" / "gold_eval_decontam_predictions.csv"
DECONTAM_AGG = ROOT / "results" / "accord" / "decontam" / "accord_v2_decontam_gold_aggregate.json"

SCORE_COL = re.compile(r"(prob|score|contra|harm|pred|logit|p_)", re.I)
DECONTAM_MARK = re.compile(r"decontam", re.I)
B = 10000
BOOT_SEED = 42


def boot_ci(y, s, B=B, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    auc, ap = [], []
    while len(auc) < B:
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n:
            auc.append(roc_auc_score(y[i], s[i]))
            ap.append(average_precision_score(y[i], s[i]))
    return {"auroc_ci95": [float(np.percentile(auc, 2.5)), float(np.percentile(auc, 97.5))],
            "ap_ci95": [float(np.percentile(ap, 2.5)), float(np.percentile(ap, 97.5))],
            "B": B, "seed": seed}


def main():
    t0 = time.time()
    with LOG.open("w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
        gold = pd.read_csv(GOLD)
        gold_ids = set(gold.pair_id)
        amb_ids = set(gold.loc[gold.pharm_label == "ambiguous", "pair_id"])
        pos_ids = set(gold.loc[gold.pharm_label == "contradiction", "pair_id"])
        assert len(gold_ids) == 299 and len(amb_ids) == 46 and len(pos_ids) == 12

        roots = [MIMIC_LOCAL_ONLY_DIR, ROOT / "results"]
        seen, candidates = set(), []
        for r in roots:
            for path in sorted(r.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in (".csv", ".json"):
                    continue
                try:
                    if path.stat().st_size > 200_000_000:
                        continue
                    real = path.resolve()
                    if real in seen:
                        continue
                    seen.add(real)
                    head = path.open("rb").read(400_000).decode("utf8", "replace")
                    if "PAIR_" not in head:
                        continue
                    ids = set(re.findall(r"PAIR_\d{4}", path.read_text(errors="replace")))
                except Exception as e:  # unreadable / binary / permission
                    candidates.append({"file": str(path), "unreadable": str(e)})
                    continue
                covered = ids & gold_ids
                if len(covered) < 200:
                    continue
                cols = []
                if path.suffix.lower() == ".csv":
                    try:
                        cols = list(pd.read_csv(path, nrows=2).columns)
                    except Exception:
                        cols = []
                candidates.append({
                    "file": str(path),
                    "n_gold_pairs_covered": len(covered),
                    "n_of_the_46_ambiguous_covered": len(ids & amb_ids),
                    "n_of_the_12_positives_covered": len(ids & pos_ids),
                    "has_score_like_columns": bool([c for c in cols if SCORE_COL.search(c)]),
                    "score_like_columns": [c for c in cols if SCORE_COL.search(c)],
                    "filename_marks_decontamination": bool(DECONTAM_MARK.search(path.name)),
                })

        qualifying = [c for c in candidates
                      if c.get("n_of_the_46_ambiguous_covered", 0) == 46
                      and c.get("filename_marks_decontamination")
                      and c.get("has_score_like_columns")]

        readings = None
        if qualifying:
            src = pathlib.Path(qualifying[0]["file"])
            d = pd.read_csv(src).merge(gold[["pair_id", "pharm_label"]], on="pair_id")
            ens = d[[c for c in d.columns if re.fullmatch(r"prob_seed\d", c)]].mean(axis=1).to_numpy()
            readings = {"source_file": str(src), "conventions": {}}
            for name, pos in (("ambiguous_counted_negative", {"contradiction"}),
                              ("ambiguous_counted_positive", {"contradiction", "ambiguous"})):
                y = d.pharm_label.isin(pos).to_numpy().astype(int)
                readings["conventions"][name] = {
                    "n": int(len(y)), "n_positives": int(y.sum()),
                    "auroc": float(roc_auc_score(y, ens)),
                    "ap": float(average_precision_score(y, ens)),
                    **boot_ci(y, ens)}

        # Provenance check that does not rely on filenames: could any file that DOES
        # cover all 46 ambiguous pairs be the decontaminated run under another name?
        # It would have to agree with the decontaminated per-pair probabilities on the
        # 253 shared rows. Report the closest match found.
        dec = pd.read_csv(DECONTAM_PRED)
        dec["ens"] = dec[[f"prob_seed{i}" for i in range(3)]].mean(axis=1)
        dec_cols = ["ens", "prob_seed0", "prob_seed1", "prob_seed2"]
        matches = []
        for c in candidates:
            p = pathlib.Path(c["file"])
            if c.get("n_of_the_46_ambiguous_covered") != 46 or p.suffix.lower() != ".csv":
                continue
            try:
                dd = pd.read_csv(p)
            except Exception:
                continue
            if "pair_id" not in dd.columns:
                continue
            m = dd.merge(dec[["pair_id"] + dec_cols], on="pair_id")
            if len(m) < 250:
                continue
            for col in dd.columns:
                if col == "pair_id":
                    continue
                v = pd.to_numeric(m[col], errors="coerce")
                if v.isna().all() or v.nunique() < 3:
                    continue
                for tgt in dec_cols:
                    matches.append({
                        "file": p.name, "column": col, "against": tgt,
                        "n_shared_rows": int(len(m)),
                        "max_abs_diff": float(np.max(np.abs(v - m[tgt]))),
                        "spearman_rho": float(spearmanr(v, m[tgt]).statistic)})
        matches.sort(key=lambda r: r["max_abs_diff"])
        provenance = {
            "rule": "a file is the decontaminated run only if some column reproduces the "
                    "decontaminated per-pair probabilities on the 253 shared rows "
                    "(max absolute difference < 1e-6)",
            "n_column_pairs_compared": len(matches),
            "closest_10": matches[:10],
            "any_exact_match": bool(matches and matches[0]["max_abs_diff"] < 1e-6),
        }

        agg = json.loads(DECONTAM_AGG.read_text())
        decontam_pred_rows = int(len(pd.read_csv(DECONTAM_PRED)))
        checkpoints = [agg["per_seed"][str(i)]["checkpoint"] for i in range(3)]
        checkpoints_present = {c: pathlib.Path(c).exists() for c in checkpoints}

    out = {
        "question": "do per-pair DECONTAMINATED-ensemble scores exist for the 46 ambiguous "
                    "gold pairs anywhere on this machine",
        "answer": "yes" if readings else "no",
        "gold_geometry": {"n_gold": 299, "n_contradiction": 12,
                          "n_not_contradiction": 241, "n_ambiguous": 46},
        "decontaminated_run": {
            "only_per_pair_score_file": str(DECONTAM_PRED),
            "rows": decontam_pred_rows,
            "covers": "the 253 non-ambiguous pairs only",
            "ambiguous_pairs_present": 0,
            "aggregate_artifact": str(DECONTAM_AGG.relative_to(ROOT)),
            "aggregate_n_pairs_field": agg["configuration"]["n_pairs"],
            "checkpoints_referenced_by_the_aggregate": checkpoints,
            "checkpoints_present_on_this_machine": checkpoints_present,
        },
        "search": {"roots": [str(MIMIC_LOCAL_ONLY_DIR), str(ROOT / "results")],
                   "n_files_with_at_least_200_gold_pair_ids": len(
                       [c for c in candidates if c.get("n_gold_pairs_covered", 0) >= 200]),
                   "candidates": candidates,
                   "n_qualifying_decontaminated_299_files": len(qualifying)},
        "provenance_check_against_decontaminated_253": provenance,
        "readings_if_available": readings,
        "what_would_have_to_be_regenerated": None if readings else {
            "file": str(DECONTAM_PRED).replace(
                "gold_eval_decontam_predictions.csv",
                "gold_eval_decontam_predictions_299.csv"),
            "how": "score all 299 gold pairs, not the 253 non-ambiguous subset, with each of "
                   "the three decontaminated checkpoints named in "
                   "results/accord/decontam/accord_v2_decontam_gold_aggregate.json "
                   "(per_seed.*.checkpoint), then average the three probabilities per pair",
            "blocker": "none of those three checkpoints is present on this machine; the paths "
                       "recorded in the aggregate are Google Drive paths from the training "
                       "environment, so this requires model inference on restored weights and "
                       "cannot be done from stored scores",
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("answer:", out["answer"])
    print("closest non-decontam file to the decontaminated scores: max_abs_diff="
          f"{provenance['closest_10'][0]['max_abs_diff']:.4f}, exact match="
          f"{provenance['any_exact_match']}")
    print("files with >=200 gold pair ids:", out["search"]["n_files_with_at_least_200_gold_pair_ids"])
    for c in candidates:
        if c.get("n_gold_pairs_covered", 0) >= 200:
            print(f"  amb={c['n_of_the_46_ambiguous_covered']:3d} pos={c['n_of_the_12_positives_covered']:3d} "
                  f"decontam_name={c['filename_marks_decontamination']!s:5s} "
                  f"{pathlib.Path(c['file']).name}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
