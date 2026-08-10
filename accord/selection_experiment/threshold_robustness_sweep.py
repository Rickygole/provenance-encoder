"""E9 threshold-robustness sweep.

C38 (accord/CORRECTIONS.md) found REFUTES for the selection-conditioned-
audit-blindness claim, but flagged its own gap: the whole sweep was run at
ONE operating point (the classifier threshold chosen from val ROC at target
FPR=0.15). C28 already showed a real-data audit read can flip across
operating points (passed all metrics at only 2/35, 9/35, 0/35 thresholds
for three different scorers). This script asks the same question of E9:
does the REFUTES verdict survive across the decision-threshold range, or
was it an artifact of one threshold choice?

Reuses `classifier.py` / `audit.py` / `data_prep.py` UNCHANGED, and reuses
`run_experiment.py`'s own `det_seed` / `solve_intercept` / `BETA_GRID` /
`TARGET_SELECTION_RATE` so the beta-grid selection draws (which beta of
doc_feature) are byte-identical to the original E9 run. The classifier is
fit ONCE (not rebuilt) and the population is scored ONCE; those frozen
scores are reused at every threshold. The ONLY thing that varies across the
outer sweep dimension is the decision threshold applied to those frozen
scores.

The pre-registered breach rule and verdict-mapping logic are taken VERBATIM
from `audit.breach_call` and `run_experiment.py`'s Step 5/6 (drift check +
verdict priority order), applied independently at every threshold.

Aggregates only. No note text is ever printed or written. Run from this
directory: `python3 threshold_robustness_sweep.py`.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import linregress, t as t_dist

import data_prep
import classifier as clf_mod
import audit as audit_mod
import run_experiment as e9  # reuse det_seed / solve_intercept / BETA_GRID / sigmoid --
                              # importing does NOT execute e9.main() (guarded by __main__).

OUT_DIR = Path("/Users/rickygole/mimic_local_only/accord_selection")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = e9.SEED
BETA_GRID = e9.BETA_GRID
TARGET_SELECTION_RATE = e9.TARGET_SELECTION_RATE
BOOTSTRAP_B = e9.BOOTSTRAP_B
det_seed = e9.det_seed
solve_intercept = e9.solve_intercept
sigmoid = e9.sigmoid

# Predicted-positive-rate target grid: base rate (~9%) to ~60%, ~3pp steps.
TARGET_PRED_POS_RATES = [round(float(r), 2) for r in np.arange(0.09, 0.60 + 1e-9, 0.03)]


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    t_start = time.time()

    # -----------------------------------------------------------------
    # STEP 1: fit the ONE reference classifier (unchanged from run_experiment.py).
    # -----------------------------------------------------------------
    section("STEP 1: reference classifier (fit ONCE, reused at every threshold)")
    vec, clf, orig_threshold, clf_info = clf_mod.fit_reference_classifier(verbose=True)
    ok, failures = clf_mod.check_non_degeneracy(clf_info)
    print(f"\nNon-degeneracy check (at original FPR=0.15 operating point): {'PASS' if ok else 'FAIL'}")
    for f in failures:
        print(f"  FAILURE: {f}")
    if not ok:
        print("STOPPING: classifier is degenerate.")
        raise SystemExit(1)

    # -----------------------------------------------------------------
    # STEP 2: build population, score ONCE. Scores are frozen from here on;
    # only the decision threshold applied to them will vary.
    # -----------------------------------------------------------------
    section("STEP 2: build population, score ONCE (frozen scores reused at every threshold)")
    pop = data_prep.build_population()
    hadm_order, texts = data_prep.load_split_text("val")
    hadm_order2, texts2 = data_prep.load_split_text("test")
    hadm_order = hadm_order + hadm_order2
    texts = texts + texts2
    scores = clf_mod.score_population(vec, clf, hadm_order, texts)
    del texts, texts2  # never retained
    assert list(pop["hadm_id"]) == hadm_order, "row-order mismatch between pop and scored text"
    pop["score"] = scores

    y = pop["label"].to_numpy()
    score = pop["score"].to_numpy()
    group1 = pop["lang_missing"].to_numpy().astype(bool)
    n = len(pop)
    base_rate = float(y.mean())
    orig_pred_pos_rate = float((score >= orig_threshold).mean())
    print(f"Population n={n}, lang_missing rate={group1.mean():.4f}, label base rate={base_rate:.4f}")
    print(f"Original E9 operating threshold={orig_threshold:.6f} -> population pred_pos_rate="
          f"{orig_pred_pos_rate:.4f}")

    # -----------------------------------------------------------------
    # Threshold grid: solve, on the FROZEN population score array, for a
    # threshold achieving each target predicted-positive rate. Report the
    # rate ACTUALLY achieved (may differ slightly from target due to ties).
    # -----------------------------------------------------------------
    threshold_grid = []
    for target_rate in TARGET_PRED_POS_RATES:
        thr = float(np.quantile(score, 1.0 - target_rate))
        achieved = float((score >= thr).mean())
        threshold_grid.append(dict(target_rate=target_rate, threshold=thr, achieved_rate=achieved))
    print("\nThreshold grid (target predicted-positive rate -> achieved rate):")
    for g in threshold_grid:
        print(f"  target={g['target_rate']:.2f}  threshold={g['threshold']:.6f}  achieved={g['achieved_rate']:.4f}")

    # -----------------------------------------------------------------
    # doc_feature + beta-grid selection draws -- IDENTICAL to run_experiment.py
    # (same det_seed calls -> same RNG streams -> same realized pools),
    # computed ONCE per beta, reused for every threshold column. These do
    # not depend on pred/score/threshold at all.
    # -----------------------------------------------------------------
    feat = data_prep.doc_feature(pop)
    print(f"\ndoc_feature = 0.7*z(total_chars) + 0.3*z(n_segments); "
          f"corr(doc_feature, label)={np.corrcoef(feat, y)[0, 1]:.4f}, "
          f"corr(doc_feature, lang_missing)={np.corrcoef(feat, group1.astype(int))[0, 1]:.4f}")

    beta_selection = {}
    for beta in BETA_GRID:
        rng_intercept = np.random.default_rng(det_seed("intercept", beta))
        b0 = solve_intercept(beta, TARGET_SELECTION_RATE, feat, rng=rng_intercept)
        p_s = sigmoid(b0 + beta * feat)
        rng_primary = np.random.default_rng(det_seed("primary_selection", beta))
        s = rng_primary.binomial(1, p_s).astype(bool)
        sel_stage = audit_mod.selection_stage_audit(group1, s, B=BOOTSTRAP_B, seed=det_seed("sel_ci", beta))
        beta_selection[beta] = dict(s=s, b0=b0, sel_stage=sel_stage)

    target_beta = min(
        BETA_GRID,
        key=lambda b: abs(beta_selection[b]["sel_stage"]["selection_four_fifths_ratio"] - 0.47),
    )
    print(f"\nTarget beta nearest selection-four-fifths=0.47: beta={target_beta}, "
          f"selection_four_fifths={beta_selection[target_beta]['sel_stage']['selection_four_fifths_ratio']:.4f} "
          f"(threshold-independent by construction -- selection happens upstream of the classifier "
          f"decision, computed once and reused for all 18 thresholds).")

    # -----------------------------------------------------------------
    # STEP 3: main sweep -- len(threshold_grid) x len(BETA_GRID) cells.
    # -----------------------------------------------------------------
    section(f"STEP 3: SWEEP -- {len(threshold_grid)} thresholds x {len(BETA_GRID)} betas = "
            f"{len(threshold_grid) * len(BETA_GRID)} cells")
    all_cells = []
    per_threshold_rows = {}

    for gcell in threshold_grid:
        target_rate, thr, achieved = gcell["target_rate"], gcell["threshold"], gcell["achieved_rate"]
        pred = (score >= thr).astype(int)
        rows = []
        for beta in BETA_GRID:
            sel = beta_selection[beta]
            s = sel["s"]
            sel_stage = sel["sel_stage"]
            y_pool, pred_pool, score_pool, g1_pool = y[s], pred[s], score[s], group1[s]
            pool_audit = audit_mod.audit_pool(y_pool, pred_pool, score_pool, g1_pool)
            pool_ci = audit_mod.bootstrap_audit_ci(
                y_pool, pred_pool, score_pool, g1_pool, B=BOOTSTRAP_B,
                seed=det_seed("thr_pool_ci", target_rate, beta),
            )
            n0_pool, n1_pool = pool_audit["g0"]["n"], pool_audit["g1"]["n"]
            p0_pool, p1_pool = pool_audit["g0"]["pred_pos_rate"], pool_audit["g1"]["pred_pos_rate"]
            mde_dp = audit_mod.mde_two_proportion(n0_pool, n1_pool, p0_pool, p1_pool)

            row = dict(
                target_rate=target_rate, threshold=thr, achieved_rate=achieved,
                beta=beta, n_pool=int(s.sum()), n0_pool=n0_pool, n1_pool=n1_pool,
                selection_four_fifths=sel_stage["selection_four_fifths_ratio"],
                demographic_parity_gap=pool_audit["demographic_parity_gap"],
                dp_gap_ci_lo=pool_ci["demographic_parity_gap"]["ci_lo"],
                dp_gap_ci_hi=pool_ci["demographic_parity_gap"]["ci_hi"],
                tpr_gap=pool_audit["tpr_gap"],
                tpr_gap_ci_lo=pool_ci["tpr_gap"]["ci_lo"], tpr_gap_ci_hi=pool_ci["tpr_gap"]["ci_hi"],
                fpr_gap=pool_audit["fpr_gap"],
                fpr_gap_ci_lo=pool_ci["fpr_gap"]["ci_lo"], fpr_gap_ci_hi=pool_ci["fpr_gap"]["ci_hi"],
                ppv_gap=pool_audit["ppv_gap"],
                ppv_gap_ci_lo=pool_ci["ppv_gap"]["ci_lo"], ppv_gap_ci_hi=pool_ci["ppv_gap"]["ci_hi"],
                equalized_odds_gap=pool_audit["equalized_odds_gap"],
                eq_odds_ci_lo=pool_ci["equalized_odds_gap"]["ci_lo"],
                eq_odds_ci_hi=pool_ci["equalized_odds_gap"]["ci_hi"],
                calibration_gap_diff=pool_audit["calibration_gap_diff"],
                calib_gap_ci_lo=pool_ci["calibration_gap_diff"]["ci_lo"],
                calib_gap_ci_hi=pool_ci["calibration_gap_diff"]["ci_hi"],
                four_fifths_ratio_pool=pool_audit["four_fifths_ratio"],
                four_fifths_ci_lo=pool_ci["four_fifths_ratio"]["ci_lo"],
                four_fifths_ci_hi=pool_ci["four_fifths_ratio"]["ci_hi"],
                mde_demographic_parity_gap=mde_dp,
                # ---- C57/F3: signed + per-group quantities, added alongside
                # the absolute ones above -- the same fields run_experiment.py
                # already persists (C52), which audit_pool()/bootstrap_audit_ci()
                # compute at every cell here too but this script previously
                # discarded before saving. Sign convention is g1
                # (lang_missing==1) MINUS g0 (lang_missing==0) throughout.
                demographic_parity_gap_signed=pool_audit["demographic_parity_gap_signed"],
                dp_gap_signed_ci_lo=pool_ci["demographic_parity_gap_signed"]["ci_lo"],
                dp_gap_signed_ci_hi=pool_ci["demographic_parity_gap_signed"]["ci_hi"],
                demographic_parity_gap_g0=pool_audit["demographic_parity_gap_g0"],
                demographic_parity_gap_g1=pool_audit["demographic_parity_gap_g1"],
                tpr_gap_signed=pool_audit["tpr_gap_signed"],
                tpr_gap_signed_ci_lo=pool_ci["tpr_gap_signed"]["ci_lo"],
                tpr_gap_signed_ci_hi=pool_ci["tpr_gap_signed"]["ci_hi"],
                tpr_gap_g0=pool_audit["tpr_gap_g0"], tpr_gap_g1=pool_audit["tpr_gap_g1"],
                fpr_gap_signed=pool_audit["fpr_gap_signed"],
                fpr_gap_signed_ci_lo=pool_ci["fpr_gap_signed"]["ci_lo"],
                fpr_gap_signed_ci_hi=pool_ci["fpr_gap_signed"]["ci_hi"],
                fpr_gap_g0=pool_audit["fpr_gap_g0"], fpr_gap_g1=pool_audit["fpr_gap_g1"],
                ppv_gap_signed=pool_audit["ppv_gap_signed"],
                ppv_gap_signed_ci_lo=pool_ci["ppv_gap_signed"]["ci_lo"],
                ppv_gap_signed_ci_hi=pool_ci["ppv_gap_signed"]["ci_hi"],
                ppv_gap_g0=pool_audit["ppv_gap_g0"], ppv_gap_g1=pool_audit["ppv_gap_g1"],
                equalized_odds_gap_signed=pool_audit["equalized_odds_gap_signed"],
                eq_odds_signed_ci_lo=pool_ci["equalized_odds_gap_signed"]["ci_lo"],
                eq_odds_signed_ci_hi=pool_ci["equalized_odds_gap_signed"]["ci_hi"],
                equalized_odds_driver=pool_audit["equalized_odds_driver"],
                calibration_gap_diff_signed=pool_audit["calibration_gap_diff_signed"],
                calib_gap_signed_ci_lo=pool_ci["calibration_gap_diff_signed"]["ci_lo"],
                calib_gap_signed_ci_hi=pool_ci["calibration_gap_diff_signed"]["ci_hi"],
                calibration_gap_diff_g0=pool_audit["calibration_gap_diff_g0"],
                calibration_gap_diff_g1=pool_audit["calibration_gap_diff_g1"],
                four_fifths_ratio_directed_numerator_group=
                    pool_audit["four_fifths_ratio_directed"]["numerator_group"],
                four_fifths_ratio_directed_denominator_group=
                    pool_audit["four_fifths_ratio_directed"]["denominator_group"],
            )
            row["dp_gap_breach"] = audit_mod.breach_call(
                row["demographic_parity_gap"], row["dp_gap_ci_lo"], row["dp_gap_ci_hi"], "gap")
            row["four_fifths_breach"] = audit_mod.breach_call(
                row["four_fifths_ratio_pool"], row["four_fifths_ci_lo"], row["four_fifths_ci_hi"],
                "four_fifths")
            row["adequately_powered"] = bool(mde_dp < audit_mod.GAP_TOLERANCE)
            row["cell_breach"] = bool(
                row["adequately_powered"]
                and (row["dp_gap_breach"] == "breach" or row["four_fifths_breach"] == "breach")
            )
            rows.append(row)
            all_cells.append(row)
        per_threshold_rows[target_rate] = rows
        n_breach_here = sum(r["cell_breach"] for r in rows)
        print(f"target_rate={target_rate:.2f} achieved={achieved:.4f} thr={thr:.6f}  "
              f"breaching betas: {n_breach_here}/{len(BETA_GRID)}")

    # -----------------------------------------------------------------
    # STEP 4: per-threshold drift check + verdict -- SAME logic as
    # run_experiment.py Step 5/6, applied independently at every threshold.
    # -----------------------------------------------------------------
    section("STEP 4: per-threshold drift check + verdict (pre-registered rule, applied verbatim)")
    per_threshold_verdicts = []
    for target_rate, rows in per_threshold_rows.items():
        betas = np.array([r["beta"] for r in rows])
        drift = {}
        for key in ["demographic_parity_gap", "tpr_gap", "fpr_gap", "ppv_gap", "equalized_odds_gap"]:
            vals = np.array([r[key] for r in rows])
            res = linregress(betas, vals)
            df = len(betas) - 2
            tcrit = t_dist.ppf(0.975, df)
            slope_ci = (res.slope - tcrit * res.stderr, res.slope + tcrit * res.stderr)
            credible_drift = not (slope_ci[0] <= 0 <= slope_ci[1])
            drift[key] = dict(slope=float(res.slope), slope_ci=list(map(float, slope_ci)),
                               credible_monotonic_drift=bool(credible_drift))

        up_to_target_betas = [r for r in rows if r["beta"] <= target_beta]
        powered_rows = [r for r in up_to_target_betas if r["adequately_powered"]]
        any_breach = any(
            (r["dp_gap_breach"] == "breach" or r["four_fifths_breach"] == "breach") and r["adequately_powered"]
            for r in up_to_target_betas
        )
        any_credible_drift = any(d["credible_monotonic_drift"] for d in drift.values())
        region_has_no_power_anywhere = len(powered_rows) == 0

        first_breach_beta = None
        for r in sorted(up_to_target_betas, key=lambda r: r["beta"]):
            if r["adequately_powered"] and (r["dp_gap_breach"] == "breach" or r["four_fifths_breach"] == "breach"):
                first_breach_beta = r["beta"]
                break

        if any_breach:
            verdict = "REFUTES"
        elif region_has_no_power_anywhere:
            verdict = "INCONCLUSIVE"
        elif any_credible_drift:
            verdict = "REFUTES (drift, no individual breach yet)"
        else:
            verdict = "SUPPORTS"

        per_threshold_verdicts.append(dict(
            target_rate=target_rate, achieved_rate=rows[0]["achieved_rate"], threshold=rows[0]["threshold"],
            verdict=verdict, first_breach_beta=first_breach_beta,
            n_breaching_cells=sum(r["cell_breach"] for r in rows),
            drift=drift,
        ))
        print(f"rate={target_rate:.2f} (achieved {rows[0]['achieved_rate']:.4f})  verdict={verdict:<32}  "
              f"first_breach_beta={first_breach_beta}")

    # -----------------------------------------------------------------
    # STEP 5: aggregate summary.
    # -----------------------------------------------------------------
    section("STEP 5: SUMMARY")
    n_thresholds = len(per_threshold_verdicts)
    refutes_rows = [r for r in per_threshold_verdicts if r["verdict"].startswith("REFUTES")]
    supports_rows = [r for r in per_threshold_verdicts if r["verdict"] == "SUPPORTS"]
    inconclusive_rows = [r for r in per_threshold_verdicts if r["verdict"] == "INCONCLUSIVE"]
    n_refutes = len(refutes_rows)
    total_cells = len(all_cells)
    n_breach_cells = sum(r["cell_breach"] for r in all_cells)

    print(f"REFUTES holds at {n_refutes}/{n_thresholds} operating points "
          f"({100 * n_refutes / n_thresholds:.1f}%).")
    print(f"SUPPORTS holds at {len(supports_rows)}/{n_thresholds} operating points.")
    print(f"INCONCLUSIVE at {len(inconclusive_rows)}/{n_thresholds} operating points.")
    print(f"Cell-level: breach confirmed at {n_breach_cells} of "
          f"{n_thresholds} x {len(BETA_GRID)} = {total_cells} cells.")

    if supports_rows:
        print("\nSUPPORTS-verdict operating points (candidate inversion vs. original E9 REFUTES at FPR=0.15):")
        for r in supports_rows:
            print(f"  rate={r['target_rate']:.2f} achieved={r['achieved_rate']:.4f} threshold={r['threshold']:.6f}")
    else:
        print("\nNo SUPPORTS-verdict operating point found -- no inversion.")

    out = dict(
        seed=SEED,
        classifier_info=clf_info,
        classifier_non_degeneracy=dict(ok=ok, failures=failures),
        original_e9_operating_threshold=float(orig_threshold),
        original_e9_population_pred_pos_rate=orig_pred_pos_rate,
        population_n=int(n), population_base_rate=base_rate,
        doc_feature_corr_with_label=float(np.corrcoef(feat, y)[0, 1]),
        doc_feature_corr_with_group=float(np.corrcoef(feat, group1.astype(int))[0, 1]),
        target_beta_for_selection_47=target_beta,
        threshold_grid=threshold_grid,
        beta_grid=BETA_GRID,
        n_operating_points=n_thresholds,
        n_refutes_operating_points=n_refutes,
        n_supports_operating_points=len(supports_rows),
        n_inconclusive_operating_points=len(inconclusive_rows),
        refutes_fraction=n_refutes / n_thresholds,
        total_cells=total_cells,
        n_breach_cells=n_breach_cells,
        per_threshold_verdicts=per_threshold_verdicts,
        cells=all_cells,
        runtime_seconds=round(time.time() - t_start, 1),
    )
    # C57/F3: this script previously discarded the signed/per-group keys
    # audit_pool() already returns at every cell. Now that they are persisted
    # (see the per-cell `row` construction above), write to NEW filenames so
    # the committed threshold_robustness_results.json /
    # threshold_robustness_cells.csv (which never had a signed variant) are
    # not overwritten.
    with (OUT_DIR / "threshold_robustness_results_signed.json").open("w") as f:
        json.dump(out, f, indent=2)

    with (OUT_DIR / "threshold_robustness_cells_signed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_cells[0].keys()))
        w.writeheader()
        w.writerows(all_cells)

    print(f"\nSaved results to {OUT_DIR}. Total runtime: {out['runtime_seconds']}s")
    print("ALL_DONE")


if __name__ == "__main__":
    main()
