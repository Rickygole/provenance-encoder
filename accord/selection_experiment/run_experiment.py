"""Orchestrator for the selection-conditioned-audit-blindness experiment on
real MIMIC-III mortality data. Order matches the task brief: lineage check ->
classifier non-degeneracy -> POSITIVE CONTROL FIRST -> sweep with CIs and
power layer -> verdict against the pre-registered decision rule in
PREREGISTRATION.md.

Aggregates only. No note text is ever printed or written. Run from this
directory: `python3 run_experiment.py`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hashlib

import numpy as np
from scipy.stats import linregress, t as t_dist


def det_seed(*parts) -> int:
    """Deterministic (process-independent) seed. Python's built-in hash() is
    salted per-process by PYTHONHASHSEED for strings/tuples-of-strings, so
    using it for RNG seeds silently breaks run-to-run reproducibility --
    caught by comparing two runs of this script (numbers differed) before
    this fix went in."""
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for src.data.lineage
sys.path.insert(0, str(Path(__file__).resolve().parent))       # this dir, for sibling imports

from src.data.lineage import task_dir_lineage, check_task_lineage

import data_prep
import classifier as clf_mod
import audit as audit_mod

OUT_DIR = Path("/Users/rickygole/mimic_local_only/accord_selection")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPO_ROOT = Path(__file__).resolve().parents[2]

SEED = 42
TARGET_SELECTION_RATE = 0.30
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
N_REPLICATES = 30           # replicate-averaged trend (stability check)
BOOTSTRAP_B = 5000
POSITIVE_CONTROL_LOGIT_SHIFT = 1.2


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def solve_intercept(beta, target_rate, feat, n=300_000, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(feat), n)
    f = feat[idx]
    lo, hi = -12.0, 12.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = sigmoid(mid + beta * f).mean()
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    t_start = time.time()

    # -----------------------------------------------------------------
    # STEP 0: lineage check -- refuse to proceed on the wrong split.
    # -----------------------------------------------------------------
    section("STEP 0: task-split lineage check")
    lineage = task_dir_lineage(Path("/Users/rickygole/mimic_local_only/task"))
    print(json.dumps({k: v for k, v in lineage.items()}, indent=2))
    check_task_lineage(lineage, REPO_ROOT / "results" / "task_lineage_pin.json")
    expected_hash = "9a9fe073f4ea9544"
    assert lineage["task_split_hash"] == expected_hash, "split hash mismatch -- STOP"
    print(f"Confirmed on pinned split. task_split_hash={lineage['task_split_hash']}")
    print(f"NOTE: on-disk positives = {lineage['train_n_pos']}(train)+"
          f"{lineage['val_n_pos']}(val)+{lineage['test_n_pos']}(test) = "
          f"{lineage['train_n_pos']+lineage['val_n_pos']+lineage['test_n_pos']} total, "
          f"NOT the 3,730/734 recalled in the task brief -- reporting measured values.")

    # -----------------------------------------------------------------
    # STEP 1: fit the ONE group-blind reference classifier on train.
    # -----------------------------------------------------------------
    section("STEP 1: reference classifier (fit once, on train only)")
    vec, clf, threshold, clf_info = clf_mod.fit_reference_classifier(verbose=True)
    ok, failures = clf_mod.check_non_degeneracy(clf_info)
    print(f"\nNon-degeneracy check: {'PASS' if ok else 'FAIL'}")
    for f in failures:
        print(f"  FAILURE: {f}")
    if not ok:
        print("STOPPING: classifier is degenerate, refusing to run the sweep.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Build the val+test population, join group attributes, score once.
    # -----------------------------------------------------------------
    section("STEP 2: build population (val+test), join ADMISSIONS, score")
    pop = data_prep.build_population()
    hadm_order, texts = data_prep.load_split_text("val")
    hadm_order2, texts2 = data_prep.load_split_text("test")
    hadm_order = hadm_order + hadm_order2
    texts = texts + texts2
    scores = clf_mod.score_population(vec, clf, hadm_order, texts)
    del texts, texts2  # never retained
    # align scores to pop's row order (pop was built as val then test, same order)
    assert list(pop["hadm_id"]) == hadm_order, "row-order mismatch between pop and scored text"
    pop["score"] = scores
    pop["pred"] = (pop["score"] >= threshold).astype(int)

    y = pop["label"].to_numpy()
    pred = pop["pred"].to_numpy()
    score = pop["score"].to_numpy()
    group1 = pop["lang_missing"].to_numpy().astype(bool)

    print(f"Population n={len(pop)}, lang_missing rate={group1.mean():.4f}, "
          f"label base rate={y.mean():.4f}")
    print(f"Population predicted-positive rate at frozen threshold={pred.mean():.4f}")

    # -----------------------------------------------------------------
    # STEP 3: POSITIVE CONTROL -- run FIRST. Plant a known gap, no
    # selection filtering (S=1 for the whole population), confirm the
    # audit code reports it.
    # -----------------------------------------------------------------
    section("STEP 3: POSITIVE CONTROL (planted gap, full population, no selection)")
    perturbed_score = score.copy()
    perturbed_score[group1] = sigmoid(logit(score[group1]) + POSITIVE_CONTROL_LOGIT_SHIFT)
    perturbed_pred = (perturbed_score >= threshold).astype(int)

    pc_audit = audit_mod.audit_pool(y, perturbed_pred, perturbed_score, group1)
    pc_ci = audit_mod.bootstrap_audit_ci(y, perturbed_pred, perturbed_score, group1,
                                          B=BOOTSTRAP_B, seed=SEED)
    dp_point = pc_audit["demographic_parity_gap"]
    dp_ci = pc_ci["demographic_parity_gap"]
    ff_point = pc_audit["four_fifths_ratio"]
    ff_ci = pc_ci["four_fifths_ratio"]
    print(f"Injected logit shift: +{POSITIVE_CONTROL_LOGIT_SHIFT} to lang_missing=1 scores only.")
    print(f"Group pred_pos_rate: g0={pc_audit['g0']['pred_pos_rate']:.4f} "
          f"g1={pc_audit['g1']['pred_pos_rate']:.4f}")
    print(f"Demographic parity gap = {dp_point:.4f}, 95% CI = "
          f"[{dp_ci['ci_lo']:.4f}, {dp_ci['ci_hi']:.4f}]")
    print(f"Four-fifths ratio = {ff_point:.4f}, 95% CI = "
          f"[{ff_ci['ci_lo']:.4f}, {ff_ci['ci_hi']:.4f}]")

    pc_pass = (dp_ci["ci_lo"] > 0.0) and (ff_point < 0.90) and (ff_ci["ci_hi"] < 1.0)
    print(f"\nPOSITIVE CONTROL: {'PASS' if pc_pass else 'FAIL'} "
          f"(CI excludes 0 on DP gap: {dp_ci['ci_lo'] > 0.0}; "
          f"four-fifths point <0.90: {ff_point < 0.90}; CI upper <1.0: {ff_ci['ci_hi'] < 1.0})")
    if not pc_pass:
        print("STOPPING: the audit code did not recover a planted gap. Every "
              "downstream number would be void. Aborting before the sweep.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # STEP 4: SWEEP -- realistic selection on documentation-volume /
    # mention-count text features, NOT on group. Full audit + bootstrap
    # CIs + selection-stage four-fifths + power (MDE) at every point.
    # -----------------------------------------------------------------
    section("STEP 4: SWEEP -- selection on outcome-predictive text features")
    feat = data_prep.doc_feature(pop)
    print(f"doc_feature = 0.7*z(total_chars) + 0.3*z(n_segments); "
          f"corr(doc_feature, label)={np.corrcoef(feat, y)[0,1]:.4f}, "
          f"corr(doc_feature, lang_missing)={np.corrcoef(feat, group1.astype(int))[0,1]:.4f}")

    sweep_rows = []
    for beta in BETA_GRID:
        rng_intercept = np.random.default_rng(det_seed("intercept", beta))
        b0 = solve_intercept(beta, TARGET_SELECTION_RATE, feat, rng=rng_intercept)

        # --- primary realized pool: one deterministic selection draw ---
        p_s = sigmoid(b0 + beta * feat)
        rng_primary = np.random.default_rng(det_seed("primary_selection", beta))
        s = rng_primary.binomial(1, p_s).astype(bool)

        sel_stage = audit_mod.selection_stage_audit(group1, s, B=BOOTSTRAP_B,
                                                      seed=det_seed("sel_ci", beta))

        y_pool, pred_pool, score_pool, g1_pool = y[s], pred[s], score[s], group1[s]
        pool_audit = audit_mod.audit_pool(y_pool, pred_pool, score_pool, g1_pool)
        pool_ci = audit_mod.bootstrap_audit_ci(
            y_pool, pred_pool, score_pool, g1_pool, B=BOOTSTRAP_B,
            seed=det_seed("pool_ci", beta),
        )

        n0_pool, n1_pool = pool_audit["g0"]["n"], pool_audit["g1"]["n"]
        p0_pool, p1_pool = pool_audit["g0"]["pred_pos_rate"], pool_audit["g1"]["pred_pos_rate"]
        mde_dp = audit_mod.mde_two_proportion(n0_pool, n1_pool, p0_pool, p1_pool)

        # --- replicate-averaged trend (stability check, not the primary CI) ---
        rep_g0 = {k: [] for k in ["tpr", "fpr", "ppv", "pred_pos_rate", "calibration_gap"]}
        rep_g1 = {k: [] for k in rep_g0}
        for r in range(N_REPLICATES):
            rng_r = np.random.default_rng(det_seed("replicate", beta, r))
            s_r = rng_r.binomial(1, p_s).astype(bool)
            gr = audit_mod.audit_pool(y[s_r], pred[s_r], score[s_r], group1[s_r])
            for k in rep_g0:
                rep_g0[k].append(gr["g0"][k])
                rep_g1[k].append(gr["g1"][k])
        rep_g0_mean = {k: float(np.nanmean(v)) for k, v in rep_g0.items()}
        rep_g1_mean = {k: float(np.nanmean(v)) for k, v in rep_g1.items()}
        rep_dp_gap = abs(rep_g1_mean["pred_pos_rate"] - rep_g0_mean["pred_pos_rate"])
        rep_tpr_gap = abs(rep_g1_mean["tpr"] - rep_g0_mean["tpr"])
        rep_fpr_gap = abs(rep_g1_mean["fpr"] - rep_g0_mean["fpr"])

        row = dict(
            beta=beta, b0=b0,
            n_pool=int(s.sum()), n0_pool=n0_pool, n1_pool=n1_pool,
            selection_rate_group0=sel_stage["selection_rate_group0"],
            selection_rate_group1=sel_stage["selection_rate_group1"],
            selection_four_fifths=sel_stage["selection_four_fifths_ratio"],
            selection_four_fifths_ci_lo=sel_stage["ci_lo"],
            selection_four_fifths_ci_hi=sel_stage["ci_hi"],
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
            replicate_mean_dp_gap=rep_dp_gap,
            replicate_mean_tpr_gap=rep_tpr_gap,
            replicate_mean_fpr_gap=rep_fpr_gap,
        )
        row["dp_gap_breach"] = audit_mod.breach_call(
            row["demographic_parity_gap"], row["dp_gap_ci_lo"], row["dp_gap_ci_hi"], "gap")
        row["four_fifths_breach"] = audit_mod.breach_call(
            row["four_fifths_ratio_pool"], row["four_fifths_ci_lo"], row["four_fifths_ci_hi"],
            "four_fifths")
        row["adequately_powered"] = bool(mde_dp < audit_mod.GAP_TOLERANCE)
        sweep_rows.append(row)

        print(f"beta={beta:4.1f}  n_pool={row['n_pool']:5d} (g0={n0_pool},g1={n1_pool})  "
              f"sel_4/5={row['selection_four_fifths']:.3f}  "
              f"DP_gap={row['demographic_parity_gap']:.4f} "
              f"[{row['dp_gap_ci_lo']:.4f},{row['dp_gap_ci_hi']:.4f}] "
              f"call={row['dp_gap_breach']:<12}  "
              f"pool_4/5={row['four_fifths_ratio_pool']:.3f} "
              f"[{row['four_fifths_ci_lo']:.3f},{row['four_fifths_ci_hi']:.3f}] "
              f"call={row['four_fifths_breach']:<12}  "
              f"MDE={mde_dp:.4f}  powered={row['adequately_powered']}")

    # -----------------------------------------------------------------
    # STEP 4b: SECONDARY AXES (ETHNICITY-unknown, INSURANCE) at three
    # representative sweep points (beta=0 null, beta nearest the
    # selection-four-fifths~0.47 target, beta=max). Reuses the IDENTICAL
    # realized pool as the primary sweep (same deterministic selection
    # draw, same beta) -- the selection rule never looks at group, so
    # re-auditing the same pool against a different group column is a
    # direct, no-extra-assumptions robustness check.
    # -----------------------------------------------------------------
    section("STEP 4b: secondary group axes (ETHNICITY-unknown, INSURANCE)")
    ethn1 = pop["ethnicity_unknown"].to_numpy().astype(bool)
    insurance = pop["INSURANCE"].to_numpy()
    ins_subset = np.isin(insurance, ["Medicaid", "Private"])
    ins_group1 = insurance == "Medicaid"  # only meaningful within ins_subset

    secondary_axes = {}

    def secondary_audit_at(beta, tag, group_mask, extra_row_mask=None):
        rng_intercept = np.random.default_rng(det_seed("intercept", beta))
        b0 = solve_intercept(beta, TARGET_SELECTION_RATE, feat, rng=rng_intercept)
        p_s = sigmoid(b0 + beta * feat)
        rng_primary = np.random.default_rng(det_seed("primary_selection", beta))
        s = rng_primary.binomial(1, p_s).astype(bool)
        if extra_row_mask is not None:
            s = s & extra_row_mask
        yb, pb, sb, gb = y[s], pred[s], score[s], group_mask[s]
        if gb.sum() == 0 or (~gb).sum() == 0:
            return None
        res = audit_mod.audit_pool(yb, pb, sb, gb)
        ci = audit_mod.bootstrap_audit_ci(yb, pb, sb, gb, B=BOOTSTRAP_B,
                                           seed=det_seed("sec_ci", tag, beta))
        return dict(
            beta=beta, n_pool=int(s.sum()), n0=res["g0"]["n"], n1=res["g1"]["n"],
            demographic_parity_gap=res["demographic_parity_gap"],
            dp_gap_ci=[ci["demographic_parity_gap"]["ci_lo"], ci["demographic_parity_gap"]["ci_hi"]],
            four_fifths_ratio=res["four_fifths_ratio"],
            four_fifths_ci=[ci["four_fifths_ratio"]["ci_lo"], ci["four_fifths_ratio"]["ci_hi"]],
        )

    # We need the target beta, which is computed after the primary sweep
    # loop finishes -- compute it here from sweep_rows directly (cheap).
    _target_row = min(sweep_rows, key=lambda r: abs(r["selection_four_fifths"] - 0.47))
    representative_betas = sorted({0.0, _target_row["beta"], BETA_GRID[-1]})

    for beta in representative_betas:
        eth_res = secondary_audit_at(beta, "ethnicity_unknown", ethn1)
        ins_res = secondary_audit_at(beta, "insurance_medicaid_v_private", ins_group1,
                                      extra_row_mask=ins_subset)
        secondary_axes[str(beta)] = dict(ethnicity_unknown=eth_res,
                                          insurance_medicaid_vs_private=ins_res)
        print(f"beta={beta:4.1f}  ETHNICITY-unknown: DP_gap="
              f"{eth_res['demographic_parity_gap']:.4f} "
              f"CI={['%.4f' % x for x in eth_res['dp_gap_ci']]}  4/5={eth_res['four_fifths_ratio']:.3f}  "
              f"| INSURANCE Medicaid-v-Private (n={ins_res['n_pool']}): DP_gap="
              f"{ins_res['demographic_parity_gap']:.4f} "
              f"CI={['%.4f' % x for x in ins_res['dp_gap_ci']]}  4/5={ins_res['four_fifths_ratio']:.3f}")

    # -----------------------------------------------------------------
    # STEP 5: monotonic-drift check (OLS slope of each gap vs beta).
    # -----------------------------------------------------------------
    section("STEP 5: monotonic-drift check (OLS slope of gap point-estimates vs beta)")
    betas = np.array([r["beta"] for r in sweep_rows])
    drift = {}
    for key in ["demographic_parity_gap", "tpr_gap", "fpr_gap", "ppv_gap", "equalized_odds_gap"]:
        vals = np.array([r[key] for r in sweep_rows])
        res = linregress(betas, vals)
        df = len(betas) - 2
        tcrit = t_dist.ppf(0.975, df)
        slope_ci = (res.slope - tcrit * res.stderr, res.slope + tcrit * res.stderr)
        credible_drift = not (slope_ci[0] <= 0 <= slope_ci[1])
        drift[key] = dict(slope=float(res.slope), slope_ci=list(map(float, slope_ci)),
                           credible_monotonic_drift=bool(credible_drift), r2=float(res.rvalue**2))
        print(f"  {key}: slope={res.slope:.5f} 95% CI=[{slope_ci[0]:.5f},{slope_ci[1]:.5f}] "
              f"credible_drift={credible_drift}  R^2={res.rvalue**2:.3f}")

    # -----------------------------------------------------------------
    # STEP 6: verdict against the pre-registered decision rule.
    # -----------------------------------------------------------------
    section("STEP 6: VERDICT (against PREREGISTRATION.md, written before any of the above)")
    target_row = min(sweep_rows, key=lambda r: abs(r["selection_four_fifths"] - 0.47))
    print(f"Sweep point nearest selection-four-fifths=0.47: beta={target_row['beta']}, "
          f"selection_four_fifths={target_row['selection_four_fifths']:.4f}")

    up_to_target_betas = [r for r in sweep_rows if r["beta"] <= target_row["beta"]]
    powered_rows = [r for r in up_to_target_betas if r["adequately_powered"]]
    unpowered_rows = [r for r in up_to_target_betas if not r["adequately_powered"]]

    # Pre-registered priority order: a powered breach is decisive regardless
    # of whether OTHER (typically higher-beta, smaller-pool) points in the
    # same region happen to be underpowered -- MDE gates each point's OWN
    # evidentiary value, it does not retroactively void an earlier,
    # adequately-powered, CI-confirmed breach. Only call INCONCLUSIVE when
    # NO point in the region has enough power to support any call at all.
    any_breach = any(
        (r["dp_gap_breach"] == "breach" or r["four_fifths_breach"] == "breach")
        and r["adequately_powered"]
        for r in up_to_target_betas
    )
    any_credible_drift = any(d["credible_monotonic_drift"] for d in drift.values())
    region_has_no_power_anywhere = len(powered_rows) == 0

    if any_breach:
        verdict = "REFUTES"
        reason = ("At least one adequately-powered sweep point at or before the "
                  "selection-four-fifths~0.47 target shows a pool-level audit gap or "
                  "four-fifths breach whose CI excludes the tolerance boundary. "
                  f"({len(unpowered_rows)} of {len(up_to_target_betas)} points in the region "
                  "are underpowered and are not used as evidence either way, but the breach "
                  "itself is found at a powered point.)")
    elif region_has_no_power_anywhere:
        verdict = "INCONCLUSIVE"
        reason = (f"None of the {len(up_to_target_betas)} sweep points up to the target beta "
                  f"are adequately powered (MDE >= {audit_mod.GAP_TOLERANCE} throughout); cannot "
                  f"distinguish 'no gap' from 'no power' in this region.")
    elif any_credible_drift:
        verdict = "REFUTES (drift, no individual breach yet)"
        reason = ("No single point breaches tolerance up to the target beta, but at least one "
                  "audit gap shows a statistically credible monotonic increase with beta "
                  "(slope CI excludes 0) -- blindness is assumption-fragile, not exact.")
    else:
        verdict = "SUPPORTS"
        reason = ("Every pool-level audit gap point estimate stays within tolerance with CI "
                  "excluding the boundary, four-fifths stays above 0.80 with CI excluding "
                  "below 0.80, and no gap shows credible monotonic drift with beta, across the "
                  "full sweep up to and including the beta nearest selection-four-fifths~0.47, "
                  "in a region that is adequately powered.")

    print(f"\nVERDICT: {verdict}")
    print(f"REASON: {reason}")

    # -----------------------------------------------------------------
    # SAVE
    # -----------------------------------------------------------------
    out = dict(
        seed=SEED,
        task_lineage=lineage,
        classifier_info=clf_info,
        classifier_non_degeneracy=dict(ok=ok, failures=failures),
        population_n=int(len(pop)), population_lang_missing_rate=float(group1.mean()),
        population_base_rate=float(y.mean()),
        population_pred_pos_rate=float(pred.mean()),
        doc_feature_corr_with_label=float(np.corrcoef(feat, y)[0, 1]),
        doc_feature_corr_with_group=float(np.corrcoef(feat, group1.astype(int))[0, 1]),
        positive_control=dict(
            logit_shift=POSITIVE_CONTROL_LOGIT_SHIFT,
            demographic_parity_gap=dp_point, dp_gap_ci=[dp_ci["ci_lo"], dp_ci["ci_hi"]],
            four_fifths_ratio=ff_point, four_fifths_ci=[ff_ci["ci_lo"], ff_ci["ci_hi"]],
            passed=pc_pass,
        ),
        sweep=sweep_rows,
        secondary_axes=secondary_axes,
        drift_check=drift,
        target_beta_for_selection_47=target_row["beta"],
        verdict=verdict,
        verdict_reason=reason,
        runtime_seconds=round(time.time() - t_start, 1),
    )
    with (OUT_DIR / "selection_experiment_results.json").open("w") as f:
        json.dump(out, f, indent=2)

    import csv
    with (OUT_DIR / "sweep_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)

    print(f"\nSaved results to {OUT_DIR}. Total runtime: {out['runtime_seconds']}s")
    print("ALL_DONE")


if __name__ == "__main__":
    main()
