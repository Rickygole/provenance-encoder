"""Generality Experiment A -- CORRECTED specification, three-cohort re-run.

Why this script exists: experiment_A.py adjusts the missing-language
selection model for log1p(total_note_chars), the SUM of the nursing note
and the discharge summary. The mining rules require the drug to appear in
BOTH notes -- the binding constraint on candidate yield is therefore the
NURSING note, not the total. On HF (n=9,783) the discharge summary is
~91% of total_note_chars (median) and the two note lengths are nearly
uncorrelated on the log scale (r=0.174), so "adjust for total" is ~91%
"adjust for the note that does not bind." This script re-runs the same
selection model on all three cohorts (HF, sepsis, AKI) with:
  (a) the full specification curve: none / total / nursing-only /
      discharge-only / nursing+discharge
  (b) NURSING-decile stratification with Breslow-Day homogeneity test,
      reporting a pooled CMH OR ONLY where homogeneity is not rejected
  (c) the language x log(nursing_chars) interaction
  (d) cluster-robust SEs on SUBJECT_ID throughout, with an explicit
      naive-vs-robust SE comparison (prior HF check: ICC ~ -0.014)

DBSOURCE / CareVue-MetaVision era confound: ICUSTAYS.csv (the only
MIMIC-III table carrying DBSOURCE) was searched for again under
/Users/rickygole/Downloads/ and /Users/rickygole/mimic_local_only/ (see
the `grep`/`find` check logged below) and is still NOT present on this
machine. Per the data rules (local only, no new credentialed downloads),
it is NOT fetched. This remains an open, undocumented-away confound and
is reported as such, not silently dropped.

Local only. Aggregates only -- counts, ORs, CIs, test statistics. No note
TEXT is read into memory beyond computing its character length, and none
is printed or written.
Seed: not applicable (no resampling; all statistics are computed exactly
from the full cohort tables).
"""
import gzip, csv, json, os, warnings
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.contingency_tables as ct
from scipy import stats

warnings.filterwarnings("ignore")

OUTDIR = "/Users/rickygole/mimic_local_only/accord_generality"
os.makedirs(OUTDIR, exist_ok=True)

log_lines = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log_lines.append(s)

# ---------------------------------------------------------------------------
# DBSOURCE / ICUSTAYS.csv availability check (re-verified, not assumed)
# ---------------------------------------------------------------------------
import subprocess
icustays_hits = []
for root in ["/Users/rickygole/Downloads", "/Users/rickygole/mimic_local_only"]:
    try:
        out = subprocess.run(["find", root, "-iname", "*ICUSTAYS*"],
                              capture_output=True, text=True, timeout=300).stdout.strip()
        if out:
            icustays_hits.extend(out.splitlines())
    except Exception as e:
        log(f"ICUSTAYS search error under {root}: {e}")
DBSOURCE_AVAILABLE = len(icustays_hits) > 0
log(f"ICUSTAYS.csv search result: {icustays_hits if icustays_hits else 'NOT FOUND'}")
log(f"DBSOURCE_AVAILABLE = {DBSOURCE_AVAILABLE}")

# ---------------------------------------------------------------------------
# Demographics table -- identical construction to experiment_A.py
# ---------------------------------------------------------------------------
adm = pd.read_csv("/Users/rickygole/Downloads/ADMISSIONS.csv", parse_dates=["ADMITTIME", "DISCHTIME"])
adm["HADM_ID"] = adm["HADM_ID"].astype(float)
pts = pd.read_csv("/Users/rickygole/Downloads/PATIENTS.csv", parse_dates=["DOB"])
demo = adm.merge(pts[["SUBJECT_ID", "GENDER", "DOB"]], on="SUBJECT_ID", how="left")

def lang_group(x):
    if pd.isna(x):
        return "MISSING"
    if x == "ENGL":
        return "ENGLISH"
    return "RECORDED_NON_ENGLISH"
demo["language_group3"] = demo["LANGUAGE"].apply(lang_group)

NAMED_MINORITY = {
    "BLACK/AFRICAN AMERICAN", "HISPANIC OR LATINO", "ASIAN",
    "ASIAN - CHINESE", "HISPANIC/LATINO - PUERTO RICAN", "BLACK/CAPE VERDEAN",
    "WHITE - RUSSIAN", "BLACK/HAITIAN", "ASIAN - ASIAN INDIAN",
    "WHITE - OTHER EUROPEAN", "HISPANIC/LATINO - DOMINICAN", "PORTUGUESE",
    "WHITE - BRAZILIAN", "ASIAN - VIETNAMESE", "AMERICAN INDIAN/ALASKA NATIVE",
    "BLACK/AFRICAN", "MIDDLE EASTERN", "HISPANIC/LATINO - GUATEMALAN",
    "WHITE - EASTERN EUROPEAN", "ASIAN - FILIPINO", "HISPANIC/LATINO - CUBAN",
    "HISPANIC/LATINO - SALVADORAN", "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
    "ASIAN - OTHER", "MULTI RACE ETHNICITY",
}
UNKNOWN_OTHER = {"UNKNOWN/NOT SPECIFIED", "UNABLE TO OBTAIN",
                 "PATIENT DECLINED TO ANSWER", "OTHER"}

def eth_group_broad(x):
    if x == "WHITE":
        return "WHITE"
    if isinstance(x, str) and x.startswith("BLACK"):
        return "BLACK"
    if isinstance(x, str) and x.startswith("HISPANIC"):
        return "HISPANIC"
    if isinstance(x, str) and x.startswith("ASIAN"):
        return "ASIAN"
    if x in UNKNOWN_OTHER:
        return "UNKNOWN_OTHER"
    return "OTHER_NAMED"
demo["ethnicity_broad"] = demo["ETHNICITY"].apply(eth_group_broad)

demo["age_years"] = demo["ADMITTIME"].dt.year - demo["DOB"].dt.year
demo["age_years"] = np.where(demo["age_years"] > 89, 91.4, demo["age_years"])
demo["LOS_hours"] = (demo["DISCHTIME"] - demo["ADMITTIME"]).dt.total_seconds() / 3600.0

demo = demo[["HADM_ID", "SUBJECT_ID", "LANGUAGE", "language_group3",
             "ETHNICITY", "ethnicity_broad", "GENDER", "age_years",
             "ADMISSION_TYPE", "LOS_hours"]].drop_duplicates("HADM_ID")
log("Demographics table built:", demo.shape)

# ---------------------------------------------------------------------------
# Per-cohort file locations
# ---------------------------------------------------------------------------
COHORT_FILES = {
    "hf": dict(
        note_pairs="/Users/rickygole/mimic_local_only/accord/hf_note_pairs.csv.gz",
        candidates="/Users/rickygole/mimic_local_only/accord/contradiction_candidates.csv.gz",
        target_n=9783,
    ),
    "sepsis": dict(
        note_pairs="/Users/rickygole/mimic_local_only/accord_sepsis/sepsis_note_pairs.csv.gz",
        candidates="/Users/rickygole/mimic_local_only/accord_sepsis/sepsis_contradiction_candidates.csv.gz",
        target_n=4339,
    ),
    "aki": dict(
        note_pairs="/Users/rickygole/mimic_local_only/accord_aki/aki_note_pairs.csv.gz",
        candidates="/Users/rickygole/mimic_local_only/accord_aki/aki_contradiction_candidates.csv.gz",
        target_n=7579,
    ),
}


def build_master(cohort: str) -> pd.DataFrame:
    """Same as experiment_A.py's build_master, EXCEPT note length is split
    by FINAL_ROLE (nursing / discharge) via pivot instead of summed, so
    total/nursing/discharge char counts are all available separately."""
    files = COHORT_FILES[cohort]
    rows = []
    with gzip.open(files["note_pairs"], "rt") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {c: i for i, c in enumerate(header)}
        for row in r:
            rows.append((float(row[idx["HADM_ID"]]), row[idx["FINAL_ROLE"]], len(row[idx["TEXT"]])))
    note_len_df = pd.DataFrame(rows, columns=["HADM_ID", "FINAL_ROLE", "n_chars"])
    del rows

    # Every admission in these cohort files has exactly one nursing and one
    # discharge row (verified: role-count distribution is (1,1) for all
    # 9,783 / 4,339 / 7,579 admissions in hf / sepsis / aki respectively).
    pivot = note_len_df.pivot_table(index="HADM_ID", columns="FINAL_ROLE",
                                     values="n_chars", aggfunc="sum").reset_index()
    pivot = pivot.rename(columns={"nursing": "nursing_chars", "discharge": "discharge_chars"})
    for col in ["nursing_chars", "discharge_chars"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["nursing_chars"] = pivot["nursing_chars"].fillna(0)
    pivot["discharge_chars"] = pivot["discharge_chars"].fillna(0)
    pivot["total_note_chars"] = pivot["nursing_chars"] + pivot["discharge_chars"]

    cc = pd.read_csv(files["candidates"])
    cc["HADM_ID"] = cc["HADM_ID"].astype(float)
    cand_counts = cc.groupby("HADM_ID").size().rename("n_candidates").reset_index()

    master = pivot.merge(demo, on="HADM_ID", how="left")
    master = master.merge(cand_counts, on="HADM_ID", how="left")
    master["n_candidates"] = master["n_candidates"].fillna(0).astype(int)
    master["has_candidate"] = (master["n_candidates"] > 0).astype(int)

    n_before_demo = master.shape[0]
    n_missing_demo = master["language_group3"].isna().sum()
    master = master.dropna(subset=["language_group3"])
    log(f"[{cohort}] note-pair cohort n={n_before_demo} (target {files['target_n']}), "
        f"missing demo join={n_missing_demo}, final master n={master.shape[0]}")
    return master


def or_ci_from_model(model, keys):
    p, ci = model.params, model.conf_int()
    out = {}
    for k in keys:
        out[k] = dict(OR=round(float(np.exp(p[k])), 4),
                       CI_low=round(float(np.exp(ci.loc[k, 0])), 4),
                       CI_high=round(float(np.exp(ci.loc[k, 1])), 4),
                       p=round(float(model.pvalues[k]), 6))
    return out


def fit_cluster(formula, data, groups):
    return smf.logit(formula, data=data).fit(disp=0, cov_type="cluster",
                                              cov_kwds={"groups": groups})


def spec_curve_row(name, formula, data, groups, lang_keys):
    m_robust = fit_cluster(formula, data, groups)
    m_naive = smf.logit(formula, data=data).fit(disp=0)
    robust = or_ci_from_model(m_robust, lang_keys)
    naive = or_ci_from_model(m_naive, lang_keys)
    return dict(spec=name, robust=robust, naive=naive), m_robust, m_naive


def run_cohort(cohort: str) -> dict:
    log("\n" + "=" * 78)
    log(f"COHORT: {cohort.upper()}")
    log("=" * 78)
    master = build_master(cohort)

    m2 = master.copy()
    m2["language_group3"] = pd.Categorical(m2["language_group3"], categories=["ENGLISH", "RECORDED_NON_ENGLISH", "MISSING"])
    m2["ethnicity_broad"] = pd.Categorical(m2["ethnicity_broad"])
    m2["log_total"] = np.log1p(m2["total_note_chars"])
    m2["log_nursing"] = np.log1p(m2["nursing_chars"])
    m2["log_discharge"] = np.log1p(m2["discharge_chars"])
    m2 = m2.dropna(subset=["LOS_hours", "age_years", "GENDER", "ADMISSION_TYPE", "SUBJECT_ID"])
    log(f"[{cohort}] Regression n = {m2.shape[0]}")

    # ---- Descriptive: nursing vs discharge length relationship ------------
    med_n, med_d = m2["nursing_chars"].median(), m2["discharge_chars"].median()
    pct_disch_of_total = float((m2["discharge_chars"] / m2["total_note_chars"].replace(0, np.nan)).median())
    r_log = float(np.corrcoef(m2["log_nursing"], m2["log_discharge"])[0, 1])
    log(f"[{cohort}] median nursing_chars={med_n:.0f}, median discharge_chars={med_d:.0f}, "
        f"median discharge share of total={pct_disch_of_total:.3f}, "
        f"corr(log_nursing, log_discharge)={r_log:.4f}")

    lang_keys = [
        "C(language_group3, Treatment('ENGLISH'))[T.RECORDED_NON_ENGLISH]",
        "C(language_group3, Treatment('ENGLISH'))[T.MISSING]",
    ]
    # NOTE on LOS_hours placement: this mirrors experiment_A.py / the original
    # accord/differential/run_experiments.py Experiment 2a spec exactly --
    # LOS_hours enters the model bundled WITH the note-volume adjustment
    # (model_base has no LOS_hours; model_full = model_base + log_note_chars
    # + LOS_hours). We reproduce that bundling here so the "none" row is the
    # literal original before_volume_adj spec and "total" is the literal
    # original after_volume_adj spec -- the two reference points the task
    # brief's numbers were computed from. Rows 3-5 vary only the note-length
    # covariate(s) against that same base+LOS bundle.
    base_terms = ("C(language_group3, Treatment('ENGLISH')) + C(ADMISSION_TYPE) "
                  "+ age_years + C(GENDER) + C(ethnicity_broad)")

    specs = {
        "none":               "has_candidate ~ " + base_terms,
        "total":              "has_candidate ~ " + base_terms + " + LOS_hours + log_total",
        "nursing_only":       "has_candidate ~ " + base_terms + " + LOS_hours + log_nursing",
        "discharge_only":     "has_candidate ~ " + base_terms + " + LOS_hours + log_discharge",
        "nursing_plus_discharge": "has_candidate ~ " + base_terms + " + LOS_hours + log_nursing + log_discharge",
    }
    groups = m2["SUBJECT_ID"]

    spec_curve = {}
    models_robust = {}
    for name, formula in specs.items():
        row, m_robust, m_naive = spec_curve_row(name, formula, m2, groups, lang_keys)
        spec_curve[name] = row
        models_robust[name] = m_robust
        missing_r = row["robust"][lang_keys[1]]
        missing_n = row["naive"][lang_keys[1]]
        log(f"[{cohort}] spec={name:24s} MISSING OR robust={missing_r['OR']:.4f} "
            f"[{missing_r['CI_low']:.4f},{missing_r['CI_high']:.4f}] p={missing_r['p']:.3g}  "
            f"| naive OR={missing_n['OR']:.4f} [{missing_n['CI_low']:.4f},{missing_n['CI_high']:.4f}] p={missing_n['p']:.3g}")

    # ---- SE ratio (robust/naive) as the cluster-effect check --------------
    se_ratios = {}
    for name in specs:
        mr, mn = models_robust[name], smf.logit(specs[name], data=m2).fit(disp=0)
        k = lang_keys[1]
        se_ratios[name] = round(float(mr.bse[k] / mn.bse[k]), 4)
    log(f"[{cohort}] robust/naive SE ratio on MISSING coef, by spec: {se_ratios}")

    # ---- SUBJECT_ID clustering structure (does clustering even bind?) -----
    dup_subjects = m2["SUBJECT_ID"].value_counts()
    n_subjects_multi = int((dup_subjects > 1).sum())
    n_admissions_multi = int(dup_subjects[dup_subjects > 1].sum())
    log(f"[{cohort}] SUBJECT_ID structure: {m2['SUBJECT_ID'].nunique()} unique subjects for "
        f"{len(m2)} admissions; {n_subjects_multi} subjects contribute >1 admission "
        f"({n_admissions_multi} admissions total in those subjects).")

    # -------------------------------------------------------------------
    # (b) Stratified analysis on NURSING-note deciles, Breslow-Day test
    # -------------------------------------------------------------------
    m2["nursing_decile"] = pd.qcut(m2["nursing_chars"], 10, labels=False, duplicates="drop")
    eng_miss = m2[m2["language_group3"].isin(["ENGLISH", "MISSING"])].copy()
    eng_miss["language_group3"] = eng_miss["language_group3"].astype(str)

    def build_stratified_tables(df, decile_col):
        # Row order MISSING-then-ENGLISH so the resulting odds ratio is
        # odds(MISSING)/odds(ENGLISH) -- the SAME convention as the
        # regression's Treatment('ENGLISH') language coefficients elsewhere
        # in this script (OR<1 means MISSING has lower candidate odds, i.e.
        # a deficit). Do not swap without also swapping every OR reported
        # below (per-stratum, pooled CMH, cross-check).
        tables = []
        strata_used = []
        for dec, sub in df.groupby(decile_col):
            ct2 = pd.crosstab(sub["language_group3"], sub["has_candidate"])
            ct2 = ct2.reindex(index=["ENGLISH", "MISSING"], columns=[0, 1]).fillna(0)
            tab = np.array([[ct2.loc["MISSING", 1], ct2.loc["MISSING", 0]],
                             [ct2.loc["ENGLISH", 1], ct2.loc["ENGLISH", 0]]])
            if (tab.sum(axis=1) > 0).all():
                tables.append(tab)
                strata_used.append(int(dec))
        return tables, strata_used

    nursing_tables, nursing_strata = build_stratified_tables(eng_miss, "nursing_decile")
    st_nursing = ct.StratifiedTable(nursing_tables)
    bd_nursing = st_nursing.test_equal_odds()
    cmh_nursing = st_nursing.test_null_odds()
    or_pooled_nursing = st_nursing.oddsratio_pooled
    or_pooled_nursing_ci = st_nursing.oddsratio_pooled_confint()
    per_stratum_or_nursing = []
    for i, tab in enumerate(nursing_tables):
        a, b = tab[0]
        c_, d = tab[1]
        or_i = (a / b) / (c_ / d) if b > 0 and c_ > 0 and d > 0 else float("nan")
        per_stratum_or_nursing.append(dict(decile=nursing_strata[i], OR=None if pd.isna(or_i) else round(float(or_i), 4),
                                            n=int(tab.sum())))
    log(f"[{cohort}] NURSING-decile strata: {len(nursing_tables)} usable strata")
    log(f"[{cohort}] NURSING-decile Breslow-Day (homogeneity): stat={bd_nursing.statistic:.4f} p={bd_nursing.pvalue:.4g}")
    log(f"[{cohort}] NURSING-decile CMH pooled OR={or_pooled_nursing:.4f} "
        f"[{or_pooled_nursing_ci[0]:.4f},{or_pooled_nursing_ci[1]:.4f}] "
        f"CMH test stat={cmh_nursing.statistic:.4f} p={cmh_nursing.pvalue:.4g}")
    log(f"[{cohort}] NURSING-decile per-stratum crude ORs: {per_stratum_or_nursing}")
    nursing_homogeneity_ok = bool(bd_nursing.pvalue >= 0.05)
    log(f"[{cohort}] NURSING-decile: homogeneity {'NOT rejected -> pooled OR is defensible' if nursing_homogeneity_ok else 'REJECTED -> pooled OR is NOT valid, report per-stratum only'}")

    # Same exercise on TOTAL-chars deciles, kept as a cross-check against
    # the reasoning in the task brief (HF reference: Breslow-Day
    # chi2=50.19, p=9.9e-8 on TOTAL deciles).
    m2["total_decile"] = pd.qcut(m2["total_note_chars"], 10, labels=False, duplicates="drop")
    eng_miss_total = m2[m2["language_group3"].isin(["ENGLISH", "MISSING"])].copy()
    eng_miss_total["language_group3"] = eng_miss_total["language_group3"].astype(str)
    total_tables, total_strata = build_stratified_tables(eng_miss_total, "total_decile")
    st_total = ct.StratifiedTable(total_tables)
    bd_total = st_total.test_equal_odds()
    cmh_total = st_total.test_null_odds()
    or_pooled_total = st_total.oddsratio_pooled
    log(f"[{cohort}] [cross-check] TOTAL-decile Breslow-Day: stat={bd_total.statistic:.4f} p={bd_total.pvalue:.4g}; "
        f"CMH pooled OR={or_pooled_total:.4f} p={cmh_total.pvalue:.4g}")

    # -------------------------------------------------------------------
    # (c) language x log(nursing_chars) interaction
    # -------------------------------------------------------------------
    inter_formula = specs["nursing_plus_discharge"] + " + C(language_group3, Treatment('ENGLISH')):log_nursing"
    m_inter = fit_cluster(inter_formula, m2, groups)
    inter_terms = [k for k in m_inter.params.index if ":log_nursing" in k]
    inter_results = {}
    for k in inter_terms:
        inter_results[k] = dict(
            coef=round(float(m_inter.params[k]), 4),
            OR_per_unit_log_nursing_chars=round(float(np.exp(m_inter.params[k])), 4),
            se_robust=round(float(m_inter.bse[k]), 4),
            p=round(float(m_inter.pvalues[k]), 6),
        )
    log(f"[{cohort}] language x log(nursing_chars) interaction (cluster-robust SE): {inter_results}")

    result = {
        "cohort": cohort,
        "n_note_pair_cohort": int(master.shape[0]),
        "n_regression": int(m2.shape[0]),
        "median_nursing_chars": float(med_n),
        "median_discharge_chars": float(med_d),
        "median_discharge_share_of_total": pct_disch_of_total,
        "corr_log_nursing_log_discharge": r_log,
        "specification_curve": spec_curve,
        "robust_over_naive_SE_ratio_MISSING_coef": se_ratios,
        "subject_clustering": dict(n_unique_subjects=int(m2["SUBJECT_ID"].nunique()),
                                    n_admissions=int(len(m2)),
                                    n_subjects_with_gt1_admission=n_subjects_multi,
                                    n_admissions_in_those_subjects=n_admissions_multi),
        "nursing_decile_stratified": {
            "n_strata": len(nursing_tables),
            "breslow_day_statistic": float(bd_nursing.statistic),
            "breslow_day_p": float(bd_nursing.pvalue),
            "homogeneity_not_rejected": nursing_homogeneity_ok,
            "cmh_pooled_OR": float(or_pooled_nursing),
            "cmh_pooled_OR_CI": [float(or_pooled_nursing_ci[0]), float(or_pooled_nursing_ci[1])],
            "cmh_test_statistic": float(cmh_nursing.statistic),
            "cmh_p": float(cmh_nursing.pvalue),
            "per_stratum_crude_OR": per_stratum_or_nursing,
            "pooled_OR_valid": nursing_homogeneity_ok,
        },
        "total_decile_stratified_crosscheck": {
            "n_strata": len(total_tables),
            "breslow_day_statistic": float(bd_total.statistic),
            "breslow_day_p": float(bd_total.pvalue),
            "cmh_pooled_OR": float(or_pooled_total),
            "cmh_p": float(cmh_total.pvalue),
        },
        "language_x_lognursingchars_interaction": inter_results,
    }
    return result


all_results = {}
for cohort in ["hf", "sepsis", "aki"]:
    all_results[cohort] = run_cohort(cohort)

log("\n" + "=" * 78)
log("DBSOURCE / CareVue-MetaVision ERA CONFOUND")
log("=" * 78)
if DBSOURCE_AVAILABLE:
    log(f"ICUSTAYS.csv WAS found this time at: {icustays_hits}. "
        "NOTE: the adjustment-set-with-DBSOURCE analysis is NOT implemented in "
        "this script; re-run must be extended before this claim can be closed.")
else:
    log("ICUSTAYS.csv (the only MIMIC-III table carrying DBSOURCE) was searched again "
        "under /Users/rickygole/Downloads/ and /Users/rickygole/mimic_local_only/ and is "
        "still NOT present on this machine. Per the data rules (local only, no new "
        "credentialed downloads), it was NOT fetched. DBSOURCE was NOT added to the "
        "adjustment set in any of the three cohorts. This remains an OPEN, unresolved "
        "confound: if LANGUAGE recording practice and/or note format shifted at the "
        "CareVue/MetaVision boundary, an unmodeled era effect remains a live "
        "alternative explanation for any surviving missing-language OR below.")

with open(os.path.join(OUTDIR, "experimentA_corrected_replication.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)
with open(os.path.join(OUTDIR, "experimentA_corrected_run_log.txt"), "w") as f:
    f.write("\n".join(log_lines))

print("EXPERIMENT_A_CORRECTED_DONE")
