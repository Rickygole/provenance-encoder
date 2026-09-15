"""Generality Experiment A -- cross-cohort replication of the Stage-1
selection effect (P(admission yields >=1 candidate) ~ language_group + ...),
run IDENTICALLY across HF (9,783), sepsis (4,339), AKI (7,579).

Regression setup (demographic grouping, master-table construction, formula)
is copied verbatim from accord/differential/run_experiments.py Experiment 2a
-- that code is already validated for HF. This script generalizes it to run
on sepsis/AKI using the cohort files built by generality/build_cohort.py.

TWO CORRECTIONS applied per coordinator review of the first pass:
  1. DBSOURCE (CareVue vs MetaVision era proxy) COULD NOT BE ADDED. The only
     MIMIC-III table carrying DBSOURCE is ICUSTAYS.csv, which is NOT present
     anywhere in the local credentialed file set (checked /Users/rickygole/
     Downloads and the whole filesystem; also flagged as missing in
     accord/CORRECTIONS.md C-entry for the HF-only analysis). Per the
     data rules ("local only"), new credentialed files were not fetched.
     This is reported as an open limitation, not silently dropped.
  2. The decile-stratified check is reported as a language x log(note_chars)
     INTERACTION regression, not a pooled Cochran-Mantel-Haenszel OR --
     Breslow-Day on the HF deciles previously rejected homogeneity
     (chi2=31.11, p=2.8e-4), so a single pooled OR is not defensible.

Local only. Aggregates only -- no note text, no per-patient rows written.
Seed: not used (no resampling in this script).
"""
import gzip, csv, json, os, warnings
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
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
# Demographics table -- IDENTICAL to accord/differential/run_experiments.py
# ---------------------------------------------------------------------------
adm = pd.read_csv("/Users/rickygole/mimic_local_only/raw_mimiciii/ADMISSIONS.csv", parse_dates=["ADMITTIME", "DISCHTIME"])
adm["HADM_ID"] = adm["HADM_ID"].astype(float)
pts = pd.read_csv("/Users/rickygole/mimic_local_only/raw_mimiciii/PATIENTS.csv", parse_dates=["DOB"])
demo = adm.merge(pts[["SUBJECT_ID", "GENDER", "DOB"]], on="SUBJECT_ID", how="left")

def lang_group(x):
    if pd.isna(x):
        return "MISSING"
    if x == "ENGL":
        return "ENGLISH"
    return "RECORDED_NON_ENGLISH"
demo["language_group3"] = demo["LANGUAGE"].apply(lang_group)
demo["language_binary"] = np.where(demo["LANGUAGE"] == "ENGL", "ENGLISH", "NON_ENGLISH_OR_MISSING")

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

def eth_group3(x):
    if x == "WHITE":
        return "WHITE"
    if x in UNKNOWN_OTHER:
        return "UNKNOWN_OTHER"
    if x in NAMED_MINORITY or (isinstance(x, str) and
        (x.startswith("BLACK") or x.startswith("HISPANIC") or x.startswith("ASIAN"))):
        return "NAMED_MINORITY"
    return "UNKNOWN_OTHER"
demo["ethnicity_group3"] = demo["ETHNICITY"].apply(eth_group3)

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

demo = demo[["HADM_ID", "SUBJECT_ID", "LANGUAGE", "language_group3", "language_binary",
             "ETHNICITY", "ethnicity_group3", "ethnicity_broad", "GENDER", "age_years",
             "ADMISSION_TYPE", "LOS_hours"]].drop_duplicates("HADM_ID")

log("Demographics table built:", demo.shape)

# ---------------------------------------------------------------------------
# Per-cohort file locations
# ---------------------------------------------------------------------------
COHORT_FILES = {
    "hf": dict(
        note_pairs="/Users/rickygole/mimic_local_only/accord/hf_note_pairs.csv.gz",
        mentions="/Users/rickygole/mimic_local_only/accord/hf_medication_mentions.csv.gz",
        candidates="/Users/rickygole/mimic_local_only/accord/contradiction_candidates.csv.gz",
        target_n=9783,
    ),
    "sepsis": dict(
        note_pairs="/Users/rickygole/mimic_local_only/accord_sepsis/sepsis_note_pairs.csv.gz",
        mentions="/Users/rickygole/mimic_local_only/accord_sepsis/sepsis_medication_mentions.csv.gz",
        candidates="/Users/rickygole/mimic_local_only/accord_sepsis/sepsis_contradiction_candidates.csv.gz",
        target_n=4339,
    ),
    "aki": dict(
        note_pairs="/Users/rickygole/mimic_local_only/accord_aki/aki_note_pairs.csv.gz",
        mentions="/Users/rickygole/mimic_local_only/accord_aki/aki_medication_mentions.csv.gz",
        candidates="/Users/rickygole/mimic_local_only/accord_aki/aki_contradiction_candidates.csv.gz",
        target_n=7579,
    ),
}


def build_master(cohort: str) -> pd.DataFrame:
    files = COHORT_FILES[cohort]
    rows = []
    with gzip.open(files["note_pairs"], "rt") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {c: i for i, c in enumerate(header)}
        for row in r:
            rows.append((float(row[idx["HADM_ID"]]), len(row[idx["TEXT"]])))
    note_len_df = pd.DataFrame(rows, columns=["HADM_ID", "n_chars"])
    note_agg = note_len_df.groupby("HADM_ID").agg(total_note_chars=("n_chars", "sum"),
                                                    n_notes=("n_chars", "size")).reset_index()
    del rows, note_len_df

    c = Counter()
    with gzip.open(files["mentions"], "rt") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {c2: i for i, c2 in enumerate(header)}
        for row in r:
            c[float(row[idx["HADM_ID"]])] += 1
    n_mentions_df = pd.DataFrame(list(c.items()), columns=["HADM_ID", "n_mentions"])

    cc = pd.read_csv(files["candidates"])
    cc["HADM_ID"] = cc["HADM_ID"].astype(float)
    cand_counts = cc.groupby("HADM_ID").size().rename("n_candidates").reset_index()

    master = note_agg.merge(demo, on="HADM_ID", how="left")
    master = master.merge(n_mentions_df, on="HADM_ID", how="left")
    master["n_mentions"] = master["n_mentions"].fillna(0)
    master = master.merge(cand_counts, on="HADM_ID", how="left")
    master["n_candidates"] = master["n_candidates"].fillna(0).astype(int)
    master["has_candidate"] = (master["n_candidates"] > 0).astype(int)

    n_before_demo = master.shape[0]
    n_missing_demo = master["language_group3"].isna().sum()
    master = master.dropna(subset=["language_group3"])
    log(f"[{cohort}] note-pair cohort n={n_before_demo} (target {files['target_n']}), "
        f"missing demo join={n_missing_demo}, final master n={master.shape[0]}")
    return master


def run_cohort(cohort: str) -> dict:
    log("\n" + "=" * 70)
    log(f"COHORT: {cohort.upper()}")
    log("=" * 70)
    master = build_master(cohort)

    m2 = master.copy()
    m2["language_group3"] = pd.Categorical(m2["language_group3"], categories=["ENGLISH", "RECORDED_NON_ENGLISH", "MISSING"])
    m2["ethnicity_broad"] = pd.Categorical(m2["ethnicity_broad"])
    m2["log_note_chars"] = np.log1p(m2["total_note_chars"])
    m2 = m2.dropna(subset=["LOS_hours", "age_years", "GENDER", "ADMISSION_TYPE"])
    log(f"[{cohort}] Regression n = {m2.shape[0]}")

    base_formula = ("has_candidate ~ C(language_group3, Treatment('ENGLISH')) + C(ADMISSION_TYPE) "
                     "+ age_years + C(GENDER) + C(ethnicity_broad)")
    full_formula = base_formula + " + log_note_chars + LOS_hours"

    model_base = smf.logit(base_formula, data=m2).fit(disp=0)
    model_full = smf.logit(full_formula, data=m2).fit(disp=0)

    def lang_or_ci(model):
        p = model.params
        ci = model.conf_int()
        rows = [k for k in p.index if "language_group3" in k]
        out = {}
        for k in rows:
            out[k] = dict(OR=round(float(np.exp(p[k])), 4),
                           CI_low=round(float(np.exp(ci.loc[k, 0])), 4),
                           CI_high=round(float(np.exp(ci.loc[k, 1])), 4),
                           p=round(float(model.pvalues[k]), 5))
        return out

    before = lang_or_ci(model_base)
    after = lang_or_ci(model_full)
    log(f"[{cohort}] BEFORE volume adj (language ORs): {before}")
    log(f"[{cohort}] AFTER volume adj  (language ORs): {after}")

    # ---- Raw yield rates by language group (unadjusted composition check) ----
    yield_by_lang = m2.groupby("language_group3")["has_candidate"].agg(["mean", "count"])
    log(f"[{cohort}] Raw yield rate by language group:\n{yield_by_lang.round(4).to_string()}")

    # ---- Ethnicity comparison: WHITE vs NAMED_MINORITY, raw + adjusted ----
    m2_eth = m2.copy()
    m2_eth["ethnicity_group3"] = pd.Categorical(m2_eth["ethnicity_group3"], categories=["WHITE", "NAMED_MINORITY", "UNKNOWN_OTHER"])
    yield_by_eth3 = m2_eth.groupby("ethnicity_group3")["has_candidate"].agg(["mean", "count"])
    named_only = m2_eth[m2_eth["ethnicity_group3"] != "UNKNOWN_OTHER"]
    ct_named = pd.crosstab(named_only["ethnicity_group3"], named_only["has_candidate"])
    if ct_named.shape == (2, 2) and (ct_named.values > 0).all():
        chi2n, pn, _, _ = stats.chi2_contingency(ct_named)
    else:
        chi2n, pn = float("nan"), float("nan")
    log(f"[{cohort}] Ethnicity yield: {yield_by_eth3.round(4).to_dict()}, "
        f"WHITE vs NAMED_MINORITY chi2={chi2n:.3f} p={pn:.4f}" if pd.notna(chi2n) else
        f"[{cohort}] Ethnicity yield: {yield_by_eth3.round(4).to_dict()} (chi2 test not computable, sparse cell)")

    # -----------------------------------------------------------------------
    # REPLACED CMH pooled-OR check: explicit language x log(note_chars)
    # interaction (pooled OR is not defensible -- Breslow-Day on the HF
    # deciles rejects stratum homogeneity, chi2=31.11 p=2.8e-4).
    # -----------------------------------------------------------------------
    inter_formula = (full_formula +
                      " + C(language_group3, Treatment('ENGLISH')):log_note_chars")
    model_inter = smf.logit(inter_formula, data=m2).fit(disp=0)
    inter_terms = [k for k in model_inter.params.index if ":log_note_chars" in k]
    inter_results = {}
    for k in inter_terms:
        inter_results[k] = dict(
            coef=round(float(model_inter.params[k]), 4),
            OR_per_unit_log_chars=round(float(np.exp(model_inter.params[k])), 4),
            p=round(float(model_inter.pvalues[k]), 5),
        )
    log(f"[{cohort}] language x log(note_chars) interaction terms: {inter_results}")

    # Simple sign-check companion: split at median log_note_chars, report the
    # language OR separately above/below median (not a pooled CMH OR).
    med = m2["log_note_chars"].median()
    below = m2[m2["log_note_chars"] <= med]
    above = m2[m2["log_note_chars"] > med]
    def crude_or_em(sub):
        sub2 = sub[sub["language_group3"] != "RECORDED_NON_ENGLISH"]
        ct = pd.crosstab(sub2["language_group3"].astype(str), sub2["has_candidate"])
        ct = ct.reindex(index=["ENGLISH", "MISSING"], columns=[0, 1]).fillna(0)
        a, b = ct.loc["ENGLISH", 1], ct.loc["ENGLISH", 0]
        cc_, d = ct.loc["MISSING", 1], ct.loc["MISSING", 0]
        if b == 0 or cc_ == 0:
            return float("nan")
        return (a / b) / (cc_ / d) if d > 0 else float("nan")
    or_below = crude_or_em(below)
    or_above = crude_or_em(above)
    log(f"[{cohort}] Crude English-vs-Missing OR below median note length: {or_below:.4f} "
        f"(n={len(below)}); above median: {or_above:.4f} (n={len(above)})")

    result = {
        "cohort": cohort,
        "n_note_pair_cohort": int(master.shape[0]),
        "n_regression": int(m2.shape[0]),
        "before_volume_adj": before,
        "after_volume_adj": after,
        "raw_yield_by_language": yield_by_lang.round(4).reset_index().to_dict(orient="records"),
        "ethnicity_yield": yield_by_eth3.round(4).reset_index().to_dict(orient="records"),
        "ethnicity_white_vs_named_minority_chi2": {"chi2": None if pd.isna(chi2n) else float(chi2n),
                                                      "p": None if pd.isna(pn) else float(pn)},
        "language_x_lognotechars_interaction": inter_results,
        "crude_OR_english_vs_missing_below_median_length": None if pd.isna(or_below) else float(or_below),
        "crude_OR_english_vs_missing_above_median_length": None if pd.isna(or_above) else float(or_above),
        "n_below_median": int(len(below)),
        "n_above_median": int(len(above)),
    }
    return result


all_results = {}
for cohort in ["hf", "sepsis", "aki"]:
    all_results[cohort] = run_cohort(cohort)

log("\n" + "=" * 70)
log("DBSOURCE LIMITATION (cannot be resolved with local files)")
log("=" * 70)
log("ICUSTAYS.csv (the only MIMIC-III table with DBSOURCE) is not present under "
    "/Users/rickygole/Downloads/ or anywhere else on this machine (filesystem search "
    "returned no matches for ICUSTAYS/CHARTEVENTS/CareVue/MetaVision). This analysis "
    "could NOT add a CareVue-vs-MetaVision era covariate. This is an open confound: "
    "if LANGUAGE recording practice and note format both shifted at the CareVue/"
    "MetaVision boundary, an unmodeled era effect remains a live alternative "
    "explanation for the missing-language OR in every cohort below. This should be "
    "treated as unresolved, not as adjusted-for.")

with open(os.path.join(OUTDIR, "experimentA_replication.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)
with open(os.path.join(OUTDIR, "experimentA_run_log.txt"), "w") as f:
    f.write("\n".join(log_lines))

print("EXPERIMENT_A_DONE")
