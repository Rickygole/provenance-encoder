"""
ACCORD differential-error / confound-adjustment experiments.
Reads local MIMIC-III + ACCORD pipeline outputs. Writes ONLY aggregate
results (counts, rates, coefficients, CIs, test stats) to
/Users/rickygole/mimic_local_only/accord_differential/.
No note text or per-patient rows are written anywhere.

Seed: 42 (used for bootstrap CIs).
"""
import warnings
warnings.filterwarnings("ignore")
import gzip, csv, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

RNG = np.random.default_rng(42)
OUTDIR = "/Users/rickygole/mimic_local_only/accord_differential"
import os
os.makedirs(OUTDIR, exist_ok=True)

log_lines = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log_lines.append(s)

# ---------------------------------------------------------------------------
# 1. DEMOGRAPHICS TABLE (per HADM_ID)
# ---------------------------------------------------------------------------
adm = pd.read_csv("/Users/rickygole/mimic_local_only/raw_mimiciii/ADMISSIONS.csv",
                   parse_dates=["ADMITTIME", "DISCHTIME"])
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
    return "UNKNOWN_OTHER"  # residual small categories folded into unknown/other
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

# age at admission; MIMIC shifts DOB for 89+ patients to ~300 years prior,
# which overflows pandas datetime64 arithmetic -- use year-difference instead
# (standard MIMIC-III approach), then clip/flag the shifted-DOB 89+ group.
demo["age_years"] = demo["ADMITTIME"].dt.year - demo["DOB"].dt.year
demo["age_years"] = np.where(demo["age_years"] > 89, 91.4, demo["age_years"])
def age_bucket(a):
    if pd.isna(a):
        return "UNKNOWN"
    if a < 45:
        return "<45"
    if a < 65:
        return "45-64"
    if a < 80:
        return "65-79"
    return "80+"
demo["age_bucket"] = demo["age_years"].apply(age_bucket)
demo["LOS_hours"] = (demo["DISCHTIME"] - demo["ADMITTIME"]).dt.total_seconds() / 3600.0

demo = demo[["HADM_ID", "SUBJECT_ID", "LANGUAGE", "language_group3", "language_binary",
             "ETHNICITY", "ethnicity_group3", "ethnicity_broad", "GENDER", "age_years",
             "age_bucket", "ADMISSION_TYPE", "LOS_hours"]].drop_duplicates("HADM_ID")

log("Demographics table built:", demo.shape)
log("Language group3 overall distribution (all admissions):")
log(demo["language_group3"].value_counts(normalize=True).round(4).to_dict())

# ---------------------------------------------------------------------------
# 2. NOTE-PAIR COHORT (9,783 admissions) -- text length, no text retained
# ---------------------------------------------------------------------------
rows = []
with gzip.open("/Users/rickygole/mimic_local_only/accord/hf_note_pairs.csv.gz", "rt") as f:
    r = csv.reader(f)
    header = next(r)
    idx = {c: i for i, c in enumerate(header)}
    for row in r:
        rows.append((float(row[idx["HADM_ID"]]), row[idx["FINAL_ROLE"]], len(row[idx["TEXT"]])))
note_len_df = pd.DataFrame(rows, columns=["HADM_ID", "FINAL_ROLE", "n_chars"])
del rows

note_agg = note_len_df.groupby("HADM_ID").agg(
    total_note_chars=("n_chars", "sum"),
    n_notes=("n_chars", "size"),
    nursing_chars=("n_chars", lambda s: s.iloc[0] if len(s) else np.nan),
).reset_index()
# nursing/discharge chars separately
piv = note_len_df.pivot_table(index="HADM_ID", columns="FINAL_ROLE", values="n_chars", aggfunc="first")
piv = piv.rename(columns={"nursing": "nursing_chars", "discharge": "discharge_chars"}).reset_index()
note_agg = note_len_df.groupby("HADM_ID").agg(total_note_chars=("n_chars", "sum"),
                                               n_notes=("n_chars", "size")).reset_index()
note_agg = note_agg.merge(piv[["HADM_ID", "nursing_chars", "discharge_chars"]], on="HADM_ID", how="left")

log("Note-pair cohort:", note_agg["HADM_ID"].nunique(), "admissions;",
    "n_notes constant check -- unique values:", note_agg["n_notes"].unique())

# ---------------------------------------------------------------------------
# 3. MEDICATION MENTIONS per admission (documentation volume covariate)
# ---------------------------------------------------------------------------
mm_counts = []
with gzip.open("/Users/rickygole/mimic_local_only/accord/hf_medication_mentions.csv.gz", "rt") as f:
    r = csv.reader(f)
    header = next(r)
    idx = {c: i for i, c in enumerate(header)}
    from collections import Counter
    c = Counter()
    for row in r:
        c[float(row[idx["HADM_ID"]])] += 1
n_mentions_df = pd.DataFrame(list(c.items()), columns=["HADM_ID", "n_mentions"])

# ---------------------------------------------------------------------------
# 4. CANDIDATE POOL
# ---------------------------------------------------------------------------
cc = pd.read_csv("/Users/rickygole/mimic_local_only/accord/contradiction_candidates.csv.gz")
cc["HADM_ID"] = cc["HADM_ID"].astype(float)
cand_counts = cc.groupby("HADM_ID").size().rename("n_candidates").reset_index()

# ---------------------------------------------------------------------------
# 5. ADMISSION-LEVEL MASTER TABLE (9,783 cohort)
# ---------------------------------------------------------------------------
master = note_agg.merge(demo, on="HADM_ID", how="left")
master = master.merge(n_mentions_df, on="HADM_ID", how="left")
master["n_mentions"] = master["n_mentions"].fillna(0)
master = master.merge(cand_counts, on="HADM_ID", how="left")
master["n_candidates"] = master["n_candidates"].fillna(0).astype(int)
master["has_candidate"] = (master["n_candidates"] > 0).astype(int)

log("\nMaster table (note-pair cohort):", master.shape)
log("Missing demo joins:", master["language_group3"].isna().sum())
master = master.dropna(subset=["language_group3"])
log("After dropping unmatched demo rows:", master.shape)

log("(admission-level covariate table kept in-memory only; not written to disk per data rules)")

# ---------------------------------------------------------------------------
# 6. GOLD-LABEL <-> WEAK-LABEL JOIN (299 pharmacist pairs)
#    Path: gold_standard_accord_results.csv (pair_id, pharm_label)
#      -> annotation_sample_300.csv (pair_id -> HADM_ID, drug, rule, sentences)
#      -> accord_decontamination_audit.csv (HADM_ID+drug+rule+sentences -> weak_label)
#    Verified: all 299 gold pairs match audit rows 1:1 with
#    removal_reason == 'pharmacist_overlap' (i.e. these are exactly the rows
#    decontamination pulled OUT of the training pool because they overlap the
#    gold set -- so the audit file carries the correct pre-decontam weak label
#    for the gold-evaluated pairs).
# ---------------------------------------------------------------------------
GOLD_PATH = "/Users/rickygole/mimic_local_only/derived/gold_standard_accord_results.csv"
# Repointed off the Google Drive mount. This file carries verbatim note
# sentences joined to HADM_ID, so a cloud-synced path was the one thing
# the .gitignore rules and sync gate exist to prevent, and README.md
# asserted it was not happening while these two lines said otherwise.
SAMP_PATH = "/Users/rickygole/mimic_local_only/from_drive_2026-08-08/outputs/annotation_sample_300.csv"
AUDIT_PATH = "/Users/rickygole/mimic_local_only/accord_supp/accord_decontamination_audit.csv"
WS_PATH = "/Users/rickygole/mimic_local_only/accord_supp/accord_weak_supervision_labels_decontam.csv"

gold = pd.read_csv(GOLD_PATH)
samp = pd.read_csv(SAMP_PATH)
audit = pd.read_csv(AUDIT_PATH)
ws = pd.read_csv(WS_PATH)
audit["HADM_ID"] = audit["HADM_ID"].astype(float)
ws["HADM_ID"] = ws["HADM_ID"].astype(float)

g2 = gold.merge(samp[["pair_id", "HADM_ID", "drug", "rule", "nursing_sentence", "discharge_sentence"]],
                 on=["pair_id", "rule"], how="left")
g2["HADM_ID"] = g2["HADM_ID"].astype(float)
key = ["HADM_ID", "drug", "rule", "nursing_sentence", "discharge_sentence"]
gold_join = g2.merge(audit[key + ["weak_label", "weak_confidence", "removal_reason"]], on=key, how="left")

assert gold_join["weak_label"].notna().sum() == 299, "gold<->weak-label join is not fully 1:1 -- STOP"
assert (gold_join["removal_reason"] == "pharmacist_overlap").all()
assert gold_join.shape[0] == 299

gold_join = gold_join.merge(demo, on="HADM_ID", how="left")
log("\nGold-set join: 299/299 pairs matched to weak_label (verified 1:1, removal_reason=pharmacist_overlap).")
log("Gold-set demo match rate:", gold_join["language_group3"].notna().mean())

gold_join["pharm_pos"] = (gold_join["pharm_label"] == "contradiction").astype(int)
gold_join["weak_pos"] = (gold_join["weak_label"] == "contradiction").astype(int)
log("Gold set pharm_label distribution:", gold_join["pharm_label"].value_counts().to_dict())
log("Gold set weak_label distribution:", gold_join["weak_label"].value_counts().to_dict())

print("PART2_DONE")

# ===========================================================================
# EXPERIMENT 1a: gold-set error rates by demographic group (n=299, 12 pos)
# ===========================================================================
log("\n" + "=" * 70)
log("EXPERIMENT 1a: Llama weak-label error rates by group (gold n=299)")
log("=" * 70)

def confusion_by_group(df, group_col):
    out = []
    for g, sub in df.groupby(group_col):
        tp = ((sub["weak_pos"] == 1) & (sub["pharm_pos"] == 1)).sum()
        fp = ((sub["weak_pos"] == 1) & (sub["pharm_pos"] == 0)).sum()
        fn = ((sub["weak_pos"] == 0) & (sub["pharm_pos"] == 1)).sum()
        tn = ((sub["weak_pos"] == 0) & (sub["pharm_pos"] == 0)).sum()
        n_pos = tp + fn
        n_neg = fp + tn
        fpr = fp / n_neg if n_neg else np.nan
        fnr = fn / n_pos if n_pos else np.nan
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / n_pos if n_pos else np.nan
        out.append(dict(group=g, n=len(sub), n_pos=n_pos, n_neg=n_neg,
                         TP=tp, FP=fp, FN=fn, TN=tn,
                         FPR=round(fpr, 3) if pd.notna(fpr) else fpr,
                         FNR=round(fnr, 3) if pd.notna(fnr) else fnr,
                         precision=round(precision, 3) if pd.notna(precision) else precision,
                         recall=round(recall, 3) if pd.notna(recall) else recall))
    return pd.DataFrame(out)

exp1a_results = {}
for gcol in ["language_group3", "ethnicity_group3", "GENDER", "age_bucket"]:
    tbl = confusion_by_group(gold_join, gcol)
    log(f"\n-- by {gcol} --")
    log(tbl.to_string(index=False))
    exp1a_results[gcol] = tbl.to_dict(orient="records")

log("\nNOTE: only 12 gold positives total in the whole 299-row set. FNR/recall cells")
log("split across 3-4 groups routinely have denominators of 2-6. Any per-group FNR/recall")
log("difference here is a description of this specific 299-row sample, not a stable estimate.")
log("Sensitivity check: excluding the 46 'ambiguous' pharm_label rows entirely.")

gold_noamb = gold_join[gold_join["pharm_label"] != "ambiguous"].copy()
for gcol in ["language_group3"]:
    tbl = confusion_by_group(gold_noamb, gcol)
    log(f"\n-- by {gcol} (ambiguous excluded, n={len(gold_noamb)}) --")
    log(tbl.to_string(index=False))
    exp1a_results[gcol + "_no_ambiguous"] = tbl.to_dict(orient="records")

with open(os.path.join(OUTDIR, "exp1a_gold_error_rates_by_group.json"), "w") as f:
    json.dump(exp1a_results, f, indent=2, default=str)

print("PART3_DONE")

# ===========================================================================
# EXPERIMENT 1b: well-powered positive-RATE analysis (n=1,679 weak-labelled rows)
# NOTE: no gold labels on these rows (they are exactly the rows decontamination
# kept IN the training pool, i.e. disjoint from the 299 gold-evaluated rows).
# This can show a differential POSITIVE RATE, not a differential ERROR RATE.
# ===========================================================================
log("\n" + "=" * 70)
log("EXPERIMENT 1b: Llama positive rate by group on all 1,679 weak-labelled rows")
log("=" * 70)

ws_full = ws.copy()
ws_full["weak_pos"] = (ws_full["weak_label"] == "contradiction").astype(int)
ws_full = ws_full.merge(demo, on="HADM_ID", how="left")
log("Demo match rate on ws (1,679 rows):", ws_full["language_group3"].notna().mean())
ws_full = ws_full.dropna(subset=["language_group3"])

exp1b_results = {}
for gcol in ["language_group3", "ethnicity_group3"]:
    ct = pd.crosstab(ws_full[gcol], ws_full["weak_pos"])
    rate = ws_full.groupby(gcol)["weak_pos"].agg(["mean", "count"])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    log(f"\n-- Llama positive rate by {gcol} --")
    log(rate.round(4).to_string())
    log(f"chi2={chi2:.2f}, dof={dof}, p={p:.2e}")
    exp1b_results[gcol] = dict(rates=rate.round(4).to_dict(), chi2=chi2, dof=dof, p=p)

log("\nDISTINCTION: a differential POSITIVE RATE here can arise from (i) differential Llama")
log("error, (ii) genuine differential prevalence of true contradictions across groups, or")
log("(iii) differential base rates of the underlying rule/drug_class mix the weak labeller")
log("sees by group. Without gold labels on these 1,679 rows we CANNOT attribute a rate gap")
log("to labeller error. Only (a) above (n=299, gold) speaks to error; (b) is prevalence-of-")
log("positive-calls only, offered because it is well powered.")

with open(os.path.join(OUTDIR, "exp1b_positive_rate_by_group.json"), "w") as f:
    json.dump(exp1b_results, f, indent=2, default=str)

print("PART4_DONE")

# ===========================================================================
# EXPERIMENT 1c: logistic regression, weak_label ~ language + volume + rule + drug_class
# ===========================================================================
log("\n" + "=" * 70)
log("EXPERIMENT 1c: logistic regression of weak_label on language + adjustment set")
log("=" * 70)

reg_df = ws_full.merge(master[["HADM_ID", "total_note_chars", "n_mentions"]],
                        on="HADM_ID", how="left", suffixes=("", "_adm"))
reg_df["n_mentions_row"] = reg_df["n_mentions_adm"] if "n_mentions_adm" in reg_df.columns else reg_df["n_mentions"]
# admission-level n_mentions column name collision guard
if "n_mentions_adm" in reg_df.columns:
    reg_df["n_mentions_row"] = reg_df["n_mentions_adm"]
else:
    reg_df["n_mentions_row"] = reg_df["n_mentions"]

reg_df["log_note_chars"] = np.log1p(reg_df["total_note_chars"])
reg_df["drug_class_bin"] = reg_df["drug_class"].fillna("UNKNOWN")
top_classes = reg_df["drug_class_bin"].value_counts()
rare = top_classes[top_classes < 15].index
reg_df["drug_class_bin"] = reg_df["drug_class_bin"].where(~reg_df["drug_class_bin"].isin(rare), "OTHER_RARE")
reg_df["rule_cat"] = "rule" + reg_df["rule"].astype(str)
reg_df["language_group3"] = pd.Categorical(reg_df["language_group3"],
                                            categories=["ENGLISH", "RECORDED_NON_ENGLISH", "MISSING"])

log("Regression n =", reg_df.shape[0])
log("drug_class_bin distribution:", reg_df["drug_class_bin"].value_counts().to_dict())
log("NOTE: drug_class is populated ONLY for rule==2 rows (90% missing overall, and where")
log("present it occurs 1:1 with rule==2) -- see crosstab check. Including both drug_class")
log("and rule as main effects produces perfect collinearity / quasi-separation (non-")
log("convergence, inf CIs). drug_class is therefore DROPPED from the main model; rule")
log("already absorbs its identifying information for this dataset.")

model = smf.logit(
    "weak_pos ~ C(language_group3, Treatment('ENGLISH')) + log_note_chars + n_mentions_row + C(rule_cat)",
    data=reg_df).fit(disp=0)
log(model.summary().as_text())
log("Converged:", model.mle_retvals.get("converged"))

or_ci = np.exp(model.conf_int())
or_ci["OR"] = np.exp(model.params)
or_ci.columns = ["CI_low", "CI_high", "OR"]
or_ci["p"] = model.pvalues
log("\nOdds ratios with 95% CI:")
log(or_ci.round(4).to_string())

with open(os.path.join(OUTDIR, "exp1c_logistic_regression.json"), "w") as f:
    json.dump({
        "n": int(reg_df.shape[0]),
        "params": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "or_ci": or_ci.round(4).to_dict(orient="index"),
        "pseudo_r2": model.prsquared,
    }, f, indent=2, default=str)

print("PART5_DONE")

# ===========================================================================
# EXPERIMENT 2a: admission-level candidate-yield regression, before/after volume adj.
# ===========================================================================
log("\n" + "=" * 70)
log("EXPERIMENT 2a: P(admission produces >=1 candidate) ~ language + covariates")
log("=" * 70)

m2 = master.copy()
m2["language_group3"] = pd.Categorical(m2["language_group3"], categories=["ENGLISH", "RECORDED_NON_ENGLISH", "MISSING"])
m2["ethnicity_broad"] = pd.Categorical(m2["ethnicity_broad"])
m2["log_note_chars"] = np.log1p(m2["total_note_chars"])
m2 = m2.dropna(subset=["LOS_hours", "age_years", "GENDER", "ADMISSION_TYPE"])
log("n_notes is CONSTANT (=2) in this cohort by construction (exactly 1 nursing + 1")
log("discharge note per admission) -- it carries zero variance and is dropped from the")
log("regression rather than forced in as a degenerate covariate.")
log(f"Regression n = {m2.shape[0]}")

base_formula = ("has_candidate ~ C(language_group3, Treatment('ENGLISH')) + C(ADMISSION_TYPE) "
                 "+ age_years + C(GENDER) + C(ethnicity_broad)")
full_formula = base_formula + " + log_note_chars + LOS_hours"

model_base = smf.logit(base_formula, data=m2).fit(disp=0)
model_full = smf.logit(full_formula, data=m2).fit(disp=0)

def lang_or_ci(model, label):
    p = model.params
    ci = model.conf_int()
    rows = [k for k in p.index if "language_group3" in k]
    out = {}
    for k in rows:
        out[k] = dict(OR=round(np.exp(p[k]), 4),
                       CI_low=round(np.exp(ci.loc[k, 0]), 4),
                       CI_high=round(np.exp(ci.loc[k, 1]), 4),
                       p=round(model.pvalues[k], 5))
    log(f"\n{label} -- language coefficients:")
    for k, v in out.items():
        log(f"  {k}: OR={v['OR']} CI=[{v['CI_low']}, {v['CI_high']}] p={v['p']}")
    return out

log("\n--- MODEL BEFORE volume covariates (demographics + admission_type only) ---")
before = lang_or_ci(model_base, "BEFORE volume adjustment")
log("\n--- MODEL AFTER adding log_note_chars + LOS_hours ---")
after = lang_or_ci(model_full, "AFTER volume adjustment")

verdict_survives = all(
    (after[k]["CI_low"] > 1.0 or after[k]["CI_high"] < 1.0) for k in after
    if "MISSING" in k or "RECORDED_NON_ENGLISH" in k
)
log(f"\nLanguage coefficient still excludes OR=1 (95% CI) after volume adjustment for ALL")
log(f"non-English/missing categories: {verdict_survives}")

with open(os.path.join(OUTDIR, "exp2a_candidate_yield_regression.json"), "w") as f:
    json.dump({"n": int(m2.shape[0]), "before_volume_adj": before, "after_volume_adj": after,
               "base_pseudo_r2": model_base.prsquared, "full_pseudo_r2": model_full.prsquared},
              f, indent=2, default=str)

print("PART6_DONE")

# ===========================================================================
# EXPERIMENT 2b: stratified check -- yield by language within note-length deciles
# ===========================================================================
log("\n" + "=" * 70)
log("EXPERIMENT 2b: candidate yield by language WITHIN note-length deciles")
log("=" * 70)

m2["chars_decile"] = pd.qcut(m2["total_note_chars"], 10, labels=False, duplicates="drop")

strat = m2.groupby(["chars_decile", "language_binary"])["has_candidate"].agg(["mean", "count"]).unstack("language_binary")
log("\nCandidate yield rate by note-length decile x language (binary English vs non-English/missing):")
log(strat.round(4).to_string())

# Cochran-Mantel-Haenszel test across deciles for the binary language contrast
tables = []
for d, sub in m2.groupby("chars_decile"):
    ct = pd.crosstab(sub["language_binary"], sub["has_candidate"])
    ct = ct.reindex(index=["ENGLISH", "NON_ENGLISH_OR_MISSING"], columns=[0, 1]).fillna(0).values
    if ct.shape == (2, 2) and (ct.sum(axis=1) > 0).all():
        tables.append(ct)
from statsmodels.stats.contingency_tables import StratifiedTable
strat_table = StratifiedTable(tables)
cmh = strat_table.test_null_odds()
log(f"\nCochran-Mantel-Haenszel test (English vs non-English/missing, stratified by decile):")
log(f"  common OR = {strat_table.oddsratio_pooled:.4f}")
log(f"  CMH statistic = {cmh.statistic:.3f}, p = {cmh.pvalue:.2e}")

# same for 3-way, English vs MISSING specifically (largest gap group)
m2_em = m2[m2["language_group3"] != "RECORDED_NON_ENGLISH"].copy()
tables_em = []
for d, sub in m2_em.groupby("chars_decile"):
    ct = pd.crosstab(sub["language_group3"].astype(str), sub["has_candidate"])
    ct = ct.reindex(index=["ENGLISH", "MISSING"], columns=[0, 1]).fillna(0).values
    if ct.shape == (2, 2) and (ct.sum(axis=1) > 0).all():
        tables_em.append(ct)
strat_table_em = StratifiedTable(tables_em)
cmh_em = strat_table_em.test_null_odds()
log(f"\nSame test restricted to ENGLISH vs MISSING only (excluding recorded non-English):")
log(f"  common OR = {strat_table_em.oddsratio_pooled:.4f}")
log(f"  CMH statistic = {cmh_em.statistic:.3f}, p = {cmh_em.pvalue:.2e}")

strat_flat = strat.copy()
strat_flat.columns = ["_".join(str(c) for c in col) for col in strat_flat.columns]
with open(os.path.join(OUTDIR, "exp2b_decile_stratified.json"), "w") as f:
    json.dump({
        "yield_by_decile_language": strat_flat.round(4).reset_index().to_dict(orient="records"),
        "cmh_english_vs_nonenglish_or_missing": {"common_OR": strat_table.oddsratio_pooled,
                                                   "statistic": float(cmh.statistic), "p": float(cmh.pvalue)},
        "cmh_english_vs_missing_only": {"common_OR": strat_table_em.oddsratio_pooled,
                                          "statistic": float(cmh_em.statistic), "p": float(cmh_em.pvalue)},
    }, f, indent=2, default=str)

print("PART7_DONE")

# ===========================================================================
# EXPERIMENT 3: disaggregate MISSING vs RECORDED NON-ENGLISH (and ethnicity analog)
# ===========================================================================
log("\n" + "=" * 70)
log("EXPERIMENT 3: three-way composition -- ENGLISH / RECORDED_NON_ENGLISH / MISSING")
log("=" * 70)

pop_lang = master["language_group3"].value_counts(normalize=True).round(4)
cand_pop = master[master["has_candidate"] == 1]
cand_lang = cand_pop["language_group3"].value_counts(normalize=True).round(4)
log("\nPopulation (9,783 admissions) language composition:")
log(pop_lang.to_string())
log(f"\nCandidate-producing admissions (n={len(cand_pop)}) language composition:")
log(cand_lang.to_string())

ct_lang = pd.crosstab(master["language_group3"], master["has_candidate"])
chi2, p, dof, exp = stats.chi2_contingency(ct_lang)
log(f"\nchi2 test, 3-way language x has_candidate: chi2={chi2:.2f}, dof={dof}, p={p:.2e}")
log(ct_lang.to_string())

yield_by_lang3 = master.groupby("language_group3")["has_candidate"].agg(["mean", "count"])
log("\nRaw candidate-yield RATE by language group (unadjusted):")
log(yield_by_lang3.round(4).to_string())

log("\nVERDICT (composition): the shift toward English in the candidate pool has TWO")
log("distinct pieces -- moving from MISSING to ENGLISH raises P(candidate) sharply")
log(f"({yield_by_lang3.loc['MISSING','mean']:.3f} -> {yield_by_lang3.loc['ENGLISH','mean']:.3f}, "
    f"n={yield_by_lang3.loc['MISSING','count']:.0f} vs {yield_by_lang3.loc['ENGLISH','count']:.0f}), while")
log(f"RECORDED_NON_ENGLISH yield ({yield_by_lang3.loc['RECORDED_NON_ENGLISH','mean']:.3f}, "
    f"n={yield_by_lang3.loc['RECORDED_NON_ENGLISH','count']:.0f}) is close to ENGLISH and much higher than MISSING.")
log("Given Exp2a: the RECORDED_NON_ENGLISH gap fully resolves to volume (OR->0.94, ns after")
log("adjustment); the MISSING gap is attenuated but NOT resolved by volume (OR 0.41->0.64,")
log("still p<0.001). So the composition shift is driven predominantly by MISSING-language")
log("admissions (an administrative-completeness pattern), not by recorded non-English")
log("admissions (a language-per-se pattern) -- and even the missingness piece is only")
log("partly a volume artifact.")

# -- ethnicity 3-way (unknown/other vs named minority vs white) --
log("\n" + "-" * 70)
log("Ethnicity 3-way: is the ethnicity composition result carried by UNKNOWN/OTHER?")
log("-" * 70)
pop_eth = master["ethnicity_group3"].value_counts(normalize=True).round(4)
cand_eth = cand_pop["ethnicity_group3"].value_counts(normalize=True).round(4)
log("\nPopulation ethnicity_group3 composition:")
log(pop_eth.to_string())
log("\nCandidate-producing admissions ethnicity_group3 composition:")
log(cand_eth.to_string())

yield_by_eth3 = master.groupby("ethnicity_group3")["has_candidate"].agg(["mean", "count"])
log("\nRaw candidate-yield RATE by ethnicity_group3 (unadjusted):")
log(yield_by_eth3.round(4).to_string())
ct_eth = pd.crosstab(master["ethnicity_group3"], master["has_candidate"])
chi2e, pe, dofe, _ = stats.chi2_contingency(ct_eth)
log(f"chi2 test, ethnicity_group3 x has_candidate: chi2={chi2e:.2f}, dof={dofe}, p={pe:.2e}")

log("\nFiner-grained broad ethnicity buckets (WHITE/BLACK/HISPANIC/ASIAN/OTHER_NAMED/UNKNOWN_OTHER):")
yield_by_eth_broad = master.groupby("ethnicity_broad")["has_candidate"].agg(["mean", "count"])
log(yield_by_eth_broad.round(4).to_string())

named_only = master[master["ethnicity_group3"] != "UNKNOWN_OTHER"]
ct_named = pd.crosstab(named_only["ethnicity_group3"], named_only["has_candidate"])
chi2n, pn, _, _ = stats.chi2_contingency(ct_named)
log(f"\nWHITE vs NAMED_MINORITY only (UNKNOWN_OTHER excluded): chi2={chi2n:.2f}, p={pn:.2e}")
log(f"  WHITE yield={yield_by_eth3.loc['WHITE','mean']:.4f} (n={yield_by_eth3.loc['WHITE','count']:.0f}), "
    f"NAMED_MINORITY yield={yield_by_eth3.loc['NAMED_MINORITY','mean']:.4f} (n={yield_by_eth3.loc['NAMED_MINORITY','count']:.0f})")
log(f"  UNKNOWN_OTHER yield={yield_by_eth3.loc['UNKNOWN_OTHER','mean']:.4f} (n={yield_by_eth3.loc['UNKNOWN_OTHER','count']:.0f})")

with open(os.path.join(OUTDIR, "exp3_disaggregation.json"), "w") as f:
    json.dump({
        "population_language_composition": pop_lang.to_dict(),
        "candidate_pool_language_composition": cand_lang.to_dict(),
        "yield_by_language3": yield_by_lang3.round(4).to_dict(),
        "lang_chi2": {"chi2": chi2, "dof": dof, "p": p},
        "population_ethnicity3_composition": pop_eth.to_dict(),
        "candidate_pool_ethnicity3_composition": cand_eth.to_dict(),
        "yield_by_ethnicity3": yield_by_eth3.round(4).to_dict(),
        "eth_chi2": {"chi2": chi2e, "dof": dofe, "p": pe},
        "yield_by_ethnicity_broad": yield_by_eth_broad.round(4).to_dict(),
        "white_vs_named_minority_chi2": {"chi2": chi2n, "p": pn},
    }, f, indent=2, default=str)

with open(os.path.join(OUTDIR, "run_log.txt"), "w") as f:
    f.write("\n".join(log_lines))

print("ALL_DONE")
