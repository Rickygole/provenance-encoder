"""Generality Experiment B, REVISED per coordinator correction.

The OR 1.69 [1.11, 2.57] p=0.015 "missing-language -> Llama over-calls
positive" finding (accord_differential/exp1c) was produced by conditioning
on C(rule_cat) -- a deterministic OUTPUT of Stage-1 mining -- inside a
sample already conditioned on has_candidate==1. That is collider-stratification
bias, not an effect of missing language on the labeller. Unadjusted, language
does NOT predict a higher positive-call rate (crude OR ~0.82, direction
reversed). This script does NOT re-attempt mediation on that effect. Instead:

  b1) Marginal model: weak_pos ~ language_group3 + log_note_chars + n_mentions_row
      WITHOUT C(rule_cat), with cluster-robust SEs on HADM_ID (rows share an
      admission when >1 candidate came from the same admission). Confirms
      whether the language effect is null once the collider is removed.
  b2) rule x language_group3 crosstab -- makes the mediator (rule composition)
      visible. Quantifies how much more of MISSING-language admissions' weak-
      labelled candidates are Rule 4 (omission, OR 0.045 for positive-call in
      the collider-biased model) vs ENGLISH admissions'.
  b3) Evidence-length vs positive-rate, MARGINAL (no demographic stratification).
      Deciles of (nursing_sentence + discharge_sentence) character and token
      count, positive rate per decile, and an explicit trend test.

Local only. Aggregates only -- no note text printed or written. Output:
/Users/rickygole/mimic_local_only/accord_generality/experimentB_*.json
Seed: 42 (not needed for anything stochastic here, but recorded for consistency).
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

SEED = 42
OUTDIR = "/Users/rickygole/mimic_local_only/accord_generality"
os.makedirs(OUTDIR, exist_ok=True)

log_lines = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log_lines.append(s)

# ---------------------------------------------------------------------------
# Demographics table (identical grouping logic to accord/differential/run_experiments.py)
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
demo = demo[["HADM_ID", "SUBJECT_ID", "language_group3"]].drop_duplicates("HADM_ID")

# ---------------------------------------------------------------------------
# HF admission-level note-volume covariates (already built; reused verbatim)
# ---------------------------------------------------------------------------
import gzip, csv
rows = []
with gzip.open("/Users/rickygole/mimic_local_only/accord/hf_note_pairs.csv.gz", "rt") as f:
    r = csv.reader(f)
    header = next(r)
    idx = {c: i for i, c in enumerate(header)}
    for row in r:
        rows.append((float(row[idx["HADM_ID"]]), len(row[idx["TEXT"]])))
note_len_df = pd.DataFrame(rows, columns=["HADM_ID", "n_chars"])
note_agg = note_len_df.groupby("HADM_ID").agg(total_note_chars=("n_chars", "sum")).reset_index()
del rows, note_len_df

from collections import Counter
c = Counter()
with gzip.open("/Users/rickygole/mimic_local_only/accord/hf_medication_mentions.csv.gz", "rt") as f:
    r = csv.reader(f)
    header = next(r)
    idx = {c2: i for i, c2 in enumerate(header)}
    for row in r:
        c[float(row[idx["HADM_ID"]])] += 1
n_mentions_df = pd.DataFrame(list(c.items()), columns=["HADM_ID", "n_mentions"])

# ---------------------------------------------------------------------------
# Weak-supervision labels (n=1,679)
# ---------------------------------------------------------------------------
ws = pd.read_csv("/Users/rickygole/mimic_local_only/accord_supp/accord_weak_supervision_labels_decontam.csv")
ws["HADM_ID"] = ws["HADM_ID"].astype(float)
ws["weak_pos"] = (ws["weak_label"] == "contradiction").astype(int)
ws = ws.merge(demo, on="HADM_ID", how="left")
ws = ws.merge(note_agg, on="HADM_ID", how="left")
ws = ws.merge(n_mentions_df, on="HADM_ID", how="left")
ws["n_mentions"] = ws["n_mentions"].fillna(0)
ws = ws.dropna(subset=["language_group3"])
log(f"Weak-supervision rows: {len(ws)} (demo-matched)")

ws["log_note_chars"] = np.log1p(ws["total_note_chars"])
ws["rule_cat"] = "rule" + ws["rule"].astype(str)
ws["language_group3"] = pd.Categorical(ws["language_group3"], categories=["ENGLISH", "RECORDED_NON_ENGLISH", "MISSING"])

# ===========================================================================
# b0: crude (unadjusted) positive-call rate by language group -- sanity check
# against coordinator's reported numbers (English 0.8618 n=1064, Missing 0.8362 n=464)
# ===========================================================================
log("\n" + "=" * 70)
log("b0: CRUDE positive-call rate by language group (no adjustment)")
log("=" * 70)
crude = ws.groupby("language_group3")["weak_pos"].agg(["mean", "count"])
log(crude.round(4).to_string())
ct = pd.crosstab(ws["language_group3"], ws["weak_pos"])
chi2, p_chi2, dof, _ = stats.chi2_contingency(ct)
log(f"chi2={chi2:.3f}, dof={dof}, p={p_chi2:.4f}")
# crude OR, English vs Missing only (2x2)
sub = ws[ws["language_group3"].isin(["ENGLISH", "MISSING"])]
ct2 = pd.crosstab(sub["language_group3"], sub["weak_pos"]).reindex(index=["ENGLISH", "MISSING"])
a, b = ct2.loc["ENGLISH", 1], ct2.loc["ENGLISH", 0]
cc_, d = ct2.loc["MISSING", 1], ct2.loc["MISSING", 0]
crude_or = (a / b) / (cc_ / d)
log(f"Crude OR (English vs Missing, positive-call): {crude_or:.4f}")

b0_results = {
    "rates": crude.round(4).to_dict(),
    "chi2": {"chi2": float(chi2), "dof": int(dof), "p": float(p_chi2)},
    "crude_or_english_vs_missing": float(crude_or),
}

# ===========================================================================
# b1: MARGINAL model -- drop C(rule_cat), cluster-robust SE on HADM_ID
# ===========================================================================
log("\n" + "=" * 70)
log("b1: MARGINAL logistic model (no rule_cat), cluster-robust SE on HADM_ID")
log("=" * 70)

model_marginal = smf.logit(
    "weak_pos ~ C(language_group3, Treatment('ENGLISH')) + log_note_chars + n_mentions",
    data=ws
).fit(disp=0, cov_type="cluster", cov_kwds={"groups": ws["HADM_ID"]})
log(model_marginal.summary().as_text())

or_ci = np.exp(model_marginal.conf_int())
or_ci["OR"] = np.exp(model_marginal.params)
or_ci.columns = ["CI_low", "CI_high", "OR"]
or_ci["p"] = model_marginal.pvalues
log("\nOdds ratios (marginal, cluster-robust SE), 95% CI:")
log(or_ci.round(4).to_string())

# Also fit the ORIGINAL (rule-conditioned, collider-biased) model on this
# same demo-matched subsample for direct side-by-side comparison.
model_conditioned = smf.logit(
    "weak_pos ~ C(language_group3, Treatment('ENGLISH')) + log_note_chars + n_mentions + C(rule_cat)",
    data=ws
).fit(disp=0)
or_ci_cond = np.exp(model_conditioned.conf_int())
or_ci_cond["OR"] = np.exp(model_conditioned.params)
or_ci_cond.columns = ["CI_low", "CI_high", "OR"]
or_ci_cond["p"] = model_conditioned.pvalues
log("\nFor comparison -- rule-CONDITIONED model (the collider-biased one, exp1c) on same rows:")
log(or_ci_cond.round(4).to_string())

b1_results = {
    "n": int(len(ws)),
    "marginal_no_rule_cluster_robust": or_ci.round(4).to_dict(orient="index"),
    "conditioned_on_rule_naive_se": or_ci_cond.round(4).to_dict(orient="index"),
}

# ===========================================================================
# b2: rule x language_group3 crosstab -- the mediator made visible
# ===========================================================================
log("\n" + "=" * 70)
log("b2: rule x language_group3 crosstab (counts and row-%, and positive rate per cell)")
log("=" * 70)

rule_lang_counts = pd.crosstab(ws["language_group3"], ws["rule"])
rule_lang_rowpct = pd.crosstab(ws["language_group3"], ws["rule"], normalize="index").round(4)
log("\nCounts:")
log(rule_lang_counts.to_string())
log("\nRow-normalized (rule composition WITHIN each language group):")
log(rule_lang_rowpct.to_string())

pos_rate_cell = ws.groupby(["language_group3", "rule"])["weak_pos"].agg(["mean", "count"]).unstack("rule")
log("\nPositive-call rate per (language_group3 x rule) cell:")
log(pos_rate_cell.round(4).to_string())
pos_rate_cell_flat = pos_rate_cell.copy()
pos_rate_cell_flat.columns = ["_".join(str(x) for x in col) for col in pos_rate_cell_flat.columns]

# chi2 test: is rule distribution independent of language group?
chi2_rl, p_rl, dof_rl, _ = stats.chi2_contingency(rule_lang_counts)
log(f"\nchi2 test, language_group3 x rule independence: chi2={chi2_rl:.3f}, dof={dof_rl}, p={p_rl:.2e}")

rule4_share = rule_lang_rowpct[4] if 4 in rule_lang_rowpct.columns else None
log(f"\nShare of each language group's weak-labelled candidates that are Rule 4 (omission):")
log(str(rule4_share))

b2_results = {
    "counts": rule_lang_counts.to_dict(),
    "row_pct_rule_composition_by_language": rule_lang_rowpct.to_dict(),
    "positive_rate_by_cell": pos_rate_cell_flat.round(4).reset_index().to_dict(orient="records"),
    "chi2_independence": {"chi2": float(chi2_rl), "dof": int(dof_rl), "p": float(p_rl)},
}

# ===========================================================================
# b3: evidence length vs positive rate, MARGINAL (no demographic stratification)
# ===========================================================================
log("\n" + "=" * 70)
log("b3: Llama positive rate vs evidence length (chars, tokens), MARGINAL, all 1,679 rows")
log("=" * 70)

# Evidence shown to the labeller = nursing_sentence + discharge_sentence.
# Character counts and whitespace-token counts only -- text itself never
# extracted into a variable beyond len()/split() here.
ns = ws["nursing_sentence"].fillna("").astype(str)
ds = ws["discharge_sentence"].fillna("").astype(str)
ws["evidence_chars"] = ns.str.len() + ds.str.len()
ws["evidence_tokens"] = ns.str.split().apply(len) + ds.str.split().apply(len)

ws["chars_decile"] = pd.qcut(ws["evidence_chars"], 10, labels=False, duplicates="drop")
decile_tbl = ws.groupby("chars_decile")["weak_pos"].agg(["mean", "count"])
decile_tbl["median_evidence_chars"] = ws.groupby("chars_decile")["evidence_chars"].median()
decile_tbl["median_evidence_tokens"] = ws.groupby("chars_decile")["evidence_tokens"].median()
log("\nPositive-call rate by evidence-length decile (all rows, no demographic split):")
log(decile_tbl.round(4).to_string())

# Trend tests: (i) Spearman rank correlation length vs weak_pos
rho, p_rho = stats.spearmanr(ws["evidence_chars"], ws["weak_pos"])
log(f"\nSpearman rank correlation (evidence_chars vs weak_pos): rho={rho:.4f}, p={p_rho:.2e}")

rho_tok, p_rho_tok = stats.spearmanr(ws["evidence_tokens"], ws["weak_pos"])
log(f"Spearman rank correlation (evidence_tokens vs weak_pos): rho={rho_tok:.4f}, p={p_rho_tok:.2e}")

# (ii) Cochran-Armitage trend test across deciles (ordered categories)
def cochran_armitage(decile_series, outcome_series):
    d = decile_series.values
    y = outcome_series.values
    scores = np.unique(d)
    n_i = np.array([np.sum(d == s) for s in scores])
    r_i = np.array([np.sum(y[d == s]) for s in scores])
    N = n_i.sum()
    p_bar = r_i.sum() / N
    num = np.sum(n_i * (scores - np.average(scores, weights=n_i)) * (r_i / n_i - p_bar))
    denom = p_bar * (1 - p_bar) * np.sum(n_i * (scores - np.average(scores, weights=n_i)) ** 2)
    z = num / np.sqrt(denom)
    p_two = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_two

z_ca, p_ca = cochran_armitage(ws["chars_decile"], ws["weak_pos"])
log(f"\nCochran-Armitage trend test across evidence-length deciles: z={z_ca:.3f}, p={p_ca:.2e}")

# (iii) simple logistic trend model on continuous log(evidence_chars+1)
ws["log_evidence_chars"] = np.log1p(ws["evidence_chars"])
trend_model = smf.logit("weak_pos ~ log_evidence_chars", data=ws).fit(disp=0)
log("\nLogistic trend model, weak_pos ~ log(evidence_chars+1):")
log(trend_model.summary().as_text())
trend_or = float(np.exp(trend_model.params["log_evidence_chars"]))
trend_p = float(trend_model.pvalues["log_evidence_chars"])
log(f"OR per unit log(evidence_chars+1): {trend_or:.4f}, p={trend_p:.4f}")

monotonic = decile_tbl["mean"].is_monotonic_increasing or decile_tbl["mean"].is_monotonic_decreasing
log(f"\nDecile positive-rate sequence strictly monotonic: {monotonic}")
log(str(decile_tbl["mean"].round(4).tolist()))

b3_results = {
    "n": int(len(ws)),
    "decile_table": decile_tbl.round(4).reset_index().to_dict(orient="records"),
    "spearman_chars": {"rho": float(rho), "p": float(p_rho)},
    "spearman_tokens": {"rho": float(rho_tok), "p": float(p_rho_tok)},
    "cochran_armitage_trend": {"z": float(z_ca), "p": float(p_ca)},
    "logistic_trend_OR_per_log_evidence_chars": trend_or,
    "logistic_trend_p": trend_p,
    "decile_sequence_strictly_monotonic": bool(monotonic),
}

# ---------------------------------------------------------------------------
with open(os.path.join(OUTDIR, "experimentB_results.json"), "w") as f:
    json.dump({"seed": SEED, "b0_crude": b0_results, "b1_marginal_vs_conditioned": b1_results,
               "b2_rule_language_crosstab": b2_results, "b3_evidence_length_trend": b3_results},
              f, indent=2, default=str)

with open(os.path.join(OUTDIR, "experimentB_run_log.txt"), "w") as f:
    f.write("\n".join(log_lines))

print("EXPERIMENT_B_DONE")
