"""
ACCORD Build 1 -- multi-LLM replication of the weak-labelling task, run locally
via Ollama, to determine whether the two effects claimed for Llama-3 in the
retracted C10 finding replicate on other models:

  (i)  GROSS MISCALIBRATION -- ~21x inflation of the positive (contradiction)
       rate relative to the 4% pharmacist gold base rate.
  (ii) THE DEMOGRAPHIC DIFFERENTIAL -- admissions with an unrecorded LANGUAGE
       field receive a positive weak label at OR ~1.69 [1.11, 2.57], adjusted
       for note length, mention count and rule.

PROMPT: reused VERBATIM from ~/mimic_local_only/accord_supp/
prompts_existing.csv, config_id == "llama_baseline" -- the system_prompt and
user_prompt template that were archived as the exact prompt used to elicit
the original Llama-3-8B weak labels. temperature=0, max_tokens=50.

MODELS: local Ollama models only (`ollama list`). No OpenAI/Anthropic/paid
API, no network inference. Unparseable generations are recorded as their own
category -- NEVER silently scored as positive (this is the exact bug in C10:
`llm_preds.append(pred if pred else pk)`).

DATA RULES: reads local, credentialed MIMIC-III-derived files. Writes ONLY
aggregate results (counts, rates, coefficients, CIs) to
~/mimic_local_only/accord_multillm/. No note text, and no
per-row raw model output text, is ever written to disk or printed.

Seed: 42 (regression / bootstrap; also passed to Ollama's `seed` option best-effort).
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import gzip, csv, json, os, sys, time
import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
from sklearn.metrics import cohen_kappa_score

ap = argparse.ArgumentParser()
ap.add_argument("--subsample-n", type=int, default=None,
                 help="If set, draw a FIXED-SEED random subsample of this "
                      "size from the 1,679-row candidate pool instead of "
                      "running all rows (the gold set, n=299, is always run "
                      "in full). Use only if the full run is too slow.")
ap.add_argument("--models", nargs="+", default=None,
                 help="Override the default model list (for smoke tests).")
ARGS = ap.parse_args()

RNG = np.random.default_rng(42)
MIMIC_LOCAL_ONLY_DIR = os.environ.get("MIMIC_LOCAL_ONLY_DIR", os.path.expanduser("~/mimic_local_only"))
OUTDIR = os.path.join(MIMIC_LOCAL_ONLY_DIR, "accord_multillm")
os.makedirs(OUTDIR, exist_ok=True)

log_lines = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log_lines.append(s)

# ---------------------------------------------------------------------------
# 0. VERBATIM PROMPT (from prompts_existing.csv, config_id=llama_baseline)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a clinical pharmacist reviewing medication reconciliation between "
    "nursing progress notes and hospital discharge summaries. Your task is to "
    "identify dangerous medication contradictions. Respond with exactly one "
    "word: contradiction or not_contradiction."
)
USER_TEMPLATE = (
    "Nursing Progress Note:\n{nursing_note}\n\n"
    "Discharge Summary:\n{discharge_summary}\n\n"
    "Is there a dangerous medication contradiction between these two notes?"
)
TEMPERATURE = 0
SEED = 42
MAX_TOKENS = 50

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = ARGS.models or ["llama3:8b", "granite3.3:8b", "qwen3:8b"]  # from `ollama list`
THINKING_MODELS = {"qwen3:8b"}  # models that need think:false to avoid
                                 # burning the token budget on <think> traces


def call_ollama(model, nursing, discharge, max_retries=4):
    """Call local Ollama, return raw response text (never logged/printed)."""
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": USER_TEMPLATE.format(
            nursing_note=nursing, discharge_summary=discharge),
        "options": {"temperature": TEMPERATURE, "seed": SEED,
                    "num_predict": MAX_TOKENS},
        "stream": False,
    }
    if model in THINKING_MODELS:
        payload["think"] = False
    for attempt in range(max_retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def parse_prediction(response):
    """
    Same case-insensitive first-word matching logic as parser.py, EXCEPT
    that no-match is returned as its own UNPARSEABLE category rather than
    silently defaulting to 0 (not_contradiction) or, as the retracted code
    did, to 1 (positive). This is the fix required by C10.
    """
    if response is None or not isinstance(response, str) or not response.strip():
        return "UNPARSEABLE"
    r = response.strip().lower()
    if "not_contradiction" in r or "not contradiction" in r:
        return "not_contradiction"
    tokens = r.split()
    if tokens:
        first = tokens[0].strip(".,!?:;\n\"'")
        if first == "not":
            return "not_contradiction"
    if "contradiction" in r:
        return "contradiction"
    return "UNPARSEABLE"


def run_model_on_rows(model, nursing_list, discharge_list, label):
    """Runs `model` over paired (nursing, discharge) text. Returns a list of
    parsed labels ('contradiction'/'not_contradiction'/'UNPARSEABLE'). Never
    retains or prints the note text or the raw model output."""
    n = len(nursing_list)
    preds = []
    t0 = time.time()
    for i in range(n):
        resp = call_ollama(model, nursing_list[i], discharge_list[i])
        preds.append(parse_prediction(resp))
        if (i + 1) % 200 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            log(f"    [{model} | {label}] {i+1}/{n}  ({elapsed:.0f}s elapsed)")
    return preds


# ---------------------------------------------------------------------------
# 1. LOAD THE 1,679-ROW CANDIDATE POOL (weak-supervision training pool)
# ---------------------------------------------------------------------------
WS_PATH = os.path.join(MIMIC_LOCAL_ONLY_DIR, "accord_supp/accord_weak_supervision_labels_decontam.csv")
ws = pd.read_csv(WS_PATH)
ws["HADM_ID"] = ws["HADM_ID"].astype(float)
ws["nursing_sentence"] = ws["nursing_sentence"].fillna("").astype(str)
ws["discharge_sentence"] = ws["discharge_sentence"].fillna("").astype(str)
log(f"Loaded candidate pool: {len(ws)} rows from {WS_PATH}")
log(f"Original (v1) Llama-3-8B weak_label positive rate on this pool: "
    f"{(ws['weak_label']=='contradiction').mean():.4f}")

# ---------------------------------------------------------------------------
# 2. LOAD THE 299-ROW PHARMACIST GOLD SET, join pair_id <-> HADM_ID
#    (identical join logic to accord/differential/run_experiments.py)
# ---------------------------------------------------------------------------
GOLD_PATH = os.path.join(MIMIC_LOCAL_ONLY_DIR, "derived/gold_standard_accord_results.csv")
# Repointed off the Google Drive mount. This file carries verbatim note
# sentences joined to HADM_ID, so a cloud-synced path was the one thing
# the .gitignore rules and sync gate exist to prevent, and README.md
# asserted it was not happening while these two lines said otherwise.
SAMP_PATH = os.path.join(MIMIC_LOCAL_ONLY_DIR, "from_drive_2026-08-08/outputs/annotation_sample_300.csv")
AUDIT_PATH = os.path.join(MIMIC_LOCAL_ONLY_DIR, "accord_supp/accord_decontamination_audit.csv")

gold = pd.read_csv(GOLD_PATH)
samp = pd.read_csv(SAMP_PATH)
audit = pd.read_csv(AUDIT_PATH)
audit["HADM_ID"] = audit["HADM_ID"].astype(float)

g2 = gold.merge(samp[["pair_id", "HADM_ID", "drug", "rule", "nursing_sentence", "discharge_sentence"]],
                 on=["pair_id", "rule"], how="left")
g2["HADM_ID"] = g2["HADM_ID"].astype(float)
key = ["HADM_ID", "drug", "rule", "nursing_sentence", "discharge_sentence"]
gold_join = g2.merge(audit[key + ["weak_label", "weak_confidence", "removal_reason"]], on=key, how="left")

assert gold_join["weak_label"].notna().sum() == 299, "gold<->weak-label join is not fully 1:1 -- STOP"
assert (gold_join["removal_reason"] == "pharmacist_overlap").all()
assert gold_join.shape[0] == 299

gold_join["pharm_pos"] = (gold_join["pharm_label"] == "contradiction").astype(int)
gold_join["nursing_sentence"] = gold_join["nursing_sentence"].fillna("").astype(str)
gold_join["discharge_sentence"] = gold_join["discharge_sentence"].fillna("").astype(str)

n_gold = len(gold_join)
n_gold_pos = int(gold_join["pharm_pos"].sum())
gold_base_rate = n_gold_pos / n_gold
log(f"\nLoaded gold set: {n_gold} pairs, {n_gold_pos} pharmacist positives "
    f"(base rate {gold_base_rate:.4f})")

# ---------------------------------------------------------------------------
# 3. DEMOGRAPHIC / DOCUMENTATION-VOLUME COVARIATES for the 1,679-row pool
#    (identical construction to run_experiments.py, needed to replicate
#    Experiment 1c: weak_pos ~ language + log_note_chars + n_mentions + rule)
# ---------------------------------------------------------------------------
adm = pd.read_csv(os.path.join(MIMIC_LOCAL_ONLY_DIR, "raw_mimiciii/ADMISSIONS.csv"),
                   parse_dates=["ADMITTIME", "DISCHTIME"])
adm["HADM_ID"] = adm["HADM_ID"].astype(float)

def lang_group(x):
    if pd.isna(x):
        return "MISSING"
    if x == "ENGL":
        return "ENGLISH"
    return "RECORDED_NON_ENGLISH"
adm["language_group3"] = adm["LANGUAGE"].apply(lang_group)
demo = adm[["HADM_ID", "language_group3"]].drop_duplicates("HADM_ID")

rows = []
with gzip.open(os.path.join(MIMIC_LOCAL_ONLY_DIR, "accord/hf_note_pairs.csv.gz"), "rt") as f:
    r = csv.reader(f)
    header = next(r)
    idx = {c: i for i, c in enumerate(header)}
    for row in r:
        rows.append((float(row[idx["HADM_ID"]]), len(row[idx["TEXT"]])))
note_len_df = pd.DataFrame(rows, columns=["HADM_ID", "n_chars"])
del rows
note_agg = note_len_df.groupby("HADM_ID").agg(total_note_chars=("n_chars", "sum")).reset_index()

mm_counts = []
with gzip.open(os.path.join(MIMIC_LOCAL_ONLY_DIR, "accord/hf_medication_mentions.csv.gz"), "rt") as f:
    r = csv.reader(f)
    header = next(r)
    idx = {c: i for i, c in enumerate(header)}
    from collections import Counter
    c = Counter()
    for row in r:
        c[float(row[idx["HADM_ID"]])] += 1
n_mentions_df = pd.DataFrame(list(c.items()), columns=["HADM_ID", "n_mentions"])

covar = demo.merge(note_agg, on="HADM_ID", how="left").merge(n_mentions_df, on="HADM_ID", how="left")
covar["n_mentions"] = covar["n_mentions"].fillna(0)
covar["log_note_chars"] = np.log1p(covar["total_note_chars"])

reg_df = ws.merge(covar, on="HADM_ID", how="left")
log(f"\nCovariate match rate on the 1,679-row pool: "
    f"{reg_df['language_group3'].notna().mean():.4f}")
reg_df = reg_df.dropna(subset=["language_group3", "log_note_chars"]).reset_index(drop=True)
reg_df["rule_cat"] = "rule" + reg_df["rule"].astype(str)
reg_df["language_group3"] = pd.Categorical(
    reg_df["language_group3"], categories=["ENGLISH", "RECORDED_NON_ENGLISH", "MISSING"])
log(f"Regression-eligible pool rows (after covariate join): {len(reg_df)}")

if ARGS.subsample_n is not None and ARGS.subsample_n < len(reg_df):
    sub_idx = RNG.choice(len(reg_df), size=ARGS.subsample_n, replace=False)
    reg_df = reg_df.iloc[np.sort(sub_idx)].reset_index(drop=True)
    log(f"*** SUBSAMPLED the candidate pool to n={len(reg_df)} rows "
        f"(fixed seed=42, np.random.default_rng) because --subsample-n was set. "
        f"The gold set (n={n_gold}) is still run in full. ***")

# ---------------------------------------------------------------------------
# 4. RUN EACH LOCAL MODEL, ON (a) the candidate pool, (b) the gold set
# ---------------------------------------------------------------------------
per_model = {}

for model in MODELS:
    log(f"\n{'='*70}\nMODEL: {model}\n{'='*70}")

    # -- (a) candidate pool (n up to 1,679; uses reg_df since it carries the
    #        covariates needed for Experiment (ii); a handful of rows are
    #        dropped for missing demo/covariate join, reported above) --
    pool_preds = run_model_on_rows(
        model, reg_df["nursing_sentence"].tolist(), reg_df["discharge_sentence"].tolist(),
        label="candidate pool")
    reg_df[f"pred_{model}"] = pool_preds

    n_pool = len(pool_preds)
    n_unparse_pool = sum(1 for p in pool_preds if p == "UNPARSEABLE")
    n_pos_pool = sum(1 for p in pool_preds if p == "contradiction")
    n_neg_pool = sum(1 for p in pool_preds if p == "not_contradiction")
    parseable_pool = n_pos_pool + n_neg_pool
    pos_rate_pool = n_pos_pool / parseable_pool if parseable_pool else float("nan")
    unparse_rate_pool = n_unparse_pool / n_pool if n_pool else float("nan")
    calib_ratio_A = pos_rate_pool / gold_base_rate if gold_base_rate else float("nan")

    # -- (b) gold set, same-population calibration + accuracy/kappa check --
    gold_preds = run_model_on_rows(
        model, gold_join["nursing_sentence"].tolist(), gold_join["discharge_sentence"].tolist(),
        label="gold set")
    gold_join[f"pred_{model}"] = gold_preds

    n_gold_unparse = sum(1 for p in gold_preds if p == "UNPARSEABLE")
    gold_parseable_mask = [p != "UNPARSEABLE" for p in gold_preds]
    gold_y_pred = [1 if p == "contradiction" else 0
                   for p, m in zip(gold_preds, gold_parseable_mask) if m]
    gold_y_true = [t for t, m in zip(gold_join["pharm_pos"].tolist(), gold_parseable_mask) if m]
    n_gold_parseable = len(gold_y_pred)
    pos_rate_gold = (sum(gold_y_pred) / n_gold_parseable) if n_gold_parseable else float("nan")
    unparse_rate_gold = n_gold_unparse / n_gold
    calib_ratio_B = pos_rate_gold / gold_base_rate if gold_base_rate else float("nan")

    if n_gold_parseable and len(set(gold_y_pred)) > 1 and len(set(gold_y_true)) > 1:
        kappa_gold = float(cohen_kappa_score(gold_y_true, gold_y_pred))
    else:
        kappa_gold = None  # degenerate (e.g. model predicts a single class)
    tp = sum(1 for t, p in zip(gold_y_true, gold_y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(gold_y_true, gold_y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(gold_y_true, gold_y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(gold_y_true, gold_y_pred) if t == 0 and p == 0)
    recall_gold = tp / (tp + fn) if (tp + fn) else float("nan")
    accuracy_gold = (tp + tn) / n_gold_parseable if n_gold_parseable else float("nan")

    log(f"\n  -- {model}: candidate pool (n={n_pool}, parseable={parseable_pool}) --")
    log(f"     positive={n_pos_pool}  negative={n_neg_pool}  UNPARSEABLE={n_unparse_pool} "
        f"({unparse_rate_pool:.3%})")
    log(f"     positive rate (of parseable) = {pos_rate_pool:.4f}")
    log(f"     calibration vs gold base rate ({gold_base_rate:.4f}): {calib_ratio_A:.2f}x")
    log(f"\n  -- {model}: gold set (n={n_gold}, parseable={n_gold_parseable}) --")
    log(f"     UNPARSEABLE={n_gold_unparse} ({unparse_rate_gold:.3%})")
    log(f"     positive rate (of parseable) = {pos_rate_gold:.4f}  "
        f"calibration = {calib_ratio_B:.2f}x")
    log(f"     confusion (parseable only): TP={tp} FP={fp} FN={fn} TN={tn}")
    log(f"     recall={recall_gold:.3f}  accuracy={accuracy_gold:.3f}  kappa={kappa_gold}")

    # -- (ii) logistic regression: model's pool prediction ~ language + adjustment set --
    reg_sub = reg_df[reg_df[f"pred_{model}"] != "UNPARSEABLE"].copy()
    reg_sub["model_pos"] = (reg_sub[f"pred_{model}"] == "contradiction").astype(int)
    n_reg = len(reg_sub)
    n_dropped_unparse = len(reg_df) - n_reg
    log(f"\n  -- {model}: logistic regression (dropping {n_dropped_unparse} "
        f"UNPARSEABLE rows from pool), n={n_reg} --")

    or_result = None
    try:
        m = smf.logit(
            "model_pos ~ C(language_group3, Treatment('ENGLISH')) + log_note_chars + "
            "n_mentions + C(rule_cat)",
            data=reg_sub).fit(disp=0)
        or_ci = np.exp(m.conf_int())
        or_ci["OR"] = np.exp(m.params)
        or_ci.columns = ["CI_low", "CI_high", "OR"]
        or_ci["p"] = m.pvalues
        key_missing = "C(language_group3, Treatment('ENGLISH'))[T.MISSING]"
        key_nonenglish = "C(language_group3, Treatment('ENGLISH'))[T.RECORDED_NON_ENGLISH]"
        or_result = {
            "n": n_reg,
            "converged": bool(m.mle_retvals.get("converged")),
            "missing_language_OR": round(float(or_ci.loc[key_missing, "OR"]), 4),
            "missing_language_CI_low": round(float(or_ci.loc[key_missing, "CI_low"]), 4),
            "missing_language_CI_high": round(float(or_ci.loc[key_missing, "CI_high"]), 4),
            "missing_language_p": round(float(or_ci.loc[key_missing, "p"]), 5),
            "recorded_non_english_OR": round(float(or_ci.loc[key_nonenglish, "OR"]), 4),
            "recorded_non_english_CI_low": round(float(or_ci.loc[key_nonenglish, "CI_low"]), 4),
            "recorded_non_english_CI_high": round(float(or_ci.loc[key_nonenglish, "CI_high"]), 4),
            "recorded_non_english_p": round(float(or_ci.loc[key_nonenglish, "p"]), 5),
            "pseudo_r2": float(m.prsquared),
        }
        log(f"     MISSING-language OR = {or_result['missing_language_OR']} "
            f"[{or_result['missing_language_CI_low']}, {or_result['missing_language_CI_high']}] "
            f"p={or_result['missing_language_p']}")
        log(f"     RECORDED_NON_ENGLISH OR = {or_result['recorded_non_english_OR']} "
            f"[{or_result['recorded_non_english_CI_low']}, {or_result['recorded_non_english_CI_high']}] "
            f"p={or_result['recorded_non_english_p']}")
    except Exception as e:
        log(f"     Regression FAILED for {model}: {e}")
        if reg_sub["model_pos"].nunique() < 2:
            log(f"     (model predicted a single class on the pool -- "
                f"class counts: {reg_sub['model_pos'].value_counts().to_dict()})")

    per_model[model] = {
        "pool": {
            "n": n_pool, "n_unparseable": n_unparse_pool,
            "unparseable_rate": unparse_rate_pool,
            "n_positive": n_pos_pool, "n_negative": n_neg_pool,
            "positive_rate_of_parseable": pos_rate_pool,
            "calibration_ratio_vs_gold_base_rate": calib_ratio_A,
        },
        "gold": {
            "n": n_gold, "n_unparseable": n_gold_unparse,
            "unparseable_rate": unparse_rate_gold,
            "n_parseable": n_gold_parseable,
            "positive_rate_of_parseable": pos_rate_gold,
            "calibration_ratio_vs_gold_base_rate": calib_ratio_B,
            "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
            "recall": recall_gold, "accuracy": accuracy_gold, "kappa": kappa_gold,
        },
        "language_regression": or_result,
    }

# ---------------------------------------------------------------------------
# 5. REFERENCE ROW -- the original (retracted) Llama-3-8B-Instruct run
#    (from the actually-executed CSV / exp1c regression already on disk;
#    NOT hardcoded here -- read from the source files)
# ---------------------------------------------------------------------------
orig_pos_rate = float((ws["weak_label"] == "contradiction").mean())
orig_calib = orig_pos_rate / gold_base_rate
EXP1C_PATH = os.path.join(MIMIC_LOCAL_ONLY_DIR, "accord_differential/exp1c_logistic_regression.json")
orig_or = None
if os.path.isfile(EXP1C_PATH):
    with open(EXP1C_PATH) as f:
        exp1c = json.load(f)
    orig_or = exp1c["or_ci"]["C(language_group3, Treatment('ENGLISH'))[T.MISSING]"]

log(f"\n{'='*70}\nREFERENCE: original Llama-3-8B-Instruct (HF Inference API, retracted C10 run)\n{'='*70}")
log(f"  pool positive rate = {orig_pos_rate:.4f}  calibration = {orig_calib:.2f}x")
if orig_or:
    log(f"  MISSING-language OR = {orig_or['OR']} [{orig_or['CI_low']}, {orig_or['CI_high']}] "
        f"p={orig_or['p']}")

# ---------------------------------------------------------------------------
# 6. SUMMARY TABLE + VERDICT
# ---------------------------------------------------------------------------
table_rows = []
table_rows.append({
    "model": "Llama-3-8B-Instruct (original, HF Inference API -- REFERENCE, not re-run)",
    "n_pool": len(ws), "pool_positive_rate": round(orig_pos_rate, 4),
    "unparseable_rate_pool": None,
    "calibration_vs_4pct_gold": round(orig_calib, 2),
    "missing_language_OR": orig_or["OR"] if orig_or else None,
    "missing_language_CI": f"[{orig_or['CI_low']}, {orig_or['CI_high']}]" if orig_or else None,
    "missing_language_p": orig_or["p"] if orig_or else None,
})
for model in MODELS:
    d = per_model[model]
    lr = d["language_regression"]
    table_rows.append({
        "model": model,
        "n_pool": d["pool"]["n"],
        "pool_positive_rate": round(d["pool"]["positive_rate_of_parseable"], 4),
        "unparseable_rate_pool": round(d["pool"]["unparseable_rate"], 4),
        "calibration_vs_4pct_gold": round(d["pool"]["calibration_ratio_vs_gold_base_rate"], 2),
        "missing_language_OR": lr["missing_language_OR"] if lr else None,
        "missing_language_CI": f"[{lr['missing_language_CI_low']}, {lr['missing_language_CI_high']}]" if lr else None,
        "missing_language_p": lr["missing_language_p"] if lr else None,
    })

table_df = pd.DataFrame(table_rows)
log(f"\n{'='*70}\nSUMMARY TABLE\n{'='*70}")
log(table_df.to_string(index=False))

# verdict on (i) gross miscalibration
replicating_models = [m for m in MODELS
                       if per_model[m]["pool"]["calibration_ratio_vs_gold_base_rate"] > 3.0]
log(f"\nVERDICT (i) gross miscalibration: models with pool calibration ratio > 3x "
    f"(i.e. clearly over-flagging relative to the 4% gold base rate): {replicating_models} "
    f"of {MODELS}")

# verdict on (ii) demographic differential
sig_missing_positive = []
for m in MODELS:
    lr = per_model[m]["language_regression"]
    if lr and lr["missing_language_p"] < 0.05 and lr["missing_language_OR"] > 1.0:
        sig_missing_positive.append(m)
log(f"VERDICT (ii) demographic differential: models replicating a significant "
    f"(p<0.05) MISSING-language OR > 1 (over-flagging the missing-language subgroup): "
    f"{sig_missing_positive} of {MODELS}")

if len(sig_missing_positive) == len(MODELS):
    log("  --> Differential replicates across ALL locally-run models: consistent with "
        "a property of the TASK, not of one model.")
elif len(sig_missing_positive) == 0:
    log("  --> Differential did NOT replicate in any locally-run model: the original "
        "finding is model-specific to Llama-3-8B-Instruct (HF Inference API) and must "
        "be reported as such.")
else:
    log(f"  --> Differential replicates in {len(sig_missing_positive)}/{len(MODELS)} "
        "locally-run models: partial replication, report per-model.")

# ---------------------------------------------------------------------------
# 7. SAVE (aggregates only -- no note text, no raw model output text)
# ---------------------------------------------------------------------------
with open(os.path.join(OUTDIR, "per_model_results.json"), "w") as f:
    json.dump({
        "prompt_source": "prompts_existing.csv, config_id=llama_baseline (verbatim)",
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_TEMPLATE,
        "temperature": TEMPERATURE, "seed": SEED, "max_tokens": MAX_TOKENS,
        "gold_base_rate": gold_base_rate, "n_gold": n_gold, "n_gold_positive": n_gold_pos,
        "original_llama3_reference": {
            "pool_positive_rate": orig_pos_rate,
            "calibration_ratio": orig_calib,
            "missing_language_OR_CI_p": orig_or,
        },
        "per_model": per_model,
        "replicating_models_gross_miscalibration": replicating_models,
        "replicating_models_demographic_differential": sig_missing_positive,
    }, f, indent=2, default=str)

table_df.to_csv(os.path.join(OUTDIR, "per_model_table.csv"), index=False)

with open(os.path.join(OUTDIR, "run_log.txt"), "w") as f:
    f.write("\n".join(log_lines))

log(f"\nSaved per_model_results.json, per_model_table.csv, run_log.txt to {OUTDIR}")
print("ALL_DONE")
