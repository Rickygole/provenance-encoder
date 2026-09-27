"""
ACCORD prevalence-replication sweep -- testing whether the clinical weak-
labeller over-calling finding (CORRECTIONS.md C20/C23: 15.8x-23.6x over-call
of an 8B open-weights model against a 4.0% expert base rate) is a property
of CLINICAL ADJUDICATION, or of asking an instruction-tuned model to make a
rare-class judgment from short evidence.

DATA: SNLI premise/hypothesis pairs (see build_pools.py for dataset choice
rationale), PUBLIC, human-labeled, non-clinical. Four pools at true
contradiction prevalence 2%, 5%, 10%, 25%, N=100 each, disjoint across
levels, fixed seed 42.

MODELS: local Ollama only -- llama3:8b, granite3.3:8b, qwen3:8b (matching
the clinical C20 replication exactly). temperature=0, seed=42.

PROMPTS (the sweep C23 says the clinical work never did):
  A. BARE -- structurally identical framing to the clinical
     "llama_baseline" prompt (see differential/run_multillm_replication.py):
     conclusion-asserting, forced binary choice, no base-rate information.
  B. BASE-RATE-AWARE + ABSTENTION -- states the pool's true prevalence
     explicitly and permits an "unsure" response.

FAILURE HANDLING (C10's lesson): unparseable generations are their own
UNPARSEABLE category and are NEVER counted as positive.

Writes only to this repo (public SNLI data + aggregate results). No
clinical/MIMIC data is read or written by this script.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
POOL_DIR = os.path.join(HERE, "data", "pools")
RESULTS_DIR = os.path.join(HERE, "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw_predictions")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+",
                 default=["llama3:8b", "granite3.3:8b", "qwen3:8b"])
ap.add_argument("--prevalence-levels", nargs="+", type=float,
                 default=[0.02, 0.05, 0.10, 0.25])
ap.add_argument("--limit-n", type=int, default=None,
                 help="For smoke tests: only run the first K items of each pool.")
ap.add_argument("--resume", action="store_true", default=True,
                 help="Skip (model, prompt, level) combos whose raw CSV already exists.")
ap.add_argument("--no-resume", dest="resume", action="store_false")
ARGS = ap.parse_args()

SEED = 42
TEMPERATURE = 0
MAX_TOKENS = 60
OLLAMA_URL = "http://localhost:11434/api/generate"
THINKING_MODELS = {"qwen3:8b"}  # needs think:false or it burns the token budget

log_lines = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log_lines.append(s)

# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------
def prompt_A(premise, hypothesis, prevalence):
    system = (
        "You are an annotator judging semantic relationships between sentence "
        "pairs. Your task is to identify contradictions. Respond with exactly "
        "one word: contradiction or not_contradiction."
    )
    user = (
        f"Sentence A:\n{premise}\n\n"
        f"Sentence B:\n{hypothesis}\n\n"
        "Is there a contradiction between these two sentences?"
    )
    return system, user

def prompt_B(premise, hypothesis, prevalence):
    system = (
        "You are an annotator judging semantic relationships between sentence "
        f"pairs. In this dataset, sentence pairs are contradictions only "
        f"rarely -- about {prevalence*100:.0f}% of pairs. Most pairs are NOT "
        "contradictions. Only answer contradiction if you are confident the "
        "two sentences genuinely cannot both be true. If you are not "
        "confident either way, say unsure. Respond with exactly one word: "
        "contradiction, not_contradiction, or unsure."
    )
    user = (
        f"Sentence A:\n{premise}\n\n"
        f"Sentence B:\n{hypothesis}\n\n"
        "Is there a contradiction between these two sentences?"
    )
    return system, user

PROMPTS = {"A_bare": prompt_A, "B_baserate_abstain": prompt_B}


def call_ollama(model, system, user, max_retries=4):
    payload = {
        "model": model,
        "system": system,
        "prompt": user,
        "options": {"temperature": TEMPERATURE, "seed": SEED, "num_predict": MAX_TOKENS},
        "stream": False,
    }
    if model in THINKING_MODELS:
        payload["think"] = False
    for attempt in range(max_retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def parse_prediction(response, allow_abstain):
    """
    Case-insensitive first-token / substring matching. An unparseable
    generation is its own UNPARSEABLE category and is NEVER scored as
    positive -- this is the C10 fix, applied identically here.
    """
    if response is None or not isinstance(response, str) or not response.strip():
        return "UNPARSEABLE"
    r = response.strip().lower()
    if allow_abstain and ("unsure" in r):
        return "ABSTAIN"
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


# ---------------------------------------------------------------------------
# MAIN SWEEP
# ---------------------------------------------------------------------------
def run_combo(model, prompt_name, prompt_fn, level, pool):
    out_csv = os.path.join(RAW_DIR, f"{model.replace(':','_')}__{prompt_name}__p{level:.2f}.csv")
    if ARGS.resume and os.path.isfile(out_csv):
        log(f"  [SKIP, exists] {out_csv}")
        return pd.read_csv(out_csv)

    n = len(pool)
    preds, raws = [], []
    t0 = time.time()
    for i, row in pool.iterrows():
        system, user = prompt_fn(row["premise"], row["hypothesis"], level)
        resp = call_ollama(model, system, user)
        pred = parse_prediction(resp, allow_abstain=(prompt_name == "B_baserate_abstain"))
        preds.append(pred)
        raws.append(resp if resp is not None else "")
        if (i + 1) % 20 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            log(f"    [{model} | {prompt_name} | p={level:.2f}] {i+1}/{n} ({elapsed:.0f}s)")

    out = pool.copy()
    out["pred"] = preds
    out["raw_response"] = raws
    out.to_csv(out_csv, index=False)
    return out


def _fmt(x, places):
    """A metric for the run log, or "undefined" where there was no denominator."""
    return "undefined" if x is None else f"{x:.{places}f}"


def combo_metrics(out_df, level):
    n = len(out_df)
    n_contra = int((out_df["pred"] == "contradiction").sum())
    n_not = int((out_df["pred"] == "not_contradiction").sum())
    n_abstain = int((out_df["pred"] == "ABSTAIN").sum())
    n_unparse = int((out_df["pred"] == "UNPARSEABLE").sum())
    decided = n_contra + n_not  # excludes abstain and unparseable
    pos_rate_of_decided = n_contra / decided if decided else None
    pos_rate_of_all_valid = n_contra / (n - n_unparse) if (n - n_unparse) else None
    true_pos = int(out_df["is_contradiction"].sum())
    true_rate = true_pos / n
    over_call_ratio = (pos_rate_of_decided / true_rate
                       if true_rate and pos_rate_of_decided is not None else None)

    # accuracy on decided rows only, vs SNLI gold
    decided_mask = out_df["pred"].isin(["contradiction", "not_contradiction"])
    if decided_mask.sum():
        y_true = out_df.loc[decided_mask, "is_contradiction"].to_numpy()
        y_pred = (out_df.loc[decided_mask, "pred"] == "contradiction").astype(int).to_numpy()
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) else None
        precision = tp / (tp + fp) if (tp + fp) else None
        accuracy = (tp + tn) / decided_mask.sum()
    else:
        tp = fp = fn = tn = 0
        recall = precision = accuracy = None

    return {
        "n": n, "n_contradiction": n_contra, "n_not_contradiction": n_not,
        "n_abstain": n_abstain, "n_unparseable": n_unparse,
        "unparseable_rate": n_unparse / n,
        "abstain_rate": n_abstain / n,
        "n_decided": decided,
        "positive_rate_of_decided": pos_rate_of_decided,
        "positive_rate_of_all_valid": pos_rate_of_all_valid,
        "true_prevalence": true_rate,
        "over_call_ratio": over_call_ratio,
        "confusion_decided_only": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "recall_decided": recall, "precision_decided": precision, "accuracy_decided": accuracy,
    }


def main():
    pools = {}
    for level in ARGS.prevalence_levels:
        p = pd.read_csv(os.path.join(POOL_DIR, f"pool_prevalence_{level:.2f}.csv"))
        if ARGS.limit_n:
            p = p.iloc[:ARGS.limit_n].reset_index(drop=True)
        pools[level] = p
        log(f"Loaded pool prevalence={level:.2f}: n={len(p)}, "
            f"n_pos={int(p['is_contradiction'].sum())}, "
            f"achieved={p['is_contradiction'].mean():.4f}")

    all_results = {}  # model -> prompt -> level -> metrics
    for model in ARGS.models:
        all_results[model] = {}
        for prompt_name, prompt_fn in PROMPTS.items():
            all_results[model][prompt_name] = {}
            for level in ARGS.prevalence_levels:
                log(f"\n{'='*70}\n{model} | {prompt_name} | prevalence={level:.2%}\n{'='*70}")
                out_df = run_combo(model, prompt_name, prompt_fn, level, pools[level])
                metrics = combo_metrics(out_df, level)
                all_results[model][prompt_name][level] = metrics
                log(f"  n={metrics['n']}  pos={metrics['n_contradiction']} "
                    f"neg={metrics['n_not_contradiction']} abstain={metrics['n_abstain']} "
                    f"unparseable={metrics['n_unparseable']}")
                log(f"  positive_rate_of_decided={_fmt(metrics['positive_rate_of_decided'], 4)} "
                    f"vs true_prevalence={metrics['true_prevalence']:.4f} "
                    f"-> over_call_ratio={_fmt(metrics['over_call_ratio'], 2)}x")

    with open(os.path.join(RESULTS_DIR, "sweep_results.json"), "w") as f:
        json.dump({
            "seed": SEED, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
            "models": ARGS.models, "prevalence_levels": ARGS.prevalence_levels,
            "prompts": {k: {"system_template": "see run_prevalence_sweep.py:prompt_" + k[0]}
                        for k in PROMPTS},
            "results": all_results,
            # allow_nan=False: NaN and Infinity are not JSON, and this file is read
            # back by the register, which ships in the published repository. An
            # undefined metric is None here and null on disk; see combo_metrics.
        }, f, indent=2, default=str, allow_nan=False)

    with open(os.path.join(RESULTS_DIR, "sweep_run_log.txt"), "w") as f:
        f.write("\n".join(log_lines))

    log(f"\nSaved sweep_results.json, sweep_run_log.txt, raw per-combo CSVs to {RESULTS_DIR}")
    print("ALL_DONE")


if __name__ == "__main__":
    main()
