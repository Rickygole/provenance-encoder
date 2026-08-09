"""
Well-powered supplement to the prevalence-inheritance question.

The N=100 pools used in run_prevalence_sweep.py turned out too small for
TF-IDF+LR to separate the classes at all -- the 5-fold-CV student collapsed
to predicting "not_contradiction" for every out-of-fold item regardless of
the teacher's marginal (see results/distillation_results.json / REPORT.md).
That is itself a reportable finding (distillation needs a training-set size
the clinical pipeline had -- ~1,679 rows -- and this experiment's N=100
pools did not), but it does not answer the question the task asks: does the
student track the TEACHER's marginal or the TRUE prevalence.

This script re-runs ONE model x ONE prompt -- `llama3:8b`, prompt A (bare),
the exact analog of the clinical "llama_baseline" reference -- at N=300 per
prevalence level, on pools disjoint from every sweep pool (see
build_pools.py's DISTILL_POOL_N section), specifically to give the
distillation student enough training rows to be non-degenerate.

Same failure handling as run_prevalence_sweep.py: UNPARSEABLE is its own
category and never defaults to positive.
"""
import json
import os
import time

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
POOL_DIR = os.path.join(HERE, "data", "pools")
RESULTS_DIR = os.path.join(HERE, "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw_predictions_distill_supplement")
os.makedirs(RAW_DIR, exist_ok=True)

MODEL = "llama3:8b"
PREVALENCE_LEVELS = [0.02, 0.05, 0.10, 0.25]
SEED = 42
TEMPERATURE = 0
MAX_TOKENS = 60
OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = (
    "You are an annotator judging semantic relationships between sentence "
    "pairs. Your task is to identify contradictions. Respond with exactly "
    "one word: contradiction or not_contradiction."
)
USER_TEMPLATE = (
    "Sentence A:\n{premise}\n\n"
    "Sentence B:\n{hypothesis}\n\n"
    "Is there a contradiction between these two sentences?"
)


def call_ollama(model, system, user, max_retries=4):
    payload = {
        "model": model, "system": system, "prompt": user,
        "options": {"temperature": TEMPERATURE, "seed": SEED, "num_predict": MAX_TOKENS},
        "stream": False,
    }
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


def parse_prediction(response):
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


def main():
    for level in PREVALENCE_LEVELS:
        out_csv = os.path.join(RAW_DIR, f"llama3_8b__A_bare__distill_p{level:.2f}.csv")
        if os.path.isfile(out_csv):
            print(f"[SKIP, exists] {out_csv}")
            continue
        pool = pd.read_csv(os.path.join(POOL_DIR, f"distill_pool_prevalence_{level:.2f}.csv"))
        print(f"\n=== {MODEL} | A_bare | distill prevalence={level:.2%} | n={len(pool)} ===")
        preds, raws = [], []
        t0 = time.time()
        for i, row in pool.iterrows():
            user = USER_TEMPLATE.format(premise=row["premise"], hypothesis=row["hypothesis"])
            resp = call_ollama(MODEL, SYSTEM_PROMPT, user)
            pred = parse_prediction(resp)
            preds.append(pred)
            raws.append(resp if resp is not None else "")
            if (i + 1) % 50 == 0 or (i + 1) == len(pool):
                print(f"    {i+1}/{len(pool)} ({time.time()-t0:.0f}s)")
        out = pool.copy()
        out["pred"] = preds
        out["raw_response"] = raws
        out.to_csv(out_csv, index=False)
        n_unparse = int((out["pred"] == "UNPARSEABLE").sum())
        n_pos = int((out["pred"] == "contradiction").sum())
        print(f"  n={len(out)} pos={n_pos} unparseable={n_unparse} "
              f"positive_rate={n_pos/(len(out)-n_unparse):.4f} vs true={level:.4f}")
    print("ALL_DONE")


if __name__ == "__main__":
    main()
