"""Per-pair replication labels on the 299-pair pharmacist gold set for the two
replication models whose per-row calls were never persisted.

`accord/differential/run_multillm_replication.py` ran three local Ollama models
(llama3:8b, granite3.3:8b, qwen3:8b) over the gold set but wrote aggregates
only. `accord/c130_reproducible_student/persist_replication_labels.py` later
recovered the per-row llama3:8b calls. granite3.3:8b and qwen3:8b were never
recovered, so no analysis could say WHERE those two models over-call. This
script recovers them, using the same prompt, the same Ollama options, the same
parser, and the same gold join.

GATE. The recovered per-pair calls must reproduce the already-registered
confusion matrix in results/accord/multillm/per_model_results.json exactly
(TP/FP/FN/TN and the unparseable count). If they do not, the script writes
nothing and exits non-zero: a mismatch means the recovered labels are not the
labels the thesis's aggregates were computed from.

TRANSPORT DEVIATION, deliberate and recorded. The original `call_ollama` uses a
120 second read timeout, four retries, and returns None on exhaustion, which
`parse_prediction` turns into UNPARSEABLE. On a contended Ollama daemon (other
processes on this machine load and evict other models between requests) that
converts a queueing delay into a fabricated UNPARSEABLE label. This script
therefore sends the request itself, with a long read timeout, a keep_alive that
pins the model for the run, and an abort instead of a fabricated label if the
request genuinely cannot be completed. The prompt, the options that reach the
model (temperature, seed, num_predict, think) and `parse_prediction` are the
original ones, lifted by static parse. The gate below is what makes this safe:
if the transport change had altered a single call, the recovered confusion
matrix would not equal the registered one.

DATA AGREEMENT. MIMIC-III credentialed, local only. Ollama is called at
127.0.0.1 only. The per-pair output carries pair_id (a synthetic key),
rule, drug and the parsed one-word label. No note text, no admission or
subject identifier, and no raw model output is written or printed.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from accord.cloud_paths import assert_local_only  # noqa: E402

MIMIC = Path(os.environ.get("MIMIC_LOCAL_ONLY_DIR", os.path.expanduser("~/mimic_local_only")))
SRC = REPO / "accord" / "differential" / "run_multillm_replication.py"
REGISTERED = REPO / "results" / "accord" / "multillm" / "per_model_results.json"
OUTDIR = assert_local_only(MIMIC / "accord_multillm")

MODELS = ["granite3.3:8b", "qwen3:8b"]
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# ---------------------------------------------------------------------------
# Lift the prompt constants and the two functions from the original script by
# static parse. The source is never executed as a module (importing it would
# run its whole top-level body, including its own Ollama calls and writes).
# ---------------------------------------------------------------------------
tree = ast.parse(SRC.read_text())
consts: dict[str, object] = {}
funcs: dict[str, ast.FunctionDef] = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if name in ("SYSTEM_PROMPT", "USER_TEMPLATE", "TEMPERATURE", "SEED", "MAX_TOKENS", "THINKING_MODELS"):
            try:
                consts[name] = ast.literal_eval(node.value)
            except ValueError:
                pass
    if isinstance(node, ast.FunctionDef) and node.name == "parse_prediction":
        funcs[node.name] = node

for k in ("SYSTEM_PROMPT", "USER_TEMPLATE", "TEMPERATURE", "SEED", "MAX_TOKENS", "THINKING_MODELS"):
    assert k in consts, f"could not statically lift {k} from {SRC}"
assert set(funcs) == {"parse_prediction"}, "could not lift parse_prediction"

NS: dict[str, object] = {}
exec(compile(ast.Module(body=list(funcs.values()), type_ignores=[]), str(SRC), "exec"), NS)
parse_prediction = NS["parse_prediction"]

SYSTEM_PROMPT = consts["SYSTEM_PROMPT"]
USER_TEMPLATE = consts["USER_TEMPLATE"]
THINKING_MODELS = consts["THINKING_MODELS"]
READ_TIMEOUT = 900          # a queued request on a contended daemon is not a failure
MAX_ATTEMPTS = 8
KEEP_ALIVE = "45m"          # pin the model so other processes cannot evict it mid-run


def call_ollama(model: str, nursing: str, discharge: str) -> str:
    """Send one request. Raises rather than returning None, so a transport
    failure can never be recorded as an UNPARSEABLE model call."""
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": USER_TEMPLATE.format(nursing_note=nursing, discharge_summary=discharge),
        "options": {"temperature": consts["TEMPERATURE"], "seed": consts["SEED"],
                    "num_predict": consts["MAX_TOKENS"]},
        "stream": False,
        "keep_alive": KEEP_ALIVE,
    }
    if model in THINKING_MODELS:
        payload["think"] = False
    last = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=(10, READ_TIMEOUT))
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:                      # noqa: BLE001
            last = type(e).__name__                # never the body: it quotes the note back
            time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f"Ollama request for {model} failed after {MAX_ATTEMPTS} attempts ({last})")

# ---------------------------------------------------------------------------
# The gold join, identical to run_multillm_replication.py section 2.
# ---------------------------------------------------------------------------
gold = pd.read_csv(MIMIC / "derived/gold_standard_accord_results.csv")
samp = pd.read_csv(MIMIC / "from_drive_2026-08-08/outputs/annotation_sample_300.csv")
audit = pd.read_csv(MIMIC / "accord_supp/accord_decontamination_audit.csv")
audit["HADM_ID"] = audit["HADM_ID"].astype(float)

g2 = gold.merge(
    samp[["pair_id", "HADM_ID", "drug", "rule", "nursing_sentence", "discharge_sentence"]],
    on=["pair_id", "rule"], how="left")
g2["HADM_ID"] = g2["HADM_ID"].astype(float)
key = ["HADM_ID", "drug", "rule", "nursing_sentence", "discharge_sentence"]
gold_join = g2.merge(audit[key + ["weak_label", "weak_confidence", "removal_reason"]], on=key, how="left")
assert gold_join["weak_label"].notna().sum() == 299, "gold <-> weak-label join is not 1:1 -- STOP"
assert gold_join.shape[0] == 299
gold_join["nursing_sentence"] = gold_join["nursing_sentence"].fillna("").astype(str)
gold_join["discharge_sentence"] = gold_join["discharge_sentence"].fillna("").astype(str)
gold_join["pharm_pos"] = (gold_join["pharm_label"] == "contradiction").astype(int)

registered = json.loads(REGISTERED.read_text())["per_model"]

for model in MODELS:
    out_path = assert_local_only(OUTDIR / f"gold_replication_labels_{model.replace(':', '_').replace('.', '')}.csv")
    preds = []
    t0 = time.time()
    for i in range(len(gold_join)):
        resp = call_ollama(model, gold_join["nursing_sentence"].iat[i], gold_join["discharge_sentence"].iat[i])
        preds.append(parse_prediction(resp))
        if (i + 1) % 50 == 0 or (i + 1) == len(gold_join):
            print(f"  [{model}] {i+1}/{len(gold_join)} ({time.time()-t0:.0f}s)", flush=True)

    df = gold_join[["pair_id", "rule", "drug", "pharm_label"]].copy()
    df["replication_label"] = preds
    pos = (df["replication_label"] == "contradiction")
    tp = int((pos & (gold_join["pharm_pos"] == 1)).sum())
    fp = int((pos & (gold_join["pharm_pos"] == 0)).sum())
    fn = int((~pos & (gold_join["pharm_pos"] == 1)).sum())
    tn = int((~pos & (gold_join["pharm_pos"] == 0)).sum())
    n_unp = int((df["replication_label"] == "UNPARSEABLE").sum())
    reg = registered[model]["gold"]
    got = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    print(f"  [{model}] recovered {got}, unparseable {n_unp}; registered {reg['confusion']}", flush=True)
    if got != reg["confusion"] or n_unp != reg["n_unparseable"]:
        sys.exit(f"GATE FAILED for {model}: recovered calls do not reproduce the registered aggregate")
    df.to_csv(out_path, index=False)
    print(f"  [{model}] gate passed, wrote {out_path}", flush=True)

    # Release this model before moving to the next one. KEEP_ALIVE pins the
    # model so a contending process cannot evict it mid-run, but leaving it
    # pinned afterwards blocks the NEXT model from loading when the daemon
    # holds only one model at a time. This cost the first run about 20 minutes.
    try:
        requests.post(OLLAMA_URL, json={"model": model, "prompt": "", "keep_alive": 0}, timeout=30)
        print(f"  [{model}] released from memory", flush=True)
    except Exception as e:                          # noqa: BLE001
        print(f"  [{model}] release request failed ({type(e).__name__}); "
              f"it will expire after {KEEP_ALIVE}", flush=True)

print("done")
