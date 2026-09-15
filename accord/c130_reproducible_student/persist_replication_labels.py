"""C130 companion: persist the per-row llama3:8b replication labels that
`accord/differential/run_multillm_replication.py` computed but, by its own
docstring, never wrote to disk ("Writes ONLY aggregate results ... No note
text, and no per-row raw model output text, is ever written to disk or
printed"). See C130_PREREGISTRATION.md's "A gap found while preparing this
run" section: `reg_df[f"pred_{model}"]` and `gold_join[f"pred_{model}"]`
existed only in that process's memory.

This script re-runs the identical llama3:8b inference (same prompt, same
Ollama options, same parsing) and writes the per-row labels the
pre-registration's gates need:

    ~/mimic_local_only/accord_multillm/pool_replication_labels_llama3_8b.csv
    ~/mimic_local_only/accord_multillm/gold_replication_labels_llama3_8b.csv

and aggregates (positive rates, parse failures, the marginal-match check
against the already-registered `overcall.llama3_8b.pool.positive_rate`) to

    results/accord/c130_reproducible_student/replication_relabel_aggregates.json

REPRODUCIBILITY OF THE LABELLING STEP. `run_multillm_replication.py` is never
imported (importing it would execute its whole top-level body: argparse,
loading all three models' worth of Ollama calls, and its own file writes --
side effects this script must not trigger just to read three constants and
two functions). Instead, `SYSTEM_PROMPT`, `USER_TEMPLATE`, `TEMPERATURE`,
`SEED`, and `MAX_TOKENS` are statically parsed (via `ast`, source never
executed) out of that file and asserted equal to the copies below; the
`parse_prediction` and `call_ollama` function BODIES are lifted by the same
static parse and executed verbatim in a private namespace, so the parsing and
request-construction logic running here is the original code, not a
rewrite of it. The one deliberate deviation: `OLLAMA_URL` is pinned to the
literal loopback IP (`127.0.0.1`, not `localhost`) per this task's data
agreement, which does not change the request the model receives.

ROW SOURCE AND ORDER. The pool is `ws` read straight from
`accord_weak_supervision_labels_decontam.csv` (the same file
`run_multillm_replication.py` loads at its section 1), in file order.
`run_multillm_replication.py` actually predicts over `reg_df` (`ws` left-joined
with demographic/note-length covariates needed only for its own downstream
logistic regression, Experiment ii) rather than `ws` directly, but its own
archived run recorded a covariate match rate of 1.0000 and
`per_model_results.json`'s `per_model["llama3:8b"]["pool"]["n"] == 1679`
(`~/mimic_local_only/accord_multillm/run_log.txt`, `per_model_results.json`):
no pool row was dropped and a left join preserves row order, so `reg_df` and
`ws` are, for every row and in the same order, the same 1,679 (nursing,
discharge) pairs. Re-deriving the covariates here would read three more
credentialed files for a join that cannot change a single label. The gold
set is `accord.control.data.load_gold()` -- the same `pair_id <-> HADM_ID`
join `run_multillm_replication.py` performs inline at its section 2 (gold
merge samp merge audit, same key, same asserted 299 rows, same order because
both are `how="left"` merges against the same left frame).

DATA AGREEMENT. Local only, MIMIC-III credentialed. This script's stdout and
stderr are redirected to a file under ~/mimic_local_only/logs (never read
back into any conversation). The per-row label CSVs (which carry note
sentences and HADM_IDs, per the pre-registration's join schema) are written
ONLY under ~/mimic_local_only/accord_multillm/, each path checked with
accord.cloud_paths.assert_local_only before writing. Ollama is called only at
127.0.0.1:11434; an egress check samples `lsof` for this process's
connections during the run and both a note (local log) and a boolean
(aggregate JSON) record whether every observed remote endpoint was loopback.

MEMORY CONSTRAINT. Before any Ollama call, this script itself polls
`pgrep -f run_c127` and blocks until C127 has exited, in addition to (not
instead of) the launcher's own wait loop -- belt and suspenders against a
race where C127 is restarted between the launcher's check and this script's
first request.

    python3 accord/c130_reproducible_student/persist_replication_labels.py
"""
from __future__ import annotations

import ast
import contextlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accord" / "control"))

from accord.cloud_paths import assert_local_only  # noqa: E402

# ---------------------------------------------------------------------------
# Paths (all under the credentialed-only tree; every write path is checked).
# ---------------------------------------------------------------------------
OUT_DIR = assert_local_only(pathlib.Path.home() / "mimic_local_only" / "accord_multillm")
OUT_DIR.mkdir(parents=True, exist_ok=True)
POOL_LABELS_PATH = assert_local_only(OUT_DIR / "pool_replication_labels_llama3_8b.csv")
GOLD_LABELS_PATH = assert_local_only(OUT_DIR / "gold_replication_labels_llama3_8b.csv")

LOG_DIR = pathlib.Path.home() / "mimic_local_only" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = assert_local_only(LOG_DIR / f"persist_replication_labels_{time.strftime('%Y%m%dT%H%M%S')}.log")

WS_PATH = pathlib.Path.home() / "mimic_local_only" / "accord_supp" / "accord_weak_supervision_labels_decontam.csv"

SRC_MODULE_PATH = ROOT / "accord" / "differential" / "run_multillm_replication.py"
FACTS_PATH = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
AGG_OUT_PATH = ROOT / "results" / "accord" / "c130_reproducible_student" / "replication_relabel_aggregates.json"
AGG_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

JOIN_KEY = ["HADM_ID", "drug", "rule", "nursing_sentence", "discharge_sentence"]
REGISTERED_MARGINAL_KEY = "overcall.llama3_8b.pool.positive_rate"
MODEL = "llama3:8b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"  # pinned to the loopback IP, per this task's data agreement
EXPECTED_POOL_N = 1679

# ---------------------------------------------------------------------------
# Verbatim prompt/options + parse_prediction + call_ollama, lifted from
# run_multillm_replication.py by static (non-executing) parse.
# ---------------------------------------------------------------------------
def _extract_constant(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"constant {name} not found in {SRC_MODULE_PATH}")


def _extract_function_source(tree, full_source, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            src = ast.get_source_segment(full_source, node)
            if src is None:
                raise AssertionError(f"could not extract source segment for {name}")
            return src
    raise AssertionError(f"function {name} not found in {SRC_MODULE_PATH}")


_full_source = SRC_MODULE_PATH.read_text()
_tree = ast.parse(_full_source)

# Copies below MUST equal the originals -- asserted immediately, not trusted.
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

assert SYSTEM_PROMPT == _extract_constant(_tree, "SYSTEM_PROMPT"), \
    "SYSTEM_PROMPT has drifted from run_multillm_replication.py -- STOP"
assert USER_TEMPLATE == _extract_constant(_tree, "USER_TEMPLATE"), \
    "USER_TEMPLATE has drifted from run_multillm_replication.py -- STOP"
assert TEMPERATURE == _extract_constant(_tree, "TEMPERATURE")
assert SEED == _extract_constant(_tree, "SEED")
assert MAX_TOKENS == _extract_constant(_tree, "MAX_TOKENS")

# parse_prediction and call_ollama: the ORIGINAL function bodies, exec'd into
# a private namespace supplying the same globals they close over. THINKING_MODELS
# is copied verbatim too (llama3:8b is never a member, so it cannot change this
# run's behaviour, but the global must exist for call_ollama's body to run).
THINKING_MODELS = {"qwen3:8b"}
_shared_ns = {
    "requests": requests,
    "time": time,
    "SYSTEM_PROMPT": SYSTEM_PROMPT,
    "USER_TEMPLATE": USER_TEMPLATE,
    "TEMPERATURE": TEMPERATURE,
    "SEED": SEED,
    "MAX_TOKENS": MAX_TOKENS,
    "OLLAMA_URL": OLLAMA_URL,
    "THINKING_MODELS": THINKING_MODELS,
}
exec(_extract_function_source(_tree, _full_source, "parse_prediction"), _shared_ns)  # noqa: S102
exec(_extract_function_source(_tree, _full_source, "call_ollama"), _shared_ns)  # noqa: S102
parse_prediction = _shared_ns["parse_prediction"]
call_ollama = _shared_ns["call_ollama"]


# ---------------------------------------------------------------------------
# Memory-contention guard (belt and suspenders on top of the launcher's own
# wait loop): never fire an Ollama call while C127 is still running.
# ---------------------------------------------------------------------------
def wait_for_c127_to_finish(poll_seconds=30):
    while True:
        result = subprocess.run(["pgrep", "-f", "run_c127"], capture_output=True, text=True)
        if result.returncode != 0:
            return
        print(f"  [guard] run_c127 still running, waiting {poll_seconds}s before any Ollama call")
        time.sleep(poll_seconds)


# ---------------------------------------------------------------------------
# Egress sampler: confirms this process's Ollama connections never leave
# loopback, for the duration this sampler is running.
# ---------------------------------------------------------------------------
_EGRESS_REMOTE_IPS: set[str] = set()


def _sample_egress_once(pid):
    try:
        out = subprocess.run(
            ["lsof", "-p", str(pid), "-i"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return
    for line in out.splitlines()[1:]:
        m = re.search(r"->([0-9a-fA-F.:]+):(\d+)", line)
        if m:
            _EGRESS_REMOTE_IPS.add(m.group(1))


def run_with_egress_sampling(fn, *, sample_every=25):
    """Runs fn(sample_cb) where sample_cb() should be called periodically
    (e.g. once per row) so lsof samples land while a request is in flight."""
    pid = os.getpid()
    counter = {"n": 0}

    def sample_cb():
        counter["n"] += 1
        if counter["n"] % sample_every == 0:
            _sample_egress_once(pid)

    return fn(sample_cb)


# ---------------------------------------------------------------------------
# Row loading (same source, same order as run_multillm_replication.py; see
# module docstring for why reg_df's covariate join can be skipped here).
# ---------------------------------------------------------------------------
def load_pool_rows() -> pd.DataFrame:
    ws = pd.read_csv(WS_PATH)
    ws["HADM_ID"] = ws["HADM_ID"].astype(float)
    ws["nursing_sentence"] = ws["nursing_sentence"].fillna("").astype(str)
    ws["discharge_sentence"] = ws["discharge_sentence"].fillna("").astype(str)
    assert len(ws) == EXPECTED_POOL_N, (
        f"expected {EXPECTED_POOL_N}-row pool, {WS_PATH} now has {len(ws)} rows -- source has drifted, STOP"
    )
    return ws


def load_gold_rows() -> pd.DataFrame:
    from data import load_gold  # accord/control/data.py -- same join run_multillm_replication.py performs inline

    gold_df = load_gold()
    gold_df["nursing_sentence"] = gold_df["nursing_sentence"].fillna("").astype(str)
    gold_df["discharge_sentence"] = gold_df["discharge_sentence"].fillna("").astype(str)
    assert len(gold_df) == 299, f"expected 299 gold pairs, got {len(gold_df)} -- STOP"
    return gold_df


def run_model_on_rows(nursing_list, discharge_list, label, sample_cb):
    n = len(nursing_list)
    preds = []
    t0 = time.time()
    for i in range(n):
        resp = call_ollama(MODEL, nursing_list[i], discharge_list[i])
        preds.append(parse_prediction(resp))
        sample_cb()
        if (i + 1) % 200 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            print(f"    [{MODEL} | {label}] {i + 1}/{n}  ({elapsed:.0f}s elapsed)")
    return preds, time.time() - t0


def summarize(preds):
    n = len(preds)
    n_unparse = sum(1 for p in preds if p == "UNPARSEABLE")
    n_pos = sum(1 for p in preds if p == "contradiction")
    n_neg = sum(1 for p in preds if p == "not_contradiction")
    parseable = n_pos + n_neg
    pos_rate = (n_pos / parseable) if parseable else None
    return {
        "n": n, "n_unparseable": n_unparse, "n_positive": n_pos, "n_negative": n_neg,
        "n_parseable": parseable, "positive_rate_of_parseable": pos_rate,
    }


def main():
    with LOG_PATH.open("w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
        print(f"persist_replication_labels starting {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"pool labels -> {POOL_LABELS_PATH}")
        print(f"gold labels -> {GOLD_LABELS_PATH}")

        ws = load_pool_rows()
        gold_df = load_gold_rows()
        print(f"loaded pool: {len(ws)} rows; gold: {len(gold_df)} rows")

        wait_for_c127_to_finish()

        pid = os.getpid()
        print(f"this process pid={pid}; sampling lsof -p {pid} -i for egress during inference")

        def _run(sample_cb):
            pool_preds, pool_seconds = run_model_on_rows(
                ws["nursing_sentence"].tolist(), ws["discharge_sentence"].tolist(),
                "candidate pool", sample_cb)
            gold_preds, gold_seconds = run_model_on_rows(
                gold_df["nursing_sentence"].tolist(), gold_df["discharge_sentence"].tolist(),
                "gold set", sample_cb)
            return pool_preds, pool_seconds, gold_preds, gold_seconds

        pool_preds, pool_seconds, gold_preds, gold_seconds = run_with_egress_sampling(_run)
        _sample_egress_once(pid)  # one last sample after the final request

        egress_observed = sorted(_EGRESS_REMOTE_IPS)
        egress_local_only = bool(egress_observed) and set(egress_observed).issubset({"127.0.0.1"})
        print(f"egress check: observed remote IPs during Ollama calls = {egress_observed}; "
              f"local_only = {egress_local_only}")

        pool_out = ws[JOIN_KEY].copy()
        pool_out["replication_label"] = pool_preds
        pool_out.to_csv(POOL_LABELS_PATH, index=False)
        print(f"wrote {len(pool_out)} pool rows to {POOL_LABELS_PATH}")

        gold_out = gold_df[JOIN_KEY].copy()
        gold_out["replication_label"] = gold_preds
        gold_out.to_csv(GOLD_LABELS_PATH, index=False)
        print(f"wrote {len(gold_out)} gold rows to {GOLD_LABELS_PATH}")

        pool_summary = summarize(pool_preds)
        gold_summary = summarize(gold_preds)

        facts = json.loads(FACTS_PATH.read_text())
        registered_rate = facts.get(REGISTERED_MARGINAL_KEY, {}).get("value")
        computed_rate = pool_summary["positive_rate_of_parseable"]
        rounds_match = (
            computed_rate is not None and registered_rate is not None
            and round(computed_rate, 4) == round(float(registered_rate), 4)
        )
        print(f"marginal-match check: computed={computed_rate}  registered({REGISTERED_MARGINAL_KEY})="
              f"{registered_rate}  rounds_match_4dp={rounds_match}")

        aggregates = {
            "model": MODEL,
            "prompt_source": "run_multillm_replication.py, statically re-parsed and asserted verbatim",
            "temperature": TEMPERATURE, "seed": SEED, "num_predict": MAX_TOKENS,
            "pool": pool_summary,
            "gold": gold_summary,
            "marginal_match": {
                "computed_pool_positive_rate": computed_rate,
                "registered_pool_positive_rate": registered_rate,
                "registered_key": REGISTERED_MARGINAL_KEY,
                "rounds_match_4dp": rounds_match,
            },
            "ollama": {
                "url": OLLAMA_URL,
                "egress_observed_remote_ips": egress_observed,
                "egress_local_only": egress_local_only,
            },
            "wall_time_seconds": {"pool": pool_seconds, "gold": gold_seconds},
            "pool_labels_path": str(POOL_LABELS_PATH),
            "gold_labels_path": str(GOLD_LABELS_PATH),
            "outcome": None if rounds_match else "C",
            "outcome_reason": None if rounds_match else (
                f"relabelled pool positive rate {computed_rate} does not match the registered "
                f"{REGISTERED_MARGINAL_KEY}={registered_rate} to 4 decimals -- per "
                "C130_PREREGISTRATION.md this is Outcome C (invalid run); training must not proceed"
            ),
        }
        AGG_OUT_PATH.write_text(json.dumps(aggregates, indent=2) + "\n")
        print(f"wrote aggregates to {AGG_OUT_PATH}")
        print("persist_replication_labels finished")

    # Only counts/booleans/paths -- never a join-key value -- reach real stdout.
    print(json.dumps({
        "pool": aggregates["pool"], "gold": aggregates["gold"],
        "marginal_match": aggregates["marginal_match"],
        "egress_local_only": aggregates["ollama"]["egress_local_only"],
        "wall_time_seconds": aggregates["wall_time_seconds"],
        "outcome": aggregates["outcome"],
    }, indent=2))
    print(f"log: {LOG_PATH}")

    if aggregates["outcome"] == "C":
        print(f"ABORT (Outcome C): {aggregates['outcome_reason']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
