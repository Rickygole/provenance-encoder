"""C138 step 1: graded teacher scores for the weak-label pool.

Scores every row of accord_weak_supervision_labels_decontam.csv (1,679 rows)
with the single configuration fixed in C138_PREREGISTRATION.md section 2:
granite3.3:8b, prompt P_rev_agree, decoding identical to
accord/prompt_sensitivity/run_c109.py::call_chat. The prompt dict and the
scoring function are IMPORTED from the C109 modules, never retyped, so the
pool scores come from byte-identical wording and an identical parse to the
gold-set cells that produced AUROC 0.7936 on G253.

The per-pair graded caches under ~/mimic_local_only/accord_prompt_sensitivity_cache/
cover only the 299 gold pairs, and ~/mimic_local_only/accord_multillm/ holds
only HARD llama3:8b pool labels (pool_replication_labels_llama3_8b.csv, a
replication_label column, no graded score). No pool-level graded score existed
before this script.

Local only. Loopback Ollama, egress evidence captured before the first call.
The cache holds a pool row index and numeric fields only: never note text,
never HADM_ID, never SUBJECT_ID. Resumable: a row already in the cache is not
re-called.

  python3 accord/c138_prevalence_matched/score_pool_granite.py --dry-run
  python3 accord/c138_prevalence_matched/score_pool_granite.py
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accord.cloud_paths import assert_local_only  # noqa: E402
from accord.prompt_sensitivity.prompts_c109 import PROMPTS_C109  # noqa: E402
from accord.prompt_sensitivity.run_c109 import (  # noqa: E402
    NUM_CTX, assert_loopback_host, call_chat, parse_and_score,
    write_egress_evidence,
)

MODEL = "granite3.3:8b"
PROMPT_NAME = "P_rev_agree"

MIMIC_LOCAL_ONLY_DIR = pathlib.Path(
    os.environ.get("MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
WS_PATH = MIMIC_LOCAL_ONLY_DIR / "accord_supp" / "accord_weak_supervision_labels_decontam.csv"
CACHE_DIR = MIMIC_LOCAL_ONLY_DIR / "accord_c138_cache"
CACHE_PATH = CACHE_DIR / "pool_granite33_8b_nothink__P_rev_agree.jsonl"
LOG_DIR = MIMIC_LOCAL_ONLY_DIR / "logs"
EGRESS_DIR = ROOT / "results" / "accord" / "c138_prevalence_matched"
HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")


def load_cache(path: pathlib.Path) -> dict:
    done = {}
    if path.is_file():
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[int(rec["ws_row"])] = rec
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows, smoke test only")
    ap.add_argument("--dry-run", action="store_true",
                    help="invented sentences, mocked call, never reads "
                         "mimic_local_only, writes to a temp dir")
    args = ap.parse_args()

    cfg = PROMPTS_C109[PROMPT_NAME]
    assert cfg["reversed"] is True, "P_rev_agree must be a reversed-wording prompt"
    run_tag = time.strftime("c138_score_%Y%m%dT%H%M%S")

    if args.dry_run:
        import tempfile
        from accord.prompt_sensitivity.dry_run_fixtures import INVENTED_GOLD
        from accord.prompt_sensitivity.run_c109 import mock_call_chat
        out_root = pathlib.Path(tempfile.mkdtemp(prefix="c138_dry_run_"))
        cache_path = out_root / "cache.jsonl"
        log_path = out_root / "run.log"
        rows = [(i, n, d) for i, (pid, n, d, _a, _b) in enumerate(INVENTED_GOLD)]
        caller = lambda system, user, ws_row, log_fh: mock_call_chat(  # noqa: E731
            MODEL, system, user, str(ws_row), cfg["pos_tokens"], cfg["neg_tokens"],
            log_fh=log_fh)
        print(f"--dry-run: {len(rows)} INVENTED rows, mocked Ollama call, "
              f"no mimic_local_only access. Writing under {out_root}")
    else:
        assert_loopback_host(HOST)
        cache_path = assert_local_only(CACHE_PATH)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        EGRESS_DIR.mkdir(parents=True, exist_ok=True)
        write_egress_evidence(run_tag, EGRESS_DIR)
        log_path = LOG_DIR / f"{run_tag}.log"
        ws = pd.read_csv(WS_PATH)
        ws["nursing_sentence"] = ws["nursing_sentence"].fillna("").astype(str)
        ws["discharge_sentence"] = ws["discharge_sentence"].fillna("").astype(str)
        rows = list(zip(range(len(ws)), ws["nursing_sentence"], ws["discharge_sentence"]))
        caller = lambda system, user, ws_row, log_fh: call_chat(  # noqa: E731
            MODEL, system, user, log_fh=log_fh)
        print(f"C138 pool scoring: {len(rows)} rows, model={MODEL}, "
              f"prompt={PROMPT_NAME}, num_ctx={NUM_CTX}")

    if args.limit:
        rows = rows[:args.limit]

    done = load_cache(cache_path)
    todo = [r for r in rows if r[0] not in done]
    print(f"cached={len(done)} to_call={len(todo)}")

    t0 = time.time()
    with open(log_path, "a") as log_fh, open(cache_path, "a") as cache_fh:
        for i, (ws_row, nursing, discharge) in enumerate(todo):
            user = cfg["user_template"].format(
                nursing_note=nursing, discharge_summary=discharge)
            reply = caller(cfg["system"], user, ws_row, log_fh)
            if reply is None:
                rec = {"ws_row": ws_row, "flag_label": None, "flag_logodds": None,
                       "degenerate": True, "parse_failure": True, "call_failed": True}
            else:
                raw_label, raw_logodds, degenerate, parse_fail = parse_and_score(
                    reply, cfg["pos_tokens"], cfg["neg_tokens"])
                # reversed prompt: the sign flip lives here only, as in run_c109.py
                rec = {
                    "ws_row": ws_row,
                    "flag_label": (1 - raw_label) if raw_label is not None else None,
                    "flag_logodds": -raw_logodds,
                    "degenerate": degenerate,
                    "parse_failure": parse_fail,
                    "call_failed": False,
                    "thinking_nonempty": reply["thinking_nonempty"],
                    "truncated": (reply["prompt_eval_count"] is not None
                                  and reply["prompt_eval_count"] >= NUM_CTX),
                }
            cache_fh.write(json.dumps(rec) + "\n")
            cache_fh.flush()
            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{len(todo)}  {rate:.2f} rows/s  "
                      f"eta {(len(todo) - i - 1) / rate / 60:.1f} min", flush=True)

    done = load_cache(cache_path)
    n = len(rows)
    sub = [done[r[0]] for r in rows if r[0] in done]
    n_fail = sum(1 for r in sub if r.get("call_failed"))
    n_parse = sum(1 for r in sub if r.get("parse_failure"))
    n_degen = sum(1 for r in sub if r.get("degenerate"))
    n_flag = sum(1 for r in sub if r.get("flag_label") == 1)
    n_trunc = sum(1 for r in sub if r.get("truncated"))
    distinct = len({round(r["flag_logodds"], 12) for r in sub
                    if r.get("flag_logodds") is not None})
    print(json.dumps({
        "model": MODEL, "prompt": PROMPT_NAME, "n_rows": n, "n_scored": len(sub),
        "n_call_failed": n_fail, "n_parse_failures": n_parse,
        "parse_failure_rate": round(n_parse / len(sub), 4) if sub else None,
        "n_degenerate": n_degen, "n_truncated": n_trunc,
        "n_flagged": n_flag,
        "hard_positive_rate": round(n_flag / len(sub), 4) if sub else None,
        "n_distinct_logodds": distinct,
        "wall_time_seconds": round(time.time() - t0, 1),
    }, indent=2))
    if args.dry_run:
        print(f"--dry-run artifacts under {log_path.parent}")


if __name__ == "__main__":
    main()
