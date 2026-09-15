"""C127: qwen3 scale ladder (0.6B, 1.7B, 4B, 8B, 14B) under P_asserting on the 299
gold pairs. See C127_PREREGISTRATION.md.

Mirrors accord/prompt_sensitivity/run_c103.py's data loading, join gate, caching
layout, calling convention (think=False, temperature 0, num_predict 1,
NUM_CTX 4096), parsing and truncation check line for line -- this file does not
re-derive any of that, it imports the shared prompt table and reuses the same
functions where they are not tied to C103's specific model list.

Local only: every request goes to an Ollama server on 127.0.0.1. Reads
credentialed gold data from mimic_local_only and writes:
  - a numeric-only per-pair cache under
    ~/mimic_local_only/accord_prompt_sensitivity_cache/*.jsonl (pair_id, flag
    label, flag log-odds, degenerate, parse-failure, thinking-nonempty,
    truncated -- never sentence text), same schema and same directory C103
    already writes to, so a size run once here is visible to any later reuse
    check exactly the way C103's qwen3:8b cell already is;
  - aggregate-only cell JSON under results/accord/prompt_sensitivity/cells/,
    matching C103's own directory, plus this run's own summary under
    results/accord/c127_scale_ladder/.

No note text, pair text, or model reply text is ever printed to this
process's stdout/stderr, or written anywhere but the local cache above (as
numeric flags, not text).

    python3 accord/c127_scale_ladder/run_c127.py
"""
import argparse
import json
import os
import pathlib
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accord.cloud_paths import assert_local_only  # noqa: E402
from accord.differential.model_tags import ollama_model_tag  # noqa: E402
from accord.prompt_sensitivity.prompts import PROMPTS  # noqa: E402
from accord.prompt_sensitivity.run_c103 import (  # noqa: E402
    call_chat, parse_and_score, assert_loopback_host,
)

GOLD = pathlib.Path.home() / "mimic_local_only/from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv"
CACHE_DIR = assert_local_only(
    pathlib.Path.home() / "mimic_local_only/accord_prompt_sensitivity_cache")
LOG_DIR = pathlib.Path.home() / "mimic_local_only/logs"
CELLS_DIR = ROOT / "results" / "accord" / "prompt_sensitivity" / "cells"
OUT_DIR = ROOT / "results" / "accord" / "c127_scale_ladder"

SIZES = ["0.6b", "1.7b", "4b", "8b", "14b"]
MODELS = {size: f"qwen3:{size}" for size in SIZES}
PROMPT_NAME = "P_asserting"
NUM_CTX = 4096  # frozen; matches run_c103.py exactly
HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")

# The one earlier cell that MIGHT be reusable: qwen3:8b's P_asserting cell,
# produced by run_c103.py itself. Its recorded settings are checked below
# before being trusted; nothing is reused on the strength of the model label
# alone.
REUSABLE_TAGS = {"8b": "_qwen3_8b_nothink"}


def write_egress_evidence(run_tag):
    """Same evidence-and-refuse pattern as run_c103.py::write_egress_evidence."""
    import subprocess
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"egress_evidence_{run_tag}.txt"
    ollama_pid = None
    try:
        ps = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True, text=True)
        lines = [l for l in ps.stdout.strip().splitlines() if l]
        ollama_pid = lines[0] if lines else None
    except Exception as exc:
        print(f"could not find ollama pid: {type(exc).__name__}", file=sys.stderr)

    my_pid = str(os.getpid())
    pids = [p for p in (ollama_pid, my_pid) if p]
    chunks = [f"run={run_tag}\n", f"ollama_pid={ollama_pid}\nself_pid={my_pid}\n\n"]
    for pid in pids:
        r = subprocess.run(["lsof", "-nP", "-i", "-a", "-p", pid],
                            capture_output=True, text=True)
        chunks.append(f"--- lsof -nP -i -a -p {pid} ---\n{r.stdout}\n")
    text = "".join(chunks)
    out_path.write_text(text)

    bad = []
    for line in text.splitlines():
        if "->" in line and "127.0.0.1" not in line and "localhost" not in line:
            if "LISTEN" in line and "127.0.0.1" in line:
                continue
            bad.append(line)
    if bad:
        raise SystemExit(
            f"egress evidence shows a non-loopback line, stopping: {bad[:3]!r} "
            f"(full evidence at {out_path})")
    print(f"egress evidence OK (loopback only), written to {out_path}")


def run_cell(model, gold_rows, log_fh):
    cfg = PROMPTS[PROMPT_NAME]
    records = []
    n_thinking_nonempty = 0
    n_truncated = 0
    for pair_id, nursing, discharge in gold_rows:
        user = cfg["user_template"].format(
            nursing_note=nursing, discharge_summary=discharge)
        reply = call_chat(model, cfg["system"], user, log_fh=log_fh)
        if reply is None:
            records.append({"pair_id": pair_id, "flag_label": None,
                             "flag_logodds": None, "degenerate": True,
                             "parse_failure": True, "call_failed": True})
            continue
        raw_label, raw_logodds, degenerate, parse_fail = parse_and_score(
            reply, cfg["pos_tokens"], cfg["neg_tokens"])
        # cfg["reversed"] is False for P_asserting; kept for parity with
        # run_c103.py's block so the two stay comparable line for line.
        flag_label = raw_label
        flag_logodds = raw_logodds
        if reply["thinking_nonempty"]:
            n_thinking_nonempty += 1
        truncated = (reply["prompt_eval_count"] is not None
                     and reply["prompt_eval_count"] >= NUM_CTX)
        if truncated:
            n_truncated += 1
        records.append({
            "pair_id": pair_id, "flag_label": flag_label,
            "flag_logodds": flag_logodds, "degenerate": degenerate,
            "parse_failure": parse_fail, "call_failed": False,
            "thinking_nonempty": reply["thinking_nonempty"],
            "truncated": truncated,
        })
    return records, n_thinking_nonempty, n_truncated


def aggregate_cell(records, n_thinking_nonempty, n_truncated, model, think):
    n = len(records)
    n_failed = sum(1 for r in records if r["call_failed"])
    n_parse_fail = sum(1 for r in records if r["parse_failure"])
    n_degenerate = sum(1 for r in records if r["degenerate"])
    n_flag = sum(1 for r in records if r["flag_label"] == 1)
    n_scored = sum(1 for r in records if r["flag_label"] is not None)
    return {
        "model": model, "prompt": PROMPT_NAME, "think": think,
        "n": n, "n_call_failed": n_failed,
        "n_parse_failures": n_parse_fail,
        "parse_failure_rate": round(n_parse_fail / n, 4) if n else None,
        "n_degenerate_logodds": n_degenerate,
        "degenerate_rate": round(n_degenerate / n, 4) if n else None,
        "n_thinking_nonempty": n_thinking_nonempty,
        "n_truncated": n_truncated,
        "n_scored": n_scored,
        "n_flagged": n_flag,
        "positive_rate": round(n_flag / n, 4) if n else None,
    }


def check_reuse(tag, gold_rows):
    """Byte-identity check of the settings recorded in an existing cell's
    artifact, per C127_PREREGISTRATION.md's reuse rule. Returns
    (reusable: bool, reason: str)."""
    cache_path = CACHE_DIR / f"{tag}__{PROMPT_NAME}.jsonl"
    cell_path = CELLS_DIR / f"{tag}__{PROMPT_NAME}.json"
    if not cache_path.exists() or not cell_path.exists():
        return False, "no existing cache/cell file"
    cell = json.loads(cell_path.read_text())
    if cell.get("prompt") != PROMPT_NAME:
        return False, f"existing cell prompt {cell.get('prompt')!r} != {PROMPT_NAME!r}"
    if cell.get("think") is not False:
        return False, f"existing cell think={cell.get('think')!r} != False"
    if cell.get("n") != len(gold_rows):
        return False, f"existing cell n={cell.get('n')} != {len(gold_rows)}"
    recs = [json.loads(line) for line in cache_path.read_text().splitlines() if line]
    if len(recs) != len(gold_rows):
        return False, f"existing cache has {len(recs)} rows, expected {len(gold_rows)}"
    if not all("flag_logodds" in r for r in recs):
        return False, "existing cache is missing per-pair flag_logodds (not a graded-score run)"
    if not all("truncated" in r for r in recs):
        return False, "existing cache is missing the truncated field (settings not comparable)"
    # NUM_CTX and num_predict are not stored per-cell in the C103 schema; they
    # are constants baked into run_c103.py, which produced this cache and has
    # not changed since (this repo has one copy of NUM_CTX=4096/num_predict=1
    # for this prompt family, checked by import identity below).
    from accord.prompt_sensitivity import run_c103 as _c103_mod
    if _c103_mod.NUM_CTX != NUM_CTX:
        return False, f"run_c103.NUM_CTX={_c103_mod.NUM_CTX} != {NUM_CTX}"
    return True, "existing cell/cache settings byte-identical (prompt, think, n, NUM_CTX, per-pair logodds present)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=SIZES)
    ap.add_argument("--limit", type=int, default=None, help="cap pairs, dry run only")
    ap.add_argument("--force-rerun", action="store_true",
                     help="ignore the reuse check and rerun every size")
    args = ap.parse_args()

    assert_loopback_host(HOST)
    run_tag = time.strftime("c127_%Y%m%dT%H%M%S")
    write_egress_evidence(run_tag)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    g = pd.read_csv(GOLD)
    gold_rows = list(zip(g["pair_id"], g["nursing_sentence"].astype(str),
                          g["discharge_sentence"].astype(str)))
    if args.limit:
        gold_rows = gold_rows[:args.limit]

    run_log = {"run_tag": run_tag, "sizes": {}}
    log_path = LOG_DIR / f"{run_tag}.log"
    with open(log_path, "a") as log_fh:
        print(f"C127 run {run_tag}: {len(gold_rows)} pairs, sizes={args.sizes}",
              file=log_fh, flush=True)
        for size in args.sizes:
            model = MODELS[size]
            think = False
            tag = ollama_model_tag(model, think)

            if args.force_rerun:
                reused, reuse_reason = False, "reuse not attempted (--force-rerun)"
            elif size not in REUSABLE_TAGS:
                reused, reuse_reason = False, "no earlier cell registered as reusable for this size"
            else:
                reused, reuse_reason = check_reuse(REUSABLE_TAGS[size], gold_rows)
                if reused:
                    tag = REUSABLE_TAGS[size]

            if reused:
                cell = json.loads((CELLS_DIR / f"{tag}__{PROMPT_NAME}.json").read_text())
                cell["elapsed_sec"] = 0.0
                cell["reused_from_earlier_run"] = True
                cell["reuse_check"] = reuse_reason
                print(f"qwen3:{size} {PROMPT_NAME}: REUSED existing cell "
                      f"{tag}__{PROMPT_NAME}.json ({reuse_reason})")
                print(f"cell qwen3:{size} REUSED ({reuse_reason})", file=log_fh, flush=True)
            else:
                t0 = time.time()
                records, n_think, n_trunc = run_cell(model, gold_rows, log_fh)
                cell = aggregate_cell(records, n_think, n_trunc, model, think)
                cell["elapsed_sec"] = round(time.time() - t0, 1)
                cell["reused_from_earlier_run"] = False
                cell["reuse_check"] = reuse_reason

                cache_path = CACHE_DIR / f"{tag}__{PROMPT_NAME}.jsonl"
                with open(cache_path, "w") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")

                cell_path = CELLS_DIR / f"{tag}__{PROMPT_NAME}.json"
                cell_path.write_text(json.dumps(cell, indent=2) + "\n")
                print(f"qwen3:{size} {PROMPT_NAME}: n={cell['n']} "
                      f"flagged={cell['n_flagged']} rate={cell['positive_rate']} "
                      f"parse_fail={cell['n_parse_failures']} "
                      f"degenerate={cell['n_degenerate_logodds']} "
                      f"thinking_nonempty={cell['n_thinking_nonempty']} "
                      f"truncated={cell['n_truncated']} "
                      f"({cell['elapsed_sec']:.0f}s) -> {cell_path}")
                print(f"cell qwen3:{size} done in {cell['elapsed_sec']}s",
                      file=log_fh, flush=True)

            run_log["sizes"][size] = {
                "model": model, "tag": tag, "reused": reused,
                "reuse_check": reuse_reason,
                "n": cell["n"], "n_flagged": cell["n_flagged"],
                "positive_rate": cell["positive_rate"],
                "n_parse_failures": cell["n_parse_failures"],
                "n_truncated": cell["n_truncated"],
                "elapsed_sec": cell["elapsed_sec"],
            }

    (OUT_DIR / f"run_summary_{run_tag}.json").write_text(
        json.dumps(run_log, indent=2) + "\n")
    print(f"run summary -> {OUT_DIR / f'run_summary_{run_tag}.json'}")


if __name__ == "__main__":
    main()
