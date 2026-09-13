"""C103: run P_asserting / P_neutral / P_reversed against llama3:8b,
granite3.3:8b and qwen3:8b on the 299 gold pairs. See C103_PREREGISTRATION.md.

Local only: every request goes to an Ollama server on 127.0.0.1. Reads
credentialed gold data from mimic_local_only and writes:
  - a numeric-only per-pair cache under
    ~/mimic_local_only/accord_prompt_sensitivity_cache/*.jsonl (pair_id,
    flag label, flag log-odds, degenerate, parse-failure, thinking-nonempty,
    truncated -- never sentence text)
  - aggregate-only cell JSON under results/accord/prompt_sensitivity/cells/

No note text, pair text, or model reply text is ever printed to this
process's stdout/stderr, or written anywhere but the local cache above (as
numeric flags, not text).
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

GOLD = pathlib.Path.home() / "mimic_local_only/from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv"
CACHE_DIR = assert_local_only(
    pathlib.Path.home() / "mimic_local_only/accord_prompt_sensitivity_cache")
LOG_DIR = pathlib.Path.home() / "mimic_local_only/logs"
CELLS_DIR = ROOT / "results" / "accord" / "prompt_sensitivity" / "cells"
EGRESS_DIR = ROOT / "results" / "accord" / "prompt_sensitivity"

MODELS = ["llama3:8b", "granite3.3:8b", "qwen3:8b"]
PROMPT_NAMES = ["P_asserting", "P_neutral", "P_reversed"]
NUM_CTX = 4096  # committed once; a truncation heuristic compares against this
HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")

_LOOPBACK_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}


def assert_loopback_host(host):
    hostname = urllib.parse.urlparse(host).hostname
    if hostname not in _LOOPBACK_HOSTNAMES and not (
            hostname and hostname.startswith("127.")):
        sys.exit(f"OLLAMA_HOST {host!r} is not loopback; refusing to run")


def write_egress_evidence(run_tag):
    """lsof for the ollama serve process and this process, before any call."""
    import subprocess
    EGRESS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EGRESS_DIR / f"egress_evidence_{run_tag}.txt"
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
            # a real connection line, not loopback
            if "LISTEN" in line and "127.0.0.1" in line:
                continue
            bad.append(line)
    if bad:
        raise SystemExit(
            f"egress evidence shows a non-loopback line, stopping: {bad[:3]!r} "
            f"(full evidence at {out_path})")
    print(f"egress evidence OK (loopback only), written to {out_path}")


def call_chat(model, system, user, retries=4, log_fh=None):
    """One /api/chat call. think=False, temperature 0, num_predict 1,
    top_logprobs 8, logprobs true -- same shape graded_score_probe.py uses.
    Returns dict(content, top_logprobs, thinking_nonempty, prompt_eval_count)
    or None on failure after retries. Never raises with the request/response
    body attached; only the exception type name is logged, and only to the
    local mimic_local_only log file, never to this process's own stdout.
    """
    body = {
        "model": model, "stream": False, "logprobs": True, "top_logprobs": 8,
        "think": False,
        "options": {"temperature": 0, "num_predict": 1, "num_ctx": NUM_CTX},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        f"{HOST}/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            msg = r.get("message") or {}
            thinking = msg.get("thinking")
            lp = r.get("logprobs") or []
            top = (lp[0].get("top_logprobs") if lp else []) or []
            return {
                "content": msg.get("content") or "",
                "top_logprobs": top,
                "thinking_nonempty": bool(thinking and thinking.strip()),
                "prompt_eval_count": r.get("prompt_eval_count"),
            }
        except Exception as exc:
            if log_fh is not None:
                print(f"attempt {attempt + 1}/{retries} failed: "
                      f"{type(exc).__name__}", file=log_fh, flush=True)
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def parse_and_score(reply, pos_tokens, neg_tokens):
    """Hard label from the actual reply's first word, and log-odds from the
    top-8 first-token logprobs. Returns (raw_label, raw_logodds, degenerate,
    parse_failure)."""
    content = (reply["content"] or "").strip()
    first = content.split()[0].strip(".,:;!?").lower() if content else ""
    pos_lower = {t.lower() for t in pos_tokens}
    neg_lower = {t.lower() for t in neg_tokens}
    if first in pos_lower:
        raw_label = 1
    elif first in neg_lower:
        raw_label = 0
    else:
        raw_label = None

    p = n_ = None
    for t in reply["top_logprobs"]:
        tok = (t.get("token") or "").strip()
        if p is None and tok in pos_tokens:
            p = t["logprob"]
        if n_ is None and tok in neg_tokens:
            n_ = t["logprob"]
    degenerate = p is None or n_ is None
    raw_logodds = (p if p is not None else -20.0) - (n_ if n_ is not None else -20.0)
    return raw_label, raw_logodds, degenerate, raw_label is None


def run_cell(model, prompt_name, gold_rows, log_fh):
    cfg = PROMPTS[prompt_name]
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
        if cfg["reversed"]:
            flag_label = (1 - raw_label) if raw_label is not None else None
            flag_logodds = -raw_logodds
        else:
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


def aggregate_cell(records, n_thinking_nonempty, n_truncated, model, prompt_name, think):
    n = len(records)
    n_failed = sum(1 for r in records if r["call_failed"])
    n_parse_fail = sum(1 for r in records if r["parse_failure"])
    n_degenerate = sum(1 for r in records if r["degenerate"])
    n_flag = sum(1 for r in records if r["flag_label"] == 1)
    n_scored = sum(1 for r in records if r["flag_label"] is not None)
    return {
        "model": model, "prompt": prompt_name, "think": think,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="cap pairs per cell, for a dry run only")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--prompts", nargs="+", default=PROMPT_NAMES)
    args = ap.parse_args()

    assert_loopback_host(HOST)
    run_tag = time.strftime("c103_%Y%m%dT%H%M%S")
    write_egress_evidence(run_tag)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CELLS_DIR.mkdir(parents=True, exist_ok=True)

    g = pd.read_csv(GOLD)
    gold_rows = list(zip(g["pair_id"], g["nursing_sentence"].astype(str),
                          g["discharge_sentence"].astype(str)))
    if args.limit:
        gold_rows = gold_rows[:args.limit]

    log_path = LOG_DIR / f"{run_tag}.log"
    with open(log_path, "a") as log_fh:
        print(f"C103 run {run_tag}: {len(gold_rows)} pairs, "
              f"{len(args.models)} models x {len(args.prompts)} prompts",
              file=log_fh, flush=True)
        for model in args.models:
            tag = ollama_model_tag(model, False)
            for prompt_name in args.prompts:
                t0 = time.time()
                records, n_think, n_trunc = run_cell(model, prompt_name, gold_rows, log_fh)
                cell = aggregate_cell(records, n_think, n_trunc, model, prompt_name, False)
                cell["elapsed_sec"] = round(time.time() - t0, 1)

                cache_path = CACHE_DIR / f"{tag}__{prompt_name}.jsonl"
                with open(cache_path, "w") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")

                cell_path = CELLS_DIR / f"{tag}__{prompt_name}.json"
                cell_path.write_text(json.dumps(cell, indent=2) + "\n")
                print(f"{model} {prompt_name}: n={cell['n']} "
                      f"flagged={cell['n_flagged']} rate={cell['positive_rate']} "
                      f"parse_fail={cell['n_parse_failures']} "
                      f"degenerate={cell['n_degenerate_logodds']} "
                      f"thinking_nonempty={cell['n_thinking_nonempty']} "
                      f"truncated={cell['n_truncated']} "
                      f"({cell['elapsed_sec']:.0f}s) -> {cell_path}")
                print(f"cell {model}/{prompt_name} done in {cell['elapsed_sec']}s",
                      file=log_fh, flush=True)


if __name__ == "__main__":
    main()
