"""C109: run three NEW reversed-question paraphrases, plus the original
P_reversed re-run as a determinism check, against llama3:8b and
granite3.3:8b on the 299 gold pairs. See C109_PREREGISTRATION.md.

Reuses accord/prompt_sensitivity/run_c103.py's conventions exactly:
  - Local only: every request goes to an Ollama server on 127.0.0.1.
  - Per-pair numeric-only cache under
    ~/mimic_local_only/accord_prompt_sensitivity_cache/*.jsonl (pair_id,
    flag label, flag log-odds, degenerate, parse-failure, thinking-nonempty,
    truncated -- never sentence text). Same directory C103 used; C109's
    prompt names (P_rev_agree, P_rev_follow, P_rev_compatible,
    P_reversed_rerun) are new filenames, so nothing there is overwritten.
  - Aggregate-only cell JSON under
    results/accord/prompt_sensitivity/c109/cells/.
  - Full stdout/stderr of any process touching ~/mimic_local_only redirected
    by the caller to ~/mimic_local_only/logs/<run>.log.
  - lsof egress evidence for the ollama serve process and this process,
    captured before the first call.
  - No `--diagnose`, no raw model replies stored, exception type names only.

No note text, pair text, or model reply text is ever printed to this
process's stdout/stderr, or written anywhere but the local cache above (as
numeric flags, not text).

--dry-run: uses INVENTED sentence pairs authored in this file (never reads
~/mimic_local_only) and mocks the Ollama call (no network call, no model
loaded). This is the only mode allowed to run without a live Ollama server;
it exists to prove the harness end-to-end before the coordinator runs the
real cells.
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
from accord.prompt_sensitivity.dry_run_fixtures import INVENTED_GOLD  # noqa: E402
from accord.prompt_sensitivity.prompts_c109 import (  # noqa: E402
    ALL_PROMPT_NAMES, PROMPTS_C109,
)

MIMIC_LOCAL_ONLY_DIR = pathlib.Path(os.environ.get("MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
GOLD = MIMIC_LOCAL_ONLY_DIR / "from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv"
MODELS = ["llama3:8b", "granite3.3:8b"]
NUM_CTX = 4096  # committed once; a truncation heuristic compares against this
HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")

_LOOPBACK_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}

# Real-run paths (mimic_local_only / results); never used in --dry-run.
REAL_CACHE_DIR = MIMIC_LOCAL_ONLY_DIR / "accord_prompt_sensitivity_cache"
REAL_LOG_DIR = MIMIC_LOCAL_ONLY_DIR / "logs"
REAL_CELLS_DIR = ROOT / "results" / "accord" / "prompt_sensitivity" / "c109" / "cells"
REAL_EGRESS_DIR = ROOT / "results" / "accord" / "prompt_sensitivity" / "c109"
REAL_ATTEMPTS_PATH = REAL_EGRESS_DIR / "attempts.json"


def assert_loopback_host(host):
    hostname = urllib.parse.urlparse(host).hostname
    if hostname not in _LOOPBACK_HOSTNAMES and not (
            hostname and hostname.startswith("127.")):
        sys.exit(f"OLLAMA_HOST {host!r} is not loopback; refusing to run")


def write_egress_evidence(run_tag, egress_dir):
    """lsof for the ollama serve process and this process, before any call."""
    import subprocess
    egress_dir.mkdir(parents=True, exist_ok=True)
    out_path = egress_dir / f"egress_evidence_{run_tag}.txt"
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


def call_chat(model, system, user, retries=4, log_fh=None):
    """One /api/chat call. think=False, temperature 0, num_predict 1,
    top_logprobs 8, logprobs true -- same shape run_c103.py / C103 uses.
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


def mock_call_chat(model, system, user, pair_id, pos_tokens, neg_tokens, retries=4, log_fh=None):
    """--dry-run stand-in for call_chat. No network call, no model loaded.

    Deterministic in (model, pair_id) ONLY -- not in the prompt wording --
    so that P_reversed and P_reversed_rerun, which differ only in wording,
    produce byte-identical output for the same pair_id, exercising the
    determinism check's pass path honestly rather than by construction of
    the check itself.
    """
    h = (hash((model, pair_id)) % 5)
    is_yes = h < 3  # deterministic 3-of-5 split, no randomness
    pos_tok = sorted(pos_tokens)[0]
    neg_tok = sorted(neg_tokens)[0]
    content = pos_tok if is_yes else neg_tok
    # Synthetic top-8 logprobs: the chosen token gets -0.1, the other -2.0.
    top = [{"token": pos_tok, "logprob": -0.1 if is_yes else -2.0},
           {"token": neg_tok, "logprob": -2.0 if is_yes else -0.1}]
    return {"content": content, "top_logprobs": top,
            "thinking_nonempty": False, "prompt_eval_count": 50}


def parse_and_score(reply, pos_tokens, neg_tokens):
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


def run_cell(model, prompt_name, gold_rows, log_fh, dry_run):
    cfg = PROMPTS_C109[prompt_name]
    records = []
    n_thinking_nonempty = 0
    n_truncated = 0
    for pair_id, nursing, discharge in gold_rows:
        user = cfg["user_template"].format(
            nursing_note=nursing, discharge_summary=discharge)
        if dry_run:
            reply = mock_call_chat(model, cfg["system"], user, pair_id,
                                    cfg["pos_tokens"], cfg["neg_tokens"],
                                    log_fh=log_fh)
        else:
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


def load_attempts(path):
    if path.exists():
        return json.loads(path.read_text())
    return {}


def bump_attempts(path, cell_key):
    """Kill-condition bookkeeping: counts, never resumed, from the first
    pair. Returns the attempt number (1 = first start). A cell started a
    3rd time (i.e. killed twice already) is refused and flagged invalid,
    per C109_PREREGISTRATION.md's Outcome C."""
    attempts = load_attempts(path)
    attempts[cell_key] = attempts.get(cell_key, 0) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(attempts, indent=2) + "\n")
    return attempts[cell_key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="cap pairs per cell, for a dry run or smoke test only")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--prompts", nargs="+", default=ALL_PROMPT_NAMES)
    ap.add_argument("--dry-run", action="store_true",
                     help="INVENTED pairs only, mocked Ollama call, never "
                          "touches mimic_local_only. Writes under a temp dir.")
    ap.add_argument("--out-dir", default=None,
                     help="override output root (dry-run default: a temp dir "
                          "printed at the end of the run)")
    args = ap.parse_args()

    run_tag = time.strftime("c109_%Y%m%dT%H%M%S")

    if args.dry_run:
        import tempfile
        out_root = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path(
            tempfile.mkdtemp(prefix="c109_dry_run_"))
        cache_dir = out_root / "cache"
        log_dir = out_root / "logs"
        cells_dir = out_root / "cells"
        egress_dir = out_root
        attempts_path = out_root / "attempts.json"
        gold_rows = [(pid, nurse, disch) for pid, nurse, disch, _, _ in INVENTED_GOLD]
        print(f"--dry-run: INVENTED {len(gold_rows)} pairs, mocked Ollama call, "
              f"no mimic_local_only access, no model loaded. Writing under {out_root}")
    else:
        out_root = REAL_EGRESS_DIR
        cache_dir = assert_local_only(REAL_CACHE_DIR)
        log_dir = REAL_LOG_DIR
        cells_dir = REAL_CELLS_DIR
        egress_dir = REAL_EGRESS_DIR
        attempts_path = REAL_ATTEMPTS_PATH
        assert_loopback_host(HOST)
        write_egress_evidence(run_tag, egress_dir)
        g = pd.read_csv(GOLD)
        gold_rows = list(zip(g["pair_id"], g["nursing_sentence"].astype(str),
                              g["discharge_sentence"].astype(str)))

    if args.limit:
        gold_rows = gold_rows[:args.limit]

    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{run_tag}.log"
    with open(log_path, "a") as log_fh:
        print(f"C109 run {run_tag} (dry_run={args.dry_run}): {len(gold_rows)} pairs, "
              f"{len(args.models)} models x {len(args.prompts)} prompts",
              file=log_fh, flush=True)
        for model in args.models:
            tag = ollama_model_tag(model, False)
            for prompt_name in args.prompts:
                cell_key = f"{tag}::{prompt_name}"
                attempt_n = bump_attempts(attempts_path, cell_key)
                if attempt_n > 2:
                    sys.exit(f"cell {cell_key} started {attempt_n} times "
                              f"(killed at least twice); refusing this 3rd "
                              f"start per the pre-registered kill condition")
                t0 = time.time()
                records, n_think, n_trunc = run_cell(
                    model, prompt_name, gold_rows, log_fh, args.dry_run)
                cell = aggregate_cell(records, n_think, n_trunc, model, prompt_name, False)
                cell["elapsed_sec"] = round(time.time() - t0, 1)
                cell["attempt_number"] = attempt_n

                cache_path = cache_dir / f"{tag}__{prompt_name}.jsonl"
                with open(cache_path, "w") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")

                # --dry-run only: also write a synthetic "C103 P_reversed"
                # reference cache for the same pair_ids/model, so the
                # determinism check in analyze_c109.py has something to
                # compare P_reversed_rerun against. Never done in a real
                # run; the real C103 cache is never touched or regenerated
                # by this file.
                if args.dry_run and prompt_name == "P_reversed_rerun":
                    ref_path = cache_dir / f"{tag}__P_reversed.jsonl"
                    with open(ref_path, "w") as f:
                        for r in records:
                            f.write(json.dumps(r) + "\n")

                cell_path = cells_dir / f"{tag}__{prompt_name}.json"
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

    if args.dry_run:
        print(f"--dry-run complete. cache_dir={cache_dir} cells_dir={cells_dir} "
              f"attempts_path={attempts_path}")
        # Print the paths so a follow-up analyze_c109.py --dry-run invocation
        # can point at the same temp root.
        print(f"DRY_RUN_OUT_ROOT={out_root}")


if __name__ == "__main__":
    main()
