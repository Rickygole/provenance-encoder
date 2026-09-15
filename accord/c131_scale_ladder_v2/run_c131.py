"""C131: qwen3 scale ladder v2, a post hoc repair of C127/C128's qwen3:4b decoding defect.
See C131_PREREGISTRATION.md.

Mirrors accord/c127_scale_ladder/run_c127.py and accord/prompt_sensitivity/run_c103.py's data
loading, join gate, caching convention, egress check and compliance redirect line for line, with
exactly one decoding change (num_predict 1 -> 8, see the pre-registration's "the remedy" section)
and a wider parse (the hard label scans every word of the reply's content, not just the first; the
graded score is read at the first answer-vocabulary TOKEN position in Ollama's per-position
logprobs list, not always position 0).

Local only: every request goes to an Ollama server on 127.0.0.1. Reads credentialed gold data from
mimic_local_only and writes:
  - a numeric-only per-pair cache under ~/mimic_local_only/accord_c131_cache/*.jsonl (pair_id, flag
    label, flag log-odds [primary and legacy-position-0], degenerate, parse-failure,
    thinking-nonempty, truncated, answer_token_position, n_generated_tokens,
    first_token_is_answer -- never sentence text). A NEW cache directory, not C103/C127's, because
    the schema and num_predict differ from those runs and writing into the same namespace would be
    ambiguous about which settings produced which file.
  - aggregate-only cell JSON under results/accord/c131_scale_ladder_v2/cells/, plus this run's own
    summary under results/accord/c131_scale_ladder_v2/.

Refuses to run against the gold CSV unless
results/accord/c131_scale_ladder_v2/synthetic_probe_validation.json already exists (the
pre-registration's validation-before-any-clinical-call requirement).

No note text, pair text, or model reply text is ever printed to this process's stdout/stderr, or
written anywhere but the local cache above (as numeric flags, not text).

    python3 accord/c131_scale_ladder_v2/run_c131.py
"""
import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accord.cloud_paths import assert_local_only  # noqa: E402
from accord.differential.model_tags import ollama_model_tag  # noqa: E402
from accord.prompt_sensitivity.prompts import PROMPTS  # noqa: E402
from accord.prompt_sensitivity.run_c103 import assert_loopback_host  # noqa: E402

GOLD = pathlib.Path.home() / "mimic_local_only/from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv"
CACHE_DIR = assert_local_only(pathlib.Path.home() / "mimic_local_only/accord_c131_cache")
LOG_DIR = pathlib.Path.home() / "mimic_local_only/logs"
CELLS_DIR = ROOT / "results" / "accord" / "c131_scale_ladder_v2" / "cells"
OUT_DIR = ROOT / "results" / "accord" / "c131_scale_ladder_v2"
VALIDATION_PATH = OUT_DIR / "synthetic_probe_validation.json"

# C127's own cache/cell locations: read-only reference for the reuse check and,
# when reuse is proven, the sole source of that size's per-pair records
# (never copied into CACHE_DIR).
C127_CACHE_DIR = pathlib.Path.home() / "mimic_local_only/accord_prompt_sensitivity_cache"
C127_CELLS_DIR = ROOT / "results" / "accord" / "prompt_sensitivity" / "cells"

SIZES = ["0.6b", "1.7b", "4b", "8b", "14b"]
MODELS = {size: f"qwen3:{size}" for size in SIZES}
PROMPT_NAME = "P_asserting"
NUM_CTX = 4096  # frozen; matches run_c103.py / run_c127.py exactly
NUM_PREDICT = 8  # the only decoding-setting change this file makes; see the pre-registration
HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")

# Reuse is a candidate ONLY for these two sizes, per C131_PREREGISTRATION.md's reuse rule.
REUSABLE_TAGS = {"8b": "_qwen3_8b_nothink", "14b": "_qwen3_14b_nothink"}


def write_egress_evidence(run_tag):
    """Same evidence-and-refuse pattern as run_c103.py / run_c127.py."""
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


def call_chat_remedy(model, system, user, retries=4, log_fh=None):
    """One /api/chat call at num_predict=8. Returns dict(content, token_logprobs,
    thinking_nonempty, prompt_eval_count) or None on failure after retries.
    token_logprobs is the FULL per-position list Ollama's response carries under
    "logprobs" -- one entry per generated token, each with its own "token",
    "logprob" and "top_logprobs" -- not just position 0, unlike
    run_c103.py::call_chat. Never raises with the request/response body
    attached; only the exception type name is logged, and only to the local
    mimic_local_only log file, never to this process's own stdout.
    """
    body = {
        "model": model, "stream": False, "logprobs": True, "top_logprobs": 8,
        "think": False,
        "options": {"temperature": 0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX},
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
            return {
                "content": msg.get("content") or "",
                "token_logprobs": r.get("logprobs") or [],
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


def parse_and_score_remedy(reply, pos_tokens, neg_tokens, max_positions=8):
    """Hard label from a scan of every word in the reply's full content (not
    just the first word); graded log-odds from the first answer-vocabulary
    TOKEN position in reply["token_logprobs"]. See C131_PREREGISTRATION.md's
    "the remedy" section; this is the only function that changes C103/C127's
    parse_and_score's behaviour, and it changes only where it reads from, not
    the scoring formula itself.

    Returns a dict: hard_label, parse_failure, flag_logodds_primary,
    degenerate, answer_token_position, n_generated_tokens,
    first_token_is_answer, flag_logodds_legacy_pos0, degenerate_pos0.
    """
    content = (reply["content"] or "").strip()
    # pos_tokens/neg_tokens are C103's frozen, fragment-based vocabulary
    # (e.g. "contr", "not"), designed around num_predict=1's content being a
    # single sub-word fragment, never a whole word. At num_predict=8 the
    # model can complete the whole word instead: "contradiction" is already
    # covered (ASSERTING_POS carries a "Contradiction" entry), but
    # "not_contradiction" -- the system prompt's own literal second answer
    # word, joined without a space -- has no whole-word counterpart in
    # ASSERTING_NEG's fragments ("not"/"n"/"no"). Both explicit whole-word
    # forms of P_asserting's own answer vocabulary are added here rather than
    # widened into a prefix match, so a genuinely different word beginning
    # "not" (e.g. "nothing") is never misread as the negative answer.
    pos_lower = {t.lower() for t in pos_tokens} | {"contradiction"}
    neg_lower = {t.lower() for t in neg_tokens} | {"not_contradiction"}

    hard_label = None
    for word in content.split():
        w = word.strip(".,:;!?").lower()
        if w in pos_lower:
            hard_label = 1
            break
        if w in neg_lower:
            hard_label = 0
            break
    parse_failure = hard_label is None

    positions = reply["token_logprobs"][:max_positions]
    n_generated = len(reply["token_logprobs"])

    def score_at(entry):
        p = n_ = None
        for t in entry.get("top_logprobs") or []:
            tok = (t.get("token") or "").strip()
            if p is None and tok in pos_tokens:
                p = t["logprob"]
            if n_ is None and tok in neg_tokens:
                n_ = t["logprob"]
        degenerate = p is None or n_ is None
        logodds = (p if p is not None else -20.0) - (n_ if n_ is not None else -20.0)
        return logodds, degenerate

    answer_pos = None
    flag_logodds_primary = None
    degenerate_primary = True
    for i, entry in enumerate(positions):
        chosen = (entry.get("token") or "").strip()
        if chosen in pos_tokens or chosen in neg_tokens:
            answer_pos = i
            flag_logodds_primary, degenerate_primary = score_at(entry)
            break

    first_token_is_answer = False
    flag_logodds_legacy_pos0 = None
    degenerate_pos0 = True
    if positions:
        chosen0 = (positions[0].get("token") or "").strip()
        first_token_is_answer = chosen0 in pos_tokens or chosen0 in neg_tokens
        flag_logodds_legacy_pos0, degenerate_pos0 = score_at(positions[0])

    return {
        "hard_label": hard_label,
        "parse_failure": parse_failure,
        "flag_logodds_primary": flag_logodds_primary,
        "degenerate_primary": degenerate_primary,
        "answer_token_position": answer_pos,
        "n_generated_tokens": n_generated,
        "first_token_is_answer": first_token_is_answer,
        "flag_logodds_legacy_pos0": flag_logodds_legacy_pos0,
        "degenerate_pos0": degenerate_pos0,
    }


def run_cell(model, gold_rows, log_fh):
    cfg = PROMPTS[PROMPT_NAME]
    records = []
    n_thinking_nonempty = 0
    n_truncated = 0
    for pair_id, nursing, discharge in gold_rows:
        user = cfg["user_template"].format(
            nursing_note=nursing, discharge_summary=discharge)
        reply = call_chat_remedy(model, cfg["system"], user, log_fh=log_fh)
        if reply is None:
            records.append({
                "pair_id": pair_id, "flag_label": None, "flag_logodds": None,
                "degenerate": True, "parse_failure": True, "call_failed": True,
                "flag_logodds_legacy_pos0": None, "answer_token_position": None,
                "n_generated_tokens": 0, "first_token_is_answer": False,
            })
            continue
        scored = parse_and_score_remedy(reply, cfg["pos_tokens"], cfg["neg_tokens"])
        if reply["thinking_nonempty"]:
            n_thinking_nonempty += 1
        truncated = (reply["prompt_eval_count"] is not None
                     and reply["prompt_eval_count"] >= NUM_CTX)
        if truncated:
            n_truncated += 1
        records.append({
            "pair_id": pair_id,
            "flag_label": scored["hard_label"],
            "flag_logodds": scored["flag_logodds_primary"],
            "degenerate": scored["degenerate_primary"],
            "parse_failure": scored["parse_failure"],
            "call_failed": False,
            "thinking_nonempty": reply["thinking_nonempty"],
            "truncated": truncated,
            "answer_token_position": scored["answer_token_position"],
            "n_generated_tokens": scored["n_generated_tokens"],
            "first_token_is_answer": scored["first_token_is_answer"],
            "flag_logodds_legacy_pos0": scored["flag_logodds_legacy_pos0"],
            "degenerate_pos0": scored["degenerate_pos0"],
        })
    return records, n_thinking_nonempty, n_truncated


def aggregate_cell(records, n_thinking_nonempty, n_truncated, model, think):
    n = len(records)
    n_failed = sum(1 for r in records if r["call_failed"])
    n_parse_fail = sum(1 for r in records if r["parse_failure"])
    n_degenerate = sum(1 for r in records if r["degenerate"])
    n_flag = sum(1 for r in records if r["flag_label"] == 1)
    n_scored = sum(1 for r in records if r["flag_label"] is not None)
    n_no_answer_pos = sum(1 for r in records
                          if not r["call_failed"] and r["answer_token_position"] is None)
    gen_counts = [r["n_generated_tokens"] for r in records if not r["call_failed"]]
    mean_gen = round(sum(gen_counts) / len(gen_counts), 3) if gen_counts else None
    return {
        "model": model, "prompt": PROMPT_NAME, "think": think,
        "num_predict": NUM_PREDICT,
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
        "n_no_answer_token_position": n_no_answer_pos,
        "mean_n_generated_tokens": mean_gen,
    }


def check_reuse_provable(tag, gold_rows):
    """Provable byte-identity check for the 8B/14B reuse candidates. See
    C131_PREREGISTRATION.md's reuse rule for the greedy-decoding argument this
    encodes: byte-identity holds iff (a) the existing cell's settings are the
    ones run_c127.check_reuse already requires, AND (b) the existing cell has
    zero parse failures and zero truncated calls -- i.e. position 0 was
    already an answer-vocabulary token on every one of the 299 pairs, which
    means num_predict=8 cannot sample anything different at position 0.
    Returns (reusable: bool, reason: str)."""
    cache_path = C127_CACHE_DIR / f"{tag}__{PROMPT_NAME}.jsonl"
    cell_path = C127_CELLS_DIR / f"{tag}__{PROMPT_NAME}.json"
    if not cache_path.exists() or not cell_path.exists():
        return False, "no existing C127/C103 cache/cell file"
    cell = json.loads(cell_path.read_text())
    if cell.get("prompt") != PROMPT_NAME:
        return False, f"existing cell prompt {cell.get('prompt')!r} != {PROMPT_NAME!r}"
    if cell.get("think") is not False:
        return False, f"existing cell think={cell.get('think')!r} != False"
    if cell.get("n") != len(gold_rows):
        return False, f"existing cell n={cell.get('n')} != {len(gold_rows)}"
    if cell.get("n_parse_failures") != 0:
        return False, f"existing cell n_parse_failures={cell.get('n_parse_failures')} != 0"
    if cell.get("n_truncated") != 0:
        return False, f"existing cell n_truncated={cell.get('n_truncated')} != 0"
    recs = [json.loads(line) for line in cache_path.read_text().splitlines() if line]
    if len(recs) != len(gold_rows):
        return False, f"existing cache has {len(recs)} rows, expected {len(gold_rows)}"
    if not all("flag_logodds" in r for r in recs):
        return False, "existing cache is missing per-pair flag_logodds"
    if any(r.get("parse_failure") for r in recs) or any(r.get("truncated") for r in recs):
        return False, "existing cache disagrees with its own cell aggregate (parse_failure/truncated)"
    from accord.prompt_sensitivity import run_c103 as _c103_mod
    if _c103_mod.NUM_CTX != NUM_CTX:
        return False, f"run_c103.NUM_CTX={_c103_mod.NUM_CTX} != {NUM_CTX}"
    return True, (
        "existing cell settings byte-identical (prompt, think, n, NUM_CTX) AND "
        "zero parse failures AND zero truncated calls in every one of the 299 "
        "rows, so position 0 is provably the first answer-vocabulary position "
        "under the remedy too (num_predict cannot change what is sampled at "
        "position 0 under greedy decoding); byte-identical by construction, "
        "not compared")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=SIZES)
    ap.add_argument("--limit", type=int, default=None, help="cap pairs, dry run only")
    ap.add_argument("--force-rerun", action="store_true",
                     help="ignore the reuse check and rerun every size")
    args = ap.parse_args()

    if not VALIDATION_PATH.exists():
        raise SystemExit(
            f"refusing to run against gold data: {VALIDATION_PATH} does not exist. "
            f"Run accord/c131_scale_ladder_v2/validate_remedy_synthetic.py first, "
            f"per C131_PREREGISTRATION.md's validation-before-any-clinical-call rule.")
    validation = json.loads(VALIDATION_PATH.read_text())
    if not validation.get("all_checks_passed"):
        raise SystemExit(
            f"refusing to run against gold data: synthetic probe validation did "
            f"not pass ({validation.get('summary')!r}); see {VALIDATION_PATH}")

    assert_loopback_host(HOST)
    run_tag = time.strftime("c131_%Y%m%dT%H%M%S")
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

    run_log = {"run_tag": run_tag, "num_predict": NUM_PREDICT,
               "per_token_logprobs_available": validation.get("per_token_logprobs_available"),
               "sizes": {}}
    log_path = LOG_DIR / f"{run_tag}.log"
    with open(log_path, "a") as log_fh:
        print(f"C131 run {run_tag}: {len(gold_rows)} pairs, sizes={args.sizes}, "
              f"num_predict={NUM_PREDICT}", file=log_fh, flush=True)
        for size in args.sizes:
            model = MODELS[size]
            think = False
            tag = ollama_model_tag(model, think)

            if args.force_rerun:
                reused, reuse_reason = False, "reuse not attempted (--force-rerun)"
            elif size not in REUSABLE_TAGS:
                reused, reuse_reason = False, "not a reuse candidate (only 8b/14b are, per the pre-registration)"
            else:
                reused, reuse_reason = check_reuse_provable(REUSABLE_TAGS[size], gold_rows)

            if reused:
                source_cell_path = C127_CELLS_DIR / f"{tag}__{PROMPT_NAME}.json"
                source_cache_path = C127_CACHE_DIR / f"{tag}__{PROMPT_NAME}.jsonl"
                source_cell = json.loads(source_cell_path.read_text())
                cell = dict(source_cell)
                cell["reused_from_c127"] = True
                cell["reuse_check"] = reuse_reason
                cell["source_cell_path"] = str(source_cell_path.relative_to(ROOT))
                cell["source_cache_path"] = str(source_cache_path)
                cell["elapsed_sec"] = 0.0
                cell_path = CELLS_DIR / f"{tag}__{PROMPT_NAME}.json"
                cell_path.write_text(json.dumps(cell, indent=2) + "\n")
                print(f"qwen3:{size} {PROMPT_NAME}: REUSED from C127 "
                      f"({reuse_reason}) -> {cell_path}")
                print(f"cell qwen3:{size} REUSED ({reuse_reason})", file=log_fh, flush=True)
            else:
                t0 = time.time()
                records, n_think, n_trunc = run_cell(model, gold_rows, log_fh)
                cell = aggregate_cell(records, n_think, n_trunc, model, think)
                cell["elapsed_sec"] = round(time.time() - t0, 1)
                cell["reused_from_c127"] = False
                cell["reuse_check"] = reuse_reason

                cache_path = CACHE_DIR / f"{tag}__{PROMPT_NAME}__c131.jsonl"
                with open(cache_path, "w") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")
                cell["source_cache_path"] = str(cache_path)

                cell_path = CELLS_DIR / f"{tag}__{PROMPT_NAME}.json"
                cell_path.write_text(json.dumps(cell, indent=2) + "\n")
                print(f"qwen3:{size} {PROMPT_NAME}: n={cell['n']} "
                      f"flagged={cell['n_flagged']} rate={cell['positive_rate']} "
                      f"parse_fail={cell['n_parse_failures']} "
                      f"degenerate={cell['n_degenerate_logodds']} "
                      f"thinking_nonempty={cell['n_thinking_nonempty']} "
                      f"truncated={cell['n_truncated']} "
                      f"no_answer_pos={cell['n_no_answer_token_position']} "
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
                "source_cache_path": cell.get("source_cache_path"),
            }

    (OUT_DIR / f"run_summary_{run_tag}.json").write_text(
        json.dumps(run_log, indent=2) + "\n")
    print(f"run summary -> {OUT_DIR / f'run_summary_{run_tag}.json'}")


if __name__ == "__main__":
    main()
