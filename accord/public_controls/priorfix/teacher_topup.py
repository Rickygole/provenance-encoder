"""C105: label the 2,000-item SNLI top-up (1,000 entailment + 1,000 neutral)
with llama3:8b under P1, the same prompt and call shape Item 11 used for its
own llama3:8b/P1 cell. Public data only; no MIMIC access.

Must not run while another Ollama model is loaded; OLLAMA_MAX_LOADED_MODELS=1
on this machine already enforces at most one loaded model, and this script
additionally checks `ollama ps` is either empty or already llama3:8b before
its first call.
"""
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from accord.public_controls.nlifactorial.build_manifest import (  # noqa: E402
    reproduce_snli_pool_cursors,
)
from accord.public_controls.nlifactorial.prompts import (  # noqa: E402
    PROMPTS, POS_TOKENS, NEG_TOKENS,
)

HERE = pathlib.Path(__file__).resolve().parent
PARTITIONS_PATH = HERE / "partitions.json"
OUT_DIR = ROOT / "results" / "accord" / "public_controls" / "priorfix" / "raw"
HOST = "http://127.0.0.1:11434"
MODEL = "llama3:8b"
PROMPT_NAME = "P1"


def assert_only_this_model_loaded():
    r = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
    lines = [l for l in r.stdout.strip().splitlines()[1:] if l.strip()]
    other = [l for l in lines if not l.startswith(MODEL)]
    if other:
        raise SystemExit(f"refusing to start: another model is loaded "
                          f"({len(other)} entries not starting with {MODEL!r}); "
                          f"unload it first (`ollama stop <model>`)")


def call_chat(model, system, user, retries=4):
    body = {"model": model, "stream": False, "logprobs": True, "top_logprobs": 20,
            "think": False, "options": {"temperature": 0, "num_predict": 1},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    req = urllib.request.Request(f"{HOST}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            msg = r.get("message") or {}
            lp = r.get("logprobs") or []
            top = (lp[0].get("top_logprobs") if lp else []) or []
            return {"content": msg.get("content") or "", "top_logprobs": top}
        except Exception as exc:
            print(f"    attempt {attempt+1}/{retries} failed: {type(exc).__name__}",
                  file=sys.stderr)
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def parse_and_score(reply):
    content = (reply["content"] or "").strip()
    first = content.split()[0].strip(".,:;!?").lower() if content else ""
    pos_lower = {t.lower() for t in POS_TOKENS}
    neg_lower = {t.lower() for t in NEG_TOKENS}
    label = 1 if first in pos_lower else (0 if first in neg_lower else None)
    p = n_ = None
    for t in reply["top_logprobs"]:
        tok = (t.get("token") or "").strip()
        if p is None and tok in POS_TOKENS:
            p = t["logprob"]
        if n_ is None and tok in NEG_TOKENS:
            n_ = t["logprob"]
    degenerate = p is None or n_ is None
    logodds = (p if p is not None else -20.0) - (n_ if n_ is not None else -20.0)
    return label, logodds, degenerate, label is None


def main():
    assert_only_this_model_loaded()
    partitions = json.loads(PARTITIONS_PATH.read_text())
    df, *_ = reproduce_snli_pool_cursors()
    cfg = PROMPTS[PROMPT_NAME]

    rows = ([("entailment", r) for r in partitions["teacher_topup_entail_rows"]]
            + [("neutral", r) for r in partitions["teacher_topup_neutral_rows"]])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "teacher_topup.jsonl"
    n_deg = n_fail = 0
    with open(out_path, "w") as f:
        for i, (cls, row_idx) in enumerate(rows):
            premise, hypothesis = str(df.loc[row_idx, "premise"]), str(df.loc[row_idx, "hypothesis"])
            user = cfg["user_template"].format(premise=premise, hypothesis=hypothesis)
            reply = call_chat(MODEL, cfg["system"], user)
            if reply is None:
                f.write(json.dumps({"row": int(row_idx), "class": cls,
                                     "hard_label": None, "logodds": None,
                                     "degenerate": True}) + "\n")
                n_fail += 1
                continue
            label, logodds, degenerate, _ = parse_and_score(reply)
            n_deg += int(degenerate)
            f.write(json.dumps({"row": int(row_idx), "class": cls,
                                 "hard_label": label, "logodds": logodds,
                                 "degenerate": degenerate}) + "\n")
            if (i + 1) % 200 == 0:
                print(f"{i+1}/{len(rows)} scored", flush=True)
    print(f"done: n={len(rows)} n_degenerate={n_deg} n_call_failed={n_fail} -> {out_path}")


if __name__ == "__main__":
    main()
