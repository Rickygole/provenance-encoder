"""Prompted-LLM baseline for note provenance, run entirely on localhost.

This is the compliant replacement for the GPT-4o evaluation in ACCORD.
PhysioNet's published guidance states the credentialed DUA prohibits sending
restricted data through hosted APIs such as OpenAI's; Ollama on 127.0.0.1 sends
nothing anywhere. Refuses to run against a non-local endpoint.

Scientific purpose: the TF-IDF probe recovered note role at 97.5-99.7%. If a
prompted 8B model also recovers it near-perfectly, the redundancy finding is not
an artifact of one model class, and the case that provenance conditioning
supplies no information gets substantially stronger.

Emits aggregate metrics only. No note text is printed or written.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from urllib import request as urlreq

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from role_probe import OUT_DIR, scrub  # noqa: E402

CACHE = OUT_DIR / "probe_sample.pkl"
LOCAL_HOSTS = ("127.0.0.1", "localhost")

PROMPT = """You are classifying a clinical note by its type.

Note excerpt:
---
{text}
---

Which type of note is this? Answer with EXACTLY one word:
nursing - a nursing note (shift observations, vitals, care delivered)
physician - a physician note (assessment and plan, clinical reasoning)
discharge - a discharge summary (narrative of the whole admission)

Answer:"""

VALID = ("nursing", "physician", "discharge")


def assert_local(host: str, port: int) -> None:
    if host not in LOCAL_HOSTS:
        raise SystemExit(
            f"refusing to send MIMIC-derived text to non-local host {host!r}. "
            "This script only talks to Ollama on localhost."
        )


def generate(host: str, port: int, model: str, prompt: str, timeout: float = 120.0) -> str:
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_predict": 6},
    }).encode()
    req = urlreq.Request(f"http://{host}:{port}/api/generate", data=body,
                         headers={"Content-Type": "application/json"})
    with urlreq.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()).get("response", "")


def parse(resp: str) -> str | None:
    low = resp.strip().lower()
    for v in VALID:
        if low.startswith(v):
            return v
    for v in VALID:
        if v in low:
            return v
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3:8b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--per-role", type=int, default=120)
    ap.add_argument("--chars", type=int, default=1500)
    ap.add_argument("--scrub-level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert_local(args.host, args.port)
    with CACHE.open("rb") as f:
        buckets = pickle.load(f)

    rng = np.random.default_rng(args.seed)
    items = []
    for role, rows in buckets.items():
        idx = rng.choice(len(rows), size=min(args.per_role, len(rows)), replace=False)
        for i in idx:
            items.append((role, scrub(rows[i][1], args.scrub_level)[: args.chars]))
    rng.shuffle(items)
    print(f"model={args.model} endpoint=http://{args.host}:{args.port} (local only)")
    print(f"evaluating {len(items)} notes, scrub level {args.scrub_level}\n", flush=True)

    correct = unparsed = 0
    confusion: Counter[tuple[str, str]] = Counter()
    t0 = time.time()
    for n, (gold, text) in enumerate(items, 1):
        try:
            pred = parse(generate(args.host, args.port, args.model, PROMPT.format(text=text)))
        except Exception as e:
            print(f"  request failed: {type(e).__name__}", flush=True)
            pred = None
        if pred is None:
            unparsed += 1
            pred = "<unparsed>"
        else:
            correct += int(pred == gold)
        confusion[(gold, pred)] += 1
        if n % 25 == 0:
            print(f"  {n}/{len(items)}  running acc {correct/n:.3f}  "
                  f"({n/(time.time()-t0):.2f} notes/s)", flush=True)

    acc = correct / len(items)
    print(f"\naccuracy      : {acc:.4f}")
    print(f"unparsed      : {unparsed} ({unparsed/len(items):.1%})")
    print(f"wall seconds  : {time.time()-t0:.0f}")

    roles = sorted({g for g, _ in confusion})
    preds = sorted({p for _, p in confusion})
    print("\nconfusion (rows = gold, cols = predicted):")
    print("           " + "".join(f"{p:>12}" for p in preds))
    for g in roles:
        print(f"  {g:<9}" + "".join(f"{confusion[(g, p)]:>12}" for p in preds))

    out = OUT_DIR / "ollama_role_baseline.json"
    out.write_text(json.dumps({
        "model": args.model, "n": len(items), "accuracy": acc,
        "unparsed": unparsed, "scrub_level": args.scrub_level,
        "confusion": {f"{g}->{p}": c for (g, p), c in confusion.items()},
    }, indent=2))
    print(f"\nwrote {out}")
    print("\nCompare: TF-IDF + logistic regression reached 99.7% scrubbed on the")
    print("full pool. If a prompted 8B model is also high, provenance redundancy")
    print("is not an artifact of one model class.")


if __name__ == "__main__":
    main()
