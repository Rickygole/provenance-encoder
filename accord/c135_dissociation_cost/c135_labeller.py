"""C135 labeller arm: local Ollama models over the displayed evidence at each planted rate.
See C135_PREREGISTRATION.md. Local Ollama socket only; no hosted API, no restricted clinical path.

    python3 accord/c135_dissociation_cost/c135_labeller.py
"""
import json
import pathlib
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import c135_pool as P  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results/accord/c135_dissociation_cost"
HOST = "http://127.0.0.1:11434/api/generate"
MODELS = ("llama3:8b", "granite3.3:8b")
N_ITEMS = 400
LLM_SEED = 7
PROMPT = (
    "You are checking one piece of evidence.\n"
    "Sentence A: {premise}\n"
    "Sentence B: {hypothesis}\n"
    "Do these two sentences contradict each other? Answer with exactly one word, yes or no."
)


def ask(model, prompt, tries=4):
    """One generation, retried on a busy or slow local server. Raises after the last try."""
    last = None
    for attempt in range(tries):
        try:
            return _ask_once(model, prompt)
        except Exception as exc:  # local server busy with another client, or reloading a model
            last = exc
            time.sleep(5 * (attempt + 1))
    raise last


def _ask_once(model, prompt):
    req = urllib.request.Request(
        HOST,
        data=json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "20m",
                "options": {"temperature": 0, "seed": LLM_SEED, "num_predict": 4},
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"]


def parse(resp):
    t = resp.strip().lower()
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    return 1 if "yes" in t and "no" not in t else 0


def main():
    groups = P.load_groups()
    # same cursor walk as c135_run.py, so the mid pool here is the mid pool there
    cursor = 0
    pools = {}
    for name, base in P.POOLS.items():
        cands, n_tp, cursor = P.build_pool(groups, cursor, P.POOL_N, base)
        pools[name] = (cands, n_tp, P.plant_order(n_tp))
    cands, n_tp, order = pools["mid"]
    rng = np.random.default_rng(P.SEED + 1)
    slots = np.sort(rng.choice(len(cands), size=N_ITEMS, replace=False))  # same slots at every rate
    true = np.array([cands[i]["true"] for i in slots], dtype=int)

    out = {
        "preregistration": "accord/c135_dissociation_cost/C135_PREREGISTRATION.md",
        "seed": P.SEED,
        "llm_seed": LLM_SEED,
        "host": "local ollama socket 127.0.0.1:11434",
        "pool": "mid",
        "n_items_per_rate": N_ITEMS,
        "prompt_template": PROMPT,
        "slot_true_positive_rate": float(true.mean()),
        "models": {},
    }
    for model in MODELS:
        rows, t0 = [], time.time()
        try:
            for r in P.RATES:
                disp, planted, _ = P.display_for(cands, n_tp, order, r)
                conf = P.confirm(cands, disp)[slots]
                flags = []
                for i in slots:
                    flags.append(
                        parse(ask(model, PROMPT.format(premise=cands[i]["premise"],
                                                       hypothesis=cands[i]["hypotheses"][disp[i]])))
                    )
                f = np.array(flags, dtype=int)
                rows.append(
                    {
                        "planted_rate": r,
                        "flag_rate": float(f.mean()),
                        "accuracy_vs_true_label": float((f == true).mean()),
                        "recall_on_true_positives": float(f[true == 1].mean()),
                        "specificity_on_true_negatives": float(1 - f[true == 0].mean()),
                        "agreement_with_displayed_evidence_judge": float((f == conf).mean()),
                        "displayed_judge_flag_rate": float(conf.mean()),
                    }
                )
                print(f"  {model} r={r} flag={f.mean():.4f} acc={(f == true).mean():.4f}", flush=True)
            out["models"][model] = {"skipped": False, "seconds": time.time() - t0, "by_rate": rows}
        except Exception as exc:  # unreachable or too slow: arm is skipped, no number used
            out["models"][model] = {"skipped": True, "reason": type(exc).__name__, "by_rate": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c135_labeller.json").write_text(json.dumps(out, indent=1) + "\n")
    print("written", OUT / "c135_labeller.json")


if __name__ == "__main__":
    main()
