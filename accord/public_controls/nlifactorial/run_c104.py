"""C104: run the P0/P1/P2 x negative-hardness x model-scale factorial on
public SNLI + HANS data only. See C104_PREREGISTRATION.md.

No MIMIC file is opened anywhere in this script. Text (premise/hypothesis/
sentence1/sentence2) is re-derived at call time from the public parquet/HANS
files by the indices/pairIDs recorded in manifest.json; the manifest itself
never carries text, and neither do this script's outputs: raw/*.jsonl store
item_id, class, log-odds and a degenerate flag only.
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from accord.public_controls.nlifactorial.build_manifest import (  # noqa: E402
    reproduce_snli_pool_cursors, HANS_PATH,
)
from accord.public_controls.nlifactorial.prompts import (  # noqa: E402
    PROMPTS, PROMPT_NAMES, POS_TOKENS, NEG_TOKENS,
)

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "manifest.json"
RAW_DIR = ROOT / "results" / "accord" / "public_controls" / "nlifactorial" / "raw"
CELLS_DIR = ROOT / "results" / "accord" / "public_controls" / "nlifactorial" / "cells"
HOST = "http://127.0.0.1:11434"

MODELS = ["qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "qwen3:8b", "qwen3:14b",
          "llama3:8b", "granite3.3:8b"]
GUARD_LATENCY_S = 2.0
GUARD_PROMPTS_IF_SLOW = ["P0", "P1"]


def _model_tag(model):
    return model.replace(":", "_").replace(".", "")


def load_text_sources():
    df, *_ = reproduce_snli_pool_cursors()
    hans = pd.read_csv(HANS_PATH, sep="\t")
    hans = hans.set_index("pairID")
    return df, hans


def get_text(item, snli_df, hans_df):
    if item["dataset"] == "snli":
        row = snli_df.loc[item["row"]]
        return str(row["premise"]), str(row["hypothesis"])
    row = hans_df.loc[item["pairID"]]
    return str(row["sentence1"]), str(row["sentence2"])


def item_id(item):
    if item["dataset"] == "snli":
        return f"snli:{item['row']}"
    return f"hans:{item['pairID']}"


def call_chat(model, system, user, retries=4):
    body = {
        "model": model, "stream": False, "logprobs": True, "top_logprobs": 20,
        "think": False,
        "options": {"temperature": 0, "num_predict": 1},
        "messages": [{"role": "system", "content": system},
                      {"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        f"{HOST}/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            dt = time.time() - t0
            msg = r.get("message") or {}
            thinking = msg.get("thinking")
            lp = r.get("logprobs") or []
            top = (lp[0].get("top_logprobs") if lp else []) or []
            return {
                "content": msg.get("content") or "",
                "top_logprobs": top,
                "thinking_nonempty": bool(thinking and thinking.strip()),
                "latency_s": dt,
            }
        except Exception as exc:
            print(f"    attempt {attempt + 1}/{retries} failed: "
                  f"{type(exc).__name__}", file=sys.stderr)
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def parse_and_score(reply):
    content = (reply["content"] or "").strip()
    first = content.split()[0].strip(".,:;!?").lower() if content else ""
    pos_lower = {t.lower() for t in POS_TOKENS}
    neg_lower = {t.lower() for t in NEG_TOKENS}
    if first in pos_lower:
        label = 1
    elif first in neg_lower:
        label = 0
    else:
        label = None
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


def label_items(model, prompt_name, items, snli_df, hans_df):
    cfg = PROMPTS[prompt_name]
    out = []
    latencies = []
    for it in items:
        premise, hypothesis = get_text(it, snli_df, hans_df)
        user = cfg["user_template"].format(premise=premise, hypothesis=hypothesis)
        reply = call_chat(model, cfg["system"], user)
        if reply is None:
            out.append({"item_id": item_id(it), "class": it["label"],
                         "hardness": it.get("hardness"), "hard_label": None,
                         "logodds": None, "degenerate": True,
                         "parse_failure": True, "call_failed": True,
                         "thinking_nonempty": False})
            continue
        label, logodds, degenerate, parse_fail = parse_and_score(reply)
        latencies.append(reply["latency_s"])
        out.append({"item_id": item_id(it), "class": it["label"],
                     "hardness": it.get("hardness"), "hard_label": label,
                     "logodds": logodds, "degenerate": degenerate,
                     "parse_failure": parse_fail, "call_failed": False,
                     "thinking_nonempty": reply["thinking_nonempty"]})
    return out, latencies


def flatten_manifest_section(section):
    """section is manifest['main'] or manifest['pilot']: dict of
    hardness -> list of {dataset, row/pairID, label, hash}. Returns one flat
    list with a 'hardness' key added to each item."""
    flat = []
    for hardness, items in section.items():
        for it in items:
            it2 = dict(it)
            it2["hardness"] = hardness
            flat.append(it2)
    return flat


def run_pilot(models, prompts, manifest, snli_df, hans_df, out_path):
    pilot_items = flatten_manifest_section(manifest["pilot"])
    results = {}
    for model in models:
        for prompt_name in prompts:
            t0 = time.time()
            recs, latencies = label_items(model, prompt_name, pilot_items, snli_df, hans_df)
            elapsed = time.time() - t0
            n = len(recs)
            n_deg = sum(1 for r in recs if r["degenerate"])
            n_think = sum(1 for r in recs if r["thinking_nonempty"])
            mean_lat = float(np.mean(latencies)) if latencies else None
            results[f"{model}::{prompt_name}"] = {
                "model": model, "prompt": prompt_name, "n": n,
                "n_degenerate": n_deg, "degenerate_rate": round(n_deg / n, 4) if n else None,
                "n_thinking_nonempty": n_think,
                "mean_latency_s": round(mean_lat, 3) if mean_lat else None,
                "elapsed_sec": round(elapsed, 1),
            }
            print(f"[pilot] {model} {prompt_name}: n={n} degenerate={n_deg} "
                  f"thinking_nonempty={n_think} mean_latency={mean_lat} "
                  f"({elapsed:.0f}s)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    return results


def determinism_ok(rec_a, rec_b, tol=0.05):
    if rec_a["hard_label"] != rec_b["hard_label"]:
        return False
    if rec_a["logodds"] is None or rec_b["logodds"] is None:
        return rec_a["logodds"] == rec_b["logodds"]
    return abs(rec_a["logodds"] - rec_b["logodds"]) <= tol


def run_cell(model, prompt_name, manifest, snli_df, hans_df):
    main_items = flatten_manifest_section(manifest["main"])
    det_items = manifest["determinism_check"]

    det_start, _ = label_items(model, prompt_name, det_items, snli_df, hans_df)
    main_recs, _ = label_items(model, prompt_name, main_items, snli_df, hans_df)
    det_end, _ = label_items(model, prompt_name, det_items, snli_df, hans_df)

    determinism_failures = sum(
        0 if determinism_ok(a, b) else 1 for a, b in zip(det_start, det_end))

    n = len(main_recs)
    n_deg = sum(1 for r in main_recs if r["degenerate"])
    n_parse_fail = sum(1 for r in main_recs if r["parse_failure"])
    n_failed = sum(1 for r in main_recs if r["call_failed"])
    n_thinking = sum(1 for r in main_recs if r["thinking_nonempty"])
    degenerate_rate = n_deg / n if n else None
    valid = (degenerate_rate is not None and degenerate_rate <= 0.10
              and determinism_failures == 0 and n_failed == 0)

    meta = {
        "model": model, "prompt": prompt_name, "n": n,
        "n_degenerate": n_deg, "degenerate_rate": round(degenerate_rate, 4) if degenerate_rate is not None else None,
        "n_parse_failures": n_parse_fail, "n_call_failed": n_failed,
        "n_thinking_nonempty": n_thinking,
        "n_determinism_items": len(det_items),
        "n_determinism_failures": determinism_failures,
        "valid": bool(valid),
    }
    return main_recs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "main"], required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--prompts", nargs="+", default=PROMPT_NAMES)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())
    snli_df, hans_df = load_text_sources()

    if args.phase == "pilot":
        out_path = (ROOT / "results" / "accord" / "public_controls"
                    / "nlifactorial" / "pilot_results.json")
        run_pilot(args.models, args.prompts, manifest, snli_df, hans_df, out_path)
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        tag = _model_tag(model)
        prompts_this_model = args.prompts
        for prompt_name in prompts_this_model:
            cell_path = CELLS_DIR / f"{tag}__{prompt_name}_meta.json"
            raw_path = RAW_DIR / f"{tag}__{prompt_name}.jsonl"
            if cell_path.exists():
                print(f"skip {model} {prompt_name}: cell already exists at {cell_path}")
                continue
            t0 = time.time()
            main_recs, meta = run_cell(model, prompt_name, manifest, snli_df, hans_df)
            meta["elapsed_sec"] = round(time.time() - t0, 1)
            with open(raw_path, "w") as f:
                for r in main_recs:
                    f.write(json.dumps({
                        "item_id": r["item_id"], "class": r["class"],
                        "hardness": r["hardness"], "hard_label": r["hard_label"],
                        "logodds": r["logodds"], "degenerate": r["degenerate"],
                    }) + "\n")
            cell_path.write_text(json.dumps(meta, indent=2) + "\n")
            print(f"{model} {prompt_name}: n={meta['n']} valid={meta['valid']} "
                  f"degenerate_rate={meta['degenerate_rate']} "
                  f"determinism_failures={meta['n_determinism_failures']} "
                  f"({meta['elapsed_sec']:.0f}s) -> {cell_path}")


if __name__ == "__main__":
    main()
