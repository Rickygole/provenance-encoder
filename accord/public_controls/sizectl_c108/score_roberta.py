"""C108: score every non-pool, gold-labelled SNLI dev/test item with
FacebookAI/roberta-large-mnli. See ../C108_PREREGISTRATION.md.

Byte-for-byte the SAME scoring procedure as C101's
`accord/public_controls/sizectl/score_roberta.py` -- same model, same
manifest, same data, same exclusion set, same score formula, same batching
(deterministic: `model.eval()`, no dropout, batch composition depends only
on string length which is unchanged) -- reused verbatim per the C108
pre-registration's "only change is the calibration fix" statement. The only
difference from C101's script is the output path, so this rerun writes to
its own artifact tree (`results/accord/public_controls/sizectl_c108/`)
rather than overwriting C101's already-committed scores.

THE OFFLINE ENV VARS ARE SET AS THE VERY FIRST LINES OF THIS MODULE, before any
other import, matching accord/differential/nli_baseline.py's discipline:
transformers and huggingface_hub read these at import time.

No hosted API, no locally-served chat model, CPU only, offline only. Public/synthetic data only
(SNLI dev/test, a public HF dataset) -- no MIMIC path is opened anywhere in
this file. Per-item output is keyed by a hash of (premise, hypothesis), never
the text itself.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
MODEL_DIR = pathlib.Path.home() / "models_public" / "roberta-large-mnli"
MANIFEST = ROOT / "results" / "accord" / "differential" / "nli_model_manifest.json"
DATA_DIR = ROOT / "accord" / "prevalence_replication" / "data"
POOLS_DIR = DATA_DIR / "pools"

OUT_META = ROOT / "results" / "accord" / "public_controls" / "sizectl_c108" / "roberta_scores_meta.json"
OUT_SCORES = ROOT / "results" / "accord" / "public_controls" / "sizectl_c108" / "roberta_scores.json"

MAX_TOKENS = 512
BATCH_SIZE = 128
torch.set_num_threads(os.cpu_count() or 8)


def _item_hash(premise: str, hypothesis: str) -> str:
    return hashlib.sha256(f"{premise}\x1f{hypothesis}".encode()).hexdigest()[:16]


def load_excluded_pairs():
    """Every (premise, hypothesis) pair already used by any prevalence/
    distillation pool -- these were drawn from the same combined SNLI
    test+validation set for a different purpose and must not double-serve
    as this control's data."""
    excluded = set()
    for csv_path in glob.glob(str(POOLS_DIR / "*.csv")):
        if "summary" in pathlib.Path(csv_path).name:
            continue
        df = pd.read_csv(csv_path)
        for premise, hypothesis in zip(df["premise"], df["hypothesis"]):
            excluded.add((premise, hypothesis))
    return excluded


def load_gold_items():
    """Every gold-labelled (label != -1) row from SNLI validation + test,
    minus every pool-excluded pair. Label convention matches build_pools.py:
    entailment=0, neutral=1, contradiction=2."""
    val = pd.read_parquet(DATA_DIR / "snli_validation.parquet")
    val["split"] = "validation"
    test = pd.read_parquet(DATA_DIR / "snli_test.parquet")
    test["split"] = "test"
    df = pd.concat([val, test], ignore_index=True)
    df = df[df["label"] != -1].reset_index(drop=True)
    excluded = load_excluded_pairs()
    keep = [(p, h) not in excluded for p, h in zip(df["premise"], df["hypothesis"])]
    n_before = len(df)
    df = df[keep].reset_index(drop=True)
    df["is_contradiction"] = (df["label"] == 2).astype(int)
    return df, n_before - len(df), len(excluded)


def load_model():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["repo_id"] == "FacebookAI/roberta-large-mnli"
    assert pathlib.Path(manifest["local_dir"]) == MODEL_DIR or MODEL_DIR.exists()
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    assert id2label == {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}, (
        f"unexpected id2label {id2label}; aborting rather than guessing the "
        "contradiction index")
    contra_idx = [k for k, v in id2label.items() if v == "CONTRADICTION"][0]
    return tok, model, contra_idx, manifest


def score_batch(tok, model, contra_idx, premises, hypotheses):
    enc = tok(list(premises), list(hypotheses), truncation=False, padding=True,
              return_tensors="pt")
    lengths = enc["attention_mask"].sum(dim=1)
    assert int(lengths.max()) <= MAX_TOKENS, (
        f"an SNLI pair encoded to {int(lengths.max())} tokens, over the "
        f"{MAX_TOKENS} limit; aborting rather than silently truncating")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1)
    p_contra = probs[:, contra_idx].clamp(1e-12, 1 - 1e-12)
    scores = torch.log(p_contra) - torch.log(1 - p_contra)
    return scores.tolist()


def main():
    t0 = time.time()
    df, n_excluded, n_pool_pairs = load_gold_items()
    tok, model, contra_idx, manifest = load_model()

    # Sort by combined text length so each batch pads to a similar length
    # (throughput only; output is keyed by hash, so order does not matter).
    # Original row position is kept as source_index before the reorder.
    df = df.reset_index(drop=True)
    df["_source_index"] = df.index
    df = df.assign(_len=df["premise"].str.len() + df["hypothesis"].str.len())
    df = df.sort_values("_len").drop(columns="_len").reset_index(drop=True)

    records = {}
    n = len(df)
    for start in range(0, n, BATCH_SIZE):
        chunk = df.iloc[start:start + BATCH_SIZE]
        scores = score_batch(tok, model, contra_idx, chunk["premise"], chunk["hypothesis"])
        for (_, row), s in zip(chunk.iterrows(), scores):
            h = _item_hash(row["premise"], row["hypothesis"])
            records[h] = {
                "source_index": int(row["_source_index"]),
                "split": row["split"],
                "label": int(row["label"]),
                "is_contradiction": int(row["is_contradiction"]),
                "score": float(s),
            }
        if (start // BATCH_SIZE) % 20 == 0:
            elapsed = time.time() - t0
            print(f"{min(start + BATCH_SIZE, n)}/{n} scored, {elapsed:.1f}s elapsed", flush=True)

    y = np.array([r["is_contradiction"] for r in records.values()])
    s = np.array([r["score"] for r in records.values()])
    a0 = float(roc_auc_score(y, s))

    OUT_SCORES.parent.mkdir(parents=True, exist_ok=True)
    OUT_SCORES.write_text(json.dumps(records) + "\n")

    meta = {
        "preregistration": "accord/public_controls/C108_PREREGISTRATION.md",
        "model": "FacebookAI/roberta-large-mnli",
        "model_manifest_sha256_of_config_json": manifest["files_sha256"]["config.json"],
        "model_commit_sha": manifest["commit_sha"],
        "id2label": {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"},
        "n_scored": int(n),
        "n_positive_contradiction": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "n_excluded_pool_overlap": int(n_excluded),
        "n_distinct_pairs_across_all_pools": int(n_pool_pairs),
        "A0_auroc": round(a0, 4),
        "wall_clock_seconds": round(time.time() - t0, 1),
        "scores_artifact": str(OUT_SCORES.relative_to(ROOT)),
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
