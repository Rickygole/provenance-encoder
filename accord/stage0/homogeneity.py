"""
Stage 0 re-diagnosis: model-free homogeneity test (step 5).

Directly tests H3 ("clinical homogeneity") against H2 ("header truncation")
by measuring whether raw, untrained BioMedBERT CLS-embedding cosine
similarity can separate matched (nursing, discharge) pairs from mismatched
pairs, under head-truncation vs. a non-header ("tail") window. This is a
model-free measure: no ATCG, no classifier head, no training at all -- just
"does the signal exist in the frozen embedding space".

If matched/mismatched cosine-similarity distributions overlap almost
completely under head truncation but separate under the tail window, H2 is
confirmed and H3 (genuine clinical homogeneity) is refuted -- the notes
*do* carry admission-specific signal, it's just not in the header block that
head-truncation exposes.

Writes an aggregate-only JSON (means/stds/percentiles/AUROC). NEVER writes
note text or per-note embeddings that could be inverted to text.
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer

from data import build_subject_split, build_fixed_pairs, tokenize_split

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MIMIC_LOCAL_ONLY_DIR = os.environ.get("MIMIC_LOCAL_ONLY_DIR", os.path.expanduser("~/mimic_local_only"))
OUT_PATH = os.path.join(MIMIC_LOCAL_ONLY_DIR, "accord_stage0/homogeneity.json")
PAIR_SEED = 3033


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def embed_cls(backbone, ids, mask, device, batch_size=32):
    backbone.eval()
    embs = []
    for i in range(0, ids.shape[0], batch_size):
        b_ids = ids[i:i + batch_size].to(device)
        b_mask = mask[i:i + batch_size].to(device)
        out = backbone(input_ids=b_ids, attention_mask=b_mask)
        cls = out.last_hidden_state[:, 0, :]
        embs.append(cls.cpu())
    return torch.cat(embs, dim=0)


def cosine_sim_matrix_pairs(nursing_emb, discharge_emb, pairs):
    n = torch.nn.functional.normalize(nursing_emb, dim=-1)
    d = torch.nn.functional.normalize(discharge_emb, dim=-1)
    sims, labels = [], []
    for n_idx, d_idx, label in pairs:
        sims.append(float((n[n_idx] * d[d_idx]).sum()))
        labels.append(label)
    return np.array(sims), np.array(labels)


def summarize(sims):
    return {
        "n": int(len(sims)),
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "p5": float(np.percentile(sims, 5)),
        "p25": float(np.percentile(sims, 25)),
        "p50": float(np.percentile(sims, 50)),
        "p75": float(np.percentile(sims, 75)),
        "p95": float(np.percentile(sims, 95)),
    }


def main():
    device = get_device()
    print(f"[homogeneity] device={device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME).to(device)

    train_df, dev_df = build_subject_split()
    full_df = train_df._append(dev_df, ignore_index=True) if hasattr(train_df, "_append") \
        else train_df.append(dev_df, ignore_index=True)
    print(f"[homogeneity] full sample admissions: {len(full_df)}")

    pairs = build_fixed_pairs(len(full_df), seed=PAIR_SEED)
    n_matched = sum(1 for _, _, l in pairs if l == 1)
    n_mismatched = sum(1 for _, _, l in pairs if l == 0)
    print(f"[homogeneity] matched={n_matched} mismatched={n_mismatched}")

    results = {}
    for window in ["head", "tail"]:
        print(f"[homogeneity] tokenizing window={window} ...")
        tok = tokenize_split(full_df, tokenizer, window)

        print(f"[homogeneity] embedding nursing notes (window={window}) ...")
        nursing_emb = embed_cls(backbone, tok.nursing_ids, tok.nursing_mask, device)
        print(f"[homogeneity] embedding discharge notes (window={window}) ...")
        discharge_emb = embed_cls(backbone, tok.discharge_ids, tok.discharge_mask, device)

        sims, labels = cosine_sim_matrix_pairs(nursing_emb, discharge_emb, pairs)
        matched_sims = sims[labels == 1]
        mismatched_sims = sims[labels == 0]

        auroc = roc_auc_score(labels, sims)

        results[window] = {
            "matched": summarize(matched_sims),
            "mismatched": summarize(mismatched_sims),
            "auroc_cosine_separates_matched": float(auroc),
        }
        print(f"[homogeneity] window={window} matched_mean={results[window]['matched']['mean']:.4f} "
              f"mismatched_mean={results[window]['mismatched']['mean']:.4f} "
              f"AUROC={auroc:.4f}")

    with open(OUT_PATH, "w") as f:
        json.dump({
            "n_admissions": len(full_df),
            "n_matched": n_matched,
            "n_mismatched": n_mismatched,
            "results": results,
        }, f, indent=2)
    print(f"[homogeneity] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
