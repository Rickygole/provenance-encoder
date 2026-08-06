"""
Control experiment (not clinical data): is the ~0.92 cosine similarity we see
for BOTH matched and mismatched (nursing, discharge) pairs a property of
clinical notes specifically (H3, "clinical homogeneity"), or is it the
well-documented anisotropy of a raw, non-fine-tuned BERT CLS token, which
produces high cosine similarity between almost ANY two pieces of text
regardless of topic?

We embed a set of clearly UNRELATED, non-clinical, non-MIMIC sentences
(public-domain-style filler text spanning cooking, astronomy, sports,
finance, etc.) with the SAME frozen BioMedBERT CLS extraction used in
homogeneity.py, and report the pairwise cosine similarity distribution
across all unrelated pairs. If unrelated topics also cluster ~0.9, the
"homogeneity" observed on MIMIC notes is a representation artifact, not a
clinical-language property.
"""
import itertools
import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
OUT_PATH = "/Users/rickygole/mimic_local_only/accord_stage0/anisotropy_control.json"

# Deliberately unrelated, non-clinical, non-MIMIC sentences.
SENTENCES = [
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Saturn's rings are made mostly of ice particles and rocky debris.",
    "The stock market rallied sharply after the earnings report.",
    "She scored the winning goal in the final minute of the match.",
    "The volcano has been dormant for over three hundred years.",
    "He tuned his guitar before the concert began.",
    "The museum's new exhibit features Renaissance-era paintings.",
    "A gentle breeze moved through the wheat fields at dusk.",
    "The programmer fixed the bug in the sorting algorithm.",
    "Tourists lined up early to see the sunrise over the canyon.",
    "The committee voted to approve the new zoning regulations.",
    "The chef garnished the plate with a sprig of fresh mint.",
    "Migratory birds travel thousands of miles every autumn.",
    "The novelist spent a decade drafting her final manuscript.",
    "The engineers tested the bridge for structural fatigue.",
    "A rainbow appeared briefly after the summer storm.",
    "The orchestra rehearsed the symphony for six hours straight.",
    "Local farmers reported a strong harvest this season.",
    "The spacecraft entered orbit after a seven-month journey.",
    "The marathon route wound through five different neighborhoods.",
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def embed(tokenizer, backbone, sentences, device):
    enc = tokenizer(sentences, padding=True, truncation=True, max_length=64, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = backbone(**enc)
    return out.last_hidden_state[:, 0, :].cpu()


def main():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    embs = embed(tokenizer, backbone, SENTENCES, device)
    embs = torch.nn.functional.normalize(embs, dim=-1)

    sims = []
    for i, j in itertools.combinations(range(len(SENTENCES)), 2):
        sims.append(float((embs[i] * embs[j]).sum()))
    sims = np.array(sims)

    result = {
        "n_sentences": len(SENTENCES),
        "n_pairs": len(sims),
        "note": "unrelated non-clinical sentence pairs, same frozen CLS extraction as homogeneity.py",
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "p5": float(np.percentile(sims, 5)),
        "p50": float(np.percentile(sims, 50)),
        "p95": float(np.percentile(sims, 95)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
    }
    print(f"[anisotropy_control] unrelated-pair cosine sim: "
          f"mean={result['mean']:.4f} std={result['std']:.4f} "
          f"min={result['min']:.4f} max={result['max']:.4f}")

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[anisotropy_control] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
