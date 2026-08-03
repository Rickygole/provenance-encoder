"""Train a WordPiece tokenizer from scratch on the bundle corpus.

From scratch means from scratch: no pretrained vocabulary. Role tokens are added
as special tokens so arm A2 can express provenance as literal text, which is the
decisive baseline -- if A2 ties A4, six vocabulary entries did the work and there
is no architectural contribution.

Local only; the corpus is MIMIC-derived and never leaves the machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.roles import ROLE_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BUNDLES = ROOT / "local_only" / "bundles"
OUT = ROOT / "local_only" / "tokenizer"

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
ROLE_TOKENS = [f"[ROLE={n}]" for i, n in ROLE_NAMES.items() if n not in ("PAD",)]
TIME_TOKENS = [f"[AGE={b}]" for b in ("0h", "6h", "24h", "72h", "7d", "gt7d")]


def iter_texts(split: str):
    p = BUNDLES / f"{split}.jsonl"
    with p.open() as f:
        for line in f:
            b = json.loads(line)
            for s in b["segments"]:
                yield s["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=32_000)
    ap.add_argument("--min-frequency", type=int, default=3)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(),
                                           normalizers.StripAccents()])
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece()

    trainer = trainers.WordPieceTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIALS + ROLE_TOKENS + TIME_TOKENS,
    )
    print("training tokenizer on train split...", flush=True)
    tok.train_from_iterator(iter_texts("train"), trainer=trainer)

    path = OUT / "wordpiece.json"
    tok.save(str(path))
    print(f"vocab size: {tok.get_vocab_size():,}")
    print(f"wrote {path}")

    ids = {t: tok.token_to_id(t) for t in SPECIALS + ROLE_TOKENS + TIME_TOKENS}
    (OUT / "special_ids.json").write_text(json.dumps(ids, indent=2))
    print("special token ids:", json.dumps(ids))


if __name__ == "__main__":
    main()
