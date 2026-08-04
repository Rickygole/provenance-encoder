"""Train a WordPiece tokenizer from scratch on the bundle corpus.

From scratch means from scratch: no pretrained vocabulary. Role tokens are added
as special tokens so arm A2 can express provenance as literal text, which is the
decisive baseline -- if A2 ties A4, six vocabulary entries did the work and there
is no architectural contribution.

DEPENDENCY (part of the pretraining-corpus contamination fix; see
scripts/build_bundles.py's docstring and src/data/lineage.py). The corpus this
reads (local_only/bundles/train.jsonl) is itself derived from the downstream
task's train-split subjects only. This script stamps a lineage.json alongside
the trained vocab recording the exact bundle-file content it was trained on
(sha256) and the task-split hash that corpus was built from, so
scripts/pretrain.py can refuse to run if the bundles are rebuilt (new content)
without retraining the tokenizer, or if the task split changes without the
whole chain being rebuilt.

Local only; the corpus is MIMIC-derived and never leaves the machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.lineage import hash_file  # noqa: E402
from src.data.roles import ROLE_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLES = ROOT / "local_only" / "bundles"
DEFAULT_OUT = ROOT / "local_only" / "tokenizer"

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
ROLE_TOKENS = [f"[ROLE={n}]" for i, n in ROLE_NAMES.items() if n not in ("PAD",)]
TIME_TOKENS = [f"[AGE={b}]" for b in ("0h", "6h", "24h", "72h", "7d", "gt7d")]


def iter_texts(bundles_dir: Path, split: str):
    p = bundles_dir / f"{split}.jsonl"
    with p.open() as f:
        for line in f:
            b = json.loads(line)
            for s in b["segments"]:
                yield s["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=32_000)
    ap.add_argument("--min-frequency", type=int, default=3)
    ap.add_argument("--bundles-dir", default=str(DEFAULT_BUNDLES),
                    help="override for smoke-testing against a scratch corpus")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="override for smoke-testing; the real path is read by "
                         "scripts/pretrain.py by default")
    args = ap.parse_args()
    BUNDLES = Path(args.bundles_dir)
    OUT = Path(args.out)

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
    print(f"training tokenizer on train split of {BUNDLES}...", flush=True)
    tok.train_from_iterator(iter_texts(BUNDLES, "train"), trainer=trainer)

    path = OUT / "wordpiece.json"
    tok.save(str(path))
    print(f"vocab size: {tok.get_vocab_size():,}")
    print(f"wrote {path}")

    ids = {t: tok.token_to_id(t) for t in SPECIALS + ROLE_TOKENS + TIME_TOKENS}
    (OUT / "special_ids.json").write_text(json.dumps(ids, indent=2))
    print("special token ids:", json.dumps(ids))

    # Lineage stamp -- see src/data/lineage.py. pretrain.py recomputes both
    # hashes fresh at startup and refuses to run if either is stale.
    bundles_meta_path = BUNDLES / "meta.json"
    task_split_hash = None
    if bundles_meta_path.exists():
        task_split_hash = json.loads(bundles_meta_path.read_text()).get("task_split_hash")
    if task_split_hash is None:
        print("WARNING: local_only/bundles/meta.json has no task_split_hash "
              "(corpus built before the contamination fix, or built with "
              "--out pointed elsewhere). pretrain.py's lineage guard will "
              "refuse to run against this tokenizer until the bundles are "
              "rebuilt with the current scripts/build_bundles.py.")
    train_jsonl = BUNDLES / "train.jsonl"
    lineage = {
        "bundles_dir": str(BUNDLES),
        "bundles_content_hash": hash_file(train_jsonl) if train_jsonl.exists() else None,
        "task_split_hash": task_split_hash,
        "vocab_size": tok.get_vocab_size(),
    }
    (OUT / "lineage.json").write_text(json.dumps(lineage, indent=2))
    print(f"wrote {OUT / 'lineage.json'}: {json.dumps(lineage)}")


if __name__ == "__main__":
    main()
