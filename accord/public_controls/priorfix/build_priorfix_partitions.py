"""C105: build disjoint SNLI partitions for the prior-fix distillation check.

Public data only (SNLI). Reproduces the same combine/dedupe/shuffle
construction accord/prevalence_replication/build_pools.py and
accord/public_controls/nlifactorial/build_manifest.py use, so exclusion of
every row either of those two has already used is exact (index-based), never
a text diff. Stores indices only, never premise/hypothesis text.

Partitions (all disjoint from each other AND from build_pools.py's pools AND
from nlifactorial's manifest):
  - teacher_positives     Item 11's own manifest["main"]["positives"] (the
                          SAME 1,000 rows llama3:8b/P1 already scored at full
                          scale in Item 11 -- reused by row index, not
                          redrawn, so this run does not re-label them).
  - teacher_topup_entail  1,000 SNLI entailment rows, newly drawn, to be
                          labelled by llama3:8b/P1 here.
  - teacher_topup_neutral 1,000 SNLI neutral rows, newly drawn, to be
                          labelled by llama3:8b/P1 here.
  - gold_pos_pool         500 SNLI contradiction rows, sampled WITH
                          replacement per replicate to build each
                          replicate's 253-item gold-only draw.
  - gold_neg_pool         500 SNLI non-contradiction rows (mixed entailment
                          + neutral), same use.
  - eval_pos              190 SNLI contradiction rows (fixed evaluation set,
                          the teacher never labels these).
  - eval_neg              3,810 SNLI non-contradiction rows (fixed
                          evaluation set).
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from accord.public_controls.nlifactorial.build_manifest import (  # noqa: E402
    reproduce_snli_pool_cursors,
)

HERE = pathlib.Path(__file__).resolve().parent
NLIFACTORIAL_MANIFEST = (ROOT / "accord" / "public_controls" / "nlifactorial"
                          / "manifest.json")
OUT_PATH = HERE / "partitions.json"

SEED = 42 + 2   # distinct from build_pools.py (42) and nlifactorial (43)
N_TOPUP_EACH = 1000
N_GOLD_POOL_EACH = 500
N_EVAL_POS = 190
N_EVAL_NEG = 3810


def snli_rows_used_by_manifest(manifest):
    used = set()
    for section in ("main", "pilot"):
        for level, items in manifest[section].items():
            for it in items:
                if it["dataset"] == "snli":
                    used.add(it["row"])
    return used


def build():
    df, pos_pool_idx, neg_pool_idx, pos_used, neg_used = reproduce_snli_pool_cursors()
    manifest = json.loads(NLIFACTORIAL_MANIFEST.read_text())
    nli_used_rows = snli_rows_used_by_manifest(manifest)

    teacher_positives = [it["row"] for it in manifest["main"]["positives"]]
    assert set(teacher_positives) <= nli_used_rows

    avail_pos_idx = np.array([i for i in pos_pool_idx[pos_used:]
                               if i not in nli_used_rows])
    avail_neg_idx = np.array([i for i in neg_pool_idx[neg_used:]
                               if i not in nli_used_rows])
    avail_neg_labels = df.loc[avail_neg_idx, "label"].to_numpy()
    avail_entail_idx = avail_neg_idx[avail_neg_labels == 0]
    avail_neutral_idx = avail_neg_idx[avail_neg_labels == 1]

    need_pos = N_GOLD_POOL_EACH + N_EVAL_POS
    need_entail = N_TOPUP_EACH + N_EVAL_NEG // 2 + N_GOLD_POOL_EACH // 2 + 100
    need_neutral = N_TOPUP_EACH + N_EVAL_NEG // 2 + N_GOLD_POOL_EACH // 2 + 100
    if len(avail_pos_idx) < need_pos:
        raise SystemExit(f"not enough available contradiction rows: "
                          f"need {need_pos}, have {len(avail_pos_idx)}")
    if len(avail_entail_idx) < need_entail:
        raise SystemExit(f"not enough available entailment rows: "
                          f"need {need_entail}, have {len(avail_entail_idx)}")
    if len(avail_neutral_idx) < need_neutral:
        raise SystemExit(f"not enough available neutral rows: "
                          f"need {need_neutral}, have {len(avail_neutral_idx)}")

    rng = np.random.default_rng(SEED)
    pos_shuf = avail_pos_idx.copy(); rng.shuffle(pos_shuf)
    entail_shuf = avail_entail_idx.copy(); rng.shuffle(entail_shuf)
    neutral_shuf = avail_neutral_idx.copy(); rng.shuffle(neutral_shuf)

    cursor_pos = 0
    gold_pos_pool = pos_shuf[cursor_pos:cursor_pos + N_GOLD_POOL_EACH]; cursor_pos += N_GOLD_POOL_EACH
    eval_pos = pos_shuf[cursor_pos:cursor_pos + N_EVAL_POS]; cursor_pos += N_EVAL_POS

    cursor_e = 0
    topup_entail = entail_shuf[cursor_e:cursor_e + N_TOPUP_EACH]; cursor_e += N_TOPUP_EACH
    gold_neg_entail = entail_shuf[cursor_e:cursor_e + N_GOLD_POOL_EACH // 2]; cursor_e += N_GOLD_POOL_EACH // 2
    eval_neg_entail = entail_shuf[cursor_e:cursor_e + N_EVAL_NEG // 2]; cursor_e += N_EVAL_NEG // 2

    cursor_n = 0
    topup_neutral = neutral_shuf[cursor_n:cursor_n + N_TOPUP_EACH]; cursor_n += N_TOPUP_EACH
    gold_neg_neutral = neutral_shuf[cursor_n:cursor_n + N_GOLD_POOL_EACH // 2]; cursor_n += N_GOLD_POOL_EACH // 2
    eval_neg_neutral = neutral_shuf[cursor_n:cursor_n + N_EVAL_NEG // 2]; cursor_n += N_EVAL_NEG // 2

    gold_neg_pool = np.concatenate([gold_neg_entail, gold_neg_neutral])
    eval_neg = np.concatenate([eval_neg_entail, eval_neg_neutral])

    all_sets = [set(teacher_positives), set(topup_entail.tolist()),
                set(topup_neutral.tolist()), set(gold_pos_pool.tolist()),
                set(gold_neg_pool.tolist()), set(eval_pos.tolist()),
                set(eval_neg.tolist())]
    for i in range(len(all_sets)):
        for j in range(i + 1, len(all_sets)):
            assert all_sets[i].isdisjoint(all_sets[j]), f"partitions {i},{j} overlap"

    out = {
        "provenance": {
            "generated_by": "accord/public_controls/priorfix/build_priorfix_partitions.py",
            "seed": SEED,
            "excludes": "prevalence_replication pools + nlifactorial manifest (main+pilot)",
        },
        "teacher_positives_rows": [int(i) for i in teacher_positives],
        "teacher_topup_entail_rows": [int(i) for i in topup_entail.tolist()],
        "teacher_topup_neutral_rows": [int(i) for i in topup_neutral.tolist()],
        "gold_pos_pool_rows": [int(i) for i in gold_pos_pool.tolist()],
        "gold_neg_pool_rows": [int(i) for i in gold_neg_pool.tolist()],
        "eval_pos_rows": [int(i) for i in eval_pos.tolist()],
        "eval_neg_rows": [int(i) for i in eval_neg.tolist()],
    }
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")
    print({k: len(v) for k, v in out.items() if isinstance(v, list)})


if __name__ == "__main__":
    build()
