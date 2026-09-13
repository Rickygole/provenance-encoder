"""C104: build the item-11 manifest (indices and hashes only, never text).

Public data only: SNLI parquet files under
accord/prevalence_replication/data/ and HANS's heuristics_evaluation_set.txt
under accord/public_controls/nlifactorial/data/ (downloaded once from the
official McCoy repository, MIT licensed, cited as mccoy2019hans). This script
never opens anything under mimic_local_only.

Exclusion of SNLI items already used by accord/prevalence_replication's pools
is done by reproducing build_pools.py's exact construction (same seed 42,
same combine-dedupe-shuffle steps) rather than reading pool_*.csv content, so
the exclusion is exact and does not require diffing text.

Partitions built here (SNLI, by original row index into the deduped,
label!=-1 combined test+validation frame; HANS, by pairID):
  - positives      1,000 SNLI contradiction (label==2)
  - easy           1,000 SNLI entailment   (label==0)
  - medium         1,000 SNLI neutral      (label==1)
  - hard           1,000 HANS entailment, stratified by heuristic (~334/333/333
                    across lexical_overlap / subsequence / constituent)
  - pilot          50 items per (SNLI positives/easy/medium; HANS hard) drawn
                    from a partition disjoint from the four above, excluded
                    from analysis
  - determinism    200 items drawn from the four main partitions (positives/
                    easy/medium/hard combined), re-run at cell start and end

Everything is written as SNLI row indices (into the reproducible combined
frame) or HANS pairIDs, plus a sha256 of (dataset, split, row index) as a
tamper check -- never premise/hypothesis/sentence1/sentence2 text.
"""
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "accord" / "prevalence_replication" / "data"
HANS_PATH = pathlib.Path(__file__).resolve().parent / "data" / "heuristics_evaluation_set.txt"
OUT_DIR = pathlib.Path(__file__).resolve().parent
MANIFEST_PATH = OUT_DIR / "manifest.json"

SEED = 42          # matches build_pools.py exactly
POOL_N = 100
DISTILL_POOL_N = 300
PREVALENCE_LEVELS = [0.02, 0.05, 0.10, 0.25]

N_MAIN = 1000      # per hardness level
N_PILOT = 50       # per hardness level, pilot partition
N_DETERMINISM = 200


def _hash_row(dataset, split, row_idx):
    return hashlib.sha256(f"{dataset}::{split}::{row_idx}".encode()).hexdigest()[:16]


def reproduce_snli_pool_cursors():
    """Exactly reproduce build_pools.py's combine/dedupe/shuffle and return
    (df, pos_pool_idx, neg_pool_idx, pos_used, neg_used) where pos_used/
    neg_used are the counts build_pools.py actually consumed (sweep pools
    plus the distillation-supplement pools), so this script's own draws can
    start strictly after them.
    """
    t = pd.read_parquet(DATA_DIR / "snli_test.parquet")
    v = pd.read_parquet(DATA_DIR / "snli_validation.parquet")
    df = pd.concat([t, v], ignore_index=True)
    df = df[df["label"] != -1].reset_index(drop=True)
    df = df.drop_duplicates(subset=["premise", "hypothesis"]).reset_index(drop=True)
    df["is_contradiction"] = (df["label"] == 2).astype(int)

    rng = np.random.default_rng(SEED)
    pos_pool_idx = df.index[df["is_contradiction"] == 1].to_numpy()
    neg_pool_idx = df.index[df["is_contradiction"] == 0].to_numpy()
    rng.shuffle(pos_pool_idx)
    rng.shuffle(neg_pool_idx)

    pos_cursor = neg_cursor = 0
    for p in PREVALENCE_LEVELS:
        n_pos, n_neg = round(POOL_N * p), POOL_N - round(POOL_N * p)
        pos_cursor += n_pos
        neg_cursor += n_neg
    for p in PREVALENCE_LEVELS:
        n_pos, n_neg = round(DISTILL_POOL_N * p), DISTILL_POOL_N - round(DISTILL_POOL_N * p)
        pos_cursor += n_pos
        neg_cursor += n_neg

    return df, pos_pool_idx, neg_pool_idx, pos_cursor, neg_cursor


def build():
    df, pos_pool_idx, neg_pool_idx, pos_used, neg_used = reproduce_snli_pool_cursors()
    avail_pos_idx = pos_pool_idx[pos_used:]              # available contradiction rows
    avail_neg_idx = neg_pool_idx[neg_used:]              # available non-contradiction rows

    avail_neg_labels = df.loc[avail_neg_idx, "label"].to_numpy()
    avail_entail_idx = avail_neg_idx[avail_neg_labels == 0]
    avail_neutral_idx = avail_neg_idx[avail_neg_labels == 1]

    n_needed_pos = N_MAIN + N_PILOT
    n_needed_neg_each = N_MAIN + N_PILOT
    if len(avail_pos_idx) < n_needed_pos:
        raise SystemExit(f"not enough available SNLI contradiction rows: "
                          f"need {n_needed_pos}, have {len(avail_pos_idx)}")
    if len(avail_entail_idx) < n_needed_neg_each:
        raise SystemExit(f"not enough available SNLI entailment rows: "
                          f"need {n_needed_neg_each}, have {len(avail_entail_idx)}")
    if len(avail_neutral_idx) < n_needed_neg_each:
        raise SystemExit(f"not enough available SNLI neutral rows: "
                          f"need {n_needed_neg_each}, have {len(avail_neutral_idx)}")

    # A second, independent RNG for this script's own draws, seeded
    # differently from build_pools.py's, but still fixed and committed, so
    # this manifest is itself reproducible from the same source parquets.
    rng2 = np.random.default_rng(SEED + 1)

    def draw(idx_pool, n_main, n_pilot):
        idx_pool = idx_pool.copy()
        rng2.shuffle(idx_pool)
        return idx_pool[:n_main], idx_pool[n_main:n_main + n_pilot]

    pos_main, pos_pilot = draw(avail_pos_idx, N_MAIN, N_PILOT)
    easy_main, easy_pilot = draw(avail_entail_idx, N_MAIN, N_PILOT)
    med_main, med_pilot = draw(avail_neutral_idx, N_MAIN, N_PILOT)

    # HANS hard partition: entailment-labelled only, stratified by heuristic.
    hans = pd.read_csv(HANS_PATH, sep="\t")
    hans_ent = hans[hans["gold_label"] == "entailment"].reset_index(drop=True)
    heuristics = sorted(hans_ent["heuristic"].unique())
    per_h_main = N_MAIN // len(heuristics)
    per_h_pilot = N_PILOT // len(heuristics)
    remainder_main = N_MAIN - per_h_main * len(heuristics)
    remainder_pilot = N_PILOT - per_h_pilot * len(heuristics)
    hard_main_ids, hard_pilot_ids = [], []
    for i, h in enumerate(heuristics):
        sub = hans_ent[hans_ent["heuristic"] == h]
        pair_ids = sub["pairID"].to_numpy().copy()
        rng2.shuffle(pair_ids)
        take_main = per_h_main + (1 if i < remainder_main else 0)
        take_pilot = per_h_pilot + (1 if i < remainder_pilot else 0)
        hard_main_ids.extend(pair_ids[:take_main].tolist())
        hard_pilot_ids.extend(pair_ids[take_main:take_main + take_pilot].tolist())
    assert len(hard_main_ids) == N_MAIN, len(hard_main_ids)
    assert len(hard_pilot_ids) == N_PILOT, len(hard_pilot_ids)

    # Determinism-check partition: 200 items drawn from the four MAIN
    # partitions combined (not pilot), one fixed draw, reused at both the
    # start and the end of every cell.
    combo = ([("snli", int(i)) for i in pos_main.tolist()]
             + [("snli", int(i)) for i in easy_main.tolist()]
             + [("snli", int(i)) for i in med_main.tolist()]
             + [("hans", pid) for pid in hard_main_ids])
    combo_arr = np.array(combo, dtype=object)
    det_sel = rng2.choice(len(combo_arr), size=N_DETERMINISM, replace=False)
    determinism_items = [{"dataset": d, "row": r} for d, r in combo_arr[det_sel]]

    def snli_items(idx_array, label_name):
        return [{"dataset": "snli", "row": int(i), "label": label_name,
                  "hash": _hash_row("snli", "test+validation_dedup", int(i))}
                for i in idx_array.tolist()]

    def hans_items(pair_ids):
        return [{"dataset": "hans", "pairID": pid, "label": "entailment",
                  "hash": _hash_row("hans", "heuristics_evaluation_set", pid)}
                for pid in pair_ids]

    manifest = {
        "provenance": {
            "generated_by": "accord/public_controls/nlifactorial/build_manifest.py",
            "snli_source": "accord/prevalence_replication/data/snli_{test,validation}.parquet",
            "hans_source": "accord/public_controls/nlifactorial/data/heuristics_evaluation_set.txt",
            "hans_citation": "mccoy2019hans",
            "excluded_snli_rows": {"positives_used_by_pools": int(pos_used),
                                    "negatives_used_by_pools": int(neg_used)},
            "seed_pools_reproduction": SEED, "seed_manifest_draw": SEED + 1,
        },
        "main": {
            "positives": snli_items(pos_main, "contradiction"),
            "easy": snli_items(easy_main, "entailment"),
            "medium": snli_items(med_main, "neutral"),
            "hard": hans_items(hard_main_ids),
        },
        "pilot": {
            "positives": snli_items(pos_pilot, "contradiction"),
            "easy": snli_items(easy_pilot, "entailment"),
            "medium": snli_items(med_pilot, "neutral"),
            "hard": hans_items(hard_pilot_ids),
        },
        "determinism_check": determinism_items,
        "counts": {
            "main": {k: len(v) for k, v in
                     {"positives": pos_main, "easy": easy_main, "medium": med_main,
                      "hard": hard_main_ids}.items()},
            "pilot": {"positives": len(pos_pilot), "easy": len(easy_pilot),
                      "medium": len(med_pilot), "hard": len(hard_pilot_ids)},
            "determinism_check": len(determinism_items),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {MANIFEST_PATH}")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    build()
