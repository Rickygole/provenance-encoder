"""
Build prevalence-swept pools for the non-clinical replication of the ACCORD
weak-labeller over-calling finding (see accord/CORRECTIONS.md C20/C23).

DATASET: SNLI (Stanford Natural Language Inference), test + validation splits,
downloaded from the public HF hub parquet mirror (huggingface.co/datasets/
stanfordnlp/snli). Human-annotated (Amazon Mechanical Turk, 5-way majority
vote in the original collection), premise/hypothesis sentence pairs, gold
label in {entailment=0, neutral=1, contradiction=2, no-gold=-1}.

WHY THIS DATASET.
  (a) Human ground truth: yes, MTurk-annotated gold labels, standard benchmark.
  (b) Short text units comparable to the clinical sentence pairs: premise and
      hypothesis are both single sentences (clinical task: nursing-note
      sentence vs discharge-summary sentence).
  (c) Same binary judgment framed the same way: "is there a contradiction
      between these two short texts?" -- this is structurally the *same*
      task as the clinical weak-labelling prompt, just non-clinical content.
      Non-contradiction (entailment + neutral, pooled) is the negative class.
  (d) Base rate is constructible: SNLI is balanced ~33/33/33 by construction,
      so contradiction pairs and non-contradiction pairs are both abundant,
      letting us subsample to ANY target prevalence -- including the clinical
      pipeline's ~4% -- with plenty of headroom. This is exactly what the
      clinical data cannot do (it has one fixed prevalence, ~4%).

POOLS: 4 prevalence levels x N=100 items, DISJOINT across levels (no
premise/hypothesis pair reused across levels), fixed seed 42.
"""
import os
import numpy as np
import pandas as pd

SEED = 42
POOL_N = 100
PREVALENCE_LEVELS = [0.02, 0.05, 0.10, 0.25]
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/data"
OUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/data/pools"
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.default_rng(SEED)

t = pd.read_parquet(os.path.join(DATA_DIR, "snli_test.parquet"))
v = pd.read_parquet(os.path.join(DATA_DIR, "snli_validation.parquet"))
df = pd.concat([t, v], ignore_index=True)
df = df[df["label"] != -1].reset_index(drop=True)
df = df.drop_duplicates(subset=["premise", "hypothesis"]).reset_index(drop=True)
df["is_contradiction"] = (df["label"] == 2).astype(int)

print(f"Combined SNLI test+validation, deduped: {len(df)} rows")
print(f"  entailment={int((df['label']==0).sum())}  "
      f"neutral={int((df['label']==1).sum())}  "
      f"contradiction={int((df['label']==2).sum())}")
print(f"  natural contradiction base rate = {df['is_contradiction'].mean():.4f}")

pos_pool_idx = df.index[df["is_contradiction"] == 1].to_numpy()
neg_pool_idx = df.index[df["is_contradiction"] == 0].to_numpy()
RNG.shuffle(pos_pool_idx)
RNG.shuffle(neg_pool_idx)

pos_cursor = 0
neg_cursor = 0
summary_rows = []

for p in PREVALENCE_LEVELS:
    n_pos = round(POOL_N * p)
    n_neg = POOL_N - n_pos
    pos_idx = pos_pool_idx[pos_cursor: pos_cursor + n_pos]
    neg_idx = neg_pool_idx[neg_cursor: neg_cursor + n_neg]
    pos_cursor += n_pos
    neg_cursor += n_neg
    assert len(pos_idx) == n_pos and len(neg_idx) == n_neg, \
        f"Not enough disjoint rows left for prevalence {p}"

    idx = np.concatenate([pos_idx, neg_idx])
    RNG.shuffle(idx)  # shuffle order so pos/neg aren't grouped
    pool = df.loc[idx, ["premise", "hypothesis", "is_contradiction"]].reset_index(drop=True)
    pool.insert(0, "item_id", [f"p{p:.2f}_{i:03d}" for i in range(len(pool))])

    achieved_rate = pool["is_contradiction"].mean()
    out_path = os.path.join(OUT_DIR, f"pool_prevalence_{p:.2f}.csv")
    pool.to_csv(out_path, index=False)
    print(f"prevalence target={p:.2%}  n={len(pool)}  n_pos={int(pool['is_contradiction'].sum())} "
          f"achieved_rate={achieved_rate:.4f}  -> {out_path}")
    summary_rows.append({
        "target_prevalence": p, "n": len(pool),
        "n_pos": int(pool["is_contradiction"].sum()),
        "achieved_prevalence": achieved_rate,
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(OUT_DIR, "pool_summary.csv"), index=False)
print("\nPool summary:")
print(summary.to_string(index=False))
print(f"\nTotal distinct items used across all pools: "
      f"{pos_cursor + neg_cursor} of {len(df)} available "
      f"(pos used {pos_cursor}/{len(pos_pool_idx)}, neg used {neg_cursor}/{len(neg_pool_idx)}) "
      f"-- disjoint by construction, no item reused across prevalence levels.")
