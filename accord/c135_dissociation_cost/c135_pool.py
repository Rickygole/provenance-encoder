"""C135 shared pool builder. Public SNLI only; opens no restricted clinical path, no network host.
See C135_PREREGISTRATION.md.
"""
import hashlib
import pathlib

import numpy as np
import pandas as pd

SEED = 20260926
RATES = (0.0, 0.10, 0.25, 0.414, 0.60)
POOLS = {"low": 0.0401, "mid": 0.50}
POOL_N = 8000
HELDOUT_N = 2000
SNLI = pathlib.Path.home() / (
    ".cache/huggingface/hub/datasets--stanfordnlp--snli/snapshots/"
    "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b/plain_text/train-00000-of-00001.parquet"
)
CONTRADICTION = 2


def load_groups():
    """Premise groups with exactly one contradiction and two non-contradiction hypotheses."""
    df = pd.read_parquet(SNLI)
    df = df[df["label"] >= 0].reset_index(drop=True)
    df["pos"] = df.groupby("premise").cumcount()  # position within the group, input row order
    sz = df.groupby("premise")["label"].agg(lambda s: tuple(sorted(s)))
    keep = set(sz.index[sz == (0, 1, 2)])
    df = df[df["premise"].isin(keep)].sort_values(["premise", "pos"], kind="stable")
    groups = []
    for premise, g in df.groupby("premise", sort=True):
        labels = g["label"].tolist()
        groups.append(
            {
                "premise": premise,
                "hypotheses": g["hypothesis"].tolist(),
                "labels": labels,
                "trigger_idx": [i for i, la in enumerate(labels) if la == CONTRADICTION],
                "nontrigger_idx": [i for i, la in enumerate(labels) if la != CONTRADICTION],
            }
        )
    groups.sort(key=lambda r: hashlib.sha256(r["premise"].encode()).hexdigest())
    return groups


def build_pool(groups, offset, n_total, base_rate):
    """Return (candidates, n_tp). TP and FP candidates come from disjoint premise groups."""
    n_tp = int(round(base_rate * n_total))
    n_fp = n_total - n_tp
    tp_groups = groups[offset : offset + n_tp]
    fp_groups = groups[offset + n_tp : offset + n_tp + n_fp]
    cands = []
    for g in tp_groups:
        cands.append(
            {
                "premise": g["premise"],
                "hypotheses": g["hypotheses"],
                "labels": g["labels"],
                "trigger_idx": list(g["trigger_idx"]),
                "nontrigger_idx": list(g["nontrigger_idx"]),
                "true": 1,
            }
        )
    for g in fp_groups:
        # extraction error: the contradiction hypothesis is absent from the group; one of the
        # remaining hypotheses is mis-read as conflicting, which admits the candidate.
        nt = list(g["nontrigger_idx"])
        cands.append(
            {
                "premise": g["premise"],
                "hypotheses": [g["hypotheses"][i] for i in nt],
                "labels": [g["labels"][i] for i in nt],
                "trigger_idx": [],
                "nontrigger_idx": list(range(len(nt))),
                "true": 0,
            }
        )
    return cands, n_tp, offset + n_tp + n_fp


def plant_order(n_tp, seed=SEED):
    """One seeded permutation of the true-positive slots, fixed once per pool: nested plant sets."""
    return np.random.default_rng(seed).permutation(n_tp)


def display_for(cands, n_tp, order, rate):
    """Displayed hypothesis index per candidate at the given planted rate, plus the planted mask."""
    n_plant = int(round(rate * n_tp))
    planted = np.zeros(len(cands), dtype=bool)
    planted[order[:n_plant]] = True
    disp = []
    for i, c in enumerate(cands):
        if c["true"] == 1 and planted[i]:
            disp.append(c["nontrigger_idx"][0])  # original positional selector: first non-trigger
        elif c["true"] == 1:
            disp.append(c["trigger_idx"][0])  # rebuild selector: a trigger-set member
        else:
            disp.append(0)  # the mis-read hypothesis, first by position
    return np.array(disp), planted, n_plant


def confirm(cands, disp):
    """The judge reading only the displayed evidence: gold label of the displayed pair."""
    return np.array(
        [1 if c["labels"][d] == CONTRADICTION else 0 for c, d in zip(cands, disp)], dtype=int
    )


def text(cands, disp):
    return [c["premise"] + " [SEP] " + c["hypotheses"][d] for c, d in zip(cands, disp)]


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)
