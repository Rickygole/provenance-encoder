"""C135 main run: bias in measured precision, student AUROC, assertion check.
See C135_PREREGISTRATION.md. Public SNLI only.

    python3 accord/c135_dissociation_cost/c135_run.py
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import c135_pool as P  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results/accord/c135_dissociation_cost"
N_BOOT = 10000


def paired_bootstrap_gap(confirmed, true, seed=P.SEED, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    n = len(confirmed)
    idx = rng.integers(0, n, size=(n_boot, n))
    g = confirmed[idx].mean(axis=1) - true[idx].mean(axis=1)
    lo, hi = np.percentile(g, [2.5, 97.5])
    return float(lo), float(hi)


# Tokens are assembled from fragments so that this auditor's own source does not contain the
# literals it forbids, which would make the check unable to pass.
FORBIDDEN = tuple(
    a + b
    for a, b in (
        ("local", "_only"), ("mim", "ic"), ("HADM", "_ID"), ("SUBJECT", "_ID"),
        ("bed", "rock"), ("open", "ai"), ("anthro", "pic"), ("physio", "net"),
    )
)


def data_provenance_audit():
    """Kill condition 4, checked rather than asserted: the check's own sources carry no restricted
    clinical path and no hosted-API host. Returns the hits found, which must be empty."""
    here = pathlib.Path(__file__).resolve().parent
    hits = []
    for f in sorted(here.glob("c135_*.py")):
        low = f.read_text().lower()
        hits += [f"{f.name}:{t}" for t in FORBIDDEN if t.lower() in low]
    return sorted(set(hits))


def main():
    groups = P.load_groups()
    n_groups = len(groups)
    cursor = 0
    pools = {}
    for name, base in P.POOLS.items():
        cands, n_tp, cursor = P.build_pool(groups, cursor, P.POOL_N, base)
        pools[name] = {"cands": cands, "n_tp": n_tp, "order": P.plant_order(n_tp)}
    held_cands, held_n_tp, cursor = P.build_pool(groups, cursor, P.HELDOUT_N, P.POOLS["mid"])
    held_disp, _, _ = P.display_for(held_cands, held_n_tp, P.plant_order(held_n_tp), 0.0)
    held_text = P.text(held_cands, held_disp)
    held_true = np.array([c["true"] for c in held_cands], dtype=int)

    out = {
        "preregistration": "accord/c135_dissociation_cost/C135_PREREGISTRATION.md",
        "seed": P.SEED,
        "data": {
            "source": "stanfordnlp/snli train split, local Hugging Face cache, read only",
            "parquet": str(P.SNLI),
            "n_usable_premise_groups": n_groups,
            "group_filter": "exactly three hypotheses with gold label multiset (0, 1, 2)",
            "provenance_audit_hits": data_provenance_audit(),
            "snli_parquet_under_hf_cache": str(P.SNLI).startswith(
                str(pathlib.Path.home() / ".cache/huggingface")
            ),
        },
        "pool_n": P.POOL_N,
        "heldout_n": P.HELDOUT_N,
        "rates": list(P.RATES),
        "pools": {},
        "assertion_check": {},
        "kill_conditions": {},
    }

    void = []
    if out["data"]["provenance_audit_hits"]:
        void.append(f"kill4: provenance audit hits {out['data']['provenance_audit_hits']}")
    if not out["data"]["snli_parquet_under_hf_cache"]:
        void.append("kill4: SNLI parquet is not under the local Hugging Face cache")
    for name, pk in pools.items():
        cands, n_tp, order = pk["cands"], pk["n_tp"], pk["order"]
        true = np.array([c["true"] for c in cands], dtype=int)
        realized = float(true.mean())
        if abs(realized - P.POOLS[name]) > 0.0005:
            void.append(f"kill3 {name}: realized base rate {realized} off target {P.POOLS[name]}")
        block = {
            "target_base_rate": P.POOLS[name],
            "realized_true_precision": realized,
            "n_true_positive": int(n_tp),
            "by_rate": [],
        }
        train_texts, train_labels_a, train_labels_b = {}, {}, {}
        for r in P.RATES:
            disp, planted, n_plant = P.display_for(cands, n_tp, order, r)
            conf = P.confirm(cands, disp)
            mp = float(conf.mean())
            tp_rate = float(true.mean())
            lo, hi = paired_bootstrap_gap(conf, true)
            mw = P.wilson(int(conf.sum()), len(conf))
            tw = P.wilson(int(true.sum()), len(true))
            # assertion check, on true-positive candidates only
            flagged, tp_idx = [], [i for i, c in enumerate(cands) if c["true"] == 1]
            for i in tp_idx:
                if int(disp[i]) not in cands[i]["trigger_idx"]:
                    flagged.append(i)
            planted_set = set(np.flatnonzero(planted).tolist())
            fp_flagged = sum(
                1 for i, c in enumerate(cands) if c["true"] == 0 and int(disp[i]) not in c["trigger_idx"]
            )
            ok = set(flagged) == planted_set
            if not ok:
                void.append(f"kill2 {name} r={r}: flagged != planted")
            if r == 0.0 and abs(mp - tp_rate) > 1e-12:
                void.append(f"kill1 {name}: gap at rate 0 is {mp - tp_rate}, not 0")
            train_texts[r] = P.text(cands, disp)
            train_labels_a[r] = conf
            train_labels_b[r] = true
            block["by_rate"].append(
                {
                    "planted_rate": r,
                    "n_planted": int(n_plant),
                    "realized_planted_share_of_true_positives": float(n_plant / n_tp),
                    "measured_precision": mp,
                    "measured_precision_wilson_lo": mw[0],
                    "measured_precision_wilson_hi": mw[1],
                    "true_precision": tp_rate,
                    "true_precision_wilson_lo": tw[0],
                    "true_precision_wilson_hi": tw[1],
                    "gap_measured_minus_true": mp - tp_rate,
                    "gap_boot_lo": lo,
                    "gap_boot_hi": hi,
                    "relative_bias": (mp - tp_rate) / tp_rate if tp_rate else float("nan"),
                    # closed form under this planting model: every planted true positive loses its
                    # confirmation and nothing else moves, so gap = -(n_planted / pool_n).
                    "analytic_gap": -n_plant / len(cands),
                    "analytic_gap_residual": (mp - tp_rate) - (-n_plant / len(cands)),
                    # would a study reporting only measured precision cover the truth?
                    "measured_wilson_covers_true_precision": bool(mw[0] <= tp_rate <= mw[1]),
                    "assertion_flagged_on_true_positives": len(flagged),
                    "assertion_flagged_equals_planted_set": bool(ok),
                    "assertion_flagged_on_false_positives": int(fp_flagged),
                    "n_false_positive": int(len(cands) - n_tp),
                }
            )
        # student arm, mid pool only (pre-registered)
        if name == "mid":
            students = []
            for r in P.RATES:
                row = {"planted_rate": r}
                for tag, y in (("A_displayed_evidence_labels", train_labels_a[r]),
                               ("B_candidate_fired_labels", train_labels_b[r])):
                    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
                    X = vec.fit_transform(train_texts[r])
                    if len(set(y.tolist())) < 2:
                        row[tag] = {"auroc": float("nan"), "train_positive_rate": float(y.mean()),
                                    "note": "single-class training labels, model not fit"}
                        continue
                    clf = LogisticRegression(max_iter=2000, random_state=P.SEED).fit(X, y)
                    s = clf.predict_proba(vec.transform(held_text))[:, 1]
                    row[tag] = {
                        "auroc": float(roc_auc_score(held_true, s)),
                        "train_positive_rate": float(y.mean()),
                    }
                students.append(row)
            block["student"] = {
                "features": "TfidfVectorizer word 1-2 grams min_df=2 over 'premise [SEP] displayed hypothesis'",
                "model": "LogisticRegression max_iter=2000",
                "heldout": "2000 candidates from disjoint premises, rate 0, trigger-based exemplars",
                "heldout_true_positive_rate": float(held_true.mean()),
                "by_rate": students,
            }
        out["pools"][name] = block

    out["kill_conditions"] = {"violations": void, "void": bool(void)}
    out["outcome"] = None
    head = [b for b in out["pools"]["low"]["by_rate"] if b["planted_rate"] == 0.414][0]
    g, lo, hi = head["gap_measured_minus_true"], head["gap_boot_lo"], head["gap_boot_hi"]
    if void:
        out["outcome"] = "VOID"
    elif g < 0 and hi < 0:
        out["outcome"] = "A"
    elif lo <= 0 <= hi:
        out["outcome"] = "B"
    elif g > 0 and lo > 0:
        out["outcome"] = "C"
    else:
        out["outcome"] = "B"
    out["headline"] = {
        "pool": "low",
        "planted_rate": 0.414,
        "gap_measured_minus_true": g,
        "gap_boot_lo": lo,
        "gap_boot_hi": hi,
        "understatement": -g,
        "understatement_lo": -hi,
        "understatement_hi": -lo,
        "relative_bias": head["relative_bias"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c135_bias_curve.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"outcome={out['outcome']} void={out['kill_conditions']['void']} "
          f"gap@0.414(low)={g:.6f} [{lo:.6f}, {hi:.6f}]")
    for name in out["pools"]:
        for b in out["pools"][name]["by_rate"]:
            print(f"  {name} r={b['planted_rate']:.3f} measured={b['measured_precision']:.6f} "
                  f"true={b['true_precision']:.6f} gap={b['gap_measured_minus_true']:+.6f} "
                  f"assert_ok={b['assertion_flagged_equals_planted_set']}")


if __name__ == "__main__":
    main()
