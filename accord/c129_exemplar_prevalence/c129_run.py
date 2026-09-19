"""C129: prevalence of the original positional selector's design landing
outside the trigger set, measured on accord/phase3_mining.py's own rebuilt
candidate pool. See C129_PREREGISTRATION.md.

Local only; aggregates only. Never prints or writes sentence text, drug names
tied to a row, HADM_ID, SUBJECT_ID, or pair_id.

    python3 accord/c129_exemplar_prevalence/c129_run.py
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accord"))

from accord.cloud_paths import assert_local_only  # noqa: E402

MIMIC_LOCAL_ONLY_DIR = pathlib.Path(os.environ.get("MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
LOG_DIR = assert_local_only(MIMIC_LOCAL_ONLY_DIR / "logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG = LOG_DIR / f"c129_run_{time.strftime('%Y%m%dT%H%M%S')}.log"
OUT = ROOT / "results" / "accord" / "c129_exemplar_prevalence" / "c129.json"
RECORDED_SUMMARY = assert_local_only(MIMIC_LOCAL_ONLY_DIR / "accord" / "phase3_mining_summary.json")
MENTIONS_PATH = assert_local_only(MIMIC_LOCAL_ONLY_DIR / "accord" / "hf_medication_mentions.csv.gz")

Z = 1.959963984540054  # scipy.stats.norm.ppf(0.975)


def wilson(k: int, n: int, z: float = Z):
    """Wilson score interval for a binomial rate. Exact closed form, no
    resampling, no seed needed."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    lo = (center - half) / denom
    hi = (center + half) / denom
    return (max(0.0, float(lo)), min(1.0, float(hi)))


def mirror_rule1(classified: pd.DataFrame, mentions_by_note: dict, pm) -> list[dict]:
    """Re-runs mine_rule1's own filtering logic (imported constants and
    functions only -- pm.pick_exemplar is called, not reimplemented) while
    additionally recording trigger-set sizes and single-mention-per-side
    flags that mine_rule1's own return rows do not expose. Every conflict
    condition (cond1/cond2), every subframe (n_sub/d_sub), and every
    exemplar call is identical to mine_rule1's source."""
    out = []
    for hadm_id, group in classified.groupby("HADM_ID"):
        nursing = group[group["note_role"] == "nursing"]
        discharge = group[group["note_role"] == "discharge"]
        if len(nursing) == 0 or len(discharge) == 0:
            continue
        for drug in set(nursing["drug"]) & set(discharge["drug"]):
            n_sub = nursing[nursing["drug"] == drug]
            d_sub = discharge[discharge["drug"] == drug]
            n_actions = set(n_sub["action"])
            d_actions = set(d_sub["action"])

            cond1 = "continuation" in n_actions and "cessation" in d_actions
            cond2 = "cessation" in n_actions and "continuation" in d_actions
            if not (cond1 or cond2):
                continue
            req_n, req_d = ("continuation", "cessation") if cond1 else ("cessation", "continuation")

            old_n = n_sub.iloc[0]
            old_d = d_sub.iloc[0]

            new_n_candidates = n_sub[n_sub["action"] == req_n]
            new_d_candidates = d_sub[d_sub["action"] == req_d]
            note_n = mentions_by_note.get((hadm_id, "nursing"))
            note_d = mentions_by_note.get((hadm_id, "discharge"))
            new_n = pm.pick_exemplar(new_n_candidates, note_n)
            new_d = pm.pick_exemplar(new_d_candidates, note_d)

            out.append({
                "rule": 1,
                "n_trigger_size": int(len(new_n_candidates)),
                "d_trigger_size": int(len(new_d_candidates)),
                "n_mentions_this_side": int(len(n_sub)),
                "d_mentions_this_side": int(len(d_sub)),
                "positional_outside_nursing": bool(old_n["action"] != req_n),
                "positional_outside_discharge": bool(old_d["action"] != req_d),
                "rebuild_outside_nursing": bool(new_n["action"] != req_n),
                "rebuild_outside_discharge": bool(new_d["action"] != req_d),
            })
    return out


def mirror_rule2(classified: pd.DataFrame, mentions_by_note: dict, pm) -> list[dict]:
    """Same mirroring as mirror_rule1, for mine_rule2's class-level branch."""
    out = []
    c = classified.copy()
    c["drug_class"] = c["drug"].map(pm.DRUG_TO_CLASS)
    c = c.dropna(subset=["drug_class"])
    for hadm_id, group in c.groupby("HADM_ID"):
        nursing = group[group["note_role"] == "nursing"]
        discharge = group[group["note_role"] == "discharge"]
        if len(nursing) == 0 or len(discharge) == 0:
            continue
        for cls in set(nursing["drug_class"]) & set(discharge["drug_class"]):
            n_sub = nursing[nursing["drug_class"] == cls]
            d_sub = discharge[discharge["drug_class"] == cls]
            if set(n_sub["drug"]) == set(d_sub["drug"]):
                continue
            n_actions = set(n_sub["action"])
            d_actions = set(d_sub["action"])
            cond1 = "continuation" in n_actions and "cessation" in d_actions
            cond2 = "cessation" in n_actions and "continuation" in d_actions
            if not (cond1 or cond2):
                continue
            req_n, req_d = ("continuation", "cessation") if cond1 else ("cessation", "continuation")

            old_n = n_sub.iloc[0]
            old_d = d_sub.iloc[0]

            new_n_candidates = n_sub[n_sub["action"] == req_n]
            new_d_candidates = d_sub[d_sub["action"] == req_d]
            note_n = mentions_by_note.get((hadm_id, "nursing"))
            note_d = mentions_by_note.get((hadm_id, "discharge"))
            new_n = pm.pick_exemplar(new_n_candidates, note_n)
            new_d = pm.pick_exemplar(new_d_candidates, note_d)

            out.append({
                "rule": 2,
                "n_trigger_size": int(len(new_n_candidates)),
                "d_trigger_size": int(len(new_d_candidates)),
                "n_mentions_this_side": int(len(n_sub)),
                "d_mentions_this_side": int(len(d_sub)),
                "positional_outside_nursing": bool(old_n["action"] != req_n),
                "positional_outside_discharge": bool(old_d["action"] != req_d),
                "rebuild_outside_nursing": bool(new_n["action"] != req_n),
                "rebuild_outside_discharge": bool(new_d["action"] != req_d),
            })
    return out


def block(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    df = pd.DataFrame(rows)
    either_positional = (df["positional_outside_nursing"] | df["positional_outside_discharge"])
    either_rebuild = (df["rebuild_outside_nursing"] | df["rebuild_outside_discharge"])

    def rate_block(mask: pd.Series) -> dict:
        k = int(mask.sum())
        lo, hi = wilson(k, n)
        return {"n": n, "count": k, "rate": round(k / n, 6),
                "wilson_lo": round(lo, 6), "wilson_hi": round(hi, 6)}

    return {
        "n_candidates": n,
        "positional_outside_trigger_set": {
            "nursing_side": rate_block(df["positional_outside_nursing"]),
            "discharge_side": rate_block(df["positional_outside_discharge"]),
            "either_side": rate_block(either_positional),
        },
        "rebuild_selector_outside_trigger_set_construction_check": {
            "nursing_side": rate_block(df["rebuild_outside_nursing"]),
            "discharge_side": rate_block(df["rebuild_outside_discharge"]),
            "either_side": rate_block(either_rebuild),
        },
        "single_mention_per_side_share": {
            "nursing_side": round(float((df["n_mentions_this_side"] == 1).mean()), 6),
            "discharge_side": round(float((df["d_mentions_this_side"] == 1).mean()), 6),
            "both_sides": round(float(((df["n_mentions_this_side"] == 1) & (df["d_mentions_this_side"] == 1)).mean()), 6),
        },
        "trigger_set_size_mean": {
            "nursing_side": round(float(df["n_trigger_size"].mean()), 4),
            "discharge_side": round(float(df["d_trigger_size"].mean()), 4),
        },
    }


def run() -> dict:
    import phase3_mining as pm  # imported for its functions/constants; not modified

    mentions = pd.read_csv(MENTIONS_PATH)
    classified = mentions[mentions["action"] != "unclassified"].copy()
    mentions_by_note = {k: v for k, v in mentions.groupby(["HADM_ID", "note_role"])}

    # Official candidate pool, via the rebuild's own unmodified functions --
    # this is the regeneration/count-match gate.
    r1 = pm.mine_rule1(classified, mentions_by_note)
    r2 = pm.mine_rule2(classified, mentions_by_note)
    r3 = pm.mine_rule3(mentions)
    r4 = pm.mine_rule4(classified, mentions, mentions_by_note)
    regen_counts = {
        "rule1_candidates": len(r1), "rule2_candidates": len(r2),
        "rule3_candidates": len(r3), "rule4_candidates": len(r4),
        "total_candidates": len(r1) + len(r2) + len(r3) + len(r4),
    }
    recorded = json.loads(RECORDED_SUMMARY.read_text())
    recorded_counts = {k: recorded[k] for k in regen_counts}
    counts_match = regen_counts == recorded_counts

    out = {
        "preregistration": "accord/c129_exemplar_prevalence/C129_PREREGISTRATION.md",
        "gate_regenerated_counts": regen_counts,
        "gate_recorded_counts": recorded_counts,
        "gate_counts_match": counts_match,
    }
    if not counts_match:
        out["outcome"] = "C"
        return out

    # Mirror mine_rule1/mine_rule2 to additionally capture trigger-set sizes
    # and the positional pick's side-level status; cross-check candidate
    # counts against the official r1/r2 lengths as an internal consistency
    # gate on the mirroring itself.
    m1 = mirror_rule1(classified, mentions_by_note, pm)
    m2 = mirror_rule2(classified, mentions_by_note, pm)
    mirror_counts_match = (len(m1) == len(r1)) and (len(m2) == len(r2))
    out["gate_mirror_counts_match_official"] = {
        "rule1": [len(m1), len(r1)], "rule2": [len(m2), len(r2)], "ok": mirror_counts_match,
    }
    if not mirror_counts_match:
        out["outcome"] = "C"
        return out

    out["rule1"] = block(m1)
    out["rule2"] = block(m2)
    out["pooled"] = block(m1 + m2)
    out["outcome"] = "A"
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with LOG.open("w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
        res = run()
    res["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"outcome {res['outcome']}")
    print("gate_counts_match", res.get("gate_counts_match"))
    if "gate_mirror_counts_match_official" in res:
        print("gate_mirror_counts_match_official", res["gate_mirror_counts_match_official"]["ok"])
    for block_name in ("rule1", "rule2", "pooled"):
        if block_name in res:
            b = res[block_name]
            print(f"\n{block_name}: n={b['n_candidates']}")
            for side, d in b["positional_outside_trigger_set"].items():
                print(f"  positional outside trigger ({side}): {d['count']}/{d['n']} "
                      f"= {d['rate']:.4f}  [{d['wilson_lo']:.4f}, {d['wilson_hi']:.4f}]")
            for side, d in b["rebuild_selector_outside_trigger_set_construction_check"].items():
                print(f"  rebuild selector outside trigger ({side}, must be 0): {d['count']}/{d['n']}")
            print(f"  single-mention-per-side share: {b['single_mention_per_side_share']}")


if __name__ == "__main__":
    main()
