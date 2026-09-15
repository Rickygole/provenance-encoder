"""C128 analysis: prospective feasibility check on a 100-pair pilot, held-out
against the remaining 199 pairs. See C128_PREREGISTRATION.md; the rule is
read, not re-derived, from that file.

Reads the same per-pair caches C127 wrote/reused (never new Ollama calls) and
the gold CSV (only to recover label_clean per pair_id). The pilot/held-out
split is a pure function of a committed seed and the gold file's row order
(see C128_PREREGISTRATION.md's pilot definition); it is recomputed here every
run and is NEVER persisted as a pair_id-to-split mapping, in results/ or
anywhere else -- only aggregate pilot/held-out counts are written to
results/accord/c128_prospective_feasibility/c128.json.

Run with stdout/stderr redirected while mimic_local_only paths are open,
e.g.:
    python3 accord/c128_prospective_feasibility/analyze_c128.py \\
        > ~/mimic_local_only/logs/c128_analyze_<ts>.log 2>&1
"""
import collections
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accord.differential.model_tags import ollama_model_tag  # noqa: E402
from accord.stats.marginal_feasibility_gate import (  # noqa: E402
    attainable_floor_p, best_table, mcc,
)

GOLD = pathlib.Path.home() / "mimic_local_only/from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv"
CACHE_DIR = pathlib.Path.home() / "mimic_local_only/accord_prompt_sensitivity_cache"
C127_OUT = ROOT / "results" / "accord" / "c127_scale_ladder" / "c127.json"
OUT_PATH = ROOT / "results" / "accord" / "c128_prospective_feasibility" / "c128.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREGISTRATION = ROOT / "accord" / "c128_prospective_feasibility" / "C128_PREREGISTRATION.md"
PREFIX = "c128"

SIZES = ["0.6b", "1.7b", "4b", "8b", "14b"]
PROMPT_NAME = "P_asserting"
PILOT_SEED = 20260915
N_PILOT = 100
N_TOTAL = 299
N_HELDOUT = N_TOTAL - N_PILOT
ALPHA = 0.05


def load_cache(tag):
    path = CACHE_DIR / f"{tag}__{PROMPT_NAME}.jsonl"
    recs = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {r["pair_id"]: r for r in recs}


def pilot_heldout_split(pair_ids_299):
    """Deterministic function of (seed, gold row order); never persisted."""
    rng = np.random.default_rng(PILOT_SEED)
    pilot_idx = sorted(rng.choice(N_TOTAL, size=N_PILOT, replace=False).tolist())
    pilot_set = set(pilot_idx)
    pilot_pairs = [pair_ids_299[i] for i in pilot_idx]
    heldout_pairs = [pair_ids_299[i] for i in range(N_TOTAL) if i not in pilot_set]
    return pilot_pairs, heldout_pairs


def main():
    g = pd.read_csv(GOLD)
    label_by_pair = dict(zip(g["pair_id"], g["label_clean"]))
    pair_ids_299 = g["pair_id"].tolist()
    y_by_pair = {pid: int(label_by_pair[pid] == "contradiction") for pid in pair_ids_299}
    if len(pair_ids_299) != N_TOTAL:
        raise SystemExit(f"gold basis mismatch: n={len(pair_ids_299)} (expected {N_TOTAL})")

    pilot_pairs, heldout_pairs = pilot_heldout_split(pair_ids_299)
    if len(pilot_pairs) != N_PILOT or len(heldout_pairs) != N_HELDOUT \
            or set(pilot_pairs) & set(heldout_pairs) \
            or set(pilot_pairs) | set(heldout_pairs) != set(pair_ids_299):
        raise SystemExit("pilot/held-out split does not partition the 299 gold "
                          "pairs exactly; Outcome C")

    c127_invalid_sizes = []
    if C127_OUT.exists():
        c127 = json.loads(C127_OUT.read_text())
        c127_invalid_sizes = c127.get("invalid_sizes", [])

    if c127_invalid_sizes:
        out = collections.OrderedDict([
            ("preregistration", "accord/c128_prospective_feasibility/C128_PREREGISTRATION.md"),
            ("outcome", "C"),
            ("outcome_reason", f"C127 gate failed for size(s) {c127_invalid_sizes}; "
                                f"the prospective check is not evaluated"),
        ])
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {OUT_PATH}; outcome=C ({out['outcome_reason']})")
        return out

    pilot_n_pos = sum(y_by_pair[pid] for pid in pilot_pairs)
    heldout_n_pos = sum(y_by_pair[pid] for pid in heldout_pairs)

    per_size = collections.OrderedDict()
    ruled_out_sizes = []
    disagreements = []
    for size in SIZES:
        tag = ollama_model_tag(f"qwen3:{size}", False)
        recs = load_cache(tag)

        pilot_n_flag = sum(1 for pid in pilot_pairs if recs.get(pid, {}).get("flag_label") == 1)
        pilot_tp = sum(1 for pid in pilot_pairs
                        if recs.get(pid, {}).get("flag_label") == 1 and y_by_pair[pid] == 1)
        pilot_floor_p = attainable_floor_p(N_PILOT, pilot_n_pos, pilot_n_flag)
        ceiling_tp, ceiling_tn, ceiling_fp, ceiling_fn = best_table(N_PILOT, pilot_n_pos, pilot_n_flag)
        pilot_ceiling_mcc = mcc(ceiling_tp, ceiling_tn, ceiling_fp, ceiling_fn)

        if pilot_n_pos <= 1:
            prediction = "not_evaluable"
        elif pilot_floor_p > ALPHA:
            prediction = "ruled_out"
        else:
            prediction = "not_ruled_out"

        heldout_n_flag = sum(1 for pid in heldout_pairs if recs.get(pid, {}).get("flag_label") == 1)
        heldout_tp = sum(1 for pid in heldout_pairs
                          if recs.get(pid, {}).get("flag_label") == 1 and y_by_pair[pid] == 1)
        heldout_p = float(hypergeom.sf(heldout_tp - 1, N_HELDOUT, heldout_n_pos, heldout_n_flag))
        heldout_outcome = "confirmed_uninformative" if heldout_p >= ALPHA else "not_confirmed"

        if prediction == "ruled_out":
            ruled_out_sizes.append(size)
            agreement = (heldout_outcome == "confirmed_uninformative")
            if not agreement:
                disagreements.append(size)
            agreement_str = "agree" if agreement else "disagree"
        elif prediction == "not_ruled_out":
            agreement_str = "not_applicable"
        else:
            agreement_str = "not_evaluable"

        per_size[size] = collections.OrderedDict([
            ("model", f"qwen3:{size}"), ("tag", tag),
            ("pilot_n", N_PILOT), ("pilot_n_pos", int(pilot_n_pos)),
            ("pilot_n_flag", int(pilot_n_flag)), ("pilot_tp", int(pilot_tp)),
            ("pilot_attainable_floor_p", pilot_floor_p),
            ("pilot_ceiling_mcc", pilot_ceiling_mcc),
            ("prediction", prediction),
            ("heldout_n", N_HELDOUT), ("heldout_n_pos", int(heldout_n_pos)),
            ("heldout_n_flag", int(heldout_n_flag)), ("heldout_tp", int(heldout_tp)),
            ("heldout_hypergeom_p_one_sided_observed", heldout_p),
            ("heldout_outcome", heldout_outcome),
            ("agreement", agreement_str),
        ])

    if not ruled_out_sizes:
        check_outcome = "not_applicable_no_ruled_out_predictions"
        check_reason = "no size's pilot prediction was ruled_out; the check has nothing to demonstrate"
    elif not disagreements:
        check_outcome = "demonstrated"
        check_reason = (f"every ruled_out pilot prediction ({ruled_out_sizes}) was "
                         f"confirmed_uninformative on its held-out set")
    else:
        check_outcome = "not_demonstrated"
        check_reason = (f"ruled_out pilot prediction(s) {disagreements} disagreed with "
                         f"their held-out outcome")

    out = collections.OrderedDict([
        ("preregistration", "accord/c128_prospective_feasibility/C128_PREREGISTRATION.md"),
        ("pilot_seed", PILOT_SEED), ("n_pilot", N_PILOT), ("n_heldout", N_HELDOUT),
        ("pilot_n_pos_pharmacist", int(pilot_n_pos)),
        ("heldout_n_pos_pharmacist", int(heldout_n_pos)),
        ("alpha", ALPHA),
        ("per_size", per_size),
        ("ruled_out_sizes", ruled_out_sizes),
        ("disagreements", disagreements),
        ("check_outcome", check_outcome),
        ("check_reason", check_reason),
        ("outcome", check_outcome),
    ])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")

    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]
    rel = OUT_PATH.relative_to(ROOT)
    n_written = 0

    def put(path_prefix, node):
        nonlocal n_written
        if isinstance(node, dict):
            for k, v in node.items():
                put(f"{path_prefix}.{k}" if path_prefix else k, v)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                put(f"{path_prefix}.{i}", v)
        elif node is None:
            return
        else:
            facts[f"{PREFIX}.{path_prefix}"] = collections.OrderedDict(
                [("artifact", f"{rel}::{path_prefix}"), ("value", node)])
            n_written += 1

    put("", out)
    FACTS.write_text(json.dumps(facts, indent=1) + "\n")

    prereg_text = PREGISTRATION.read_text()
    outcome_line = (
        f"\n**Outcome: {check_outcome}** (run analyzed "
        f"{__import__('time').strftime('%Y-%m-%d')}; artifact "
        f"`results/accord/c128_prospective_feasibility/c128.json`). {check_reason}. "
        f"Per-size predictions: "
        + ", ".join(f"{s}={per_size[s]['prediction']}" for s in SIZES)
        + f"; held-out outcomes: "
        + ", ".join(f"{s}={per_size[s]['heldout_outcome']}" for s in SIZES)
        + ".\n"
    )
    if "(appended after the run; the rule above is not edited)" in prereg_text:
        prereg_text = prereg_text.replace(
            "(appended after the run; the rule above is not edited)",
            "(appended after the run; the rule above is not edited)\n" + outcome_line)
        PREGISTRATION.write_text(prereg_text)

    print(f"wrote {OUT_PATH}; registered {n_written} c128.* keys; "
          f"check_outcome={check_outcome} ({check_reason})")
    return out


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        tb = traceback.extract_tb(exc.__traceback__)
        last = tb[-1] if tb else None
        debug = {
            "exception_type": type(exc).__name__,
            "last_frame_function": last.name if last else None,
            "last_frame_lineno": last.lineno if last else None,
            "last_frame_file": last.filename if last else None,
        }
        debug_path = ROOT / "results" / "accord" / "c128_prospective_feasibility" / "analyze_debug.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(debug, indent=2) + "\n")
        raise SystemExit(f"failed: {type(exc).__name__} (see {debug_path})")
