"""C127 analysis: scale ladder attenuation rule, marginal ceiling, graded AUROC,
Mann-Whitney and label-permutation p, on the same per-pair caches run_c127.py
wrote (or reused). See C127_PREREGISTRATION.md; the rule below is read, not
re-derived, from that file.

Opens the local numeric-only per-pair cache under
~/mimic_local_only/accord_prompt_sensitivity_cache/ (pair_id, flag label, flag
log-odds, degenerate, parse-failure -- never sentence text) and the gold CSV
(only to recover label_clean per pair_id). Writes AGGREGATE ONLY to
results/accord/c127_scale_ladder/c127.json, and registers every scalar leaf
under the c127.* prefix in results/accord/VERIFIED_FACTS.json.

Per the data-use compliance rule, run this with its stdout/stderr redirected
to a log file under ~/mimic_local_only/logs/, e.g.:

    python3 accord/c127_scale_ladder/analyze_c127.py \\
        > ~/mimic_local_only/logs/c127_analyze_<ts>.log 2>&1

(This script's own final report -- aggregate numbers only -- is also printed
to real stdout by the caller after that redirected run; see run_all.sh /
report_c127.py, matching analyze_c103.py's own convention of leaving the
redirection to the invoking shell.)
"""
import collections
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accord.differential.model_tags import ollama_model_tag  # noqa: E402
from accord.stats.marginal_feasibility_gate import (  # noqa: E402
    attainable_floor_p, best_table, mcc,
)

GOLD = pathlib.Path.home() / "mimic_local_only/from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv"
CACHE_DIR = pathlib.Path.home() / "mimic_local_only/accord_prompt_sensitivity_cache"
OUT_PATH = ROOT / "results" / "accord" / "c127_scale_ladder" / "c127.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREGISTRATION = ROOT / "accord" / "c127_scale_ladder" / "C127_PREREGISTRATION.md"
PREFIX = "c127"

SIZES = ["0.6b", "1.7b", "4b", "8b", "14b"]
PARAMS_BILLIONS = {"0.6b": 0.6, "1.7b": 1.7, "4b": 4.0, "8b": 8.0, "14b": 14.0}
PROMPT_NAME = "P_asserting"
GOLD_BASE_RATE_299 = 12 / 299
OVERCALL_PATH = ROOT / "results" / "accord" / "stats" / "overcall_decomposition.json"
MDE_PATH = ROOT / "results" / "accord" / "mde_recomputed.json"
B_PERM = 20000
PERM_SEED = 42
RHO_THRESHOLD = -0.9
RATIO_14B_CEILING = 3.5
AUROC_RULE_THRESHOLD = 0.7355
MW_RULE_ALPHA = 0.05
PARSE_FAIL_GATE = 0.05


def load_cache(tag):
    path = CACHE_DIR / f"{tag}__{PROMPT_NAME}.jsonl"
    recs = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {r["pair_id"]: r for r in recs}


def perm_p_one_sided(y, s, B, seed):
    """One-sided label-permutation null holding scores fixed. Mirrors
    accord/tfidf_permutation/tfidfperm_run.py::perm_null's convention."""
    rng = np.random.default_rng(seed)
    n = len(y)
    obs_auc = float(roc_auc_score(y, s))
    idx = np.arange(n)
    n_ge = 0
    for _ in range(B):
        yp = y[rng.permutation(idx)]
        if roc_auc_score(yp, s) >= obs_auc:
            n_ge += 1
    return obs_auc, float(n_ge / B)


def main():
    g = pd.read_csv(GOLD)
    label_by_pair = dict(zip(g["pair_id"], g["label_clean"]))
    pair_ids_299 = g["pair_id"].tolist()
    pair_ids_253 = [pid for pid in pair_ids_299 if label_by_pair[pid] != "ambiguous"]
    y299_by_pair = {pid: int(label_by_pair[pid] == "contradiction") for pid in pair_ids_299}
    y253_by_pair = {pid: y299_by_pair[pid] for pid in pair_ids_253}
    n_pos_299 = sum(y299_by_pair.values())
    if n_pos_299 != 12 or len(pair_ids_299) != 299 or len(pair_ids_253) != 253:
        raise SystemExit(f"gold basis mismatch: n299={len(pair_ids_299)} "
                          f"n253={len(pair_ids_253)} n_pos_299={n_pos_299} "
                          f"(expected 299/253/12)")

    ceiling_ratio = json.loads(OVERCALL_PATH.read_text())["public"]["max_over_call_ratio"]
    mde_auc = json.loads(MDE_PATH.read_text())["strata"]["gold_ambiguous_excluded"]["minimum_detectable_auc"]

    per_size = collections.OrderedDict()
    invalid_sizes = []
    for size in SIZES:
        tag = ollama_model_tag(f"qwen3:{size}", False)
        recs = load_cache(tag)
        n = len(recs)
        n_parse_fail = sum(1 for r in recs.values() if r["parse_failure"])
        n_degenerate = sum(1 for r in recs.values() if r["degenerate"])
        n_thinking = sum(1 for r in recs.values() if r.get("thinking_nonempty"))
        n_truncated = sum(1 for r in recs.values() if r.get("truncated"))
        n_flag_299 = sum(1 for r in recs.values() if r.get("flag_label") == 1)
        n_scored = sum(1 for r in recs.values() if r.get("flag_label") is not None)
        parse_fail_rate = (n_parse_fail / n) if n else None

        gate_ok = (n == 299 and parse_fail_rate is not None
                   and parse_fail_rate <= PARSE_FAIL_GATE and n_truncated == 0)
        if not gate_ok:
            invalid_sizes.append(size)

        # Observed confusion on G_299 (ambiguous counts negative): "flagged"
        # matches n_flagged's own definition (flag_label == 1); anything else
        # (explicit 0 or unparsed) is "not flagged", exactly as positive_rate
        # is already computed in run_c103.py/run_c127.py.
        tp = tn = fp = fn = 0
        for pid in pair_ids_299:
            flagged = recs.get(pid, {}).get("flag_label") == 1
            positive = y299_by_pair[pid] == 1
            if flagged and positive:
                tp += 1
            elif flagged and not positive:
                fp += 1
            elif not flagged and positive:
                fn += 1
            else:
                tn += 1
        observed_mcc = mcc(tp, tn, fp, fn)

        ceiling_tp, ceiling_tn, ceiling_fp, ceiling_fn = best_table(299, n_pos_299, n_flag_299)
        ceiling_mcc = mcc(ceiling_tp, ceiling_tn, ceiling_fp, ceiling_fn)
        floor_p = attainable_floor_p(299, n_pos_299, n_flag_299)

        positive_rate_299 = (n_flag_299 / n) if n else None
        over_call_ratio = (positive_rate_299 / GOLD_BASE_RATE_299) if positive_rate_299 is not None else None

        # G_253 graded scores: only pairs whose cell scored a real log-odds
        # (excludes call failures, which set flag_logodds to None).
        scored_253 = [(recs[pid]["flag_logodds"], y253_by_pair[pid])
                      for pid in pair_ids_253
                      if recs.get(pid) is not None and recs[pid].get("flag_logodds") is not None]
        auroc = mw_p = perm_p = None
        if len({y for _, y in scored_253}) == 2:
            scores_253 = np.array([s for s, _ in scored_253], dtype=float)
            labels_253 = np.array([y for _, y in scored_253], dtype=int)
            auroc = float(roc_auc_score(labels_253, scores_253))
            pos = scores_253[labels_253 == 1]
            neg = scores_253[labels_253 == 0]
            _, mw_p = mannwhitneyu(pos, neg, alternative="greater")
            mw_p = float(mw_p)
            _, perm_p = perm_p_one_sided(labels_253, scores_253, B_PERM, PERM_SEED)

        meets_auroc_rule = bool(auroc is not None and auroc >= AUROC_RULE_THRESHOLD
                                  and mw_p is not None and mw_p < MW_RULE_ALPHA)

        per_size[size] = collections.OrderedDict([
            ("model", f"qwen3:{size}"), ("params_billions", PARAMS_BILLIONS[size]),
            ("tag", tag),
            ("n", n), ("n_parse_failures", n_parse_fail),
            ("parse_failure_rate", round(parse_fail_rate, 4) if parse_fail_rate is not None else None),
            ("n_degenerate_logodds", n_degenerate),
            ("n_thinking_nonempty", n_thinking), ("n_truncated", n_truncated),
            ("n_scored", n_scored), ("n_flagged_299", n_flag_299),
            ("positive_rate_299", positive_rate_299),
            ("over_calling_ratio", over_call_ratio),
            ("gate_passed", bool(gate_ok)),
            ("observed_confusion", {"tp": tp, "tn": tn, "fp": fp, "fn": fn}),
            ("observed_mcc", observed_mcc),
            ("ceiling_confusion", {"tp": ceiling_tp, "tn": ceiling_tn,
                                    "fp": ceiling_fp, "fn": ceiling_fn}),
            ("ceiling_mcc", ceiling_mcc),
            ("attainable_floor_p", floor_p),
            ("auroc_253", auroc),
            ("mannwhitney_p_one_sided_253", mw_p),
            ("permutation_p_one_sided_253", perm_p),
            ("permutation_B", B_PERM if perm_p is not None else None),
            ("meets_registered_auroc_rule", meets_auroc_rule),
        ])

    spearman_result = None
    verdict = None
    if invalid_sizes:
        outcome = "C"
        outcome_reason = (f"gate failed for size(s) {invalid_sizes}: parse-failure "
                           f"rate above {PARSE_FAIL_GATE} or a truncated call; the "
                           f"Spearman rule is not evaluated")
    else:
        log_params = [math.log(PARAMS_BILLIONS[s]) for s in SIZES]
        ratios = [per_size[s]["over_calling_ratio"] for s in SIZES]
        rho, sp_p = spearmanr(log_params, ratios)
        rho = float(rho)
        ratio_14b = per_size["14b"]["over_calling_ratio"]
        attenuates = bool(rho <= RHO_THRESHOLD and ratio_14b < RATIO_14B_CEILING)
        verdict = "attenuates" if attenuates else "does_not_attenuate"
        outcome = "A" if attenuates else "B"
        spearman_result = {
            "rho": rho, "p_value": float(sp_p),
            "rho_threshold": RHO_THRESHOLD,
            "ratio_14b": ratio_14b, "ratio_14b_ceiling": RATIO_14B_CEILING,
            "attenuates": attenuates,
        }
        outcome_reason = (f"rho={rho:.4f} (threshold <= {RHO_THRESHOLD}), "
                           f"14B ratio={ratio_14b:.4f} (ceiling < {RATIO_14B_CEILING}) "
                           f"-> {verdict}")

    out = collections.OrderedDict([
        ("preregistration", "accord/c127_scale_ladder/C127_PREREGISTRATION.md"),
        ("gold_base_rate_299", GOLD_BASE_RATE_299),
        ("ceiling_ratio_source", "results/accord/stats/overcall_decomposition.json::public.max_over_call_ratio"),
        ("ceiling_ratio", ceiling_ratio),
        ("mde_auroc_253", mde_auc),
        ("per_size", per_size),
        ("invalid_sizes", invalid_sizes),
        ("spearman", spearman_result),
        ("verdict", verdict),
        ("outcome", outcome),
        ("outcome_reason", outcome_reason),
    ])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")

    # Register: delete this prefix wholesale, then re-add every scalar leaf,
    # pointing at OUT_PATH::dotted.path, matching the existing convention
    # (accord/stats/register_overcall_decomposition.py and c126_run.py).
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

    # Append the outcome to the pre-registration file (rule text untouched).
    prereg_text = PREGISTRATION.read_text()
    outcome_line = (
        f"\n**Outcome {outcome}** (run analyzed "
        f"{__import__('time').strftime('%Y-%m-%d')}; artifact "
        f"`results/accord/c127_scale_ladder/c127.json`). {outcome_reason}. "
        f"Per-size over-calling ratios: "
        + ", ".join(f"{s}={per_size[s]['over_calling_ratio']:.3f}" for s in SIZES)
        + f". Per-size graded AUROC (G_253): "
        + ", ".join(
            f"{s}={per_size[s]['auroc_253']:.4f}" if per_size[s]['auroc_253'] is not None else f"{s}=n/a"
            for s in SIZES)
        + ".\n"
    )
    if "(appended after the run; the rule above is not edited)" in prereg_text:
        prereg_text = prereg_text.replace(
            "(appended after the run; the rule above is not edited)",
            "(appended after the run; the rule above is not edited)\n" + outcome_line)
        PREGISTRATION.write_text(prereg_text)

    print(f"wrote {OUT_PATH}; registered {n_written} c127.* keys; "
          f"outcome={outcome} ({outcome_reason})")


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
        debug_path = ROOT / "results" / "accord" / "c127_scale_ladder" / "analyze_debug.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(debug, indent=2) + "\n")
        raise SystemExit(f"failed: {type(exc).__name__} (see {debug_path})")
