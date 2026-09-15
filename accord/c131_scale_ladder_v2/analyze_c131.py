"""C131 analysis: the C127 attenuation rule AND the C128 prospective-feasibility
check, both applied to C131's own (repaired) per-pair caches. See
C131_PREREGISTRATION.md; every rule here is read, not re-derived, from that
file and from C127_PREREGISTRATION.md / C128_PREREGISTRATION.md, which this
file's two halves mirror line for line except for where the caches come from.

Opens the local numeric-only per-pair caches run_c131.py wrote (or, for the
8B/14B reuse candidates, C127's own caches, read directly and never copied)
and the gold CSV (only to recover label_clean per pair_id). Writes AGGREGATE
ONLY to results/accord/c131_scale_ladder_v2/c131.json. Registration to
results/accord/VERIFIED_FACTS.json under the c131.* prefix is done by the
separate register_c131.py, not by this file.

Per the data-use compliance rule, run this with its stdout/stderr redirected
to a log file under ~/mimic_local_only/logs/, e.g.:

    python3 accord/c131_scale_ladder_v2/analyze_c131.py \\
        > ~/mimic_local_only/logs/c131_analyze_<ts>.log 2>&1
"""
import collections
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accord.differential.model_tags import ollama_model_tag  # noqa: E402
from accord.stats.marginal_feasibility_gate import (  # noqa: E402
    attainable_floor_p, best_table, mcc,
)
from accord.c128_prospective_feasibility.analyze_c128 import (  # noqa: E402
    pilot_heldout_split, PILOT_SEED, N_PILOT, N_TOTAL, N_HELDOUT, ALPHA,
)

GOLD = pathlib.Path.home() / "mimic_local_only/from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv"
CELLS_DIR = ROOT / "results" / "accord" / "c131_scale_ladder_v2" / "cells"
VALIDATION_PATH = ROOT / "results" / "accord" / "c131_scale_ladder_v2" / "synthetic_probe_validation.json"
OUT_PATH = ROOT / "results" / "accord" / "c131_scale_ladder_v2" / "c131.json"
PREGISTRATION = ROOT / "accord" / "c131_scale_ladder_v2" / "C131_PREREGISTRATION.md"

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


def load_cache(path):
    recs = [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line]
    return {r["pair_id"]: r for r in recs}


def graded_score(rec, use_primary):
    """The graded score to use for this pair, per C131_PREREGISTRATION.md's
    primary/fallback rule. A reused (8B/14B) record has only one score field
    (C127's original, position-0-only by construction, since every one of
    its 299 rows already answered at position 0) -- it is used as-is under
    either branch. A freshly-run C131 record carries both flag_logodds (the
    primary, first-answer-token-position score) and
    flag_logodds_legacy_pos0 (gated on first_token_is_answer) for the
    fallback."""
    if rec.get("call_failed"):
        return None
    if "flag_logodds_legacy_pos0" not in rec:
        return rec.get("flag_logodds")
    if use_primary:
        return rec.get("flag_logodds")
    return rec.get("flag_logodds_legacy_pos0") if rec.get("first_token_is_answer") else None


def perm_p_one_sided(y, s, B, seed):
    """One-sided label-permutation null holding scores fixed. Mirrors
    accord/tfidf_permutation/tfidfperm_run.py::perm_null's convention (also
    accord/c127_scale_ladder/analyze_c127.py's own copy of this function)."""
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
    if not VALIDATION_PATH.exists():
        raise SystemExit(
            f"{VALIDATION_PATH} does not exist; C131 is invalid per its own "
            f"pre-registration (no clinical run should have happened without it)")
    validation = json.loads(VALIDATION_PATH.read_text())
    if not validation.get("all_checks_passed"):
        raise SystemExit(
            f"synthetic probe validation did not pass ({validation.get('summary')!r}); "
            f"C131 is invalid per its own pre-registration")
    use_primary = bool(validation.get("per_token_logprobs_available"))

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

    pilot_pairs, heldout_pairs = pilot_heldout_split(pair_ids_299)
    if len(pilot_pairs) != N_PILOT or len(heldout_pairs) != N_HELDOUT \
            or set(pilot_pairs) & set(heldout_pairs) \
            or set(pilot_pairs) | set(heldout_pairs) != set(pair_ids_299):
        raise SystemExit("pilot/held-out split does not partition the 299 gold "
                          "pairs exactly; Outcome C")
    pilot_n_pos = sum(y299_by_pair[pid] for pid in pilot_pairs)
    heldout_n_pos = sum(y299_by_pair[pid] for pid in heldout_pairs)

    per_size = collections.OrderedDict()
    invalid_sizes = []
    for size in SIZES:
        tag = ollama_model_tag(f"qwen3:{size}", False)
        cell_path = CELLS_DIR / f"{tag}__{PROMPT_NAME}.json"
        if not cell_path.exists():
            raise SystemExit(f"missing C131 cell for qwen3:{size}: {cell_path} "
                              f"(run_c131.py has not produced this size yet)")
        cell = json.loads(cell_path.read_text())
        recs = load_cache(cell["source_cache_path"])
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

        scored_253 = []
        for pid in pair_ids_253:
            rec = recs.get(pid)
            if rec is None:
                continue
            s = graded_score(rec, use_primary)
            if s is not None:
                scored_253.append((s, y253_by_pair[pid]))
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

        # C128-style pilot/held-out prospective check, on this size's own C131 cache.
        pilot_n_flag = sum(1 for pid in pilot_pairs if recs.get(pid, {}).get("flag_label") == 1)
        pilot_tp = sum(1 for pid in pilot_pairs
                        if recs.get(pid, {}).get("flag_label") == 1 and y299_by_pair[pid] == 1)
        pilot_floor_p = attainable_floor_p(N_PILOT, pilot_n_pos, pilot_n_flag)
        pceil_tp, pceil_tn, pceil_fp, pceil_fn = best_table(N_PILOT, pilot_n_pos, pilot_n_flag)
        pilot_ceiling_mcc = mcc(pceil_tp, pceil_tn, pceil_fp, pceil_fn)
        if pilot_n_pos <= 1:
            prediction = "not_evaluable"
        elif pilot_floor_p > ALPHA:
            prediction = "ruled_out"
        else:
            prediction = "not_ruled_out"

        heldout_n_flag = sum(1 for pid in heldout_pairs if recs.get(pid, {}).get("flag_label") == 1)
        heldout_tp = sum(1 for pid in heldout_pairs
                          if recs.get(pid, {}).get("flag_label") == 1 and y299_by_pair[pid] == 1)
        heldout_p = float(hypergeom.sf(heldout_tp - 1, N_HELDOUT, heldout_n_pos, heldout_n_flag))
        heldout_outcome = "confirmed_uninformative" if heldout_p >= ALPHA else "not_confirmed"
        if prediction == "ruled_out":
            agreement = "agree" if heldout_outcome == "confirmed_uninformative" else "disagree"
        elif prediction == "not_ruled_out":
            agreement = "not_applicable"
        else:
            agreement = "not_evaluable"

        per_size[size] = collections.OrderedDict([
            ("model", f"qwen3:{size}"), ("params_billions", PARAMS_BILLIONS[size]),
            ("tag", tag), ("reused_from_c127", bool(cell.get("reused_from_c127"))),
            ("source_cache_path", cell["source_cache_path"]),
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
            ("graded_score_method", "primary" if use_primary else "fallback_legacy_pos0"),
            ("n_graded_scored_253", len(scored_253)),
            ("auroc_253", auroc),
            ("mannwhitney_p_one_sided_253", mw_p),
            ("permutation_p_one_sided_253", perm_p),
            ("permutation_B", B_PERM if perm_p is not None else None),
            ("meets_registered_auroc_rule", meets_auroc_rule),
            ("pilot_n", N_PILOT), ("pilot_n_flag", int(pilot_n_flag)), ("pilot_tp", int(pilot_tp)),
            ("pilot_attainable_floor_p", pilot_floor_p),
            ("pilot_ceiling_mcc", pilot_ceiling_mcc),
            ("prospective_prediction", prediction),
            ("heldout_n", N_HELDOUT), ("heldout_n_flag", int(heldout_n_flag)),
            ("heldout_tp", int(heldout_tp)),
            ("heldout_hypergeom_p_one_sided_observed", heldout_p),
            ("heldout_outcome", heldout_outcome),
            ("prospective_agreement", agreement),
        ])

    # -- C127-style attenuation rule --
    spearman_result = None
    attenuation_verdict = None
    if invalid_sizes:
        attenuation_outcome = "C"
        attenuation_reason = (f"gate failed for size(s) {invalid_sizes}: parse-failure "
                               f"rate above {PARSE_FAIL_GATE} or a truncated call; the "
                               f"Spearman rule is not evaluated")
    else:
        log_params = [math.log(PARAMS_BILLIONS[s]) for s in SIZES]
        ratios = [per_size[s]["over_calling_ratio"] for s in SIZES]
        rho, sp_p = spearmanr(log_params, ratios)
        rho = float(rho)
        ratio_14b = per_size["14b"]["over_calling_ratio"]
        attenuates = bool(rho <= RHO_THRESHOLD and ratio_14b < RATIO_14B_CEILING)
        attenuation_verdict = "attenuates" if attenuates else "does_not_attenuate"
        attenuation_outcome = "A" if attenuates else "B"
        spearman_result = {
            "rho": rho, "p_value": float(sp_p), "rho_threshold": RHO_THRESHOLD,
            "ratio_14b": ratio_14b, "ratio_14b_ceiling": RATIO_14B_CEILING,
            "attenuates": attenuates,
        }
        attenuation_reason = (f"rho={rho:.4f} (threshold <= {RHO_THRESHOLD}), "
                               f"14B ratio={ratio_14b:.4f} (ceiling < {RATIO_14B_CEILING}) "
                               f"-> {attenuation_verdict}")

    # -- C128-style prospective feasibility check, on this run's own cells --
    if invalid_sizes:
        prospective_outcome = "C"
        prospective_reason = (f"C131's own gate failed for size(s) {invalid_sizes}; "
                               f"the prospective check is not evaluated")
        ruled_out_sizes, disagreements = [], []
    else:
        ruled_out_sizes = [s for s in SIZES if per_size[s]["prospective_prediction"] == "ruled_out"]
        disagreements = [s for s in ruled_out_sizes
                          if per_size[s]["prospective_agreement"] == "disagree"]
        if not ruled_out_sizes:
            prospective_outcome = "not_applicable_no_ruled_out_predictions"
            prospective_reason = "no size's pilot prediction was ruled_out; the check has nothing to demonstrate"
        elif not disagreements:
            prospective_outcome = "demonstrated"
            prospective_reason = (f"every ruled_out pilot prediction ({ruled_out_sizes}) was "
                                   f"confirmed_uninformative on its held-out set")
        else:
            prospective_outcome = "not_demonstrated"
            prospective_reason = (f"ruled_out pilot prediction(s) {disagreements} disagreed "
                                   f"with their held-out outcome")

    out = collections.OrderedDict([
        ("preregistration", "accord/c131_scale_ladder_v2/C131_PREREGISTRATION.md"),
        ("validation_artifact", "results/accord/c131_scale_ladder_v2/synthetic_probe_validation.json"),
        ("per_token_logprobs_available", use_primary),
        ("gold_base_rate_299", GOLD_BASE_RATE_299),
        ("ceiling_ratio_source", "results/accord/stats/overcall_decomposition.json::public.max_over_call_ratio"),
        ("ceiling_ratio", ceiling_ratio),
        ("mde_auroc_253", mde_auc),
        ("per_size", per_size),
        ("invalid_sizes", invalid_sizes),
        ("spearman", spearman_result),
        ("attenuation_verdict", attenuation_verdict),
        ("attenuation_outcome", attenuation_outcome),
        ("attenuation_outcome_reason", attenuation_reason),
        ("pilot_seed", PILOT_SEED), ("n_pilot", N_PILOT), ("n_heldout", N_HELDOUT),
        ("pilot_n_pos_pharmacist", int(pilot_n_pos)),
        ("heldout_n_pos_pharmacist", int(heldout_n_pos)),
        ("ruled_out_sizes", ruled_out_sizes),
        ("disagreements", disagreements),
        ("prospective_check_outcome", prospective_outcome),
        ("prospective_check_reason", prospective_reason),
        ("outcome", attenuation_outcome),
    ])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")

    prereg_text = PREGISTRATION.read_text()
    marker = "(appended after the run; the rule above is not edited)"
    outcome_line = (
        f"\n**Outcome {attenuation_outcome}** / prospective check: {prospective_outcome} "
        f"(run analyzed {__import__('time').strftime('%Y-%m-%d')}; artifact "
        f"`results/accord/c131_scale_ladder_v2/c131.json`). {attenuation_reason}. "
        f"{prospective_reason}. Per-size over-calling ratios: "
        + ", ".join(f"{s}={per_size[s]['over_calling_ratio']:.3f}" for s in SIZES)
        + ". Per-size graded AUROC (G_253): "
        + ", ".join(
            f"{s}={per_size[s]['auroc_253']:.4f}" if per_size[s]['auroc_253'] is not None else f"{s}=n/a"
            for s in SIZES)
        + ".\n"
    )
    if marker in prereg_text and prereg_text.count(marker) >= 1:
        # Append after the LAST occurrence of the marker (the "Outcome" section's own,
        # not the validation-addendum section, which uses different marker text).
        idx = prereg_text.rfind(marker)
        prereg_text = (prereg_text[:idx + len(marker)] + "\n" + outcome_line
                        + prereg_text[idx + len(marker):])
        PREGISTRATION.write_text(prereg_text)

    print(f"wrote {OUT_PATH}; attenuation_outcome={attenuation_outcome} ({attenuation_reason}); "
          f"prospective_check_outcome={prospective_outcome} ({prospective_reason})")


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
        debug_path = ROOT / "results" / "accord" / "c131_scale_ladder_v2" / "analyze_debug.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(debug, indent=2) + "\n")
        raise SystemExit(f"failed: {type(exc).__name__} (see {debug_path})")
