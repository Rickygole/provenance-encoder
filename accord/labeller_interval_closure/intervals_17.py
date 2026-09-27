"""Closes two gaps in the labeller graded-score evidence, from stored
per-pair scores only. No new inference, no model call, no hosted API.

GAP 1 (missing intervals). The register carries
secondary.bootstrap95_auroc and secondary.lopo_auroc_min/_max for the four
llama3:8b C109 cells only; the granite3.3:8b cells carry None and the nine
C103 promptsens cells carry nothing. This script computes, for ALL 17
graded-score configurations on G_253 and on the Rules 1-2 and Rules 3-4
strata:
  - percentile bootstrap 95 percent interval for AUROC, B = 10000, fixed
    recorded seed, single-class resamples discarded and counted
  - leave-one-positive-out minimum and maximum AUROC across the positives

GAP 2 (confound check for the untested cells). The nine C103 promptsens
cells were never put through C109's registered confound test (can the score
separate Rules 1-2 from Rules 3-4 among the CONFIRMED NEGATIVES?). This
script runs that identical test on all 17 cells, reusing C109's registered
thresholds (AUROC >= 0.65 and permutation p < 0.05 => confounded).

The 17 configurations:
  C103 promptsens   3 models x {P_asserting, P_neutral, P_reversed}   = 9
  C109 promptconfirm 2 models x {P_rev_agree, P_rev_follow,
                                 P_rev_compatible, P_reversed_rerun}  = 8

Methods are copied from the committed code, never re-derived:
  auroc_and_perm_p, lopo_range  -- accord/prompt_sensitivity/analyze_c109.py
  bootstrap_auroc_ci            -- accord/control/report.py (percentile,
                                   B = 10000, single-class resamples skipped)

Opens the local numeric-only per-pair caches under
~/mimic_local_only/accord_prompt_sensitivity_cache/ (pair_id, flag label,
flag log-odds, flags -- never sentence text), the gold CSV (label_clean per
pair_id only) and the derived results CSV (rule per pair_id only). Writes
AGGREGATE ONLY to
results/accord/labeller_interval_closure/intervals_17.json: counts, AUROC,
p-values, interval bounds, booleans. No pair_id, HADM_ID, SUBJECT_ID or note
text ever leaves this process.

Per the project compliance rule, any process that opens ~/mimic_local_only
sends its full stdout/stderr to a log file under ~/mimic_local_only/logs/:

    python3 accord/labeller_interval_closure/intervals_17.py \
        > ~/mimic_local_only/logs/intervals_17_<ts>.log 2>&1
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accord.prompt_sensitivity.analyze_c109 import (  # noqa: E402
    CONFOUND_AUROC_THRESHOLD,
    CONFOUND_P_THRESHOLD,
    PERM_DRAWS,
    PERM_SEED,
    auroc_and_perm_p,
    lopo_range,
)

MIMIC_LOCAL_ONLY_DIR = pathlib.Path(
    os.environ.get("MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
GOLD = MIMIC_LOCAL_ONLY_DIR / "from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv"
DERIVED = MIMIC_LOCAL_ONLY_DIR / "derived/gold_standard_accord_results.csv"
CACHE_DIR = MIMIC_LOCAL_ONLY_DIR / "accord_prompt_sensitivity_cache"
OUT_PATH = ROOT / "results/accord/labeller_interval_closure/intervals_17.json"
FACTS_PATH = ROOT / "results/accord/VERIFIED_FACTS.json"

MODEL_TAGS = {
    "llama3:8b": "_llama3_8b_nothink",
    "granite3.3:8b": "_granite33_8b_nothink",
    "qwen3:8b": "_qwen3_8b_nothink",
}
# The 17 graded-score configurations, (family, model, prompt).
C103_PROMPTS = ["P_asserting", "P_neutral", "P_reversed"]
C109_PROMPTS = ["P_rev_agree", "P_rev_follow", "P_rev_compatible", "P_reversed_rerun"]
CONFIGS = (
    [("promptsens", m, p) for m in ["llama3:8b", "granite3.3:8b", "qwen3:8b"] for p in C103_PROMPTS]
    + [("promptconfirm", m, p) for m in ["llama3:8b", "granite3.3:8b"] for p in C109_PROMPTS]
)

N_BOOT = 10_000
BOOT_SEED = 20260806  # accord/control/report.py::BOOT_SEED, the project's documented bootstrap seed
REGISTER_PREFIX = "labellerintervals"


PERM_DRAWS_HIGH = 200_000
PERM_SEED_HIGH = 1090


def perm_p_high_precision(y, s, B=PERM_DRAWS_HIGH, seed=PERM_SEED_HIGH, chunk=20_000):
    """The SAME one-sided label-permutation p as
    analyze_c109.py::auroc_and_perm_p, at ten times the draws, chunked so the
    rank matrix never exceeds the B = 20000 memory footprint.

    This exists for one reason. Bonferroni over all seventeen
    configurations puts alpha at 0.05/17 = 0.002941, and at B = 20000 the
    smallest resolvable p is 1/20001 = 0.00005 with a Monte Carlo standard
    error at p = 0.003 of about 0.00039. Two cells sit within one standard
    error of that boundary, so at B = 20000 the COUNT of Bonferroni
    survivors is not a stable quantity. The registered B = 20000 p stays
    primary; this is reported alongside it and is what the survivor count is
    read off.
    """
    y = np.asarray(y)
    s = np.asarray(s, float)
    n = len(y)
    k = int(y.sum())
    if k == 0 or k == n:
        return None, None
    ranks = rankdata(s, method="average")
    a = float((ranks[y == 1].sum() - k * (k + 1) / 2) / (k * (n - k)))
    rng = np.random.default_rng(seed)
    cnt = 0
    done = 0
    while done < B:
        m = min(chunk, B - done)
        idx = np.argsort(rng.random((m, n)), axis=1)[:, :k]
        auc_perm = (ranks[idx].sum(axis=1) - k * (k + 1) / 2) / (k * (n - k))
        cnt += int(np.sum(auc_perm >= a - 1e-12))
        done += m
    return a, float((cnt + 1) / (B + 1))


def bootstrap_auroc_ci(y, scores, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap 95 percent CI on AUROC, resampling the rows with
    replacement. Construction identical to accord/control/report.py's
    bootstrap_auroc_ci; the only addition is that the number of DISCARDED
    single-class resamples is returned rather than thrown away."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = np.asarray(scores, float)
    n = len(y)
    aucs = []
    attempts = 0
    discarded = 0
    max_attempts = n_boot * 20
    while len(aucs) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == n:
            discarded += 1
            continue
        aucs.append(roc_auc_score(yy, scores[idx]))
    if not aucs:
        return {"n_boot_valid": 0, "n_resamples_discarded_single_class": discarded,
                "n_attempts": attempts, "ci95": None, "boot_mean": None,
                "boot_std": None, "lower_bound_above_half": False, "seed": seed}
    a = np.asarray(aucs)
    lo, hi = float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
    return {
        "n_boot_valid": int(len(a)),
        "n_resamples_discarded_single_class": int(discarded),
        "n_attempts": int(attempts),
        "ci95": [lo, hi],
        "boot_mean": float(a.mean()),
        "boot_std": float(a.std()),
        "lower_bound_above_half": bool(lo > 0.5),
        "seed": seed,
        "n_boot_requested": n_boot,
    }


def load_cache(tag, prompt_name):
    path = CACHE_DIR / f"{tag}__{prompt_name}.jsonl"
    if not path.exists():
        return {}
    recs = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {r["pair_id"]: r for r in recs}


def block(y, s, with_bootstrap=True):
    """AUROC + permutation p + LOPO range + bootstrap CI for one score set."""
    y = list(y)
    s = list(s)
    n, n_pos = len(y), int(sum(y))
    out = {"n": n, "n_pos": n_pos}
    if n_pos == 0 or n_pos == n:
        out.update({"auroc": None, "permutation_p": None, "lopo_auroc_min": None,
                    "lopo_auroc_max": None, "bootstrap": None,
                    "note": "single-class stratum; AUROC undefined"})
        return out
    auc, p = auroc_and_perm_p(y, s)
    lo, hi = lopo_range(y, s)
    out.update({
        "auroc": auc,
        "permutation_p": p,
        "permutation_draws": PERM_DRAWS,
        "permutation_seed": PERM_SEED,
        "lopo_auroc_min": lo,
        "lopo_auroc_max": hi,
        "lopo_n_positives_left_out": n_pos,
        "bootstrap": bootstrap_auroc_ci(y, s) if with_bootstrap else None,
    })
    return out


def main():
    g = pd.read_csv(GOLD)
    label_by_pair = dict(zip(g["pair_id"], g["label_clean"]))
    pair_ids_299 = g["pair_id"].tolist()
    dg = pd.read_csv(DERIVED)
    rule_by_pair = dict(zip(dg["pair_id"], dg["rule"]))

    pair_ids_253 = [pid for pid in pair_ids_299 if label_by_pair[pid] != "ambiguous"]
    y253_by_pair = {pid: int(label_by_pair[pid] == "contradiction") for pid in pair_ids_253}

    facts = json.loads(FACTS_PATH.read_text())

    cells = collections.OrderedDict()
    for family, model, prompt in CONFIGS:
        tag = MODEL_TAGS[model]
        recs = load_cache(tag, prompt)
        cell_key = f"{family}::{model}::{prompt}"
        if not recs:
            cells[cell_key] = {"cache_present": False}
            continue

        # ---- G_253 and the two rule strata -------------------------------
        def ids_for(rules=None):
            return [pid for pid in pair_ids_253
                    if pid in recs and recs[pid].get("flag_logodds") is not None
                    and (rules is None or rule_by_pair.get(pid) in rules)]

        blocks = {}
        for name, rules in [("g253", None), ("rules_1_2", (1, 2)), ("rules_3_4", (3, 4))]:
            ids = ids_for(rules)
            blocks[name] = block([y253_by_pair[pid] for pid in ids],
                                 [recs[pid]["flag_logodds"] for pid in ids])
        ids253 = ids_for(None)
        _, p_high = perm_p_high_precision(
            [y253_by_pair[pid] for pid in ids253],
            [recs[pid]["flag_logodds"] for pid in ids253])
        blocks["g253"]["permutation_p_high_precision"] = p_high
        blocks["g253"]["permutation_draws_high_precision"] = PERM_DRAWS_HIGH
        blocks["g253"]["permutation_seed_high_precision"] = PERM_SEED_HIGH

        # ---- registered confound test, identical to analyze_c109.py ------
        neg_ids = [pid for pid in pair_ids_253
                   if label_by_pair[pid] == "not_contradiction"
                   and pid in recs and recs[pid].get("flag_logodds") is not None
                   and rule_by_pair.get(pid) in (1, 2, 3, 4)]
        rule_bin = [1 if rule_by_pair[pid] in (1, 2) else 0 for pid in neg_ids]
        neg_scores = [recs[pid]["flag_logodds"] for pid in neg_ids]
        c_auc, c_p = auroc_and_perm_p(rule_bin, neg_scores)
        confounded = bool(c_auc is not None and c_auc >= CONFOUND_AUROC_THRESHOLD
                          and c_p is not None and c_p < CONFOUND_P_THRESHOLD)
        # The registered C109 test is ONE-SIDED in the "higher score means
        # Rules 1-2" direction: it asks only whether AUROC >= 0.65 with a
        # one-sided permutation p. A cell whose score ranks Rules 3-4 ABOVE
        # Rules 1-2 has exactly as much of a rule-type cue, and the
        # registered rule cannot see it (its AUROC is far BELOW 0.5, and its
        # one-sided p is near 1). This direction-free diagnostic applies the
        # SAME two thresholds to max(AUROC, 1 - AUROC) and to the two-sided
        # permutation p. The registered rule is not changed; both are
        # reported, and `confounded` above remains the registered verdict.
        c_auc_rev, c_p_rev = auroc_and_perm_p(
            [1 - b for b in rule_bin], neg_scores)
        c_auc_df = max(c_auc, 1.0 - c_auc) if c_auc is not None else None
        c_p_df = None
        if c_p is not None and c_p_rev is not None:
            c_p_df = min(1.0, 2.0 * min(c_p, c_p_rev))
        confounded_df = bool(c_auc_df is not None and c_auc_df >= CONFOUND_AUROC_THRESHOLD
                             and c_p_df is not None and c_p_df < CONFOUND_P_THRESHOLD)
        cue_direction = None
        if c_auc is not None:
            cue_direction = ("higher_score_means_rules_1_2" if c_auc > 0.5
                             else "higher_score_means_rules_3_4" if c_auc < 0.5 else "none")

        # ---- agreement with the already-registered AUROC, where one exists
        short = tag.strip('_').replace('_nothink', '')
        reg_key = f"{family}.{short}.{prompt}.auroc_253"
        reg = facts.get(reg_key, {}).get("value")
        recomputed = blocks["g253"]["auroc"]
        agrees = (reg is not None and recomputed is not None
                  and abs(float(reg) - recomputed) < 1e-9)

        cells[cell_key] = {
            "cache_present": True,
            "family": family,
            "model": model,
            "prompt": prompt,
            "n_cache_rows": len(recs),
            "g253": blocks["g253"],
            "stratum_rules_1_2": blocks["rules_1_2"],
            "stratum_rules_3_4": blocks["rules_3_4"],
            "confound_rule12_vs_rule34_among_confirmed_negatives": {
                "n_negatives": len(neg_ids),
                "n_rule12": int(sum(rule_bin)),
                "n_rule34": int(len(rule_bin) - sum(rule_bin)),
                "auroc_rule12_vs_rule34": c_auc,
                "permutation_p": c_p,
                "auroc_threshold": CONFOUND_AUROC_THRESHOLD,
                "p_threshold": CONFOUND_P_THRESHOLD,
                "confounded": confounded,
                "auroc_reverse_direction": c_auc_rev,
                "permutation_p_reverse_direction": c_p_rev,
                "direction_free_auroc_max_auc_1minusauc": c_auc_df,
                "direction_free_permutation_p_two_sided": c_p_df,
                "direction_free_confounded_same_thresholds": confounded_df,
                "cue_direction": cue_direction,
                "registered_rule_is_one_sided": True,
                "already_registered": f"{family}.{tag.strip('_').replace('_nothink','')}.{prompt}.confound.confounded" in facts,
            },
            "registered_auroc_253_key": reg_key,
            "registered_auroc_253_value": reg,
            "recomputed_auroc_253_matches_register": bool(agrees),
        }

    # ---- summaries -------------------------------------------------------
    scored = {k: v for k, v in cells.items() if v.get("cache_present")}
    lower_above_half = sorted(
        k for k, v in scored.items()
        if v["g253"]["bootstrap"] and v["g253"]["bootstrap"]["lower_bound_above_half"])
    confounded_cells = sorted(
        k for k, v in scored.items()
        if v["confound_rule12_vs_rule34_among_confirmed_negatives"]["confounded"])
    confounded_direction_free = sorted(
        k for k, v in scored.items()
        if v["confound_rule12_vs_rule34_among_confirmed_negatives"]
        ["direction_free_confounded_same_thresholds"])
    rule_cue_significant_any_direction = sorted(
        k for k, v in scored.items()
        if (v["confound_rule12_vs_rule34_among_confirmed_negatives"]
            ["direction_free_permutation_p_two_sided"] or 1.0) < CONFOUND_P_THRESHOLD)
    newly_confounded = sorted(
        k for k in confounded_cells
        if not scored[k]["confound_rule12_vs_rule34_among_confirmed_negatives"]["already_registered"])
    alpha_bonf = 0.05 / len(CONFIGS)
    rejects_05 = sorted(k for k, v in scored.items()
                        if (v["g253"]["permutation_p"] or 1.0) < 0.05)
    survives_bonf = sorted(k for k, v in scored.items()
                           if (v["g253"]["permutation_p"] or 1.0) < alpha_bonf)
    survives_bonf_high = sorted(
        k for k, v in scored.items()
        if (v["g253"]["permutation_p_high_precision"] or 1.0) < alpha_bonf)
    # The 17 "configurations" are not 17 distinct cells: P_reversed_rerun is
    # C109's determinism re-run of C103's P_reversed on the same model, and
    # both are counted. Identify the duplicate AUROC values so the count is
    # not read as 17 independent wordings.
    by_auroc = collections.defaultdict(list)
    for k, v in scored.items():
        if v["g253"]["auroc"] is not None:
            by_auroc[round(v["g253"]["auroc"], 10)].append(k)
    duplicate_groups = [sorted(g) for g in by_auroc.values() if len(g) > 1]
    mismatches = sorted(
        k for k, v in scored.items()
        if v["registered_auroc_253_value"] is not None
        and not v["recomputed_auroc_253_matches_register"])

    out = {
        "what_this_closes": [
            "bootstrap 95 percent AUROC interval and LOPO range for all 17 "
            "graded-score labeller configurations on G_253 and on the Rules "
            "1-2 / Rules 3-4 strata (previously present for the four "
            "llama3:8b C109 cells only)",
            "the C109 registered confound test (rule-type cue among confirmed "
            "negatives) run on the nine C103 promptsens cells, which never "
            "carried it",
        ],
        "inputs": {
            "per_pair_score_caches": "~/mimic_local_only/accord_prompt_sensitivity_cache/*.jsonl (flag_logodds, the graded score; never the hard label)",
            "gold_labels": "pharmacist_labeled_clean.csv::label_clean",
            "rules": "derived/gold_standard_accord_results.csv::rule",
            "no_new_inference": True,
        },
        "method": {
            "auroc_and_permutation_p": "accord/prompt_sensitivity/analyze_c109.py::auroc_and_perm_p",
            "permutation_draws": PERM_DRAWS,
            "permutation_seed": PERM_SEED,
            "lopo": "accord/prompt_sensitivity/analyze_c109.py::lopo_range",
            "bootstrap": "accord/control/report.py::bootstrap_auroc_ci (percentile, single-class resamples discarded)",
            "bootstrap_draws": N_BOOT,
            "bootstrap_seed": BOOT_SEED,
            "confound_test": "accord/prompt_sensitivity/analyze_c109.py registered thresholds",
            "note_on_earlier_bootstrap": (
                "the four llama3:8b C109 bootstrap intervals already in the "
                "register came from analyze_c109.py::stratified_bootstrap_ci, "
                "a STRATIFIED bootstrap at B = 5000, seed 42. This run uses "
                "the plain percentile bootstrap at B = 10000, the "
                "construction accord/control/report.py uses everywhere else "
                "in the thesis, so all 17 cells are now on one method; the "
                "llama3:8b numbers here therefore need not equal the "
                "stratified ones already registered."),
        },
        "n_configurations": len(CONFIGS),
        "n_configurations_with_cache": len(scored),
        "g253_n": len(pair_ids_253),
        "g253_n_pos": int(sum(y253_by_pair.values())),
        "cells": cells,
        "summary": {
            "cells_with_bootstrap_lower_bound_above_0_5_on_g253": lower_above_half,
            "n_cells_with_bootstrap_lower_bound_above_0_5_on_g253": len(lower_above_half),
            "confounded_cells": confounded_cells,
            "confounded_cells_direction_free_same_thresholds": confounded_direction_free,
            "cells_with_a_significant_rule_type_cue_in_either_direction": rule_cue_significant_any_direction,
            "confounded_cells_not_previously_tested": newly_confounded,
            "n_confounded_cells": len(confounded_cells),
            "recomputed_auroc_mismatches_vs_register": mismatches,
            "n_configurations_rejecting_chance_at_0_05": len(rejects_05),
            "configurations_not_rejecting_chance_at_0_05": sorted(
                set(scored) - set(rejects_05)),
            "bonferroni_alpha_over_17": alpha_bonf,
            "n_surviving_bonferroni_at_registered_B_20000": len(survives_bonf),
            "surviving_bonferroni_at_registered_B_20000": survives_bonf,
            "n_surviving_bonferroni_at_B_200000": len(survives_bonf_high),
            "surviving_bonferroni_at_B_200000": survives_bonf_high,
            "duplicate_auroc_groups": duplicate_groups,
            "duplicate_note": (
                "P_reversed_rerun is C109's determinism re-run of C103's "
                "P_reversed on the same model and the same 299 pairs. It "
                "reproduces P_reversed's AUROC to every printed digit, which "
                "is what a passing determinism check means. The 17 "
                "configurations therefore contain 2 duplicate pairs and "
                "cover 15 distinct model-and-wording cells; a count of 17 "
                "must not be read as 17 independent wordings."),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")
    print(f"cells with bootstrap lower bound > 0.5 on G_253: {lower_above_half}")
    print(f"confounded cells (registered one-sided rule): {confounded_cells}")
    print(f"confounded cells (direction-free, same thresholds): {confounded_direction_free}")
    print(f"significant rule-type cue in either direction: {rule_cue_significant_any_direction}")
    print(f"newly tested and confounded: {newly_confounded}")
    print(f"register AUROC mismatches: {mismatches}")


if __name__ == "__main__":
    main()
