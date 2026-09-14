"""C109 analysis: primary decision (llama3:8b, 3 new paraphrases vs the
registered MDE), the determinism check against C103's committed
P_reversed cache, the registered confound test (rule-type cue among
negatives), the registered stratum report (Rules 1-2 vs Rules 3-4), and
the secondary granite3.3:8b + LOPO/bootstrap numbers. See
C109_PREREGISTRATION.md; the decision rule and every threshold here are
copied from that file, never re-derived.

Opens the local numeric-only per-pair cache under
~/mimic_local_only/accord_prompt_sensitivity_cache/ (pair_id, flag label,
flag log-odds, degenerate, parse-failure -- never sentence text), the gold
CSV (only to recover label_clean per pair_id) and the derived results CSV
(only to recover rule per pair_id). Writes AGGREGATE ONLY to
results/accord/prompt_sensitivity/c109/analysis.json.

Per the task's compliance rule, every process that opens
~/mimic_local_only must send its full stdout/stderr to a log file under
~/mimic_local_only/logs/; run this script with that redirection, e.g.:

    python3 accord/prompt_sensitivity/analyze_c109.py \\
        > ~/mimic_local_only/logs/c109_analyze_<ts>.log 2>&1

--dry-run: reads from a --cache-dir / --gold / --derived override instead
of ~/mimic_local_only, so the harness can be exercised end-to-end on
INVENTED data (see dry_run_fixtures.py) without ever touching real MIMIC
data. In --dry-run, the gold/derived tables are the invented fixture
itself, not read from a CSV.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accord.prompt_sensitivity.dry_run_fixtures import INVENTED_GOLD  # noqa: E402
from accord.prompt_sensitivity.prompts_c109 import NEW_PARAPHRASE_NAMES  # noqa: E402

GOLD = pathlib.Path.home() / "mimic_local_only/from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv"
DERIVED = pathlib.Path.home() / "mimic_local_only/derived/gold_standard_accord_results.csv"
REAL_CACHE_DIR = pathlib.Path.home() / "mimic_local_only/accord_prompt_sensitivity_cache"
REAL_ATTEMPTS_PATH = ROOT / "results/accord/prompt_sensitivity/c109/attempts.json"
OUT_PATH = ROOT / "results/accord/prompt_sensitivity/c109/analysis.json"

MODELS = ["llama3:8b", "granite3.3:8b"]
MODEL_TAGS = {"llama3:8b": "_llama3_8b_nothink", "granite3.3:8b": "_granite33_8b_nothink"}
NEW_PROMPTS = list(NEW_PARAPHRASE_NAMES)  # P_rev_agree, P_rev_follow, P_rev_compatible
ALL_PROMPTS = NEW_PROMPTS + ["P_reversed_rerun"]

MDE_AUROC_253 = 0.7355  # results/accord/mde_recomputed.json::strata.gold_ambiguous_excluded.minimum_detectable_auc
MDE_AUROC_DEFECT_FREE = 0.8058  # results/accord/mde_recomputed.json::strata.defect_free_subset.minimum_detectable_auc
BONFERRONI_ALPHA_3 = 0.05 / 3
CONFOUND_AUROC_THRESHOLD = 0.65
CONFOUND_P_THRESHOLD = 0.05
PERM_DRAWS = 20000
PERM_SEED = 109
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 42
DETERMINISM_SCORE_TOL = 0.05


def load_cache(cache_dir, tag, prompt_name):
    path = cache_dir / f"{tag}__{prompt_name}.jsonl"
    if not path.exists():
        return {}
    recs = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {r["pair_id"]: r for r in recs}


def auroc_and_perm_p(y, s, B=PERM_DRAWS, seed=PERM_SEED):
    """One-sided label-permutation p that a permuted AUROC >= the observed
    one, matching audit_reversed.py's convention exactly (shuffle the
    labels, recompute AUROC on the fixed scores, p = (1 + count >=
    observed) / (1 + B)).

    Vectorized over all B draws at once via the rank-sum (Mann-Whitney U)
    identity for AUROC -- mathematically identical to calling
    sklearn.metrics.roc_auc_score(y_permuted, s) B times (both reduce to
    the same tie-aware rank-sum statistic for the binary, unweighted case),
    but tens of thousands of times faster: only the label/positive-set
    assignment changes across draws, never the scores, so the score ranks
    are computed once and reused.
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
    # For each draw, a uniformly random size-k subset of positions is the
    # permuted "positive" set (equivalent to shuffling y and reading off
    # the new positive positions).
    idx = np.argsort(rng.random((B, n)), axis=1)[:, :k]
    ranksum_perm = ranks[idx].sum(axis=1)
    auc_perm = (ranksum_perm - k * (k + 1) / 2) / (k * (n - k))
    cnt = int(np.sum(auc_perm >= a - 1e-12))
    p = (cnt + 1) / (B + 1)
    return a, float(p)


def lopo_range(y, s):
    """Leave-one-positive-out AUROC range, same method audit_reversed.py used."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    pos_idx = np.where(y == 1)[0]
    vals = []
    for j in pos_idx:
        m = np.ones(len(s), bool)
        m[j] = False
        if len(set(y[m].tolist())) < 2:
            continue
        vals.append(float(roc_auc_score(y[m], s[m])))
    if not vals:
        return None, None
    return float(min(vals)), float(max(vals))


def stratified_bootstrap_ci(y, s, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    y = np.asarray(y)
    s = np.asarray(s, float)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(draws):
        bi = np.concatenate([rng.choice(pos_idx, len(pos_idx)),
                              rng.choice(neg_idx, len(neg_idx))])
        bs.append(roc_auc_score(y[bi], s[bi]))
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def load_gold_and_rule(dry_run, gold_path, derived_path):
    """Returns (label_by_pair, rule_by_pair, pair_ids_299).

    Real run: label_clean from the gold CSV, rule from the derived CSV,
    joined on pair_id. --dry-run: both come from the same INVENTED fixture
    table, never from a CSV.
    """
    if dry_run:
        label_by_pair = {pid: label for pid, _, _, label, _ in INVENTED_GOLD}
        rule_by_pair = {pid: rule for pid, _, _, _, rule in INVENTED_GOLD}
        pair_ids_299 = [pid for pid, _, _, _, _ in INVENTED_GOLD]
        return label_by_pair, rule_by_pair, pair_ids_299

    g = pd.read_csv(gold_path)
    label_by_pair = dict(zip(g["pair_id"], g["label_clean"]))
    pair_ids_299 = g["pair_id"].tolist()
    dg = pd.read_csv(derived_path)
    rule_by_pair = dict(zip(dg["pair_id"], dg["rule"]))
    return label_by_pair, rule_by_pair, pair_ids_299


def cell_diagnostics(recs, n_expected):
    n = len(recs)
    n_parse_fail = sum(1 for r in recs.values() if r.get("parse_failure"))
    n_degenerate = sum(1 for r in recs.values() if r.get("degenerate"))
    n_truncated = sum(1 for r in recs.values() if r.get("truncated"))
    n_thinking = sum(1 for r in recs.values() if r.get("thinking_nonempty"))
    parse_fail_rate = n_parse_fail / n if n else None
    degenerate_rate = n_degenerate / n if n else None
    return {
        "n": n, "n_expected": n_expected,
        "n_parse_failures": n_parse_fail,
        "parse_failure_rate": round(parse_fail_rate, 4) if parse_fail_rate is not None else None,
        "n_degenerate": n_degenerate,
        "degenerate_rate": round(degenerate_rate, 4) if degenerate_rate is not None else None,
        "n_truncated": n_truncated,
        "n_thinking_nonempty": n_thinking,
        "meets_parse_failure_floor": bool(n and parse_fail_rate is not None and parse_fail_rate <= 0.05),
        "meets_degenerate_floor": bool(n and degenerate_rate is not None and degenerate_rate <= 0.05),
        "meets_truncation_floor": n_truncated == 0,
    }


def determinism_check(rerun_recs, c103_recs, pair_ids):
    """P_reversed_rerun vs C103's committed P_reversed cache: hard labels
    must match exactly on every commonly-scored pair, and flag log-odds
    within DETERMINISM_SCORE_TOL."""
    common = [pid for pid in pair_ids
              if pid in rerun_recs and pid in c103_recs
              and rerun_recs[pid].get("flag_label") is not None
              and c103_recs[pid].get("flag_label") is not None]
    n_label_mismatch = sum(1 for pid in common
                            if rerun_recs[pid]["flag_label"] != c103_recs[pid]["flag_label"])
    score_diffs = [abs(rerun_recs[pid]["flag_logodds"] - c103_recs[pid]["flag_logodds"])
                   for pid in common
                   if rerun_recs[pid].get("flag_logodds") is not None
                   and c103_recs[pid].get("flag_logodds") is not None]
    max_score_diff = max(score_diffs) if score_diffs else None
    passes = bool(len(common) > 0 and n_label_mismatch == 0
                  and (max_score_diff is None or max_score_diff <= DETERMINISM_SCORE_TOL))
    return {
        "n_common_scored": len(common),
        "n_c103_cache_present": len(c103_recs),
        "n_rerun_cache_present": len(rerun_recs),
        "n_label_mismatch": n_label_mismatch,
        "max_score_diff": max_score_diff,
        "score_tolerance": DETERMINISM_SCORE_TOL,
        "passes": passes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                     help="read from --cache-dir (INVENTED fixture labels), "
                          "never touch ~/mimic_local_only")
    ap.add_argument("--cache-dir", default=None,
                     help="override the per-pair cache directory "
                          "(required with --dry-run)")
    ap.add_argument("--attempts-path", default=None,
                     help="override attempts.json path (dry-run)")
    ap.add_argument("--out", default=None, help="override output path")
    args = ap.parse_args()

    if args.dry_run:
        if not args.cache_dir:
            sys.exit("--dry-run requires --cache-dir pointing at the temp "
                      "dir run_c109.py --dry-run printed")
        cache_dir = pathlib.Path(args.cache_dir)
        attempts_path = pathlib.Path(args.attempts_path) if args.attempts_path else cache_dir.parent / "attempts.json"
    else:
        cache_dir = REAL_CACHE_DIR
        attempts_path = REAL_ATTEMPTS_PATH

    out_path = pathlib.Path(args.out) if args.out else OUT_PATH

    label_by_pair, rule_by_pair, pair_ids_299 = load_gold_and_rule(
        args.dry_run, GOLD, DERIVED)
    pair_ids_253 = [pid for pid in pair_ids_299 if label_by_pair[pid] != "ambiguous"]
    y253_by_pair = {pid: int(label_by_pair[pid] == "contradiction") for pid in pair_ids_253}

    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else {}

    per_model_prompt = {}
    invalid_cells = []
    killed_twice_cells = []
    for model in MODELS:
        tag = MODEL_TAGS[model]
        by_prompt = {}
        for prompt_name in ALL_PROMPTS:
            recs = load_cache(cache_dir, tag, prompt_name)
            diag = cell_diagnostics(recs, len(pair_ids_299))
            attempt_n = attempts.get(f"{tag}::{prompt_name}", 1)
            killed_twice = attempt_n > 2
            if killed_twice:
                killed_twice_cells.append(f"{model}::{prompt_name}")

            ids253 = [pid for pid in pair_ids_253
                      if pid in recs and recs[pid].get("flag_logodds") is not None]
            y253 = [y253_by_pair[pid] for pid in ids253]
            s253 = [recs[pid]["flag_logodds"] for pid in ids253]
            auc253, p253 = auroc_and_perm_p(y253, s253)

            valid = bool(diag["n"] == len(pair_ids_299)
                         and diag["meets_parse_failure_floor"]
                         and diag["meets_degenerate_floor"]
                         and diag["meets_truncation_floor"]
                         and not killed_twice)
            if not valid:
                invalid_cells.append(f"{model}::{prompt_name}")

            by_prompt[prompt_name] = {
                **diag,
                "attempt_number": attempt_n,
                "killed_twice": killed_twice,
                "auroc_253": auc253,
                "permutation_p_253": p253,
                "valid": valid,
            }
        per_model_prompt[model] = by_prompt

    # Determinism check: P_reversed_rerun vs C103's committed P_reversed cache.
    determinism = {}
    for model in MODELS:
        tag = MODEL_TAGS[model]
        rerun_recs = load_cache(cache_dir, tag, "P_reversed_rerun")
        c103_recs = load_cache(cache_dir, tag, "P_reversed")
        d = determinism_check(rerun_recs, c103_recs, pair_ids_299)
        determinism[model] = d
        if not d["passes"]:
            invalid_cells.append(f"{model}::determinism_check")
            # Per C109_PREREGISTRATION.md: "a granite3.3:8b-only invalidity
            # does not by itself force Outcome C for the primary decision
            # ... it is still reported and still disqualifies
            # granite3.3:8b's cells from the Secondary section." A failed
            # determinism check for a model disqualifies ALL of that
            # model's cells (not only P_reversed_rerun) from being read as
            # trustworthy Secondary evidence -- e.g. a model known to drift
            # beyond the 0.05 score tolerance at temperature 0 on this
            # machine (C104) cannot be presumed reproducible on its OTHER
            # three cells either. This never touches the primary decision,
            # which is stated only over llama3:8b.
            for prompt_name in ALL_PROMPTS:
                per_model_prompt[model][prompt_name]["valid"] = False
                per_model_prompt[model][prompt_name]["disqualified_reason"] = (
                    f"{model} determinism check failed "
                    f"(n_label_mismatch={d['n_label_mismatch']}, "
                    f"max_score_diff={d['max_score_diff']}); all of this "
                    f"model's cells are disqualified from Secondary "
                    f"reporting, per the pre-registered handling of a "
                    f"single model's invalidity")
                cell_key = f"{model}::{prompt_name}"
                if cell_key not in invalid_cells:
                    invalid_cells.append(cell_key)

    # Registered confound test: among negatives only, AUROC of score for
    # rule 1-2 membership vs rule 3-4 membership.
    confound = {}
    for model in MODELS:
        tag = MODEL_TAGS[model]
        for prompt_name in ALL_PROMPTS:
            recs = load_cache(cache_dir, tag, prompt_name)
            neg_ids = [pid for pid in pair_ids_253
                       if label_by_pair[pid] == "not_contradiction"
                       and pid in recs and recs[pid].get("flag_logodds") is not None
                       and rule_by_pair.get(pid) in (1, 2, 3, 4)]
            rule_bin = [1 if rule_by_pair[pid] in (1, 2) else 0 for pid in neg_ids]
            scores = [recs[pid]["flag_logodds"] for pid in neg_ids]
            auc, p = auroc_and_perm_p(rule_bin, scores)
            confounded = bool(auc is not None and auc >= CONFOUND_AUROC_THRESHOLD
                               and p is not None and p < CONFOUND_P_THRESHOLD)
            confound[f"{model}::{prompt_name}"] = {
                "n_negatives": len(neg_ids),
                "n_rule12": int(sum(rule_bin)), "n_rule34": int(len(rule_bin) - sum(rule_bin)),
                "auroc_rule12_vs_rule34": auc, "permutation_p": p,
                "confounded": confounded,
            }

    # Registered stratum report: Rules 1-2 and Rules 3-4, ambiguous dropped.
    stratum = {}
    for model in MODELS:
        tag = MODEL_TAGS[model]
        for prompt_name in ALL_PROMPTS:
            recs = load_cache(cache_dir, tag, prompt_name)
            cell = {}
            for stratum_name, rules in [("rules_1_2", (1, 2)), ("rules_3_4", (3, 4))]:
                ids = [pid for pid in pair_ids_253
                       if rule_by_pair.get(pid) in rules
                       and pid in recs and recs[pid].get("flag_logodds") is not None]
                y = [y253_by_pair[pid] for pid in ids]
                s = [recs[pid]["flag_logodds"] for pid in ids]
                auc, p = auroc_and_perm_p(y, s)
                cell[stratum_name] = {
                    "n": len(ids), "n_pos": int(sum(y)),
                    "auroc": auc, "permutation_p": p,
                }
            cell["rules_3_4"]["mde_defect_free_registered"] = MDE_AUROC_DEFECT_FREE
            cell["rules_3_4"]["note"] = ("cannot confirm a null or an effect at "
                                          "this sample size; only an AUROC >= "
                                          "0.8058 with p<0.05 clears this run's "
                                          "own resolution on this stratum")
            stratum[f"{model}::{prompt_name}"] = cell

    # Primary decision rule: llama3:8b, the 3 NEW paraphrases only.
    llama_valid = all(f"llama3:8b::{p}" not in invalid_cells for p in ALL_PROMPTS) \
        and determinism["llama3:8b"]["passes"]
    paraphrase_verdicts = {}
    n_satisfying = 0
    satisfying_names = []
    for prompt_name in NEW_PROMPTS:
        cell = per_model_prompt["llama3:8b"][prompt_name]
        auc, p = cell["auroc_253"], cell["permutation_p_253"]
        satisfies = bool(auc is not None and auc >= MDE_AUROC_253
                          and p is not None and p < BONFERRONI_ALPHA_3)
        paraphrase_verdicts[prompt_name] = {
            "auroc_253": auc, "permutation_p_253": p,
            "mde_auroc_253": MDE_AUROC_253,
            "bonferroni_alpha_over_3": BONFERRONI_ALPHA_3,
            "satisfies_outcome_a_criterion": satisfies,
        }
        if satisfies:
            n_satisfying += 1
            satisfying_names.append(prompt_name)

    if not llama_valid:
        outcome = "C"
        outcome_note = None
        outcome_reason = (f"llama3:8b cell(s) invalid or determinism check "
                           f"failed: {[c for c in invalid_cells if c.startswith('llama3:8b')]}")
    elif n_satisfying >= 2:
        # Check the registered confound test over the satisfying paraphrases.
        n_confounded_of_satisfying = sum(
            1 for name in satisfying_names
            if confound[f"llama3:8b::{name}"]["confounded"])
        if n_confounded_of_satisfying >= 2:
            outcome = "A"
            outcome_reason = (f"{n_satisfying} of 3 new paraphrases satisfy the "
                               f"AUROC+permutation criteria, but "
                               f"{n_confounded_of_satisfying} of those are "
                               f"registered as carrying a rule-type cue; "
                               f"reported as wording-robust but confounded "
                               f"with rule type, per the fixed interpretation")
            outcome_note = "wording_robust_but_confounded_with_rule_type"
        else:
            outcome = "A"
            outcome_reason = (f"{n_satisfying} of 3 new paraphrases satisfy the "
                               f"AUROC>={MDE_AUROC_253} and permutation "
                               f"p<{BONFERRONI_ALPHA_3} criteria on G_253")
            outcome_note = "wording_robust"
    else:
        outcome = "B"
        outcome_reason = (f"only {n_satisfying} of 3 new paraphrases satisfy "
                           f"the criteria (need >= 2 of 3)")
        outcome_note = None

    # Secondary: LOPO range + stratified bootstrap for each llama3 cell.
    secondary_llama = {}
    for prompt_name in ALL_PROMPTS:
        tag = MODEL_TAGS["llama3:8b"]
        recs = load_cache(cache_dir, tag, prompt_name)
        ids = [pid for pid in pair_ids_253
               if pid in recs and recs[pid].get("flag_logodds") is not None]
        y = [y253_by_pair[pid] for pid in ids]
        s = [recs[pid]["flag_logodds"] for pid in ids]
        lo, hi = lopo_range(y, s)
        ci = stratified_bootstrap_ci(y, s)
        secondary_llama[prompt_name] = {
            "n": len(ids), "n_pos": int(sum(y)),
            "lopo_auroc_min": lo, "lopo_auroc_max": hi,
            "bootstrap95_auroc": ci,
        }

    out = {
        "preregistration": "accord/prompt_sensitivity/C109_PREREGISTRATION.md",
        "mde_auroc_253": MDE_AUROC_253,
        "mde_auroc_defect_free_rules_3_4": MDE_AUROC_DEFECT_FREE,
        "bonferroni_alpha_over_3": BONFERRONI_ALPHA_3,
        "confound_auroc_threshold": CONFOUND_AUROC_THRESHOLD,
        "confound_p_threshold": CONFOUND_P_THRESHOLD,
        "per_model_prompt": per_model_prompt,
        "determinism_check": determinism,
        "confound_test": confound,
        "stratum_report": stratum,
        "primary_paraphrase_verdicts_llama3_8b": paraphrase_verdicts,
        "n_new_paraphrases_satisfying_llama3_8b": n_satisfying,
        "satisfying_paraphrases_llama3_8b": satisfying_names,
        "secondary_llama3_8b_lopo_bootstrap": secondary_llama,
        "invalid_cells": invalid_cells,
        "killed_twice_cells": killed_twice_cells,
        "outcome": outcome,
        "outcome_note": outcome_note,
        "outcome_reason": outcome_reason,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}; outcome={outcome} ({outcome_note}) -- {outcome_reason}")


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
        debug_path = (pathlib.Path(__file__).resolve().parents[2] / "results"
                      / "accord" / "prompt_sensitivity" / "c109" / "analyze_debug.json")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(debug, indent=2) + "\n")
        raise SystemExit(f"failed: {type(exc).__name__} (see {debug_path})")
