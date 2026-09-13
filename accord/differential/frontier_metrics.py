"""Aggregate metrics for the compliant frontier run.

Reads the two committed prediction artifacts and the pharmacist labels, and
writes one aggregate JSON. Every number the thesis quotes about this run comes
from here rather than from a hand computation, which is the standard C15 and R1
set for the rest of the project.

Inputs:
  results/accord/differential/frontier_anthropic_{zero,few}_shot.json
      pair_id and pred only, already committed
  ACCORD_LABELS  path to pharmacist_labeled_clean.csv, which lives outside the
      version-controlled tree because it carries verbatim note sentences
  ACCORD_FRONTIER_SUFFIX    optional, e.g. _rep5 to score a replicated run
  ACCORD_FRONTIER_PROVIDER  optional, defaults to anthropic; set to bedrock to
                            score the within-vendor run of C68
  ACCORD_FRONTIER_MODEL_TAG optional, the model tag the runner put in the
                            filename, before the strategy
      instead of the single-draw one. Moves the output file by the same suffix.

Output:
  results/accord/differential/frontier_anthropic_metrics.json
      aggregates only. No pair identifier, no note text, no admission or
      subject identifier is read into the output or printed.

Two evaluation bases, because the thesis reports on both and mixing them is the
error tab:denoms exists to prevent:
  G_299  every pair carrying a pharmacist label
  G_253  the same set with the 46 pharmacist-ambiguous pairs dropped

    export ACCORD_LABELS=/path/to/pharmacist_labeled_clean.csv
    python3 accord/differential/frontier_metrics.py
"""

import json
import math
import collections
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom

B = 10000
SEED = 42          # the seed convention used by every other bootstrap here
STRATEGIES = ("zero_shot", "few_shot")
# Which prediction artifacts to score. Empty is the committed single-draw run;
# ACCORD_FRONTIER_SUFFIX=_rep5 scores a replicated run instead. It moves the
# output too, because frontier_anthropic_metrics.json is the artifact
# VERIFIED_FACTS.json cites and a replicated run must not overwrite it.
SUFFIX = os.environ.get("ACCORD_FRONTIER_SUFFIX", "")
# Which compliant channel's artifacts to score. The runner writes the provider
# into the filename, so this must match it or the open fails loudly rather than
# scoring the wrong model, which is the failure this file already carries a
# comment about below.
PROVIDER = os.environ.get("ACCORD_FRONTIER_PROVIDER", "anthropic")
# The runner tags the model BEFORE the strategy (frontier_anthropic<tag>_zero_shot)
# and the replicate suffix AFTER it (..._zero_shot_rep5). Two different positions,
# so two different variables. An earlier version had only the trailing one and
# looked for a file the runner never writes.
MODEL_TAG = os.environ.get("ACCORD_FRONTIER_MODEL_TAG", "")
OUT = ("results/accord/differential/"
       f"frontier_{PROVIDER}_metrics{MODEL_TAG}{SUFFIX}.json")


def cells(y, yh):
    return (int(((y == 1) & (yh == 1)).sum()), int(((y == 0) & (yh == 1)).sum()),
            int(((y == 1) & (yh == 0)).sum()), int(((y == 0) & (yh == 0)).sum()))


def mcc_from(tp, fp, fn, tn):
    d = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


def evaluate(y, yh):
    """Point metrics plus a percentile bootstrap on MCC."""
    tp, fp, fn, tn = cells(y, yh)
    n = len(y)
    recall = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rng = np.random.default_rng(SEED)
    boot = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        boot[b] = mcc_from(*cells(y[i], yh[i]))
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    n_pos, n_flag = int(y.sum()), int(yh.sum())
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_positives": n_pos, "n_flagged": n_flag,
        "recall": recall, "specificity": spec, "precision": prec,
        "fpr": 1.0 - spec,
        "balanced_accuracy": (recall + spec) / 2.0,
        "f1": (2 * prec * recall / (prec + recall)) if prec and recall else float("nan"),
        "mcc": mcc_from(tp, fp, fn, tn),
        "mcc_ci_lo": lo, "mcc_ci_hi": hi,
        "mcc_ci_excludes_zero": bool(lo > 0.0),
        "mcc_bootstrap_B": B, "mcc_bootstrap_seed": SEED,
        "base_rate": n_pos / n,
        "flag_rate": n_flag / n,
        "over_calling_ratio": (n_flag / n) / (n_pos / n) if n_pos else float("nan"),
        "hypergeom_expected_tp_under_null": n_pos * n_flag / n,
        "hypergeom_p_one_sided": float(hypergeom.sf(tp - 1, n, n_pos, n_flag)),
    }


FACTS = "results/accord/VERIFIED_FACTS.json"
# One prefix per labelled run. The default Anthropic run owns the bare prefix
# because the chapters already cite it; every other provider and model tag gets
# its own, so a second labeller cannot overwrite the first's numbers in the
# register any more than it can overwrite its artifacts. Hand-maintained keys
# are what let a withdrawn label-noise construction keep 96 live keys in this
# file after the script behind them was replaced; see CORRECTIONS.md C69.
PREFIX = ("frontier" if PROVIDER == "anthropic" and not MODEL_TAG
          else f"frontier_{PROVIDER}{MODEL_TAG}")


def update_register(out):
    """Rewrite this run's block of the fact register from what it just computed."""
    facts = json.load(open(FACTS), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    def put(key, value):
        facts[f"{PREFIX}.{key}"] = collections.OrderedDict(
            [("artifact", OUT), ("value", value)])

    n = 0
    for strat, sd in out["by_strategy"].items():
        for basis in ("G_299", "G_253"):
            if basis not in sd:
                continue
            # Every scalar, not a chosen subset. Choosing a subset is how the
            # first version of this function deleted keys the chapters cite,
            # including the Matthews interval bounds, and replaced them with
            # nothing: the deletion is by prefix and cannot be selective, so the
            # write has to be total.
            for field, value in sd[basis].items():
                if isinstance(value, (int, float, bool)) and (
                        value == value):        # drop NaN, which has no fact
                    put(f"{strat}.{basis}.{field}", value)
                    n += 1
    for strat, g in out.get("ambiguity_gradient", {}).items():
        for field in ("odds_ratio", "fisher_p_one_sided", "flagged_ambiguous",
                      "flagged_clear_negative", "n_ambiguous",
                      "n_clear_negative", "rate_ambiguous",
                      "rate_clear_negative"):
            if field in g:
                put(f"ambiguity_gradient.{strat}.{field}", g[field])
                n += 1
    json.dump(facts, open(FACTS, "w"), indent=1)
    return n


def main():
    labels_path = os.environ.get("ACCORD_LABELS")
    if not labels_path:
        sys.exit("ACCORD_LABELS is not set. It must point at "
                 "pharmacist_labeled_clean.csv outside the repository.")
    lab = pd.read_csv(labels_path, usecols=["pair_id", "label_clean"])
    lab["gold"] = (lab.label_clean == "contradiction").astype(int)

    bases = {"G_299": lab,
             "G_253": lab[lab.label_clean != "ambiguous"]}

    out = {
        "provenance": {
            "generated_by": "accord/differential/frontier_metrics.py",
            "prediction_artifacts": [
                f"results/accord/differential/frontier_{PROVIDER}{MODEL_TAG}_{s}{SUFFIX}.json"
                for s in STRATEGIES],
            "labels": "pharmacist_labeled_clean.csv (outside version control)",
            "note": ("Aggregates only. The labels file carries verbatim note "
                     "sentences and is never copied, quoted or summarised here."),
        },
        "label_counts": {k: int(v) for k, v in
                         lab.label_clean.value_counts().items()},
        "by_strategy": {},
        "ambiguity_gradient": {},
    }

    expected_model = os.environ.get("ACCORD_FRONTIER_EXPECTED_MODEL")
    expected_think_raw = os.environ.get("ACCORD_FRONTIER_EXPECTED_THINK")
    if MODEL_TAG and not (expected_model and expected_think_raw is not None):
        sys.exit("ACCORD_FRONTIER_EXPECTED_MODEL and "
                 "ACCORD_FRONTIER_EXPECTED_THINK must both be set when "
                 "ACCORD_FRONTIER_MODEL_TAG is set, so this script cannot "
                 "silently score whichever artifact happens to sit at that "
                 "tag under a model or think setting nobody asked for.")
    expected_think = ({"true": True, "false": False}[expected_think_raw.lower()]
                      if MODEL_TAG else None)

    raw_by_strategy = {}
    for strat in STRATEGIES:
        art = ("results/accord/differential/"
               f"frontier_{PROVIDER}{MODEL_TAG}_{strat}{SUFFIX}.json")
        raw = json.load(open(art))
        if MODEL_TAG:
            if raw.get("model") != expected_model:
                sys.exit(f"{art}: model {raw.get('model')!r} != expected "
                         f"{expected_model!r}")
            if raw.get("think") != expected_think:
                sys.exit(f"{art}: think {raw.get('think')!r} != expected "
                         f"{expected_think!r}")
        raw_by_strategy[strat] = raw
        pr = pd.DataFrame(raw["predictions"])
        # A replicated run writes pred=null where the draws split evenly or none
        # parsed. Those pairs carry no label, so they are dropped and counted:
        # left in they arrive as NaN, cells() puts them in none of tp/fp/fn/tn
        # while n still counts them, and int(yh.sum()) in evaluate() raises.
        n_unresolved = int(pr.pred.isna().sum())
        pr = pr[pr.pred.notna()]
        out["by_strategy"][strat] = {
            "model": raw["model"], "channel": raw["channel"],
            "n_pairs_scored": raw["n_pairs"],
            "n_pairs_unresolved_dropped": n_unresolved,
            "failed_parses": raw["failed_parses"],
            "temperature_supported": raw["temperature_supported"],
            "max_tokens": raw["max_tokens"],
            "max_tokens_original_config": raw.get("max_tokens_original_config"),
            "scored_not_in_gold": int(len(set(pr.pair_id) - set(lab.pair_id))),
        }
        for name, gset in bases.items():
            m = pr.merge(gset[["pair_id", "gold"]], on="pair_id", how="inner")
            out["by_strategy"][strat][name] = evaluate(
                m.gold.to_numpy(), m.pred.to_numpy())

        # Do the false positives land on pairs the pharmacist found hard? If
        # they do, the ambiguous category is measuring evidential difficulty
        # rather than annotator noise, and dropping those 46 rows penalises
        # whichever system is more sensitive to it.
        m = pr.merge(lab, on="pair_id", how="inner")
        amb = m[m.label_clean == "ambiguous"]
        neg = m[m.label_clean == "not_contradiction"]
        a, c = int(amb.pred.sum()), int(neg.pred.sum())
        odds, p = fisher_exact([[a, len(amb) - a], [c, len(neg) - c]],
                               alternative="greater")
        out["ambiguity_gradient"][strat] = {
            "n_ambiguous": len(amb), "flagged_ambiguous": a,
            "rate_ambiguous": a / len(amb),
            "n_clear_negative": len(neg), "flagged_clear_negative": c,
            "rate_clear_negative": c / len(neg),
            "odds_ratio": float(odds), "fisher_p_one_sided": float(p),
        }

    # Both strategy artifacts must be the SAME model run under the SAME think
    # setting, sharing one digest. Scoring a zero-shot run of one model against
    # a few-shot run of another under one register prefix is exactly the class
    # of defect CLAUDE.md records twice.
    if PROVIDER == "ollama":
        models = {raw_by_strategy[s].get("model") for s in STRATEGIES}
        thinks = {raw_by_strategy[s].get("think") for s in STRATEGIES}
        digests = {raw_by_strategy[s].get("model_digest") for s in STRATEGIES}
        if len(models) != 1 or len(thinks) != 1 or len(digests) != 1:
            sys.exit(f"strategy artifacts disagree on model/think/digest: "
                     f"models={models} thinks={thinks} digests={digests}")
        out["model"] = models.pop()
        out["think"] = thinks.pop()
        out["model_digest"] = digests.pop()

    print(f"register keys under {PREFIX}.*: {update_register(out)} written")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"wrote {OUT}")
    for strat in STRATEGIES:
        s = out["by_strategy"][strat]["G_253"]
        print(f"  {strat:<10} G_253  MCC {s['mcc']:+.4f} "
              f"[{s['mcc_ci_lo']:+.4f}, {s['mcc_ci_hi']:+.4f}]  "
              f"over-calling {s['over_calling_ratio']:.2f}x  "
              f"bal.acc {s['balanced_accuracy']:.3f}")


if __name__ == "__main__":
    main()
