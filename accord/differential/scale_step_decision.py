"""C99: within-family scale step, qwen3 8B to 14B. See C99_PREREGISTRATION.md.

Reads the two zero-shot prediction artifacts directly (never opens the gold
CSV itself; McNemar needs only pair_id and pred, joined against the pair_id
universe the pharmacist labels file defines) and the two register keys the
statistic and the ceiling are read from, never recomputed here a second way:

  S = capability.within_family_step_qwen3_8b_to_14b   (capability_ladder.py)
  D = capability.spread_8b_frontier_zero_shot_g299     (capability_ladder.py,
      frozen at the commit that landed qwen3:8b, BEFORE any 14B datum; that
      commit hash is passed in via ACCORD_D_FROZEN_COMMIT and recorded in the
      outcome, never re-derived)

Environment:
  ACCORD_LABELS                path to pharmacist_labeled_clean.csv
  ACCORD_D_FROZEN_COMMIT       the commit hash that landed D, required
  ACCORD_QWEN3_8B_EXPECTED_DIGEST, ACCORD_QWEN3_14B_EXPECTED_DIGEST
                               the digests recorded in C99_PREREGISTRATION.md
  ACCORD_SCALE_STEP_KILLED_TWICE   "1" if the 14B run was OOM-killed a second
                               time (Outcome C by the pre-registered rule),
                               default unset/"0"

Writes: results/accord/differential/scale_step_c99.json

    python3 accord/differential/scale_step_decision.py
"""

import collections
import json
import os
import pathlib
import sys

import pandas as pd
from scipy.stats import binomtest

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIFF = ROOT / "results" / "accord" / "differential"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
OUT = DIFF / "scale_step_c99.json"

ART_8B = DIFF / "frontier_ollama_qwen3_8b_nothink_zero_shot.json"
ART_14B = DIFF / "frontier_ollama_qwen3_14b_nothink_zero_shot.json"

S_KEY = "capability.within_family_step_qwen3_8b_to_14b"
D_KEY = "capability.spread_8b_frontier_zero_shot_g299"


def need(var):
    v = os.environ.get(var)
    if not v:
        sys.exit(f"{var} is not set")
    return v


def exact_mcnemar_two_sided(b, c):
    """b, c: discordant counts (model1-flag-only, model2-flag-only). Exact
    two-sided McNemar is a two-sided exact binomial test on the discordant
    pairs against p=0.5."""
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    return float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)


def main():
    labels_path = need("ACCORD_LABELS")
    d_commit = need("ACCORD_D_FROZEN_COMMIT")
    exp_digest_8b = os.environ.get("ACCORD_QWEN3_8B_EXPECTED_DIGEST")
    exp_digest_14b = os.environ.get("ACCORD_QWEN3_14B_EXPECTED_DIGEST")
    killed_twice = os.environ.get("ACCORD_SCALE_STEP_KILLED_TWICE", "0") == "1"

    if not ART_8B.exists() or not ART_14B.exists():
        sys.exit(f"missing prediction artifact(s): "
                 f"{ART_8B.exists()=} {ART_14B.exists()=}")
    a8 = json.loads(ART_8B.read_text())
    a14 = json.loads(ART_14B.read_text())

    gold = pd.read_csv(labels_path, usecols=["pair_id", "label_clean"])

    invalid_reasons = []
    for name, art, exp_digest in (("8b", a8, exp_digest_8b),
                                  ("14b", a14, exp_digest_14b)):
        if art.get("failed_parses", 1) != 0:
            invalid_reasons.append(f"{name}: failed_parses={art.get('failed_parses')}")
        if art.get("n_thinking_nonempty", 1) != 0:
            invalid_reasons.append(f"{name}: n_thinking_nonempty="
                                   f"{art.get('n_thinking_nonempty')}")
        if art.get("n_prompt_truncated", 1) != 0:
            invalid_reasons.append(f"{name}: n_prompt_truncated="
                                   f"{art.get('n_prompt_truncated')}")
        if art.get("n_pairs") != 300:
            invalid_reasons.append(f"{name}: n_pairs={art.get('n_pairs')} != 300")
        if exp_digest and art.get("model_digest") != exp_digest:
            invalid_reasons.append(
                f"{name}: model_digest={art.get('model_digest')!r} != "
                f"expected {exp_digest!r}")
    if killed_twice:
        invalid_reasons.append("14b run was OOM-killed a second time")

    p8 = pd.DataFrame(a8["predictions"])[["pair_id", "pred"]].rename(
        columns={"pred": "pred_8b"})
    p14 = pd.DataFrame(a14["predictions"])[["pair_id", "pred"]].rename(
        columns={"pred": "pred_14b"})
    joined = (gold[["pair_id"]].merge(p8, on="pair_id", how="inner")
             .merge(p14, on="pair_id", how="inner"))
    n_joined = len(joined)
    if n_joined != 299:
        invalid_reasons.append(f"joined n={n_joined} != 299")
    if joined.pred_8b.isna().any() or joined.pred_14b.isna().any():
        invalid_reasons.append("an unresolved (null) prediction survived the join")

    joined = joined.dropna(subset=["pred_8b", "pred_14b"])
    b = int(((joined.pred_8b == 1) & (joined.pred_14b == 0)).sum())
    c = int(((joined.pred_8b == 0) & (joined.pred_14b == 1)).sum())
    n_discordant = b + c
    mcnemar_p = exact_mcnemar_two_sided(b, c)

    facts = json.loads(FACTS.read_text())
    if S_KEY not in facts:
        sys.exit(f"{S_KEY} not in the register; run capability_ladder.py "
                 f"after both qwen3:8b and qwen3:14b have landed")
    if D_KEY not in facts:
        sys.exit(f"{D_KEY} not in the register; run capability_ladder.py "
                 f"after qwen3:8b has landed")
    S = facts[S_KEY]["value"]
    D = facts[D_KEY]["value"]

    invalid = bool(invalid_reasons)
    if invalid:
        outcome = "C"
        direction = None
    else:
        clears_ceiling = (S >= D) or (S <= 1.0 / D)
        significant = mcnemar_p < 0.05
        if clears_ceiling and significant:
            outcome = "A"
            direction = ("8B_over_calls_more" if S >= D else "14B_over_calls_more")
        else:
            outcome = "B"
            direction = None

    out = collections.OrderedDict([
        ("preregistration", "accord/differential/C99_PREREGISTRATION.md"),
        ("S", S), ("S_source", S_KEY),
        ("D", D), ("D_source", D_KEY),
        ("D_frozen_at_commit", d_commit),
        ("n_discordant", n_discordant), ("b_8b_flag_only", b),
        ("c_14b_flag_only", c),
        ("mcnemar_exact_two_sided_p", mcnemar_p),
        ("n_pairs_8b", a8.get("n_pairs")), ("n_pairs_14b", a14.get("n_pairs")),
        ("failed_parses_8b", a8.get("failed_parses")),
        ("failed_parses_14b", a14.get("failed_parses")),
        ("n_thinking_nonempty_8b", a8.get("n_thinking_nonempty")),
        ("n_thinking_nonempty_14b", a14.get("n_thinking_nonempty")),
        ("n_prompt_truncated_8b", a8.get("n_prompt_truncated")),
        ("n_prompt_truncated_14b", a14.get("n_prompt_truncated")),
        ("model_digest_8b", a8.get("model_digest")),
        ("model_digest_14b", a14.get("model_digest")),
        ("n_joined_pairs", n_joined),
        ("invalid_reasons", invalid_reasons),
        ("outcome", outcome),
        ("direction", direction),
        ("note_14b_not_frontier_scale", True),
    ])
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  S={S:.4f}  D={D:.4f}  discordant b={b} c={c}  "
          f"McNemar p={mcnemar_p:.4f}  outcome={outcome} direction={direction}")
    if invalid_reasons:
        print(f"  INVALID: {invalid_reasons}")


if __name__ == "__main__":
    main()
