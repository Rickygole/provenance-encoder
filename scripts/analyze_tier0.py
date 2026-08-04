"""Final arm comparison across seeds.

Every arm is measured as a fraction of the segment-index bias's gain over
vanilla, because that is the quantity the project's claim is about. Arms that
carry NO segment information (T5-RPE, ALiBi) test whether the effect is really
about documents. The hard block mask tests whether "learned" earns its keep.
Misaligned blocks test whether the word "document" belongs in the claim at all.
"""

from __future__ import annotations

import json
import statistics as stats
from pathlib import Path

RUNS = Path("/Users/rickygole/mimic_local_only/runs")

# (arm, control-suffix) -> (label, uses segment info?)
ARMS = [
    ("A1_10M_st1000_s{s}_shuf", "vanilla (no bias)", False),
    ("A3_10M_st1000_s{s}_rpe_shuf", "T5 relative position bias", False),
    ("A3_10M_st1000_s{s}_alibi_shuf", "ALiBi distance bias", False),
    ("A4_10M_st1000_s{s}_segidx_misalign_shuf", "misaligned block bias", False),
    ("A4_10M_st1000_s{s}_shuf", "true role labels", True),
    ("A4_10M_st1000_s{s}_segidx_shuf", "segment-index bias", True),
    ("A3_10M_st1000_s{s}_hardblk_shuf", "hard block mask (fixed)", True),
]


def load(pat: str) -> list[float]:
    out = []
    for s in (0, 1, 2):
        p = RUNS / pat.format(s=s) / "summary.json"
        if p.exists():
            out.append(json.loads(p.read_text())["final_val_loss"])
    return out


def main() -> None:
    data = {lab: (load(pat), seg) for pat, lab, seg in ARMS}
    have = {k: v for k, (v, _) in data.items() if v}
    if "vanilla (no bias)" not in have:
        raise SystemExit("no vanilla runs yet")

    base = stats.mean(have["vanilla (no bias)"])
    ref = have.get("segment-index bias")
    span = base - stats.mean(ref) if ref else None

    print(f"{'arm':<32}{'n':>3}{'val_loss':>18}{'vs vanilla':>12}{'% of seg-idx':>14}{'seg?':>6}")
    print("-" * 85)
    for lab, (vals, seg) in data.items():
        if not vals:
            continue
        m = stats.mean(vals)
        sd = stats.stdev(vals) if len(vals) > 1 else 0.0
        d = m - base
        pct = f"{(base - m)/span:.0%}" if span else "--"
        print(f"{lab:<32}{len(vals):>3}{m:>11.4f} ±{sd:<5.4f}{d:>12.4f}{pct:>14}"
              f"{'yes' if seg else 'no':>6}")

    print("\n" + "=" * 85)
    print("WHAT EACH CONTROL DECIDES")
    print("=" * 85)
    g = lambda k: stats.mean(have[k]) if k in have else None  # noqa: E731
    segidx, alibi, rpe, hard, mis, role = (
        g("segment-index bias"), g("ALiBi distance bias"),
        g("T5 relative position bias"), g("hard block mask (fixed)"),
        g("misaligned block bias"), g("true role labels"))

    if alibi and segidx and span:
        frac = (base - alibi) / span
        print(f"\nALiBi recovers {frac:.0%}. Residual for segment alignment: "
              f"{alibi - segidx:+.4f} nats.")
        if frac > 0.85:
            print("  -> plain locality explains it. Finding reduces to Press et al. 2022.")
        else:
            print("  -> locality is most of it, but a real segment-alignment residual remains.")
    if rpe and span:
        print(f"\nT5-RPE recovers {(base - rpe)/span:.0%}. "
              f"{'Explains it -- Raffel 2020.' if (base-rpe)/span > 0.85 else 'Does not explain it.'}")
    if mis and segidx and span:
        print(f"\nMisaligned boundaries recover {(base - mis)/span:.0%} vs 100% aligned.")
        if abs(mis - segidx) < 0.05:
            print("  -> alignment does not matter. Remove 'document' from every claim;")
            print("     the finding is 'any coarse blocking helps'.")
        else:
            print(f"  -> alignment is worth {mis - segidx:+.4f} nats. Document boundaries")
            print("     genuinely matter, not just blocking.")
    if hard and segidx:
        print(f"\nHard fixed mask vs learned bias: {hard - segidx:+.4f} nats.")
        if hard < segidx:
            print("  -> the FIXED mask WINS. 'Learned' is decoration. The operative")
            print("     method is intra-document masking (Zhao et al., ACL 2024), and")
            print("     this work reproduces it rather than improving on it.")
        else:
            print("  -> the learned bias beats hard masking. Partial cross-document")
            print("     attention beats none -- this contradicts the standard recipe")
            print("     and is the only genuinely novel claim available here.")
    if role and segidx:
        print(f"\nProvenance semantics: true roles are {role - segidx:+.4f} nats vs an")
        print("  arbitrary segment ordinal. Provenance contributes nothing.")


if __name__ == "__main__":
    main()
