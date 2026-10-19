"""Generate the margin-constrained ceiling constants the manuscript quotes.

    python3 accord/c143_ceiling_sweep/ceiling_constants.py

Three constants reached the manuscript's prose as typed arithmetic with no
artifact behind them. This script computes them from Equation 1 and writes both
a JSON artifact and register keys, so the digit gate has something to scan.

  1. The gate's ceiling threshold at the 300-item grid. C136 registers the
     failure set as flag counts {0} union [235, 300]; the threshold is the
     ceiling at the smallest failing positive flag count. Reported as 0.1074.
  2. The non-degenerate band for that same grid, the flag rates whose ceiling
     falls in [0.20, 0.60]. Reported as 0.1067 to 0.5100.
  3. The band for the 253-pair clinical join, which c143.py carries as a pair
     of hardcoded literals with a `# from make_fig2.py` comment. Recomputing it
     here closes that chain too, and the recomputation is asserted against the
     literals c143.py used so the two can never drift apart silently.

The band edges are a property of the grid, not of a run: no datum enters, and
nothing here is a hypothesis test, so this script carries no pre-registration
and records no outcome letter. It registers under its own top-level prefix
`c143constants` rather than under `c143`, because CLAUDE.md records that a
generator deletes its own prefix wholesale: a later register_c143.py keying on
`c143` would otherwise eat these.
"""
import collections
import json
import math
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
C136 = "results/accord/c136_gate_validation/c136.json"
C143 = "results/accord/c143_ceiling_sweep/c143.json"
OUT = ROOT / "results/accord/c143_ceiling_sweep/ceiling_constants.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "c143constants"

# The band is the closed interval of ceilings inside which a reported
# coefficient of 0.2 reads differently beside its ceiling than it does alone.
# Figure 1 draws it; these are the two edges it is drawn at.
BAND_CEILING_LO, BAND_CEILING_HI = 0.20, 0.60

c136 = json.loads((ROOT / C136).read_text())
c143 = json.loads((ROOT / C143).read_text())


def mcc(tp, fp, fn, tn):
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d > 0 else 0.0


def ceiling(n, N, K):
    """Equation 1: the margin-constrained maximum at N items, K positives, n flags."""
    tp = min(n, K)
    fp = n - tp
    fn = K - tp
    tn = N - n - fn
    return mcc(tp, fp, fn, tn)


def band(N, K):
    """The flag rates whose ceiling lies in the band, on the n >= K branch."""
    inside = [n for n in range(K, N + 1)
              if BAND_CEILING_LO <= ceiling(n, N, K) <= BAND_CEILING_HI]
    lo, hi = min(inside), max(inside)
    # Closed interval: the neighbours outside it must really be outside, or the
    # band is not the contiguous run the figure draws.
    assert lo == K or not (BAND_CEILING_LO <= ceiling(lo - 1, N, K) <= BAND_CEILING_HI)
    assert hi == N or not (BAND_CEILING_LO <= ceiling(hi + 1, N, K) <= BAND_CEILING_HI)
    assert inside == list(range(lo, hi + 1)), "band is not contiguous"
    return lo, hi


entries = collections.OrderedDict()
out = collections.OrderedDict()


def add(key, artifact, value):
    full = f"{PREFIX}.{key}"
    assert full not in entries, f"duplicate register key {full}"
    entries[full] = collections.OrderedDict(
        [("artifact", artifact), ("value", value)])
    node = out
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, collections.OrderedDict())
    node[parts[-1]] = value


# ------------------------------------------------- 1. the grid gate threshold
GN = c136["frames"]["pilot"][0]
GK = c136["frames"]["pilot"][1]
region = c136["gate"]["fail_region_flag_counts"]
smallest_failing = c136["gate"]["smallest_failing_positive_flag_count"]
assert region[0] == 0 and region[1:] == list(range(smallest_failing, GN + 1)), (
    "c136's failure set is not {0} union an upper tail; the threshold below "
    "would not describe it")
threshold = ceiling(smallest_failing, GN, GK)
# The defining property, asserted rather than assumed: the flag counts whose
# ceiling is at or below the threshold are exactly c136's failure set.
sublevel = [n for n in range(GN + 1) if ceiling(n, GN, GK) <= threshold]
assert sublevel == region, (
    f"the sublevel set of {threshold} is {sublevel[:3]}... not c136's "
    f"failure set")
add("grid.N", f"{C136}::frames.pilot.0", GN)
add("grid.K", f"{C136}::frames.pilot.1", GK)
add("grid.gate_ceiling_threshold",
    f"Equation 1 at n = {C136}::gate.smallest_failing_positive_flag_count",
    threshold)
add("grid.gate_threshold_sublevel_set_equals_c136_failure_set",
    f"Equation 1 sublevel set against {C136}::gate.fail_region_flag_counts",
    True)

# ---------------------------------------------------- 2. the grid's own band
glo, ghi = band(GN, GK)
add("grid.band_flag_count_lo", f"Equation 1 on {C136}::frames.pilot", glo)
add("grid.band_flag_count_hi", f"Equation 1 on {C136}::frames.pilot", ghi)
add("grid.band_flag_rate_lo", f"Equation 1 on {C136}::frames.pilot", glo / GN)
add("grid.band_flag_rate_hi", f"Equation 1 on {C136}::frames.pilot", ghi / GN)
add("grid.band_ceiling_at_lo", f"Equation 1 on {C136}::frames.pilot",
    ceiling(glo, GN, GK))
add("grid.band_ceiling_at_hi", f"Equation 1 on {C136}::frames.pilot",
    ceiling(ghi, GN, GK))

# Which labellers the grid band selects, against the registered clinical band.
reg_lo = c143["q2_non_degenerate"]["band_flag_rate_lo"]
reg_hi = c143["q2_non_degenerate"]["band_flag_rate_hi"]
rows = c143["q3_table"]
in_grid = sorted(r["labeller"] for r in rows
                 if glo / GN <= r["flag_rate"] <= ghi / GN)
in_registered = sorted(r["labeller"] for r in rows
                       if reg_lo <= r["flag_rate"] <= reg_hi)
qualifying = sorted(r["labeller"] for r in c143["q2_non_degenerate"]["qualifying"])
add("grid.band_n_labellers_inside", f"{C143}::q3_table under the grid band",
    len(in_grid))
add("registered.band_n_labellers_inside",
    f"{C143}::q3_table under q2_non_degenerate band", len(in_registered))
add("grid.band_selects_the_registered_qualifiers",
    f"{C143}::q2_non_degenerate.qualifying against the grid band",
    all(q in in_grid for q in qualifying))
add("grid.band_qualifying_labellers",
    f"{C143}::q2_non_degenerate.qualifying", qualifying)

# ------------------------------------- 3. the clinical band, chain closed out
CN = c143["n_items"] if "n_items" in c143 else 253
CK = 12
clo, chi = band(CN, CK)
assert abs(clo / CN - reg_lo) < 1e-12 and abs(chi / CN - reg_hi) < 1e-12, (
    f"recomputed clinical band {clo/CN}, {chi/CN} does not match the band "
    f"c143.py carried as literals, {reg_lo}, {reg_hi}")
add("clinical.N", f"recomputed against {C143}::q2_non_degenerate", CN)
add("clinical.K", f"recomputed against {C143}::q2_non_degenerate", CK)
add("clinical.band_flag_rate_lo", "Equation 1, matches "
    f"{C143}::q2_non_degenerate.band_flag_rate_lo", clo / CN)
add("clinical.band_flag_rate_hi", "Equation 1, matches "
    f"{C143}::q2_non_degenerate.band_flag_rate_hi", chi / CN)
add("clinical.band_ceiling_at_lo",
    "Equation 1 at the clinical band's lower edge, the value Figure 1 quotes",
    ceiling(clo, CN, CK))
add("clinical.band_ceiling_at_hi",
    "Equation 1 at the clinical band's upper edge, the value Figure 1 quotes",
    ceiling(chi, CN, CK))
add("clinical.band_matches_c143_literals",
    f"Equation 1 against {C143}::q2_non_degenerate", True)
# ------------------- 4. the analogues for the coefficients the field reports
# The survey records no paper reporting the Matthews correlation, so the ceiling
# is given here for the two statistics those papers do report. Both are fixed by
# the same three integers. Brute force over every attainable 2x2 at the clinical
# margins checks each closed form rather than trusting the algebra.
CLIN_N, CLIN_K = 253, 12
CLIN_FLAGS = c136_clinical_flags = 223


def attainable(N, K, n):
    """Every 2x2 consistent with N items, K positives and n flags."""
    for tp in range(max(0, n - (N - K)), min(n, K) + 1):
        yield tp, n - tp, K - tp, N - n - (K - tp)


prec_closed = CLIN_K / CLIN_FLAGS
f1_closed = 2 * CLIN_K / (CLIN_FLAGS + CLIN_K)
prec_brute = max(tp / (tp + fp) if tp + fp else 0.0
                 for tp, fp, fn, tn in attainable(CLIN_N, CLIN_K, CLIN_FLAGS))
f1_brute = max(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
               for tp, fp, fn, tn in attainable(CLIN_N, CLIN_K, CLIN_FLAGS))
assert abs(prec_closed - prec_brute) < 1e-12, (prec_closed, prec_brute)
assert abs(f1_closed - f1_brute) < 1e-12, (f1_closed, f1_brute)
add("clinical.flags", f"{C136}-sibling clinical join, n flags", CLIN_FLAGS)
add("clinical.precision_ceiling",
    "K / n, checked against every attainable 2x2 at these margins", prec_closed)
add("clinical.f1_ceiling",
    "2K / (n + K), checked against every attainable 2x2 at these margins",
    f1_closed)
add("clinical.precision_ceiling_matches_brute_force",
    "closed form against enumeration of every attainable 2x2", True)
add("clinical.f1_ceiling_matches_brute_force",
    "closed form against enumeration of every attainable 2x2", True)

add("band_ceiling_lo", "the band Figure 1 draws", BAND_CEILING_LO)
add("band_ceiling_hi", "the band Figure 1 draws", BAND_CEILING_HI)

# ------------------------------------------------------------------ the write
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(
    collections.OrderedDict([
        ("generator", "accord/c143_ceiling_sweep/ceiling_constants.py"),
        ("note", "margin-constrained ceiling constants quoted in the "
                 "manuscript; deterministic, no datum enters, no "
                 "pre-registration"),
        ("constants", out)]), indent=1) + "\n")

original_text = FACTS.read_text()
facts = json.loads(original_text, object_pairs_hook=collections.OrderedDict)
mine = [k for k in facts if k.split(".")[0] == PREFIX]
others_before = collections.OrderedDict(
    (k, v) for k, v in facts.items() if k not in set(mine))
for k in mine:
    del facts[k]
facts.update(entries)
assert len(facts) == len(others_before) + len(entries), "key accounting"
for k, v in others_before.items():
    assert facts[k] == v, f"this script would have changed a foreign key: {k}"

fd, tmp = tempfile.mkstemp(dir=str(FACTS.parent), suffix=".tmp")
try:
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(facts, indent=1) + "\n")
    os.replace(tmp, FACTS)
except BaseException:
    if os.path.exists(tmp):
        os.unlink(tmp)
    raise

back = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
if back != facts:
    FACTS.write_text(original_text)
    sys.exit("register changed under this script; restored it and wrote nothing")

print(f"gate ceiling threshold      {threshold:.10f}  -> prose 0.1074")
print(f"grid band flag rates        {glo/GN:.10f} to {ghi/GN:.10f}"
      f"  -> prose 0.1067 to 0.5100")
print(f"grid band flag counts       {glo} to {ghi} of {GN}")
print(f"grid band labellers inside  {len(in_grid)}; registered band "
      f"{len(in_registered)}; selects the registered qualifiers "
      f"{all(q in in_grid for q in qualifying)} {qualifying}")
print(f"clinical band recomputed    {clo/CN:.10f} to {chi/CN:.10f}  "
      f"(matches the c143.py literals)")
print(f"clinical band edge ceilings {ceiling(clo, CN, CK):.10f} to {ceiling(chi, CN, CK):.10f}  -> caption 0.5971, 0.2005")
print(f"precision ceiling           {prec_closed:.10f}  (K/n, brute-force checked)")
print(f"F1 ceiling                  {f1_closed:.10f}  (2K/(n+K), brute-force checked)")
print(f"registered {len(entries)} keys under {PREFIX}; "
      f"{len(facts)} keys total, was {len(others_before) + len(mine)}")
