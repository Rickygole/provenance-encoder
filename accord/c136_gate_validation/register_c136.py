"""Register C136 (prefix c136; deletes and re-adds) from its artifacts.
See C136_PREREGISTRATION.md.

    python3 accord/c136_gate_validation/register_c136.py

C136's superseded first attempt is not a separate check, so its two surviving
values are held under `withdrawn.superseded_c136_first_attempt.c136.*`, the
scoped form C111, C114 and C120 already use. This script deletes that scoped
namespace and the `c136` prefix, and nothing else: other generators run
concurrently in this tree, and CLAUDE.md records that a generator which deletes
more than its own prefix eats whatever else is parked there. Every other key is
snapshotted before the write and compared value by value after it; the write is
rolled back rather than left half applied if anything else moved.
"""
import collections
import json
import os
import pathlib
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/c136_gate_validation/c136.json"
SRC_SUP = ("results/accord/c136_gate_validation/superseded_first_attempt/"
           "superseded.json")
SRC_SUP_SYN = ("results/accord/c136_gate_validation/superseded_first_attempt/"
               "superseded_synthetic.json")
PREREG = "accord/c136_gate_validation/C136_PREREGISTRATION.md"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "c136"
SUPERSEDED = f"withdrawn.superseded_c136_first_attempt.{PREFIX}"
# Pre-registered decision floors. They live in the rule text, not in a run, so
# they are sourced to the rule file and the outcome letter computed from them is
# cross-checked against the letter the rule file recorded.
PASS_FLOOR_SENSITIVITY = 0.90
PASS_FLOOR_PPV = 0.90
# c135 must be untouched by this script; its count is asserted, not assumed.
# The c135 prefix grows when its own registrar is re-run, so a hardcoded count
# becomes a tripwire that blocks this script forever. Snapshot it instead and
# require only that this write leaves it untouched.
C135_KEYS_EXPECTED = None

d = json.loads((ROOT / SRC).read_text())
sup = json.loads((ROOT / SRC_SUP).read_text())
sup_syn = json.loads((ROOT / SRC_SUP_SYN).read_text())

entries = collections.OrderedDict()


def add(key, src, artifact_path, value, prefix=PREFIX):
    full = f"{prefix}.{key}"
    assert full not in entries, f"duplicate register key {full}"
    entries[full] = collections.OrderedDict(
        [("artifact", f"{src}::{artifact_path}"), ("value", value)]
    )


def thr_tag(threshold):
    return "thr" + f"{threshold:.2f}".replace(".", "p")


def arm_block(arm, student):
    """The artifact node and the register segment for one arm and student."""
    return f"{arm}_{student}", f"{arm}.{student}"


# ------------------------------------------------------------------ the rule
add("check", SRC, "check", d["check"])
add("prereg", SRC, "prereg", d["prereg"])
add("seed", SRC, "seed", d["seed"])
add("data_source", SRC, "data_source", d["data_source"])
add("primary_student", SRC, "primary_student", d["primary_student"])
add("rule.gate_alpha", SRC, "gate.alpha", d["gate"]["alpha"])
add("rule.validation_floor_sensitivity", PREREG, "Reading rule, Outcome A",
    PASS_FLOOR_SENSITIVITY)
add("rule.validation_floor_ppv", PREREG, "Reading rule, Outcome A",
    PASS_FLOOR_PPV)

# Design parameters.
for frame in ("pilot", "pool", "test"):
    n, k = d["frames"][frame]
    add(f"design.{frame}_n", SRC, f"frames.{frame}.0", n)
    add(f"design.{frame}_positives", SRC, f"frames.{frame}.1", k)
add("design.pool_test_disjoint", SRC, "frames.pool_test_disjoint",
    d["frames"]["pool_test_disjoint"])
for student, ref in d["clean_label_reference_auroc"].items():
    add(f"design.clean_label_reference_auroc.{student}", SRC,
        f"clean_label_reference_auroc.{student}", ref)
for field in ("floor", "passed", "primary_reference_auroc"):
    add(f"headroom.{field}", SRC, f"headroom_precondition.{field}",
        d["headroom_precondition"][field])

# ------------------------------------------------------- the gate, as a rule
gate = d["gate"]
for field in ("N", "K", "fail_region_flag_counts",
              "fail_region_is_zero_plus_upper_tail",
              "smallest_failing_positive_flag_count",
              "smallest_failing_positive_flag_rate"):
    add(f"gate.{field}", SRC, f"gate.{field}", gate[field])
region = gate["fail_region_flag_counts"]
add("gate.n_fail_region_flag_counts", SRC,
    "gate.fail_region_flag_counts (length)", len(region))
add("gate.fail_region_includes_zero_flags", SRC,
    "gate.fail_region_flag_counts (0 in region)", 0 in region)

# ------------------------------------------------- arms, students, thresholds
ARMS = ("synthetic", "real", "pooled")
STUDENTS = ("minilm", "tfidf")
GATE_FIELDS = ("cell_gatefail_studentfailed", "cell_gatefail_studentuseful",
               "cell_gatepass_studentfailed", "cell_gatepass_studentuseful",
               "n_gate_fail", "n_gate_pass", "n_labellers", "n_student_failed",
               "n_student_useful", "sensitivity", "specificity", "ppv", "npv",
               "youden_j", "frac_gatepass_still_failing")
SCREEN_FIELDS = GATE_FIELDS

thresholds = sorted(
    float(k.rsplit("_", 1)[1])
    for k in d["synthetic_minilm"]["screening"]
)
add("rule.student_failure_threshold_primary", SRC,
    "synthetic_minilm.screening.auroc_lt_0.60.gate.threshold",
    d["synthetic_minilm"]["screening"]["auroc_lt_0.60"]["gate"]["threshold"])
add("rule.student_failure_threshold_sensitivity", SRC,
    "synthetic_minilm.screening.auroc_lt_0.70.gate.threshold",
    d["synthetic_minilm"]["screening"]["auroc_lt_0.70"]["gate"]["threshold"])

for arm in ARMS:
    for student in STUDENTS:
        node, seg = arm_block(arm, student)
        block = d[node]
        add(f"{seg}.n_labellers", SRC, f"{node}.n_labellers",
            block["n_labellers"])
        for stat in ("spearman_ceilingmcc_vs_auroc", "spearman_floorp_vs_auroc"):
            if stat in block:
                for f in ("rho", "p"):
                    add(f"{seg}.{stat}.{f}", SRC, f"{node}.{stat}.{f}",
                        block[stat][f])
        for threshold in thresholds:
            anode = f"auroc_lt_{threshold:.2f}"
            s = block["screening"][anode]
            base = f"{seg}.{thr_tag(threshold)}"
            for f in GATE_FIELDS:
                add(f"{base}.{f}", SRC, f"{node}.screening.{anode}.gate.{f}",
                    s["gate"][f])
            for f in ("gate_youden_j", "gate_adds_nothing_over_raw_ratio",
                      "ratio_beats_gate"):
                add(f"{base}.{f}", SRC, f"{node}.screening.{anode}.{f}", s[f])
            # The redundancy finding: the gate against a threshold on the raw
            # flag rate ratio. The verdict comparison is recorded for both
            # students, since "the gate adds nothing over the raw ratio" is read
            # off it in every arm. The ratio predictor's own screening cells are
            # recorded for the primary student only: the tfidf student failed
            # everywhere, so its cells carry no outcome variance to screen and
            # its threshold sweep is null in the artifact.
            eq = s["ratio_at_gate_equivalent_threshold"]
            if eq is not None:
                for f in ("ratio_threshold", "identical_to_gate_verdicts",
                          "disagreements"):
                    add(f"{base}.ratio_equivalent.{f}", SRC,
                        f"{node}.screening.{anode}."
                        f"ratio_at_gate_equivalent_threshold.{f}", eq[f])
            if eq is not None and student == d["primary_student"]:
                for f in SCREEN_FIELDS:
                    add(f"{base}.ratio_equivalent.{f}", SRC,
                        f"{node}.screening.{anode}."
                        f"ratio_at_gate_equivalent_threshold.screening.{f}",
                        eq["screening"][f])
            best = s["ratio_best_over_all_thresholds"]
            if best is not None:
                add(f"{base}.ratio_best.ratio_threshold", SRC,
                    f"{node}.screening.{anode}."
                    f"ratio_best_over_all_thresholds.ratio_threshold",
                    best["ratio_threshold"])
                for f in SCREEN_FIELDS:
                    add(f"{base}.ratio_best.{f}", SRC,
                        f"{node}.screening.{anode}."
                        f"ratio_best_over_all_thresholds.{f}", best[f])

# ---------------------------------------------------------- the two arms' rows
syn_rows = [r for r in d["labellers"] if r["family"] == "synthetic"]
real_rows = [r for r in d["labellers"] if r["family"] == "real_ollama"]
add("n_real_labellers", SRC, "n_real_labellers", d["n_real_labellers"])
add("synthetic.n_labellers_in_grid", SRC,
    "labellers (family == synthetic, count)", len(syn_rows))
add("synthetic.pilot_flag_count_min", SRC,
    "labellers (family == synthetic, min flag_count)",
    min(r["flag_count"] for r in syn_rows))
add("synthetic.pilot_flag_count_max", SRC,
    "labellers (family == synthetic, max flag_count)",
    max(r["flag_count"] for r in syn_rows))
add("real.n_models_flagging_zero_pilot_items", SRC,
    "labellers (family == real_ollama, flag_count == 0, count)",
    sum(1 for r in real_rows if r["flag_count"] == 0))

REAL_FIELDS = ("flag_count", "flag_rate", "pilot_true_positives_flagged",
               "pilot_unparsed", "pool_flag_count", "pool_unparsed", "floor_p",
               "ceiling_mcc", "ratio_flagrate_over_baserate", "gate_fail",
               "student_auroc_minilm", "student_auroc_tfidf",
               "student_degenerate_minilm", "student_degenerate_tfidf")
for i, r in enumerate(real_rows):
    m = r["labeller"].replace(":", "_").replace(".", "p")
    for f in REAL_FIELDS:
        add(f"real.{m}.{f}", SRC, f"labellers.{i}.{f}", r[f])

# --------------------------------------------------------- the outcome letter
prim = d["synthetic_minilm"]["screening"]["auroc_lt_0.60"]
pg = prim["gate"]
sens, ppv = pg["sensitivity"], pg["ppv"]
clears = ((sens or 0.0) >= PASS_FLOOR_SENSITIVITY
          and (ppv or 0.0) >= PASS_FLOOR_PPV)
if not d["headroom_precondition"]["passed"] or not d["frames"]["pool_test_disjoint"]:
    outcome = "D"
elif clears and not prim["gate_adds_nothing_over_raw_ratio"]:
    outcome = "A"
elif clears:
    outcome = "B"
else:
    outcome = "C"
recorded = re.search(r"^## Outcome ([A-D])\b", (ROOT / PREREG).read_text(),
                     re.M)
assert recorded, f"no outcome letter recorded in {PREREG}"
assert recorded.group(1) == outcome, (
    f"outcome letter computed from {SRC} is {outcome} but {PREREG} records "
    f"{recorded.group(1)}")
add("outcome", SRC,
    "synthetic_minilm.screening.auroc_lt_0.60 (reading rule of "
    "C136_PREREGISTRATION.md)", outcome)

# ------------------------------------ the superseded first attempt, on record
sup_prim = sup["primary_synthetic"]["screening"]["auroc_lt_0.60"]["gate"]
add("clean_label_reference_auroc", SRC_SUP_SYN,
    "clean_label_oracle_student_auroc",
    sup_syn["clean_label_oracle_student_auroc"], prefix=SUPERSEDED)
add("reference_margin_above_primary_failure_line", SRC_SUP_SYN,
    "clean_label_oracle_student_auroc minus the 0.60 failure line",
    sup_syn["clean_label_oracle_student_auroc"] - sup_prim["threshold"],
    prefix=SUPERSEDED)
for f in ("n_student_failed", "n_student_useful", "n_labellers", "sensitivity",
          "specificity", "ppv", "npv"):
    add(f"synthetic.{f}", SRC_SUP,
        f"primary_synthetic.screening.auroc_lt_0.60.gate.{f}", sup_prim[f],
        prefix=SUPERSEDED)
add("abandoned_because", SRC_SUP,
    "primary_synthetic.screening.auroc_lt_0.60.gate",
    "no student headroom: the pre-registered TF-IDF student's clean-label "
    "reference AUROC sat just above its own 0.60 failure line, so every "
    "labeller in the grid produced a failing student, the outcome variable had "
    "no variance, and specificity was undefined", prefix=SUPERSEDED)

# ------------------------------------------------------------------ the write
# Re-read immediately before writing: another generator may have written since
# this script started, and holding a stale copy across the work above is how a
# concurrent prefix gets eaten.
original_text = FACTS.read_text()
facts = json.loads(original_text, object_pairs_hook=collections.OrderedDict)
mine = [k for k in facts
        if k.split(".")[0] == PREFIX or k.startswith(SUPERSEDED + ".")]
others_before = collections.OrderedDict(
    (k, v) for k, v in facts.items() if k not in set(mine))
for k in mine:
    del facts[k]
facts.update(entries)

c135_keys = sum(1 for k in facts if k.split(".")[0] == "c135")
assert C135_KEYS_EXPECTED is None or c135_keys == C135_KEYS_EXPECTED, (
    f"c135 holds {c135_keys} keys, expected {C135_KEYS_EXPECTED}; refusing to "
    "write")
assert len(facts) == len(others_before) + len(entries), (
    f"key accounting: {len(facts)} != {len(others_before)} others + "
    f"{len(entries)} c136")
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
    sys.exit("register changed under this script; restored it and wrote "
             "nothing")

print(f"registered {PREFIX}: outcome={outcome}; "
      f"{sum(1 for k in facts if k.startswith(PREFIX + '.'))} keys under "
      f"prefix {PREFIX}, "
      f"{sum(1 for k in facts if k.startswith(SUPERSEDED + '.'))} under "
      f"{SUPERSEDED}; {len(facts)} keys total, was {len(others_before) + len(mine)}")
