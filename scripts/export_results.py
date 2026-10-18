"""Build the deposit tree the paper promises, as an allow-list.

    python3 -I scripts/export_results.py OUTDIR [--strip-withdrawn]

The paper's Data availability statement promises four categories: the analysis
code, the pre-registrations, the coded survey table, and the aggregate result
artifacts every number is read from. It promises nothing else, and the deposit
must contain nothing else.

This is an allow-list, not a filter. A filter has to anticipate every file that
must not ship; an allow-list only has to name the files that must. The project
working agreement records a gate that reported zero because it scanned nothing,
and a deny-list here would be the same shape of mistake: the withdrawn hosted
outputs, the per-event clinical table and the credentialed-local annotation
sheets are excluded because they are not named below, not because a pattern
happened to catch them.

Two refusals are hard. The script exits nonzero and writes nothing if an
allow-listed file is missing, or if any file it is about to copy carries an
identifier column or a free-text field long enough to hold note content. Both
are checks on the aggregate artifacts only; no note-derived clinical file is
opened, named or copied.

The register retains 823 withdrawn.hosted.* keys on purpose, so that the
correction record can say what each withdrawal cost. Depositing them means
depositing the numbers from the withdrawn runs. That is a judgement for the
author, so this script never decides it silently: it reports the count, and
strips the prefix only when asked with --strip-withdrawn.
"""
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Aggregate result artifacts every reported number is read from.
ARTIFACTS = [
    "results/accord/VERIFIED_FACTS.json",
    "results/accord/c120_exact_join/c120.json",
    "results/accord/c136_gate_validation/c136.json",
    "results/accord/c139_rule_prior/c139.json",
    "results/accord/c140_rival_adequacy/c140.json",
    "results/accord/c141_crossfit_repair/c141.json",
    "results/accord/c141_crossfit_repair/c142_delivery.json",
    "results/accord/c143_ceiling_sweep/c143.json",
    "results/accord/c143_ceiling_sweep/ceiling_constants.json",
    "results/accord/labeller_interval_closure/intervals_17.json",
]
# The pre-registrations, rules and recorded outcomes together.
PREREGS = [
    "accord/c136_gate_validation/C136_PREREGISTRATION.md",
    "accord/c143_ceiling_sweep/C143_PREREGISTRATION.md",
]
# The coded survey table and the documents that define its coding and frame.
SURVEY = [
    "papers/jbi_weak_supervision/premise_survey/survey.json",
    "papers/jbi_weak_supervision/premise_survey/survey_method.md",
    "papers/jbi_weak_supervision/premise_survey/survey_findings.md",
    "papers/jbi_weak_supervision/premise_survey/survey_refs.bib",
]
# The analysis code that produces the artifacts above.
CODE = [
    "accord/c136_gate_validation/c136_run.py",
    "accord/c136_gate_validation/c136_outcome.py",
    "accord/c136_gate_validation/register_c136.py",
    "accord/c141_crossfit_repair/register_c14x.py",
    "accord/c143_ceiling_sweep/c143.py",
    "accord/c143_ceiling_sweep/ceiling_constants.py",
    "papers/jbi_weak_supervision/make_figures.py",
    "papers/jbi_weak_supervision/make_fig2.py",
]
CATEGORIES = [("artifacts", ARTIFACTS), ("preregistrations", PREREGS),
              ("survey", SURVEY), ("code", CODE)]

# Any of these appearing as a key, a column name or a field is a hard stop.
IDENTIFIERS = ("hadm_id", "subject_id", "icustay_id", "row_id")
# A string field this long in an aggregate artifact should not exist, and is the
# shape a leaked note sentence would take.
MAX_STRING = 600


def scan(path, obj, trail=""):
    """Refuse an identifier key or an overlong string anywhere in the file."""
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).strip().lower() in IDENTIFIERS:
                bad.append(f"{path}: identifier key at {trail}.{k}")
            bad += scan(path, v, f"{trail}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += scan(path, v, f"{trail}[{i}]")
    elif isinstance(obj, str):
        if len(obj) > MAX_STRING:
            bad.append(f"{path}: string of {len(obj)} chars at {trail}")
    return bad


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip().splitlines()[2].strip())
    out = pathlib.Path(sys.argv[1]).resolve()
    strip = "--strip-withdrawn" in sys.argv[2:]

    wanted = [p for _, group in CATEGORIES for p in group]
    missing = [p for p in wanted if not (ROOT / p).is_file()]
    if missing:
        sys.exit("refusing to export, these allow-listed files are missing:\n  "
                 + "\n  ".join(missing))

    problems = []
    for p in wanted:
        if p.endswith(".json"):
            problems += scan(p, json.loads((ROOT / p).read_text()))
    if problems:
        sys.exit("refusing to export, disclosure check failed:\n  "
                 + "\n  ".join(problems))

    if out.exists():
        shutil.rmtree(out)
    for p in wanted:
        dest = out / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / p, dest)

    facts = out / "results/accord/VERIFIED_FACTS.json"
    reg = json.loads(facts.read_text())
    withdrawn = [k for k in reg if k.split(".")[0] == "withdrawn"]
    if strip:
        for k in withdrawn:
            del reg[k]
        facts.write_text(json.dumps(reg, indent=1) + "\n")

    for name, group in CATEGORIES:
        print(f"{name:18} {len(group)} files")
    print(f"{'total':18} {len(wanted)} files -> {out}")
    print(f"withdrawn.* keys in the deposited register: "
          f"{0 if strip else len(withdrawn)} of {len(withdrawn)}"
          f"{' (stripped)' if strip else ' (retained; pass --strip-withdrawn to remove)'}")
    print("not deposited, because not allow-listed: the withdrawn hosted "
          "outputs, the per-event clinical table, the credentialed-local "
          "annotation sheets, and the repository history")


if __name__ == "__main__":
    main()
