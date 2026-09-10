#!/usr/bin/env bash
# Export aggregate results into the repo and mirror them to Drive.
#
# Run metrics live under /Users/rickygole/mimic_local_only/, which never leaves
# the machine because it also holds MIMIC-derived text. The metrics themselves
# are scalars -- counts, odds ratios, CIs, losses -- and carry no patient data,
# so they belong in the repo and on Drive where they can be cited.
#
# Every file is scanned for note text before it is copied anywhere. A hit is
# skipped loudly rather than filtered, so a new text-bearing field cannot leak
# by being forgotten.

set -euo pipefail

# DRIVE_ROOT holds the author's Google Drive mount path, which embeds an
# institutional email address, so it is not hardcoded here. Set it in the
# environment before running this script, e.g.:
#   export DRIVE_ROOT="/Users/you/Library/CloudStorage/GoogleDrive-you@example.edu/My Drive/thesis"
: "${DRIVE_ROOT:?DRIVE_ROOT is not set. Export DRIVE_ROOT to the thesis folder inside your Google Drive mount before running this script.}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL="/Users/rickygole/mimic_local_only"
DRIVE="$DRIVE_ROOT/results_2026-08"
DRIVE_CODE="$DRIVE_ROOT/code_2026-08"

# MIMIC-III de-identification surrogates and the section headers that would
# only appear inside a real note.
TEXT_MARKERS='\[\*\*|admission date|chief complaint|discharge medications|nursing_sentence|discharge_sentence'

# The code gate needs a narrower set than the data gate. `nursing_sentence` and
# `discharge_sentence` are column names, and the pipeline scripts that build
# those columns necessarily contain the strings: phase3_mining.py,
# run_multillm_replication.py and run_frontier_compliant.py all do. Scanning
# code with the full data set aborts on the project's own source every run, and
# a gate that always fires is a gate that gets commented out. What must never
# appear in a .py is note CONTENT, so the code gate keeps the surrogate and
# section-header markers and drops the two column names.
CODE_TEXT_MARKERS='\[\*\*|admission date|chief complaint|discharge medications'

# Identifiers matter as much as text -- HADM_ID joined to anything is restricted
# data, not a derived aggregate -- but only when they are a DATA KEY. These
# files legitimately discuss identifiers in prose ("audit.csv keys on
# HADM_ID+drug"), and a gate that fires on prose gets ignored, which is worse
# than no gate. So match the structural forms only: a JSON key, or a CSV header
# at the start of a line.
# The trailing-column alternative matters: a header ending in the identifier,
# such as pair_id,drug,HADM_ID, matched none of the first three forms and passed
# the gate clean.
ID_MARKERS='"(HADM_ID|SUBJECT_ID)"[[:space:]]*:|^(HADM_ID|SUBJECT_ID),|,(HADM_ID|SUBJECT_ID),|,(HADM_ID|SUBJECT_ID)[[:space:]]*$'

# Structural exclusion beats pattern matching: a note sentence carrying no
# surrogate and no section header would pass a content scan. Row-level formats
# therefore never enter the tree at all -- only .json/.txt/.md are eligible for
# ingest, enforced by the find filter below.

cd "$REPO"
skipped=0

for src in "$LOCAL"/accord_*; do
  [ -d "$src" ] || continue
  dst="results/accord/$(basename "$src" | sed 's/^accord_//')"
  mkdir -p "$dst"
  while IFS= read -r f; do
    if grep -qiE "$TEXT_MARKERS" "$f" 2>/dev/null; then
      echo "SKIP (note text): $f"
      skipped=$((skipped + 1))
    else
      cp "$f" "$dst/"
    fi
  done < <(find "$src" -maxdepth 1 -type f \( -name '*.json' -o -name '*.txt' -o -name '*.md' \))
done

python3 - <<'PY'
import json, pathlib
R = pathlib.Path("results/accord")
out = {}
for p in sorted(R.rglob("*.json")):
    if p.name == "ALL_RESULTS.json":
        continue
    try:
        out[str(p.relative_to(R))] = json.loads(p.read_text())
    except Exception as e:
        out[str(p.relative_to(R))] = {"_unreadable": str(e)}
dest = R / "ALL_RESULTS.json"
dest.write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
print(f"consolidated {len(out)} result files -> {dest}")
PY

# Final gate, both directions: no row-level format and no note text or
# identifier may reach the repo's results tree or, through it, Drive.
# Note on scope: the ingest loop above admits only .json/.txt/.md, so no
# row-level format can ENTER the tree from local_only. results/accord/fairness/
# holds hand-placed aggregate .csv tables (chi-square GOF, cohort composition)
# that predate this script and are one-way marginals with no identifiers. Those
# are content-checked here rather than excluded by extension -- blanket-blocking
# .csv in the tree would abort on legitimate aggregates every run.
# -print0 with a null-delimited read, not $(find ...). Unquoted command
# substitution word-splits on spaces, so a filename containing a space became
# two nonexistent paths, grep failed on both, 2>/dev/null swallowed the error and
# the file was never scanned at all.
while IFS= read -r -d '' p; do
  if grep -qiE "$TEXT_MARKERS" "$p" 2>/dev/null; then
    echo "ABORT: note text in $p"
    exit 1
  fi
  if grep -qE "$ID_MARKERS" "$p" 2>/dev/null; then
    echo "ABORT: identifier used as a data key in $p"
    exit 1
  fi
done < <(find results/accord -type f -print0)

mkdir -p "$DRIVE"
cp -R results/accord/. "$DRIVE/"

# Gate the code BEFORE it is copied, scanning the source tree. An earlier
# version scanned $DRIVE_CODE after the rsync and carried a comment claiming
# that meant "what is ON Drive is what passed". That was backwards: the copy had
# already happened and nothing was removed on abort, so a failing file reached
# Drive and stayed there. Aborting here costs a re-run; aborting after the copy
# cost nothing and prevented nothing.
while IFS= read -r -d '' p; do
  case "$p" in
    # These hold the markers as patterns, not as content. stage0/data.py:45 is
    # the section-header regex the note splitter matches on; it has to contain
    # the string in order to find it.
    */sync_results.sh|*/export_results.py|*/stage0/data.py) continue ;;
  esac
  if grep -qiE "$CODE_TEXT_MARKERS" "$p" 2>/dev/null; then
    echo "ABORT: note text in code file $p (nothing was copied to $DRIVE_CODE)"
    exit 1
  fi
done < <(find accord scripts src tests -type f \
           \( -name '*.py' -o -name '*.sh' -o -name '*.tex' \) -print0)

# Code goes to Drive too, so the evidence and the thing that produced it stay
# together. Code is not just safe to share -- PhysioNet License 1.5.0 Clause 8
# obliges contributing analysis code to an open repository. What is excluded is
# the local_only symlink (MIMIC-derived), caches, and git internals.
mkdir -p "$DRIVE_CODE"
rsync -a --delete \
  --exclude 'local_only' --exclude 'local_only/**' \
  --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '.git' --exclude '.DS_Store' \
  --exclude '*.pt' --exclude '*.ckpt' --exclude '*.bin' \
  accord/ scripts/ src/ tests/ "$DRIVE_CODE/" 2>/dev/null || {
    for d in accord scripts src tests; do
      [ -d "$d" ] && cp -R "$d" "$DRIVE_CODE/" 2>/dev/null
    done
  }
find "$DRIVE_CODE" \( -name '__pycache__' -o -name '*.pyc' \) -prune -exec rm -rf {} + 2>/dev/null

echo "repo:    $(find results/accord -type f | wc -l | tr -d ' ') result files"
echo "drive:   $(find "$DRIVE" -type f | wc -l | tr -d ' ') result files -> $DRIVE"
echo "code:    $(find "$DRIVE_CODE" -type f | wc -l | tr -d ' ') files -> $DRIVE_CODE"
echo "skipped for note text: $skipped"
