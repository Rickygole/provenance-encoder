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

REPO="/Users/rickygole/provenance-encoder"
LOCAL="/Users/rickygole/mimic_local_only"
DRIVE="/Users/rickygole/Library/CloudStorage/GoogleDrive-ricky.gole@morgan.edu/My Drive/thesis/results_2026-08"

# MIMIC-III de-identification surrogates and the section headers that would
# only ever appear inside a real note.
TEXT_MARKERS='\[\*\*|admission date|chief complaint|discharge medications|nursing_sentence|discharge_sentence'

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

# Final gate: nothing in the repo's results tree may contain note text.
for p in $(find results/accord -type f); do
  if grep -qiE "$TEXT_MARKERS" "$p" 2>/dev/null; then
    echo "ABORT: note text in $p"
    exit 1
  fi
done

mkdir -p "$DRIVE"
cp -R results/accord/. "$DRIVE/"

echo "repo:  $(find results/accord -type f | wc -l | tr -d ' ') files"
echo "drive: $(find "$DRIVE" -type f | wc -l | tr -d ' ') files -> $DRIVE"
echo "skipped for note text: $skipped"
