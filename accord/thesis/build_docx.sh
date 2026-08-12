#!/bin/bash
# Build the editable Word version.
#
# Two steps, because neither alone is sufficient:
#   pandoc renders the content but discards \newpage, so every chapter runs on
#   from the previous one; finish_docx.py inserts real Word page breaks after.
#
# The reference document carries the Morgan State styles. Pandoc writes body
# text into BodyText, FirstParagraph and Compact rather than Normal, so all of
# those are defined there, not just Normal.
set -e
cd "$(dirname "$0")"
REF="${1:-msu_reference.docx}"
pandoc main_docx.tex --from=latex --to=docx \
  --reference-doc="$REF" \
  --resource-path=.:figures \
  --toc --toc-depth=3 \
  -o /tmp/_thesis_raw.docx
python3 ../stats/finish_docx.py /tmp/_thesis_raw.docx Gole_MS_Thesis_EDITABLE.docx
echo "built Gole_MS_Thesis_EDITABLE.docx"
