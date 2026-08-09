#!/usr/bin/env bash
# Build the thesis. Four passes: pdflatex, bibtex, pdflatex x2 -- the last two
# resolve cross-references and the bibliography respectively.
set -euo pipefail
export PATH="/Library/TeX/texbin:$PATH"
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex > /tmp/thesis_build.log 2>&1
bibtex main >> /tmp/thesis_build.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >> /tmp/thesis_build.log 2>&1
pdflatex -interaction=nonstopmode main.tex >> /tmp/thesis_build.log 2>&1
echo "$(grep -o 'Output written.*' /tmp/thesis_build.log | tail -1)"
echo "undefined refs:      $(grep -c 'Reference .* undefined' /tmp/thesis_build.log || true)"
echo "undefined citations: $(grep -c 'Citation .* undefined' /tmp/thesis_build.log || true)"
echo "open TODOs in text:  $(grep -ho '\[TODO' ch*.tex main.tex | wc -l | tr -d ' ')"
