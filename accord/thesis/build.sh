#!/usr/bin/env bash
# Build the thesis. Four passes: pdflatex, bibtex, pdflatex x2 -- the last two
# resolve cross-references and the bibliography respectively.
set -euo pipefail
export PATH="/Library/TeX/texbin:$PATH"

# PACKAGES. This document needs apacite (APA 7 references, FORMAT.md section 8),
# plus tocloft, newtx, chngcntr, needspace, placeins and xurl. None ships with
# TeX Live basic. They do NOT need root: the TeX tree is owned by root, but
# tlmgr installs into the user tree instead, which kpsewhich finds automatically.
#
#   tlmgr --usermode init-usertree   # once, if ~/Library/texmf does not exist
#   tlmgr --usermode install apacite tocloft newtx chngcntr needspace placeins xurl
#
# A missing package here fails loudly. The failure mode worth knowing about is the
# quiet one: main_supplement.tex was missing graphicx for weeks and built a broken
# PDF that nothing checked, because this script did not build it.
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main_msu.tex > /tmp/thesis_build.log 2>&1
bibtex main_msu >> /tmp/thesis_build.log 2>&1 || true
pdflatex -interaction=nonstopmode main_msu.tex >> /tmp/thesis_build.log 2>&1
pdflatex -interaction=nonstopmode main_msu.tex > /tmp/thesis_final_pass.log 2>&1
cat /tmp/thesis_final_pass.log >> /tmp/thesis_build.log
echo "$(grep -o 'Output written.*' /tmp/thesis_final_pass.log | tail -1)"
echo "undefined refs:      $(grep -c 'Reference .* undefined' /tmp/thesis_final_pass.log || true)"
echo "undefined citations: $(grep -c 'Citation .* undefined' /tmp/thesis_final_pass.log || true)"
# The glob here was ch*.tex, which matches nothing: the chapters are msu_ch*.tex.
# So this line reported 0 open TODOs by counting no files, and the surrounding
# `set -e` did not fire because grep's failure is swallowed by the subshell. A
# gate that cannot fail is not a gate.
echo "open TODOs in text:  $(grep -ho '\[TODO' msu_ch*.tex msu_app*.tex frontmatter/*.tex main_msu.tex 2>/dev/null | wc -l | tr -d ' ')"

# The supplement carries Appendices A-full, B-provenance, C-trace, D-survey and
# E-anomaly, and it was NOT built here. Its PDF therefore went stale every time a
# correction landed, silently, because the verification harness reads main_msu.pdf
# only. Both documents are built from one command now so they cannot diverge.
pdflatex -interaction=nonstopmode main_supplement.tex > /tmp/supp_build.log 2>&1
bibtex main_supplement >> /tmp/supp_build.log 2>&1 || true
pdflatex -interaction=nonstopmode main_supplement.tex >> /tmp/supp_build.log 2>&1
pdflatex -interaction=nonstopmode main_supplement.tex > /tmp/supp_final.log 2>&1
echo "supplement:          $(grep -o 'Output written.*' /tmp/supp_final.log | tail -1)"
echo "supplement refs:     $(grep -c 'Reference .* undefined' /tmp/supp_final.log || true)"
