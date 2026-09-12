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
# biblatex-apa requires biber, which refuses to be a bibtex drop-in. biber reads
# the .bcf the first pdflatex pass writes. It lives in /opt/homebrew/bin because
# tlmgr cannot install a binary package in user mode.
PATH="/opt/homebrew/bin:$PATH" biber main_msu >> /tmp/thesis_build.log 2>&1 || true
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

# One document. The companion supplement is no longer built: the survey coding
# table and the selection-channel anomaly are bound in the thesis as Appendices
# B and C, and the register of reported quantities and the claim-and-evidence
# trace are released with the code rather than printed. main_supplement.tex is
# kept in the tree unbuilt rather than deleted, so nothing is lost if a separate
# supplement is ever wanted again. The caution that produced this block still
# holds and now has nowhere to hide: a second document that nothing builds goes
# stale silently, which is why there is only one.
