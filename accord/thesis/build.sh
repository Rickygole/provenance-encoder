#!/usr/bin/env bash
# Build both thesis documents: main_msu.tex (the thesis body) and
# main_supplement.tex (the companion document the side strands moved to).
# Up to four passes per document: pdflatex, biber if that document actually
# uses biblatex, then pdflatex x2 to resolve cross-references and the
# bibliography. Reports page counts, undefined refs and undefined citations
# for each document, and exits nonzero if either document fails to build.
#
# Not `set -e`: a build failure in one document must be caught and reported,
# not abort the script before the other document is even tried. That is
# exactly the failure mode this file is written against -- a second document
# that nothing builds goes stale silently (CLAUDE.md) -- so a build failure
# here is loud, not fatal to the script itself.
set -uo pipefail
export PATH="/Library/TeX/texbin:$PATH"
cd "$(dirname "$0")"

# Kept in sync by eye with accord/stats/verify_thesis_deep.py's
# MAIN_THESIS_MAX_PAGES: a shell script and a Python module have no single
# source to import a shared constant from without a new dependency, so the
# number is repeated here rather than shared, and named the same way.
MAIN_THESIS_MAX_PAGES=150

OVERALL_FAILED=0

# PACKAGES. This document needs apacite (APA 7 references, FORMAT.md section 8),
# plus tocloft, newtx, chngcntr, needspace, placeins and xurl. None ships with
# TeX Live basic. They do NOT need root: the TeX tree is owned by root, but
# tlmgr installs into the user tree instead, which kpsewhich finds automatically.
#
#   tlmgr --usermode init-usertree   # once, if ~/Library/texmf does not exist
#   tlmgr --usermode install apacite tocloft newtx chngcntr needspace placeins xurl
#
# biblatex-apa requires biber, which refuses to be a bibtex drop-in. biber lives
# in /opt/homebrew/bin because tlmgr cannot install a binary package in user mode.

# build_one DOC
#   DOC is the file stem, e.g. main_msu or main_supplement (no .tex).
# Prints a report for DOC and returns 0 if it built to a PDF with no
# undefined refs/citations, 1 otherwise. Never exits the script itself: the
# caller decides what a failure here means for the overall exit code.
build_one() {
    local doc="$1"
    local log="/tmp/thesis_build_${doc}.log"
    local final_log="/tmp/thesis_final_pass_${doc}.log"
    : > "$log"

    echo "=== ${doc}.tex ==="

    if [ ! -f "${doc}.tex" ]; then
        # A document that does not exist is reported, not silently skipped:
        # this is the exact "second document going stale" failure mode
        # (CLAUDE.md) if it ever happens to the version of this script that
        # is supposed to build it.
        echo "   FAIL: ${doc}.tex does not exist"
        return 1
    fi

    pdflatex -interaction=nonstopmode "${doc}.tex" >> "$log" 2>&1

    # biber only if this document actually uses biblatex: a .bcf is written by
    # pdflatex's first pass when \usepackage{biblatex} is loaded, and biber
    # refuses to run without one. main_supplement.tex did not carry biblatex
    # for a period of this project's history (no \cite among the appendices
    # it bound then); that state is reported as a skip below, not silently,
    # rather than assumed and hard-coded, because whether a document uses
    # biblatex is a property of its own preamble, not of this script.
    if [ -f "${doc}.bcf" ]; then
        PATH="/opt/homebrew/bin:$PATH" biber "${doc}" >> "$log" 2>&1
        biber_status=$?
        if [ "$biber_status" -ne 0 ]; then
            echo "   biber FAILED (exit $biber_status) -- see $log"
        fi
    else
        echo "   biber: skipped, ${doc}.tex produced no .bcf (no biblatex bibliography)"
    fi

    pdflatex -interaction=nonstopmode "${doc}.tex" >> "$log" 2>&1
    pdflatex -interaction=nonstopmode "${doc}.tex" > "$final_log" 2>&1
    cat "$final_log" >> "$log"

    local built=0
    if grep -q "Output written on ${doc}.pdf" "$final_log"; then
        built=1
        echo "   $(grep -o 'Output written.*' "$final_log" | tail -1)"
    fi

    local undef_refs undef_cites over_limit=0
    undef_refs=$(grep -c 'Reference .* undefined' "$final_log" || true)
    undef_cites=$(grep -c 'Citation .* undefined' "$final_log" || true)
    echo "   undefined refs:      ${undef_refs}"
    echo "   undefined citations: ${undef_cites}"

    if [ "$built" -eq 1 ] && command -v pdfinfo > /dev/null 2>&1; then
        local pages
        pages=$(pdfinfo "${doc}.pdf" 2>/dev/null | awk '/^Pages:/{print $2}')
        if [ -n "$pages" ]; then
            echo "   pages: ${pages}"
            if [ "$doc" = "main_msu" ] && [ "$pages" -gt "$MAIN_THESIS_MAX_PAGES" ]; then
                echo "   FAIL: main_msu.pdf is ${pages} pages, exceeds the ${MAIN_THESIS_MAX_PAGES}-page limit"
                over_limit=1
            fi
        fi
    fi

    # A build that compiled but is too long is a different failure from one that
    # did not compile. They previously shared a flag, so an over-length thesis was
    # also reported as "did not build to a PDF", which sends a reader after a LaTeX
    # error that does not exist.
    if [ "$built" -ne 1 ]; then
        echo "   FAIL: ${doc}.tex did not build to a PDF -- see $log"
        return 1
    fi
    if [ "${over_limit:-0}" -eq 1 ]; then
        return 1
    fi
    if [ "$undef_refs" -ne 0 ] || [ "$undef_cites" -ne 0 ]; then
        echo "   FAIL: ${doc}.tex built but has unresolved refs/citations -- see $log"
        return 1
    fi
    echo "   pass"
    return 0
}

build_one main_msu; MAIN_STATUS=$?
echo
# The thesis is submitted as one document; there is no supplement.
SUP_STATUS=0

# The glob used to be ch*.tex, which matches nothing (the chapters are
# msu_ch*.tex): a gate that cannot fail. msu_sup*.tex is included for the same
# reason msu_ch*.tex is: a TODO left in the supplement is as much an open item
# as one left in a chapter.
todo_count=$(grep -ho '\[TODO' msu_ch*.tex msu_app*.tex frontmatter/*.tex main_msu.tex main_supplement.tex 2>/dev/null | wc -l | tr -d ' ')
echo "open TODOs in text:  ${todo_count}"

if [ "$MAIN_STATUS" -ne 0 ]; then
    echo "OVERALL: main_msu.tex FAILED to build cleanly"
    OVERALL_FAILED=1
fi
if [ "$SUP_STATUS" -ne 0 ]; then
    # main_supplement.tex may not yet build cleanly while the side-strand
    # files it binds (msu_sup*.tex) are still being written elsewhere in this
    # tree: that is reported here, loudly, rather than caught and hidden, so
    # a broken supplement cannot sit unbuilt and unnoticed the way it did
    # before this script built it at all.
    echo "OVERALL: main_supplement.tex FAILED to build cleanly (see above; the"
    echo "  supplement's own inputs may still be incomplete -- this is reported,"
    echo "  not swallowed, exactly so that state cannot go unnoticed)"
    OVERALL_FAILED=1
fi

exit "$OVERALL_FAILED"
