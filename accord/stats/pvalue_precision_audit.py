"""Every p value printed in the thesis, with its precision and its four-decimal form.

Three reviews have noted that the chapters print p values at two precisions: most
at four decimals, five at five decimals (0.01335, 0.00005, 0.10035, 0.17875 and
0.96465), which are the permutation tests whose null sizes were 20,000 draws and
whose p values therefore land on odd multiples of 1/20000 or 1/40000. This script
does not edit anything. It enumerates every p value in the chapters and
appendices, records its printed precision, gives the value rounded to four
decimals, and flags any value whose four-decimal form reads differently against a
conventional threshold (0.05, 0.01, 0.001) or whose four-decimal form loses the
fact that the value is at its null's resolution floor.

Two sources of p values are collected:

  1 prose and table notes, by the pattern `p` followed by a relation and a
    number, including the LaTeX forms `$p = 0.1234$` and `p = 0.1234`;
  2 table columns, by parsing every tabular environment, finding the header row
    and any column whose header carries a standalone `p` (for example
    "Mann-Whitney $p$", "Floor $p$", "$p$"), then reading that column out of
    every body row;
  3 the Appendix C claim-and-evidence trace, whose Value column reprints every
    quantity the chapters print. A trace row is a p value when its register key
    names one (p_permutation, p_two_sided, p_one_sided, p_value, and the like).
    These rows are mirrors of chapter values, so they carry the same precision
    defect and would need the same edit.

One p value in Chapter 4 is printed without its own `p` marker: the NLI model's
permutation p on G_299 appears as the bare second value in
"($p = 0.0134$ on $G_{253}$, $0.0120$ on $G_{299}$)". A regex over `p` cannot
see it, so it is declared here and its presence in the source is asserted.

No MIMIC data is read. Inputs are the committed .tex sources only.

    python3 accord/stats/pvalue_precision_audit.py
"""
from __future__ import annotations

import collections
import decimal
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
THESIS = ROOT / "accord" / "thesis"
OUT = ROOT / "results" / "accord" / "stats" / "pvalue_precision_audit.json"

FILES = sorted(THESIS.glob("msu_ch*.tex")) + sorted(THESIS.glob("msu_app*.tex"))
FACTS = json.loads((ROOT / "results" / "accord" / "VERIFIED_FACTS.json").read_text())

# p values that carry no `p` marker of their own, declared with the string that
# must be present in the named file for the declaration to stand.
DECLARED_BARE = [
    {"file": "msu_ch4_results.tex", "printed": "0.0120",
     "context_must_contain": "$p = 0.0134$ on $G_{253}$, $0.0120$ on",
     "what": "NLI average-precision label-permutation p on G_299, printed as the "
             "bare companion of the G_253 value"},
]

TRACE_FILE = "msu_appC_trace.tex"
TRACE_P_KEY = re.compile(r"p_permutation|p_two_sided|p_one_sided|p_value"
                         r"|p_null_ge|[._]p$")
# keys that read like a p value but are not one: a coverage percentage whose name
# begins with the prefix p_ for "proportion"
TRACE_P_KEY_EXCLUDE = re.compile(r"p_interval_contains")

PROSE = re.compile(
    r"(?<![A-Za-z])p\$?\s*(=|<|>|\\leq|\\geq|\\le|\\ge)\s*\$?\s*([0-9]*\.[0-9]+)")
HEADER_P = re.compile(r"(?<![A-Za-z\\])p(?![A-Za-z])")
NUMBER = re.compile(r"^-?[0-9]*\.[0-9]+$")
THRESHOLDS = (0.05, 0.01, 0.001)


def decimals(printed: str) -> int:
    return len(printed.split(".")[1]) if "." in printed else 0


def round4(printed: str) -> str:
    """Four-decimal form under round-half-up, the convention a reader assumes."""
    q = decimal.Decimal(printed).quantize(decimal.Decimal("0.0001"),
                                          rounding=decimal.ROUND_HALF_UP)
    return f"{q:.4f}"


def strip_tex(cell: str) -> str:
    s = cell.strip()
    for tok in ("\\textbf{", "\\emph{", "\\mathbf{", "$", "}", "{", "\\,"):
        s = s.replace(tok, "")
    return s.strip()


def table_p_columns(text: str, fname: str):
    """Every value in a tabular column whose header carries a standalone p."""
    out = []
    for tab in re.finditer(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", text,
                           re.S):
        block = tab.group(1)
        offset = tab.start(1)
        # logical rows: the source wraps a header across lines, so join on \\
        rows = [r for r in block.split("\\\\")]
        header = None
        header_idx = None
        for i, r in enumerate(rows):
            clean = r.replace("\\toprule", "").replace("\\midrule", "")
            if "&" in clean and "\\textbf" in clean:
                header = [strip_tex(c) for c in clean.split("&")]
                header_idx = i
                break
        if header is None:
            continue
        pcols = [j for j, h in enumerate(header) if HEADER_P.search(h)]
        if not pcols:
            continue
        label_m = re.search(r"\\label\{([^}]*)\}", text[tab.end():tab.end() + 4000])
        label = label_m.group(1) if label_m else "unlabelled"
        for r in rows[header_idx + 1:]:
            clean = (r.replace("\\midrule", "").replace("\\bottomrule", "")
                      .replace("\\toprule", ""))
            if "&" not in clean:
                continue
            cells = [strip_tex(c) for c in clean.split("&")]
            for j in pcols:
                if j >= len(cells):
                    continue
                v = cells[j]
                if NUMBER.match(v) and not v.startswith("-"):
                    line = text.count("\n", 0, offset + block.index(r)
                                      if r in block else offset) + 1
                    out.append({"file": fname, "line": line, "source": "table column",
                                "table": label, "column_header": header[j],
                                "relation": "=", "printed": v})
    return out


def trace_p_rows(text: str, fname: str):
    """Appendix C trace rows whose register key names a p value."""
    out = []
    for i, line in enumerate(text.split("\n"), 1):
        if "&" not in line or not line.rstrip().endswith("\\\\"):
            continue
        cells = line.rstrip().rstrip("\\").split("&")
        if len(cells) < 2:
            continue
        value = strip_tex(re.sub(r"\\textsuperscript\{[a-z]\}", "", cells[0]))
        key = cells[1].replace("\\allowbreak{}", "").replace("\\_", "_").strip()
        if not NUMBER.match(value) or value.startswith("-"):
            continue
        if not TRACE_P_KEY.search(key) or TRACE_P_KEY_EXCLUDE.search(key):
            continue
        rec = {"file": fname, "line": i, "source": "Appendix C trace, Value column",
               "table": "app:trace", "column_header": "Value",
               "register_key": key, "relation": "=", "printed": value}
        entry = FACTS.get(key)
        if entry is not None and isinstance(entry.get("value"), (int, float)):
            reg = float(entry["value"])
            rec["registered_value"] = reg
            # the trace prints the digit string it found in a chapter, so a row
            # whose printed value does not round-trip to its register key is a
            # resolution collision: the digits matched some other quantity
            dp = decimals(value)
            rec["trace_value_agrees_with_register"] = f"{reg:.{dp}f}" == value
        out.append(rec)
    return out


def main():
    records = []
    for path in FILES:
        text = path.read_text()
        if path.name == TRACE_FILE:
            records.extend(trace_p_rows(text, path.name))
            continue
        for i, line in enumerate(text.split("\n"), 1):
            for m in PROSE.finditer(line):
                a, b = max(0, m.start() - 70), min(len(line), m.end() + 30)
                records.append({"file": path.name, "line": i, "source": "prose or note",
                                "table": None, "column_header": None,
                                "relation": m.group(1), "printed": m.group(2),
                                # line numbers move as the chapters are edited, so
                                # each record carries its own locating context
                                "context": line[a:b].strip()})
        records.extend(table_p_columns(text, path.name))

    for d in DECLARED_BARE:
        text = (THESIS / d["file"]).read_text()
        assert d["context_must_contain"] in text, (
            f"declared bare p value {d['printed']} no longer has its context in "
            f"{d['file']}; the declaration is stale and must be rechecked")
        line = text.count("\n", 0, text.index(d["context_must_contain"])) + 1
        records.append({"file": d["file"], "line": line,
                        "source": "prose, bare companion value", "table": None,
                        "column_header": None, "relation": "=",
                        "printed": d["printed"], "what": d["what"]})

    for r in records:
        p = r["printed"]
        r["decimals_printed"] = decimals(p)
        r["rounded_to_4dp"] = round4(p)
        r["changes_at_4dp"] = round4(p) != p
        crossings = []
        for t in THRESHOLDS:
            below_now = float(p) < t
            below_4dp = float(round4(p)) < t
            if below_now != below_4dp:
                crossings.append(t)
        r["threshold_reading_changes_at"] = crossings
        # a value at its null's resolution floor reads as an exact figure once
        # rounded, which is a different and stronger claim than the test made
        r["four_dp_form_is_a_floor_not_a_value"] = (
            r["decimals_printed"] > 4 and float(p) < 1e-4)
        if (crossings or r["four_dp_form_is_a_floor_not_a_value"]
                or r.get("trace_value_agrees_with_register") is False):
            r["flag"] = True

    records.sort(key=lambda r: (r["file"], r["line"], r["printed"]))
    by_dp = collections.Counter(r["decimals_printed"] for r in records)
    non4 = [r for r in records if r["decimals_printed"] != 4]
    flagged = [r for r in records if r.get("flag")]

    out = {
        "what": "every p value printed in the thesis chapters and appendices, with "
                "its printed precision and its four-decimal form",
        "addresses": "the chapters print five p values at five decimals beside "
                     "four-decimal values; three reviews have asked for one precision",
        "edits_made": "none; this script reads the .tex sources and writes JSON only",
        "files_scanned": [p.name for p in FILES],
        "n_p_values": len(records),
        "count_by_printed_decimals": {str(k): v for k, v in sorted(by_dp.items())},
        "n_not_at_four_decimals": len(non4),
        "n_that_change_when_rounded_to_four": sum(
            1 for r in records if r["changes_at_4dp"]),
        "not_at_four_decimals": [
            {**{k: r[k] for k in ("file", "line", "printed", "decimals_printed",
                                  "rounded_to_4dp", "threshold_reading_changes_at",
                                  "four_dp_form_is_a_floor_not_a_value")},
             "context": r.get("context") or r.get("register_key")}
            for r in non4],
        "flagged_rounding_to_four_changes_how_it_reads": [
            {k: r[k] for k in ("file", "line", "printed", "rounded_to_4dp",
                               "threshold_reading_changes_at",
                               "four_dp_form_is_a_floor_not_a_value")}
            for r in flagged],
        "trace_rows_whose_value_disagrees_with_their_register_key": [
            {k: r[k] for k in ("file", "line", "printed", "register_key",
                               "registered_value")}
            for r in records if r.get("trace_value_agrees_with_register") is False],
        "thresholds_checked": list(THRESHOLDS),
        "rounding_convention": "round half up, the convention a reader applies by hand; "
                               "Python's round() is half-to-even and would give a "
                               "different answer for a value ending in 5",
        "all_p_values": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{len(records)} p values; by printed decimals: {dict(sorted(by_dp.items()))}")
    for r in records:
        mark = "  <-- FLAG" if r.get("flag") else ""
        if r["decimals_printed"] != 4:
            print(f"{r['file']}:{r['line']:5d}  {r['printed']:>9s} "
                  f"({r['decimals_printed']} dp) -> {r['rounded_to_4dp']}{mark}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
