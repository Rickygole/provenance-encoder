#!/usr/bin/env python3
"""Cross-chapter consistency: find quantities described by the same phrase but
carrying different values in different chapters (the C53 failure class).

Re-run: python3 accord/stats/audit_cross_consistency.py
"""
import json, os, re, collections

THESIS = "/Users/rickygole/provenance-encoder/accord/thesis"
OUT = "/Users/rickygole/provenance-encoder/results/accord/_audit_cross_consistency.json"

# (probe name, regex over the raw line, group index holding the number)
PROBES = [
    ("top15_true_contradictions", r"(fifteen|15) highest-scored pairs contain (\w+)", 2),
    ("auroc_denominator",         r"the \$?(299|253)\$? (?:pharmacist-)?annotated pairs[^.]{0,80}AUROC", 1),
    ("gpt4o_pair_count",          r"zero-shot[^.]{0,120}?same \$?(299|253)\$? pairs", 1),
    ("alpha_move",                r"moved? by \$?(\d\.\d)\\%\$? and", 1),
    ("weak_labeller_mcc",         r"labeller[^.]{0,80}?scores \$?\+?(-?0\.\d+)", 1),
    ("accord_mcc",                r"Matthews correlation is \$?(-?[\+]?0\.\d+)", 1),
    ("mednli_acc",                r"(83\.9\d?)\\%", 1),
    ("exemplar_wrong_rate",       r"(41\.4)\\%", 1),
    ("base_rate",                 r"base rate of \$?(\d\.\d+)\\?%?", 1),
    ("rule_f1",                   r"\$F_1\$ of \$?(0\.9\d+)", 1),
]

def main():
    hits = collections.defaultdict(list)
    for fn in sorted(os.listdir(THESIS)):
        if not fn.endswith(".tex"):
            continue
        for i, line in enumerate(open(os.path.join(THESIS, fn)), 1):
            if line.lstrip().startswith("%"):
                continue
            for name, rx, gi in PROBES:
                for m in re.finditer(rx, line):
                    hits[name].append({"file": fn, "line": i, "value": m.group(gi),
                                       "context": line.strip()[:180]})
    report = {}
    for name, lst in hits.items():
        vals = sorted({h["value"] for h in lst})
        report[name] = {"distinct_values": vals, "consistent": len(vals) <= 1,
                        "occurrences": lst}
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=1)
    for name, r in report.items():
        flag = "OK  " if r["consistent"] else "DIFF"
        print(f"{flag} {name}: {r['distinct_values']}")
        if not r["consistent"]:
            for h in r["occurrences"]:
                print(f"       {h['file']}:{h['line']} -> {h['value']}")

if __name__ == "__main__":
    main()
