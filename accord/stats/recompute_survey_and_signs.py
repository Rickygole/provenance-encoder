import csv, json, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
S = ROOT/"results/accord/survey"

rows = list(csv.DictReader(open(S/"coding_table.csv")))
print("coding_table.csv rows =", len(rows))
print("columns:", list(rows[0].keys()))
for col in rows[0].keys():
    c = collections.Counter(r[col] for r in rows)
    if len(c) <= 10:
        print(" ", col, dict(c))

print("\n--- second_coder_table.csv ---")
sc = list(csv.DictReader(open(S/"second_coder_table.csv")))
print("rows =", len(sc), "columns:", list(sc[0].keys()))
for r in sc: print("  ", r)

print("\n--- second_coder_reliability.json ---")
print(json.dumps(json.load(open(S/"second_coder_reliability.json")), indent=1)[:3000])

print("\n--- coding_summary.json ---")
print(json.dumps(json.load(open(S/"coding_summary.json")), indent=1)[:3000])

print("\n--- signed_gap_analysis.json ---")
print(json.dumps(json.load(open(ROOT/"results/accord/signed_gap_analysis.json")), indent=1)[:4000])

print("\n########## RECOMPUTED CHECKS ##########")
# --- survey reliability: join the 19 double-coded papers to coder 1 ---
c1 = {r["paper_id"]: r for r in rows}
fields = ["audits_selection", "audits_selection_literal", "null_attribute_handling"]
print("second-coder overlap n =", len(sc), "; all in coder1 table:", all(r["paper_id"] in c1 for r in sc))
def kappa(a, b):
    cats = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c)/n) * (b.count(c)/n) for c in cats)
    return po, (None if abs(1-pe) < 1e-12 else (po-pe)/(1-pe)), pe
for f in fields:
    a = [c1[r["paper_id"]][f] for r in sc]; b = [r[f] for r in sc]
    po, k, pe = kappa(a, b)
    print(f"  {f}: agreement={po:.6f} kappa={k if k is None else round(k,6)} (pe={pe:.6f}, n={len(a)})")

# --- signed gaps: min/max signed demographic_parity_gap per experiment ---
sg = json.load(open(ROOT/"results/accord/signed_gap_analysis.json"))
print("\n signed_gap_analysis overall_max_discrepancy =", sg["reproduction_check"]["overall_max_discrepancy"])
for exp, blk in sg["zero_crossing_analysis"].items():
    pm = blk.get("per_metric", {}).get("demographic_parity_gap")
    if not pm: continue
    vals = [p["signed_value"] for p in pm["points"]]
    print(f"  {exp}: n={len(vals)} min={min(vals):.6f} max={max(vals):.6f} "
          f"first={vals[0]:.6f} last={vals[-1]:.6f} any_sign_change={len({v>0 for v in vals})>1}")
    for kk, vv in pm.items():
        if kk != "points": print("     ", kk, vv)
