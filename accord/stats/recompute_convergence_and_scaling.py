"""Recompute Table tab:conv and tab:scaling from the raw run lists.
The run records carry BOTH `val_accuracy` and `accuracy`; both are computed so the
chapter's numerator can be identified rather than assumed.
"""
import json, statistics as st
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[2]

conv = json.load(open(ROOT/"results/accord/scaling/convergence_results.json"))
print("n_runs =", len(conv), " keys per run:", sorted(conv[0].keys()))
tot = sum(r["minutes"] for r in conv)
print("total_minutes = %.4f -> rounded %d" % (tot, round(tot)))

for metric in ("accuracy", "val_accuracy", "macro_f1"):
    cells = defaultdict(list)
    for r in conv:
        cells[(r["tag"], r["budget"])].append(r[metric])
    tags = sorted({t for t,_ in cells}); budgets = sorted({b for _,b in cells})
    print(f"\n=== tab:conv under metric '{metric}' ===")
    for t in tags:
        print(" ", t, " ".join(f"{st.mean(cells[(t,b)]):.4f}({st.stdev(cells[(t,b)]):.4f}|{st.pstdev(cells[(t,b)]):.4f})" for b in budgets))
    m = lambda t,b: st.mean(cells[(t,b)])
    p = "pretrained_bert"; scr = [t for t in tags if t != p]
    for b in budgets:
        best = max(scr, key=lambda s: m(s,b))
        print(f"   budget {b}: gap = {100*(m(p,b)-m(best,b)):.4f} pts (best scratch {best})")
    print("   pretrained 3->24 change (pts): %.4f" % (100*(m(p,24)-m(p,3))))
    for s in scr:
        print(f"   {s} 3->24 change (pts): %.4f" % (100*(m(s,24)-m(s,3))))
    print("   max cell sd: ddof1=%.6f ddof0=%.6f" % (max(st.stdev(v) for v in cells.values()),
                                                     max(st.pstdev(v) for v in cells.values())))

print("\n=== scaling_results.json (tab:scaling public rows) ===")
sc = json.load(open(ROOT/"results/accord/scaling/scaling_results.json"))
print("keys per run:", sorted(sc[0].keys()))
for metric in ("val_accuracy","accuracy"):
    g = defaultdict(list); pc = {}
    for r in sc:
        if metric in r: g[r["tag"]].append(r[metric])
        pc[r["tag"]] = r.get("non_emb_params")
    print(f" -- {metric}")
    for t in sorted(g):
        print(f"    {t}: params={pc[t]} n={len(g[t])} mean={st.mean(g[t]):.6f} sd1={st.stdev(g[t]):.6f} sd0={st.pstdev(g[t]):.6f}")

print("\n=== task_runs.json (tab:scaling clinical rows) ===")
tr = json.load(open(ROOT/"results/task_runs.json"))
for k, v in sorted(tr.items()):
    if isinstance(v, dict):
        print(" ", k, {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))})
