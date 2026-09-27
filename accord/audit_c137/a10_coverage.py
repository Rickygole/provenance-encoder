"""Audit pass 10: every numeric literal in every thesis tex file against the register."""
import json,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/"results/accord/audit_c137"
F=json.load(open(ROOT/"results/accord/VERIFIED_FACTS.json"))
vals=set()
def add(v):
    if isinstance(v,(int,float)) and not isinstance(v,bool):
        for nd in range(0,7): vals.add(f"{round(float(v),nd):.{nd}f}".rstrip('0').rstrip('.') if nd else str(int(round(v))))
        vals.add(repr(float(v))); vals.add(str(v))
        for nd in range(0,7): vals.add(f"{float(v):.{nd}f}")
for k,v in F.items():
    x=v.get("value") if isinstance(v,dict) else v
    if isinstance(x,list):
        for e in x: add(e)
    else: add(x)
R={}
for tex in sorted((ROOT/"accord/thesis").glob("msu_ch*.tex")):
    body=tex.read_text()
    body=re.sub(r"%.*","",body)          # strip comments
    lits=re.findall(r"(?<![A-Za-z0-9._])(\d+\.\d{3,})",body)
    miss={}
    for l in sorted(set(lits)):
        f=float(l)
        ok = l in vals or f"{f:.4f}" in vals or any(abs(f-float(c))<5e-5 for c in [] )
        if not ok:
            # tolerant check against every register float
            ok=False
            for k,v in F.items():
                x=v.get("value") if isinstance(v,dict) else v
                if isinstance(x,(int,float)) and not isinstance(x,bool) and abs(round(float(x),len(l.split('.')[1]))-f)<1e-12:
                    ok=True; break
        if not ok: miss[l]=body.count(l)
    R[tex.name]=dict(n_literals=len(set(lits)),unmatched=miss)
json.dump(R,open(OUT/"a10_coverage.json","w"),indent=1,sort_keys=True); print(json.dumps(R,indent=1,sort_keys=True))
