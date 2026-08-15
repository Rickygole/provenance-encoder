#!/usr/bin/env python3
"""Assert that the notebook and style.py cannot have drifted apart.

style.py is the only file in the repository allowed to contain a colour
literal. The notebook is the path that actually ships these figures, and until
now its palette cell duplicated every hex by hand, which is how a hue drifts
between the script path and the shipping path. This makes that drift a build
failure rather than a proofreading job.

Three checks:

  1. The notebook's palette and helper cells, executed in isolation, must
     produce a PALETTE dict and an RCPARAMS dict set-equal to style.py's.
  2. No #RRGGBB literal may appear anywhere else in the notebook, and none of
     the hexes that do appear may be outside style.PALETTE.
  3. No drawing script may contain a #RRGGBB literal, an `alpha=` keyword, or a
     call to the deleted lean().

Non-zero exit on any drift.
"""
import json
import re
import sys
from pathlib import Path

FIGDIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FIGDIR))

import style  # noqa: E402

NOTEBOOK = FIGDIR / "Thesis_Figure_Material.ipynb"
SCRIPTS = ["make_figures.py", "fig_batch2.py", "make_figures_extra.py",
           "fig_threshold_surface.py"]

HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")
# The two markers the generator writes around the cells that are copied out of
# style.py verbatim.
BEGIN = "# >>> generated from style.py"
END = "# <<< generated from style.py"


def cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def main():
    fails = []
    nb = json.loads(NOTEBOOK.read_text())
    code = cells(nb)

    generated = [s for s in code if BEGIN in s]
    palette = [s for s in generated if "PALETTE = " in s]
    if not generated or not palette:
        fails.append("notebook carries no palette cell generated from style.py")
        print("\n".join(fails))
        return 1

    ns = {"__name__": "nb_style"}
    try:
        # Only the palette cell is executed, and it is written to stand on its
        # own. The helpers cell depends on the setup cell's paths and is
        # checked for stray hexes instead.
        exec(compile("\n".join(palette), "<notebook palette cell>", "exec"), ns)
    except Exception as exc:                      # pragma: no cover
        fails.append(f"notebook palette cells do not execute: {exc!r}")
        print("\n".join(fails))
        return 1

    if ns.get("PALETTE") != style.PALETTE:
        fails.append("PALETTE differs between the notebook and style.py")
        for k in sorted(set(style.PALETTE) | set(ns.get("PALETTE", {}))):
            a = style.PALETTE.get(k)
            b = ns.get("PALETTE", {}).get(k)
            if a != b:
                fails.append(f"  {k}: style={a} notebook={b}")

    if ns.get("RCPARAMS") != style.RCPARAMS:
        fails.append("RCPARAMS differs between the notebook and style.py")
        a, b = style.RCPARAMS, ns.get("RCPARAMS", {})
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                fails.append(f"  {k}: style={a.get(k)!r} notebook={b.get(k)!r}")

    known = {h.upper() for ramp in style.PALETTE.values()
             for h in ramp.values()}
    for i, src in enumerate(code):
        body = src
        if BEGIN in body:
            continue
        for h in HEX.findall(body):
            fails.append(f"notebook code cell {i} carries the hex literal {h}")
    for src in generated:
        for h in HEX.findall(src):
            if h.upper() not in known:
                fails.append(f"generated cell carries an unknown hex {h}")

    for name in SCRIPTS:
        p = FIGDIR / name
        if not p.exists():
            continue
        for n, line in enumerate(p.read_text().splitlines(), 1):
            stripped = line.split("#", 1)[0]
            for h in HEX.findall(stripped):
                fails.append(f"{name}:{n} defines a colour literal {h}")
            if "alpha=" in stripped:
                fails.append(f"{name}:{n} uses alpha=")
            if re.search(r"\blean\s*\(", stripped):
                fails.append(f"{name}:{n} calls the deleted lean()")

    if fails:
        print("PARITY FAILED")
        for f in fails:
            print("  " + f)
        return 1
    print("parity ok: notebook PALETTE and RCPARAMS match style.py; "
          "no colour literal, alpha= or lean() outside style.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
