#!/usr/bin/env python3
"""Regenerate Thesis_Figure_Material.ipynb from style.py and the drawing
scripts, so the shipping path and the script path cannot differ.

The notebook's palette and helper cells are style.py's own source, copied
verbatim between two markers; every figure cell is the corresponding drawing
block, copied verbatim from the script that owns it. tools/check_parity.py then
gates the pair.

Run from anywhere: python figures/tools/regen_notebook.py
"""
import json
import re
import sys
import textwrap
from pathlib import Path

FIGDIR = Path(__file__).resolve().parent.parent
NB = FIGDIR / "Thesis_Figure_Material.ipynb"
BEGIN = "# >>> generated from style.py"
END = "# <<< generated from style.py"

style_src = (FIGDIR / "style.py").read_text()
mf = (FIGDIR / "make_figures.py").read_text()
b2 = (FIGDIR / "fig_batch2.py").read_text()
ex = (FIGDIR / "make_figures_extra.py").read_text()
ts = (FIGDIR / "fig_threshold_surface.py").read_text()


def style_section(start_marker, end_marker=None):
    i = style_src.index(start_marker)
    j = len(style_src) if end_marker is None else style_src.index(end_marker)
    return style_src[i:j].rstrip() + "\n"


def fn_body(name):
    """The body of a make_figures.py function, dedented, minus its docstring."""
    m = re.search(rf"^def {name}\(\):\n(.*?)(?=\n\n\n)", mf, re.S | re.M)
    body = textwrap.dedent(m.group(1))
    body = re.sub(r'^\s*""".*?"""\n', "", body, count=1, flags=re.S)
    return body.rstrip() + "\n"


SEP = "# " + "=" * 77


def block(src, title):
    """A `# ==== / # Fig N -- name / # ====` delimited block of a linear script.

    The search for the block's END has to start after the header's OWN closing
    separator. Starting it at the title matched that closing rule three lines
    later and returned the header alone, so four of the twelve figure cells
    shipped as a two-line comment: the notebook could not draw its own figures
    and check_parity, which only greps for colour literals, still passed.
    """
    i = src.index(title)
    start = src.rindex(SEP, 0, i)
    close = src.index(SEP, i)                 # the header's closing separator
    nxt = src.find(SEP, close + len(SEP))     # the next block's opening one
    return src[start:len(src) if nxt == -1 else nxt].rstrip() + "\n"


PALETTE_CELL = (
    BEGIN + "\n"
    "# style.py sections 1-6: the ramps, the structure tones, the entity map,\n"
    "# the type scale, the line weights and the rcParams. Copied verbatim, so\n"
    "# tools/check_parity.py can assert set-equality with the module itself.\n"
    "import matplotlib\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.patheffects as mpe\n"
    "from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, ArrowStyle\n\n"
    + style_section("# ---------------------------------------------------------------------------\n# 1. THE RAMPS",
                    "# ---------------------------------------------------------------------------\n# 7. MEASURE")
    + "\nsetup()\n" + END + "\n")

HELPERS_CELL = (
    BEGIN + "\n"
    "# style.py sections 7-10: measure, the frame system, marks and grounds,\n"
    "# and the one interval this set is allowed to compute.\n"
    + style_section("# ---------------------------------------------------------------------------\n# 7. MEASURE")
    + """

# Notebook-local bindings the drawing cells below expect. They exist in the
# scripts as module globals and are reproduced here so a figure cell is the
# same code in both places.
import statistics as st, glob
ROOT = pathlib.Path(REPO)
HERE = pathlib.Path(FIGDIR)
SUPP = pathlib.Path(RESULTS_ACCORD) / "decontam"
MANIFEST = []
man = {}


def load(*parts):
    path = os.path.join(RESULTS_ACCORD, *parts)
    with open(path) as f:
        return json.load(f), path


def register(fname, label, sources, note=""):
    MANIFEST.append({"file": os.path.join("figures", fname), "label": label,
                     "sources": sources, "note": note})


def mcc_from_counts(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else num / den
""" + END + "\n")


def fix_paths(code):
    code = code.replace("(REPO / ", "(ROOT / ")
    code = code.replace("(S / ", "(SUPP / ")
    code = code.replace('pathlib.Path(__file__).parent / "fig-threshold-surface.pdf"',
                        'HERE / "fig-threshold-surface.pdf"')
    return code


FIGURE_CELLS = {
    "fig-pipeline-status":    fn_body("make_fig_pipeline_status"),
    "fig-intro-headline":     fn_body("make_fig_intro_headline"),
    "fig-cohort-flow":        fn_body("make_fig_cohort_flow"),
    "fig-exemplar":           fn_body("make_fig_exemplar"),
    "fig-prevalence-response": fix_paths(block(b2, "# Fig 5 -- fig-prevalence-response")),
    "fig-falsification":      fix_paths(ex[ex.index("# ---- fig:falsification"):
                                           ex.index('p = HERE / "manifest_extra.json"')].rstrip() + "\n"),
    "fig-power":              fix_paths(block(b2, "# Fig 7 -- fig-power")),
    "fig-forest":             fix_paths(block(b2, "# Fig 8 -- fig-forest")),
    "fig-scaling":            fix_paths(block(b2, "# Fig 9 -- fig-scaling")),
    "fig-threshold-surface":  fix_paths(ts[ts.index('ROOT = pathlib.Path('):].rstrip() + "\n"),
    "fig-disc-actionability": fn_body("make_fig_disc_actionability"),
    "fig-met-exchange":       fn_body("make_fig_met_exchange"),
}
CUT = ("fig-survey-audit", "fig-disc-strata")

# A cell that does not save its own figure is not that figure's cell. This is
# the assertion whose absence let four two-line stubs ship.
for _label, _code in FIGURE_CELLS.items():
    if f'"{_label}.pdf"' not in _code and f"{_label}.pdf" not in _code:
        raise SystemExit(f"regen_notebook: extracted cell for {_label} does "
                         f"not write {_label}.pdf ({len(_code)} chars)")

# fig-forest's provenance cell used to prepare the v2 family layout as well as
# load the values, and that layout carried the retired hue names. The Step 2
# cell now builds the whole figure, so this one only reads and prints.
STEP1_OVERRIDE = {"fig-forest": """iv = json.load(open(os.path.join(
    RESULTS_ACCORD, "stats", "intervals.json")))["E2_mcc_intervals"]
dec = json.load(open(os.path.join(
    RESULTS_ACCORD, "decontam", "corrected_gold_metrics.json")))["aggregate"]
df34 = json.load(open(os.path.join(
    RESULTS_ACCORD, "control", "final_report.json")))[
        "ensemble_mean_of_seed_scores"]["gold_defect_free_rule34"]
tr = json.load(open(os.path.join(RESULTS, "task_runs.json")))
sc = json.load(open(os.path.join(
    RESULTS_ACCORD, "scaling", "scaling_results.json")))

print("fig-forest: every headline estimate in the thesis, read against zero")
for k, v in iv.items():
    print(f"  MCC {k:32s} {v['mcc']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
print(f"  AUROC decontaminated ensemble        {dec['ensemble_auroc']-0.5:+.4f}"
      f"  [{dec['ensemble_ci95'][0]-0.5:+.4f}, {dec['ensemble_ci95'][1]-0.5:+.4f}]")
print(f"  AUROC defect-free stratum            {df34['auroc']-0.5:+.4f}"
      f"  [{df34['bootstrap']['ci95'][0]-0.5:+.4f}, {df34['bootstrap']['ci95'][1]-0.5:+.4f}]")
print("  pretraining gaps are recomputed in the figure cell from task_runs.json"
      " and scaling_results.json")
"""}


def src_lines(text):
    lines = text.splitlines(keepends=True)
    return lines


def main():
    nb = json.loads(NB.read_text())
    cells = nb["cells"]
    out = []
    i = 0
    n_fig, n_cut = 0, 0
    while i < len(cells):
        c = cells[i]
        s = "".join(c["source"])
        if c["cell_type"] == "code" and "%matplotlib inline" in s:
            out.append(c); i += 1; continue
        # the palette cell and the helpers cell, in that order
        if c["cell_type"] == "code" and (s.lstrip().startswith("# Structure.")
                                         or "PALETTE = " in s):
            c["source"] = src_lines(PALETTE_CELL); c["outputs"] = []
            c["execution_count"] = None
            out.append(c); i += 1; continue
        if c["cell_type"] == "code" and ("rcParams block" in s
                                         or "def wilson_ci" in s):
            c["source"] = src_lines(HELPERS_CELL); c["outputs"] = []
            c["execution_count"] = None
            out.append(c); i += 1; continue
        if c["cell_type"] == "markdown":
            m = re.search(r"## Figure \d+: `(fig-[a-z-]+)`", s)
            if m:
                label = m.group(1)
                if label in CUT:
                    # drop this heading and the two cells under it
                    i += 1
                    while i < len(cells) and not (
                            cells[i]["cell_type"] == "markdown"
                            and re.search(r"^## ", "".join(cells[i]["source"]))):
                        i += 1
                    n_cut += 1
                    continue
                n_fig += 1
                c["source"] = src_lines(f"## Figure {n_fig}: `{label}`\n")
                out.append(c); i += 1
                # step 1 cell (provenance print) is left alone; step 2 is replaced
                while i < len(cells):
                    cs = "".join(cells[i]["source"])
                    if cells[i]["cell_type"] == "markdown" and cs.startswith("## "):
                        break
                    if (cells[i]["cell_type"] == "code"
                            and label in STEP1_OVERRIDE
                            and "**Step 2." not in cs):
                        cells[i]["source"] = src_lines(STEP1_OVERRIDE[label])
                        cells[i]["outputs"] = []
                        cells[i]["execution_count"] = None
                        out.append(cells[i]); i += 1; continue
                    if cells[i]["cell_type"] == "markdown" and "**Step 2." in cs:
                        out.append(cells[i]); i += 1
                        cells[i]["source"] = src_lines(FIGURE_CELLS[label])
                        cells[i]["outputs"] = []
                        cells[i]["execution_count"] = None
                        out.append(cells[i]); i += 1
                        continue
                    out.append(cells[i]); i += 1
                continue
        if c["cell_type"] == "code" and "EXPECTED = [" in s:
            c["source"] = src_lines(
                "EXPECTED = [\n"
                + "".join(f'    "{k}.pdf",\n' for k in FIGURE_CELLS)
                + "]\n"
                "missing = [f for f in EXPECTED if not os.path.exists(os.path.join(FIGDIR, f))]\n"
                'print("figures written:", len(EXPECTED) - len(missing), "of", len(EXPECTED))\n'
                'for f in missing:\n'
                '    print("  MISSING", f)\n'
                'assert not missing, missing\n')
            c["outputs"] = []; c["execution_count"] = None
            out.append(c); i += 1; continue
        if c["cell_type"] == "markdown" and "Final: verify all fourteen" in s:
            c["source"] = src_lines("## Final: verify all twelve exist\n")
            out.append(c); i += 1; continue
        out.append(c); i += 1

    nb["cells"] = out
    NB.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"regenerated {NB.name}: {len(out)} cells, {n_fig} figures, {n_cut} cut")
    return 0


if __name__ == "__main__":
    sys.exit(main())
