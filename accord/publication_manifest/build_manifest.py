#!/usr/bin/env python3
"""Recompute the minimal publication manifest for this thesis.

WHAT THIS COMPUTES
------------------
The set of files that is necessary and sufficient for a clone to (a) build
main_msu.pdf, (b) re-derive every number the thesis prints from a committed
artifact, and (c) return the SAME verdict this repository returns on its own six
gates:

    python3 accord/stats/verify_thesis.py
    python3 accord/stats/verify_thesis_deep.py
    python3 accord/stats/make_appendices.py && python3 accord/stats/trace_claims.py
    cd accord/thesis/figures && python3 tools/check_figure_invariants.py
    cd accord/thesis/figures && python3 tools/measure_ink.py
    cd accord/thesis && bash build.sh

HOW, AND WHY NOT JUST STATIC ANALYSIS
-------------------------------------
Static import analysis is not enough here: the gates resolve artifact paths out
of the register at run time, and the figure generators read result JSONs by
string. So the closure is measured in two stages.

  STAGE 1, instrumented runs. The six commands run in a throwaway copy of the
  repository with builtins.open, io.open and os.open wrapped, recording every
  path opened under the tree, plus every sys.modules entry whose __file__ is in
  the tree. main_msu.tex is compiled with -recorder so main_msu.fls gives the
  authoritative LaTeX input list.

  STAGE 2, fixpoint sufficiency. An open() hook is NOT trustworthy on its own;
  the first version of this analysis missed two pathlib reads inside the figure
  generators and would have shipped a manifest that does not build. So a tree
  containing only the stage 1 candidates is assembled, the six commands are run
  IN IT, every path named in a FileNotFoundError or equivalent is copied in, and
  the loop repeats until a round adds nothing. Then the minimal tree's verdicts
  are compared to the full tree's. Sufficiency is demonstrated, not argued.

NOTHING IS WRITTEN OUTSIDE --out AND THE SCRATCH DIRECTORY. The gates regenerate
appendices and CLAIM_TRACE.json, which is why they are never run in place.
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
GATES = [
    (".", [sys.executable, "accord/stats/verify_thesis.py"], "verify_thesis"),
    (".", [sys.executable, "accord/stats/verify_thesis_deep.py"], "verify_deep"),
    (".", [sys.executable, "accord/stats/make_appendices.py"], "make_appendices"),
    (".", [sys.executable, "accord/stats/trace_claims.py"], "trace_claims"),
    ("accord/thesis/figures", [sys.executable, "tools/check_figure_invariants.py"], "figinv"),
    ("accord/thesis/figures", [sys.executable, "tools/measure_ink.py"], "measure_ink"),
    ("accord/thesis", ["bash", "build.sh"], "build"),
]
# Run directly too: pass 10 launches these as subprocesses and swallows stderr,
# so a missing input of theirs would otherwise show up only as "no generator
# rebuilt it" with no path to act on.
# verify_thesis_deep.py's own FIG_SCRIPTS lists only the first five. fig_batch3.py
# builds fig-scale-ladder and fig-answer-polarity, both of which
# check_figure_invariants.py requires to be present, and pass 10 never runs it: a
# gate that cannot fail for the two newest figures. It is run here so its data
# input (results/accord/framing_mechanism/answer_polarity.json) enters the closure.
FIG_SCRIPTS = ["make_figures.py", "fig_batch2.py", "fig_batch3.py",
               "fig_threshold_surface.py", "fig_frontier.py", "fig_ceiling.py"]

# LaTeX byproducts. build.sh regenerates all of them from main_msu.tex, so they
# are never part of the closure even though they exist in the working tree and
# get named in build logs. main_msu.toc and main_msu.pdf are the exceptions and
# are added on purpose: deep pass 9 guards its TOC-prefix, page-count and
# internal-filename checks with exists(), so without them the gate passes by
# not looking.
BYPRODUCT = re.compile(r"\.(aux|bcf|bbl|blg|log|lof|lot|out|fls|run\.xml|synctex\.gz)$")

TRACER = '''
import atexit, builtins, io, json, os, sys
OUT = os.environ["TRACE_OUT"]; PRE = os.path.realpath(os.environ["TRACE_ROOT"])
opened = {}
def _rec(p):
    try: p = os.path.realpath(os.path.abspath(os.fspath(p)))
    except Exception: return
    if p.startswith(PRE): opened[p] = 1
_open = builtins.open
def open(f, mode="r", *a, **k):
    _rec(f); return _open(f, mode, *a, **k)
builtins.open = open; io.open = open
_oo = os.open
def _osopen(p, fl, *a, **k):
    _rec(p); return _oo(p, fl, *a, **k)
os.open = _osopen
@atexit.register
def _dump():
    mods = {}
    for n, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not f: continue
        try: p = os.path.realpath(os.path.abspath(f))
        except Exception: continue
        if p.startswith(PRE): mods[p] = n
    with _open(OUT, "a") as fh:
        fh.write(json.dumps({"opened": sorted(opened), "modules": sorted(mods)}) + "\\n")
'''


def is_file(p):
    """Path.is_file() that survives nonsense. verify_thesis.py resolves the
    register's artifact strings as paths, and many of them are prose ("derived:
    0.5255 raised to the 23rd power, ..."), which os.stat rejects with ENAMETOOLONG
    rather than returning False."""
    try:
        return pathlib.Path(p).is_file()
    except OSError:
        return False


def run(tree, cwd, cmd, env=None):
    e = dict(os.environ, PATH="/Library/TeX/texbin:/opt/homebrew/bin:" + os.environ["PATH"])
    e.update(env or {})
    p = subprocess.run(cmd, cwd=str(pathlib.Path(tree) / cwd), env=e,
                       capture_output=True, text=True)
    return p.stdout + p.stderr


def run_all(tree, trace_out=None):
    """Run the seven commands plus the five figure generators. Returns the log."""
    env = {}
    if trace_out:
        sc = pathlib.Path(tempfile.mkdtemp()) / "sitecustomize.py"
        sc.write_text(TRACER)
        env = {"PYTHONPATH": str(sc.parent), "TRACE_OUT": str(trace_out),
               "TRACE_ROOT": str(tree)}
    log = ""
    for cwd, cmd, name in GATES:
        log += f"\n### {name}\n" + run(tree, cwd, cmd, env)
    figout = tempfile.mkdtemp()
    for s in FIG_SCRIPTS:
        log += f"\n### fig {s}\n" + run(tree, "accord/thesis/figures",
                                       [sys.executable, s],
                                       dict(env, ACCORD_FIG_OUT=figout))
    return log


def verdicts(log):
    """The lines that say whether each gate passed, for full-vs-minimal parity."""
    keep = (r"^RESULT:", r"^OVERALL:", r"all gating figure invariants",
            r"^all figures at or above", r"^\s+pass$", r"^Appendix [AB]:",
            r"^traced\s", r"stale: ", r"^   pages:", r"FAIL")
    return [l.strip() for l in log.splitlines()
            if any(re.search(k, l) for k in keep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/accord/publication_manifest"))
    ap.add_argument("--scratch", default=None)
    a = ap.parse_args()
    scratch = pathlib.Path(a.scratch or tempfile.mkdtemp(prefix="pubmanifest-"))
    full, minimal = scratch / "full", scratch / "minimal"
    trace = scratch / "trace.jsonl"

    print(f"scratch: {scratch}")
    print("copying the repository (excluding .git, local_only, __pycache__) ...")
    full.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--exclude", ".git", "--exclude", "local_only",
                    "--exclude", "__pycache__", str(ROOT) + "/", str(full) + "/"],
                   check=True)

    print("stage 1: instrumented runs in the full copy ...")
    full_log = run_all(full, trace_out=trace)
    # pdflatex -recorder: the authoritative LaTeX input list.
    run(full, "accord/thesis", ["pdflatex", "-recorder", "-interaction=nonstopmode",
                                "main_msu.tex"])
    cand, why = set(), {}

    def note(rel, method, by):
        why.setdefault(rel, []).append({"method": method, "by": by})
        cand.add(rel)

    for line in trace.read_text().splitlines():
        d = json.loads(line)
        for p in d["opened"]:
            r = os.path.relpath(p, os.path.realpath(full))
            if is_file(full / r) and not BYPRODUCT.search(r):
                note(r, "instrumented run: open() hook recorded a read", "gate suite")
        for p in d["modules"]:
            note(os.path.relpath(p, os.path.realpath(full)),
                 "instrumented run: imported as a Python module", "gate suite")
    fls = full / "accord/thesis/main_msu.fls"
    if fls.exists():
        for m in re.finditer(r"^INPUT (\./)?(?!/)(.*)$", fls.read_text(), re.M):
            r = "accord/thesis/" + m.group(2)
            if is_file(full / r) and not BYPRODUCT.search(r):
                note(r, "pdflatex -recorder: INPUT line in main_msu.fls", "build.sh")
    for r in ["accord/thesis/build.sh", "accord/thesis/main_msu.tex",
              "accord/thesis/main_msu.toc", "accord/thesis/main_msu.pdf",
              "accord/thesis/refs.bib", "environment.yml"]:
        note(r, "entry point, build driver, or a check that silently skips when absent",
             "task definition")

    print(f"stage 1 candidates: {len(cand)}")
    print("stage 2: fixpoint sufficiency test in a minimal tree ...")
    if minimal.exists():
        shutil.rmtree(minimal)
    for r in sorted(cand):
        (minimal / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(full / r, minimal / r)
    for rnd in range(1, 13):
        log = run_all(minimal)
        new = 0
        for m in re.finditer(re.escape(str(minimal)) + r"/([A-Za-z0-9_./-]+)", log):
            r = m.group(1)
            if (minimal / r).exists() or not is_file(full / r) or BYPRODUCT.search(r):
                continue
            (minimal / r).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(full / r, minimal / r)
            note(r, f"fixpoint sufficiency test: the minimal tree failed until this "
                    f"file was copied in (round {rnd})", "minimal tree")
            new += 1
        # A figure PDF that a gate names without a path is reported, not raised.
        # check_figure_invariants.py REPORTS a missing figure PDF instead of
        # raising, in three different wordings (bound, pending, retired), so the
        # path never appears in a traceback and the fixpoint loop above cannot
        # see it. All three are matched here. Every figure PDF the gate names is
        # load-bearing: the orphan check is what keeps a retired figure from
        # quietly becoming an untracked one.
        for m in re.finditer(r"(fig-[a-z0-9-]+): (?:.*PDF is absent"
                             r"|declares category grounds but is not a live)", log):
            r = f"accord/thesis/figures/{m.group(1)}.pdf"
            if (minimal / r).exists() or not is_file(full / r):
                continue
            shutil.copy2(full / r, minimal / r)
            note(r, f"fixpoint sufficiency test: check_figure_invariants.py reported it "
                    f"absent (round {rnd})", "minimal tree")
            new += 1
        print(f"  round {rnd}: added {new}")
        if not new:
            break
    min_log = log

    fv, mv = verdicts(full_log), verdicts(min_log)
    parity = fv == mv
    print(f"verdict parity full vs minimal: {'IDENTICAL' if parity else 'DIFFERENT'}")
    if not parity:
        print("  only in full:   ", [x for x in fv if x not in mv][:12])
        print("  only in minimal:", [x for x in mv if x not in fv][:12])

    paths = sorted(cand)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest_include.txt").write_text("\n".join(paths) + "\n")
    ev = {p: {"size_bytes": (ROOT / p).stat().st_size if (ROOT / p).exists() else None,
              "evidence": why[p]} for p in paths}
    (out / "closure_evidence.json").write_text(json.dumps({
        "how_this_was_computed": __doc__.strip().splitlines(),
        "verdict_parity": {"identical": parity, "full": fv, "minimal": mv},
        "counts": {"files": len(paths),
                   "bytes": sum(v["size_bytes"] or 0 for v in ev.values())},
        "paths": ev}, indent=1))
    print(f"wrote {out/'manifest_include.txt'} ({len(paths)} files)")
    print("manifest_exclude_rationale.md and audit_findings.md are prose and are "
          "maintained by hand; re-read them if this run changed the include set.")
    return 0 if parity else 1


if __name__ == "__main__":
    sys.exit(main())
