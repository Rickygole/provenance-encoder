"""Safe, side-effect-free access to named pieces of the frozen ablations file.

`accord/detector/accord_phase3_v2_ablations.py` is frozen evidence (recovered
Colab source for Chapter 4's RQ3 falsifications): it runs `pip install` at
import time, mounts Google Drive, and reads data. It must never be imported
or executed as a whole, and this file never modifies it.

What Stage 3/4 of the C102 planted-defect benchmark need from it is a small,
named set of declarations: four `nn.Module` classes and seven module-level
assignments. This loader gets them without running the rest of the file:

  1. Read the file's bytes, assert the sha256 pin (the file changing
     underneath this loader, even by one byte, must be a loud failure, not a
     silently different frozen artifact).
  2. Parse the text with `ast` (never `exec` the whole module).
  3. Walk the parsed module body for exactly the ClassDef and Assign nodes
     named below; anything else in the file (the pip installs, the Drive
     mount, the training loop) is never visited.
  4. `compile` and `exec` ONLY those nodes' source, in a controlled namespace
     that supplies torch/torch.nn/torch.nn.functional (the only names the
     four class bodies reference at class-definition time; none of their
     `__init__` methods run until something the caller instantiates).

Every name request is explicit and pinned; there is no "load everything ast
finds," so a future edit to the frozen file that adds a fifth class or a
Drive-reading top-level statement cannot enter this loader unnoticed.
"""
from __future__ import annotations

import ast
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FROZEN_FILE = ROOT / "accord" / "detector" / "accord_phase3_v2_ablations.py"

# Pinned at the time this loader was written. `tests/test_public_controls_pins.py`
# re-derives this from the live file and fails loudly if it drifts.
FROZEN_FILE_SHA256 = "5595bb96f18ae62440dbea06a8360bba6d1c6f219b01c649d3dabd54c2124cdf"

ALLOWED_CLASSES = (
    "AsymmetricCrossNoteEncoder",
    "ClinicalDecisionEncoder",
    "AuthorityTemporalConflictGraph",
    "GuidelineGroundedHarmEstimator",
    "ACCORDv2Full",
)
ALLOWED_ASSIGNS = (
    "LABEL_MAP",
    "RULE_MAP",
    "ACTION_MAP",
    "DRUG_CLASS_MAP",
    "DRUG_CLASS_DANGER",
    "CONTRADICTION_WEIGHT",
    "RULE_TYPE_WEIGHT",
)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assign_target_name(node: ast.Assign):
    if len(node.targets) != 1:
        return None
    t = node.targets[0]
    return t.id if isinstance(t, ast.Name) else None


def load(names=None, extra_globals=None):
    """Return a namespace dict containing only the requested names (default:
    every name in ALLOWED_CLASSES + ALLOWED_ASSIGNS), extracted from the
    frozen file as described above. Raises if the sha256 pin does not match
    or if a requested name is not found among the allowed nodes.
    """
    actual = _sha256(FROZEN_FILE)
    if actual != FROZEN_FILE_SHA256:
        raise SystemExit(
            f"ABORT: {FROZEN_FILE} sha256 is {actual}, pinned value is "
            f"{FROZEN_FILE_SHA256}. Refusing to load from a frozen-evidence "
            "file that changed underneath this loader.")

    wanted = set(names) if names is not None else set(ALLOWED_CLASSES) | set(ALLOWED_ASSIGNS)
    unknown = wanted - (set(ALLOWED_CLASSES) | set(ALLOWED_ASSIGNS))
    if unknown:
        raise ValueError(f"not in the allow-list, refusing: {sorted(unknown)}")

    tree = ast.parse(FROZEN_FILE.read_text(), filename=str(FROZEN_FILE))
    # Keep the file's own order: ACCORDv2Full's __init__ instantiates the
    # other four classes by name, so if it is requested they must all be
    # defined, in the SAME exec call, in the order the frozen file defines
    # them -- a dict keyed by name (as this loop builds) and then filtered
    # and re-ordered by `tree.body` position below preserves that.
    nodes_by_name = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ALLOWED_CLASSES:
            nodes_by_name[node.name] = node
        elif isinstance(node, ast.Assign):
            tname = _assign_target_name(node)
            if tname in ALLOWED_ASSIGNS:
                nodes_by_name[tname] = node

    missing = wanted - set(nodes_by_name)
    if missing:
        raise SystemExit(
            f"ABORT: requested name(s) {sorted(missing)} not found among the "
            "pinned ClassDef/Assign nodes in the frozen file. The file's "
            "structure changed; the sha256 pin above should already have "
            "caught this.")

    # ACCORDv2Full.__init__ instantiates the other four classes; if it is
    # requested, pull its three undeclared dependencies (ACNE, CDE, ATCG,
    # GGHE) in too so the exec namespace can actually construct it, in
    # file order so later classes can see earlier ones.
    to_exec = set(wanted)
    if "ACCORDv2Full" in to_exec:
        to_exec |= {"AsymmetricCrossNoteEncoder", "ClinicalDecisionEncoder",
                    "AuthorityTemporalConflictGraph", "GuidelineGroundedHarmEstimator"}
    ordered_nodes = [node for node in tree.body
                     if (isinstance(node, ast.ClassDef) and node.name in to_exec)
                     or (isinstance(node, ast.Assign) and _assign_target_name(node) in to_exec)]

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    namespace = {"torch": torch, "nn": nn, "F": F, "__builtins__": __builtins__}
    if extra_globals:
        namespace.update(extra_globals)

    module = ast.Module(body=ordered_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    code = compile(module, filename=f"<frozen:{FROZEN_FILE.name}>", mode="exec")
    exec(code, namespace)  # noqa: S102 -- only the pinned, allow-listed nodes above

    return {n: namespace[n] for n in wanted}
