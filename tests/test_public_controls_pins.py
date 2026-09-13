"""Pins for the public-controls work (C101 / C102, pipeline stage 2 BUILD).

These are not behavioral unit tests of a feature; each one pins an assumption
this work depends on staying true in files it is FORBIDDEN to edit:
`accord/stats/intervals.py`'s bootstrap loop, `accord/detector/
accord_phase3_v2_ablations.py`'s frozen structure, and the fact that
`accord/control/verify_rq3_falsifications.py` cannot fail. If any of these
pins breaks, the public-controls code built on top of that assumption needs
re-review before it is trusted again.

Also pins the compliance rules from the task: nothing under
`accord/public_controls/` may import MIMIC-derived dictionaries, open a MIMIC
path, or talk to Ollama.

Run: python3 -m pytest tests/test_public_controls_pins.py -q
"""
from __future__ import annotations

import ast
import hashlib
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accord.public_controls import evaluation as ev  # noqa: E402
from accord.public_controls import frozen_ablations_loader as fl  # noqa: E402

INTERVALS_PY = ROOT / "accord" / "stats" / "intervals.py"
ABLATIONS_PY = ROOT / "accord" / "detector" / "accord_phase3_v2_ablations.py"
FALSIFICATIONS_PY = ROOT / "accord" / "control" / "verify_rq3_falsifications.py"
PUBLIC_CONTROLS_DIR = ROOT / "accord" / "public_controls"

# The exact stratified-bootstrap loop this project reimplements (intervals.py,
# currently lines 71-77: the pos/neg index split, the fresh rng, the B-loop
# body, and the array cast). Pinned verbatim; if intervals.py's text changes
# here, `stratified_bootstrap_auroc` needs re-review against the new text.
PINNED_INTERVALS_SNIPPET = (
    "pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]\n"
    "rng = np.random.default_rng(SEED)\n"
    "strat = []\n"
    "for _ in range(B):\n"
    "    i = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]\n"
    "    strat.append(roc_auc_score(y[i], s[i]))\n"
    "strat = np.array(strat)"
)


def test_intervals_py_bootstrap_snippet_unchanged():
    """We must never import intervals.py (it reads MIMIC at import time), so
    this reads its TEXT and asserts the exact lines `stratified_bootstrap_auroc`
    reimplements are still there, unedited."""
    assert INTERVALS_PY.exists(), "intervals.py moved; public-controls pin needs updating"
    text = INTERVALS_PY.read_text()
    assert PINNED_INTERVALS_SNIPPET in text, (
        "accord/stats/intervals.py's stratified bootstrap loop changed. "
        "accord/public_controls/evaluation.py:stratified_bootstrap_auroc is a "
        "reimplementation of exactly this snippet and must be re-reviewed "
        "against the new text before this pin is updated.")


def test_intervals_py_seed_and_B_unchanged():
    text = INTERVALS_PY.read_text()
    assert "SEED, B = 42, 10_000" in text, (
        "intervals.py's SEED/B constants changed; evaluation.py's B=10_000 "
        "default and pinned RNG order assume SEED=42, B=10_000.")


def test_bootstrap_fast_matches_slow_reimplementation():
    """The vectorized bootstrap (`stratified_bootstrap_auroc_fast`, used for
    the heavy C101/C102 replicate counts) must give bit-for-bit the same
    result as the slow, pinned, loop reimplementation for the same seed."""
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(12, int), np.zeros(241, int)]
    s = np.r_[rng.normal(1.2, 1, 12), rng.normal(0, 1, 241)]
    slow = ev.stratified_bootstrap_auroc(y, s, seed=7, B=300)
    fast = ev.stratified_bootstrap_auroc_fast(y, s, seed=7, B=300)
    assert np.allclose(slow, fast, atol=1e-9)


def test_rule_c98_matches_graded_score_probe_condition():
    """C98's own rule text (AUROC >= 0.7355 AND one-sided MW p < 0.05) is
    never edited; this pins that evaluation.py's measurement copy of the
    condition uses the same MDE constant and the same boolean structure."""
    text = (ROOT / "accord" / "differential" / "C98_PREREGISTRATION.md").read_text()
    assert "0.7355" in text
    assert "one-sided Mann-Whitney" in text
    y = np.r_[np.ones(12, int), np.zeros(241, int)]
    s_strong = np.r_[np.full(12, 5.0), np.zeros(241)]
    assert ev.rule_c98(y, s_strong) is True
    s_flat = np.zeros(253)
    assert ev.rule_c98(y, s_flat) is False


# ---------------------------------------------------------------------------
# The frozen ablations file: sha256 pin + ast-only, allow-listed extraction.
# ---------------------------------------------------------------------------

def test_ablations_file_sha256_pin():
    actual = hashlib.sha256(ABLATIONS_PY.read_bytes()).hexdigest()
    assert actual == fl.FROZEN_FILE_SHA256, (
        "accord/detector/accord_phase3_v2_ablations.py changed. It is frozen "
        "evidence and must never be edited; if this is intentional elsewhere, "
        "frozen_ablations_loader.FROZEN_FILE_SHA256 needs a deliberate, "
        "reviewed update, not a silent one.")


def test_ablations_file_never_imported_as_a_module():
    """Guard against a regression where some public-controls script imports
    the frozen file directly (which would run its pip installs and Drive
    mount at import time)."""
    forbidden_import_patterns = (
        "import accord_phase3_v2_ablations",
        "from accord.detector.accord_phase3_v2_ablations import",
        "from accord.detector import accord_phase3_v2_ablations",
        "from accord_phase3_v2_ablations import",
    )
    for py in PUBLIC_CONTROLS_DIR.rglob("*.py"):
        if py.name == "frozen_ablations_loader.py":
            continue
        text = py.read_text()
        for pattern in forbidden_import_patterns:
            assert pattern not in text, f"{py} imports the frozen file directly: {pattern}"


def test_ablations_file_still_has_the_side_effects_the_loader_avoids():
    """Confirms the reason this loader exists is still true: the file still
    pip-installs and mounts Drive at module scope, so importing it directly
    would still be wrong."""
    tree = ast.parse(ABLATIONS_PY.read_text())
    top_level_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "google.colab":
            top_level_calls.append("google.colab")
    assert "google.colab" in top_level_calls
    assert "'pip', 'install'" in ABLATIONS_PY.read_text()


def test_frozen_loader_extracts_allowlisted_classes_only():
    ns = fl.load(["AuthorityTemporalConflictGraph", "GuidelineGroundedHarmEstimator",
                  "LABEL_MAP", "DRUG_CLASS_DANGER"])
    assert set(ns) == {"AuthorityTemporalConflictGraph", "GuidelineGroundedHarmEstimator",
                       "LABEL_MAP", "DRUG_CLASS_DANGER"}
    assert ns["LABEL_MAP"] == {"contradiction": 0, "not_contradiction": 1, "ambiguous": 2}


def test_frozen_loader_rejects_names_outside_the_allowlist():
    with pytest.raises(ValueError):
        fl.load(["ACCORDv2NoACNE"])  # a real class in the file, but not allow-listed


def test_frozen_loader_instantiates_tiny_authority_and_harm_modules():
    """Sanity that the ast-extracted classes are the real, usable classes
    (not stubs), at dimensions small enough to run in a CPU smoke test."""
    import torch
    ns = fl.load(["AuthorityTemporalConflictGraph", "GuidelineGroundedHarmEstimator"])
    atcg = ns["AuthorityTemporalConflictGraph"](node_dim=8, hidden_dim=8, num_heads=2)
    n_node, d_node = torch.randn(3, 8), torch.randn(3, 8)
    edge_repr, conflict_score, _ = atcg(n_node, d_node)
    assert edge_repr.shape == (3, 16)
    assert conflict_score.shape == (3, 1)
    gghe = ns["GuidelineGroundedHarmEstimator"](
        edge_dim=16, prior_dim=4, hidden_dim=8, num_labels=3, num_rules=4)
    prior = torch.randn(3, 4)
    c_logits, r_logits, harm, expl, attn = gghe(edge_repr, prior)
    assert c_logits.shape == (3, 3)
    assert r_logits.shape == (3, 4)
    assert harm.shape == (3, 1)


# ---------------------------------------------------------------------------
# verify_rq3_falsifications.py cannot fail; pin that fact so nothing ever
# relies on it as if it were a real check.
# ---------------------------------------------------------------------------

def test_verify_rq3_falsifications_is_hardcoded_and_must_not_be_relied_on():
    text = FALSIFICATIONS_PY.read_text()
    assert text.count('"falsified": True') >= 3, (
        "verify_rq3_falsifications.py no longer hardcodes its verdict; if it "
        "now computes a real, switchable result this pin (and the note in "
        "the C102 pre-registration that Stage 3 replaces it) should be "
        "revisited, but that script is read-only for this work either way.")
    for py in PUBLIC_CONTROLS_DIR.rglob("*.py"):
        assert "verify_rq3_falsifications" not in py.read_text(), (
            f"{py} depends on a check that is hardcoded to always pass")


# ---------------------------------------------------------------------------
# Compliance: no MIMIC, no dictionaries.py, no Ollama, from anything under
# accord/public_controls/.
# ---------------------------------------------------------------------------

FORBIDDEN_SUBSTRINGS = (
    "mimic_local_only",
    "Downloads MIMIC",
    "accord.dictionaries",
    "from accord import dictionaries",
    "ACTION_PATTERNS",
    "DOSE_RE",
    "11434",  # the Ollama local port used elsewhere in this repo
    "ollama",
    "api.openai.com",
    "api.anthropic.com",
    "bedrock",
)


def test_public_controls_has_no_mimic_or_ollama_or_hosted_api_references():
    offenders = []
    for py in PUBLIC_CONTROLS_DIR.rglob("*.py"):
        text = py.read_text().lower()
        for marker in FORBIDDEN_SUBSTRINGS:
            if marker.lower() in text:
                offenders.append((py.name, marker))
    assert not offenders, f"forbidden reference(s) found: {offenders}"


def test_public_controls_model_loading_is_offline_and_cpu_only_where_present():
    """Any script under accord/public_controls/ that loads a HF model must set
    the offline/CPU env flags this task requires; scripts that don't load a
    model at all are exempt."""
    for py in PUBLIC_CONTROLS_DIR.rglob("*.py"):
        text = py.read_text()
        if "from_pretrained" not in text:
            continue
        assert "HF_HUB_OFFLINE" in text and "TRANSFORMERS_OFFLINE" in text, (
            f"{py} loads a model but does not force offline mode")
        assert "local_files_only=True" in text or "local_files_only = True" in text, (
            f"{py} loads a model without local_files_only=True")
        assert 'device' not in text.lower() or "cuda" not in text.lower() or "cpu" in text.lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
