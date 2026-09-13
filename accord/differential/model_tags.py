"""One function, one place, that turns (model label, think) into a filename tag.

WHY THIS EXISTS
---------------
`run_frontier_compliant.py` used to build the Ollama filename tag as
`model_label.split("/")[-1].split(":")[0]`, which drops everything after the
colon. `qwen3:14b` and `qwen3:8b` both produce `qwen3`, so a 14B run would have
overwritten the committed 8B artifact silently: exactly the class of defect
CLAUDE.md already records twice ("A model tag in the wrong filename position").

The fix is to put the size in the tag and to make thinking mode part of the
identity too, since the same model served with `think` on and off is not the
same measurement (see the superseded qwen3:8b zero-shot artifact, think on, 67
parse failures, discarded under C94's rule).

Three tags are already committed to disk and cited by the register:
llama3:8b -> _llama3, granite3.3:8b -> _granite33, qwen3:8b (think unset,
i.e. the pre-C94/C94-era runs, made before this project tracked think at all)
-> _qwen3. Those must never change, so they are frozen in LEGACY_TAGS rather
than derived, and a test in accord/stats/test_register.py checks the map is
byte-for-byte what shipped.

Everything else, including qwen3:8b run again with think recorded explicitly
(true or false), gets a NEW tag under the family_size(+_nothink) scheme below,
which cannot collide with a legacy tag because every legacy tag is missing the
size segment the new scheme requires.
"""

import re
import types

# The three tags already committed and cited by the register. `think=None`
# means "the run was made before this project recorded think at all" (the
# pre-C94 / C94 runs), which is the ONLY key that maps to a legacy tag. Any
# other think value for these same labels (True or False, recorded explicitly)
# is a different, later run and gets a new-scheme tag instead, never one of
# these three strings.
LEGACY_TAGS = types.MappingProxyType({
    ("llama3:8b", None): "_llama3",
    ("granite3.3:8b", None): "_granite33",
    ("qwen3:8b", None): "_qwen3",
})

_ALLOWED = re.compile(r"^[a-z0-9_]+$")
_SANITIZE = re.compile(r"[^a-z0-9]+")


def _sanitize(piece):
    return _SANITIZE.sub("", piece.lower())


def ollama_model_tag(label, think):
    """Return the filename tag (with a leading underscore) for (label, think).

    label   the Ollama model label, e.g. "qwen3:14b". Must carry an explicit
            size after a colon; a bare family name is refused, because a tag
            with no size segment is exactly how the collision this file fixes
            happened in the first place.
    think   True, False, or None. None means "not recorded" and is accepted
            ONLY for the three frozen LEGACY_TAGS entries; every other label
            must pass an explicit True or False.
    """
    key = (label, think)
    if key in LEGACY_TAGS:
        return LEGACY_TAGS[key]
    if think is None:
        raise ValueError(
            f"ollama_model_tag({label!r}, None): think must be recorded "
            f"explicitly (True or False) for any label outside the frozen "
            f"LEGACY_TAGS three")
    if ":" not in label:
        raise ValueError(
            f"ollama_model_tag({label!r}, ...): no explicit size after a "
            f"colon; refusing rather than guessing one, which is the defect "
            f"this file exists to close")
    family, _, size = label.partition(":")
    family, size = _sanitize(family), _sanitize(size)
    if not family or not size:
        raise ValueError(
            f"ollama_model_tag({label!r}, ...): family or size is empty "
            f"after sanitizing to [a-z0-9]")
    tag = f"_{family}_{size}"
    if think is False:
        tag += "_nothink"
    elif think is not True:
        raise ValueError(
            f"ollama_model_tag({label!r}, {think!r}): think must be True, "
            f"False, or None (legacy only)")
    if not _ALLOWED.match(tag[1:]):
        raise ValueError(f"ollama_model_tag produced an out-of-charset tag: "
                         f"{tag!r}")
    return tag
