"""Frozen role taxonomy. Derived from MIMIC-III NOTEEVENTS.CATEGORY.

Counts are from a full scan of the 2,083,180-row table (scripts/scan_notes.py).

Deliberately NOT included: an `authority_rank` ordinal. CAREGIVERS.csv is not
available locally, so any ranking would be a hypothesis written into an input
and then "recovered" from the model -- which is exactly the ACCORD failure
(authority_scale initialized at [0.5, 1.0], reported as converging to
0.508/1.004). Role identity from CATEGORY is uncontroversial factual metadata.
Authority, if it exists, must be read off the learned bias table.
"""

from __future__ import annotations

PAD_ROLE = 0
UNK_ROLE = 7

ROLE_NAMES: dict[int, str] = {
    0: "PAD",
    1: "NURSING",
    2: "PHYSICIAN",
    3: "DISCHARGE",
    4: "RADIOLOGY",
    5: "CARDIAC_DIAG",
    6: "ALLIED",
    7: "UNK",
}

N_ROLES = len(ROLE_NAMES)

ROLE_MAP: dict[str, int] = {
    "Nursing": 1,
    "Nursing/other": 1,
    "Physician ": 2,
    "Physician": 2,
    "Discharge summary": 3,
    "Radiology": 4,
    "ECG": 5,
    "Echo": 5,
    "Respiratory": 6,
    "Nutrition": 6,
    "Rehab Services": 6,
    "Social Work": 6,
    "Case Management": 6,
    "General": 6,
    "Pharmacy": 6,
    "Consult": 6,
}

# Discharge summaries are authored AT discharge and state the outcome verbatim.
# Including them in an outcome-prediction bundle is catastrophic label leakage of
# exactly the kind that produced ACCORD's Rule F1 = 0.953.
OUTCOME_EXCLUDED_ROLES: frozenset[int] = frozenset({3})


def role_of(category: str | None) -> int:
    if not category:
        return UNK_ROLE
    return ROLE_MAP.get(category.strip(), UNK_ROLE)
